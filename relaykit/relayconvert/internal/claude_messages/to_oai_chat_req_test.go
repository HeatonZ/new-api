package claudemessages

import (
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
)

func TestStripVolatileCCH(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "strip cch only, keep stable fields",
			in:   "x-anthropic-billing-header: cc_version=2.1.177.c0b; cc_entrypoint=cli; cch=b8e89;",
			want: "x-anthropic-billing-header: cc_version=2.1.177.c0b; cc_entrypoint=cli",
		},
		{
			name: "no cch present",
			in:   "You are a helpful assistant.",
			want: "You are a helpful assistant.",
		},
		{
			name: "cch at end without trailing semicolon",
			in:   "prefix; cch=abc123",
			want: "prefix",
		},
		{
			name: "cleanup double semicolons",
			in:   "a=1; cch=x;; b=2;",
			want: "a=1; b=2",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := stripVolatileCCH(c.in); got != c.want {
				t.Errorf("stripVolatileCCH(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestStripAnthropicBillingSystem(t *testing.T) {
	t.Run("string system that is entirely billing header -> nil", func(t *testing.T) {
		req := &dto.ClaudeRequest{
			System: "x-anthropic-billing-header: cc_version=2.1.177.c0b; cc_entrypoint=cli; cch=b8e89;",
		}
		stripAnthropicBillingSystem(req)
		if req.System != nil {
			t.Errorf("expected nil system, got %v", req.System)
		}
	})

	t.Run("string system with billing header + prompt -> cch stripped, prompt kept", func(t *testing.T) {
		req := &dto.ClaudeRequest{
			System: "x-anthropic-billing-header: cc_version=2.1.177.c0b; cc_entrypoint=cli; cch=b8e89;\n\nYou are helpful.",
		}
		stripAnthropicBillingSystem(req)
		got := req.GetStringSystem()
		if got != "x-anthropic-billing-header: cc_version=2.1.177.c0b; cc_entrypoint=cli;\n\nYou are helpful." {
			t.Errorf("unexpected system: %q", got)
		}
	})

	t.Run("array system filters billing blocks and strips cch from others", func(t *testing.T) {
		text1 := "x-anthropic-billing-header: cc_version=2.1.177.c0b; cc_entrypoint=cli; cch=b8e89;"
		text2 := "You are a coding assistant; cch=volatile; stay focused."
		req := &dto.ClaudeRequest{
			System: []dto.ClaudeMediaMessage{
				{Type: "text", Text: &text1},
				{Type: "text", Text: &text2},
			},
		}
		stripAnthropicBillingSystem(req)
		systems := req.ParseSystem()
		if len(systems) != 1 {
			t.Fatalf("expected 1 system block, got %d: %+v", len(systems), systems)
		}
		if got := *systems[0].Text; got != "You are a coding assistant; stay focused." {
			t.Errorf("unexpected remaining block: %q", got)
		}
	})

	t.Run("array system all billing blocks -> nil", func(t *testing.T) {
		text := "x-anthropic-billing-header: cc_version=2.1.177.c0b; cch=abc;"
		req := &dto.ClaudeRequest{
			System: []dto.ClaudeMediaMessage{{Type: "text", Text: &text}},
		}
		stripAnthropicBillingSystem(req)
		if req.System != nil {
			t.Errorf("expected nil system, got %v", req.System)
		}
	})

	t.Run("nil system is a no-op", func(t *testing.T) {
		req := &dto.ClaudeRequest{}
		stripAnthropicBillingSystem(req)
		if req.System != nil {
			t.Errorf("expected nil system, got %v", req.System)
		}
	})
}

func TestRewriteHistorySystemToUser(t *testing.T) {
	t.Run("mid-conversation system reminders become user", func(t *testing.T) {
		req := &dto.ClaudeRequest{
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "hi"},
				{Role: "system", Content: "Task tool reminder"},
				{Role: "assistant", Content: "ok"},
				{Role: "SYSTEM", Content: "date changed"},
			},
		}
		rewriteHistorySystemToUser(req)
		want := []string{"user", "user", "assistant", "user"}
		for i, w := range want {
			if req.Messages[i].Role != w {
				t.Errorf("message %d role = %q, want %q", i, req.Messages[i].Role, w)
			}
		}
	})

	t.Run("no system messages is a no-op", func(t *testing.T) {
		req := &dto.ClaudeRequest{
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "hi"},
				{Role: "assistant", Content: "ok"},
			},
		}
		rewriteHistorySystemToUser(req)
		if req.Messages[0].Role != "user" || req.Messages[1].Role != "assistant" {
			t.Errorf("roles changed unexpectedly: %+v", req.Messages)
		}
	})
}

func TestClaudeMessagesRequestToOpenAIChatCacheStability(t *testing.T) {
	// Full conversion: billing header must be stripped and mid-conversation
	// system reminders demoted, so the outbound prefix is stable across turns.
	text1 := "x-anthropic-billing-header: cc_version=2.1.177.c0b; cc_entrypoint=cli; cch=b8e89;"
	req := dto.ClaudeRequest{
		Model: "deepseek-v4-flash",
		System: []dto.ClaudeMediaMessage{
			{Type: "text", Text: &text1},
		},
		Messages: []dto.ClaudeMessage{
			{Role: "user", Content: "hello"},
			{Role: "system", Content: "date reminder"},
			{Role: "assistant", Content: "hi there"},
		},
	}
	out, err := ClaudeMessagesRequestToOpenAIChat(req, nil)
	if err != nil {
		t.Fatalf("conversion failed: %v", err)
	}
	if len(out.Messages) == 0 {
		t.Fatal("no messages in output")
	}
	// The first message must be the user message (billing block dropped entirely).
	if out.Messages[0].Role != "user" {
		t.Errorf("first message role = %q, want user (billing header block should be dropped)", out.Messages[0].Role)
	}
	// Mid-conversation system reminder demoted to user.
	roles := map[string]bool{}
	for _, m := range out.Messages {
		roles[m.Role] = true
	}
	if roles["system"] {
		t.Errorf("no message should remain system after rewrite, got roles %v", roles)
	}
}

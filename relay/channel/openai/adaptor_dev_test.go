package openai

import (
	"encoding/json"
	"testing"

	"github.com/gin-gonic/gin"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/relaykit/dto"
)

// TestConvertOpenAIRequestNormalizesDeveloperRoleToSystem 验证 chat 路径上
// developer role 在任意位置都被归一化为 system（与 responses 转换器行为一致）。
// 背景：opencode Console Go / DeepSeek 等上游只认 system/user/assistant/tool，
// developer 直接透传会 400 "unknown variant `developer`"。
func TestConvertOpenAIRequestNormalizesDeveloperRoleToSystem(t *testing.T) {
	adaptor := &Adaptor{}
	info := &relaycommon.RelayInfo{
		ChannelMeta: &relaycommon.ChannelMeta{
			UpstreamModelName: "deepseek-v4-flash-free",
			ChannelType:       constant.ChannelTypeOpenAI,
		},
	}

	c, _ := gin.CreateTestContext(nil)

	toolCalls, _ := json.Marshal([]map[string]any{{"id": "call_1", "type": "function", "function": map[string]any{"name": "f", "arguments": "{}"}}})

	cases := []struct {
		name      string
		messages  []dto.Message
		wantRoles []string
	}{
		{
			name: "developer first, user second",
			messages: []dto.Message{
				{Role: "developer", Content: "You are helpful"},
				{Role: "user", Content: "hello"},
			},
			wantRoles: []string{"system", "user"},
		},
		{
			name: "developer at arbitrary index",
			messages: []dto.Message{
				{Role: "user", Content: "hi"},
				{Role: "developer", Content: "be concise"},
				{Role: "user", Content: "explain"},
			},
			wantRoles: []string{"user", "system", "user"},
		},
		{
			name: "assistant and tool roles untouched",
			messages: []dto.Message{
				{Role: "developer", Content: "rules"},
				{Role: "user", Content: "q"},
				{Role: "assistant", Content: "a", ToolCalls: toolCalls},
				{Role: "tool", Content: "tool result"},
			},
			wantRoles: []string{"system", "user", "assistant", "tool"},
		},
		{
			name: "no developer leaves everything as-is",
			messages: []dto.Message{
				{Role: "system", Content: "sys"},
				{Role: "user", Content: "u"},
			},
			wantRoles: []string{"system", "user"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := &dto.GeneralOpenAIRequest{Messages: tc.messages}
			_, err := adaptor.ConvertOpenAIRequest(c, info, req)
			if err != nil {
				t.Fatalf("ConvertOpenAIRequest returned error: %v", err)
			}
			if len(req.Messages) != len(tc.wantRoles) {
				t.Fatalf("message count = %d, want %d", len(req.Messages), len(tc.wantRoles))
			}
			for i, want := range tc.wantRoles {
				if req.Messages[i].Role != want {
					t.Errorf("messages[%d].Role = %q, want %q", i, req.Messages[i].Role, want)
				}
			}
		})
	}
}

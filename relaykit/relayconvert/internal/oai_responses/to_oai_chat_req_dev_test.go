package oairesponses

import (
	"encoding/json"
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestResponsesRequestToChatCompletionsDeveloperRoleNormalized verifies that a
// Responses-API "developer" input item is normalized to "system" during the
// responses->chat conversion, while other roles (user/assistant/tool) pass
// through untouched. DeepSeek / opencode zen-go reject "developer" with a 400.
func TestResponsesRequestToChatCompletionsDeveloperRoleNormalized(t *testing.T) {
	tests := []struct {
		name  string
		input json.RawMessage
		want  []dto.Message
	}{
		{
			name: "developer first, user second",
			input: json.RawMessage(`[
				{"role": "developer", "content": [{"type": "input_text", "text": "you are a helpful assistant"}]},
				{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
			]`),
			want: []dto.Message{
				{Role: "system", Content: "you are a helpful assistant"},
				{Role: "user", Content: "hi"},
			},
		},
		{
			name:  "developer with plain string content",
			input: json.RawMessage(`[{"role": "developer", "content": "system rules"}]`),
			want:  []dto.Message{{Role: "system", Content: "system rules"}},
		},
		{
			name: "assistant and tool roles untouched",
			input: json.RawMessage(`[
				{"role": "user", "content": "hello"},
				{"role": "assistant", "content": "hi there"},
				{"type": "function_call_output", "call_id": "call_1", "output": "ok"}
			]`),
			want: []dto.Message{
				{Role: "user", Content: "hello"},
				{Role: "assistant", Content: "hi there"},
				{Role: "tool", ToolCallId: "call_1", Content: "ok"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
				Model: "gpt-test",
				Input: tt.input,
			})
			require.NoError(t, err)
			require.Len(t, got.Messages, len(tt.want))
			for i := range tt.want {
				assert.Equal(t, tt.want[i].Role, got.Messages[i].Role, "messages[%d].role", i)
				assert.Equal(t, tt.want[i].Content, got.Messages[i].Content, "messages[%d].content", i)
				assert.Equal(t, tt.want[i].ToolCallId, got.Messages[i].ToolCallId, "messages[%d].tool_call_id", i)
			}
		})
	}
}

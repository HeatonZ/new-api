package oairesponses

import (
	"encoding/json"
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func ptr(s string) *string { return &s }

// TestResponsesRequestToChatCompletionsConversion verifies the responses->chat
// converter keeps native shapes (developer role and custom tool type pass
// through untouched) and only performs format-necessary mappings:
// custom_tool_call_output -> chat "tool" message (chat has no "custom" role).
// Upstream-specific normalization (developer->system, custom->function) is done
// at the channel adaptor layer (advancedcustom) where the upstream is known.
func TestResponsesRequestToChatCompletionsConversion(t *testing.T) {
	tests := []struct {
		name  string
		input json.RawMessage
		want  []dto.Message
	}{
		{
			name: "developer role passes through untouched",
			input: json.RawMessage(`[
				{"role": "developer", "content": [{"type": "input_text", "text": "you are a helpful assistant"}]},
				{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
			]`),
			want: []dto.Message{
				{Role: "developer", Content: "you are a helpful assistant"},
				{Role: "user", Content: "hi"},
			},
		},
		{
			name:  "developer with plain string content",
			input: json.RawMessage(`[{"role": "developer", "content": "system rules"}]`),
			want:  []dto.Message{{Role: "developer", Content: "system rules"}},
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
		{
			// custom_tool_call_output items carry "role":"custom"; chat has no
			// such role, so the converter maps them to a "tool" message. This is
			// format-necessary for ANY chat upstream, not opencode-specific.
			name: "custom_tool_call_output with role custom maps to tool",
			input: json.RawMessage(`[
				{"role": "user", "content": "run the tool"},
				{"type": "custom_tool_call", "id": "ct_1", "name": "web_search", "input": "{\"q\":\"x\"}"},
				{"type": "custom_tool_call_output", "id": "ct_1", "role": "custom", "output": "search results"}
			]`),
			want: []dto.Message{
				{Role: "user", Content: "run the tool"},
				{Role: "assistant", ToolCalls: json.RawMessage(`[{"id":"ct_1","type":"custom","custom":{"type":"custom_tool_call","id":"ct_1","name":"web_search","input":"{\"q\":\"x\"}"},"function":{"name":"web_search","arguments":"{\"q\":\"x\"}"}}]`)},
				{Role: "tool", ToolCallId: "ct_1", Content: "search results"},
			},
		},
		{
			name: "custom_tool_call_output falls back to id when call_id missing",
			input: json.RawMessage(`[
				{"type": "custom_tool_call_output", "id": "ct_9", "role": "custom", "output": "done"}
			]`),
			want: []dto.Message{
				{Role: "tool", ToolCallId: "ct_9", Content: "done"},
			},
		},
		{
			// DeepSeek thinking mode requires historical assistant messages to
			// carry reasoning_content back to the API. A Responses "reasoning"
			// item in input must fold into the preceding assistant message.
			name: "reasoning item folds into preceding assistant reasoning_content",
			input: json.RawMessage(`[
				{"role": "user", "content": "think hard"},
				{"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "step one"}], "content": [{"type": "output_text", "text": "step two"}]},
				{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}
			]`),
			want: []dto.Message{
				{Role: "user", Content: "think hard"},
				{Role: "assistant", Content: "answer", ReasoningContent: ptr("step onestep two")},
			},
		},
		{
			name: "reasoning item without preceding assistant is dropped",
			input: json.RawMessage(`[
				{"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "orphan reasoning"}]}
			]`),
			want: []dto.Message{},
		},
		{
			// Tool-call turns also carry a reasoning item before the function
			// call; the reasoning must land on the assistant message that holds
			// the tool_calls so DeepSeek thinking mode accepts the replay.
			name: "reasoning folds into assistant with tool calls",
			input: json.RawMessage(`[
				{"role": "user", "content": "look it up"},
				{"type": "reasoning", "id": "rs_2", "summary": [{"type": "summary_text", "text": "planning"}], "content": [{"type": "output_text", "text": "details"}]},
				{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\":\"x\"}"},
				{"type": "function_call_output", "call_id": "call_1", "output": "result"}
			]`),
			want: []dto.Message{
				{Role: "user", Content: "look it up"},
				{Role: "assistant", ReasoningContent: ptr("planningdetails"), ToolCalls: json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]`)},
				{Role: "tool", ToolCallId: "call_1", Content: "result"},
			},
		},
		{
			// A reasoning item followed directly by a tool-output item leaves the
			// placeholder mid-array (tool branch appends without consuming it).
			// It must be dropped, never serialized with Role=="".
			name: "stray reasoning placeholder before tool output is dropped",
			input: json.RawMessage(`[
				{"role": "user", "content": "run it"},
				{"type": "reasoning", "id": "rs_3", "summary": [{"type": "summary_text", "text": "orphan before tool"}]},
				{"type": "custom_tool_call_output", "id": "ct_1", "role": "custom", "output": "done"}
			]`),
			want: []dto.Message{
				{Role: "user", Content: "run it"},
				{Role: "tool", ToolCallId: "ct_1", Content: "done"},
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
				assert.Equal(t, tt.want[i].GetReasoningContent(), got.Messages[i].GetReasoningContent(), "messages[%d].reasoning_content", i)
				if len(tt.want[i].ToolCalls) == 0 && len(got.Messages[i].ToolCalls) == 0 {
					continue
				}
				var wantTC, gotTC []map[string]any
				if len(tt.want[i].ToolCalls) > 0 {
					require.NoError(t, json.Unmarshal(tt.want[i].ToolCalls, &wantTC))
				}
				if len(got.Messages[i].ToolCalls) > 0 {
					require.NoError(t, json.Unmarshal(got.Messages[i].ToolCalls, &gotTC))
				}
				assert.Equal(t, wantTC, gotTC, "messages[%d].tool_calls", i)
			}
		})
	}
}

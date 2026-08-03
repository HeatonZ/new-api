package advancedcustom

import (
	"encoding/json"
	"testing"

	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsOpenCodeUpstream(t *testing.T) {
	cases := []struct {
		name string
		info *relaycommon.RelayInfo
		want bool
	}{
		{"nil info is not opencode", nil, false},
		{"empty base url is not opencode", &relaycommon.RelayInfo{ChannelMeta: &relaycommon.ChannelMeta{ChannelBaseUrl: ""}}, false},
		{"opencode zen is opencode", &relaycommon.RelayInfo{ChannelMeta: &relaycommon.ChannelMeta{ChannelBaseUrl: "https://opencode.ai/zen"}}, true},
		{"opencode go is opencode", &relaycommon.RelayInfo{ChannelMeta: &relaycommon.ChannelMeta{ChannelBaseUrl: "https://opencode.ai/zen/go"}}, true},
		{"case insensitive", &relaycommon.RelayInfo{ChannelMeta: &relaycommon.ChannelMeta{ChannelBaseUrl: "HTTPS://OPENCODE.AI/ZEN"}}, true},
		{"openai native is not opencode", &relaycommon.RelayInfo{ChannelMeta: &relaycommon.ChannelMeta{ChannelBaseUrl: "https://api.openai.com/v1"}}, false},
		{"deepseek direct is not opencode", &relaycommon.RelayInfo{ChannelMeta: &relaycommon.ChannelMeta{ChannelBaseUrl: "https://api.deepseek.com"}}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, isOpenCodeUpstream(tc.info))
		})
	}
}

func TestNormalizeRequestForOpenCode(t *testing.T) {
	toolCallsRaw := json.RawMessage(`[
		{"id":"ct_1","type":"custom","custom":{"type":"custom_tool_call","id":"ct_1","name":"web_search","input":"{}"},"function":{"name":"web_search","arguments":"{}"}},
		{"id":"fc_2","type":"function","function":{"name":"lookup","arguments":"{}"}}
	]`)

	req := &dto.GeneralOpenAIRequest{
		Messages: []dto.Message{
			{Role: "developer", Content: "rules"},
			{Role: "user", Content: "hi"},
			{Role: "assistant", ToolCalls: toolCallsRaw},
			{Role: "tool", ToolCallId: "ct_1", Content: "result"},
		},
		Tools: []dto.ToolCallRequest{
			{Type: "custom", Custom: json.RawMessage(`{"type":"custom","name":"web_search","description":"search the web","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}`)},
			{Type: "function", Function: dto.FunctionRequest{Name: "lookup"}},
		},
	}

	normalizeRequestForOpenCode(req)

	assert.Equal(t, "system", req.Messages[0].Role, "developer should become system")
	assert.Equal(t, "user", req.Messages[1].Role, "user untouched")

	toolCalls := req.Messages[2].ParseToolCalls()
	require.Len(t, toolCalls, 2)
	assert.Equal(t, "function", toolCalls[0].Type, "custom tool call type -> function")
	assert.Equal(t, "function", toolCalls[1].Type, "function tool call untouched")
	// raw custom payload survives for upstreams that can read it
	assert.Equal(t, "custom_tool_call", func() string {
		var m map[string]any
		_ = json.Unmarshal(toolCalls[0].Custom, &m)
		return m["type"].(string)
	}(), "raw custom payload preserved")

	// tools declarations normalized the same way
	require.Len(t, req.Tools, 2)
	assert.Equal(t, "function", req.Tools[0].Type, "custom tool declaration type -> function")
	assert.Equal(t, "function", req.Tools[1].Type, "function tool declaration untouched")
	// function fields backfilled from the raw custom payload so opencode's
	// serde gets a non-empty name
	assert.Equal(t, "web_search", req.Tools[0].Function.Name, "custom tool name backfilled into function.name")
	assert.Equal(t, "search the web", req.Tools[0].Function.Description, "custom tool description backfilled")
	require.NotNil(t, req.Tools[0].Function.Parameters, "input_schema aliased to parameters")

	// nil request is a no-op
	normalizeRequestForOpenCode(nil)
}

func TestNormalizeNamespaceToolsForOpenCode(t *testing.T) {
	// Codex wraps MCP tools in {"type":"namespace","name":"mcp__open_websearch__",
	// "tools":[{function...}]} (openai/codex#23186). opencode zen/go rejects the
	// wrapper, so each child must be flattened into a top-level function tool.
	namespaceRaw := json.RawMessage(`{
		"type": "namespace",
		"name": "mcp__open_websearch__",
		"tools": [
			{"type": "function", "name": "search", "description": "web search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
			{"type": "function", "name": "fetchWebContent", "description": "fetch url", "input_schema": {"type": "object"}}
		]
	}`)

	req := &dto.GeneralOpenAIRequest{
		Tools: []dto.ToolCallRequest{
			{Type: "namespace", Raw: namespaceRaw},
			{Type: "function", Function: dto.FunctionRequest{Name: "lookup"}},
		},
	}

	normalizeRequestForOpenCode(req)

	require.Len(t, req.Tools, 3, "namespace flattens into 2 children + 1 function")
	assert.Equal(t, "function", req.Tools[0].Type)
	assert.Equal(t, "mcp__open_websearch___search", req.Tools[0].Function.Name, "child name prefixed with namespace")
	assert.Equal(t, "web search", req.Tools[0].Function.Description)
	require.NotNil(t, req.Tools[0].Function.Parameters, "parameters carried over")
	assert.Equal(t, "mcp__open_websearch___fetchWebContent", req.Tools[1].Function.Name)
	require.NotNil(t, req.Tools[1].Function.Parameters, "input_schema aliased to parameters")
	assert.Equal(t, "lookup", req.Tools[2].Function.Name, "plain function untouched")
}

func TestNormalizeNamespaceToolsFromCustomFallback(t *testing.T) {
	// responses->chat path keeps the raw tool in Custom instead of Raw; the
	// normalization must fall back to it.
	req := &dto.GeneralOpenAIRequest{
		Tools: []dto.ToolCallRequest{
			{Type: "namespace", Custom: json.RawMessage(`{
				"type": "namespace",
				"name": "mcp__fs__",
				"tools": [
					{"type": "function", "name": "read", "description": "read file", "parameters": {"type": "object"}}
				]
			}`)},
		},
	}

	normalizeRequestForOpenCode(req)

	require.Len(t, req.Tools, 1)
	assert.Equal(t, "function", req.Tools[0].Type)
	assert.Equal(t, "mcp__fs___read", req.Tools[0].Function.Name)
}

func TestToolCallRequestUnmarshalPreservesRaw(t *testing.T) {
	raw := json.RawMessage(`{"id":"ct_1","type":"namespace","name":"mcp__x__","tools":[]}`)
	var tc dto.ToolCallRequest
	require.NoError(t, json.Unmarshal(raw, &tc))
	assert.Equal(t, "namespace", tc.Type)
	assert.Equal(t, `{"id":"ct_1","type":"namespace","name":"mcp__x__","tools":[]}`, string(tc.Raw), "raw JSON preserved for upstream normalization")
	// re-marshal must not leak the Raw field
	out, err := json.Marshal(tc)
	require.NoError(t, err)
	assert.NotContains(t, string(out), "\"raw\"", "raw field must not serialize")
}

func TestBackfillReasoningContentForToolCallAssistants(t *testing.T) {
	// DeepSeek thinking mode (behind opencode) 400s when an assistant message
	// carrying tool_calls has no reasoning_content. Codex clients send
	// reasoning items with only encrypted_content (no extractable text), so
	// the converter cannot recover the real reasoning — a placeholder must be
	// synthesized. Plain assistant history does not need one (verified against
	// the live upstream).
	existing := "actual reasoning"
	req := &dto.GeneralOpenAIRequest{
		Messages: []dto.Message{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "plain history"},
			{Role: "assistant", ToolCalls: json.RawMessage(`[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]`)},
			{Role: "tool", ToolCallId: "c1", Content: "sunny"},
			{Role: "assistant", ReasoningContent: &existing, ToolCalls: json.RawMessage(`[{"id":"c2","type":"function","function":{"name":"lookup","arguments":"{}"}}]`)},
			{Role: "tool", ToolCallId: "c2", Content: "found"},
		},
	}

	normalizeRequestForOpenCode(req)

	assert.Nil(t, req.Messages[1].ReasoningContent, "plain assistant history untouched")
	require.NotNil(t, req.Messages[2].ReasoningContent, "tool-call assistant without reasoning gets placeholder")
	assert.NotEmpty(t, *req.Messages[2].ReasoningContent)
	require.NotNil(t, req.Messages[4].ReasoningContent)
	assert.Equal(t, existing, *req.Messages[4].ReasoningContent, "existing reasoning_content preserved")
}

func TestNormalizeUnknownToolTypesForOpenCode(t *testing.T) {
	// Variant #8 seen in production: codex-family clients declare tools with
	// type "tool_search"; opencode's serde only accepts "function". Unknown
	// types must convert when named, or be dropped when unnamed — never pass
	// through, which guarantees an upstream 400.
	req := &dto.GeneralOpenAIRequest{
		Tools: []dto.ToolCallRequest{
			{Type: "tool_search", Function: dto.FunctionRequest{Name: "tool_search"}},
			{Type: "tool_search", Raw: json.RawMessage(`{"type":"tool_search","name":"find_tool","description":"search tools","input_schema":{"type":"object"}}`)},
			{Type: "some_future_type"},
			{Type: "function", Function: dto.FunctionRequest{Name: "lookup"}},
		},
	}

	normalizeRequestForOpenCode(req)

	require.Len(t, req.Tools, 3, "named unknown types converted, unnamed dropped")
	assert.Equal(t, "function", req.Tools[0].Type, "tool_search with parsed name -> function")
	assert.Equal(t, "tool_search", req.Tools[0].Function.Name)
	assert.Equal(t, "function", req.Tools[1].Type, "tool_search with raw name -> function")
	assert.Equal(t, "find_tool", req.Tools[1].Function.Name, "name recovered from raw JSON")
	assert.Equal(t, "search tools", req.Tools[1].Function.Description)
	require.NotNil(t, req.Tools[1].Function.Parameters, "input_schema aliased to parameters")
	assert.Equal(t, "lookup", req.Tools[2].Function.Name, "plain function untouched")
}

func TestToolCallRequestFlatFunctionBackfill(t *testing.T) {
	// Chat clients (Codex namespace children, flat declarations) put name at the
	// top level instead of nested under function; the DTO must backfill it.
	var tc dto.ToolCallRequest
	require.NoError(t, json.Unmarshal(json.RawMessage(`{"type":"function","name":"lookup","description":"find things","input_schema":{"type":"object"}}`), &tc))
	assert.Equal(t, "function", tc.Type)
	assert.Equal(t, "lookup", tc.Function.Name, "top-level name backfilled into function.name")
	assert.Equal(t, "find things", tc.Function.Description, "top-level description backfilled")
	require.NotNil(t, tc.Function.Parameters, "input_schema aliased to parameters")

	// nested function form still works
	var nested dto.ToolCallRequest
	require.NoError(t, json.Unmarshal(json.RawMessage(`{"type":"function","function":{"name":"nested"}}`), &nested))
	assert.Equal(t, "nested", nested.Function.Name, "nested function.name untouched")

	// namespace with flat children: top-level name preserved in Raw, children
	// backfilled at parse time
	var ns dto.ToolCallRequest
	require.NoError(t, json.Unmarshal(json.RawMessage(`{"type":"namespace","name":"mcp__fs__","tools":[{"type":"function","name":"read","parameters":{"type":"object"}}]}`), &ns))
	assert.Equal(t, "namespace", ns.Type)
	assert.NotEmpty(t, ns.Raw, "namespace raw preserved")
}

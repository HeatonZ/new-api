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
			{Type: "custom", Function: dto.FunctionRequest{Name: "web_search"}},
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

	// nil request is a no-op
	normalizeRequestForOpenCode(nil)
}

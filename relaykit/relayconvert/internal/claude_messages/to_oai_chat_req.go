package claudemessages

import (
	"fmt"
	"strings"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/convmeta"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
)

const (
	webSearchMaxUsesLow    = 1
	webSearchMaxUsesMedium = 5
	webSearchMaxUsesHigh   = 10
)

type openRouterRequestReasoning struct {
	Enabled   bool   `json:"enabled"`
	Effort    string `json:"effort,omitempty"`
	MaxTokens int    `json:"max_tokens,omitempty"`
	Exclude   bool   `json:"exclude,omitempty"`
}

const anthropicBillingHeaderMarker = "x-anthropic-billing-header"

// stripVolatileCCH removes the volatile cch=<hex> segment from an Anthropic
// billing header, keeping stable fields such as cc_version and cc_entrypoint
// intact, and cleans up leftover separators. opencode/DeepSeek disk cache
// matches on a byte-identical body prefix, and the per-turn cch value would
// otherwise break prefix matching on every request.
func stripVolatileCCH(s string) string {
	idx := strings.Index(s, "cch=")
	if idx >= 0 {
		end := strings.IndexByte(s[idx:], ';')
		if end < 0 {
			s = s[:idx]
		} else {
			s = s[:idx] + s[idx+end:]
		}
	}
	s = strings.ReplaceAll(s, ";;", ";")
	s = strings.ReplaceAll(s, "; ;", ";")
	s = strings.TrimSuffix(s, ";")
	s = strings.TrimSuffix(s, "; ")
	return strings.TrimSpace(s)
}

// isAnthropicBillingHeader reports whether the whole trimmed text is an
// Anthropic billing header line (injected by Claude Code into system[0]).
// A system block that mixes the header with real prompt content is not
// treated as a pure billing header; it is only stripped of its volatile
// cch segment instead.
func isAnthropicBillingHeader(text string) bool {
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(strings.ToLower(trimmed), anthropicBillingHeaderMarker) {
		return false
	}
	// The header line ends at the first newline. If anything meaningful
	// follows the header line, treat the block as prompt content.
	if nl := strings.IndexByte(trimmed, '\n'); nl >= 0 {
		after := strings.TrimSpace(trimmed[nl:])
		return after == ""
	}
	rest := trimmed[len(anthropicBillingHeaderMarker):]
	return !strings.Contains(rest, "\n") && strings.Contains(rest, "=")
}

// stripAnthropicBillingSystem removes the per-request billing header that
// Claude Code injects into system[0]. Its cch=<hex> segment changes on every
// turn and, after conversion, sits at the very front of the outbound system
// prefix, destroying byte-identical prefix matching for opencode/DeepSeek
// cache. A string system that is entirely the billing header is dropped;
// otherwise the volatile cch segment is stripped from each block.
func stripAnthropicBillingSystem(req *dto.ClaudeRequest) {
	if req.System == nil {
		return
	}
	if req.IsStringSystem() {
		system := req.GetStringSystem()
		if isAnthropicBillingHeader(system) {
			req.System = nil
			return
		}
		req.SetStringSystem(stripVolatileCCH(system))
		return
	}
	systems := req.ParseSystem()
	if len(systems) == 0 {
		return
	}
	kept := make([]dto.ClaudeMediaMessage, 0, len(systems))
	for _, system := range systems {
		if system.Text == nil {
			kept = append(kept, system)
			continue
		}
		text := *system.Text
		if isAnthropicBillingHeader(text) {
			continue
		}
		cleaned := stripVolatileCCH(text)
		system.Text = &cleaned
		kept = append(kept, system)
	}
	if len(kept) == 0 {
		req.System = nil
		return
	}
	req.System = kept
}

// rewriteHistorySystemToUser rewrites mid-conversation system reminders
// (tool hints, date changes, etc. injected by Claude Code) to role "user".
// OpenAI-compatible upstreams hoist every system message to the front of the
// prompt, so an inserted reminder rewrites the whole system prefix and
// evicts everything after it from cache. Moving reminders to the tail keeps
// the shared prefix byte-identical across turns. The top-level
// request.System (the real system prompt) is left untouched.
func rewriteHistorySystemToUser(req *dto.ClaudeRequest) {
	for i := range req.Messages {
		if strings.EqualFold(req.Messages[i].Role, "system") {
			req.Messages[i].Role = "user"
		}
	}
}

func ClaudeMessagesRequestToOpenAIChat(claudeRequest dto.ClaudeRequest, info convmeta.Meta) (*dto.GeneralOpenAIRequest, error) {
	// Cache-stability preprocessing for opencode/DeepSeek: opencode matches
	// disk cache on the byte-identical body prefix, so strip the volatile
	// per-turn billing header Claude Code injects into system[0] and demote
	// mid-conversation system reminders to user so they land at the tail.
	stripAnthropicBillingSystem(&claudeRequest)
	rewriteHistorySystemToUser(&claudeRequest)

	openAIRequest := dto.GeneralOpenAIRequest{
		Model:       claudeRequest.Model,
		Temperature: claudeRequest.Temperature,
	}
	if claudeRequest.MaxTokens != nil {
		openAIRequest.MaxTokens = kitutil.GetPointer(*claudeRequest.MaxTokens)
	}
	if claudeRequest.TopP != nil {
		openAIRequest.TopP = kitutil.GetPointer(*claudeRequest.TopP)
	}
	if claudeRequest.TopK != nil {
		openAIRequest.TopK = kitutil.GetPointer(*claudeRequest.TopK)
	}
	if claudeRequest.Stream != nil {
		openAIRequest.Stream = kitutil.GetPointer(*claudeRequest.Stream)
	}

	isOpenRouter := convmeta.OptionsOf(info).OpenRouterDialect
	if isOpenRouter {
		if effort := claudeRequest.GetEfforts(); effort != "" {
			effortBytes, _ := kitutil.Marshal(effort)
			openAIRequest.Verbosity = effortBytes
		}
		if claudeRequest.Thinking != nil {
			var reasoningConfig openRouterRequestReasoning
			if claudeRequest.Thinking.Type == "enabled" {
				reasoningConfig = openRouterRequestReasoning{
					Enabled:   true,
					MaxTokens: claudeRequest.Thinking.GetBudgetTokens(),
				}
			} else if claudeRequest.Thinking.Type == "adaptive" {
				reasoningConfig = openRouterRequestReasoning{
					Enabled: true,
				}
			}
			reasoningJSON, err := kitutil.Marshal(reasoningConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal reasoning: %w", err)
			}
			openAIRequest.Reasoning = reasoningJSON
		}
	} else if info != nil {
		thinkingSuffix := "-thinking"
		if strings.HasSuffix(info.GetOriginModelName(), thinkingSuffix) &&
			!strings.HasSuffix(openAIRequest.Model, thinkingSuffix) {
			openAIRequest.Model = openAIRequest.Model + thinkingSuffix
		}
	}

	if len(claudeRequest.StopSequences) == 1 {
		openAIRequest.Stop = claudeRequest.StopSequences[0]
	} else if len(claudeRequest.StopSequences) > 1 {
		openAIRequest.Stop = claudeRequest.StopSequences
	}

	tools, _ := kitutil.Any2Type[[]dto.Tool](claudeRequest.Tools)
	openAITools := make([]dto.ToolCallRequest, 0)
	for _, claudeTool := range tools {
		openAITool := dto.ToolCallRequest{
			Type: "function",
			Function: dto.FunctionRequest{
				Name:        claudeTool.Name,
				Description: claudeTool.Description,
				Parameters:  claudeTool.InputSchema,
			},
		}
		openAITools = append(openAITools, openAITool)
	}
	openAIRequest.Tools = openAITools

	openAIMessages := make([]dto.Message, 0)
	if claudeRequest.System != nil {
		if claudeRequest.IsStringSystem() && claudeRequest.GetStringSystem() != "" {
			openAIMessage := dto.Message{
				Role: "system",
			}
			openAIMessage.SetStringContent(claudeRequest.GetStringSystem())
			openAIMessages = append(openAIMessages, openAIMessage)
		} else {
			systems := claudeRequest.ParseSystem()
			if len(systems) > 0 {
				openAIMessage := dto.Message{
					Role: "system",
				}
				isOpenRouterClaude := isOpenRouter && strings.HasPrefix(convmeta.UpstreamModelName(info), "anthropic/claude")
				if isOpenRouterClaude {
					systemMediaMessages := make([]dto.MediaContent, 0, len(systems))
					for _, system := range systems {
						message := dto.MediaContent{
							Type:         "text",
							Text:         system.GetText(),
							CacheControl: system.CacheControl,
						}
						systemMediaMessages = append(systemMediaMessages, message)
					}
					openAIMessage.SetMediaContent(systemMediaMessages)
				} else {
					systemStr := ""
					for _, system := range systems {
						if system.Text != nil {
							systemStr += *system.Text
						}
					}
					openAIMessage.SetStringContent(systemStr)
				}
				openAIMessages = append(openAIMessages, openAIMessage)
			}
		}
	}

	for _, claudeMessage := range claudeRequest.Messages {
		openAIMessage := dto.Message{
			Role: claudeMessage.Role,
		}
		if claudeMessage.IsStringContent() {
			openAIMessage.SetStringContent(claudeMessage.GetStringContent())
		} else {
			content, err := claudeMessage.ParseContent()
			if err != nil {
				return nil, err
			}
			var toolCalls []dto.ToolCallRequest
			mediaMessages := make([]dto.MediaContent, 0, len(content))

			for _, mediaMsg := range content {
				switch mediaMsg.Type {
				case "text", "input_text":
					message := dto.MediaContent{
						Type:         "text",
						Text:         mediaMsg.GetText(),
						CacheControl: mediaMsg.CacheControl,
					}
					mediaMessages = append(mediaMessages, message)
				case "thinking", "redacted_thinking":
					// Claude Code thinking blocks carry the reasoning text DeepSeek
					// thinking mode requires on every assistant turn. Surface it as
					// reasoning_content so the history round-trips (mirrors the
					// oai_responses placeholder handling); the opencode
					// /v1/messages passthrough needs it verbatim.
					if mediaMsg.Thinking != nil && *mediaMsg.Thinking != "" {
						if openAIMessage.ReasoningContent == nil {
							openAIMessage.ReasoningContent = new(string)
						}
						*openAIMessage.ReasoningContent = *mediaMsg.Thinking
					}
				case "image":
					imageData := fmt.Sprintf("data:%s;base64,%s", mediaMsg.Source.MediaType, mediaMsg.Source.Data)
					mediaMessage := dto.MediaContent{
						Type:     "image_url",
						ImageUrl: &dto.MessageImageUrl{Url: imageData},
					}
					mediaMessages = append(mediaMessages, mediaMessage)
				case "tool_use":
					toolCall := dto.ToolCallRequest{
						ID:   mediaMsg.Id,
						Type: "function",
						Function: dto.FunctionRequest{
							Name:      mediaMsg.Name,
							Arguments: requestToJSONString(mediaMsg.Input),
						},
					}
					toolCalls = append(toolCalls, toolCall)
				case "tool_result":
					toolName := mediaMsg.Name
					if toolName == "" {
						toolName = claudeRequest.SearchToolNameByToolCallId(mediaMsg.ToolUseId)
					}
					oaiToolMessage := dto.Message{
						Role:       "tool",
						Name:       &toolName,
						ToolCallId: mediaMsg.ToolUseId,
					}
					if mediaMsg.IsStringContent() {
						oaiToolMessage.SetStringContent(mediaMsg.GetStringContent())
					} else {
						mediaContents := mediaMsg.ParseMediaContent()
						encodedJSON, _ := kitutil.Marshal(mediaContents)
						oaiToolMessage.SetStringContent(string(encodedJSON))
					}
					openAIMessages = append(openAIMessages, oaiToolMessage)
				}
			}

			if len(toolCalls) > 0 {
				openAIMessage.SetToolCalls(toolCalls)
			}
			if len(mediaMessages) > 0 && len(toolCalls) == 0 {
				openAIMessage.SetMediaContent(mediaMessages)
			}
		}
		if len(openAIMessage.ParseContent()) > 0 || len(openAIMessage.ToolCalls) > 0 {
			openAIMessages = append(openAIMessages, openAIMessage)
		}
	}

	openAIRequest.Messages = openAIMessages
	return &openAIRequest, nil
}

func requestToJSONString(v interface{}) string {
	b, err := kitutil.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

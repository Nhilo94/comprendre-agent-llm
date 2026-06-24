package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
)

// --- Types de l'API (compatibles OpenAI / LM Studio) ---

type ChatRequest struct {
	Messages      []Message      `json:"messages"`
	Model         string         `json:"model"`
	Temperature   float64        `json:"temperature"`
	MaxTokens     int            `json:"max_tokens,omitempty"`
	Tools         []ToolSchema   `json:"tools,omitempty"`
	Stream        bool           `json:"stream,omitempty"`
	StreamOptions *StreamOptions `json:"stream_options,omitempty"`
}

// StreamOptions demande au serveur d'inclure le décompte de tokens (usage) dans
// le flux SSE — sinon "usage" n'est renvoyé qu'en mode non-streamé.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// Usage : décompte de tokens réel renvoyé par le serveur (≠ notre estimation).
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
	// ReasoningContent : "thinking" des modèles à raisonnement (qwen, deepseek).
	// omitempty : on ne le renvoie pas au serveur dans l'historique.
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // chaîne JSON
}

// ToolSchema décrit un outil au modèle (champ "tools" de la requête).
type ToolSchema struct {
	Type     string         `json:"type"` // "function"
	Function FunctionSchema `json:"function"`
}

type FunctionSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"` // JSON Schema
}

// Réponse non-streamée.
type ChatResponse struct {
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// Fragment de réponse streamée (SSE).
type ChatStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content          string          `json:"content"`
			ReasoningContent string          `json:"reasoning_content"`
			Reasoning        string          `json:"reasoning"`
			ToolCalls        []ToolCallDelta `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	// Le serveur envoie un dernier fragment "usage" (choices vide) si on a
	// demandé include_usage.
	Usage *Usage `json:"usage"`
}

type ToolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type ModelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

// --- Estimation de tokens ---
// Heuristique simple : ~4 caractères par token. Suffisant pour piloter le
// compactage sans embarquer un vrai tokenizer.

func estimateTokens(m Message) int {
	n := len([]rune(m.Content))
	for _, tc := range m.ToolCalls {
		n += len([]rune(tc.Function.Arguments)) + len([]rune(tc.Function.Name))
	}
	return n/4 + 4 // +4 d'overhead par message
}

func totalTokens(msgs []Message) int {
	sum := 0
	for _, m := range msgs {
		sum += estimateTokens(m)
	}
	return sum
}

// --- Client LLM ---
// Encapsule la communication avec le serveur. Gère le streaming (SSE) et les
// appels d'outils natifs (function calling). Tous les appels réseau acceptent
// un context.Context : un Ctrl-C (ou un timeout) annule proprement la requête
// en cours.

type LLMClient struct {
	BaseURL     string
	Model       string
	Temperature float64
	MaxTokens   int
	Stream      bool
	HTTP        *http.Client
	ui          *UI

	// Cumul de tokens sur la session (compteurs réels du serveur).
	cumPrompt     int
	cumCompletion int
}

func NewLLMClient(c *Config, ui *UI) *LLMClient {
	return &LLMClient{
		BaseURL:     c.BaseURL,
		Model:       c.Model,
		Temperature: c.Temperature,
		MaxTokens:   c.MaxTokens,
		Stream:      c.Stream,
		// Pas de timeout global sur le client : la durée est gérée par un context
		// dérivé à chaque requête (un long streaming ne doit pas être coupé).
		HTTP: &http.Client{},
		ui:   ui,
	}
}

func (c *LLMClient) chatURL() string {
	return strings.TrimRight(c.BaseURL, "/") + "/v1/chat/completions"
}

func (c *LLMClient) modelsURL() string {
	return strings.TrimRight(c.BaseURL, "/") + "/v1/models"
}

// AvailableModels interroge le serveur pour lister les modèles chargés.
func (c *LLMClient) AvailableModels(ctx context.Context) ([]string, error) {
	c.ui.logStep("MODELES", "récupération des modèles depuis %s", c.modelsURL())

	ctx, cancel := context.WithTimeout(ctx, modelsTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.modelsURL(), nil)
	if err != nil {
		return nil, fmt.Errorf("erreur requête modèles: %w", err)
	}

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("erreur requête modèles: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("erreur lecture modèles: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("erreur HTTP modèles %d: %s", resp.StatusCode, string(body))
	}

	var modelsResp ModelsResponse
	if err := json.Unmarshal(body, &modelsResp); err != nil {
		return nil, fmt.Errorf("erreur parsing modèles: %w", err)
	}

	models := make([]string, 0, len(modelsResp.Data))
	for _, model := range modelsResp.Data {
		if strings.TrimSpace(model.ID) != "" {
			models = append(models, model.ID)
		}
	}

	c.ui.logStep("MODELES", "%d modèle(s) disponible(s)", len(models))
	return models, nil
}

// Chat envoie l'historique + les outils au modèle et renvoie le message
// assistant complet (contenu et/ou appels d'outils).
func (c *LLMClient) Chat(ctx context.Context, history []Message, tools []ToolSchema) (Message, error) {
	req := ChatRequest{
		Messages:    history,
		Model:       c.Model,
		Temperature: c.Temperature,
		MaxTokens:   c.MaxTokens,
		Tools:       tools,
		Stream:      c.Stream,
	}
	c.ui.logStep("LLM", "requête (%s) modèle %s temp %.2f, %d outil(s)", streamLabel(c.Stream), c.Model, c.Temperature, len(tools))
	c.ui.logHistory(history)

	if c.Stream {
		return c.chatStream(ctx, req)
	}
	return c.chatOnce(ctx, req)
}

func streamLabel(stream bool) string {
	if stream {
		return "stream"
	}
	return "sync"
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// recordUsage met à jour le cumul de session et affiche le détail des tokens
// (compteurs réels du serveur) sous la réponse.
func (c *LLMClient) recordUsage(u *Usage) {
	if u == nil || u.TotalTokens == 0 {
		return
	}
	c.cumPrompt += u.PromptTokens
	c.cumCompletion += u.CompletionTokens
	c.ui.tokenUsage(u, c.cumPrompt+c.cumCompletion)
}

// chatOnce : requête classique, réponse complète d'un bloc.
func (c *LLMClient) chatOnce(ctx context.Context, req ChatRequest) (Message, error) {
	req.Stream = false

	body, err := c.send(ctx, req)
	if err != nil {
		return Message{}, err
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return Message{}, fmt.Errorf("erreur parsing JSON: %w\nRéponse brute: %s", err, string(body))
	}
	if len(chatResp.Choices) == 0 {
		return Message{}, fmt.Errorf("pas de réponse du modèle")
	}

	msg := chatResp.Choices[0].Message
	msg.Role = "assistant"
	c.ui.logStep("LLM", "réponse reçue (%d tool_calls)", len(msg.ToolCalls))
	c.recordUsage(&chatResp.Usage)
	return msg, nil
}

func (c *LLMClient) send(ctx context.Context, req ChatRequest) ([]byte, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("erreur création JSON: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, requestTimeout)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.chatURL(), bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("erreur requête: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTP.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("erreur requête: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("erreur lecture réponse: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("erreur HTTP %d: %s", resp.StatusCode, string(body))
	}
	return body, nil
}

// chatStream : lit le flux SSE, affiche le contenu token par token et
// réassemble les appels d'outils (qui arrivent en fragments).
func (c *LLMClient) chatStream(ctx context.Context, req ChatRequest) (Message, error) {
	req.Stream = true
	req.StreamOptions = &StreamOptions{IncludeUsage: true} // pour recevoir "usage"

	jsonData, err := json.Marshal(req)
	if err != nil {
		return Message{}, fmt.Errorf("erreur création JSON: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, requestTimeout)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.chatURL(), bytes.NewReader(jsonData))
	if err != nil {
		return Message{}, fmt.Errorf("erreur requête: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := c.HTTP.Do(httpReq)
	if err != nil {
		return Message{}, fmt.Errorf("erreur requête: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return Message{}, fmt.Errorf("erreur HTTP %d: %s", resp.StatusCode, string(b))
	}

	msg := Message{Role: "assistant"}
	var content, reasoning strings.Builder
	toolCalls := make(map[int]*ToolCall)
	var indices []int
	var usage *Usage
	var md *mdStream        // rendu Markdown du contenu, créé au 1er token réel
	contentStarted := false // l'en-tête 🤖 a-t-il été imprimé ?
	reasoningOpen := false  // un bloc 💭 est-il en cours ?

	reader := bufio.NewReader(resp.Body)
	for {
		line, readErr := reader.ReadString('\n')

		if strings.HasPrefix(line, "data:") {
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data == "[DONE]" {
				break
			}

			var chunk ChatStreamChunk
			if json.Unmarshal([]byte(data), &chunk) == nil {
				if chunk.Usage != nil {
					usage = chunk.Usage // fragment final (choices vide)
				}
				if len(chunk.Choices) > 0 {
					delta := chunk.Choices[0].Delta

					// Raisonnement (qwen, deepseek…) : affiché en grisé, à part.
					if r := firstNonEmpty(delta.ReasoningContent, delta.Reasoning); r != "" {
						if !reasoningOpen {
							c.ui.reasoningStart()
							reasoningOpen = true
						}
						c.ui.streamToken(r)
						reasoning.WriteString(r)
					}

					if delta.Content != "" {
						content.WriteString(delta.Content)
						if reasoningOpen { // fermer le 💭 avant la réponse
							c.ui.reasoningEnd()
							reasoningOpen = false
						}
						// On n'imprime l'en-tête 🤖 qu'au 1er caractère réel : pas
						// de bloc agent vide quand le modèle passe direct aux outils.
						if !contentStarted {
							if strings.TrimSpace(content.String()) != "" {
								contentStarted = true
								c.ui.streamPrefix()
								md = c.ui.newMarkdownStream()
								md.write(strings.TrimLeft(content.String(), " \t\r\n"))
							}
						} else {
							md.write(delta.Content)
						}
					}

					for _, d := range delta.ToolCalls {
						tc, ok := toolCalls[d.Index]
						if !ok {
							tc = &ToolCall{Type: "function"}
							toolCalls[d.Index] = tc
							indices = append(indices, d.Index)
						}
						if d.ID != "" {
							tc.ID = d.ID
						}
						if d.Type != "" {
							tc.Type = d.Type
						}
						if d.Function.Name != "" {
							tc.Function.Name = d.Function.Name
						}
						tc.Function.Arguments += d.Function.Arguments
					}
				}
			}
		}

		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return Message{}, fmt.Errorf("erreur lecture flux: %w", readErr)
		}
	}
	if reasoningOpen {
		c.ui.reasoningEnd()
	}
	if contentStarted {
		md.flush() // rend la dernière ligne partielle + saut de ligne final
	}

	msg.Content = content.String()
	msg.ReasoningContent = reasoning.String()
	sort.Ints(indices)
	for _, idx := range indices {
		msg.ToolCalls = append(msg.ToolCalls, *toolCalls[idx])
	}
	c.ui.logStep("LLM", "flux terminé (%d tool_calls)", len(msg.ToolCalls))
	c.recordUsage(usage)
	return msg, nil
}

// Summarize produit un résumé concis d'un bloc de messages (requête one-shot,
// non-streamée). Utilisé par le compactage de contexte.
func (c *LLMClient) Summarize(ctx context.Context, messages []Message) (string, error) {
	var b strings.Builder
	for _, m := range messages {
		line := m.Content
		for _, tc := range m.ToolCalls {
			line += fmt.Sprintf(" [appel %s(%s)]", tc.Function.Name, tc.Function.Arguments)
		}
		fmt.Fprintf(&b, "%s: %s\n", m.Role, line)
	}

	req := ChatRequest{
		Model:       c.Model,
		Temperature: 0.2,
		Messages: []Message{
			{Role: "system", Content: "Résume en français, de façon concise (puces, max 200 mots), la conversation suivante. Conserve les décisions prises, les fichiers touchés, les commandes importantes et l'état actuel de la tâche."},
			{Role: "user", Content: b.String()},
		},
	}
	msg, err := c.chatOnce(ctx, req)
	if err != nil {
		return "", err
	}
	return msg.Content, nil
}

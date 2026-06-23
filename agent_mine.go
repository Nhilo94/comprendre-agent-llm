package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

// --- Configuration ---
const (
	DEFAULT_BASE_URL     = "http://localhost:1234"
	DEFAULT_MODEL_NAME   = "local-model"
	DEFAULT_TEMPERATURE  = 0.7
	DEFAULT_MAX_TOKENS   = 0 // 0 = pas de limite imposée au modèle
	MAX_AGENT_STEPS      = 10
	MAX_HISTORY_MESSAGES = 20 // fenêtre coulissante (système exclu du décompte)
	REQUEST_TIMEOUT      = 120 * time.Second
	DEBUG_LOGS           = true
)

// --- Types de l'API (compatibles OpenAI / LM Studio) ---

type ChatRequest struct {
	Messages    []Message `json:"messages"`
	Model       string    `json:"model"`
	Temperature float64   `json:"temperature"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
}

type ModelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

// --- Logging (debug) ---

func logStep(label string, format string, args ...any) {
	if !DEBUG_LOGS {
		return
	}

	message := fmt.Sprintf(format, args...)
	fmt.Printf("\n[LOG %s] %s: %s\n", time.Now().Format("15:04:05"), label, message)
}

func logHistory(history []Message) {
	if !DEBUG_LOGS {
		return
	}

	logStep("HISTORIQUE", "%d message(s)", len(history))
	for i, message := range history {
		content := strings.ReplaceAll(message.Content, "\n", `\n`)
		if len(content) > 220 {
			content = content[:220] + "..."
		}
		fmt.Printf("  #%02d %-9s %s\n", i+1, message.Role, content)
	}
}

// --- Client LLM (amélioration #3) ---
// Encapsule la communication avec le serveur. Permet d'ajuster Temperature,
// MaxTokens ou BaseURL sans toucher au reste du code.

type LLMClient struct {
	BaseURL     string
	Model       string
	Temperature float64
	MaxTokens   int
	HTTP        *http.Client
}

func NewLLMClient(baseURL string) *LLMClient {
	return &LLMClient{
		BaseURL:     baseURL,
		Model:       DEFAULT_MODEL_NAME,
		Temperature: DEFAULT_TEMPERATURE,
		MaxTokens:   DEFAULT_MAX_TOKENS,
		HTTP:        &http.Client{Timeout: REQUEST_TIMEOUT},
	}
}

func (c *LLMClient) chatURL() string {
	return strings.TrimRight(c.BaseURL, "/") + "/v1/chat/completions"
}

func (c *LLMClient) modelsURL() string {
	return strings.TrimRight(c.BaseURL, "/") + "/v1/models"
}

// AvailableModels interroge le serveur pour lister les modèles chargés.
func (c *LLMClient) AvailableModels() ([]string, error) {
	logStep("MODELES", "récupération des modèles depuis %s", c.modelsURL())

	resp, err := c.HTTP.Get(c.modelsURL())
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

	logStep("MODELES", "%d modèle(s) disponible(s)", len(models))
	return models, nil
}

// Chat envoie l'historique au modèle et renvoie la réponse textuelle.
func (c *LLMClient) Chat(history []Message) (string, error) {
	logStep("LLM", "requête vers %s (modèle %s, temp %.2f)", c.chatURL(), c.Model, c.Temperature)
	logHistory(history)

	reqBody := ChatRequest{
		Messages:    history,
		Model:       c.Model,
		Temperature: c.Temperature,
		MaxTokens:   c.MaxTokens,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("erreur création JSON: %w", err)
	}
	logStep("LLM", "payload JSON prêt (%d octets)", len(jsonData))

	resp, err := c.HTTP.Post(c.chatURL(), "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("erreur requête: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("erreur lecture réponse: %w", err)
	}
	logStep("LLM", "réponse HTTP %d reçue (%d octets)", resp.StatusCode, len(body))

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("erreur HTTP %d: %s", resp.StatusCode, string(body))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("erreur parsing JSON: %w\nRéponse brute: %s", err, string(body))
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("pas de réponse du modèle")
	}

	reply := chatResp.Choices[0].Message.Content
	logStep("LLM", "contenu assistant reçu: %s", reply)

	return reply, nil
}

// --- Outils & Registre (amélioration #4) ---
// Un ToolFunc reçoit les arguments parsés et renvoie un résultat textuel
// ou une erreur. Le ToolRegistry permet d'ajouter des outils sans modifier
// la boucle principale ni le parseur.

type ToolFunc func(args map[string]string) (string, error)

type Tool struct {
	Name        string
	Description string
	Usage       string
	Run         ToolFunc
}

type ToolRegistry struct {
	order []string // préserve l'ordre d'enregistrement (regex déterministe)
	tools map[string]Tool
}

func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{tools: make(map[string]Tool)}
}

func (r *ToolRegistry) Register(t Tool) {
	if _, exists := r.tools[t.Name]; !exists {
		r.order = append(r.order, t.Name)
	}
	r.tools[t.Name] = t
}

func (r *ToolRegistry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

func (r *ToolRegistry) Names() []string {
	return append([]string(nil), r.order...)
}

// Prompt génère la description des outils pour le prompt système.
func (r *ToolRegistry) Prompt() string {
	var b strings.Builder
	for _, name := range r.order {
		t := r.tools[name]
		fmt.Fprintf(&b, "- %s : %s\n  Usage : %s\n", t.Name, t.Description, t.Usage)
	}
	return b.String()
}

// registerDefaultTools enregistre les outils de base de l'agent.
func registerDefaultTools(r *ToolRegistry) {
	r.Register(Tool{
		Name:        "execute_shell",
		Description: "Exécute une commande shell (Linux/macOS), supporte pipes et redirections.",
		Usage:       `execute_shell(command="ls -la")`,
		Run:         toolExecuteShell,
	})
	r.Register(Tool{
		Name:        "read_file",
		Description: "Lit le contenu d'un fichier texte.",
		Usage:       `read_file(path="chemin/vers/fichier.txt")`,
		Run:         toolReadFile,
	})
	r.Register(Tool{
		Name:        "write_file",
		Description: "Écrit (ou écrase) un fichier texte avec le contenu fourni.",
		Usage:       `write_file(path="fichier.txt", content="bonjour")`,
		Run:         toolWriteFile,
	})
}

// toolExecuteShell : on utilise "sh -c" pour permettre les pipes (|),
// redirections (>) et variables. Les échecs de commande (exit != 0) sont
// renvoyés comme résultat (pas comme erreur Go) afin que le modèle puisse
// les analyser et réessayer.
func toolExecuteShell(args map[string]string) (string, error) {
	command := strings.TrimSpace(args["command"])
	if command == "" {
		return "", fmt.Errorf(`argument "command" manquant ou vide`)
	}

	fmt.Printf("\n[ACTION] Exécution terminal : %s\n", command)
	logStep("OUTIL", "commande shell reçue: %s", command)

	cmd := exec.Command("sh", "-c", command)

	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		result := fmt.Sprintf("ERREUR : %v\nStderr: %s", err, stderr.String())
		logStep("OUTIL", "commande terminée avec erreur: %s", result)
		return result, nil
	}

	result := fmt.Sprintf("Succès : %s", out.String())
	logStep("OUTIL", "commande terminée avec succès: %s", result)
	return result, nil
}

func toolReadFile(args map[string]string) (string, error) {
	path := strings.TrimSpace(args["path"])
	if path == "" {
		return "", fmt.Errorf(`argument "path" manquant`)
	}

	fmt.Printf("\n[ACTION] Lecture fichier : %s\n", path)
	logStep("OUTIL", "lecture fichier: %s", path)

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("lecture impossible: %w", err)
	}
	return fmt.Sprintf("Contenu de %s :\n%s", path, string(data)), nil
}

func toolWriteFile(args map[string]string) (string, error) {
	path := strings.TrimSpace(args["path"])
	if path == "" {
		return "", fmt.Errorf(`argument "path" manquant`)
	}
	content := args["content"]

	fmt.Printf("\n[ACTION] Écriture fichier : %s (%d octets)\n", path, len(content))
	logStep("OUTIL", "écriture fichier: %s", path)

	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("écriture impossible: %w", err)
	}
	return fmt.Sprintf("Fichier écrit: %s (%d octets)", path, len(content)), nil
}

// --- Parsing des actions (amélioration #1 : Regex) ---
// Capture chaque paire clé="valeur" en tolérant les espaces et les
// guillemets échappés (\").
var argRe = regexp.MustCompile(`([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"`)

// parseAction détecte un appel d'outil dans la réponse du modèle.
// Le motif est construit à partir des noms réellement enregistrés, ce qui
// évite les faux positifs sur du texte qui ressemble à un appel de fonction.
func parseAction(reply string, registry *ToolRegistry) (name string, args map[string]string, found bool) {
	names := registry.Names()
	if len(names) == 0 {
		return "", nil, false
	}

	logStep("PARSER", "recherche d'une action parmi: %s", strings.Join(names, ", "))

	// (?s) : '.' matche aussi les sauts de ligne. (.*) est gourmand donc on
	// capture jusqu'à la dernière parenthèse fermante (utile si la commande
	// contient elle-même des parenthèses).
	pattern := `(?s)\b(` + strings.Join(names, "|") + `)\s*\(\s*(.*)\)`
	re := regexp.MustCompile(pattern)

	m := re.FindStringSubmatch(reply)
	if m == nil {
		logStep("PARSER", "aucune action outil détectée")
		return "", nil, false
	}

	name = m[1]
	args = parseArgs(m[2])
	logStep("PARSER", "action détectée: %s args=%v", name, args)
	return name, args, true
}

func parseArgs(raw string) map[string]string {
	args := make(map[string]string)
	for _, kv := range argRe.FindAllStringSubmatch(raw, -1) {
		args[kv[1]] = unescape(kv[2])
	}
	return args
}

func unescape(s string) string {
	s = strings.ReplaceAll(s, `\"`, `"`)
	s = strings.ReplaceAll(s, `\\`, `\`)
	return s
}

// --- Gestion du contexte (amélioration #2 : fenêtre coulissante) ---
// Conserve tous les messages système en tête + les `max` derniers messages
// pour ne jamais dépasser la fenêtre de contexte du modèle.
func trimHistory(history []Message, max int) []Message {
	// On sépare les messages système initiaux du reste.
	i := 0
	for i < len(history) && history[i].Role == "system" {
		i++
	}
	systems := history[:i]
	rest := history[i:]

	if len(rest) <= max {
		return history
	}

	trimmed := rest[len(rest)-max:]
	logStep("HISTORIQUE", "fenêtre coulissante: %d -> %d messages (hors système)", len(rest), len(trimmed))

	out := make([]Message, 0, len(systems)+len(trimmed))
	out = append(out, systems...)
	out = append(out, trimmed...)
	return out
}

// --- Sélection du modèle ---

func chooseModel(client *LLMClient, scanner *bufio.Scanner) string {
	models, err := client.AvailableModels()
	if err != nil {
		fmt.Println("Impossible de récupérer la liste des modèles :", err)
		fmt.Printf("Modèle par défaut proposé : %s\n", DEFAULT_MODEL_NAME)
	}

	if len(models) > 0 {
		fmt.Println("\nModèles disponibles :")
		for i, model := range models {
			fmt.Printf("  %d. %s\n", i+1, model)
		}
	}

	fmt.Printf("\nChoisis un modèle par numéro ou tape son nom [%s] : ", DEFAULT_MODEL_NAME)
	if !scanner.Scan() {
		return DEFAULT_MODEL_NAME
	}

	choice := strings.TrimSpace(scanner.Text())
	if choice == "" {
		logStep("MODELES", "modèle par défaut sélectionné: %s", DEFAULT_MODEL_NAME)
		return DEFAULT_MODEL_NAME
	}

	for i, model := range models {
		if choice == fmt.Sprintf("%d", i+1) {
			logStep("MODELES", "modèle sélectionné par numéro: %s", model)
			return model
		}
	}

	logStep("MODELES", "modèle sélectionné par nom: %s", choice)
	return choice
}

// --- Logique de l'Agent ---

func main() {
	registry := NewToolRegistry()
	registerDefaultTools(registry)

	// Le prompt système est généré à partir du registre : ajouter un outil
	// met automatiquement à jour les capacités annoncées au modèle.
	systemPrompt := fmt.Sprintf(`Tu es un agent de codage autonome. Tu peux interagir avec le système via des outils.

Outils disponibles :
%s
Instructions :
1. Discute avec l'utilisateur en français.
2. Quand l'utilisateur donne une mission, analyse-la et décompose-la en étapes.
3. Utilise un outil seulement quand une action est nécessaire (une seule action par réponse).
4. Pour appeler un outil, écris exactement : Action: NOM_OUTIL(arg="valeur").
5. Les guillemets à l'intérieur d'une valeur doivent être échappés avec \".
6. Si un outil renvoie une erreur, analyse-la et corrige ta commande ou ton format.
7. Quand une mission est totalement accomplie, réponds "FIN" puis propose de continuer la discussion.`, registry.Prompt())

	history := []Message{
		{Role: "system", Content: systemPrompt},
	}

	client := NewLLMClient(DEFAULT_BASE_URL)
	scanner := bufio.NewScanner(os.Stdin)
	client.Model = chooseModel(client, scanner)

	fmt.Println("Agent prêt. Tape une mission ou une question. Tape 'exit' ou 'quit' pour quitter.")
	logStep("INIT", "agent initialisé avec %d message(s) système", len(history))
	logStep("INIT", "modèle actif: %s | outils: %s", client.Model, strings.Join(registry.Names(), ", "))

	for {
		fmt.Print("\n[USER] ")
		if !scanner.Scan() {
			break
		}

		userInput := strings.TrimSpace(scanner.Text())
		if userInput == "" {
			logStep("USER", "entrée vide ignorée")
			continue
		}
		if userInput == "exit" || userInput == "quit" {
			logStep("USER", "demande de sortie reçue: %s", userInput)
			fmt.Println("Au revoir.")
			break
		}

		logStep("USER", "message reçu: %s", userInput)
		history = append(history, Message{Role: "user", Content: userInput})
		logStep("HISTORIQUE", "message utilisateur ajouté")

		for step := 0; step < MAX_AGENT_STEPS; step++ {
			logStep("BOUCLE", "étape agent %d/%d", step+1, MAX_AGENT_STEPS)

			history = trimHistory(history, MAX_HISTORY_MESSAGES)

			reply, err := client.Chat(history)
			if err != nil {
				// Amélioration #5 : on ne tue plus le programme. On signale
				// l'erreur et on rend la main à l'utilisateur.
				fmt.Println("Erreur modèle :", err)
				logStep("BOUCLE", "erreur LLM, retour à l'utilisateur")
				break
			}

			fmt.Printf("\n[AGENT] %s\n", reply)
			history = append(history, Message{Role: "assistant", Content: reply})
			logStep("HISTORIQUE", "réponse assistant ajoutée")

			name, args, ok := parseAction(reply, registry)
			if !ok {
				logStep("BOUCLE", "pas d'action à exécuter, retour à l'utilisateur")
				break
			}

			tool, exists := registry.Get(name)
			if !exists {
				// Sécurité : ne devrait pas arriver (regex basée sur le registre).
				feedback := fmt.Sprintf("ERREUR: outil inconnu %q. Outils disponibles: %s", name, strings.Join(registry.Names(), ", "))
				fmt.Println(feedback)
				history = append(history, Message{Role: "user", Content: feedback})
				continue
			}

			result, runErr := tool.Run(args)
			if runErr != nil {
				// Amélioration #5 : l'erreur d'exécution/format est réinjectée
				// pour que l'agent "comprenne" sa faute et se corrige.
				feedback := fmt.Sprintf("Résultat outil %s (ERREUR): %v\nFormat attendu: %s", name, runErr, tool.Usage)
				fmt.Printf("\n[OUTIL] %s\n", feedback)
				history = append(history, Message{Role: "user", Content: feedback})
				logStep("HISTORIQUE", "erreur outil réinjectée dans la conversation")
				continue
			}

			history = append(history, Message{Role: "user", Content: fmt.Sprintf("Résultat outil %s:\n%s", name, result)})
			logStep("HISTORIQUE", "résultat outil ajouté à la conversation")
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Erreur lecture utilisateur :", err)
	}
}

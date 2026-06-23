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
	"strings"
	"time"
)

// --- Configuration ---
const (
	LM_STUDIO_CHAT_URL   = "http://localhost:1234/v1/chat/completions"
	LM_STUDIO_MODELS_URL = "http://localhost:1234/v1/models"
	DEFAULT_MODEL_NAME   = "local-model"
	DEBUG_LOGS           = true
)

type ChatRequest struct {
	Messages []Message `json:"messages"`
	Model    string    `json:"model"`
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

// --- Outils de l'Agent (Les "Mains") ---

type Tools struct{}

// executeShell permet à l'agent de taper n'importe quelle commande Linux/macOS
func (t *Tools) executeShell(command string) string {
	fmt.Printf("\n[ACTION] Exécution terminal : %s\n", command)
	logStep("OUTIL", "commande shell reçue: %s", command)

	// On utilise "sh -c" pour permettre les pipes (|), redirections (>) et variables
	cmd := exec.Command("sh", "-c", command)

	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		result := fmt.Sprintf("ERREUR : %v\nStderr: %s", err, stderr.String())
		logStep("OUTIL", "commande terminée avec erreur: %s", result)
		return result
	}

	result := fmt.Sprintf("Succès : %s", out.String())
	logStep("OUTIL", "commande terminée avec succès: %s", result)
	return result
}

func fetchAvailableModels() ([]string, error) {
	logStep("MODELES", "récupération des modèles depuis %s", LM_STUDIO_MODELS_URL)

	resp, err := http.Get(LM_STUDIO_MODELS_URL)
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
	err = json.Unmarshal(body, &modelsResp)
	if err != nil {
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

func chooseModel(scanner *bufio.Scanner) string {
	models, err := fetchAvailableModels()
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

func callModel(history []Message, modelName string) (string, error) {
	logStep("LLM", "préparation requête vers %s avec le modèle %s", LM_STUDIO_CHAT_URL, modelName)
	logHistory(history)

	reqBody := ChatRequest{
		Messages: history,
		Model:    modelName,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("erreur création JSON: %w", err)
	}
	logStep("LLM", "payload JSON prêt (%d octets)", len(jsonData))

	resp, err := http.Post(LM_STUDIO_CHAT_URL, "application/json", bytes.NewBuffer(jsonData))
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
	err = json.Unmarshal(body, &chatResp)
	if err != nil {
		return "", fmt.Errorf("erreur parsing JSON: %w\nRéponse brute: %s", err, string(body))
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("pas de réponse du modèle")
	}

	reply := chatResp.Choices[0].Message.Content
	logStep("LLM", "contenu assistant reçu: %s", reply)

	return reply, nil
}

func extractShellCommand(reply string) (string, bool) {
	logStep("PARSER", "recherche d'une action execute_shell")

	start := strings.Index(reply, `execute_shell(command="`)
	if start == -1 {
		logStep("PARSER", "aucune action shell détectée")
		return "", false
	}

	start += len(`execute_shell(command="`)
	end := strings.Index(reply[start:], `")`)
	if end == -1 {
		logStep("PARSER", "action détectée mais format incomplet")
		return "", false
	}

	command := reply[start : start+end]
	logStep("PARSER", "commande extraite: %s", command)
	return command, true
}

// --- Logique de l'Agent ---

func main() {
	// Le prompt système est crucial : il définit les capacités de l'agent
	systemPrompt := `Tu es un agent de codage autonome. Tu peux interagir avec le système via l'outil :
execute_shell(command="LA_COMMANDE_LINUX").

Instructions :
1. Discute avec l'utilisateur en français.
2. Quand l'utilisateur donne une mission, analyse-la et décompose-la en étapes.
3. Utilise execute_shell seulement quand une action terminal est nécessaire.
4. Si une commande renvoie une erreur, analyse-la et réessaie avec une correction.
5. Quand une mission est totalement accomplie, réponds "FIN" puis propose à l'utilisateur de continuer la discussion.

Format de réponse pour les outils :
Action: execute_shell(command="...")`

	history := []Message{
		{Role: "system", Content: systemPrompt},
	}

	agent := Tools{}
	scanner := bufio.NewScanner(os.Stdin)
	modelName := chooseModel(scanner)

	fmt.Println("Agent prêt. Tape une mission ou une question. Tape 'exit' ou 'quit' pour quitter.")
	logStep("INIT", "agent initialisé avec %d message(s) système", len(history))
	logStep("INIT", "modèle actif: %s", modelName)

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

		for step := 0; step < 10; step++ {
			logStep("BOUCLE", "étape agent %d/10", step+1)

			reply, err := callModel(history, modelName)
			if err != nil {
				fmt.Println("Erreur modèle :", err)
				return
			}

			fmt.Printf("\n[AGENT] %s\n", reply)
			history = append(history, Message{Role: "assistant", Content: reply})
			logStep("HISTORIQUE", "réponse assistant ajoutée")

			command, ok := extractShellCommand(reply)
			if !ok {
				logStep("BOUCLE", "pas d'action à exécuter, retour à l'utilisateur")
				break
			}

			result := agent.executeShell(command)
			history = append(history, Message{Role: "user", Content: "Résultat outil execute_shell:\n" + result})
			logStep("HISTORIQUE", "résultat outil ajouté à la conversation")
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Erreur lecture utilisateur :", err)
	}
}

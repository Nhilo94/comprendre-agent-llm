package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
)

func main() {
	cfg := LoadConfig()
	ui := NewUI(os.Stdout, cfg)

	registry := NewToolRegistry()
	registerDefaultTools(registry)

	client := NewLLMClient(cfg, ui)

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024) // tolère les entrées longues (collage)

	// Sélection du modèle : interactive si non fourni via -model / AGENT_MODEL.
	if client.Model == "" {
		client.Model = chooseModel(context.Background(), client, ui, scanner)
	}

	agent := &Agent{
		client:   client,
		registry: registry,
		ui:       ui,
		scanner:  scanner,
		history:  []Message{{Role: "system", Content: buildSystemPrompt(registry)}},
		maxSteps: cfg.MaxSteps,
		maxCtx:   cfg.MaxContext,
	}

	ui.welcome(client, cfg)
	ui.logStep("INIT", "modèle: %s | streaming: %t | outils: %s", client.Model, client.Stream, strings.Join(registry.Names(), ", "))

	for {
		ui.printf("\n%s %s ", ui.colorize(cGreen+cBold, iUser), ui.colorize(cGreen+cBold, "vous ›"))
		if !scanner.Scan() {
			break
		}

		userInput := strings.TrimSpace(scanner.Text())
		if userInput == "" {
			ui.logStep("USER", "entrée vide ignorée")
			continue
		}
		if userInput == "exit" || userInput == "quit" {
			ui.logStep("USER", "demande de sortie reçue: %s", userInput)
			ui.println(ui.colorize(cGray, "Au revoir."))
			break
		}

		ui.logStep("USER", "message reçu: %s", userInput)
		agent.history = append(agent.history, Message{Role: "user", Content: userInput})

		// Le handler de signal n'est armé que pendant le tour : un Ctrl-C annule
		// l'appel LLM / la commande en cours et rend la main au prompt. Hors tour
		// (au prompt), Ctrl-C garde son comportement par défaut (quitter).
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		agent.handleTurn(ctx)
		stop()
	}

	if err := scanner.Err(); err != nil {
		ui.println(ui.colorize(cRed, fmt.Sprintf("Erreur lecture utilisateur : %v", err)))
	}
}

// chooseModel liste les modèles du serveur et laisse l'utilisateur en choisir un
// (par numéro ou par nom).
func chooseModel(ctx context.Context, client *LLMClient, ui *UI, scanner *bufio.Scanner) string {
	models, err := client.AvailableModels(ctx)
	if err != nil {
		ui.printf("%s %v\n", ui.colorize(cRed, "Impossible de récupérer la liste des modèles :"), err)
		ui.printf("Modèle par défaut proposé : %s\n", ui.colorize(cBold, defaultModel))
	}

	if len(models) > 0 {
		ui.println(ui.colorize(cBold, "\nModèles disponibles :"))
		for i, model := range models {
			ui.printf("  %s %s\n", ui.colorize(cCyan, fmt.Sprintf("%d.", i+1)), model)
		}
	}

	ui.printf("\nChoisis un modèle par numéro ou tape son nom [%s] : ", ui.colorize(cBold, defaultModel))
	if !scanner.Scan() {
		return defaultModel
	}

	choice := strings.TrimSpace(scanner.Text())
	if choice == "" {
		ui.logStep("MODELES", "modèle par défaut sélectionné: %s", defaultModel)
		return defaultModel
	}

	for i, model := range models {
		if choice == fmt.Sprintf("%d", i+1) {
			ui.logStep("MODELES", "modèle sélectionné par numéro: %s", model)
			return model
		}
	}

	ui.logStep("MODELES", "modèle sélectionné par nom: %s", choice)
	return choice
}

func buildSystemPrompt(r *ToolRegistry) string {
	return fmt.Sprintf(`Tu es un agent de codage autonome qui travaille en français.

Tu disposes des outils suivants (fournis aussi via l'API de fonction) :
%s
Règles :
1. Décompose les missions en étapes et appelle les outils nécessaires.
2. Analyse le résultat de chaque outil ; en cas d'erreur, corrige puis réessaie.
3. Si ton client ne supporte pas les appels d'outils natifs, écris : Action: nom_outil(arg="valeur") (échappe les guillemets internes avec \").
4. Une action destructive peut demander une confirmation à l'utilisateur.
5. Quand la mission est accomplie, fais un court récapitulatif puis rends la main.`, r.Prompt())
}

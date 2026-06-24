package main

import (
	"bufio"
	"context"
	"fmt"
	"strings"
)

// Agent orchestre la boucle agentique. Ses dépendances (client, registre, UI,
// entrée) sont injectées : aucun état global, ce qui le rend testable et
// recomposable.
type Agent struct {
	client   *LLMClient
	registry *ToolRegistry
	ui       *UI
	scanner  *bufio.Scanner
	history  []Message
	maxSteps int
	maxCtx   int

	lastStepSig string // signature des actions de l'étape précédente (anti-boucle)
}

// handleTurn fait tourner la boucle agent jusqu'à ce qu'il n'y ait plus
// d'action à exécuter (ou que la limite d'étapes soit atteinte). Le context
// permet d'interrompre proprement un tour (Ctrl-C, timeout).
func (a *Agent) handleTurn(ctx context.Context) {
	a.lastStepSig = "" // nouveau tour : on réinitialise la détection de boucle
	for step := 0; step < a.maxSteps; step++ {
		a.ui.stepHeader(step+1, a.maxSteps)

		a.compactContext(ctx)

		msg, err := a.client.Chat(ctx, a.history, a.registry.Schemas())
		if err != nil {
			// On ne tue pas le programme : on rend la main à l'utilisateur.
			if ctx.Err() != nil {
				a.ui.banner(cYellow, "INTERROMPU", "Tour interrompu (Ctrl-C).")
			} else {
				a.ui.banner(cRed, "ERREUR", "%v", err)
			}
			return
		}
		// En streaming, contenu et raisonnement sont déjà affichés à la volée ;
		// en non-stream, on les affiche ici.
		if !a.client.Stream {
			if strings.TrimSpace(msg.ReasoningContent) != "" {
				a.ui.reasoningBlock(msg.ReasoningContent)
			}
			if strings.TrimSpace(msg.Content) != "" {
				a.ui.agentMarkdown(msg.Content)
			}
		}

		// Le raisonnement a servi à l'affichage : on ne le renvoie pas au serveur
		// (contexte gonflé pour rien, et certains serveurs le rejettent à l'entrée).
		msg.ReasoningContent = ""
		a.history = append(a.history, msg)

		// Chemin natif : appels d'outils structurés (function calling).
		if len(msg.ToolCalls) > 0 {
			if a.isLoop(toolCallsSig(msg.ToolCalls), step) {
				return
			}
			for _, call := range msg.ToolCalls {
				result := a.runTool(ctx, call.Function.Name, parseJSONArgs(call.Function.Arguments))
				a.history = append(a.history, Message{
					Role:       "tool",
					ToolCallID: call.ID,
					Name:       call.Function.Name,
					Content:    result,
				})
			}
			continue
		}

		// Chemin de secours : parsing texte pour modèles sans tool_calls.
		name, args, found := parseAction(msg.Content, a.registry)
		if !found {
			a.ui.logStep("PARSER", "aucune action texte détectée")
			a.ui.agentDone(step + 1)
			return
		}
		if a.isLoop(actionSig(name, args), step) {
			return
		}
		a.ui.logStep("PARSER", "action texte détectée: %s args=%v", name, args)
		result := a.runTool(ctx, name, args)
		a.history = append(a.history, Message{Role: "user", Content: fmt.Sprintf("Résultat outil %s:\n%s", name, result)})
	}

	a.ui.banner(cYellow, "LIMITE", "Limite de %d étapes atteinte — la mission n'est peut-être pas terminée.", a.maxSteps)
}

// isLoop arrête le tour si l'agent redemande exactement la/les même(s)
// action(s) qu'à l'étape précédente : aucun progrès, on évite la boucle (vaut
// pour le function calling natif comme pour le parsing texte).
func (a *Agent) isLoop(sig string, step int) bool {
	if step > 0 && sig == a.lastStepSig {
		a.ui.banner(cYellow, "BOUCLE", "Action identique répétée — arrêt pour éviter une boucle (l'agent ne progresse plus).")
		return true
	}
	a.lastStepSig = sig
	return false
}

// actionSig / toolCallsSig produisent une signature stable d'une action, pour
// comparer deux étapes successives.
func actionSig(name string, args map[string]string) string {
	return name + "(" + formatArgs(args) + ")"
}

func toolCallsSig(calls []ToolCall) string {
	parts := make([]string, 0, len(calls))
	for _, c := range calls {
		parts = append(parts, c.Function.Name+"("+c.Function.Arguments+")")
	}
	return strings.Join(parts, ";")
}

// runTool affiche l'action, applique le garde-fou, exécute l'outil, gère les
// erreurs et tronque la sortie. Renvoie le texte (résultat ou message d'erreur)
// à réinjecter au modèle.
func (a *Agent) runTool(ctx context.Context, name string, args map[string]string) string {
	a.ui.toolCallView(name, args) // partie "action" de la boucle

	tool, ok := a.registry.Get(name)
	if !ok {
		feedback := fmt.Sprintf("ERREUR: outil inconnu %q. Outils disponibles: %s", name, strings.Join(a.registry.Names(), ", "))
		a.ui.toolResultView(feedback)
		return feedback
	}

	if tool.Confirm != nil {
		if need, why := tool.Confirm(args); need {
			if !a.confirm(why) {
				feedback := "ERREUR: action refusée par l'utilisateur. Propose une alternative ou demande des précisions avant de continuer."
				a.ui.toolResultView(feedback)
				return feedback
			}
		}
	}

	a.ui.logStep("OUTIL", "exécution %s %s", name, formatArgs(args))
	result, err := tool.Run(ctx, args)
	if err != nil {
		// L'erreur est réinjectée pour permettre l'auto-correction du modèle.
		feedback := fmt.Sprintf("ERREUR outil %s: %v\nFormat attendu: %s", name, err, tool.Usage)
		a.ui.toolResultView(feedback)
		return feedback
	}
	out := truncateOutput(result, maxToolOutput)
	a.ui.toolResultView(out) // partie "feedback" de la boucle
	return out
}

// confirm demande une validation interactive à l'utilisateur.
func (a *Agent) confirm(reason string) bool {
	a.ui.printf("\n%s %s\n%s ", a.ui.colorize(cYellow+cBold, "[CONFIRMATION]"), reason, a.ui.colorize(cYellow, "Exécuter ? [o/N]"))
	if !a.scanner.Scan() {
		return false
	}
	answer := strings.ToLower(strings.TrimSpace(a.scanner.Text()))
	return answer == "o" || answer == "oui" || answer == "y" || answer == "yes"
}

// compactContext applique une fenêtre par tokens : si le budget est dépassé,
// les plus vieux messages sont résumés et remplacés par une note système, les
// messages récents étant conservés intacts.
func (a *Agent) compactContext(ctx context.Context) {
	if totalTokens(a.history) <= a.maxCtx {
		return
	}

	// Séparer les messages système initiaux du reste.
	i := 0
	for i < len(a.history) && a.history[i].Role == "system" {
		i++
	}
	systems := a.history[:i]
	rest := a.history[i:]

	// Garder les messages récents tenant dans ~60% du budget.
	keepBudget := a.maxCtx * 6 / 10
	keepFrom := len(rest)
	used := 0
	for j := len(rest) - 1; j >= 0; j-- {
		used += estimateTokens(rest[j])
		if used > keepBudget {
			break
		}
		keepFrom = j
	}
	if keepFrom <= 0 {
		return // impossible de compacter sans tout perdre
	}

	older := rest[:keepFrom]
	recent := stripLeadingToolMessages(rest[keepFrom:]) // cohérence function calling

	summary, err := a.client.Summarize(ctx, older)
	if err != nil {
		a.ui.logStep("HISTORIQUE", "résumé impossible (%v) : troncature simple", err)
		a.history = append(append([]Message{}, systems...), recent...)
		return
	}

	a.ui.logStep("HISTORIQUE", "compactage: %d messages résumés", len(older))
	out := make([]Message, 0, len(systems)+1+len(recent))
	out = append(out, systems...)
	out = append(out, Message{Role: "system", Content: "Résumé des échanges précédents :\n" + summary})
	out = append(out, recent...)
	a.history = out
}

// stripLeadingToolMessages retire d'éventuels messages "tool" orphelins en tête
// (un message tool doit suivre l'assistant qui a émis l'appel correspondant).
func stripLeadingToolMessages(msgs []Message) []Message {
	k := 0
	for k < len(msgs) && msgs[k].Role == "tool" {
		k++
	}
	return msgs[k:]
}

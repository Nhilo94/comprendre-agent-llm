package main

import (
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Icônes de la vue d'échange (la boucle agentique vue par l'utilisateur).
const (
	iUser   = "🧑"
	iAgent  = "🤖"
	iThink  = "💭"
	iTool   = "🔧"
	iResult = "📥"
	iTokens = "🧮"
	iDone   = "✓"
	iErr    = "✗"
)

// Codes ANSI pour la coloration.
const (
	cReset   = "\033[0m"
	cBold    = "\033[1m"
	cDim     = "\033[2m"
	cRed     = "\033[31m"
	cGreen   = "\033[32m"
	cYellow  = "\033[33m"
	cBlue    = "\033[34m"
	cMagenta = "\033[35m"
	cCyan    = "\033[36m"
	cGray    = "\033[90m"
)

// UI centralise tout l'affichage terminal. En l'injectant dans le client et
// l'agent (plutôt que d'imprimer via des fonctions globales), on rend les
// sorties configurables (couleur, logs activables) et testables : il suffit de
// passer un autre io.Writer.
//
// Deux flux distincts, pour bien voir comment marche l'agent :
//   - la "vue de l'échange" (toujours affichée) : la boucle observable
//     message → réflexion → appel d'outil → résultat → réponse.
//   - les "logs de debug" (champ debug) : la mécanique interne (HTTP, parsing…).
type UI struct {
	out         io.Writer
	color       bool
	debug       bool
	fullHistory bool
}

func NewUI(out io.Writer, c *Config) *UI {
	return &UI{out: out, color: c.Color, debug: c.Debug, fullHistory: c.FullHistory}
}

func (u *UI) colorize(color, s string) string {
	if !u.color || color == "" {
		return s
	}
	return color + s + cReset
}

// Petits wrappers d'écriture (toute sortie passe par u.out).
func (u *UI) printf(format string, a ...any) { fmt.Fprintf(u.out, format, a...) }
func (u *UI) println(s string)               { fmt.Fprintln(u.out, s) }

// --- Logs de debug (mécanique interne) ---

func (u *UI) logStep(label, format string, args ...any) {
	if !u.debug {
		return
	}
	ts := u.colorize(cGray, time.Now().Format("15:04:05"))
	sep := u.colorize(cGray, "│")
	lbl := u.colorize(labelColor(label), fmt.Sprintf("%-*s", logLabelWidth, label))
	msg := formatLogMessage(fmt.Sprintf(format, args...))
	fmt.Fprintf(u.out, "%s %s %s %s\n", ts, sep, lbl, msg)
}

func (u *UI) logHistory(history []Message) {
	if !u.debug {
		return
	}
	// Par défaut : une seule ligne (montre la croissance du contexte envoyé au
	// modèle à chaque étape). Le détail message-par-message est réservé au mode
	// verbeux pour ne pas noyer la vue de l'échange.
	u.logStep("HISTORIQUE", "%d message(s) envoyés au modèle (~%d tokens)", len(history), totalTokens(history))
	if !u.fullHistory {
		return
	}
	indent := strings.Repeat(" ", len("15:04:05")+3) // aligné sous les libellés
	for i, message := range history {
		branch := "├─"
		if i == len(history)-1 {
			branch = "└─"
		}
		role := u.colorize(roleColor(message.Role), fmt.Sprintf("%-9s", message.Role))
		content := formatLogMessage(message.Content)
		if len(message.ToolCalls) > 0 {
			calls := make([]string, 0, len(message.ToolCalls))
			for _, tc := range message.ToolCalls {
				calls = append(calls, tc.Function.Name+"("+tc.Function.Arguments+")")
			}
			content = u.colorize(cYellow, "→ "+strings.Join(calls, ", ")) + " " + content
		}
		fmt.Fprintf(u.out, "%s%s #%02d %s %s\n", indent, u.colorize(cGray, branch), i+1, role, content)
	}
}

// --- Vue de l'échange (pédagogique) ---
//   ▸ Étape N   → une itération de la boucle
//   🤖 agent    → ce que "pense"/répond le modèle
//   🔧 outil    → l'action qu'il décide de lancer (et ses arguments)
//   📥 résultat → l'observation réinjectée au modèle (le feedback)
//   ✓ terminé   → plus d'action : la main revient à l'utilisateur

// banner : en-tête mis en avant (erreurs, limites, interruptions).
func (u *UI) banner(color, tag, format string, args ...any) {
	fmt.Fprintf(u.out, "\n%s %s\n", u.colorize(color+cBold, "["+tag+"]"), fmt.Sprintf(format, args...))
}

func (u *UI) stepHeader(step, max int) {
	fmt.Fprintf(u.out, "\n%s %s\n", u.colorize(cGray, "▸"), u.colorize(cBold, fmt.Sprintf("Étape %d/%d", step, max)))
}

func (u *UI) agentLine(content string) {
	fmt.Fprintf(u.out, "%s %s\n", u.colorize(cCyan, iAgent), content)
}

// Streaming : en-tête imprimé une fois, puis les tokens à la volée.
func (u *UI) streamPrefix()        { fmt.Fprintf(u.out, "%s ", u.colorize(cCyan, iAgent)) }
func (u *UI) streamToken(s string) { fmt.Fprint(u.out, s) }
func (u *UI) streamEnd()           { fmt.Fprintln(u.out) }

// Bloc de raisonnement streamé (modèles "thinking"). On ouvre la couleur grisée
// au début et on la referme à la fin, pour teinter tout le bloc sans ré-émettre
// de code couleur à chaque token.
func (u *UI) reasoningStart() {
	if u.color {
		fmt.Fprintf(u.out, "%s %s", u.colorize(cGray, "💭"), cDim)
	} else {
		fmt.Fprint(u.out, "💭 ")
	}
}

func (u *UI) reasoningEnd() {
	if u.color {
		fmt.Fprint(u.out, cReset)
	}
	fmt.Fprintln(u.out)
}

// reasoningBlock affiche un raisonnement déjà complet (chemin non-streamé).
func (u *UI) reasoningBlock(s string) {
	fmt.Fprintf(u.out, "%s %s\n", u.colorize(cGray, "💭"), u.colorize(cDim, formatLogMessage(s)))
}

// tokenUsage affiche le décompte réel de tokens renvoyé par le serveur, plus le
// cumul de la session (utile pour comprendre/maîtriser le coût d'un agent).
func (u *UI) tokenUsage(t *Usage, cumTotal int) {
	line := fmt.Sprintf("🧮 tokens : %d contexte · %d réponse · %d total  ·  cumul session %d",
		t.PromptTokens, t.CompletionTokens, t.TotalTokens, cumTotal)
	fmt.Fprintf(u.out, "%s\n", u.colorize(cGray, line))
}

// toolCallView affiche l'action décidée par le modèle, avant exécution.
func (u *UI) toolCallView(name string, args map[string]string) {
	fmt.Fprintf(u.out, "%s %s %s\n", u.colorize(cYellow, iTool), u.colorize(cYellow+cBold, name), u.colorize(cDim, formatArgs(args)))
}

// toolResultView affiche un aperçu du résultat — c'est l'observation que le
// modèle recevra à l'étape suivante (la moitié "feedback" de la boucle).
func (u *UI) toolResultView(result string) {
	preview, meta := previewResult(result)
	icon, color := iResult, cGreen
	if strings.HasPrefix(strings.TrimSpace(result), "ERREUR") {
		icon, color = iErr, cRed
	}
	fmt.Fprintf(u.out, "%s %s %s\n", u.colorize(color, icon), preview, u.colorize(cDim, meta))
}

// agentDone signale la fin du tour (plus aucune action à exécuter).
func (u *UI) agentDone(steps int) {
	u.logStep("BOUCLE", "fin du tour, retour à l'utilisateur")
	fmt.Fprintf(u.out, "%s %s\n", u.colorize(cGreen, iDone), u.colorize(cDim, fmt.Sprintf("terminé en %d étape(s)", steps)))
}

// welcome affiche l'en-tête de démarrage + une légende des symboles.
func (u *UI) welcome(c *LLMClient, cfg *Config) {
	u.println("")
	u.println(u.colorize(cCyan+cBold, "  ⚙  Agent de codage"))
	u.printf("  %s\n", u.colorize(cDim, fmt.Sprintf("modèle %s · streaming %s · %d étapes max · contexte ~%d tokens",
		c.Model, onOff(c.Stream), cfg.MaxSteps, cfg.MaxContext)))
	u.printf("  %s\n", u.colorize(cGray, fmt.Sprintf("légende : %s vous   %s agent   %s outil   %s résultat", iUser, iAgent, iTool, iResult)))
	u.printf("  %s\n", u.colorize(cGray, "tape une mission · 'exit'/'quit' pour quitter · Ctrl-C interrompt un tour"))
}

// --- Helpers de formatage (purs, sans couleur : testables isolément) ---

// labelColor associe une couleur à chaque catégorie de log.
func labelColor(label string) string {
	switch label {
	case "LLM":
		return cCyan
	case "OUTIL":
		return cYellow
	case "PARSER":
		return cMagenta
	case "HISTORIQUE":
		return cGray
	case "MODELES":
		return cBlue
	case "BOUCLE":
		return cGreen
	case "INIT", "USER":
		return cBold
	case "ERREUR":
		return cRed
	default:
		return ""
	}
}

func roleColor(role string) string {
	switch role {
	case "system":
		return cMagenta
	case "user":
		return cGreen
	case "assistant":
		return cCyan
	case "tool":
		return cYellow
	default:
		return ""
	}
}

// formatLogMessage met le message sur une seule ligne (saut de ligne -> ⏎)
// et le tronque pour garder des logs lisibles et alignés.
func formatLogMessage(msg string) string {
	msg = strings.ReplaceAll(msg, "\r\n", "\n")
	msg = strings.ReplaceAll(msg, "\n", " ⏎ ")
	runes := []rune(msg)
	if len(runes) > logMsgMaxLen {
		msg = string(runes[:logMsgMaxLen]) + "…"
	}
	return msg
}

// formatArgs met les arguments d'un appel d'outil sur une ligne lisible.
func formatArgs(args map[string]string) string {
	if len(args) == 0 {
		return "(sans argument)"
	}
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%q", k, formatLogMessage(args[k])))
	}
	return strings.Join(parts, " ")
}

// previewResult extrait la première ligne significative + des métadonnées
// (nombre de lignes, taille) pour donner un aperçu sans tout déverser.
func previewResult(s string) (preview, meta string) {
	nBytes := len(s)
	nLines := strings.Count(strings.TrimRight(s, "\n"), "\n") + 1
	first, _, _ := strings.Cut(s, "\n")
	first = formatLogMessage(strings.TrimSpace(first))
	runes := []rune(first)
	if len(runes) > resultPreviewLen {
		first = string(runes[:resultPreviewLen]) + "…"
	}
	if first == "" {
		first = "(vide)"
	}
	return first, fmt.Sprintf("(%d ligne(s), %d octets)", nLines, nBytes)
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

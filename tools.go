package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
)

// --- Outils & Registre ---
// Un outil est une fonction pure (ctx, args) → (résultat, erreur) : il ne fait
// aucun affichage lui-même. La présentation (appel, résultat) est centralisée
// dans l'agent, ce qui garde les outils simples, testables et réutilisables.

type ToolFunc func(ctx context.Context, args map[string]string) (string, error)

type Tool struct {
	Name        string
	Description string
	Usage       string
	Parameters  map[string]any                                       // JSON Schema des arguments
	Confirm     func(args map[string]string) (need bool, why string) // garde-fou optionnel
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

// Prompt génère la description des outils pour le prompt système (utile pour
// les modèles qui n'exploitent pas le function calling natif).
func (r *ToolRegistry) Prompt() string {
	var b strings.Builder
	for _, name := range r.order {
		t := r.tools[name]
		fmt.Fprintf(&b, "- %s : %s\n  Usage : %s\n", t.Name, t.Description, t.Usage)
	}
	return b.String()
}

// Schemas convertit le registre au format "tools" attendu par l'API.
func (r *ToolRegistry) Schemas() []ToolSchema {
	out := make([]ToolSchema, 0, len(r.order))
	for _, name := range r.order {
		t := r.tools[name]
		params := t.Parameters
		if params == nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, ToolSchema{
			Type: "function",
			Function: FunctionSchema{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}
	return out
}

// registerDefaultTools enregistre les outils de base de l'agent.
func registerDefaultTools(r *ToolRegistry) {
	r.Register(Tool{
		Name:        "execute_shell",
		Description: "Exécute une commande shell (Linux/macOS), supporte pipes et redirections.",
		Usage:       `execute_shell(command="ls -la")`,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{"type": "string", "description": "La commande shell à exécuter"},
			},
			"required": []string{"command"},
		},
		Confirm: func(args map[string]string) (bool, string) {
			return dangerousCommand(args["command"])
		},
		Run: toolExecuteShell,
	})
	r.Register(Tool{
		Name:        "read_file",
		Description: "Lit le contenu d'un fichier texte.",
		Usage:       `read_file(path="chemin/vers/fichier.txt")`,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{"type": "string", "description": "Chemin du fichier à lire"},
			},
			"required": []string{"path"},
		},
		Run: toolReadFile,
	})
	r.Register(Tool{
		Name:        "write_file",
		Description: "Écrit (ou écrase) un fichier texte avec le contenu fourni.",
		Usage:       `write_file(path="fichier.txt", content="bonjour")`,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":    map[string]any{"type": "string", "description": "Chemin du fichier à écrire"},
				"content": map[string]any{"type": "string", "description": "Contenu à écrire dans le fichier"},
			},
			"required": []string{"path", "content"},
		},
		Run: toolWriteFile,
	})
}

// dangerousPatterns : commandes qui déclenchent une demande de confirmation.
var dangerousPatterns = []*regexp.Regexp{
	regexp.MustCompile(`\brm\s+-[a-zA-Z]*[rf]`), // rm -r / -rf / -fr
	regexp.MustCompile(`\bmkfs\b`),
	regexp.MustCompile(`\bdd\s+if=`),
	regexp.MustCompile(`>\s*/dev/sd`),
	regexp.MustCompile(`:\s*\(\)\s*\{`), // fork bomb
	regexp.MustCompile(`\bgit\s+reset\s+--hard\b`),
	regexp.MustCompile(`\bgit\s+clean\s+-[a-zA-Z]*f`),
	regexp.MustCompile(`\b(shutdown|reboot|halt)\b`),
	regexp.MustCompile(`\bchmod\s+-R\b`),
	regexp.MustCompile(`\bsudo\b`),
}

func dangerousCommand(cmd string) (bool, string) {
	for _, re := range dangerousPatterns {
		if re.MatchString(cmd) {
			return true, fmt.Sprintf("Commande potentiellement destructive : %s", cmd)
		}
	}
	return false, ""
}

// toolExecuteShell : "sh -c" permet pipes (|), redirections (>) et variables.
// Le context (annulation Ctrl-C) est combiné à un timeout : une commande
// bloquante ne fige pas l'agent. Les échecs de commande sont renvoyés comme
// résultat (pas comme erreur Go) afin que le modèle puisse les analyser et
// réessayer.
func toolExecuteShell(ctx context.Context, args map[string]string) (string, error) {
	command := strings.TrimSpace(args["command"])
	if command == "" {
		return "", fmt.Errorf(`argument "command" manquant ou vide`)
	}

	ctx, cancel := context.WithTimeout(ctx, shellTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return fmt.Sprintf("ERREUR : commande interrompue après %s (timeout).\nStdout: %s\nStderr: %s", shellTimeout, out.String(), stderr.String()), nil
	}
	if err != nil {
		return fmt.Sprintf("ERREUR : %v\nStderr: %s", err, stderr.String()), nil
	}
	return fmt.Sprintf("Succès : %s", out.String()), nil
}

func toolReadFile(_ context.Context, args map[string]string) (string, error) {
	path := strings.TrimSpace(args["path"])
	if path == "" {
		return "", fmt.Errorf(`argument "path" manquant`)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("lecture impossible: %w", err)
	}
	return fmt.Sprintf("Contenu de %s :\n%s", path, string(data)), nil
}

func toolWriteFile(_ context.Context, args map[string]string) (string, error) {
	path := strings.TrimSpace(args["path"])
	if path == "" {
		return "", fmt.Errorf(`argument "path" manquant`)
	}
	content := args["content"]

	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("écriture impossible: %w", err)
	}
	return fmt.Sprintf("Fichier écrit: %s (%d octets)", path, len(content)), nil
}

// truncateOutput plafonne une sortie d'outil pour ne pas saturer le contexte.
func truncateOutput(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	cut := strings.ToValidUTF8(s[:maxBytes], "")
	return cut + fmt.Sprintf("\n…(sortie tronquée, %d octets supprimés)", len(s)-len(cut))
}

// --- Parsing de secours (fallback texte si pas de tool_calls natifs) ---
// Fonctions pures : elles ne loguent rien (l'agent journalise le résultat).

// argRe capture chaque paire clé="valeur" en tolérant espaces et guillemets
// échappés (\").
var argRe = regexp.MustCompile(`([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"`)

// parseAction détecte un appel d'outil écrit en texte par le modèle (pour les
// modèles sans function calling natif). On exige le préfixe "Action:" EN DÉBUT
// DE LIGNE — c'est la convention donnée dans le prompt système. Cette ancre
// évite un piège classique : quand le modèle RÉCAPITULE ("1. Action: write_file
// (...)"), il cite l'action sans la redemander ; sans l'ancre, on la
// ré-exécuterait en boucle.
func parseAction(reply string, registry *ToolRegistry) (name string, args map[string]string, found bool) {
	names := registry.Names()
	if len(names) == 0 {
		return "", nil, false
	}

	// (?m) : ^ s'ancre en début de ligne ; (?s) : . traverse les sauts de ligne
	// (arguments multi-lignes). \s* ne mange que de l'espace, donc "1. Action:"
	// (précédé de "1. ") ne matche pas.
	pattern := `(?sm)^[ \t]*Action\s*:\s*(` + strings.Join(names, "|") + `)\s*\(\s*(.*)\)`
	re := regexp.MustCompile(pattern)

	m := re.FindStringSubmatch(reply)
	if m == nil {
		return "", nil, false
	}
	return m[1], parseArgs(m[2]), true
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

// parseJSONArgs convertit les arguments JSON d'un tool_call natif en
// map[string]string (les valeurs non-string sont formatées). Un JSON invalide
// renvoie une map vide : l'outil signalera alors l'argument manquant.
func parseJSONArgs(raw string) map[string]string {
	out := make(map[string]string)
	if strings.TrimSpace(raw) == "" {
		return out
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		return out
	}
	for k, v := range m {
		if s, ok := v.(string); ok {
			out[k] = s
		} else {
			out[k] = fmt.Sprintf("%v", v)
		}
	}
	return out
}

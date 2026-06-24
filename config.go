package main

import (
	"flag"
	"os"
	"time"
)

// Constantes opérationnelles : réglages internes non destinés à l'utilisateur
// final (un flag pour chacun serait du bruit). Les réglages utiles sont dans
// Config, pilotés par des flags.
const (
	shellTimeout   = 60 * time.Second  // timeout d'une commande shell
	requestTimeout = 600 * time.Second // timeout d'un appel au modèle
	modelsTimeout  = 10 * time.Second  // timeout du listing des modèles
	maxToolOutput  = 8000              // octets max d'une sortie d'outil réinjectée

	logLabelWidth    = 10  // alignement des libellés de log
	logMsgMaxLen     = 200 // longueur max d'un message de log
	resultPreviewLen = 120 // longueur max de l'aperçu d'un résultat d'outil

	defaultBaseURL = "http://localhost:1234"
	defaultModel   = "local-model"
)

// Config regroupe tous les réglages runtime de l'agent. Renseignée depuis les
// flags de ligne de commande, avec repli sur les variables d'environnement puis
// les valeurs par défaut. Centraliser ainsi la configuration évite les
// constantes figées dispersées et rend le comportement réglable sans recompiler.
type Config struct {
	BaseURL     string
	Model       string // "" => sélection interactive au démarrage
	Temperature float64
	MaxTokens   int
	Stream      bool
	MaxSteps    int
	MaxContext  int

	// Affichage
	Color       bool // coloration ANSI
	Debug       bool // logs internes horodatés
	FullHistory bool // détail message-par-message de l'historique
}

// LoadConfig lit les flags + l'environnement et renvoie la configuration.
// Précédence : flag explicite > variable d'environnement > valeur par défaut.
func LoadConfig() *Config {
	c := &Config{}
	var quiet, verbose, noColor, noStream bool

	flag.StringVar(&c.BaseURL, "url", envOr("AGENT_BASE_URL", defaultBaseURL), "URL du serveur LLM (compatible OpenAI)")
	flag.StringVar(&c.Model, "model", os.Getenv("AGENT_MODEL"), "modèle à utiliser (vide = choix interactif)")
	flag.Float64Var(&c.Temperature, "temp", 0.7, "température d'échantillonnage")
	flag.IntVar(&c.MaxTokens, "max-tokens", 0, "tokens max par réponse (0 = illimité)")
	flag.IntVar(&c.MaxSteps, "max-steps", 10, "nombre max d'étapes par tour d'agent")
	flag.IntVar(&c.MaxContext, "max-context", 6000, "budget contexte (tokens) avant compactage")
	flag.BoolVar(&noStream, "no-stream", false, "désactive le streaming token-par-token")
	flag.BoolVar(&quiet, "quiet", false, "n'affiche que l'échange (masque les logs de debug)")
	flag.BoolVar(&verbose, "verbose", false, "logs de debug + historique complet à chaque étape")
	flag.BoolVar(&noColor, "no-color", false, "désactive la coloration ANSI")
	flag.Parse()

	c.Stream = !noStream
	c.Debug = verbose || !quiet // -quiet coupe les logs ; -verbose les force
	c.FullHistory = verbose
	c.Color = !noColor && colorSupported()
	return c
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// colorSupported désactive la couleur si la sortie est redirigée (pipe, fichier)
// ou si la convention NO_COLOR est respectée.
func colorSupported() bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

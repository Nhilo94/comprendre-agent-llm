# Comprendre les agents LLM de codage

> Un agent de codage **réel, minimal et commenté** — ~1000 lignes de Go,
> **zéro dépendance** (bibliothèque standard uniquement). Conçu pour
> *comprendre* comment fonctionne un agent : la boucle, les outils, le
> function calling, le streaming, les garde-fous et la gestion du contexte.

🔗 **Guide pédagogique illustré : [agentic.ploy.irisoa-network.space](https://agentic.ploy.irisoa-network.space)**

Cet agent parle à **n'importe quel serveur compatible OpenAI** : LM Studio,
Ollama, llama.cpp, vLLM, ou une API cloud. Pas de framework, pas de SDK —
chaque mécanisme reste lisible et démystifié.

```
🧑 vous › ajoute un test pour la fonction Somme et lance-le
▸ Étape 1/10
💭 Je dois d'abord lire le fichier pour voir la signature…
🔧 read_file      path="math.go"
📥 Contenu de math.go : func Somme(a, b int) int { … }
▸ Étape 2/10
🔧 write_file     path="math_test.go"
📥 Fichier écrit: math_test.go (212 octets)
▸ Étape 3/10
🔧 execute_shell  command="go test ./..."
📥 Succès : ok  agentic  0.004s
✓ terminé en 3 étape(s)
```

## Démarrage

Prérequis : **Go ≥ 1.26** et un serveur LLM compatible OpenAI accessible
(par défaut `http://localhost:1234`, comme LM Studio).

```bash
git clone https://github.com/Nhilo94/comprendre-agent-llm.git
cd comprendre-agent-llm
go run .
```

Au lancement, l'agent liste les modèles du serveur et vous laisse en choisir un,
puis ouvre une invite de discussion. Tapez `exit` pour quitter, `Ctrl-C` pour
interrompre un tour en cours.

### Réglages utiles (flags)

| Flag             | Défaut                  | Rôle                                            |
| ---------------- | ----------------------- | ----------------------------------------------- |
| `-url`           | `http://localhost:1234` | URL du serveur LLM (ou `AGENT_BASE_URL`)        |
| `-model`         | *(choix interactif)*    | modèle à utiliser (ou `AGENT_MODEL`)            |
| `-temp`          | `0.7`                   | température d'échantillonnage                    |
| `-max-steps`     | `10`                    | nombre max d'étapes par tour                     |
| `-max-context`   | `6000`                  | budget contexte (tokens) avant compactage        |
| `-no-stream`     | `false`                 | désactive le streaming token-par-token           |
| `-quiet`         | `false`                 | masque les logs internes (n'affiche que l'échange) |
| `-verbose`       | `false`                 | logs détaillés + historique complet              |

Exemple : `go run . -url http://localhost:11434 -model qwen2.5-coder -quiet`

## Anatomie (6 fichiers)

| Fichier       | Rôle                                                                       |
| ------------- | -------------------------------------------------------------------------- |
| `main.go`     | Point d'entrée, boucle de lecture (REPL), prompt système, choix du modèle. |
| `agent.go`    | **La boucle agentique** : `handleTurn`, exécution d'outils, anti-boucle, compactage. |
| `tools.go`    | Définition des outils, registre, `read_file` / `write_file` / `execute_shell`, garde-fous, repli texte. |
| `llm.go`      | Client LLM compatible OpenAI : function calling, streaming SSE, résumé, comptage de tokens. |
| `ui.go`       | Affichage terminal : la « vue de l'échange » pédagogique + logs de debug.  |
| `config.go`   | Réglages runtime (flags + variables d'environnement).                      |

## Comment ça marche, en bref

1. **La boucle** — à chaque tour, on envoie l'historique au modèle. S'il demande
   un outil, on l'exécute et on lui renvoie le résultat, puis on recommence.
   Sinon, le tour est terminé.
2. **Les outils** — de simples fonctions `(ctx, args) → (résultat, erreur)`,
   décrites au modèle par un schéma JSON. Le modèle choisit lequel appeler.
3. **Parler au modèle** — function calling natif, avec un repli par parsing texte
   (`Action: outil(arg="…")`) pour les modèles qui ne le supportent pas.
4. **Les garde-fous** — confirmation des commandes dangereuses, détection de
   boucle, timeouts, troncature des sorties. Les erreurs sont réinjectées au
   modèle pour qu'il se corrige.
5. **Le contexte** — quand l'historique dépasse le budget, les anciens messages
   sont résumés ; les récents restent intacts.

Le détail, illustré et expliqué pas à pas, est dans le
**[guide pédagogique](https://agentic.ploy.irisoa-network.space)**.

## Licence

[MIT](LICENSE) — © 2026 Fanilo Rakotovao.

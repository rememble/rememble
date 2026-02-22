# Rememble

Local-first MCP memory server with hybrid search, knowledge graph, and RAG context assembly.

SQLite + sqlite-vec + FTS5 backend. Works with any MCP client.

## Features

- **Hybrid search** — BM25 full-text + vector KNN + temporal scoring, fused with RRF
- **Knowledge graph** — entities, observations, relations with fuzzy search
- **RAG context** — token-budgeted context assembly with expansion + snippets
- **Auto-chunking** — long text split with sliding window overlap
- **Multiple providers** — Ollama, OpenAI, OpenRouter, Cohere (any OpenAI-compatible API)
- **Auto-dimension detection** — probes provider at startup, migrates DB automatically on change
- **Automatic memory capture** — Claude Code hooks for session recall, contextual injection, and transcript summarization
- **Setup wizard** — auto-detects and configures Claude Code, Claude Desktop, Cursor, Windsurf, OpenCode, Codex

## Install

```bash
uv tool install rememble
```

Upgrade:

```bash
uv tool upgrade rememble
```

## Quick Start

```bash
rememble setup
```

The setup wizard will:

1. **Configure embeddings** — pick a provider (Ollama, OpenAI, OpenRouter, Cohere) and model
2. **Configure agents** — auto-detect installed AI agents and register rememble as an MCP server
3. **Install hooks** — (Claude Code only) register session-start, prompt-submit, and session-end hooks

Non-interactive:

```bash
rememble setup --provider cohere --api-key YOUR_KEY --model embed-english-v3.0 --agents claude-code --yes
```

Provider slugs: `ollama`, `openai`, `openrouter`, `cohere`
Agent slugs: `claude-code`, `claude-desktop`, `opencode`, `codex`, `cursor`, `windsurf`

## MCP Tools

Once configured, your AI agent gets these tools:

| Tool | Description |
|------|-------------|
| `remember` | Store a memory (auto-chunks, embeds, indexes) |
| `recall` | Semantic search with RAG context assembly |
| `forget` | Soft-delete a memory |
| `list_memories` | Browse memories with filters |
| `memory_stats` | DB statistics and provider info |
| `create_entities` | Create knowledge graph entities with observations |
| `create_relations` | Link entities in the knowledge graph |
| `add_observations` | Add facts to existing entities |
| `search_graph` | Search entities and observations |
| `delete_entities` | Remove entities (cascades to relations) |

## CLI

Rememble also has a CLI that talks to the HTTP API server:

```bash
rememble serve -d                   # start HTTP server as daemon
rememble remember "project uses uv" --source manual --tags tools
rememble recall "what tools does the project use"
rememble list --source manual
rememble stats
rememble forget 42
rememble entity create --name Python --type language
rememble graph search "Python"
rememble stop                       # stop daemon
rememble hook session-start         # (used by hooks, not direct use)
rememble hook prompt-submit
rememble hook session-end
```

## Hooks (Claude Code)

When set up with Claude Code, rememble installs three [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks) for automatic memory capture:

| Hook | Event | What it does |
|------|-------|--------------|
| `session-start` | Startup / resume | Recalls recent project context and injects it into the session |
| `prompt-submit` | Each user prompt | Searches memories relevant to the prompt and injects as context (5s timeout, skips trivial inputs) |
| `session-end` | Session close | Summarizes the transcript via LLM (Haiku), stores global + project-scoped memories |

Hooks are installed automatically by `rememble setup` into `~/.claude/settings.json`. All hooks gracefully degrade — if the rememble server is unreachable, they return empty and Claude Code continues normally.

## Configuration

Config: `~/.rememble/config.json` (created by `rememble setup` or on first run).

```bash
rememble config list                       # show all config
rememble config get embedding.api_url      # get a value
rememble config set embedding.dimensions 512  # set a value
```

All fields support env var overrides with `REMEMBLE_` prefix:

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `embedding.api_url` | `REMEMBLE_EMBEDDING_API_URL` | `http://localhost:11434/v1` | Embedding API endpoint |
| `embedding.api_key` | `REMEMBLE_EMBEDDING_API_KEY` | — | API key |
| `embedding.model` | `REMEMBLE_EMBEDDING_MODEL` | `nomic-embed-text` | Model name |
| `embedding.dimensions` | `REMEMBLE_EMBEDDING_DIMENSIONS` | `768` | Fallback dimensions (auto-detected at startup) |
| `port` | `REMEMBLE_PORT` | `9909` | HTTP server port |

Sub-configs (`search.*`, `rag.*`, `chunking.*`, `embedding.*`) are available via `rememble config list`.

## Embedding Providers

### Ollama (default, local)

```bash
rememble setup --provider ollama --model nomic-embed-text
```

Models: `nomic-embed-text` (768d), `mxbai-embed-large` (1024d), `all-minilm` (384d), `bge-large` (1024d), `snowflake-arctic-embed` (1024d)

### OpenAI

```bash
rememble setup --provider openai --api-key sk-... --model text-embedding-3-small
```

### OpenRouter

```bash
rememble setup --provider openrouter --api-key sk-or-... --model openai/text-embedding-3-small
```

### Cohere

```bash
rememble setup --provider cohere --api-key ... --model embed-english-v3.0
```

Dimensions are auto-detected from the provider at startup. If you switch providers, the DB migrates automatically (re-embeds all existing memories).

## Docker

```bash
docker compose up
```

Mounts `~/.rememble` as `/data`. Runs the HTTP API server on port 9909.

## Development

```bash
make dev      # install deps (all extras)
make test     # run tests
make lint     # ruff + basedpyright
make fmt      # format + fix imports
make check    # fmt + lint + test
```

## Uninstall

```bash
rememble uninstall
```

Removes MCP entries and instructions from all configured agents. Optionally deletes `~/.rememble/` (database + config).

## License

MIT

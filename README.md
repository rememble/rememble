# Rememble

Local-first memory server with hybrid search, knowledge graph, and RAG context assembly.

SQLite + sqlite-vec + FTS5 backend. Works with any MCP client.

## Features

- Hybrid search: BM25 + vector KNN + temporal scoring (RRF fusion)
- Knowledge graph: entities, observations, relations
- Token-budgeted RAG context assembly
- Multiple embedding providers: Ollama, local (sentence-transformers), OpenAI-compat

## Install

```bash
uv sync
```

## MCP Client Setup

### Claude Code / Claude Desktop

```json
{
  "mcpServers": {
    "rememble": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/Rememble", "rememble"]
    }
  }
}
```

## Configuration

Config file: `~/.rememble/config.json` (auto-created on first run).

All fields can be overridden via env vars with `REMEMBLE_` prefix and `__` as nested delimiter.

| Env var | Default | Description |
|---------|---------|-------------|
| `REMEMBLE_DB_PATH` | `~/.rememble/memory.db` | SQLite database path |
| `REMEMBLE_EMBEDDING__PROVIDER` | `ollama` | `ollama` \| `local` \| `compat` |
| `REMEMBLE_EMBEDDING__MODEL` | `nomic-embed-text` | Model name for active provider |
| `REMEMBLE_EMBEDDING__DIMENSIONS` | `768` | Embedding dimensions |
| `REMEMBLE_EMBEDDING__OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `REMEMBLE_EMBEDDING__API_TYPE` | `openrouter` | Label for compat provider (logging only) |
| `REMEMBLE_EMBEDDING__API_ENDPOINT` | `https://openrouter.ai/api/v1` | OpenAI-compat base URL |
| `REMEMBLE_EMBEDDING__API_KEY` | — | API key (or set `OPENROUTER_API_KEY` / `OPENAI_API_KEY`) |

## Embedding Providers

### Ollama (default)
```json
{ "embedding": { "provider": "ollama", "model": "nomic-embed-text", "dimensions": 768 } }
```

### Local (sentence-transformers, no network)
```bash
uv sync --extra local
```
```json
{ "embedding": { "provider": "local" } }
```

### OpenAI-compat (OpenRouter, OpenAI, Cohere compat, etc.)

**OpenRouter:**
```json
{
  "embedding": {
    "provider": "compat",
    "api_type": "openrouter",
    "api_endpoint": "https://openrouter.ai/api/v1",
    "model": "openai/text-embedding-3-small",
    "dimensions": 1536
  }
}
```

**Cohere via OpenAI compat API:**
```json
{
  "embedding": {
    "provider": "compat",
    "api_type": "cohere",
    "api_endpoint": "https://api.cohere.com/compatibility/v1",
    "model": "embed-english-light-v3.0",
    "dimensions": 384
  }
}
```

Set `REMEMBLE_EMBEDDING__API_KEY` or `OPENROUTER_API_KEY` / `OPENAI_API_KEY` env var.

## Development

```bash
make dev      # install deps
make test     # run tests
make check    # fmt + lint + test
```

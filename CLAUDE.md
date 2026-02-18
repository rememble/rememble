# CLAUDE.md

## Project Overview

Rememble is a local-first MCP memory server providing semantic memory for Claude Code / Claude Desktop. SQLite + sqlite-vec + FTS5 backend with hybrid search (BM25 + vector KNN + temporal scoring), knowledge graph, and token-budgeted RAG context assembly.

**Config:** `~/.rememble/config.json`
**DB:** `~/.rememble/memory.db`
**Entry point:** `rememble.server:main` (stdio transport)

## Architecture

### Core Components

- `src/rememble/server.py` — FastMCP server, tool/resource definitions, lifespan context (db + embedder init)
- `src/rememble/config.py` — Pydantic config models, loads/creates `~/.rememble/config.json`
- `src/rememble/models.py` — Pydantic models: Memory, Entity, Observation, Relation, SearchResult, FusedResult, RAGItem, RAGContext
- `src/rememble/db.py` — SQLite + sqlite-vec + FTS5 setup, migrations, CRUD for memories + knowledge graph

### Embeddings (`src/rememble/embeddings/`)

- `base.py` — `EmbeddingProvider` Protocol (`dimensions`, `name`, `embed()`, `embedOne()`)
- `ollama.py` — Ollama `/api/embed` endpoint, batch support, health check
- `local.py` — sentence-transformers ONNX backend (all-MiniLM-L6-v2, 384-dim)
- `cohere.py` — Cohere embed API v2, `forQuery()` for input_type switching
- `factory.py` — `createProvider()` with fallback chain: configured → ollama → local → error

### Search (`src/rememble/search/`)

- `vector.py` — sqlite-vec KNN search, score = 1 - distance
- `text.py` — FTS5 BM25 with porter tokenizer, AND-joined quoted tokens, OR fallback
- `temporal.py` — Three-component importance score (age + frequency + recency)
- `graph.py` — Fuzzy LIKE match on entities + observations, 1-hop relation traversal
- `fusion.py` — RRF fusion across all lanes, configurable weights

### Ingest + RAG

- `src/rememble/ingest/chunker.py` — tiktoken cl100k_base chunking with sliding window overlap
- `src/rememble/rag/context.py` — Token-budgeted context assembly (expansion + snippets + graph)

### MCP Interface

**Tools:** `remember`, `recall`, `forget`, `list_memories`, `memory_stats`, `create_entities`, `create_relations`, `add_observations`, `search_graph`, `delete_entities`

**Resources:** `memory://stats`, `memory://recent`, `memory://graph`, `memory://{memory_id}`

## Key Patterns

- **FastMCP decorators** — `@mcp.tool` / `@mcp.resource` wrap functions into `FunctionTool` / `FunctionResource` objects. Access original callable via `.fn` (relevant for testing)
- **Global state via lifespan** — `_db`, `_embedder`, `_config` set during `async lifespan()`, accessed via `_getDb()` etc. with assert guards
- **Sync DB, async embedders** — All sqlite operations are synchronous (sqlite3.Connection). Embedding providers are async (httpx). Search pipeline is sync despite async-compatible wrappers
- **Functional config** — `loadConfig()` returns frozen Pydantic model, creates defaults if `~/.rememble/config.json` missing
- **FTS5 sync triggers** — Insert/update/delete triggers keep `fts_memories` in sync with `memories` table
- **sqlite-vec virtual table** — Created separately from `executescript()` due to extension requirement

## Development

### Dependencies

Runtime: `fastmcp`, `sqlite-vec`, `tiktoken`, `pydantic`, `httpx`
Optional: `sentence-transformers` + `onnxruntime` (local embeddings), `cohere` (Cohere API)
Dev: `pytest`, `pytest-asyncio`, `ruff`, `basedpyright`

### Commands

```bash
uv sync                              # install deps
uv sync --all-extras                  # install all optional deps
uv run pytest -v                      # run tests (64 tests)
uv run ruff check src/                # lint
uv run ruff format src/               # format
uv run basedpyright src/              # type check
uv run rememble                       # start MCP server (stdio)
```

### Testing

- `conftest.py` — `FakeEmbedder` (deterministic hash-based 4-dim vectors), `tmp_db_path`, `config`, `db` fixtures
- `test_db.py` — Memory + entity CRUD, soft delete, access stats, cascades
- `test_embeddings.py` — Protocol conformance, mocked Ollama/Cohere, factory fallback chain
- `test_search.py` — Vector KNN, FTS5 BM25, temporal scoring, graph search, RRF fusion
- `test_ingest.py` — Chunking, token counting, truncation
- `test_rag.py` — Context assembly, token budget enforcement
- `test_server.py` — All MCP tools + resources via `.fn` accessor; patches `srv._db`/`srv._embedder`/`srv._config` globals

### Code Style

- Python 3.11+ (enables `tomllib` stdlib; no `type` alias statements — use `TypeAlias`)
- Ruff: `line-length = 100`, select `E,F,I,UP,B,SIM`
- basedpyright: `standard` mode
- Type hints on all function signatures
- camelCase function names (e.g., `loadConfig`, `insertMemory`, `embedOne`)
- snake_case file names

### Claude Code Integration

```json
{
  "mcpServers": {
    "rememble": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/n/Projects/Rememble", "rememble"]
    }
  }
}
```

### Docker

```bash
docker compose up                     # start with ~/.rememble:/data mount
```

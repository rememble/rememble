# CLAUDE.md

## Project Overview

Rememble is a local-first MCP memory server providing semantic memory for Claude Code / Claude Desktop. SQLite + sqlite-vec + FTS5 backend with hybrid search (BM25 + vector KNN + temporal scoring), knowledge graph, and token-budgeted RAG context assembly.

**Config:** `~/.rememble/config.json`
**DB:** `~/.rememble/memory.db`
**Entry point:** `rememble.server:main` (stdio transport)

## Architecture

### Core Components

- `rememble/server.py` — FastMCP server, tool/resource definitions, lifespan context (db + embedder init)
- `rememble/config.py` — Pydantic config models, loads/creates `~/.rememble/config.json`
- `rememble/models.py` — Pydantic models: Memory, Entity, Observation, Relation, SearchResult, FusedResult, RAGItem, RAGContext
- `rememble/db.py` — SQLite + sqlite-vec + FTS5 setup, migrations, CRUD for memories + knowledge graph

### Embeddings (`rememble/embeddings/`)

- `base.py` — `EmbeddingProvider` Protocol (`dimensions`, `name`, `embed()`, `embedOne()`)
- `ollama.py` — Ollama `/api/embed` endpoint, batch support, health check
- `local.py` — sentence-transformers ONNX backend (all-MiniLM-L6-v2, 384-dim)
- `compat.py` — OpenAI-compatible provider (OpenRouter, OpenAI, Cohere compat, etc.)
- `factory.py` — `createProvider()` with fallback chain: configured → ollama → local → error

### Search (`rememble/search/`)

- `vector.py` — sqlite-vec KNN search, score = 1 - distance
- `text.py` — FTS5 BM25 with porter tokenizer, AND-joined quoted tokens, OR fallback
- `temporal.py` — ACT-R base-level activation: `B_i(t) = ln(Σ(t-t_j)^(-d))`, sigmoid-normalized to [0,1]. Tracks per-memory access history (JSON array of epoch-second timestamps, max 50). Legacy `temporalScore()` synthesizes history from count+timestamps for backward compat
- `graph.py` — Fuzzy LIKE match on entities + observations, 1-hop relation traversal
- `fusion.py` — RRF fusion across all lanes, configurable weights. Uses ACT-R activation from `access_history_json` with legacy fallback
- `need.py` — Deterministic memory need classifier (<1ms, no LLM). Classifies queries as none/temporal/identity/fact_lookup/open_loop/broad_context/prospective/general via compiled regex. Skips recall for acks, short queries. Configurable via `need_analysis.enabled`

### Ingest + RAG

- `rememble/ingest/chunker.py` — tiktoken cl100k_base chunking with sliding window overlap
- `rememble/rag/context.py` — Token-budgeted context assembly (expansion + snippets + graph)

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
Optional: `sentence-transformers` + `onnxruntime` (local embeddings)
Dev: `pytest`, `pytest-asyncio`, `ruff`, `basedpyright`

### Commands

```bash
uv sync                              # install deps
uv sync --all-extras                  # install all optional deps
uv run pytest -v                      # run tests (284 tests)
uv run ruff check rememble/           # lint
uv run ruff format rememble/          # format
uv run basedpyright rememble/         # type check
uv run rememble                       # start MCP server (stdio)
```

### Testing

- `conftest.py` — `FakeEmbedder` (deterministic hash-based 4-dim vectors), `tmp_db_path`, `config`, `db` fixtures
- `test_db.py` — Memory + entity CRUD, soft delete, access stats, cascades
- `test_embeddings.py` — Protocol conformance, mocked Ollama/Compat, factory fallback chain
- `test_search.py` — Vector KNN, FTS5 BM25, temporal scoring, graph search, RRF fusion
- `test_activation.py` — ACT-R activation scoring, synthesizeHistory, access history DB ops, v2→v3 migration
- `test_need.py` — Memory need analysis: skip/temporal/identity/open_loop/broad_context/prospective/fact_lookup patterns
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

```bash
uv tool install rememble
```

```json
{
  "mcpServers": {
    "rememble": {
      "command": "rememble"
    }
  }
}
```

### Docker

```bash
docker compose up                     # start with ~/.rememble:/data mount
```

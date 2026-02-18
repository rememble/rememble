# Rememble

Local-first MCP memory server for Claude Code / Claude Desktop.

SQLite + sqlite-vec + FTS5 backend with hybrid search (BM25 + vector KNN + temporal scoring), knowledge graph, and token-budgeted RAG context assembly.

## Install

```bash
uv sync
```

## Usage

Add to Claude Code config (`~/.claude.json`):

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

## Development

```bash
make dev      # install deps
make test     # run tests
make check    # fmt + lint + test
make release  # bump, tag, publish
```

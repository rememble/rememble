"""Rememble MCP server — memory tools, knowledge graph, and RAG."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Any, get_args, get_origin

import typer
from fastmcp import FastMCP
from pydantic import BaseModel
from rich import box
from rich.console import Console
from rich.table import Table

from rememble.config import ChunkingConfig, RAGConfig, RemembleConfig, SearchConfig, loadConfig
from rememble.db import (
    addObservation,
    addRelation,
    connect,
    deleteEntity,
    getMemory,
    insertMemory,
    listMemories,
    memoryStats,
    softDeleteMemory,
    upsertEntity,
)
from rememble.embeddings.base import EmbeddingProvider
from rememble.embeddings.factory import createProvider
from rememble.ingest.chunker import chunkText, countTokens
from rememble.rag.context import buildContext
from rememble.search.fusion import hybridSearch
from rememble.search.graph import graphSearch

logger = logging.getLogger("rememble")

# Globals set during lifespan
_db: sqlite3.Connection | None = None
_embedder: EmbeddingProvider | None = None
_config: RemembleConfig | None = None


@asynccontextmanager
async def lifespan(server):
    global _db, _embedder, _config
    _config = loadConfig()
    logger.info("Rememble starting — db: %s", _config.db_path)

    _db = connect(_config)
    _embedder = await createProvider(_config)
    logger.info("Embedding provider: %s (%d dims)", _embedder.name, _embedder.dimensions)

    yield {"db": _db, "embedder": _embedder, "config": _config}

    if _db:
        _db.close()
    logger.info("Rememble shut down.")


mcp = FastMCP("rememble", lifespan=lifespan)


def _getDb() -> sqlite3.Connection:
    assert _db is not None, "DB not initialized"
    return _db


def _getEmbedder() -> EmbeddingProvider:
    assert _embedder is not None, "Embedder not initialized"
    return _embedder


def _getConfig() -> RemembleConfig:
    assert _config is not None, "Config not initialized"
    return _config


# ============================================================
# Core Memory Tools
# ============================================================


@mcp.tool
async def remember(
    content: str,
    source: str | None = None,
    tags: str | None = None,
    metadata: str | None = None,
) -> dict:
    """Store a memory. Auto-chunks long text, embeds, and indexes.

    Args:
        content: Text content to remember.
        source: Where it came from (e.g., "conversation", "file", "manual").
        tags: Comma-separated tags for filtering.
        metadata: Optional JSON string with extra data.
    """
    db = _getDb()
    embedder = _getEmbedder()
    config = _getConfig()

    chunks = chunkText(content, config.chunking.target_tokens, config.chunking.overlap_tokens)
    ids: list[int] = []

    for i, chunk in enumerate(chunks):
        embedding = await embedder.embedOne(chunk)
        chunk_meta = metadata
        if len(chunks) > 1:
            # Add parent tracking for chunked content
            meta_dict = json.loads(metadata) if metadata else {}
            meta_dict["chunk_index"] = i
            meta_dict["chunk_count"] = len(chunks)
            if ids:
                meta_dict["parent_id"] = ids[0]
            chunk_meta = json.dumps(meta_dict)

        memory_id = insertMemory(
            db,
            content=chunk,
            embedding=embedding,
            source=source,
            tags=tags,
            metadata_json=chunk_meta,
        )
        ids.append(memory_id)

    return {
        "stored": True,
        "memory_ids": ids,
        "chunks": len(chunks),
        "tokens": countTokens(content),
    }


@mcp.tool
async def recall(
    query: str,
    limit: int = 10,
    use_rag: bool = True,
) -> dict:
    """Search memories by semantic similarity. Returns ranked results with context.

    Args:
        query: Natural language search query.
        limit: Max results to return.
        use_rag: If True, returns token-budgeted RAG context. If False, raw search results.
    """
    db = _getDb()
    embedder = _getEmbedder()
    config = _getConfig()

    query_embedding = await embedder.embedOne(query)

    if use_rag:
        context = buildContext(db, query, query_embedding, config.search, config.rag)
        return {
            "query": context.query,
            "total_tokens": context.total_tokens,
            "items": [item.model_dump() for item in context.items],
            "entities": [
                {
                    "name": e.entity.name,
                    "type": e.entity.entity_type,
                    "observations": [o.content for o in e.observations],
                }
                for e in context.entities
            ],
        }

    # Raw search mode
    result = hybridSearch(db, query, query_embedding, config.search, limit=limit)
    return {
        "query": query,
        "results": [r.model_dump() for r in result.results[:limit]],
        "graph": [
            {
                "name": g.entity.name,
                "type": g.entity.entity_type,
                "observations": [o.content for o in g.observations],
            }
            for g in result.graph
        ],
    }


@mcp.tool
async def forget(memory_id: int) -> dict:
    """Soft-delete a memory by ID.

    Args:
        memory_id: The ID of the memory to forget.
    """
    db = _getDb()
    deleted = softDeleteMemory(db, memory_id)
    return {"forgotten": deleted, "memory_id": memory_id}


@mcp.tool
async def list_memories(
    source: str | None = None,
    tags: str | None = None,
    status: str = "active",
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Browse memories with optional filters.

    Args:
        source: Filter by source (e.g., "conversation").
        tags: Filter by tag substring.
        status: Filter by status: "active", "deleted", or "archived".
        limit: Max results per page.
        offset: Pagination offset.
    """
    db = _getDb()
    rows = listMemories(db, source=source, tags=tags, status=status, limit=limit, offset=offset)
    return {
        "memories": [
            {
                "id": r["id"],
                "content": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                "source": r["source"],
                "tags": r["tags"],
                "created_at": r["created_at"],
                "access_count": r["access_count"],
                "status": r["status"],
            }
            for r in rows
        ],
        "count": len(rows),
        "offset": offset,
    }


@mcp.tool
async def memory_stats() -> dict:
    """Get database statistics: memory counts, index sizes, provider info."""
    db = _getDb()
    embedder = _getEmbedder()
    config = _getConfig()

    stats = memoryStats(db)
    db_path = os.path.expanduser(config.db_path)
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0

    return {
        **stats,
        "db_size_bytes": db_size,
        "db_size_mb": round(db_size / (1024 * 1024), 2),
        "embedding_provider": embedder.name,
        "embedding_dimensions": embedder.dimensions,
    }


# ============================================================
# Knowledge Graph Tools
# ============================================================


@mcp.tool
async def create_entities(
    entities: list[dict],
) -> dict:
    """Create entities in the knowledge graph.

    Args:
        entities: List of dicts with keys: name, entity_type, observations (optional).
    """
    db = _getDb()
    created: list[dict] = []

    for e in entities:
        entity_id = upsertEntity(db, e["name"], e["entity_type"])
        obs_ids = []
        for obs in e.get("observations", []):
            obs_id = addObservation(db, entity_id, obs)
            obs_ids.append(obs_id)
        created.append({"entity_id": entity_id, "name": e["name"], "observations": len(obs_ids)})

    return {"created": created}


@mcp.tool
async def create_relations(
    relations: list[dict],
) -> dict:
    """Create relations between entities in the knowledge graph.

    Args:
        relations: List of dicts with keys: from_name (str), to_name (str), relation_type (str).
    """
    db = _getDb()
    created: list[dict] = []

    for r in relations:
        from_id = upsertEntity(db, r["from_name"], "unknown")
        to_id = upsertEntity(db, r["to_name"], "unknown")
        rel_id = addRelation(db, from_id, to_id, r["relation_type"], r.get("metadata"))
        created.append(
            {
                "relation_id": rel_id,
                "from": r["from_name"],
                "to": r["to_name"],
                "type": r["relation_type"],
            }
        )

    return {"created": created}


@mcp.tool
async def add_observations(
    entity_name: str,
    observations: list[str],
    source: str | None = None,
) -> dict:
    """Add observations (facts) to an existing entity.

    Args:
        entity_name: Name of the entity to add observations to.
        observations: List of observation strings.
        source: Optional source of the observations.
    """
    db = _getDb()
    row = db.execute("SELECT id FROM entities WHERE name = ?", (entity_name,)).fetchone()
    if not row:
        return {"error": f"Entity '{entity_name}' not found"}

    entity_id = row["id"]
    added = []
    for obs in observations:
        obs_id = addObservation(db, entity_id, obs, source=source)
        added.append(obs_id)

    return {"entity": entity_name, "observations_added": len(added)}


@mcp.tool
async def search_graph(query: str, limit: int = 10) -> dict:
    """Search the knowledge graph by entity name or observation content.

    Args:
        query: Search text to match against entity names and observations.
        limit: Max entities to return.
    """
    db = _getDb()
    results = graphSearch(db, query, limit=limit)
    return {
        "entities": [
            {
                "name": r.entity.name,
                "type": r.entity.entity_type,
                "observations": [o.content for o in r.observations],
                "relations": [
                    {
                        "type": rwe.relation.relation_type,
                        "entity": rwe.entity.name,
                        "direction": rwe.direction,
                    }
                    for rwe in r.relations
                ],
            }
            for r in results
        ],
    }


@mcp.tool
async def delete_entities(names: list[str]) -> dict:
    """Delete entities from the knowledge graph (cascades to relations and observations).

    Args:
        names: List of entity names to delete.
    """
    db = _getDb()
    deleted = []
    for name in names:
        if deleteEntity(db, name):
            deleted.append(name)
    return {"deleted": deleted}


# ============================================================
# MCP Resources
# ============================================================


@mcp.resource("memory://stats")
def resource_stats() -> dict:
    """Current database statistics."""
    db = _getDb()
    return memoryStats(db)


@mcp.resource("memory://recent")
def resource_recent() -> list[dict]:
    """20 most recent memories."""
    db = _getDb()
    rows = listMemories(db, limit=20)
    return [
        {
            "id": r["id"],
            "content": r["content"][:200],
            "source": r["source"],
            "tags": r["tags"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@mcp.resource("memory://graph")
def resource_graph() -> dict:
    """Full entity/relation graph."""
    db = _getDb()
    entities = db.execute("SELECT * FROM entities ORDER BY name").fetchall()
    relations = db.execute(
        """SELECT r.*, e1.name AS from_name, e2.name AS to_name
           FROM relations r
           JOIN entities e1 ON e1.id = r.from_entity_id
           JOIN entities e2 ON e2.id = r.to_entity_id"""
    ).fetchall()
    return {
        "entities": [{"name": e["name"], "type": e["entity_type"]} for e in entities],
        "relations": [
            {"from": r["from_name"], "to": r["to_name"], "type": r["relation_type"]}
            for r in relations
        ],
    }


@mcp.resource("memory://{memory_id}")
def resource_memory(memory_id: str) -> dict:
    """Fetch a specific memory by ID."""
    db = _getDb()
    row = getMemory(db, int(memory_id))
    if not row:
        return {"error": "Memory not found"}
    return {
        "id": row["id"],
        "content": row["content"],
        "source": row["source"],
        "tags": row["tags"],
        "metadata": row["metadata_json"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "accessed_at": row["accessed_at"],
        "access_count": row["access_count"],
        "status": row["status"],
    }


# ============================================================
# Config CLI helpers
# ============================================================


def _fmtVal(v: Any) -> str:
    if v is None:
        return "[dim](not set)[/dim]"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _annStr(ann: Any) -> str:
    """Return a simple string representation of a type annotation."""
    args = get_args(ann)
    if args:
        non_none = [a for a in args if a is not type(None)]
        has_none = type(None) in args
        base = non_none[0] if non_none else args[0]
        name = getattr(base, "__name__", str(base))
        return f"{name} | None" if has_none else name
    return getattr(ann, "__name__", str(ann))


def _unwrapModel(ann: Any) -> type[BaseModel] | None:
    """Extract a BaseModel subclass from Optional[X] / Union[X, None]."""
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    for arg in get_args(ann):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _getFieldAnnotation(dotpath: str) -> Any:
    """Walk model fields for dotpath, return annotation or None."""
    parts = dotpath.split(".")
    model: type[BaseModel] = RemembleConfig
    for part in parts[:-1]:
        f = model.model_fields.get(part)
        if f is None:
            return None
        model = _unwrapModel(f.annotation)
        if model is None:
            return None
    f = model.model_fields.get(parts[-1])
    return f.annotation if f else None


def _coerceTyped(value: str, annotation: Any) -> Any:
    """Coerce string value using the field annotation."""
    origin = get_origin(annotation)
    args = get_args(annotation) if origin else ()
    types = [a for a in args if a is not type(None)] if args else [annotation]
    base = types[0] if types else str

    if value.lower() in ("none", "null") and type(None) in (args or []):
        return None
    if base is bool:
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        raise ValueError(f"Expected bool, got {value!r}")
    if base is int:
        return int(value)
    if base is float:
        return float(value)
    return value


def _renderConfigSection(title: str, pairs: list[tuple[str, Any, Any]]) -> None:
    """Print a section with title + key/value table. pairs = (key, value, default)."""
    _console.print(f"\n[bold]{title}[/bold]")
    t = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    t.add_column("key", style="dim")
    t.add_column("val")
    for key, val, default in pairs:
        fmt = _fmtVal(val)
        if val != default:
            fmt = f"[yellow]{fmt}[/yellow]"
        t.add_row(key, fmt)
    _console.print(t)


# ============================================================
# CLI (typer)
# ============================================================

_cli = typer.Typer(
    name="rememble",
    help="Local MCP memory server with hybrid search and knowledge graph.",
    no_args_is_help=False,
    rich_markup_mode="rich",
)
_config_cli = typer.Typer(help="Read/write [bold]~/.rememble/config.json[/bold].")
_cli.add_typer(_config_cli, name="config")

_console = Console()


@_cli.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Start the MCP server (default when no subcommand given)."""
    if ctx.invoked_subcommand is None:
        logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
        mcp.run(transport="stdio")


@_cli.command()
def setup(
    provider: str | None = typer.Option(
        None, "--provider", help="Provider slug: ollama|openai|openrouter|cohere"
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="API key (skips prompt)"),
    model: str | None = typer.Option(None, "--model", help="Model name (skips prompt)"),
    agents: str | None = typer.Option(
        None, "--agents", help="Comma-sep agent slugs, e.g. claude-code,cursor"
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Skip agent install prompts (install all selected)"
    ),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Interactive setup wizard — configure embedding and AI agents."""
    if format not in ("human", "json"):
        raise typer.BadParameter(f"Invalid format {format!r}; choose human or json")
    from rememble.setup import runSetup

    runSetup(provider=provider, api_key=api_key, model=model, agents=agents, yes=yes, format=format)


@_cli.command()
def uninstall(
    yes: bool = typer.Option(False, "--yes", help="Skip DB deletion prompt (assume yes)"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Remove rememble MCP entries and instructions from all agent configs."""
    if format not in ("human", "json"):
        raise typer.BadParameter(f"Invalid format {format!r}; choose human or json")
    from rememble.setup import runUninstall

    runUninstall(yes=yes, format=format)


@_config_cli.command("list")
def config_list(
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Pretty-print the current config grouped by section."""
    if format not in ("human", "json"):
        raise typer.BadParameter(f"Invalid format {format!r}; choose human or json")
    cfg = loadConfig()

    if format == "json":
        print(json.dumps(cfg.model_dump()))
        raise typer.Exit()

    defaults = RemembleConfig()
    d = cfg.model_dump()
    dd = defaults.model_dump()

    embedding_keys = [
        "db_path",
        "embedding_api_url",
        "embedding_api_key",
        "embedding_api_model",
        "embedding_dimensions",
    ]
    _renderConfigSection("Embedding", [(k, d[k], dd[k]) for k in embedding_keys])

    def _subPairs(cfg_sub: Any, def_sub: Any, model: Any) -> list[tuple[str, Any, Any]]:
        return [(k, getattr(cfg_sub, k), getattr(def_sub, k)) for k in model.model_fields]

    _renderConfigSection("Search", _subPairs(cfg.search, defaults.search, SearchConfig))
    _renderConfigSection("RAG", _subPairs(cfg.rag, defaults.rag, RAGConfig))
    _renderConfigSection("Chunking", _subPairs(cfg.chunking, defaults.chunking, ChunkingConfig))


@_config_cli.command("get")
def config_get(
    dotpath: str = typer.Argument(help="Dot-separated key, e.g. search.rrf_k"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Get a single config value."""
    if format not in ("human", "json"):
        raise typer.BadParameter(f"Invalid format {format!r}; choose human or json")
    node: Any = loadConfig().model_dump()
    for part in dotpath.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            if format == "json":
                print(json.dumps({"ok": False, "error": f"Key not found: {dotpath}"}))
            else:
                _console.print(f"[red]Key not found:[/red] {dotpath}")
            raise typer.Exit(1)
    ann = _getFieldAnnotation(dotpath)
    if format == "json":
        print(
            json.dumps({"key": dotpath, "value": node, "type": _annStr(ann) if ann else "unknown"})
        )
    else:
        type_hint = f"  [dim]({_annStr(ann)})[/dim]" if ann else ""
        _console.print(f"[bold]{dotpath}[/bold] = {_fmtVal(node)}{type_hint}")


@_config_cli.command("set")
def config_set(
    dotpath: str = typer.Argument(help="Dot-separated key path"),
    value: str = typer.Argument(help="Value (type-coerced via schema)"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Set a config value."""
    if format not in ("human", "json"):
        raise typer.BadParameter(f"Invalid format {format!r}; choose human or json")
    from rememble.config import CONFIG_PATH

    ann = _getFieldAnnotation(dotpath)
    try:
        coerced = _coerceTyped(value, ann) if ann else value
    except ValueError as e:
        if format == "json":
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            _console.print(f"[red]Invalid:[/red] {e}")
        raise typer.Exit(1) from e

    raw: dict = {}
    if CONFIG_PATH.exists():
        with contextlib.suppress(json.JSONDecodeError):
            raw = json.loads(CONFIG_PATH.read_text())

    parts = dotpath.split(".")
    node = raw
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = coerced

    try:
        RemembleConfig(**raw)
    except Exception as e:
        if format == "json":
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            _console.print(f"[red]Invalid value:[/red] {e}")
        raise typer.Exit(1) from e

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(raw, indent=2) + "\n")
    if format == "json":
        print(json.dumps({"ok": True, "key": dotpath, "value": coerced}))
    else:
        _console.print(f"[green]Set[/green] {dotpath} = {coerced!r}")


def main() -> None:
    _cli()


if __name__ == "__main__":
    main()

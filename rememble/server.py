"""Rememble MCP server — memory tools, knowledge graph, and RAG."""

from __future__ import annotations

import contextlib
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, get_args, get_origin

import typer
from fastmcp import FastMCP
from pydantic import BaseModel
from rich import box
from rich.console import Console
from rich.table import Table

from rememble.config import ChunkingConfig, RAGConfig, RemembleConfig, SearchConfig, loadConfig
from rememble.service import (
    svcAddObservations,
    svcCreateEntities,
    svcCreateRelations,
    svcDeleteEntities,
    svcForget,
    svcListMemories,
    svcMemoryStats,
    svcRecall,
    svcRemember,
    svcResourceGraph,
    svcResourceMemory,
    svcResourceRecent,
    svcResourceStats,
    svcSearchGraph,
)
from rememble.state import AppState, createAppState

logger = logging.getLogger("rememble")

# Single global state, set during lifespan
_state: AppState | None = None


def _getState() -> AppState:
    assert _state is not None, "AppState not initialized"
    return _state


@asynccontextmanager
async def lifespan(server):
    global _state
    _state = await createAppState()

    yield {"db": _state.db, "embedder": _state.embedder, "config": _state.config}

    if _state.db:
        _state.db.close()
    logger.info("Rememble shut down.")


mcp = FastMCP("rememble", lifespan=lifespan)


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
    return await svcRemember(_getState(), content, source, tags, metadata)


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
    return await svcRecall(_getState(), query, limit, use_rag)


@mcp.tool
async def forget(memory_id: int) -> dict:
    """Soft-delete a memory by ID.

    Args:
        memory_id: The ID of the memory to forget.
    """
    return await svcForget(_getState(), memory_id)


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
    return await svcListMemories(_getState(), source, tags, status, limit, offset)


@mcp.tool
async def memory_stats() -> dict:
    """Get database statistics: memory counts, index sizes, provider info."""
    return await svcMemoryStats(_getState())


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
    return await svcCreateEntities(_getState(), entities)


@mcp.tool
async def create_relations(
    relations: list[dict],
) -> dict:
    """Create relations between entities in the knowledge graph.

    Args:
        relations: List of dicts with keys: from_name (str), to_name (str), relation_type (str).
    """
    return await svcCreateRelations(_getState(), relations)


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
    return await svcAddObservations(_getState(), entity_name, observations, source)


@mcp.tool
async def search_graph(query: str, limit: int = 10) -> dict:
    """Search the knowledge graph by entity name or observation content.

    Args:
        query: Search text to match against entity names and observations.
        limit: Max entities to return.
    """
    return await svcSearchGraph(_getState(), query, limit)


@mcp.tool
async def delete_entities(names: list[str]) -> dict:
    """Delete entities from the knowledge graph (cascades to relations and observations).

    Args:
        names: List of entity names to delete.
    """
    return await svcDeleteEntities(_getState(), names)


# ============================================================
# MCP Resources
# ============================================================


@mcp.resource("memory://stats")
def resource_stats() -> dict:
    """Current database statistics."""
    return svcResourceStats(_getState())


@mcp.resource("memory://recent")
def resource_recent() -> list[dict]:
    """20 most recent memories."""
    return svcResourceRecent(_getState())


@mcp.resource("memory://graph")
def resource_graph() -> dict:
    """Full entity/relation graph."""
    return svcResourceGraph(_getState())


@mcp.resource("memory://{memory_id}")
def resource_memory(memory_id: str) -> dict:
    """Fetch a specific memory by ID."""
    return svcResourceMemory(_getState(), memory_id)


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
        unwrapped = _unwrapModel(f.annotation)
        if unwrapped is None:
            return None
        model = unwrapped
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
    no_args_is_help=True,
    rich_markup_mode="rich",
)
_config_cli = typer.Typer(help="Read/write [bold]~/.rememble/config.json[/bold].")
_cli.add_typer(_config_cli, name="config")
_entity_cli = typer.Typer(help="Knowledge graph entity operations.")
_cli.add_typer(_entity_cli, name="entity")
_graph_cli = typer.Typer(help="Knowledge graph search.")
_cli.add_typer(_graph_cli, name="graph")

_console = Console()


def _getClient():
    """Lazy-import and return a RemembleClient connected to running daemon."""
    from rememble.client import RemembleClient

    cfg = loadConfig()
    return RemembleClient(f"http://localhost:{cfg.port}")


@_cli.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Local MCP memory server with hybrid search and knowledge graph."""
    pass


# ── serve / stop / status ────────────────────────────────────


@_cli.command()
def serve(
    mcp_mode: bool = typer.Option(False, "--mcp", help="Run as MCP stdio server instead of HTTP"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run HTTP server in background"),
    port: int | None = typer.Option(None, "--port", "-p", help="HTTP port (default from config)"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Start the HTTP API server (or MCP stdio with --mcp)."""
    if mcp_mode:
        logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
        mcp.run(transport="stdio")
        return

    cfg = loadConfig()
    p = port or cfg.port

    if daemon:
        from rememble.daemon import daemonize, isRunning

        if isRunning(cfg.pid_path):
            if format == "json":
                print(json.dumps({"ok": False, "error": "daemon already running"}))
            else:
                _console.print("[red]Daemon already running[/red]")
            raise typer.Exit(1)

        daemonize(cfg.pid_path)

    if format == "json" and not daemon:
        print(json.dumps({"ok": True, "port": p, "daemon": daemon}))

    import uvicorn

    from rememble.api import app

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=p, log_level="info")


@_cli.command()
def stop(
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Stop the running daemon."""
    from rememble.daemon import stopDaemon

    cfg = loadConfig()
    stopped = stopDaemon(cfg.pid_path)
    if format == "json":
        print(json.dumps({"ok": stopped}))
    elif stopped:
        _console.print("[green]Daemon stopped[/green]")
    else:
        _console.print("[yellow]No daemon running[/yellow]")


@_cli.command()
def status(
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Check if the daemon is running."""
    from rememble.daemon import isRunning, readPid

    cfg = loadConfig()
    running = isRunning(cfg.pid_path)
    pid = readPid(cfg.pid_path)
    if format == "json":
        print(json.dumps({"running": running, "pid": pid, "port": cfg.port}))
    elif running:
        _console.print(f"[green]Running[/green] — pid {pid}, port {cfg.port}")
    else:
        _console.print("[dim]Not running[/dim]")


# ── client commands (talk to running daemon) ─────────────────


@_cli.command("remember")
def cli_remember(
    content: str = typer.Argument(help="Text content to remember"),
    source: str | None = typer.Option(None, "--source", "-s", help="Source label"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Store a memory via the HTTP API."""
    try:
        with _getClient() as c:
            result = c.remember(content, source=source, tags=tags)
    except Exception as e:
        if format == "json":
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            _console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if format == "json":
        print(json.dumps(result))
    else:
        ids = result.get("memory_ids", [])
        _console.print(f"[green]Stored[/green] {len(ids)} chunk(s) — ids: {ids}")


@_cli.command("recall")
def cli_recall(
    query: str = typer.Argument(help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Raw results instead of RAG context"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Search memories via the HTTP API."""
    try:
        with _getClient() as c:
            result = c.recall(query, limit=limit, use_rag=not no_rag)
    except Exception as e:
        if format == "json":
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            _console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if format == "json":
        print(json.dumps(result))
        return

    if no_rag:
        for r in result.get("results", []):
            _console.print(f"  [bold]#{r['memory_id']}[/bold] (score: {r['score']:.3f})")
            if r.get("snippet"):
                _console.print(f"    {r['snippet'][:120]}")
    else:
        for item in result.get("items", []):
            _console.print(f"  [{item['kind']}] {item['text'][:120]}")
        _console.print(f"[dim]tokens: {result.get('total_tokens', 0)}[/dim]")


@_cli.command("forget")
def cli_forget(
    memory_id: int = typer.Argument(help="Memory ID to forget"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Soft-delete a memory via the HTTP API."""
    try:
        with _getClient() as c:
            result = c.forget(memory_id)
    except Exception as e:
        if format == "json":
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            _console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if format == "json":
        print(json.dumps(result))
    elif result.get("forgotten"):
        _console.print(f"[green]Forgotten[/green] memory #{memory_id}")
    else:
        _console.print(f"[yellow]Not found[/yellow] memory #{memory_id}")


@_cli.command("list")
def cli_list(
    source: str | None = typer.Option(None, "--source", "-s", help="Filter by source"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Filter by tags"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """List memories via the HTTP API."""
    try:
        with _getClient() as c:
            result = c.listMemories(source=source, tags=tags, limit=limit)
    except Exception as e:
        if format == "json":
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            _console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if format == "json":
        print(json.dumps(result))
        return

    t = Table(box=box.SIMPLE)
    t.add_column("ID", style="bold")
    t.add_column("Content")
    t.add_column("Source", style="dim")
    t.add_column("Tags", style="dim")
    for m in result.get("memories", []):
        t.add_row(str(m["id"]), m["content"][:80], m.get("source") or "", m.get("tags") or "")
    _console.print(t)
    _console.print(f"[dim]{result.get('count', 0)} memories[/dim]")


@_cli.command("stats")
def cli_stats(
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Show database stats via the HTTP API."""
    try:
        with _getClient() as c:
            result = c.stats()
    except Exception as e:
        if format == "json":
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            _console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if format == "json":
        print(json.dumps(result))
        return

    t = Table(show_header=False, box=box.SIMPLE)
    t.add_column("key", style="dim")
    t.add_column("val")
    for k, v in result.items():
        t.add_row(k, str(v))
    _console.print(t)


# ── entity + graph subcommands ───────────────────────────────


@_entity_cli.command("create")
def cli_entity_create(
    name: str = typer.Option(..., "--name", help="Entity name"),
    entity_type: str = typer.Option(..., "--type", help="Entity type"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Create an entity via the HTTP API."""
    try:
        with _getClient() as c:
            result = c.createEntities([{"name": name, "entity_type": entity_type}])
    except Exception as e:
        if format == "json":
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            _console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if format == "json":
        print(json.dumps(result))
    else:
        _console.print(f"[green]Created[/green] entity '{name}' (type: {entity_type})")


@_graph_cli.command("search")
def cli_graph_search(
    query: str = typer.Argument(help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human|json"),
) -> None:
    """Search the knowledge graph via the HTTP API."""
    try:
        with _getClient() as c:
            result = c.searchGraph(query, limit=limit)
    except Exception as e:
        if format == "json":
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            _console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if format == "json":
        print(json.dumps(result))
        return

    for ent in result.get("entities", []):
        _console.print(f"  [bold]{ent['name']}[/bold] ({ent['type']})")
        for obs in ent.get("observations", []):
            _console.print(f"    - {obs}")
        for rel in ent.get("relations", []):
            _console.print(f"    → {rel['type']} → {rel['entity']}")


# ── setup / uninstall / config (unchanged) ───────────────────


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

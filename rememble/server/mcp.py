"""Rememble MCP server — memory tools, knowledge graph, and RAG."""

from __future__ import annotations

from fastmcp import FastMCP

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
from rememble.state import getState

mcp = FastMCP("rememble")


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
    return await svcRemember(getState(), content, source, tags, metadata)


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
    return await svcRecall(getState(), query, limit, use_rag)


@mcp.tool
async def forget(memory_id: int) -> dict:
    """Soft-delete a memory by ID.

    Args:
        memory_id: The ID of the memory to forget.
    """
    return await svcForget(getState(), memory_id)


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
    return await svcListMemories(getState(), source, tags, status, limit, offset)


@mcp.tool
async def memory_stats() -> dict:
    """Get database statistics: memory counts, index sizes, provider info."""
    return await svcMemoryStats(getState())


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
    return await svcCreateEntities(getState(), entities)


@mcp.tool
async def create_relations(
    relations: list[dict],
) -> dict:
    """Create relations between entities in the knowledge graph.

    Args:
        relations: List of dicts with keys: from_name (str), to_name (str), relation_type (str).
    """
    return await svcCreateRelations(getState(), relations)


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
    return await svcAddObservations(getState(), entity_name, observations, source)


@mcp.tool
async def search_graph(query: str, limit: int = 10) -> dict:
    """Search the knowledge graph by entity name or observation content.

    Args:
        query: Search text to match against entity names and observations.
        limit: Max entities to return.
    """
    return await svcSearchGraph(getState(), query, limit)


@mcp.tool
async def delete_entities(names: list[str]) -> dict:
    """Delete entities from the knowledge graph (cascades to relations and observations).

    Args:
        names: List of entity names to delete.
    """
    return await svcDeleteEntities(getState(), names)


# ============================================================
# MCP Resources
# ============================================================


@mcp.resource("memory://stats")
def resource_stats() -> dict:
    """Current database statistics."""
    return svcResourceStats(getState())


@mcp.resource("memory://recent")
def resource_recent() -> list[dict]:
    """20 most recent memories."""
    return svcResourceRecent(getState())


@mcp.resource("memory://graph")
def resource_graph() -> dict:
    """Full entity/relation graph."""
    return svcResourceGraph(getState())


@mcp.resource("memory://{memory_id}")
def resource_memory(memory_id: str) -> dict:
    """Fetch a specific memory by ID."""
    return svcResourceMemory(getState(), memory_id)

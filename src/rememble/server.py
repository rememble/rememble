"""Rememble MCP server — memory tools, knowledge graph, and RAG."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from rememble.config import RemembleConfig, loadConfig
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
    _embedder = await createProvider(_config.embedding)
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
# Entry point
# ============================================================


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

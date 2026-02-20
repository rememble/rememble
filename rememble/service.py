"""Service layer — all business logic extracted from MCP tools."""

from __future__ import annotations

import json
import os

from rememble.db import (
    addObservation,
    addRelation,
    deleteEntity,
    getMemory,
    insertMemory,
    listMemories,
    memoryStats,
    softDeleteMemory,
    upsertEntity,
)
from rememble.ingest.chunker import chunkText, countTokens
from rememble.rag.context import buildContext
from rememble.search.fusion import bm25Probe, hybridSearch, hybridSearchTextOnly
from rememble.search.graph import graphSearch
from rememble.state import AppState

# ── Core Memory ──────────────────────────────────────────────


async def svcRemember(
    state: AppState,
    content: str,
    source: str | None = None,
    tags: str | None = None,
    metadata: str | None = None,
) -> dict:
    """Store a memory. Auto-chunks, embeds, indexes."""
    cfg = state.config
    c = cfg.chunking
    chunks = chunkText(content, c.target_tokens, c.overlap_tokens, c.markdown_aware)
    ids: list[int] = []

    for i, chunk in enumerate(chunks):
        embedding = await state.embedder.embedOne(chunk)
        chunk_meta = metadata
        if len(chunks) > 1:
            meta_dict = json.loads(metadata) if metadata else {}
            meta_dict["chunk_index"] = i
            meta_dict["chunk_count"] = len(chunks)
            if ids:
                meta_dict["parent_id"] = ids[0]
            chunk_meta = json.dumps(meta_dict)

        memory_id = insertMemory(
            state.db,
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


async def svcRecall(
    state: AppState,
    query: str,
    limit: int = 10,
    use_rag: bool = True,
) -> dict:
    """Search memories by semantic similarity. Uses BM25 probe to short-circuit embedding."""
    cfg = state.config
    threshold = cfg.search.bm25_shortcircuit_threshold

    # Phase 1: BM25 probe
    top_score, text_results = bm25Probe(state.db, query, cfg.search, limit)

    if top_score >= threshold and threshold < 1.0:
        # Short-circuit: skip embedding, fuse BM25 + temporal + graph only
        hybrid = hybridSearchTextOnly(state.db, query, text_results, cfg.search, limit)
        if use_rag:
            context = buildContext(state.db, query, [], cfg.search, cfg.rag, precomputed=hybrid)
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
        return {
            "query": query,
            "results": [r.model_dump() for r in hybrid.results[:limit]],
            "graph": [
                {
                    "name": g.entity.name,
                    "type": g.entity.entity_type,
                    "observations": [o.content for o in g.observations],
                }
                for g in hybrid.graph
            ],
        }

    # Phase 2: full path (embed + vector + text + temporal + graph)
    query_embedding = await state.embedder.embedOne(query)

    if use_rag:
        context = buildContext(state.db, query, query_embedding, cfg.search, cfg.rag)
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

    result = hybridSearch(state.db, query, query_embedding, cfg.search, limit=limit)
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


async def svcForget(state: AppState, memory_id: int) -> dict:
    """Soft-delete a memory by ID."""
    deleted = softDeleteMemory(state.db, memory_id)
    return {"forgotten": deleted, "memory_id": memory_id}


async def svcListMemories(
    state: AppState,
    source: str | None = None,
    tags: str | None = None,
    status: str = "active",
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Browse memories with optional filters."""
    rows = listMemories(
        state.db, source=source, tags=tags, status=status, limit=limit, offset=offset
    )
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


async def svcMemoryStats(state: AppState) -> dict:
    """Get database statistics."""
    stats = memoryStats(state.db)
    db_path = os.path.expanduser(state.config.db_path)
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0

    return {
        **stats,
        "db_size_bytes": db_size,
        "db_size_mb": round(db_size / (1024 * 1024), 2),
        "embedding_provider": state.embedder.name,
        "embedding_dimensions": state.embedder.dimensions,
    }


# ── Knowledge Graph ──────────────────────────────────────────


async def svcCreateEntities(state: AppState, entities: list[dict]) -> dict:
    """Create entities in the knowledge graph."""
    created: list[dict] = []
    for e in entities:
        entity_id = upsertEntity(state.db, e["name"], e["entity_type"])
        obs_ids = []
        for obs in e.get("observations", []):
            obs_id = addObservation(state.db, entity_id, obs)
            obs_ids.append(obs_id)
        created.append({"entity_id": entity_id, "name": e["name"], "observations": len(obs_ids)})
    return {"created": created}


async def svcCreateRelations(state: AppState, relations: list[dict]) -> dict:
    """Create relations between entities."""
    created: list[dict] = []
    for r in relations:
        from_id = upsertEntity(state.db, r["from_name"], "unknown")
        to_id = upsertEntity(state.db, r["to_name"], "unknown")
        rel_id = addRelation(state.db, from_id, to_id, r["relation_type"], r.get("metadata"))
        created.append(
            {
                "relation_id": rel_id,
                "from": r["from_name"],
                "to": r["to_name"],
                "type": r["relation_type"],
            }
        )
    return {"created": created}


async def svcAddObservations(
    state: AppState,
    entity_name: str,
    observations: list[str],
    source: str | None = None,
) -> dict:
    """Add observations to an existing entity."""
    row = state.db.execute("SELECT id FROM entities WHERE name = ?", (entity_name,)).fetchone()
    if not row:
        return {"error": f"Entity '{entity_name}' not found"}

    entity_id = row["id"]
    added = []
    for obs in observations:
        obs_id = addObservation(state.db, entity_id, obs, source=source)
        added.append(obs_id)
    return {"entity": entity_name, "observations_added": len(added)}


async def svcSearchGraph(state: AppState, query: str, limit: int = 10) -> dict:
    """Search knowledge graph by entity name or observation content."""
    results = graphSearch(state.db, query, limit=limit)
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


async def svcDeleteEntities(state: AppState, names: list[str]) -> dict:
    """Delete entities (cascades to relations and observations)."""
    deleted = []
    for name in names:
        if deleteEntity(state.db, name):
            deleted.append(name)
    return {"deleted": deleted}


# ── Resources (sync) ─────────────────────────────────────────


def svcResourceStats(state: AppState) -> dict:
    """Current database statistics."""
    return memoryStats(state.db)


def svcResourceRecent(state: AppState) -> list[dict]:
    """20 most recent memories."""
    rows = listMemories(state.db, limit=20)
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


def svcResourceGraph(state: AppState) -> dict:
    """Full entity/relation graph."""
    entities = state.db.execute("SELECT * FROM entities ORDER BY name").fetchall()
    relations = state.db.execute(
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


def svcResourceMemory(state: AppState, memory_id: str) -> dict:
    """Fetch a specific memory by ID."""
    row = getMemory(state.db, int(memory_id))
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

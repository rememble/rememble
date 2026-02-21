"""REST API routes — mounted under /api by the combined app."""

from __future__ import annotations

from fastapi import APIRouter, Query

from rememble.server.api_models import (
    AddObservationsRequest,
    CreateEntitiesRequest,
    CreateRelationsRequest,
    DeleteEntitiesRequest,
    ForgetRequest,
    RecallRequest,
    RememberRequest,
)
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
    svcSearchGraph,
)
from rememble.state import getState

router = APIRouter()


# ── Health ───────────────────────────────────────────────────


@router.get("/health")
async def health():
    return {"status": "ok"}


# ── Core Memory ──────────────────────────────────────────────


@router.post("/remember")
async def api_remember(req: RememberRequest):
    return await svcRemember(
        getState(), req.content, req.source, req.tags, req.metadata, project=req.project
    )


@router.post("/recall")
async def api_recall(req: RecallRequest):
    return await svcRecall(getState(), req.query, req.limit, req.use_rag, project=req.project)


@router.post("/forget")
async def api_forget(req: ForgetRequest):
    return await svcForget(getState(), req.memory_id)


@router.get("/memories")
async def api_list_memories(
    source: str | None = Query(None),
    tags: str | None = Query(None),
    status: str = Query("active"),
    limit: int = Query(20),
    offset: int = Query(0),
    project: str | None = Query(None),
):
    return await svcListMemories(
        getState(), source, tags, status, limit, offset, project=project
    )


@router.get("/stats")
async def api_stats():
    return await svcMemoryStats(getState())


# ── Knowledge Graph ──────────────────────────────────────────


@router.post("/entities")
async def api_create_entities(req: CreateEntitiesRequest):
    entities = [e.model_dump() for e in req.entities]
    return await svcCreateEntities(getState(), entities, project=req.project)


@router.post("/relations")
async def api_create_relations(req: CreateRelationsRequest):
    relations = [r.model_dump() for r in req.relations]
    return await svcCreateRelations(getState(), relations)


@router.post("/observations")
async def api_add_observations(req: AddObservationsRequest):
    return await svcAddObservations(getState(), req.entity_name, req.observations, req.source)


@router.get("/graph")
async def api_search_graph(
    query: str = Query(...),
    limit: int = Query(10),
    project: str | None = Query(None),
):
    return await svcSearchGraph(getState(), query, limit, project=project)


@router.delete("/entities")
async def api_delete_entities(req: DeleteEntitiesRequest):
    return await svcDeleteEntities(getState(), req.names)

"""FastAPI HTTP API — routes calling the shared service layer."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

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
from rememble.state import AppState, createAppState

_state: AppState | None = None


def _getState() -> AppState:
    assert _state is not None, "AppState not initialized"
    return _state


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    _state = await createAppState(check_same_thread=False)
    yield
    if _state and _state.db:
        _state.db.close()


app = FastAPI(title="Rememble", lifespan=lifespan)


# ── Health ───────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Core Memory ──────────────────────────────────────────────


@app.post("/remember")
async def api_remember(req: RememberRequest):
    return await svcRemember(_getState(), req.content, req.source, req.tags, req.metadata)


@app.post("/recall")
async def api_recall(req: RecallRequest):
    return await svcRecall(_getState(), req.query, req.limit, req.use_rag)


@app.post("/forget")
async def api_forget(req: ForgetRequest):
    return await svcForget(_getState(), req.memory_id)


@app.get("/memories")
async def api_list_memories(
    source: str | None = Query(None),
    tags: str | None = Query(None),
    status: str = Query("active"),
    limit: int = Query(20),
    offset: int = Query(0),
):
    return await svcListMemories(_getState(), source, tags, status, limit, offset)


@app.get("/stats")
async def api_stats():
    return await svcMemoryStats(_getState())


# ── Knowledge Graph ──────────────────────────────────────────


@app.post("/entities")
async def api_create_entities(req: CreateEntitiesRequest):
    entities = [e.model_dump() for e in req.entities]
    return await svcCreateEntities(_getState(), entities)


@app.post("/relations")
async def api_create_relations(req: CreateRelationsRequest):
    relations = [r.model_dump() for r in req.relations]
    return await svcCreateRelations(_getState(), relations)


@app.post("/observations")
async def api_add_observations(req: AddObservationsRequest):
    return await svcAddObservations(_getState(), req.entity_name, req.observations, req.source)


@app.get("/graph")
async def api_search_graph(
    query: str = Query(...),
    limit: int = Query(10),
):
    return await svcSearchGraph(_getState(), query, limit)


@app.delete("/entities")
async def api_delete_entities(req: DeleteEntitiesRequest):
    return await svcDeleteEntities(_getState(), req.names)

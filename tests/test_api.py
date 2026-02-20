"""REST API integration tests."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rememble.config import RemembleConfig
from rememble.db import connect, insertMemory, upsertEntity
from rememble.server.api import router
from rememble.state import AppState, setState
from tests.conftest import FakeEmbedder


@pytest.fixture
def api_db(config: RemembleConfig) -> sqlite3.Connection:
    """DB with check_same_thread=False for TestClient cross-thread access."""
    conn, _ = connect(config, check_same_thread=False)
    yield conn
    conn.close()


@pytest.fixture
def api_state(config: RemembleConfig, api_db: sqlite3.Connection, fake_embedder: FakeEmbedder):
    """Set shared state for testing."""
    state = AppState(db=api_db, embedder=fake_embedder, config=config)
    setState(state)
    yield state
    setState(None)  # type: ignore[arg-type]


@pytest.fixture
def client(api_state):
    """FastAPI TestClient with patched state (skip lifespan to avoid Ollama connect)."""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app, raise_server_exceptions=True)


# ── Health ───────────────────────────────────────────────────


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── Core Memory ──────────────────────────────────────────────


def test_remember(client):
    r = client.post("/api/remember", json={"content": "hello world", "source": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["stored"] is True
    assert len(data["memory_ids"]) == 1


def test_recall_rag(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("search test")
    insertMemory(api_state.db, "search test content", emb, source="test")

    r = client.post("/api/recall", json={"query": "search test", "use_rag": True})
    assert r.status_code == 200
    data = r.json()
    assert "query" in data
    assert "total_tokens" in data


def test_recall_raw(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("raw")
    insertMemory(api_state.db, "raw search content", emb, source="test")

    r = client.post("/api/recall", json={"query": "raw", "use_rag": False})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data


def test_forget(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("f")
    mid = insertMemory(api_state.db, "to forget", emb)

    r = client.post("/api/forget", json={"memory_id": mid})
    assert r.status_code == 200
    assert r.json()["forgotten"] is True


def test_list_memories(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("l")
    insertMemory(api_state.db, "listed", emb, source="api")

    r = client.get("/api/memories")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


def test_list_memories_with_filter(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("l")
    insertMemory(api_state.db, "alpha", emb, source="file")
    insertMemory(api_state.db, "beta", emb, source="chat")

    r = client.get("/api/memories", params={"source": "file"})
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_stats(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("s")
    insertMemory(api_state.db, "stat", emb)

    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_memories"] >= 1
    assert "embedding_provider" in data


# ── Knowledge Graph ──────────────────────────────────────────


def test_create_entities(client):
    r = client.post(
        "/api/entities",
        json={
            "entities": [{"name": "Python", "entity_type": "language", "observations": ["Fast"]}],
        },
    )
    assert r.status_code == 200
    assert len(r.json()["created"]) == 1


def test_create_relations(client, api_state):
    upsertEntity(api_state.db, "A", "test")
    upsertEntity(api_state.db, "B", "test")

    r = client.post(
        "/api/relations",
        json={"relations": [{"from_name": "A", "to_name": "B", "relation_type": "linked"}]},
    )
    assert r.status_code == 200
    assert len(r.json()["created"]) == 1


def test_add_observations(client, api_state):
    upsertEntity(api_state.db, "Python", "language")

    r = client.post(
        "/api/observations",
        json={"entity_name": "Python", "observations": ["3.12 released"]},
    )
    assert r.status_code == 200
    assert r.json()["observations_added"] == 1


def test_search_graph(client, api_state):
    from rememble.db import addObservation

    eid = upsertEntity(api_state.db, "SQLite", "database")
    addObservation(api_state.db, eid, "Embedded")

    r = client.get("/api/graph", params={"query": "SQLite"})
    assert r.status_code == 200
    assert len(r.json()["entities"]) == 1


def test_delete_entities(client, api_state):
    upsertEntity(api_state.db, "ToDelete", "test")

    r = client.request("DELETE", "/api/entities", json={"names": ["ToDelete"]})
    assert r.status_code == 200
    assert "ToDelete" in r.json()["deleted"]

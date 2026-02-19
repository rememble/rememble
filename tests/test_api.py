"""FastAPI HTTP API integration tests."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

import rememble.api as api_mod
from rememble.config import RemembleConfig
from rememble.db import connect, insertMemory, upsertEntity
from rememble.state import AppState
from tests.conftest import FakeEmbedder


@pytest.fixture
def api_db(config: RemembleConfig) -> sqlite3.Connection:
    """DB with check_same_thread=False for TestClient cross-thread access."""
    conn = connect(config, check_same_thread=False)
    yield conn
    conn.close()


@pytest.fixture
def api_state(config: RemembleConfig, api_db: sqlite3.Connection, fake_embedder: FakeEmbedder):
    """Patch api module state for testing."""
    api_mod._state = AppState(db=api_db, embedder=fake_embedder, config=config)
    yield api_mod._state
    api_mod._state = None


@pytest.fixture
def client(api_state):
    """FastAPI TestClient with patched state (skip lifespan to avoid Ollama connect)."""
    return TestClient(api_mod.app, raise_server_exceptions=True)


# ── Health ───────────────────────────────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── Core Memory ──────────────────────────────────────────────


def test_remember(client):
    r = client.post("/remember", json={"content": "hello world", "source": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["stored"] is True
    assert len(data["memory_ids"]) == 1


def test_recall_rag(client, api_state):
    # Seed a memory first
    from tests.conftest import FakeEmbedder

    emb = FakeEmbedder(4)._fakeVec("search test")
    insertMemory(api_state.db, "search test content", emb, source="test")

    r = client.post("/recall", json={"query": "search test", "use_rag": True})
    assert r.status_code == 200
    data = r.json()
    assert "query" in data
    assert "total_tokens" in data


def test_recall_raw(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("raw")
    insertMemory(api_state.db, "raw search content", emb, source="test")

    r = client.post("/recall", json={"query": "raw", "use_rag": False})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data


def test_forget(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("f")
    mid = insertMemory(api_state.db, "to forget", emb)

    r = client.post("/forget", json={"memory_id": mid})
    assert r.status_code == 200
    assert r.json()["forgotten"] is True


def test_list_memories(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("l")
    insertMemory(api_state.db, "listed", emb, source="api")

    r = client.get("/memories")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


def test_list_memories_with_filter(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("l")
    insertMemory(api_state.db, "alpha", emb, source="file")
    insertMemory(api_state.db, "beta", emb, source="chat")

    r = client.get("/memories", params={"source": "file"})
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_stats(client, api_state):
    emb = FakeEmbedder(4)._fakeVec("s")
    insertMemory(api_state.db, "stat", emb)

    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_memories"] >= 1
    assert "embedding_provider" in data


# ── Knowledge Graph ──────────────────────────────────────────


def test_create_entities(client):
    r = client.post(
        "/entities",
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
        "/relations",
        json={"relations": [{"from_name": "A", "to_name": "B", "relation_type": "linked"}]},
    )
    assert r.status_code == 200
    assert len(r.json()["created"]) == 1


def test_add_observations(client, api_state):
    upsertEntity(api_state.db, "Python", "language")

    r = client.post(
        "/observations",
        json={"entity_name": "Python", "observations": ["3.12 released"]},
    )
    assert r.status_code == 200
    assert r.json()["observations_added"] == 1


def test_search_graph(client, api_state):
    from rememble.db import addObservation

    eid = upsertEntity(api_state.db, "SQLite", "database")
    addObservation(api_state.db, eid, "Embedded")

    r = client.get("/graph", params={"query": "SQLite"})
    assert r.status_code == 200
    assert len(r.json()["entities"]) == 1


def test_delete_entities(client, api_state):
    upsertEntity(api_state.db, "ToDelete", "test")

    r = client.request("DELETE", "/entities", json={"names": ["ToDelete"]})
    assert r.status_code == 200
    assert "ToDelete" in r.json()["deleted"]

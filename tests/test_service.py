"""Tests for service layer — business logic decoupled from MCP transport."""

from __future__ import annotations

import json
import sqlite3

import pytest

from rememble.config import RemembleConfig
from rememble.db import (
    addObservation,
    addRelation,
    getMemory,
    insertMemory,
    upsertEntity,
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
    svcResourceGraph,
    svcResourceMemory,
    svcResourceRecent,
    svcResourceStats,
    svcSearchGraph,
)
from rememble.state import AppState
from tests.conftest import FakeEmbedder


@pytest.fixture
def state(config: RemembleConfig, db: sqlite3.Connection, fake_embedder: FakeEmbedder) -> AppState:
    return AppState(db=db, embedder=fake_embedder, config=config)


# ── Core Memory ──────────────────────────────────────────────


class TestSvcRemember:
    @pytest.mark.asyncio
    async def test_storeShortText(self, state, db):
        result = await svcRemember(state, "SQLite is fast", source="test", tags="db,sqlite")
        assert result["stored"] is True
        assert result["chunks"] == 1
        assert len(result["memory_ids"]) == 1

        row = getMemory(db, result["memory_ids"][0])
        assert row is not None
        assert row["content"] == "SQLite is fast"

    @pytest.mark.asyncio
    async def test_storeLongTextChunks(self, state, db):
        long_text = "word " * 1000
        result = await svcRemember(state, long_text, source="test")
        assert result["chunks"] > 1
        assert len(result["memory_ids"]) == result["chunks"]

        second = getMemory(db, result["memory_ids"][1])
        meta = json.loads(second["metadata_json"])
        assert meta["parent_id"] == result["memory_ids"][0]

    @pytest.mark.asyncio
    async def test_storeWithMetadata(self, state, db):
        result = await svcRemember(state, "test", metadata='{"key": "value"}')
        row = getMemory(db, result["memory_ids"][0])
        meta = json.loads(row["metadata_json"])
        assert meta["key"] == "value"


class TestSvcRecall:
    @pytest.mark.asyncio
    async def test_recallWithRag(self, state, db, fake_embedder):
        emb = await fake_embedder.embedOne("vector search")
        insertMemory(db, "vector search is great", emb, source="test")

        result = await svcRecall(state, "vector search", limit=5, use_rag=True)
        assert "query" in result
        assert "total_tokens" in result

    @pytest.mark.asyncio
    async def test_recallRawMode(self, state, db, fake_embedder):
        emb = await fake_embedder.embedOne("python code")
        insertMemory(db, "python is awesome", emb, source="test")

        result = await svcRecall(state, "python", limit=5, use_rag=False)
        assert "results" in result
        assert "graph" in result

    @pytest.mark.asyncio
    async def test_recallEmptyDb(self, state):
        result = await svcRecall(state, "nothing", limit=5, use_rag=True)
        assert result["total_tokens"] == 0


class TestSvcForget:
    @pytest.mark.asyncio
    async def test_forgetMemory(self, state, db, fake_embedder):
        emb = await fake_embedder.embedOne("to forget")
        mid = insertMemory(db, "to forget", emb)

        result = await svcForget(state, mid)
        assert result["forgotten"] is True
        assert getMemory(db, mid)["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_forgetNonexistent(self, state):
        result = await svcForget(state, 99999)
        assert result["forgotten"] is False


class TestSvcListMemories:
    @pytest.mark.asyncio
    async def test_listWithFilters(self, state, db, fake_embedder):
        emb = await fake_embedder.embedOne("a")
        insertMemory(db, "alpha", emb, source="file", tags="greek")
        insertMemory(db, "beta", emb, source="chat", tags="greek")
        insertMemory(db, "gamma", emb, source="file", tags="other")

        result = await svcListMemories(state, source="file")
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_listTruncatesContent(self, state, db, fake_embedder):
        emb = await fake_embedder.embedOne("long")
        insertMemory(db, "x" * 300, emb)

        result = await svcListMemories(state)
        assert result["memories"][0]["content"].endswith("...")


class TestSvcMemoryStats:
    @pytest.mark.asyncio
    async def test_stats(self, state, db, fake_embedder):
        emb = await fake_embedder.embedOne("s")
        insertMemory(db, "one", emb)
        insertMemory(db, "two", emb)

        result = await svcMemoryStats(state)
        assert result["total_memories"] == 2
        assert result["embedding_provider"] == "fake/test"


# ── Knowledge Graph ──────────────────────────────────────────


class TestSvcCreateEntities:
    @pytest.mark.asyncio
    async def test_createWithObservations(self, state):
        result = await svcCreateEntities(
            state,
            [{"name": "Python", "entity_type": "language", "observations": ["Typed", "GIL"]}],
        )
        assert len(result["created"]) == 1
        assert result["created"][0]["observations"] == 2

    @pytest.mark.asyncio
    async def test_createWithoutObservations(self, state):
        result = await svcCreateEntities(state, [{"name": "Rust", "entity_type": "language"}])
        assert result["created"][0]["observations"] == 0


class TestSvcCreateRelations:
    @pytest.mark.asyncio
    async def test_createRelation(self, state, db):
        upsertEntity(db, "Alice", "person")
        upsertEntity(db, "Acme", "company")
        result = await svcCreateRelations(
            state, [{"from_name": "Alice", "to_name": "Acme", "relation_type": "works_at"}]
        )
        assert result["created"][0]["type"] == "works_at"


class TestSvcAddObservations:
    @pytest.mark.asyncio
    async def test_addToExisting(self, state, db):
        upsertEntity(db, "Python", "language")
        result = await svcAddObservations(state, "Python", ["Version 3.12", "Async/await"])
        assert result["observations_added"] == 2

    @pytest.mark.asyncio
    async def test_addToNonexistent(self, state):
        result = await svcAddObservations(state, "Ghost", ["Not real"])
        assert "error" in result


class TestSvcSearchGraph:
    @pytest.mark.asyncio
    async def test_searchByName(self, state, db):
        eid = upsertEntity(db, "SQLite", "database")
        addObservation(db, eid, "Embedded database")
        result = await svcSearchGraph(state, "SQLite")
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "SQLite"


class TestSvcDeleteEntities:
    @pytest.mark.asyncio
    async def test_deleteCascades(self, state, db):
        eid = upsertEntity(db, "ToDelete", "test")
        addObservation(db, eid, "Will be gone")
        result = await svcDeleteEntities(state, ["ToDelete"])
        assert "ToDelete" in result["deleted"]


# ── Resources ────────────────────────────────────────────────


class TestSvcResources:
    def test_resourceStats(self, state, db, fake_embedder):
        emb = fake_embedder._fakeVec("r")
        insertMemory(db, "resource test", emb)
        result = svcResourceStats(state)
        assert result["total_memories"] == 1

    def test_resourceRecent(self, state, db, fake_embedder):
        emb = fake_embedder._fakeVec("r")
        insertMemory(db, "recent memory", emb)
        result = svcResourceRecent(state)
        assert len(result) == 1

    def test_resourceGraph(self, state, db):
        upsertEntity(db, "A", "test")
        upsertEntity(db, "B", "test")
        addRelation(db, 1, 2, "linked_to")
        result = svcResourceGraph(state)
        assert len(result["entities"]) == 2
        assert len(result["relations"]) == 1

    def test_resourceMemoryById(self, state, db, fake_embedder):
        emb = fake_embedder._fakeVec("r")
        mid = insertMemory(db, "specific memory", emb, source="test")
        result = svcResourceMemory(state, str(mid))
        assert result["content"] == "specific memory"

    def test_resourceMemoryNotFound(self, state):
        result = svcResourceMemory(state, "99999")
        assert "error" in result

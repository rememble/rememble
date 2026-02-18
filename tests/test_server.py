"""Tests for MCP server tools and resources."""

from __future__ import annotations

import json
import sqlite3

import pytest

# Import server module and patch globals for testing
import rememble.server as srv
from rememble.config import RemembleConfig
from rememble.db import (
    addObservation,
    addRelation,
    getMemory,
    insertMemory,
    upsertEntity,
)
from tests.conftest import FakeEmbedder

# FastMCP wraps decorated functions in FunctionTool/FunctionResource objects.
# Access the original callable via `.fn`.
_remember = srv.remember.fn
_recall = srv.recall.fn
_forget = srv.forget.fn
_list_memories = srv.list_memories.fn
_memory_stats = srv.memory_stats.fn
_create_entities = srv.create_entities.fn
_create_relations = srv.create_relations.fn
_add_observations = srv.add_observations.fn
_search_graph = srv.search_graph.fn
_delete_entities = srv.delete_entities.fn
_resource_stats = srv.resource_stats.fn
_resource_recent = srv.resource_recent.fn
_resource_graph = srv.resource_graph.fn
_resource_memory = srv.resource_memory.fn


@pytest.fixture
def server_ctx(config: RemembleConfig, db: sqlite3.Connection, fake_embedder: FakeEmbedder):
    """Set up server globals for tool testing."""
    srv._db = db
    srv._embedder = fake_embedder
    srv._config = config
    yield
    srv._db = None
    srv._embedder = None
    srv._config = None


# -- Core Memory Tools --


class TestRemember:
    @pytest.mark.asyncio
    async def test_storeShortText(self, server_ctx, db):
        result = await _remember("SQLite is fast", source="test", tags="db,sqlite")
        assert result["stored"] is True
        assert result["chunks"] == 1
        assert len(result["memory_ids"]) == 1

        row = getMemory(db, result["memory_ids"][0])
        assert row is not None
        assert row["content"] == "SQLite is fast"
        assert row["source"] == "test"
        assert row["tags"] == "db,sqlite"

    @pytest.mark.asyncio
    async def test_storeLongTextChunks(self, server_ctx, db):
        long_text = "word " * 1000
        result = await _remember(long_text, source="test")
        assert result["chunks"] > 1
        assert len(result["memory_ids"]) == result["chunks"]

        # Second chunk should have parent_id metadata
        second = getMemory(db, result["memory_ids"][1])
        assert second is not None
        meta = json.loads(second["metadata_json"])
        assert meta["parent_id"] == result["memory_ids"][0]
        assert meta["chunk_index"] == 1

    @pytest.mark.asyncio
    async def test_storeWithMetadata(self, server_ctx, db):
        result = await _remember("test", metadata='{"key": "value"}')
        row = getMemory(db, result["memory_ids"][0])
        meta = json.loads(row["metadata_json"])
        assert meta["key"] == "value"


class TestRecall:
    @pytest.mark.asyncio
    async def test_recallWithRag(self, server_ctx, db, fake_embedder):
        emb = await fake_embedder.embedOne("vector search")
        insertMemory(db, "vector search is great", emb, source="test")

        result = await _recall("vector search", limit=5, use_rag=True)
        assert "query" in result
        assert "total_tokens" in result
        assert "items" in result

    @pytest.mark.asyncio
    async def test_recallRawMode(self, server_ctx, db, fake_embedder):
        emb = await fake_embedder.embedOne("python code")
        insertMemory(db, "python is awesome", emb, source="test")

        result = await _recall("python", limit=5, use_rag=False)
        assert "results" in result
        assert "graph" in result

    @pytest.mark.asyncio
    async def test_recallEmptyDb(self, server_ctx):
        result = await _recall("nothing", limit=5, use_rag=True)
        assert result["total_tokens"] == 0
        assert result["items"] == []


class TestForget:
    @pytest.mark.asyncio
    async def test_forgetMemory(self, server_ctx, db, fake_embedder):
        emb = await fake_embedder.embedOne("to forget")
        mid = insertMemory(db, "to forget", emb)

        result = await _forget(mid)
        assert result["forgotten"] is True

        row = getMemory(db, mid)
        assert row["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_forgetNonexistent(self, server_ctx):
        result = await _forget(99999)
        assert result["forgotten"] is False


class TestListMemories:
    @pytest.mark.asyncio
    async def test_listWithFilters(self, server_ctx, db, fake_embedder):
        emb = await fake_embedder.embedOne("a")
        insertMemory(db, "alpha", emb, source="file", tags="greek")
        insertMemory(db, "beta", emb, source="chat", tags="greek")
        insertMemory(db, "gamma", emb, source="file", tags="other")

        result = await _list_memories(source="file")
        assert result["count"] == 2

        result = await _list_memories(tags="greek")
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_listTruncatesContent(self, server_ctx, db, fake_embedder):
        emb = await fake_embedder.embedOne("long")
        insertMemory(db, "x" * 300, emb)

        result = await _list_memories()
        assert result["memories"][0]["content"].endswith("...")


class TestMemoryStats:
    @pytest.mark.asyncio
    async def test_stats(self, server_ctx, db, fake_embedder):
        emb = await fake_embedder.embedOne("s")
        insertMemory(db, "one", emb)
        insertMemory(db, "two", emb)

        result = await _memory_stats()
        assert result["total_memories"] == 2
        assert result["embedding_provider"] == "fake/test"


# -- Knowledge Graph Tools --


class TestCreateEntities:
    @pytest.mark.asyncio
    async def test_createWithObservations(self, server_ctx):
        result = await _create_entities(
            [
                {
                    "name": "Python",
                    "entity_type": "language",
                    "observations": ["Dynamically typed", "GIL"],
                },
            ]
        )
        assert len(result["created"]) == 1
        assert result["created"][0]["name"] == "Python"
        assert result["created"][0]["observations"] == 2

    @pytest.mark.asyncio
    async def test_createWithoutObservations(self, server_ctx):
        result = await _create_entities(
            [
                {"name": "Rust", "entity_type": "language"},
            ]
        )
        assert result["created"][0]["observations"] == 0


class TestCreateRelations:
    @pytest.mark.asyncio
    async def test_createRelation(self, server_ctx, db):
        upsertEntity(db, "Alice", "person")
        upsertEntity(db, "Acme", "company")

        result = await _create_relations(
            [
                {"from_name": "Alice", "to_name": "Acme", "relation_type": "works_at"},
            ]
        )
        assert len(result["created"]) == 1
        assert result["created"][0]["type"] == "works_at"


class TestAddObservations:
    @pytest.mark.asyncio
    async def test_addToExisting(self, server_ctx, db):
        upsertEntity(db, "Python", "language")
        result = await _add_observations("Python", ["Version 3.12", "Async/await"])
        assert result["observations_added"] == 2

    @pytest.mark.asyncio
    async def test_addToNonexistent(self, server_ctx):
        result = await _add_observations("Ghost", ["Not real"])
        assert "error" in result


class TestSearchGraph:
    @pytest.mark.asyncio
    async def test_searchByName(self, server_ctx, db):
        eid = upsertEntity(db, "SQLite", "database")
        addObservation(db, eid, "Embedded database")

        result = await _search_graph("SQLite")
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "SQLite"
        assert "Embedded database" in result["entities"][0]["observations"]


class TestDeleteEntities:
    @pytest.mark.asyncio
    async def test_deleteCascades(self, server_ctx, db):
        eid = upsertEntity(db, "ToDelete", "test")
        addObservation(db, eid, "Will be gone")

        result = await _delete_entities(["ToDelete"])
        assert "ToDelete" in result["deleted"]

        obs = db.execute("SELECT * FROM observations WHERE entity_id = ?", (eid,)).fetchall()
        assert len(obs) == 0


# -- MCP Resources --


class TestResources:
    def test_resourceStats(self, server_ctx, db, fake_embedder):
        emb = fake_embedder._fakeVec("r")
        insertMemory(db, "resource test", emb)

        result = _resource_stats()
        assert result["total_memories"] == 1

    def test_resourceRecent(self, server_ctx, db, fake_embedder):
        emb = fake_embedder._fakeVec("r")
        insertMemory(db, "recent memory", emb)

        result = _resource_recent()
        assert len(result) == 1
        assert result[0]["content"] == "recent memory"

    def test_resourceGraph(self, server_ctx, db):
        upsertEntity(db, "A", "test")
        upsertEntity(db, "B", "test")
        addRelation(db, 1, 2, "linked_to")

        result = _resource_graph()
        assert len(result["entities"]) == 2
        assert len(result["relations"]) == 1

    def test_resourceMemoryById(self, server_ctx, db, fake_embedder):
        emb = fake_embedder._fakeVec("r")
        mid = insertMemory(db, "specific memory", emb, source="test")

        result = _resource_memory(str(mid))
        assert result["content"] == "specific memory"

    def test_resourceMemoryNotFound(self, server_ctx):
        result = _resource_memory("99999")
        assert "error" in result

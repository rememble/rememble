"""Tests for database operations."""

from __future__ import annotations

import sqlite3

from rememble.db import (
    addObservation,
    addRelation,
    deleteEntity,
    getMemory,
    insertMemory,
    listMemories,
    memoryStats,
    softDeleteMemory,
    updateAccessStats,
    upsertEntity,
)


def test_insertAndGetMemory(db: sqlite3.Connection):
    mid = insertMemory(db, "hello world", [0.1, 0.2, 0.3, 0.4], source="test")
    assert mid > 0
    row = getMemory(db, mid)
    assert row is not None
    assert row["content"] == "hello world"
    assert row["source"] == "test"
    assert row["status"] == "active"


def test_softDeleteMemory(db: sqlite3.Connection):
    mid = insertMemory(db, "to delete", [0.1, 0.2, 0.3, 0.4])
    assert softDeleteMemory(db, mid)
    row = getMemory(db, mid)
    assert row["status"] == "deleted"
    # Can't delete again
    assert not softDeleteMemory(db, mid)


def test_listMemoriesFilters(db: sqlite3.Connection):
    insertMemory(db, "alpha", [0.1, 0.2, 0.3, 0.4], source="chat", tags="project")
    insertMemory(db, "beta", [0.5, 0.6, 0.7, 0.8], source="file", tags="docs")
    insertMemory(db, "gamma", [0.9, 0.1, 0.2, 0.3], source="chat", tags="project,docs")

    all_rows = listMemories(db)
    assert len(all_rows) == 3

    chat_only = listMemories(db, source="chat")
    assert len(chat_only) == 2

    docs_tag = listMemories(db, tags="docs")
    assert len(docs_tag) == 2


def test_updateAccessStats(db: sqlite3.Connection):
    mid = insertMemory(db, "stats test", [0.1, 0.2, 0.3, 0.4])
    row = getMemory(db, mid)
    assert row["access_count"] == 0

    updateAccessStats(db, [mid])
    row = getMemory(db, mid)
    assert row["access_count"] == 1

    updateAccessStats(db, [mid])
    row = getMemory(db, mid)
    assert row["access_count"] == 2


def test_memoryStats(db: sqlite3.Connection):
    insertMemory(db, "a", [0.1, 0.2, 0.3, 0.4])
    insertMemory(db, "b", [0.5, 0.6, 0.7, 0.8])
    stats = memoryStats(db)
    assert stats["total_memories"] == 2
    assert stats["entities"] == 0


# -- Knowledge Graph --


def test_entityCRUD(db: sqlite3.Connection):
    eid = upsertEntity(db, "Python", "language")
    assert eid > 0
    # Upsert same name returns same id
    eid2 = upsertEntity(db, "Python", "language")
    assert eid2 == eid


def test_observationsAndRelations(db: sqlite3.Connection):
    py_id = upsertEntity(db, "Python", "language")
    rust_id = upsertEntity(db, "Rust", "language")

    obs_id = addObservation(db, py_id, "Created by Guido van Rossum")
    assert obs_id > 0
    # Duplicate observation returns existing id
    obs_id2 = addObservation(db, py_id, "Created by Guido van Rossum")
    assert obs_id2 == obs_id

    rel_id = addRelation(db, py_id, rust_id, "competes_with")
    assert rel_id > 0

    stats = memoryStats(db)
    assert stats["entities"] == 2
    assert stats["relations"] == 1
    assert stats["observations"] == 1


def test_deleteEntityCascades(db: sqlite3.Connection):
    eid = upsertEntity(db, "Temp", "thing")
    addObservation(db, eid, "Will be deleted")
    eid2 = upsertEntity(db, "Other", "thing")
    addRelation(db, eid, eid2, "related_to")

    assert deleteEntity(db, "Temp")
    stats = memoryStats(db)
    assert stats["entities"] == 1  # Only "Other" remains
    assert stats["observations"] == 0
    assert stats["relations"] == 0

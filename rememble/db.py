"""SQLite + sqlite-vec + FTS5 database setup and operations."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from rememble.config import RemembleConfig

SCHEMA_VERSION = 1


def connect(config: RemembleConfig) -> sqlite3.Connection:
    """Open DB, load sqlite-vec, run migrations."""
    db_path = Path(config.db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=5000")

    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    _migrate(db, config.embedding_dimensions)
    return db


def _migrate(db: sqlite3.Connection, dimensions: int) -> None:
    """Create tables if they don't exist."""
    db.executescript("""
        -- Core memories table
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT,
            tags TEXT,
            metadata_json TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            accessed_at INTEGER NOT NULL,
            access_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        );

        -- Full-text search index
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
            content,
            source,
            tags,
            content='memories',
            content_rowid='id',
            tokenize='porter unicode61'
        );

        -- FTS sync triggers
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO fts_memories(rowid, content, source, tags)
            VALUES (new.id, new.content, new.source, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO fts_memories(fts_memories, rowid, content, source, tags)
            VALUES ('delete', old.id, old.content, old.source, old.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO fts_memories(fts_memories, rowid, content, source, tags)
            VALUES ('delete', old.id, old.content, old.source, old.tags);
            INSERT INTO fts_memories(rowid, content, source, tags)
            VALUES (new.id, new.content, new.source, new.tags);
        END;

        -- Knowledge graph: entities
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            entity_type TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

        -- Knowledge graph: observations
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            source TEXT,
            valid_from INTEGER,
            valid_to INTEGER,
            created_at INTEGER NOT NULL,
            UNIQUE(entity_id, content)
        );

        -- Knowledge graph: relations
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            to_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            metadata_json TEXT,
            created_at INTEGER NOT NULL,
            UNIQUE(from_entity_id, to_entity_id, relation_type)
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_entity_id);
        CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_entity_id);
        CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id);
        CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
        CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at);
    """)

    # sqlite-vec virtual table (can't be in executescript due to extension)
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
            memory_id INTEGER PRIMARY KEY,
            embedding float[{dimensions}],
            created_at INTEGER
        )
    """)
    db.commit()


# -- Memory CRUD --


def insertMemory(
    db: sqlite3.Connection,
    content: str,
    embedding: list[float],
    source: str | None = None,
    tags: str | None = None,
    metadata_json: str | None = None,
) -> int:
    """Insert a memory + its embedding. Returns memory ID."""
    now = int(time.time() * 1000)
    cursor = db.execute(
        """INSERT INTO memories (content, source, tags, metadata_json,
           created_at, updated_at, accessed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (content, source, tags, metadata_json, now, now, now),
    )
    memory_id = cursor.lastrowid
    assert memory_id is not None

    db.execute(
        "INSERT INTO vec_memories (memory_id, embedding, created_at) VALUES (?, ?, ?)",
        (memory_id, serialize_float32(embedding), now),
    )
    db.commit()
    return memory_id


def getMemory(db: sqlite3.Connection, memory_id: int) -> sqlite3.Row | None:
    """Fetch a single memory by ID."""
    return db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()


def softDeleteMemory(db: sqlite3.Connection, memory_id: int) -> bool:
    """Soft-delete a memory. Returns True if found."""
    now = int(time.time() * 1000)
    cursor = db.execute(
        "UPDATE memories SET status = 'deleted', updated_at = ? WHERE id = ? AND status = 'active'",
        (now, memory_id),
    )
    db.commit()
    return cursor.rowcount > 0


def updateAccessStats(db: sqlite3.Connection, memory_ids: list[int]) -> None:
    """Bump accessed_at and access_count for recalled memories."""
    if not memory_ids:
        return
    now = int(time.time() * 1000)
    placeholders = ",".join("?" for _ in memory_ids)
    db.execute(
        f"""UPDATE memories SET accessed_at = ?, access_count = access_count + 1
            WHERE id IN ({placeholders})""",
        [now, *memory_ids],
    )
    db.commit()


def listMemories(
    db: sqlite3.Connection,
    source: str | None = None,
    tags: str | None = None,
    status: str = "active",
    limit: int = 20,
    offset: int = 0,
) -> list[sqlite3.Row]:
    """List memories with optional filters."""
    conditions = ["status = ?"]
    params: list[str | int] = [status]
    if source:
        conditions.append("source = ?")
        params.append(source)
    if tags:
        conditions.append("tags LIKE ?")
        params.append(f"%{tags}%")
    where = " AND ".join(conditions)
    params.extend([limit, offset])
    return db.execute(
        f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params,
    ).fetchall()


def memoryStats(db: sqlite3.Connection) -> dict:
    """Return DB statistics."""
    total = db.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone()[0]
    deleted = db.execute("SELECT COUNT(*) FROM memories WHERE status = 'deleted'").fetchone()[0]
    entities = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    relations = db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    observations = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    return {
        "total_memories": total,
        "deleted_memories": deleted,
        "entities": entities,
        "relations": relations,
        "observations": observations,
    }


# -- Knowledge Graph CRUD --


def upsertEntity(
    db: sqlite3.Connection,
    name: str,
    entity_type: str,
) -> int:
    """Create or get an entity. Returns entity ID."""
    now = int(time.time() * 1000)
    row = db.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
    if row:
        db.execute("UPDATE entities SET updated_at = ? WHERE id = ?", (now, row["id"]))
        db.commit()
        return row["id"]
    cursor = db.execute(
        "INSERT INTO entities (name, entity_type, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (name, entity_type, now, now),
    )
    db.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def addObservation(
    db: sqlite3.Connection,
    entity_id: int,
    content: str,
    source: str | None = None,
    valid_from: int | None = None,
    valid_to: int | None = None,
) -> int:
    """Add an observation to an entity. Returns observation ID."""
    now = int(time.time() * 1000)
    cursor = db.execute(
        """INSERT OR IGNORE INTO observations
           (entity_id, content, source, valid_from, valid_to, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (entity_id, content, source, valid_from, valid_to, now),
    )
    db.commit()
    if cursor.lastrowid and cursor.rowcount > 0:
        return cursor.lastrowid
    # Already existed
    row = db.execute(
        "SELECT id FROM observations WHERE entity_id = ? AND content = ?",
        (entity_id, content),
    ).fetchone()
    return row["id"]


def addRelation(
    db: sqlite3.Connection,
    from_entity_id: int,
    to_entity_id: int,
    relation_type: str,
    metadata_json: str | None = None,
) -> int:
    """Create a relation between entities. Returns relation ID."""
    now = int(time.time() * 1000)
    cursor = db.execute(
        """INSERT OR IGNORE INTO relations
           (from_entity_id, to_entity_id, relation_type, metadata_json, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (from_entity_id, to_entity_id, relation_type, metadata_json, now),
    )
    db.commit()
    if cursor.lastrowid and cursor.rowcount > 0:
        return cursor.lastrowid
    row = db.execute(
        """SELECT id FROM relations
           WHERE from_entity_id = ? AND to_entity_id = ? AND relation_type = ?""",
        (from_entity_id, to_entity_id, relation_type),
    ).fetchone()
    return row["id"]


def deleteEntity(db: sqlite3.Connection, name: str) -> bool:
    """Delete entity by name (cascades to observations + relations). Returns True if found."""
    cursor = db.execute("DELETE FROM entities WHERE name = ?", (name,))
    db.commit()
    return cursor.rowcount > 0

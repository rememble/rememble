"""SQLite + sqlite-vec + FTS5 database setup and operations."""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from rememble.config import RemembleConfig

SCHEMA_VERSION = 2
VEC_GLOBAL = "_global"  # sentinel for NULL project in vec_memories
logger = logging.getLogger("rememble")


def connect(
    config: RemembleConfig,
    *,
    dimensions: int | None = None,
    check_same_thread: bool = True,
) -> tuple[sqlite3.Connection, bool]:
    """Open DB, load sqlite-vec, run migrations.

    Returns (connection, needs_reembed) — caller must re-embed if True.
    """
    db_path = Path(config.db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=5000")

    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    dims = dimensions or config.embedding.dimensions
    needs_reembed = _migrate(db, dims)
    return db, needs_reembed


def _migrate(db: sqlite3.Connection, dimensions: int) -> bool:
    """Create tables if they don't exist. Returns True if re-embedding needed."""
    db.executescript("""
        -- Core memories table (v2: includes project)
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT,
            tags TEXT,
            metadata_json TEXT,
            project TEXT,
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

        -- Knowledge graph: entities (v2: project column, no UNIQUE on name alone)
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            project TEXT,
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

    # Meta table for tracking settings across restarts
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")

    # v1 → v2 migration: add project column if missing
    needs_reembed = False
    cols = {col[1] for col in db.execute("PRAGMA table_info(memories)").fetchall()}
    if "project" not in cols:
        try:
            needs_reembed = _migrateV1ToV2(db)
        except Exception:
            logger.exception("Schema migration v1→v2 failed")
            raise RuntimeError(
                "Failed to migrate database to v2 (project namespace). "
                "Back up and delete ~/.rememble/memory.db to start fresh, "
                "or check logs for details."
            ) from None

    # Project indexes — must run AFTER migration adds the column
    db.execute("CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project)")

    # Entity uniqueness: (name, project) with NULL handling via partial indexes
    db.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_project
           ON entities(name, project) WHERE project IS NOT NULL"""
    )
    db.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_global
           ON entities(name) WHERE project IS NULL"""
    )

    # Detect dimension mismatch — vec_memories is locked to the dimension it was created with
    stored = db.execute("SELECT value FROM meta WHERE key = 'vec_dimensions'").fetchone()
    stored_dims = int(stored[0]) if stored else None

    if stored_dims is not None and stored_dims != dimensions:
        logger.info(
            "Embedding dims changed (%d → %d), rebuilding vec_memories",
            stored_dims,
            dimensions,
        )
        db.execute("DROP TABLE IF EXISTS vec_memories")
        needs_reembed = True

    # sqlite-vec virtual table with project auxiliary column
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
            memory_id INTEGER PRIMARY KEY,
            embedding float[{dimensions}],
            project TEXT,
            created_at INTEGER
        )
    """)
    db.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('vec_dimensions', ?)",
        (str(dimensions),),
    )
    db.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    db.commit()
    return needs_reembed


def _migrateV1ToV2(db: sqlite3.Connection) -> bool:
    """Migrate v1 → v2: add project column, rebuild entities uniqueness."""
    logger.info("Migrating schema v1 → v2: adding project namespace")

    db.execute("ALTER TABLE memories ADD COLUMN project TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project)")
    db.commit()

    # Recreate entities table: drop UNIQUE(name), add project column
    db.executescript("""
        PRAGMA foreign_keys=OFF;

        CREATE TABLE entities_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            project TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        INSERT INTO entities_new (id, name, entity_type, project, created_at, updated_at)
            SELECT id, name, entity_type, NULL, created_at, updated_at FROM entities;
        DROP TABLE entities;
        ALTER TABLE entities_new RENAME TO entities;

        PRAGMA foreign_keys=ON;
    """)

    db.execute("CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project)")

    # Drop vec_memories — caller recreates with project auxiliary column
    db.execute("DROP TABLE IF EXISTS vec_memories")
    db.commit()
    return True


# -- Memory CRUD --


def insertMemory(
    db: sqlite3.Connection,
    content: str,
    embedding: list[float],
    source: str | None = None,
    tags: str | None = None,
    metadata_json: str | None = None,
    project: str | None = None,
) -> int:
    """Insert a memory + its embedding. Returns memory ID."""
    now = int(time.time() * 1000)
    cursor = db.execute(
        """INSERT INTO memories (content, source, tags, metadata_json, project,
           created_at, updated_at, accessed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, source, tags, metadata_json, project, now, now, now),
    )
    memory_id = cursor.lastrowid
    assert memory_id is not None

    vec_project = project or VEC_GLOBAL
    db.execute(
        "INSERT INTO vec_memories (memory_id, embedding, project, created_at) VALUES (?, ?, ?, ?)",
        (memory_id, serialize_float32(embedding), vec_project, now),
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
    project: str | None = None,
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
    if project is not None:
        conditions.append("(project = ? OR project IS NULL)")
        params.append(project)
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
    project: str | None = None,
) -> int:
    """Create or get an entity scoped by (name, project). Returns entity ID."""
    now = int(time.time() * 1000)
    if project is not None:
        row = db.execute(
            "SELECT id FROM entities WHERE name = ? AND project = ?", (name, project)
        ).fetchone()
    else:
        row = db.execute(
            "SELECT id FROM entities WHERE name = ? AND project IS NULL", (name,)
        ).fetchone()
    if row:
        db.execute("UPDATE entities SET updated_at = ? WHERE id = ?", (now, row["id"]))
        db.commit()
        return row["id"]
    cursor = db.execute(
        """INSERT INTO entities (name, entity_type, project, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (name, entity_type, project, now, now),
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

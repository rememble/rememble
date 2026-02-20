"""Application state container — replaces scattered globals."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field

from sqlite_vec import serialize_float32

from rememble.config import RemembleConfig, loadConfig
from rememble.db import connect
from rememble.embeddings.base import EmbeddingProvider
from rememble.embeddings.factory import createProvider

logger = logging.getLogger("rememble")

DEFAULT_PORT = 7707


@dataclass(frozen=True)
class AppState:
    db: sqlite3.Connection
    embedder: EmbeddingProvider
    config: RemembleConfig
    port: int = field(default=DEFAULT_PORT)


async def createAppState(
    config: RemembleConfig | None = None,
    port: int | None = None,
    check_same_thread: bool = True,
) -> AppState:
    """Create AppState — loads config, opens DB, initialises embedder."""
    cfg = config or loadConfig()
    # Embedder first — probe gives us actual dimensions
    embedder = await createProvider(cfg)
    db, needs_reembed = connect(
        cfg, dimensions=embedder.dimensions, check_same_thread=check_same_thread
    )
    if needs_reembed:
        await _reembed(db, embedder)
    p = port or cfg.port
    logger.info("Rememble starting — db: %s, port: %d", cfg.db_path, p)
    return AppState(db=db, embedder=embedder, config=cfg, port=p)


async def _reembed(db: sqlite3.Connection, embedder: EmbeddingProvider) -> None:
    """Re-embed all active memories after a dimension change."""
    rows = db.execute("SELECT id, content FROM memories WHERE status = 'active'").fetchall()
    if not rows:
        logger.info("No memories to re-embed")
        return
    logger.info("Re-embedding %d memories with %s...", len(rows), embedder.name)
    # Batch in groups of 64
    batch_size = 64
    now = int(time.time() * 1000)
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [r["content"] for r in batch]
        embeddings = await embedder.embed(texts)
        for row, emb in zip(batch, embeddings, strict=True):
            db.execute(
                "INSERT INTO vec_memories (memory_id, embedding, created_at) VALUES (?, ?, ?)",
                (row["id"], serialize_float32(emb), now),
            )
    db.commit()
    logger.info("Re-embedded %d memories", len(rows))

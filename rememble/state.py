"""Application state container — replaces scattered globals."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

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
    db = connect(cfg, check_same_thread=check_same_thread)
    embedder = await createProvider(cfg)
    p = port or cfg.port
    logger.info("Rememble starting — db: %s, port: %d", cfg.db_path, p)
    logger.info("Embedding provider: %s (%d dims)", embedder.name, embedder.dimensions)
    return AppState(db=db, embedder=embedder, config=cfg, port=p)

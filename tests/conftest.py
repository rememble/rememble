"""Shared test fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rememble.config import EmbeddingConfig, RemembleConfig
from rememble.db import connect


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def config(tmp_db_path: str) -> RemembleConfig:
    return RemembleConfig(
        db_path=tmp_db_path,
        embedding=EmbeddingConfig(dimensions=4),  # tiny dims for tests
    )


@pytest.fixture
def db(config: RemembleConfig) -> sqlite3.Connection:
    conn = connect(config)
    yield conn
    conn.close()


class FakeEmbedder:
    """Deterministic fake embedder for tests. Returns normalized [hash % 1.0, ...] vectors."""

    def __init__(self, dimensions: int = 4):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return "fake/test"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._fakeVec(t) for t in texts]

    async def embedOne(self, text: str) -> list[float]:
        return self._fakeVec(text)

    def _fakeVec(self, text: str) -> list[float]:
        """Deterministic fake embedding based on text hash."""
        h = hash(text)
        vec = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self._dimensions)]
        # L2 normalize
        mag = sum(v * v for v in vec) ** 0.5
        if mag > 0:
            vec = [v / mag for v in vec]
        return vec


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dimensions=4)

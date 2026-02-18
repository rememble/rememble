"""Embedding provider protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface all embedding providers implement."""

    @property
    def dimensions(self) -> int: ...

    @property
    def name(self) -> str: ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...

    async def embedOne(self, text: str) -> list[float]:
        """Embed a single text. Default: calls embed([text])[0]."""
        ...

"""Cohere embedding provider via REST API."""

from __future__ import annotations

import os

import httpx


class CohereProvider:
    """Embedding provider using Cohere's embed API."""

    def __init__(
        self,
        model: str = "embed-v4.0",
        api_key: str | None = None,
        dimensions: int = 1024,
    ):
        self._model = model
        self._dimensions = dimensions
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Cohere API key required: set COHERE_API_KEY or config.embedding.cohere_api_key"
            )
        self._client = httpx.AsyncClient(
            base_url="https://api.cohere.com/v2",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=60.0,
        )
        self._input_type = "search_document"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"cohere/{self._model}"

    def forQuery(self) -> CohereProvider:
        """Return a copy configured for query embedding (not document)."""
        provider = CohereProvider.__new__(CohereProvider)
        provider._model = self._model
        provider._dimensions = self._dimensions
        provider._api_key = self._api_key
        provider._client = self._client
        provider._input_type = "search_query"
        return provider

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed via Cohere API."""
        resp = await self._client.post(
            "/embed",
            json={
                "model": self._model,
                "texts": texts,
                "input_type": self._input_type,
                "embedding_types": ["float"],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]["float"]

    async def embedOne(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]

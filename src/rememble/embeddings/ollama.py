"""Ollama embedding provider via /api/embed endpoint."""

from __future__ import annotations

import httpx


class OllamaProvider:
    """Embedding provider using Ollama's local API."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        url: str = "http://localhost:11434",
    ):
        self._model = model
        self._dimensions = dimensions
        self._url = url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._url, timeout=60.0)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed via Ollama /api/embed."""
        resp = await self._client.post("/api/embed", json={"model": self._model, "input": texts})
        resp.raise_for_status()
        return resp.json()["embeddings"]

    async def embedOne(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]

    async def healthCheck(self) -> bool:
        """Check if Ollama is reachable and has the model."""
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check model name with or without :latest tag
            return any(self._model in m for m in models)
        except (httpx.HTTPError, KeyError):
            return False

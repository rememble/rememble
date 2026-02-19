"""OpenAI-compatible embeddings provider (Ollama, OpenRouter, OpenAI, Cohere, etc.)."""

from __future__ import annotations

import httpx


class CompatProvider:
    def __init__(
        self,
        model: str,
        api_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        dimensions: int = 768,
    ):
        self._model = model
        self._dimensions = dimensions
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=api_url.rstrip("/"),
            headers=headers,
            timeout=60.0,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"compat/{self._model}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post("/embeddings", json={"model": self._model, "input": texts})
        resp.raise_for_status()
        items = sorted(resp.json()["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def embedOne(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

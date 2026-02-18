"""OpenAI-compatible embeddings provider (OpenRouter, OpenAI, Cohere compat, etc.)."""

from __future__ import annotations

import os

import httpx


class CompatProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        dimensions: int = 1536,
        url: str = "https://openrouter.ai/api/v1",
        api_type: str = "openrouter",
    ):
        self._model = model
        self._dimensions = dimensions
        self._api_type = api_type
        self._api_key = (
            api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "API key required: set OPENROUTER_API_KEY, OPENAI_API_KEY, or config.embedding.api_key"  # noqa: E501
            )
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=60.0,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"{self._api_type}/{self._model}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post("/embeddings", json={"model": self._model, "input": texts})
        resp.raise_for_status()
        items = sorted(resp.json()["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def embedOne(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

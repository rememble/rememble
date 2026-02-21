"""Sync HTTP client for the Rememble API."""

from __future__ import annotations

from typing import Any

import httpx

from rememble.state import DEFAULT_PORT


class RemembleClient:
    """Sync httpx client wrapping the Rememble HTTP API."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        url = base_url or f"http://localhost:{DEFAULT_PORT}/api"
        self._client = httpx.Client(base_url=url, timeout=timeout)

    # ── lifecycle ─────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RemembleClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── helpers ───────────────────────────────────────────────

    def _post(self, path: str, **kwargs: Any) -> dict:
        r = self._client.post(path, json=kwargs)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: dict | None = None) -> Any:
        r = self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str, **kwargs: Any) -> dict:
        r = self._client.request("DELETE", path, json=kwargs)
        r.raise_for_status()
        return r.json()

    # ── health ────────────────────────────────────────────────

    def health(self) -> dict:
        return self._get("/health")

    # ── core memory ───────────────────────────────────────────

    def remember(
        self,
        content: str,
        source: str | None = None,
        tags: str | None = None,
        metadata: str | None = None,
        project: str | None = None,
    ) -> dict:
        return self._post(
            "/remember",
            content=content, source=source, tags=tags, metadata=metadata, project=project,
        )

    def recall(
        self,
        query: str,
        limit: int = 10,
        use_rag: bool = True,
        project: str | None = None,
    ) -> dict:
        return self._post(
            "/recall", query=query, limit=limit, use_rag=use_rag, project=project
        )

    def forget(self, memory_id: int) -> dict:
        return self._post("/forget", memory_id=memory_id)

    def listMemories(
        self,
        source: str | None = None,
        tags: str | None = None,
        status: str = "active",
        limit: int = 20,
        offset: int = 0,
        project: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"status": status, "limit": limit, "offset": offset}
        if source:
            params["source"] = source
        if tags:
            params["tags"] = tags
        if project:
            params["project"] = project
        return self._get("/memories", params=params)

    def stats(self) -> dict:
        return self._get("/stats")

    # ── knowledge graph ───────────────────────────────────────

    def createEntities(self, entities: list[dict], project: str | None = None) -> dict:
        return self._post("/entities", entities=entities, project=project)

    def createRelations(self, relations: list[dict]) -> dict:
        return self._post("/relations", relations=relations)

    def addObservations(
        self, entity_name: str, observations: list[str], source: str | None = None
    ) -> dict:
        return self._post(
            "/observations", entity_name=entity_name, observations=observations, source=source
        )

    def searchGraph(self, query: str, limit: int = 10, project: str | None = None) -> dict:
        params: dict[str, Any] = {"query": query, "limit": limit}
        if project:
            params["project"] = project
        return self._get("/graph", params=params)

    def deleteEntities(self, names: list[str]) -> dict:
        return self._delete("/entities", names=names)

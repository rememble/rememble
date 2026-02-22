"""Vector search via sqlite-vec KNN."""

from __future__ import annotations

import sqlite3

from sqlite_vec import serialize_float32

from rememble.db import VEC_GLOBAL
from rememble.models import SearchResult


def vectorSearch(
    db: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 20,
    time_range: tuple[int, int] | None = None,
    project: str | None = None,
) -> list[SearchResult]:
    """KNN search via sqlite-vec. Returns results scored as 1 - cosine_distance."""
    query_blob = serialize_float32(query_embedding)

    if project:
        # Separate KNN for project-scoped + global, then merge top-k
        proj_rows = _knn(db, query_blob, limit, time_range, project)
        global_rows = _knn(db, query_blob, limit, time_range, VEC_GLOBAL)

        best: dict[int, float] = {}
        for row in [*proj_rows, *global_rows]:
            mid, dist = row["memory_id"], row["distance"]
            if mid not in best or dist < best[mid]:
                best[mid] = dist
        ranked = sorted(best.items(), key=lambda x: x[1])[:limit]
        return [SearchResult(memory_id=m, score=1.0 - d, source="vector") for m, d in ranked]

    # No project filter: search everything
    rows = _knn(db, query_blob, limit, time_range)
    return [
        SearchResult(memory_id=row["memory_id"], score=1.0 - row["distance"], source="vector")
        for row in rows
    ]


def _knn(
    db: sqlite3.Connection,
    query_blob: bytes,
    limit: int,
    time_range: tuple[int, int] | None = None,
    project_val: str | None = None,
) -> list[sqlite3.Row]:
    """Run a single KNN query with optional time_range and project filters."""
    conditions = ["embedding MATCH ?", "k = ?"]
    params: list[object] = [query_blob, limit]

    if project_val is not None:
        conditions.append("project = ?")
        params.append(project_val)

    if time_range:
        conditions.append("created_at >= ?")
        conditions.append("created_at <= ?")
        params.extend([time_range[0], time_range[1]])

    where = " AND ".join(conditions)
    return db.execute(
        f"SELECT memory_id, distance FROM vec_memories WHERE {where} ORDER BY distance",
        params,
    ).fetchall()

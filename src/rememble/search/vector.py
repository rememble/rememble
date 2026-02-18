"""Vector search via sqlite-vec KNN."""

from __future__ import annotations

import sqlite3

from sqlite_vec import serialize_float32

from rememble.models import SearchResult


def vectorSearch(
    db: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 20,
    time_range: tuple[int, int] | None = None,
) -> list[SearchResult]:
    """KNN search via sqlite-vec. Returns results scored as 1 - cosine_distance."""
    query_blob = serialize_float32(query_embedding)

    if time_range:
        rows = db.execute(
            """SELECT memory_id, distance
               FROM vec_memories
               WHERE embedding MATCH ? AND k = ?
                 AND created_at >= ? AND created_at <= ?
               ORDER BY distance""",
            (query_blob, limit, time_range[0], time_range[1]),
        ).fetchall()
    else:
        rows = db.execute(
            """SELECT memory_id, distance
               FROM vec_memories
               WHERE embedding MATCH ? AND k = ?
               ORDER BY distance""",
            (query_blob, limit),
        ).fetchall()

    results = []
    for row in rows:
        score = 1.0 - row["distance"]  # cosine: lower distance = more similar
        results.append(
            SearchResult(
                memory_id=row["memory_id"],
                score=score,
                source="vector",
            )
        )
    return results

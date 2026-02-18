"""BM25 full-text search via SQLite FTS5."""

from __future__ import annotations

import re
import sqlite3

from rememble.models import SearchResult


def _sanitizeFtsQuery(query: str) -> str:
    """Sanitize query for FTS5 MATCH. Quotes tokens, strips special chars."""
    # Remove FTS5 operators and special characters
    cleaned = re.sub(r"[^\w\s]", " ", query)
    tokens = cleaned.split()
    if not tokens:
        return '""'
    # Quote each token and AND-join
    quoted = [f'"{t}"' for t in tokens if t.strip()]
    return " ".join(quoted)


def _orQuery(query: str) -> str:
    """Build an OR-expanded FTS5 query as fallback."""
    cleaned = re.sub(r"[^\w\s]", " ", query)
    tokens = cleaned.split()
    if not tokens:
        return '""'
    quoted = [f'"{t}"' for t in tokens if t.strip()]
    return " OR ".join(quoted)


def textSearch(
    db: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """FTS5 BM25 search. Returns results with snippet previews."""
    fts_query = _sanitizeFtsQuery(query)

    try:
        rows = db.execute(
            """SELECT m.id AS memory_id, bm25(fts_memories) AS rank,
                      snippet(fts_memories, 0, '[', ']', '...', 10) AS snippet
               FROM fts_memories
               JOIN memories m ON m.id = fts_memories.rowid
               WHERE fts_memories MATCH ? AND m.status = 'active'
               ORDER BY rank ASC
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # Fallback to OR query on FTS5 syntax errors
        rows = db.execute(
            """SELECT m.id AS memory_id, bm25(fts_memories) AS rank,
                      snippet(fts_memories, 0, '[', ']', '...', 10) AS snippet
               FROM fts_memories
               JOIN memories m ON m.id = fts_memories.rowid
               WHERE fts_memories MATCH ? AND m.status = 'active'
               ORDER BY rank ASC
               LIMIT ?""",
            (_orQuery(query), limit),
        ).fetchall()

    results = []
    for row in rows:
        # BM25 rank is negative (lower = better), flip to positive higher = better
        score = -row["rank"]
        results.append(
            SearchResult(
                memory_id=row["memory_id"],
                score=score,
                snippet=row["snippet"],
                source="text",
            )
        )
    return results

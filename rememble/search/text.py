"""BM25 full-text search via SQLite FTS5."""

from __future__ import annotations

import re
import sqlite3

from rememble.models import SearchResult

_FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}


def _normalizeBm25(rank: float) -> float:
    """Map BM25 rank to [0, 1). Monotonic — ranking unchanged."""
    raw = abs(rank)
    return raw / (1.0 + raw)


def buildFts5Query(query: str) -> str:
    """Build FTS5 MATCH expression supporting phrases, negation, operator stripping.

    - "exact phrase" → passed through
    - -term → NOT
    - AND/OR/NOT/NEAR stripped from bare terms
    - Bare tokens sanitized and AND-joined
    """
    negative: list[str] = []

    # Extract quoted phrases first, then process remaining tokens
    parts: list[str] = []
    remaining = query
    for m in re.finditer(r'"([^"]*)"', query):
        phrase = m.group(1).strip()
        if phrase:
            parts.append(f'"{phrase}"')
    # Remove quoted parts from remaining
    remaining = re.sub(r'"[^"]*"', " ", remaining)

    for token in remaining.split():
        if token.startswith("-") and len(token) > 1:
            clean = re.sub(r"[^\w]", "", token[1:])
            if clean and clean.upper() not in _FTS5_OPERATORS:
                negative.append(f'"{clean}"')
        else:
            clean = re.sub(r"[^\w]", "", token)
            if clean and clean.upper() not in _FTS5_OPERATORS:
                parts.append(f'"{clean}"')

    if not parts and not negative:
        return '""'

    # AND-join positive terms + phrases
    expr = " ".join(parts) if parts else '""'

    # Chain negation: ((pos) NOT neg1) NOT neg2
    for neg in negative:
        expr = f"({expr}) NOT {neg}"

    return expr


def _orQuery(query: str) -> str:
    """Build an OR-expanded FTS5 query as fallback."""
    positive: list[str] = []

    for m in re.finditer(r'"([^"]*)"', query):
        phrase = m.group(1).strip()
        if phrase:
            positive.append(f'"{phrase}"')
    remaining = re.sub(r'"[^"]*"', " ", query)

    for token in remaining.split():
        if token.startswith("-"):
            continue
        clean = re.sub(r"[^\w]", "", token)
        if clean and clean.upper() not in _FTS5_OPERATORS:
            positive.append(f'"{clean}"')

    if not positive:
        return '""'
    return " OR ".join(positive)


def textSearch(
    db: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """FTS5 BM25 search. Returns results with snippet previews."""
    fts_query = buildFts5Query(query)

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
        score = _normalizeBm25(row["rank"])
        results.append(
            SearchResult(
                memory_id=row["memory_id"],
                score=score,
                snippet=row["snippet"],
                source="text",
            )
        )
    return results

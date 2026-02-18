"""Reciprocal Rank Fusion across search lanes."""

from __future__ import annotations

import sqlite3

from rememble.config import SearchConfig
from rememble.db import updateAccessStats
from rememble.models import FusedResult, GraphResult
from rememble.search.graph import graphSearch
from rememble.search.temporal import temporalScore
from rememble.search.text import textSearch
from rememble.search.vector import vectorSearch


def _rrfScore(rank: int, weight: float, k: int) -> float:
    """RRF contribution: weight / (k + rank)."""
    return weight / (k + rank)


class HybridSearchResult:
    def __init__(self, results: list[FusedResult], graph: list[GraphResult]):
        self.results = results
        self.graph = graph


def hybridSearch(
    db: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    config: SearchConfig,
    limit: int | None = None,
    time_range: tuple[int, int] | None = None,
) -> HybridSearchResult:
    """Run all search lanes, fuse with RRF, return ranked results.

    Lanes:
    1. BM25 text search (weight: bm25_weight)
    2. Vector KNN search (weight: vector_weight)
    3. Temporal scoring (weight: temporal_weight) — applied to candidates from lanes 1+2
    4. Knowledge graph — entities appended as graph-sourced results
    """
    effective_limit = limit or config.default_limit
    candidate_limit = min(effective_limit * 3, 200)
    k = config.rrf_k

    # Lane 1: BM25
    text_results = textSearch(db, query, limit=candidate_limit)

    # Lane 2: Vector
    vector_results = vectorSearch(db, query_embedding, limit=candidate_limit, time_range=time_range)

    # Collect all candidate memory IDs
    all_ids: set[int] = set()
    for r in text_results:
        all_ids.add(r.memory_id)
    for r in vector_results:
        all_ids.add(r.memory_id)

    # Lane 3: Temporal scoring for all candidates
    temporal_scores: dict[int, float] = {}
    if all_ids and config.temporal_weight > 0:
        placeholders = ",".join("?" for _ in all_ids)
        rows = db.execute(
            f"""SELECT id, created_at, accessed_at, access_count
                FROM memories WHERE id IN ({placeholders}) AND status = 'active'""",
            list(all_ids),
        ).fetchall()
        for row in rows:
            temporal_scores[row["id"]] = temporalScore(
                row["created_at"],
                row["accessed_at"],
                row["access_count"],
                config.recency_half_life_days,
            )

    # Build temporal ranking (sorted by temporal score desc)
    temporal_ranked = sorted(temporal_scores.items(), key=lambda x: x[1], reverse=True)

    # RRF accumulator
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    sources: dict[int, list[str]] = {}
    snippets: dict[int, str | None] = {}

    # Accumulate text lane
    for rank, r in enumerate(text_results, 1):
        scores[r.memory_id] = scores.get(r.memory_id, 0) + _rrfScore(rank, config.bm25_weight, k)
        best_rank[r.memory_id] = min(best_rank.get(r.memory_id, rank), rank)
        sources.setdefault(r.memory_id, []).append("text")
        if r.snippet:
            snippets[r.memory_id] = r.snippet

    # Accumulate vector lane
    for rank, r in enumerate(vector_results, 1):
        scores[r.memory_id] = scores.get(r.memory_id, 0) + _rrfScore(rank, config.vector_weight, k)
        best_rank[r.memory_id] = min(best_rank.get(r.memory_id, rank), rank)
        sources.setdefault(r.memory_id, []).append("vector")

    # Accumulate temporal lane
    for rank, (mid, _tscore) in enumerate(temporal_ranked, 1):
        scores[mid] = scores.get(mid, 0) + _rrfScore(rank, config.temporal_weight, k)
        best_rank[mid] = min(best_rank.get(mid, rank), rank)
        sources.setdefault(mid, []).append("temporal")

    # Sort: fused score desc → best rank asc → memory ID asc
    ranked = sorted(
        scores.keys(),
        key=lambda mid: (-scores[mid], best_rank.get(mid, 999999), mid),
    )

    # Fetch content for top results
    top_ids = ranked[:effective_limit]
    content_map: dict[int, str] = {}
    if top_ids:
        placeholders = ",".join("?" for _ in top_ids)
        rows = db.execute(
            f"SELECT id, content FROM memories WHERE id IN ({placeholders})",
            top_ids,
        ).fetchall()
        for row in rows:
            content_map[row["id"]] = row["content"]

    results = [
        FusedResult(
            memory_id=mid,
            score=scores[mid],
            best_rank=best_rank.get(mid, 0),
            sources=sources.get(mid, []),
            snippet=snippets.get(mid),
            content=content_map.get(mid),
        )
        for mid in top_ids
    ]

    # Update access stats for returned results
    updateAccessStats(db, top_ids)

    # Lane 4: Graph search — separate lane, not fused numerically
    graph_results = graphSearch(db, query, limit=5)

    return HybridSearchResult(results=results, graph=graph_results)

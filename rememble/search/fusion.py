"""Reciprocal Rank Fusion across search lanes."""

from __future__ import annotations

import sqlite3

from rememble.config import SearchConfig
from rememble.db import updateAccessStats
from rememble.models import FusedResult, GraphResult, SearchResult
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


def _temporalScores(
    db: sqlite3.Connection, ids: set[int], config: SearchConfig
) -> dict[int, float]:
    """Compute temporal scores for a set of memory IDs."""
    if not ids or config.temporal_weight <= 0:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = db.execute(
        f"""SELECT id, created_at, accessed_at, access_count
            FROM memories WHERE id IN ({placeholders}) AND status = 'active'""",
        list(ids),
    ).fetchall()
    return {
        row["id"]: temporalScore(
            row["created_at"],
            row["accessed_at"],
            row["access_count"],
            config.recency_half_life_days,
        )
        for row in rows
    }


def _fuseResults(
    db: sqlite3.Connection,
    config: SearchConfig,
    effective_limit: int,
    text_results: list[SearchResult],
    vector_results: list[SearchResult] | None = None,
) -> list[FusedResult]:
    """RRF accumulation + content fetch for fused lanes.

    vector_results=None → text-only fusion (BM25 + temporal).
    """
    k = config.rrf_k

    all_ids: set[int] = {r.memory_id for r in text_results}
    if vector_results:
        all_ids |= {r.memory_id for r in vector_results}

    temporal_scores = _temporalScores(db, all_ids, config)
    temporal_ranked = sorted(temporal_scores.items(), key=lambda x: x[1], reverse=True)

    # RRF accumulator
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    sources: dict[int, list[str]] = {}
    snippets: dict[int, str | None] = {}

    # Text lane
    for rank, r in enumerate(text_results, 1):
        scores[r.memory_id] = scores.get(r.memory_id, 0) + _rrfScore(rank, config.bm25_weight, k)
        best_rank[r.memory_id] = min(best_rank.get(r.memory_id, rank), rank)
        sources.setdefault(r.memory_id, []).append("text")
        if r.snippet:
            snippets[r.memory_id] = r.snippet

    # Vector lane (optional)
    if vector_results:
        for rank, r in enumerate(vector_results, 1):
            scores[r.memory_id] = scores.get(r.memory_id, 0) + _rrfScore(
                rank, config.vector_weight, k
            )
            best_rank[r.memory_id] = min(best_rank.get(r.memory_id, rank), rank)
            sources.setdefault(r.memory_id, []).append("vector")

    # Temporal lane
    for rank, (mid, _tscore) in enumerate(temporal_ranked, 1):
        scores[mid] = scores.get(mid, 0) + _rrfScore(rank, config.temporal_weight, k)
        best_rank[mid] = min(best_rank.get(mid, rank), rank)
        sources.setdefault(mid, []).append("temporal")

    ranked = sorted(
        scores.keys(),
        key=lambda mid: (-scores[mid], best_rank.get(mid, 999999), mid),
    )

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

    updateAccessStats(db, top_ids)
    return results


def bm25Probe(
    db: sqlite3.Connection,
    query: str,
    config: SearchConfig,
    limit: int,
    project: str | None = None,
) -> tuple[float, list[SearchResult]]:
    """Run BM25 text search, return (top_normalized_score, results)."""
    candidate_limit = min(limit * 3, 200)
    results = textSearch(db, query, limit=candidate_limit, project=project)
    top_score = results[0].score if results else 0.0
    return top_score, results


def hybridSearchTextOnly(
    db: sqlite3.Connection,
    query: str,
    text_results: list[SearchResult],
    config: SearchConfig,
    limit: int | None = None,
    project: str | None = None,
) -> HybridSearchResult:
    """Fuse BM25 + temporal + graph, skip vector lane."""
    effective_limit = limit or config.default_limit
    fused = _fuseResults(db, config, effective_limit, text_results)
    graph_results = graphSearch(db, query, limit=5, project=project)
    return HybridSearchResult(results=fused, graph=graph_results)


def hybridSearch(
    db: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    config: SearchConfig,
    limit: int | None = None,
    time_range: tuple[int, int] | None = None,
    project: str | None = None,
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

    text_results = textSearch(db, query, limit=candidate_limit, project=project)
    vector_results = vectorSearch(
        db, query_embedding, limit=candidate_limit, time_range=time_range, project=project
    )

    fused = _fuseResults(db, config, effective_limit, text_results, vector_results)
    graph_results = graphSearch(db, query, limit=5, project=project)

    return HybridSearchResult(results=fused, graph=graph_results)

"""Tests for search pipeline: vector, text, temporal, graph, fusion."""

from __future__ import annotations

import sqlite3
import time

from rememble.config import SearchConfig
from rememble.db import addObservation, addRelation, insertMemory, upsertEntity
from rememble.search.fusion import hybridSearch
from rememble.search.graph import graphSearch
from rememble.search.temporal import temporalScore
from rememble.search.text import textSearch
from rememble.search.vector import vectorSearch


def _insertWithEmbedding(db, content, embedding, source=None, tags=None):
    return insertMemory(db, content, embedding, source=source, tags=tags)


class TestVectorSearch:
    def test_basicKnn(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "cats are nice", [1.0, 0.0, 0.0, 0.0])
        _insertWithEmbedding(db, "dogs are cool", [0.0, 1.0, 0.0, 0.0])
        _insertWithEmbedding(db, "cats and dogs", [0.7, 0.7, 0.0, 0.0])

        results = vectorSearch(db, [1.0, 0.0, 0.0, 0.0], limit=2)
        assert len(results) == 2
        assert results[0].memory_id == 1  # exact match
        assert results[0].score > results[1].score

    def test_emptyDb(self, db: sqlite3.Connection):
        results = vectorSearch(db, [1.0, 0.0, 0.0, 0.0], limit=5)
        assert len(results) == 0


class TestTextSearch:
    def test_basicFts(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "python programming language", [0.1, 0.2, 0.3, 0.4])
        _insertWithEmbedding(db, "rust systems programming", [0.5, 0.6, 0.7, 0.8])
        _insertWithEmbedding(db, "cooking recipes for dinner", [0.9, 0.1, 0.2, 0.3])

        results = textSearch(db, "programming", limit=5)
        assert len(results) == 2
        assert all(r.source == "text" for r in results)

    def test_emptyQuery(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "some content", [0.1, 0.2, 0.3, 0.4])
        results = textSearch(db, "", limit=5)
        # Empty query should not crash
        assert isinstance(results, list)


class TestTemporalScore:
    def test_newHigherThanOld(self):
        now = int(time.time() * 1000)
        week_ago = now - 7 * 24 * 3_600_000

        new_score = temporalScore(now, now, 0)
        old_score = temporalScore(week_ago, week_ago, 0)
        assert new_score > old_score

    def test_frequentlyAccessedHigher(self):
        now = int(time.time() * 1000)
        score_0 = temporalScore(now, now, 0)
        score_10 = temporalScore(now, now, 10)
        assert score_10 > score_0


class TestGraphSearch:
    def test_findByName(self, db: sqlite3.Connection):
        eid = upsertEntity(db, "Python", "language")
        addObservation(db, eid, "High-level language")

        results = graphSearch(db, "Python")
        assert len(results) == 1
        assert results[0].entity.name == "Python"
        assert len(results[0].observations) == 1

    def test_findByObservation(self, db: sqlite3.Connection):
        eid = upsertEntity(db, "Rust", "language")
        addObservation(db, eid, "Memory safe without garbage collector")

        results = graphSearch(db, "garbage collector")
        assert len(results) == 1
        assert results[0].entity.name == "Rust"

    def test_relationsIncluded(self, db: sqlite3.Connection):
        py = upsertEntity(db, "Python", "language")
        django = upsertEntity(db, "Django", "framework")
        addRelation(db, django, py, "written_in")

        results = graphSearch(db, "Django")
        assert len(results) == 1
        assert len(results[0].relations) == 1
        assert results[0].relations[0].entity.name == "Python"


class TestHybridSearch:
    def test_fusesResults(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "machine learning with python", [1.0, 0.0, 0.0, 0.0])
        _insertWithEmbedding(db, "deep learning neural networks", [0.9, 0.1, 0.0, 0.0])
        _insertWithEmbedding(db, "cooking pasta recipes", [0.0, 0.0, 1.0, 0.0])

        config = SearchConfig()
        result = hybridSearch(db, "machine learning", [1.0, 0.0, 0.0, 0.0], config, limit=3)
        assert len(result.results) > 0
        # First result should be most relevant
        assert result.results[0].memory_id == 1
        # Should have multiple source lanes
        assert len(result.results[0].sources) > 0

"""Tests for search pipeline: vector, text, temporal, graph, fusion."""

from __future__ import annotations

import sqlite3
import time

from rememble.config import SearchConfig
from rememble.db import addObservation, addRelation, insertMemory, upsertEntity
from rememble.search.fusion import bm25Probe, hybridSearch, hybridSearchTextOnly
from rememble.search.graph import graphSearch
from rememble.search.temporal import temporalScore
from rememble.search.text import _normalizeBm25, buildFts5Query, textSearch
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


class TestBm25Normalization:
    def test_zeroMapsToZero(self):
        assert _normalizeBm25(0.0) == 0.0

    def test_oneMapsToHalf(self):
        assert _normalizeBm25(-1.0) == 0.5

    def test_nineMapsToPointNine(self):
        assert abs(_normalizeBm25(-9.0) - 0.9) < 1e-9

    def test_monotonic(self):
        scores = [_normalizeBm25(-x) for x in range(20)]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]

    def test_boundedBelow1(self):
        assert _normalizeBm25(-1000.0) < 1.0

    def test_textSearchScoresNormalized(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "alpha beta gamma", [0.1, 0.2, 0.3, 0.4])
        _insertWithEmbedding(db, "alpha delta epsilon", [0.5, 0.6, 0.7, 0.8])
        results = textSearch(db, "alpha", limit=5)
        for r in results:
            assert 0.0 <= r.score < 1.0


class TestFts5QueryBuilder:
    def test_bareTerms(self):
        assert buildFts5Query("hello world") == '"hello" "world"'

    def test_quotedPhrase(self):
        q = buildFts5Query('"exact phrase"')
        assert '"exact phrase"' in q

    def test_negation(self):
        q = buildFts5Query("python -java")
        assert "NOT" in q
        assert '"java"' in q
        assert '"python"' in q

    def test_multipleNegation(self):
        q = buildFts5Query("python -java -rust")
        assert q.count("NOT") == 2

    def test_operatorsStripped(self):
        q = buildFts5Query("python AND OR NOT rust")
        # AND/OR/NOT should be stripped as bare terms
        assert '"python"' in q
        assert '"rust"' in q
        assert q.count('"AND"') == 0
        assert q.count('"OR"') == 0

    def test_empty(self):
        assert buildFts5Query("") == '""'

    def test_onlyNegation(self):
        # Only negative terms → '""' as base with NOT chains
        q = buildFts5Query("-bad -worse")
        assert '""' in q
        assert "NOT" in q
        assert q.count("NOT") == 2

    def test_phraseSearchReturnsExactMatch(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "the quick brown fox jumps", [0.1, 0.2, 0.3, 0.4])
        _insertWithEmbedding(db, "quick thinking brown bear", [0.5, 0.6, 0.7, 0.8])
        results = textSearch(db, '"quick brown"', limit=5)
        assert len(results) == 1
        assert results[0].memory_id == 1

    def test_negationExcludesTarget(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "python programming language", [0.1, 0.2, 0.3, 0.4])
        _insertWithEmbedding(db, "java programming language", [0.5, 0.6, 0.7, 0.8])
        results = textSearch(db, "programming -java", limit=5)
        assert len(results) == 1
        assert results[0].memory_id == 1


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


class TestBm25Probe:
    def test_probeReturnsTopScore(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "alpha beta gamma", [0.1, 0.2, 0.3, 0.4])
        config = SearchConfig()
        top_score, results = bm25Probe(db, "alpha", config, limit=5)
        assert top_score > 0.0
        assert top_score < 1.0
        assert len(results) >= 1

    def test_probeEmptyDb(self, db: sqlite3.Connection):
        config = SearchConfig()
        top_score, results = bm25Probe(db, "nothing", config, limit=5)
        assert top_score == 0.0
        assert results == []


class TestHybridSearchTextOnly:
    def test_textOnlyFusion(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "python programming language", [0.1, 0.2, 0.3, 0.4])
        _insertWithEmbedding(db, "rust systems programming", [0.5, 0.6, 0.7, 0.8])
        config = SearchConfig()
        _, text_results = bm25Probe(db, "programming", config, limit=5)
        result = hybridSearchTextOnly(db, "programming", text_results, config, limit=5)
        assert len(result.results) >= 1
        # Should not have vector source
        for r in result.results:
            assert "vector" not in r.sources


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


# -- Project-scoped search --


class TestProjectScopedTextSearch:
    def test_projectFilterIncludesGlobal(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "global python tip", [0.1, 0.2, 0.3, 0.4])
        insertMemory(db, "myapp python tip", [0.5, 0.6, 0.7, 0.8], project="myapp")
        insertMemory(db, "other python tip", [0.9, 0.1, 0.2, 0.3], project="other")

        results = textSearch(db, "python", limit=10, project="myapp")
        # Should find global + myapp, not other
        assert len(results) == 2
        # Verify we got the right ones by checking global and myapp are included
        all_content = []
        for r in results:
            row = db.execute("SELECT project FROM memories WHERE id = ?", (r.memory_id,)).fetchone()
            all_content.append(row["project"])
        assert None in all_content  # global
        assert "myapp" in all_content

    def test_noProjectSearchesAll(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "global python tip", [0.1, 0.2, 0.3, 0.4])
        insertMemory(db, "myapp python tip", [0.5, 0.6, 0.7, 0.8], project="myapp")
        insertMemory(db, "other python tip", [0.9, 0.1, 0.2, 0.3], project="other")

        results = textSearch(db, "python", limit=10)
        assert len(results) == 3


class TestProjectScopedVectorSearch:
    def test_projectFilterIncludesGlobal(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "global vec", [1.0, 0.0, 0.0, 0.0])
        insertMemory(db, "myapp vec", [0.9, 0.1, 0.0, 0.0], project="myapp")
        insertMemory(db, "other vec", [0.8, 0.2, 0.0, 0.0], project="other")

        results = vectorSearch(db, [1.0, 0.0, 0.0, 0.0], limit=10, project="myapp")
        projects = set()
        for r in results:
            row = db.execute("SELECT project FROM memories WHERE id = ?", (r.memory_id,)).fetchone()
            projects.add(row["project"])
        assert None in projects  # global
        assert "myapp" in projects
        assert "other" not in projects

    def test_noProjectSearchesAll(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "global vec", [1.0, 0.0, 0.0, 0.0])
        insertMemory(db, "myapp vec", [0.9, 0.1, 0.0, 0.0], project="myapp")
        insertMemory(db, "other vec", [0.8, 0.2, 0.0, 0.0], project="other")

        results = vectorSearch(db, [1.0, 0.0, 0.0, 0.0], limit=10)
        assert len(results) == 3


class TestProjectScopedGraphSearch:
    def test_projectFilterIncludesGlobal(self, db: sqlite3.Connection):
        eid1 = upsertEntity(db, "Python", "language")
        addObservation(db, eid1, "Global language")
        eid2 = upsertEntity(db, "MyLib", "library", project="myapp")
        addObservation(db, eid2, "myapp library")
        eid3 = upsertEntity(db, "OtherLib", "library", project="other")
        addObservation(db, eid3, "other library")

        results = graphSearch(db, "lib", limit=10, project="myapp")
        names = {r.entity.name for r in results}
        assert "MyLib" in names
        assert "OtherLib" not in names

    def test_noProjectSearchesAll(self, db: sqlite3.Connection):
        upsertEntity(db, "A", "test")
        upsertEntity(db, "B", "test", project="myapp")
        upsertEntity(db, "C", "test", project="other")

        results = graphSearch(db, "", limit=10)
        assert len(results) == 3


class TestProjectScopedFusion:
    def test_hybridSearchWithProject(self, db: sqlite3.Connection):
        _insertWithEmbedding(db, "global machine learning", [1.0, 0.0, 0.0, 0.0])
        insertMemory(db, "myapp machine learning", [0.9, 0.1, 0.0, 0.0], project="myapp")
        insertMemory(db, "other machine learning", [0.8, 0.2, 0.0, 0.0], project="other")

        config = SearchConfig()
        result = hybridSearch(
            db, "machine learning", [1.0, 0.0, 0.0, 0.0], config, limit=10, project="myapp"
        )
        projects = set()
        for r in result.results:
            row = db.execute("SELECT project FROM memories WHERE id = ?", (r.memory_id,)).fetchone()
            projects.add(row["project"])
        assert "other" not in projects

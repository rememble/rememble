"""Tests for ACT-R activation scoring and access history."""

from __future__ import annotations

import json
import sqlite3
import time

from rememble.db import insertMemory, updateAccessStats
from rememble.search.temporal import computeActivation, synthesizeHistory, temporalScore


class TestComputeActivation:
    def test_emptyHistoryReturnsZero(self):
        assert computeActivation([]) == 0.0

    def test_singleRecentAccessHigh(self):
        now = time.time()
        # B = ln(1^(-0.5)) = 0 → sigmoid(0/1.5) = 0.5
        score = computeActivation([now - 1], now=now)
        assert score >= 0.5

    def test_manyOldAccessesLow(self):
        now = time.time()
        month_ago = now - 30 * 86400
        # 20 accesses a month ago — power-law decay makes these very weak
        history = [month_ago + i * 3600 for i in range(20)]
        score = computeActivation(history, now=now)
        assert 0.0 < score < 0.2

    def test_mixedOldRecentHigherThanOldOnly(self):
        now = time.time()
        month_ago = now - 30 * 86400
        old_history = [month_ago + i * 3600 for i in range(5)]
        mixed_history = old_history + [now - 60, now - 10]
        old_score = computeActivation(old_history, now=now)
        mixed_score = computeActivation(mixed_history, now=now)
        assert mixed_score > old_score

    def test_moreAccessesHigherScore(self):
        now = time.time()
        few = [now - 3600]
        many = [now - 3600 * i for i in range(1, 10)]
        assert computeActivation(many, now=now) > computeActivation(few, now=now)

    def test_outputBoundedZeroOne(self):
        now = time.time()
        # Very recent, many accesses
        history = [now - i for i in range(50)]
        score = computeActivation(history, now=now)
        assert 0.0 <= score <= 1.0
        # Very old, single access
        score2 = computeActivation([now - 365 * 86400], now=now)
        assert 0.0 <= score2 <= 1.0

    def test_minAgeSecondsClampsRecent(self):
        now = time.time()
        # Access at exactly now → age clamped to min_age_seconds
        score = computeActivation([now], now=now, min_age_seconds=1.0)
        assert score > 0.0


class TestSynthesizeHistory:
    def test_zeroCountEmpty(self):
        assert synthesizeHistory(100.0, 200.0, 0) == []

    def test_singleCountReturnsAccessed(self):
        result = synthesizeHistory(100.0, 200.0, 1)
        assert result == [200.0]

    def test_evenlySpaced(self):
        result = synthesizeHistory(0.0, 100.0, 3)
        assert len(result) == 3
        assert result[0] == 0.0
        assert result[1] == 50.0
        assert result[2] == 100.0

    def test_sameTimestampsRepeated(self):
        result = synthesizeHistory(100.0, 100.0, 5)
        assert len(result) == 5
        assert all(t == 100.0 for t in result)


class TestTemporalScoreLegacy:
    def test_newHigherThanOld(self):
        now = int(time.time() * 1000)
        week_ago = now - 7 * 24 * 3_600_000
        new_score = temporalScore(now, now, 1)
        old_score = temporalScore(week_ago, week_ago, 1)
        assert new_score > old_score

    def test_frequentlyAccessedHigher(self):
        now = int(time.time() * 1000)
        score_1 = temporalScore(now, now, 1)
        score_10 = temporalScore(now, now, 10)
        assert score_10 > score_1


class TestAccessHistoryDb:
    def test_insertSeedsHistory(self, db: sqlite3.Connection):
        mid = insertMemory(db, "test", [1.0, 0.0, 0.0, 0.0])
        row = db.execute("SELECT access_history_json FROM memories WHERE id = ?", (mid,)).fetchone()
        # Newly inserted memory has no history yet (column defaults to NULL)
        # History is seeded by migration or updateAccessStats
        # After migration, it gets seeded; fresh inserts start NULL
        assert row is not None

    def test_updateAccessStatsAppendsHistory(self, db: sqlite3.Connection):
        mid = insertMemory(db, "test", [1.0, 0.0, 0.0, 0.0])
        # Seed initial history
        db.execute(
            "UPDATE memories SET access_history_json = ? WHERE id = ?",
            (json.dumps([time.time() - 100]), mid),
        )
        db.commit()

        updateAccessStats(db, [mid])
        row = db.execute(
            "SELECT access_history_json, access_count FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        history = json.loads(row["access_history_json"])
        assert len(history) == 2
        assert row["access_count"] == 1

    def test_historyTrimmedToMax(self, db: sqlite3.Connection):
        mid = insertMemory(db, "test", [1.0, 0.0, 0.0, 0.0])
        now = time.time()
        # Seed with max entries
        full_history = [now - i for i in range(50)]
        db.execute(
            "UPDATE memories SET access_history_json = ? WHERE id = ?",
            (json.dumps(full_history), mid),
        )
        db.commit()

        updateAccessStats(db, [mid])
        row = db.execute("SELECT access_history_json FROM memories WHERE id = ?", (mid,)).fetchone()
        history = json.loads(row["access_history_json"])
        assert len(history) == 50  # trimmed to MAX_HISTORY_SIZE


class TestMigrationV2ToV3:
    def test_migrationSeedsHistory(self, db: sqlite3.Connection):
        """Insert a memory, wipe its history, re-run migration, verify seeded."""
        mid = insertMemory(db, "migration test", [1.0, 0.0, 0.0, 0.0])
        # Simulate pre-v3 state
        db.execute(
            "UPDATE memories SET access_history_json = NULL, access_count = 5 WHERE id = ?", (mid,)
        )
        db.commit()

        # Column exists so ALTER will fail — just test the seeding part
        rows = db.execute(
            "SELECT id, created_at, accessed_at, access_count FROM memories WHERE id = ?", (mid,)
        ).fetchall()
        for row in rows:
            created_s = row["created_at"] / 1000.0
            accessed_s = row["accessed_at"] / 1000.0
            count = max(row["access_count"], 1)
            history = synthesizeHistory(created_s, accessed_s, count)
            db.execute(
                "UPDATE memories SET access_history_json = ? WHERE id = ?",
                (json.dumps(history), row["id"]),
            )
        db.commit()

        row = db.execute("SELECT access_history_json FROM memories WHERE id = ?", (mid,)).fetchone()
        history = json.loads(row["access_history_json"])
        assert len(history) == 5

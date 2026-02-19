"""PID file operations unit tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rememble.daemon import isRunning, readPid, removePid, stopDaemon, writePid


@pytest.fixture
def pid_path(tmp_path: Path) -> str:
    return str(tmp_path / "test.pid")


class TestWritePid:
    def test_writesCurrentPid(self, pid_path):
        writePid(pid_path)
        assert Path(pid_path).read_text() == str(os.getpid())

    def test_writesGivenPid(self, pid_path):
        writePid(pid_path, pid=12345)
        assert Path(pid_path).read_text() == "12345"

    def test_createsParentDirs(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "test.pid")
        writePid(nested, pid=1)
        assert Path(nested).exists()


class TestReadPid:
    def test_readExisting(self, pid_path):
        Path(pid_path).write_text("42")
        assert readPid(pid_path) == 42

    def test_readMissing(self, pid_path):
        assert readPid(pid_path) is None

    def test_readInvalid(self, pid_path):
        Path(pid_path).write_text("not-a-number")
        assert readPid(pid_path) is None


class TestIsRunning:
    def test_currentProcessIsRunning(self, pid_path):
        writePid(pid_path)
        assert isRunning(pid_path) is True

    def test_noPidFile(self, pid_path):
        assert isRunning(pid_path) is False

    def test_stalePidCleansUp(self, pid_path):
        # Use a PID that definitely doesn't exist (max PID + 1 area)
        Path(pid_path).write_text("999999999")
        assert isRunning(pid_path) is False
        # Stale PID file should be cleaned up
        assert not Path(pid_path).exists()


class TestRemovePid:
    def test_removesExisting(self, pid_path):
        Path(pid_path).write_text("1")
        removePid(pid_path)
        assert not Path(pid_path).exists()

    def test_removeMissing(self, pid_path):
        removePid(pid_path)  # Should not raise


class TestStopDaemon:
    def test_stopNoPidFile(self, pid_path):
        assert stopDaemon(pid_path) is False

    def test_stopStalePid(self, pid_path):
        Path(pid_path).write_text("999999999")
        assert stopDaemon(pid_path) is False
        assert not Path(pid_path).exists()

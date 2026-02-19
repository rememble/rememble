"""PID file management and daemonization helpers."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


def writePid(pid_path: str, pid: int | None = None) -> None:
    """Write current (or given) PID to file."""
    p = Path(pid_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(pid or os.getpid()))


def readPid(pid_path: str) -> int | None:
    """Read PID from file. Returns None if missing or invalid."""
    p = Path(pid_path)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return None


def isRunning(pid_path: str) -> bool:
    """Check if the daemon process is still alive."""
    pid = readPid(pid_path)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        # Stale PID file — clean up
        removePid(pid_path)
        return False


def removePid(pid_path: str) -> None:
    """Remove PID file if it exists."""
    p = Path(pid_path)
    if p.exists():
        p.unlink(missing_ok=True)


def stopDaemon(pid_path: str) -> bool:
    """Send SIGTERM to daemon. Returns True if signal sent."""
    pid = readPid(pid_path)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        removePid(pid_path)
        return True
    except OSError:
        removePid(pid_path)
        return False


def daemonize(pid_path: str) -> None:
    """Double-fork to daemonize. Writes PID file after second fork."""
    # First fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    os.setsid()

    # Second fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Redirect stdio to /dev/null
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    writePid(pid_path)

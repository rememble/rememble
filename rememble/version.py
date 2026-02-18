"""Package version via importlib.metadata."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rememble")
except PackageNotFoundError:
    __version__ = "unknown"

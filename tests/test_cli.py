"""CLI command tests — non-interactive paths via typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from rememble.server import _cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path):
    """Redirect CONFIG_DIR / CONFIG_PATH to tmp_path for every test."""
    cfg_dir = tmp_path / ".rememble"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.json"
    with (
        patch("rememble.config.CONFIG_DIR", cfg_dir),
        patch("rememble.config.CONFIG_PATH", cfg_path),
        patch("rememble.setup.CONFIG_PATH", cfg_path, create=True),
    ):
        yield


# ── config list ──────────────────────────────────────────────


def test_configList_json():
    result = runner.invoke(_cli, ["config", "list", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "embedding_api_url" in data
    assert "search" in data
    assert "rag" in data


# ── config get ───────────────────────────────────────────────


def test_configGet_json():
    result = runner.invoke(_cli, ["config", "get", "embedding_api_url", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["key"] == "embedding_api_url"
    assert "value" in data
    assert "type" in data


def test_configGet_missing_json():
    result = runner.invoke(_cli, ["config", "get", "nonexistent.key", "--format", "json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["ok"] is False
    assert "error" in data


# ── config set ───────────────────────────────────────────────


def test_configSet_json(tmp_path: Path):
    result = runner.invoke(
        _cli, ["config", "set", "embedding_dimensions", "512", "--format", "json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["ok"] is True
    assert data["key"] == "embedding_dimensions"
    assert data["value"] == 512


def test_configSet_invalid_json():
    result = runner.invoke(
        _cli, ["config", "set", "embedding_dimensions", "not_a_number", "--format", "json"]
    )
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["ok"] is False


# ── setup ────────────────────────────────────────────────────


def test_setup_nonInteractive_json():
    result = runner.invoke(
        _cli,
        [
            "setup",
            "--provider",
            "ollama",
            "--model",
            "nomic-embed-text",
            "--agents",
            "",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["ok"] is True
    assert "embedding" in data


def test_setup_badProvider():
    result = runner.invoke(
        _cli,
        ["setup", "--provider", "bogus", "--agents", "", "--format", "json"],
    )
    # bad provider prints error JSON but exits 0 (no typer.Exit(1) raised)
    data = json.loads(result.output)
    assert data["ok"] is False
    assert "bogus" in data["error"]


# ── uninstall ────────────────────────────────────────────────


def test_uninstall_yes_json():
    result = runner.invoke(_cli, ["uninstall", "--yes", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["ok"] is True
    assert "agents" in data
    assert "db_deleted" in data

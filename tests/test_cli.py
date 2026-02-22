"""CLI command tests — non-interactive paths via typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from rememble.server.cli import _cli

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
    assert "embedding" in data
    assert "api_url" in data["embedding"]
    assert "search" in data
    assert "rag" in data


# ── config get ───────────────────────────────────────────────


def test_configGet_json():
    result = runner.invoke(_cli, ["config", "get", "embedding.api_url", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["key"] == "embedding.api_url"
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
        _cli, ["config", "set", "embedding.dimensions", "512", "--format", "json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["ok"] is True
    assert data["key"] == "embedding.dimensions"
    assert data["value"] == 512


def test_configSet_invalid_json():
    result = runner.invoke(
        _cli, ["config", "set", "embedding.dimensions", "not_a_number", "--format", "json"]
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


# ── status ──────────────────────────────────────────────────


def test_status_notRunning():
    result = runner.invoke(_cli, ["status", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["running"] is False


# ── stop ────────────────────────────────────────────────────


def test_stop_notRunning():
    result = runner.invoke(_cli, ["stop", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["ok"] is False


# ── client commands (mocked RemembleClient) ─────────────────


def _mockClient(**methods):
    """Create a mock RemembleClient with specified method return values."""
    mock = MagicMock()
    for name, retval in methods.items():
        getattr(mock, name).return_value = retval
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_cli_remember_json():
    mock = _mockClient(remember={"stored": True, "memory_ids": [1], "chunks": 1, "tokens": 3})
    with patch("rememble.server.cli._getClient", return_value=mock):
        result = runner.invoke(_cli, ["remember", "hello world", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["stored"] is True


def test_cli_recall_json():
    mock = _mockClient(recall={"query": "q", "total_tokens": 10, "items": []})
    with patch("rememble.server.cli._getClient", return_value=mock):
        result = runner.invoke(_cli, ["recall", "query", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["query"] == "q"


def test_cli_forget_json():
    mock = _mockClient(forget={"forgotten": True, "memory_id": 1})
    with patch("rememble.server.cli._getClient", return_value=mock):
        result = runner.invoke(_cli, ["forget", "1", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["forgotten"] is True


def test_cli_list_json():
    mock = _mockClient(listMemories={"memories": [], "count": 0, "offset": 0})
    with patch("rememble.server.cli._getClient", return_value=mock):
        result = runner.invoke(_cli, ["list", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["count"] == 0


def test_cli_stats_json():
    mock = _mockClient(stats={"total_memories": 5})
    with patch("rememble.server.cli._getClient", return_value=mock):
        result = runner.invoke(_cli, ["stats", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total_memories"] == 5


def test_cli_entity_create_json():
    mock = _mockClient(createEntities={"created": [{"name": "A", "entity_id": 1}]})
    with patch("rememble.server.cli._getClient", return_value=mock):
        result = runner.invoke(
            _cli, ["entity", "create", "--name", "A", "--type", "test", "--format", "json"]
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["created"]) == 1


def test_cli_graph_search_json():
    mock = _mockClient(searchGraph={"entities": [{"name": "A", "type": "test"}]})
    with patch("rememble.server.cli._getClient", return_value=mock):
        result = runner.invoke(_cli, ["graph", "search", "A", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["entities"]) == 1


def test_cli_remember_connectionError():
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.remember.side_effect = Exception("Connection refused")
    with patch("rememble.server.cli._getClient", return_value=mock):
        result = runner.invoke(_cli, ["remember", "test", "--format", "json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["ok"] is False

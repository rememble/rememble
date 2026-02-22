"""Tests for hook handlers and setup hook config management."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from rememble.hooks import (
    _deriveProject,
    _extractText,
    _extractTranscript,
    _formatRecallContext,
    hookPromptSubmit,
    hookSessionEnd,
    hookSessionStart,
)
from rememble.server.cli import _cli
from rememble.setup import (
    _buildHookConfig,
    _isRemembleHookEntry,
    _mergeHooks,
    _removeHooks,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path):
    """Redirect config to tmp_path for every test."""
    cfg_dir = tmp_path / ".rememble"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.json"
    with (
        patch("rememble.config.CONFIG_DIR", cfg_dir),
        patch("rememble.config.CONFIG_PATH", cfg_path),
    ):
        yield


# ── _deriveProject ───────────────────────────────────────────


def test_deriveProject_gitRemote():
    mock_result = MagicMock(returncode=0, stdout="git@github.com:user/my-project.git\n")
    with patch("rememble.hooks.subprocess.run", return_value=mock_result):
        assert _deriveProject("/some/path") == "my-project"


def test_deriveProject_httpsRemote():
    mock_result = MagicMock(returncode=0, stdout="https://github.com/user/repo\n")
    with patch("rememble.hooks.subprocess.run", return_value=mock_result):
        assert _deriveProject("/some/path") == "repo"


def test_deriveProject_fallbackBasename():
    mock_result = MagicMock(returncode=1, stdout="", stderr="")
    with patch("rememble.hooks.subprocess.run", return_value=mock_result):
        assert _deriveProject("/home/user/my-project") == "my-project"


def test_deriveProject_empty():
    assert _deriveProject("") is None


def test_deriveProject_gitTimeout():
    with patch("rememble.hooks.subprocess.run", side_effect=TimeoutError):
        assert _deriveProject("/home/user/proj") == "proj"


# ── _extractText ─────────────────────────────────────────────


def test_extractText_string():
    assert _extractText("hello world") == "hello world"


def test_extractText_contentBlocks():
    blocks = [
        {"type": "text", "text": "first"},
        {"type": "tool_use", "id": "123", "name": "Read"},
        {"type": "text", "text": "second"},
    ]
    assert _extractText(blocks) == "first\nsecond"


def test_extractText_toolResultOnly():
    blocks = [{"type": "tool_result", "tool_use_id": "123", "content": "file data"}]
    assert _extractText(blocks) == ""


def test_extractText_none():
    assert _extractText(None) == ""


# ── _extractTranscript ───────────────────────────────────────


def test_extractTranscript(tmp_path: Path):
    transcript = tmp_path / "transcript.jsonl"
    lines = [
        json.dumps(
            {
                "type": "human",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Fix the bug"}],
                },
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll fix it"},
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                    ],
                },
            }
        ),
        # Tool result — should be skipped (no text blocks)
        json.dumps(
            {
                "type": "human",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "file data"}
                    ],
                },
            }
        ),
        json.dumps(
            {
                "type": "human",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Looks good"}],
                },
            }
        ),
    ]
    transcript.write_text("\n".join(lines))
    result = _extractTranscript(str(transcript))
    assert "User: Fix the bug" in result
    assert "Assistant: I'll fix it" in result
    assert "User: Looks good" in result
    assert "file data" not in result


def test_extractTranscript_missingFile():
    assert _extractTranscript("/nonexistent/path.jsonl") == ""


# ── _formatRecallContext ─────────────────────────────────────


def test_formatRecallContext_withItems():
    result = {"items": [{"kind": "memory", "text": "Use ruff for linting"}]}
    ctx = _formatRecallContext(result)
    assert "## Recalled Memories" in ctx
    assert "Use ruff for linting" in ctx


def test_formatRecallContext_empty():
    assert _formatRecallContext({"items": []}) == ""
    assert _formatRecallContext({}) == ""


# ── hookSessionStart ─────────────────────────────────────────


def test_hookSessionStart_withContext():
    recall_result = {"items": [{"kind": "memory", "text": "prior work context"}]}
    with (
        patch("rememble.hooks._callApi", return_value=recall_result),
        patch("rememble.hooks._deriveProject", return_value="myproject"),
    ):
        result = hookSessionStart({"cwd": "/tmp/myproject", "source": "startup"})
    assert "additionalContext" in result
    assert "prior work context" in result["additionalContext"]


def test_hookSessionStart_apiDown():
    with (
        patch("rememble.hooks._callApi", return_value=None),
        patch("rememble.hooks._deriveProject", return_value="proj"),
    ):
        result = hookSessionStart({"cwd": "/tmp/proj", "source": "startup"})
    assert result == {}


def test_hookSessionStart_emptyResult():
    with (
        patch("rememble.hooks._callApi", return_value={"items": []}),
        patch("rememble.hooks._deriveProject", return_value="proj"),
    ):
        result = hookSessionStart({"cwd": "/tmp/proj"})
    assert result == {}


# ── hookPromptSubmit ─────────────────────────────────────────


def test_hookPromptSubmit_trivialInput():
    assert hookPromptSubmit({"prompt": "yes"}) == {}
    assert hookPromptSubmit({"prompt": "ok"}) == {}
    assert hookPromptSubmit({"prompt": "y"}) == {}
    assert hookPromptSubmit({"prompt": "lgtm"}) == {}


def test_hookPromptSubmit_shortInput():
    assert hookPromptSubmit({"prompt": "fix it"}) == {}


def test_hookPromptSubmit_withContext():
    recall_result = {"items": [{"kind": "memory", "text": "auth uses JWT tokens"}]}
    with patch("rememble.hooks._callApi", return_value=recall_result):
        result = hookPromptSubmit({"prompt": "refactor the authentication module"})
    assert "additionalContext" in result
    assert "JWT tokens" in result["additionalContext"]


def test_hookPromptSubmit_apiDown():
    with patch("rememble.hooks._callApi", return_value=None):
        result = hookPromptSubmit({"prompt": "refactor the authentication module"})
    assert result == {}


# ── hookSessionEnd ───────────────────────────────────────────


def test_hookSessionEnd_missingTranscript():
    result = hookSessionEnd({"transcript_path": "/nonexistent", "session_id": "s1", "cwd": "/tmp"})
    assert result == {}


def test_hookSessionEnd_withLlmSummary(tmp_path: Path):
    transcript = tmp_path / "transcript.jsonl"
    # Create a transcript long enough (>100 chars after extraction)
    entries = []
    for i in range(10):
        entries.append(
            json.dumps(
                {
                    "type": "human",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Question {i} about the architecture"}
                        ],
                    },
                }
            )
        )
        entries.append(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"Detailed answer {i} about patterns"}
                        ],
                    },
                }
            )
        )
    transcript.write_text("\n".join(entries))

    llm_result = {
        "global": [{"content": "User prefers camelCase", "tags": "preference"}],
        "project": [{"content": "Uses SQLite for storage", "tags": "architecture"}],
    }
    with (
        patch("rememble.hooks._llmSummarize", return_value=llm_result),
        patch("rememble.hooks._callApi") as mock_api,
        patch("rememble.hooks._deriveProject", return_value="testproj"),
    ):
        result = hookSessionEnd(
            {
                "transcript_path": str(transcript),
                "session_id": "sess1",
                "cwd": str(tmp_path),
            }
        )

    assert result == {}
    # Should have stored 2 memories (1 global + 1 project)
    assert mock_api.call_count == 2
    # Check global memory (no project param)
    global_call = mock_api.call_args_list[0]
    assert global_call[0][0] == "/remember"
    assert global_call[0][1]["content"] == "User prefers camelCase"
    assert "project" not in global_call[0][1]
    # Check project memory
    project_call = mock_api.call_args_list[1]
    assert project_call[0][1]["project"] == "testproj"


def test_hookSessionEnd_llmFails_fallbackToChunks(tmp_path: Path):
    transcript = tmp_path / "transcript.jsonl"
    entries = []
    for i in range(10):
        entries.append(
            json.dumps(
                {
                    "type": "human",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Message {i} with some content"}],
                    },
                }
            )
        )
    transcript.write_text("\n".join(entries))

    with (
        patch("rememble.hooks._llmSummarize", return_value=None),
        patch("rememble.hooks._callApi") as mock_api,
        patch("rememble.hooks._deriveProject", return_value="proj"),
    ):
        hookSessionEnd(
            {
                "transcript_path": str(transcript),
                "session_id": "sess2",
                "cwd": str(tmp_path),
            }
        )

    # Should have stored raw chunks
    assert mock_api.call_count >= 1
    first_call = mock_api.call_args_list[0]
    assert first_call[0][1]["tags"] == "transcript,raw"


# ── Hook config helpers (setup.py) ───────────────────────────


def test_isRemembleHookEntry_true():
    entry = {"hooks": [{"type": "command", "command": "/usr/bin/rememble hook session-start"}]}
    assert _isRemembleHookEntry(entry) is True


def test_isRemembleHookEntry_false():
    entry = {"hooks": [{"type": "command", "command": "/usr/bin/other-tool hook start"}]}
    assert _isRemembleHookEntry(entry) is False


def test_isRemembleHookEntry_noHooks():
    assert _isRemembleHookEntry({}) is False


def test_buildHookConfig():
    config = _buildHookConfig("/usr/bin/rememble")
    assert "SessionStart" in config
    assert "UserPromptSubmit" in config
    assert "SessionEnd" in config
    assert config["SessionStart"][0]["matcher"] == "startup|resume"
    assert (
        "/usr/bin/rememble hook session-start" in config["SessionStart"][0]["hooks"][0]["command"]
    )


def test_mergeHooks_newFile(tmp_path: Path):
    settings = tmp_path / "settings.json"
    config = _buildHookConfig("/usr/bin/rememble")
    _mergeHooks(settings, config)

    data = json.loads(settings.read_text())
    assert "hooks" in data
    assert len(data["hooks"]["SessionStart"]) == 1
    assert "rememble" in data["hooks"]["SessionStart"][0]["hooks"][0]["command"]


def test_mergeHooks_preservesExisting(tmp_path: Path):
    settings = tmp_path / "settings.json"
    # Write existing hook from another tool
    existing = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "/usr/bin/other-tool start"}]}
            ]
        },
        "other_setting": True,
    }
    settings.write_text(json.dumps(existing))

    config = _buildHookConfig("/usr/bin/rememble")
    _mergeHooks(settings, config)

    data = json.loads(settings.read_text())
    assert data["other_setting"] is True
    # Should have both hooks
    assert len(data["hooks"]["SessionStart"]) == 2
    commands = [e["hooks"][0]["command"] for e in data["hooks"]["SessionStart"]]
    assert "/usr/bin/other-tool start" in commands
    assert "/usr/bin/rememble hook session-start" in commands


def test_mergeHooks_idempotent(tmp_path: Path):
    settings = tmp_path / "settings.json"
    config = _buildHookConfig("/usr/bin/rememble")
    _mergeHooks(settings, config)
    _mergeHooks(settings, config)  # merge again

    data = json.loads(settings.read_text())
    # Should not duplicate rememble entries
    assert len(data["hooks"]["SessionStart"]) == 1


def test_removeHooks_removes(tmp_path: Path):
    settings = tmp_path / "settings.json"
    config = _buildHookConfig("/usr/bin/rememble")
    _mergeHooks(settings, config)

    changed = _removeHooks(settings)
    assert changed is True

    data = json.loads(settings.read_text())
    assert "hooks" not in data


def test_removeHooks_preservesOtherHooks(tmp_path: Path):
    settings = tmp_path / "settings.json"
    existing = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "/usr/bin/other-tool start"}]},
                {"hooks": [{"type": "command", "command": "/usr/bin/rememble hook session-start"}]},
            ]
        }
    }
    settings.write_text(json.dumps(existing))

    changed = _removeHooks(settings)
    assert changed is True

    data = json.loads(settings.read_text())
    assert len(data["hooks"]["SessionStart"]) == 1
    assert "other-tool" in data["hooks"]["SessionStart"][0]["hooks"][0]["command"]


def test_removeHooks_noFile(tmp_path: Path):
    assert _removeHooks(tmp_path / "nonexistent.json") is False


def test_removeHooks_noRemembleHooks(tmp_path: Path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"hooks": {"SessionStart": [{"hooks": [{"command": "other"}]}]}})
    )
    assert _removeHooks(settings) is False


# ── CLI hook subcommand integration ──────────────────────────


def test_cli_hookSessionStart():
    recall_result = {"items": [{"kind": "memory", "text": "remembered context"}]}
    with (
        patch("rememble.hooks._callApi", return_value=recall_result),
        patch("rememble.hooks._deriveProject", return_value="proj"),
    ):
        result = runner.invoke(
            _cli,
            ["hook", "session-start"],
            input=json.dumps({"session_id": "s1", "cwd": "/tmp/proj", "source": "startup"}),
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "additionalContext" in data


def test_cli_hookPromptSubmit_trivial():
    result = runner.invoke(
        _cli,
        ["hook", "prompt-submit"],
        input=json.dumps({"prompt": "yes"}),
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {}


def test_cli_hookPromptSubmit_withRecall():
    recall_result = {"items": [{"kind": "memory", "text": "relevant context"}]}
    with patch("rememble.hooks._callApi", return_value=recall_result):
        result = runner.invoke(
            _cli,
            ["hook", "prompt-submit"],
            input=json.dumps({"prompt": "refactor the database module"}),
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "additionalContext" in data


def test_cli_hookSessionEnd_noTranscript():
    result = runner.invoke(
        _cli,
        ["hook", "session-end"],
        input=json.dumps({"transcript_path": "/nonexistent", "session_id": "s1", "cwd": "/tmp"}),
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {}


def test_cli_hook_invalidJson():
    result = runner.invoke(
        _cli,
        ["hook", "session-start"],
        input="not valid json",
    )
    # Should exit cleanly (no crash)
    assert result.exit_code == 0

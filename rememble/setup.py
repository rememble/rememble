"""rememble setup — detect AI agents and configure MCP + instructions."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import questionary
from rich.console import Console

console = Console()

_MARKER = "<!-- rememble -->"

_INSTRUCTIONS = f"""{_MARKER}
## Rememble memory

You have persistent memory via MCP tools. Use proactively.

- `remember(content, source=..., tags=...)` — store decisions, preferences, patterns, context
- `recall(query)` — search before starting work; returns ranked context
- `create_entities` / `add_observations` — structured facts about people, projects, codebases
- `search_graph(query)` — search the knowledge graph

Tag with project/topic (`source="project:myapp"`) for easy filtering.
"""

_MCP_ENTRY: dict = {"command": "rememble"}

# (label, api_url, needs_key, [(model_name, dimensions)])
_PROVIDER_SLUGS: dict[str, int] = {
    "ollama": 0,
    "openai": 1,
    "openrouter": 2,
    "cohere": 3,
}

_AGENT_SLUGS: dict[str, int] = {
    "claude-code": 0,
    "claude-desktop": 1,
    "opencode": 2,
    "codex": 3,
    "cursor": 4,
    "windsurf": 5,
}

_PROVIDERS: list[tuple[str, str, bool, list[tuple[str, int]]]] = [
    (
        "Ollama",
        "http://localhost:11434/v1",
        False,
        [
            ("nomic-embed-text", 768),
            ("mxbai-embed-large", 1024),
            ("all-minilm", 384),
            ("bge-large", 1024),
            ("snowflake-arctic-embed", 1024),
        ],
    ),
    (
        "OpenAI",
        "https://api.openai.com/v1",
        True,
        [
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("text-embedding-ada-002", 1536),
        ],
    ),
    (
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        True,
        [
            ("openai/text-embedding-3-small", 1536),
            ("openai/text-embedding-3-large", 3072),
            ("cohere/embed-english-v3.0", 1024),
            ("cohere/embed-multilingual-v3.0", 1024),
        ],
    ),
    (
        "Cohere",
        "https://api.cohere.ai/compatibility/v1",
        True,
        [
            ("embed-english-v3.0", 1024),
            ("embed-multilingual-v3.0", 1024),
            ("embed-english-light-v3.0", 384),
            ("embed-multilingual-light-v3.0", 384),
        ],
    ),
]


# ============================================================
# File helpers
# ============================================================


def _mergeJson(path: Path, updates: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    for k, v in updates.items():
        if k in existing and isinstance(existing[k], dict) and isinstance(v, dict):
            existing[k].update(v)
        else:
            existing[k] = v
    path.write_text(json.dumps(existing, indent=2) + "\n")


def _appendToml(path: Path, section: str, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        content = path.read_text()
        if section in content:
            return
        path.write_text(content.rstrip() + "\n\n" + block + "\n")
    else:
        path.write_text(block + "\n")


def _appendInstructions(path: Path) -> bool:
    """Append instructions block if marker absent. Returns True if appended."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _MARKER in path.read_text():
            return False
        existing = path.read_text()
        path.write_text(existing.rstrip() + "\n\n" + _INSTRUCTIONS + "\n")
    else:
        path.write_text(_INSTRUCTIONS + "\n")
    return True


def _stripInstructions(path: Path) -> bool:
    """Remove from <!-- rememble --> to EOF. Returns True if removed."""
    if not path.exists():
        return False
    text = path.read_text()
    idx = text.find(_MARKER)
    if idx == -1:
        return False
    path.write_text(text[:idx].rstrip() + "\n")
    return True


def _removeJsonKey(path: Path, *keys: str) -> bool:
    """Navigate nested keys and delete leaf. Returns True if key existed."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    node = data
    for k in keys[:-1]:
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    if not isinstance(node, dict) or keys[-1] not in node:
        return False
    del node[keys[-1]]
    path.write_text(json.dumps(data, indent=2) + "\n")
    return True


def _removeTomlSection(path: Path, section: str) -> bool:
    """Remove section header + all following lines until next [section]."""
    if not path.exists():
        return False
    lines = path.read_text().splitlines(keepends=True)
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == section:
            start = i
        elif start is not None and stripped.startswith("[") and stripped != section:
            end = i
            break
    if start is None:
        return False
    while start > 0 and lines[start - 1].strip() == "":
        start -= 1
    path.write_text("".join(lines[:start] + lines[end:]))
    return True


# ============================================================
# Config wizard
# ============================================================


def _configWizard(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    format: str = "human",
) -> dict[str, Any] | None:
    """Config prompts (interactive or non-interactive). Returns changes dict or None on cancel."""
    from rememble.config import CONFIG_PATH, RemembleConfig, loadConfig

    config = loadConfig()

    # Resolve provider index
    if provider is not None:
        slug = provider.lower()
        if slug not in _PROVIDER_SLUGS:
            valid = ", ".join(_PROVIDER_SLUGS)
            if format == "json":
                print(
                    json.dumps(
                        {"ok": False, "error": f"Unknown provider {provider!r}; valid: {valid}"}
                    )
                )
            else:
                console.print(f"[red]Unknown provider:[/red] {provider!r}. Valid: {valid}")
            return None
        provider_idx: int | None = _PROVIDER_SLUGS[slug]
    else:
        default_provider = next(
            (i for i, (_, url, _, _) in enumerate(_PROVIDERS) if url == config.embedding_api_url), 0
        )
        provider_idx = questionary.select(
            "Select embedding provider:",
            choices=[
                questionary.Choice(title=f"{label}  ({url})", value=i)
                for i, (label, url, _, _) in enumerate(_PROVIDERS)
            ],
            default=default_provider,  # type: ignore[arg-type]  # questionary stubs too narrow
        ).ask()
        if provider_idx is None:
            return None  # ctrl-c

    label, url, needs_key, models = _PROVIDERS[provider_idx]

    # API key
    resolved_key: str | None = None
    if needs_key:
        if api_key is not None:
            resolved_key = api_key if api_key.strip() else config.embedding_api_key
        else:
            current_key = config.embedding_api_key
            hint = " (press enter to keep existing)" if current_key else ""
            new_key = questionary.password(f"API key{hint}:").ask()
            if new_key is None:
                return None  # ctrl-c
            resolved_key = new_key.strip() if new_key.strip() else current_key

    # Resolve model
    if model is not None:
        match = next(((m, d) for m, d in models if m == model), None)
        if match is None:
            valid_models = ", ".join(m for m, _ in models)
            if format == "json":
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "error": f"Unknown model {model!r} for {label}; valid: {valid_models}",
                        }
                    )
                )
            else:
                console.print(
                    f"[red]Unknown model:[/red] {model!r}. Valid for {label}: {valid_models}"
                )
            return None
        selected_model, selected_dims = match
    else:
        default_model = next(
            (i for i, (m, _) in enumerate(models) if m == config.embedding_api_model), 0
        )
        model_result: tuple[str, int] | None = questionary.select(
            "Select embedding model:",
            choices=[
                questionary.Choice(title=f"{m}  ({dims} dims)", value=(m, dims))
                for m, dims in models
            ],
            default=models[default_model],  # type: ignore[arg-type]
        ).ask()
        if model_result is None:
            return None  # ctrl-c
        selected_model, selected_dims = model_result

    updates: dict[str, Any] = {
        "embedding_api_url": url,
        "embedding_api_key": resolved_key,
        "embedding_api_model": selected_model,
        "embedding_dimensions": selected_dims,
    }

    changed = {k: v for k, v in updates.items() if getattr(config, k) != v}
    if not changed:
        if format == "human":
            console.print("No changes to embedding config.")
        return {}

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw: dict = {}
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            raw = {}
    raw.update(changed)
    RemembleConfig(**raw)  # validate before writing
    CONFIG_PATH.write_text(json.dumps(raw, indent=2) + "\n")
    if format == "human":
        console.print(f"Config saved to {CONFIG_PATH}")
    return {"provider": label, "model": selected_model, "dimensions": selected_dims}


# ============================================================
# Agent detection + setup functions
# ============================================================


def _isDetected(agent_name: str) -> bool:
    checks: dict[str, Any] = {
        "Claude Code": lambda: bool(shutil.which("claude")),
        "Claude Desktop": lambda: (
            Path.home() / "Library" / "Application Support" / "Claude"
        ).exists(),
        "OpenCode": lambda: bool(shutil.which("opencode")),
        "Codex CLI": lambda: bool(shutil.which("codex")),
        "Cursor": lambda: (Path.home() / ".config" / "cursor").exists(),
        "Windsurf": lambda: (Path.home() / ".codeium" / "windsurf").exists(),
    }
    fn = checks.get(agent_name)
    return fn() if fn else False


def _setupClaudeCode() -> str:
    if not shutil.which("claude"):
        return "skip"
    try:
        result = subprocess.run(
            ["claude", "mcp", "add", "--scope", "user", "rememble", "rememble"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"error: claude mcp add failed — {result.stderr.strip()}"
    except Exception as e:
        return f"error: {e}"

    instructions_path = Path.home() / ".claude" / "CLAUDE.md"
    added = _appendInstructions(instructions_path)
    suffix = f", instructions added to {instructions_path}" if added else ""
    return f"ok:MCP configured{suffix}"


def _setupClaudeDesktop() -> str:
    app_dir = Path.home() / "Library" / "Application Support" / "Claude"
    if not app_dir.exists():
        return "skip"
    cfg = app_dir / "claude_desktop_config.json"
    _mergeJson(cfg, {"mcpServers": {"rememble": _MCP_ENTRY}})
    return "ok:MCP configured"


def _setupOpenCode() -> str:
    if not shutil.which("opencode"):
        return "skip"
    cfg = Path.home() / ".config" / "opencode" / "opencode.json"
    _mergeJson(cfg, {"mcp": {"rememble": {"type": "local", "command": ["rememble"]}}})
    instructions_path = Path.home() / ".config" / "opencode" / "AGENTS.md"
    added = _appendInstructions(instructions_path)
    suffix = f", instructions added to {instructions_path}" if added else ""
    return f"ok:MCP configured{suffix}"


def _setupCodex() -> str:
    if not shutil.which("codex"):
        return "skip"
    cfg = Path.home() / ".codex" / "config.toml"
    block = '[mcp_servers.rememble]\ncommand = "rememble"\nargs = []'
    _appendToml(cfg, "[mcp_servers.rememble]", block)
    instructions_path = Path.home() / ".codex" / "AGENTS.md"
    added = _appendInstructions(instructions_path)
    suffix = f", instructions added to {instructions_path}" if added else ""
    return f"ok:MCP configured{suffix}"


def _setupCursor() -> str:
    cursor_dir = Path.home() / ".config" / "cursor"
    if not cursor_dir.exists():
        return "skip"
    cfg = cursor_dir / "mcp.json"
    _mergeJson(cfg, {"mcpServers": {"rememble": _MCP_ENTRY}})
    return "ok:MCP configured"


def _setupWindsurf() -> str:
    ws_dir = Path.home() / ".codeium" / "windsurf"
    if not ws_dir.exists():
        return "skip"
    cfg = ws_dir / "mcp_config.json"
    _mergeJson(cfg, {"mcpServers": {"rememble": _MCP_ENTRY}})
    return "ok:MCP configured"


_AGENTS: list[tuple[str, str, Any]] = [
    ("Claude Code", "~/.claude/CLAUDE.md", _setupClaudeCode),
    ("Claude Desktop", "~/Library/Application Support/Claude/", _setupClaudeDesktop),
    ("OpenCode", "~/.config/opencode/opencode.json", _setupOpenCode),
    ("Codex CLI", "~/.codex/config.toml", _setupCodex),
    ("Cursor", "~/.config/cursor/mcp.json", _setupCursor),
    ("Windsurf", "~/.codeium/windsurf/mcp_config.json", _setupWindsurf),
]


# ============================================================
# Uninstall helpers + per-agent uninstall
# ============================================================


def _uninstallClaudeCode() -> str:
    if not shutil.which("claude"):
        return "skip"
    try:
        result = subprocess.run(
            ["claude", "mcp", "remove", "--scope", "user", "rememble"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"error: claude mcp remove failed — {result.stderr.strip()}"
    except Exception as e:
        return f"error: {e}"

    instructions_path = Path.home() / ".claude" / "CLAUDE.md"
    removed = _stripInstructions(instructions_path)
    suffix = f", instructions removed from {instructions_path}" if removed else ""
    return f"ok:MCP removed{suffix}"


def _uninstallClaudeDesktop() -> str:
    app_dir = Path.home() / "Library" / "Application Support" / "Claude"
    if not app_dir.exists():
        return "skip"
    cfg = app_dir / "claude_desktop_config.json"
    removed = _removeJsonKey(cfg, "mcpServers", "rememble")
    return "ok:MCP removed" if removed else "ok:not configured"


def _uninstallOpenCode() -> str:
    if not shutil.which("opencode"):
        return "skip"
    cfg = Path.home() / ".config" / "opencode" / "opencode.json"
    removed = _removeJsonKey(cfg, "mcp", "rememble")
    instructions_path = Path.home() / ".config" / "opencode" / "AGENTS.md"
    inst_removed = _stripInstructions(instructions_path)
    parts = []
    if removed:
        parts.append("MCP removed")
    if inst_removed:
        parts.append(f"instructions removed from {instructions_path}")
    return "ok:" + ", ".join(parts) if parts else "ok:not configured"


def _uninstallCodex() -> str:
    if not shutil.which("codex"):
        return "skip"
    cfg = Path.home() / ".codex" / "config.toml"
    removed = _removeTomlSection(cfg, "[mcp_servers.rememble]")
    instructions_path = Path.home() / ".codex" / "AGENTS.md"
    inst_removed = _stripInstructions(instructions_path)
    parts = []
    if removed:
        parts.append("MCP removed")
    if inst_removed:
        parts.append(f"instructions removed from {instructions_path}")
    return "ok:" + ", ".join(parts) if parts else "ok:not configured"


def _uninstallCursor() -> str:
    cursor_dir = Path.home() / ".config" / "cursor"
    if not cursor_dir.exists():
        return "skip"
    cfg = cursor_dir / "mcp.json"
    removed = _removeJsonKey(cfg, "mcpServers", "rememble")
    return "ok:MCP removed" if removed else "ok:not configured"


def _uninstallWindsurf() -> str:
    ws_dir = Path.home() / ".codeium" / "windsurf"
    if not ws_dir.exists():
        return "skip"
    cfg = ws_dir / "mcp_config.json"
    removed = _removeJsonKey(cfg, "mcpServers", "rememble")
    return "ok:MCP removed" if removed else "ok:not configured"


_UNINSTALL_AGENTS: list[tuple[str, Any]] = [
    ("Claude Code", _uninstallClaudeCode),
    ("Claude Desktop", _uninstallClaudeDesktop),
    ("OpenCode", _uninstallOpenCode),
    ("Codex CLI", _uninstallCodex),
    ("Cursor", _uninstallCursor),
    ("Windsurf", _uninstallWindsurf),
]


def _printAgentResult(name: str, result: str) -> None:
    if result == "skip":
        console.print(f"  [dim]{name:<16} — not detected[/dim]")
    elif result.startswith("error:"):
        console.print(f"  [yellow]⚠[/yellow]  {name:<16} — {result[6:].strip()}")
    else:
        detail = result[3:] if result.startswith("ok:") else result
        console.print(f"  [green]✓[/green]  {name:<16} — {detail}")


def runUninstall(yes: bool = False, format: str = "human") -> None:
    if format == "human":
        console.rule("[bold]Rememble Uninstall[/bold]")
        console.print()

    agent_results: list[dict[str, str]] = []
    for name, fn in _UNINSTALL_AGENTS:
        outcome = fn()
        agent_results.append({"agent": name, "result": outcome})
        if format == "human":
            _printAgentResult(name, outcome)

    if format == "human":
        console.print()

    if yes:
        confirm = True
    else:
        confirm = questionary.confirm(
            "Delete ~/.rememble/ (database and config)?", default=False
        ).ask()

    db_deleted = False
    if confirm:
        from rememble.config import CONFIG_DIR

        if CONFIG_DIR.exists():
            shutil.rmtree(CONFIG_DIR)
            db_deleted = True
            if format == "human":
                console.print(f"[red]Deleted[/red] {CONFIG_DIR}")
        elif format == "human":
            console.print("[dim]~/.rememble/ not found, nothing to delete.[/dim]")

    if format == "human":
        console.print()
        console.print("[bold green]Done.[/bold green] Rememble has been uninstalled.")
    else:
        print(json.dumps({"ok": True, "agents": agent_results, "db_deleted": db_deleted}))


# ============================================================
# Utility
# ============================================================


def _coerceValue(value: str) -> Any:
    """Auto-coerce string to int/float/bool/str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


# ============================================================
# Main setup flow
# ============================================================


def runSetup(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    agents: str | None = None,
    yes: bool = False,
    format: str = "human",
) -> None:
    if format == "human":
        console.rule("[bold blue]Rememble Setup[/bold blue]")
        console.print()
        console.rule("Step 1/2 — Embedding", align="left")

    wizard_result = _configWizard(provider=provider, api_key=api_key, model=model, format=format)
    if wizard_result is None:
        return  # cancelled or error

    if format == "human":
        console.print()
        console.rule("Step 2/2 — Agents", align="left")

    # Resolve agent indices
    if agents is not None or yes:
        if agents is not None:
            slugs = [s.strip() for s in agents.split(",") if s.strip()]
            selected_indices: list[int] = []
            for slug in slugs:
                idx = _AGENT_SLUGS.get(slug.lower())
                if idx is None:
                    valid = ", ".join(_AGENT_SLUGS)
                    if format == "json":
                        print(
                            json.dumps(
                                {"ok": False, "error": f"Unknown agent {slug!r}; valid: {valid}"}
                            )
                        )
                    else:
                        console.print(f"[red]Unknown agent:[/red] {slug!r}. Valid: {valid}")
                    return
                selected_indices.append(idx)
        else:
            # --yes with no --agents: install all detected
            selected_indices = [i for i, (name, _, _) in enumerate(_AGENTS) if _isDetected(name)]
    else:
        agent_choices = [
            questionary.Choice(
                title=f"{name}  ({'detected' if _isDetected(name) else 'not detected'})",
                value=i,
                checked=_isDetected(name),
            )
            for i, (name, _, _) in enumerate(_AGENTS)
        ]
        result = questionary.checkbox(
            "Select agents to configure:",
            choices=agent_choices,
        ).ask()
        if result is None:
            return  # ctrl-c
        selected_indices = result

    if format == "human":
        console.print()
        console.print("[bold]Applying...[/bold]")
        console.print()

    agents_configured: list[dict[str, str]] = []
    any_applied = False
    for i, (name, _, fn) in enumerate(_AGENTS):
        if i not in selected_indices:
            continue
        any_applied = True
        outcome = fn()
        agents_configured.append({"agent": name, "result": outcome})
        if format == "human":
            _printAgentResult(name, outcome)

    if format == "human":
        if not any_applied:
            console.print("[dim]No agents selected.[/dim]")
        console.print()
        console.print(
            "[bold green]Done.[/bold green] Restart any running agents to pick up the new MCP server."  # noqa: E501
        )
    else:
        print(
            json.dumps(
                {
                    "ok": True,
                    "embedding": wizard_result,
                    "agents_configured": agents_configured,
                }
            )
        )

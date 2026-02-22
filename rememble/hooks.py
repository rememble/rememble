"""Claude Code hook handlers — session-start, prompt-submit, session-end."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx

from rememble.config import loadConfig

_TRIVIAL = re.compile(r"^(y|yes|ok|no|n|k|sure|done|thx|thanks|ty|lgtm)$", re.IGNORECASE)


# ── Helpers ──────────────────────────────────────────────────


def _deriveProject(cwd: str) -> str | None:
    """Derive project name from cwd — git remote or directory basename."""
    if not cwd:
        return None
    try:
        r = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            name = r.stdout.strip().rstrip("/").rsplit("/", 1)[-1]
            return name.removesuffix(".git") or None
    except Exception:
        pass
    return Path(cwd).name or None


def _callApi(endpoint: str, payload: dict[str, Any], timeout: float = 5.0) -> dict | None:
    """POST to rememble REST API. Returns response dict or None on any failure."""
    cfg = loadConfig()
    url = f"http://localhost:{cfg.port}/api{endpoint}"
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _extractText(content: Any) -> str:
    """Extract text from message content (string or content block list)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return ""


def _extractTranscript(path: str) -> str:
    """Parse JSONL transcript into conversation text, skipping tool I/O."""
    lines: list[str] = []
    try:
        with open(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                kind = entry.get("type", "")
                msg = entry.get("message", {})
                content = msg.get("content", "")

                if kind == "human":
                    # _extractText skips tool_result blocks (only grabs type=text)
                    text = _extractText(content)
                    if text:
                        lines.append(f"User: {text}")
                elif kind == "assistant":
                    text = _extractText(content)
                    if text:
                        lines.append(f"Assistant: {text}")
    except Exception:
        return ""
    return "\n\n".join(lines)


def _formatRecallContext(result: dict) -> str:
    """Format RAG recall result as markdown context string."""
    items = result.get("items", [])
    if not items:
        return ""
    parts = ["## Recalled Memories"]
    for item in items:
        text = item.get("text", "")
        if text:
            parts.append(f"- {text}")
    return "\n".join(parts) if len(parts) > 1 else ""


_LLM_PROMPT = """\
Extract key insights from this coding session. Output valid JSON:
{
  "global": [
    {"content": "...", "tags": "preference,python"}
  ],
  "project": [
    {"content": "...", "tags": "architecture,search"}
  ]
}

Rules:
- "global" = universal lessons: user preferences, tool conventions, workflow habits
- "project" = project-specific: architecture decisions, bug fixes, codebase patterns
- Prioritize USER FEEDBACK — corrections, preferences expressed, "always do X", "never do Y"
- Include: decisions + rationale, bugs fixed (what/why/how), patterns discovered, user preferences
- Exclude: transient debugging, file contents, routine operations
- Each item should be self-contained, useful memory (1-3 sentences)
- Maximum 10 global + 15 project items per session
- If there are no meaningful insights, output {"global": [], "project": []}"""


def _llmSummarize(transcript: str, project: str | None) -> dict | None:
    """Summarize transcript via `claude --print --model haiku`. Returns parsed JSON or None."""
    if len(transcript) > 100_000:
        transcript = transcript[:100_000] + "\n...(truncated)"

    prompt = _LLM_PROMPT
    if project:
        prompt += f"\n\nProject: {project}"

    full_input = f"{prompt}\n\n---\nSession transcript:\n\n{transcript}"
    try:
        r = subprocess.run(
            ["claude", "--print", "--model", "haiku"],
            input=full_input,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            return None

        output = r.stdout.strip()
        # Strip markdown code fences if present
        if "```json" in output:
            output = output.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in output:
            output = output.split("```", 1)[1].split("```", 1)[0]

        return json.loads(output.strip())
    except Exception:
        return None


def _storeMemories(memories: dict, session_id: str, project: str | None) -> None:
    """Store parsed LLM memories via API."""
    source = f"session:{session_id}"

    for item in memories.get("global", []):
        content = item.get("content", "")
        if content:
            _callApi(
                "/remember",
                {"content": content, "source": source, "tags": item.get("tags", "")},
                timeout=10.0,
            )

    for item in memories.get("project", []):
        content = item.get("content", "")
        if content:
            _callApi(
                "/remember",
                {
                    "content": content,
                    "source": source,
                    "tags": item.get("tags", ""),
                    "project": project,
                },
                timeout=10.0,
            )


def _storeRawChunks(transcript: str, session_id: str, project: str | None) -> None:
    """Fallback: store raw transcript as chunked memories."""
    source = f"session:{session_id}"
    chunk_size = 2000
    for i in range(0, len(transcript), chunk_size):
        _callApi(
            "/remember",
            {
                "content": transcript[i : i + chunk_size],
                "source": source,
                "tags": "transcript,raw",
                "project": project,
            },
            timeout=10.0,
        )


# ── Hook entry points ────────────────────────────────────────


def hookSessionStart(data: dict) -> dict:
    """SessionStart hook: recall recent project context. Returns hook response."""
    cwd = data.get("cwd", "")
    project = _deriveProject(cwd)
    query = f"recent work on {project}" if project else "recent work"

    result = _callApi(
        "/recall",
        {"query": query, "limit": 5, "use_rag": True, "project": project},
        timeout=10.0,
    )
    if result:
        ctx = _formatRecallContext(result)
        if ctx:
            return {"additionalContext": ctx}
    return {}


def hookPromptSubmit(data: dict) -> dict:
    """UserPromptSubmit hook: recall context relevant to user prompt. Returns hook response."""
    prompt = data.get("prompt", "").strip()
    if len(prompt) < 10 or _TRIVIAL.match(prompt):
        return {}

    result = _callApi(
        "/recall",
        {"query": prompt, "limit": 5, "use_rag": True},
        timeout=5.0,
    )
    if result:
        ctx = _formatRecallContext(result)
        if ctx:
            return {"additionalContext": ctx}
    return {}


def hookSessionEnd(data: dict) -> dict:
    """SessionEnd hook: summarize transcript and store memories. Returns hook response."""
    transcript_path = data.get("transcript_path", "")
    session_id = data.get("session_id", "unknown")
    cwd = data.get("cwd", "")

    if not transcript_path or not Path(transcript_path).exists():
        return {}

    project = _deriveProject(cwd)
    transcript = _extractTranscript(transcript_path)

    if not transcript or len(transcript) < 100:
        return {}

    memories = _llmSummarize(transcript, project)
    if memories:
        _storeMemories(memories, session_id, project)
    else:
        _storeRawChunks(transcript, session_id, project)

    return {}

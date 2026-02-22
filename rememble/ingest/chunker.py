"""Token-aware text chunking with overlap and optional markdown structure awareness."""

from __future__ import annotations

import re

import tiktoken

_encoder: tiktoken.Encoding | None = None

# Markdown break-point scoring
_BREAK_SCORES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^#{1,2}\s"), 100),  # h1-h2
    (re.compile(r"^#{3}\s|^```"), 80),  # h3, code fence
    (re.compile(r"^---+\s*$|^\*\*\*+\s*$|^___+\s*$"), 60),  # hr
    (re.compile(r"^#{4,6}\s"), 50),  # h4-h6
    (re.compile(r"^\s*$"), 20),  # blank line
    (re.compile(r"^\s*[-*+]\s|^\s*\d+\.\s"), 5),  # list item
]
_NEWLINE_SCORE = 1

_MARKDOWN_PATTERNS = [
    re.compile(r"^#{1,6}\s"),
    re.compile(r"^```"),
    re.compile(r"^\s*[-*+]\s"),
    re.compile(r"^\s*\d+\.\s"),
    re.compile(r"^---+\s*$|^\*\*\*+\s*$"),
    re.compile(r"^\[.*\]\(.*\)"),
    re.compile(r"^>\s"),
]


def _getEncoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _looksLikeMarkdown(text: str) -> bool:
    """Heuristic: >=2 markdown patterns in first 50 lines."""
    lines = text.split("\n")[:50]
    hits = 0
    for line in lines:
        for pat in _MARKDOWN_PATTERNS:
            if pat.match(line):
                hits += 1
                break
        if hits >= 2:
            return True
    return False


def _lineBreakScore(line: str) -> int:
    """Score a line as a potential chunk break point."""
    for pat, score in _BREAK_SCORES:
        if pat.match(line):
            return score
    return _NEWLINE_SCORE


def _findCodeFenceRanges(lines: list[str]) -> set[int]:
    """Return set of line indices inside code fences (never split here)."""
    inside: set[int] = set()
    in_fence = False
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            if in_fence:
                inside.add(i)  # closing fence itself is inside
                in_fence = False
            else:
                in_fence = True
                inside.add(i)
        elif in_fence:
            inside.add(i)
    return inside


def _chunkByMarkdown(text: str, target_tokens: int = 400, overlap_tokens: int = 40) -> list[str]:
    """Markdown-aware chunking using break-point scoring with distance decay."""
    enc = _getEncoder()
    lines = text.split("\n")

    # Map each line to its starting token offset
    line_token_offsets: list[int] = []
    cumulative = 0
    line_token_counts: list[int] = []
    for line in lines:
        line_token_offsets.append(cumulative)
        count = len(enc.encode(line + "\n"))
        line_token_counts.append(count)
        cumulative += count

    total_tokens = cumulative
    if total_tokens <= target_tokens:
        return [text]

    fence_lines = _findCodeFenceRanges(lines)

    chunks: list[str] = []
    start_line = 0
    window = 200  # tokens to search backward for break points

    while start_line < len(lines):
        start_offset = line_token_offsets[start_line]
        target_offset = start_offset + target_tokens

        # Find the line that exceeds target_offset
        end_line = start_line
        for i in range(start_line, len(lines)):
            if line_token_offsets[i] + line_token_counts[i] > target_offset:
                end_line = i
                break
        else:
            # Remaining text fits in one chunk
            chunk = "\n".join(lines[start_line:])
            if chunk.strip():
                chunks.append(chunk)
            break

        # Search backward in window for best break point
        window_start_offset = max(start_offset, target_offset - window)
        best_line = end_line
        best_score = -1.0

        for i in range(end_line, start_line, -1):
            if i in fence_lines:
                continue
            offset = line_token_offsets[i]
            if offset < window_start_offset:
                break
            distance = target_offset - offset
            base = _lineBreakScore(lines[i])
            decay = 1.0 - (distance / window) ** 2 * 0.7 if window > 0 else 1.0
            score = base * decay
            if score > best_score:
                best_score = score
                best_line = i

        # Build chunk from start_line to best_line (exclusive)
        if best_line <= start_line:
            best_line = end_line  # fallback: hard boundary

        chunk = "\n".join(lines[start_line:best_line])
        if chunk.strip():
            chunks.append(chunk)

        # Apply overlap: back up from cut point
        if overlap_tokens > 0 and best_line < len(lines):
            overlap_start = best_line
            overlap_offset = line_token_offsets[best_line]
            for i in range(best_line - 1, start_line - 1, -1):
                if overlap_offset - line_token_offsets[i] >= overlap_tokens:
                    overlap_start = i + 1
                    break
            else:
                overlap_start = start_line
            start_line = max(overlap_start, start_line + 1)
        else:
            start_line = best_line

    return chunks if chunks else [text]


def _chunkByTokens(text: str, target_tokens: int = 400, overlap_tokens: int = 40) -> list[str]:
    """Original token-boundary chunking (fallback)."""
    enc = _getEncoder()
    tokens = enc.encode(text)

    if len(tokens) <= target_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(1, target_tokens - overlap_tokens)

    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        chunk_text = enc.decode(tokens[start:end])
        if chunk_text.strip():
            chunks.append(chunk_text)
        start += step

    return chunks


def chunkText(
    text: str,
    target_tokens: int = 400,
    overlap_tokens: int = 40,
    markdown_aware: bool = True,
) -> list[str]:
    """Split text into token-aware chunks with overlap.

    Short texts (< target_tokens) returned as single-element list.
    When markdown_aware=True and text looks like markdown, uses structure-aware splitting.
    """
    enc = _getEncoder()
    tokens = enc.encode(text)

    if len(tokens) <= target_tokens:
        return [text]

    if markdown_aware and _looksLikeMarkdown(text):
        return _chunkByMarkdown(text, target_tokens, overlap_tokens)

    return _chunkByTokens(text, target_tokens, overlap_tokens)


def countTokens(text: str) -> int:
    """Count tokens using cl100k_base."""
    return len(_getEncoder().encode(text))


def truncateToTokens(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens, decoding back to valid UTF-8."""
    enc = _getEncoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

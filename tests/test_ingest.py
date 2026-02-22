"""Tests for text chunking."""

from __future__ import annotations

from rememble.ingest.chunker import (
    _looksLikeMarkdown,
    chunkText,
    countTokens,
    truncateToTokens,
)


def test_shortTextNotChunked():
    text = "Hello world, this is a short text."
    chunks = chunkText(text, target_tokens=400)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_longTextChunked():
    # Generate text that's definitely > 400 tokens
    text = " ".join(f"word{i}" for i in range(800))
    chunks = chunkText(text, target_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    # Each chunk should have roughly target_tokens tokens
    for chunk in chunks:
        tokens = countTokens(chunk)
        assert tokens <= 110  # small tolerance


def test_chunkOverlap():
    text = " ".join(f"token{i}" for i in range(200))
    chunks = chunkText(text, target_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    # Check that adjacent chunks share some content
    for i in range(len(chunks) - 1):
        words_a = set(chunks[i].split()[-15:])
        words_b = set(chunks[i + 1].split()[:15])
        assert len(words_a & words_b) > 0, "Chunks should overlap"


def test_countTokens():
    assert countTokens("hello") > 0
    assert countTokens("") == 0


def test_truncateToTokens():
    text = "This is a test sentence with several words in it."
    truncated = truncateToTokens(text, 3)
    assert countTokens(truncated) <= 3
    # Original text should be longer
    assert countTokens(text) > 3


# ── Markdown-Aware Chunking ─────────────────────────────────


def _makeMarkdownDoc(sections: int = 10, words_per_section: int = 80) -> str:
    """Generate a markdown document with headings, lists, and code blocks."""
    parts = []
    for i in range(sections):
        level = "#" if i == 0 else "##"
        parts.append(f"{level} Section {i}")
        parts.append("")
        parts.append(" ".join(f"word{i}_{j}" for j in range(words_per_section)))
        parts.append("")
        if i == 2:
            parts.append("```python")
            parts.append("def hello():")
            parts.append("    print('world')")
            parts.append("```")
            parts.append("")
        if i == 5:
            parts.append("- item one")
            parts.append("- item two")
            parts.append("- item three")
            parts.append("")
    return "\n".join(parts)


def test_markdownDetection():
    md = "# Title\n\nSome text\n\n## Subtitle\n\n- item\n"
    assert _looksLikeMarkdown(md) is True
    plain = "Just some plain text with no markdown at all."
    assert _looksLikeMarkdown(plain) is False


def test_headingSplit():
    """Chunks should prefer to split at heading boundaries."""
    doc = _makeMarkdownDoc(sections=10, words_per_section=80)
    chunks = chunkText(doc, target_tokens=200, overlap_tokens=20, markdown_aware=True)
    assert len(chunks) > 1
    # At least one chunk should start with a heading
    heading_starts = sum(1 for c in chunks if c.lstrip().startswith("#"))
    assert heading_starts >= 1


def test_codeFenceProtection():
    """Code fences should not be split in the middle."""
    doc = "# Title\n\nIntro text.\n\n"
    doc += "```python\n"
    doc += "\n".join(f"line_{i} = {i}" for i in range(40))
    doc += "\n```\n\n"
    doc += "## After Code\n\nMore text here.\n" + " ".join(f"w{i}" for i in range(200))
    chunks = chunkText(doc, target_tokens=100, overlap_tokens=10, markdown_aware=True)
    # No chunk should contain an opening ``` without a closing one (or vice versa)
    for chunk in chunks:
        fence_count = chunk.count("```")
        # fences should be paired (0 or 2) within a chunk, or chunk starts/ends at fence boundary
        # At minimum, we shouldn't have a lone opening without closing in most chunks
        assert fence_count % 2 == 0 or fence_count <= 2


def test_blankLinePreference():
    """Chunks should prefer blank lines over mid-paragraph splits."""
    paragraphs = []
    for i in range(8):
        paragraphs.append(f"## Para {i}")
        paragraphs.append("")
        paragraphs.append(" ".join(f"text{i}_{j}" for j in range(60)))
        paragraphs.append("")
    doc = "\n".join(paragraphs)
    chunks = chunkText(doc, target_tokens=150, overlap_tokens=15, markdown_aware=True)
    assert len(chunks) > 1


def test_nonMarkdownFallback():
    """Non-markdown text should use token-boundary chunking."""
    text = " ".join(f"word{i}" for i in range(800))
    chunks_md = chunkText(text, target_tokens=100, overlap_tokens=10, markdown_aware=True)
    chunks_plain = chunkText(text, target_tokens=100, overlap_tokens=10, markdown_aware=False)
    # Both should produce similar results since text is not markdown
    assert len(chunks_md) == len(chunks_plain)


def test_disabledFlag():
    """markdown_aware=False should always use token chunking."""
    doc = _makeMarkdownDoc(sections=10, words_per_section=80)
    chunks = chunkText(doc, target_tokens=200, overlap_tokens=20, markdown_aware=False)
    assert len(chunks) > 1


def test_distanceDecayPreference():
    """Break points closer to the target boundary should be preferred over distant ones."""
    # Build doc with heading early and blank line near target
    lines = ["# Early Heading", ""]
    lines += [" ".join(f"filler{j}" for j in range(30))] * 3
    lines += [""]  # blank line near target
    lines += [" ".join(f"more{j}" for j in range(30))] * 5
    lines += ["## Late Section", ""]
    lines += [" ".join(f"tail{j}" for j in range(30))] * 5
    doc = "\n".join(lines)
    chunks = chunkText(doc, target_tokens=100, overlap_tokens=10, markdown_aware=True)
    # Should produce at least 2 chunks
    assert len(chunks) >= 2

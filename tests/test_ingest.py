"""Tests for text chunking."""

from __future__ import annotations

from rememble.ingest.chunker import chunkText, countTokens, truncateToTokens


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

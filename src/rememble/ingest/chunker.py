"""Token-aware text chunking with overlap."""

from __future__ import annotations

import tiktoken

_encoder: tiktoken.Encoding | None = None


def _getEncoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def chunkText(text: str, target_tokens: int = 400, overlap_tokens: int = 40) -> list[str]:
    """Split text into token-aware chunks with overlap.

    Short texts (< target_tokens) returned as single-element list.
    """
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

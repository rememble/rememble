"""Embedding provider factory with auto-detection fallback chain."""

from __future__ import annotations

import logging

from rememble.config import EmbeddingConfig
from rememble.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


async def createProvider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Create embedding provider from config. Falls back: configured → ollama → local → error."""
    provider = config.provider.lower()

    if provider == "ollama":
        return await _tryOllama(config)
    elif provider == "compat":
        return _createCompat(config)
    elif provider == "local":
        return _createLocal(config)
    else:
        # Auto-detect: try ollama first, then local
        return await _autoDetect(config)


async def _tryOllama(config: EmbeddingConfig) -> EmbeddingProvider:
    from rememble.embeddings.ollama import OllamaProvider

    p = OllamaProvider(model=config.model, dimensions=config.dimensions, url=config.ollama_url)
    if await p.healthCheck():
        logger.info("Using Ollama provider: %s", p.name)
        return p  # type: ignore[return-value]
    raise ConnectionError(
        f"Ollama not reachable at {config.ollama_url} or model '{config.model}' not found"
    )


def _createCompat(config: EmbeddingConfig) -> EmbeddingProvider:
    from rememble.embeddings.compat import CompatProvider

    p = CompatProvider(
        model=config.model,
        api_key=config.api_key,
        dimensions=config.dimensions,
        url=config.api_endpoint,
        api_type=config.api_type,
    )
    logger.info("Using compat provider (%s): %s", config.api_type, p.name)
    return p  # type: ignore[return-value]


def _createLocal(config: EmbeddingConfig) -> EmbeddingProvider:
    from rememble.embeddings.local import LocalProvider

    p = LocalProvider(model=config.local_model, backend=config.local_backend)
    logger.info("Using local provider: %s", p.name)
    return p  # type: ignore[return-value]


async def _autoDetect(config: EmbeddingConfig) -> EmbeddingProvider:
    """Try ollama → local → error."""
    try:
        return await _tryOllama(config)
    except (ConnectionError, Exception):
        logger.info("Ollama not available, trying local embeddings...")

    try:
        return _createLocal(config)
    except ImportError:
        pass

    raise RuntimeError(
        "No embedding provider available. Install sentence-transformers "
        "('uv pip install rememble[local]') or start Ollama."
    )

"""Embedding provider factory."""

from __future__ import annotations

import logging

from rememble.config import RemembleConfig
from rememble.embeddings.base import EmbeddingProvider
from rememble.embeddings.compat import CompatProvider

logger = logging.getLogger(__name__)


async def createProvider(config: RemembleConfig) -> EmbeddingProvider:
    p = CompatProvider(
        model=config.embedding_api_model,
        api_url=config.embedding_api_url,
        api_key=config.embedding_api_key,
        dimensions=config.embedding_dimensions,
    )
    logger.info("Embedding provider: %s (%d dims)", p.name, p.dimensions)
    return p  # type: ignore[return-value]

"""Embedding providers: ollama, local (sentence-transformers), openai-compat."""

from rememble.embeddings.base import EmbeddingProvider
from rememble.embeddings.factory import createProvider

__all__ = ["EmbeddingProvider", "createProvider"]

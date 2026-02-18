"""Embedding providers: ollama, local (sentence-transformers), cohere."""

from rememble.embeddings.base import EmbeddingProvider
from rememble.embeddings.factory import createProvider

__all__ = ["EmbeddingProvider", "createProvider"]

"""Local embedding provider using sentence-transformers (ONNX backend)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LocalProvider:
    """Embedding provider using sentence-transformers with ONNX backend."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        backend: str = "onnx",
    ):
        self._model_name = model
        self._backend = backend
        self._model = None
        self._dims: int | None = None

    @property
    def dimensions(self) -> int:
        if self._dims is None:
            # Default for all-MiniLM-L6-v2
            return 384
        return self._dims

    @property
    def name(self) -> str:
        return f"local/{self._model_name}"

    def _loadModel(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: uv pip install 'rememble[local]'"
            ) from e
        logger.info("Loading model: %s (backend=%s)", self._model_name, self._backend)
        self._model = SentenceTransformer(self._model_name, backend=self._backend)
        dim = self._model.get_sentence_embedding_dimension()
        self._dims = dim if isinstance(dim, int) else 384

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed using sentence-transformers."""
        self._loadModel()
        assert self._model is not None
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [vec.tolist() for vec in embeddings]

    async def embedOne(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]

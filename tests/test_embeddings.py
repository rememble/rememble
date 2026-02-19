"""Tests for embedding providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rememble.config import RemembleConfig
from rememble.embeddings.base import EmbeddingProvider
from rememble.embeddings.compat import CompatProvider

# -- Protocol conformance --


class TestEmbeddingProtocol:
    def test_fakeEmbedderConforms(self, fake_embedder):
        assert isinstance(fake_embedder, EmbeddingProvider)

    def test_compatConforms(self):
        p = CompatProvider(model="test-model")
        assert isinstance(p, EmbeddingProvider)


# -- CompatProvider --


class TestCompatProvider:
    def test_properties(self):
        p = CompatProvider(model="nomic-embed-text", dimensions=768)
        assert p.name == "compat/nomic-embed-text"
        assert p.dimensions == 768

    def test_noApiKeyAllowed(self):
        p = CompatProvider(model="nomic-embed-text", api_url="http://localhost:11434/v1")
        assert p.dimensions == 768

    def test_apiKeySetsBearerHeader(self):
        p = CompatProvider(model="test-model", api_key="sk-test")
        assert "Authorization" in p._client.headers
        assert p._client.headers["Authorization"] == "Bearer sk-test"

    def test_noApiKeyNoAuthHeader(self):
        p = CompatProvider(model="test-model")
        assert "authorization" not in {k.lower() for k in p._client.headers}

    @pytest.mark.asyncio
    async def test_embed(self):
        p = CompatProvider(model="text-embedding-3-small", dimensions=1536)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2], "index": 0}]}
        mock_resp.raise_for_status = MagicMock()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=mock_resp)

        result = await p.embed(["hello"])
        assert result == [[0.1, 0.2]]
        p._client.post.assert_called_once_with(
            "/embeddings", json={"model": "text-embedding-3-small", "input": ["hello"]}
        )

    @pytest.mark.asyncio
    async def test_embedOrderPreserved(self):
        p = CompatProvider(model="test-model")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"embedding": [0.3, 0.4], "index": 1},
                {"embedding": [0.1, 0.2], "index": 0},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=mock_resp)

        result = await p.embed(["first", "second"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_embedOne(self):
        p = CompatProvider(model="test-model")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"embedding": [0.5, 0.6], "index": 0}]}
        mock_resp.raise_for_status = MagicMock()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=mock_resp)

        result = await p.embedOne("test")
        assert result == [0.5, 0.6]


# -- Factory --


class TestFactory:
    @pytest.mark.asyncio
    async def test_createProvider(self):
        config = RemembleConfig(
            embedding_api_url="http://localhost:11434/v1",
            embedding_api_model="nomic-embed-text",
            embedding_dimensions=768,
        )
        from rememble.embeddings.factory import createProvider

        with patch("rememble.embeddings.factory.CompatProvider") as MockCompat:
            MockCompat.return_value = MagicMock(spec=EmbeddingProvider)
            result = await createProvider(config)
            MockCompat.assert_called_once_with(
                model="nomic-embed-text",
                api_url="http://localhost:11434/v1",
                api_key=None,
                dimensions=768,
            )
            assert result is MockCompat.return_value

    @pytest.mark.asyncio
    async def test_createProviderWithApiKey(self):
        config = RemembleConfig(
            embedding_api_url="https://openrouter.ai/api/v1",
            embedding_api_key="sk-test",
            embedding_api_model="text-embedding-3-small",
            embedding_dimensions=1536,
        )
        from rememble.embeddings.factory import createProvider

        with patch("rememble.embeddings.factory.CompatProvider") as MockCompat:
            MockCompat.return_value = MagicMock(spec=EmbeddingProvider)
            await createProvider(config)
            MockCompat.assert_called_once_with(
                model="text-embedding-3-small",
                api_url="https://openrouter.ai/api/v1",
                api_key="sk-test",
                dimensions=1536,
            )

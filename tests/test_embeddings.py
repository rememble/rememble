"""Tests for embedding providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from rememble.config import EmbeddingConfig
from rememble.embeddings.base import EmbeddingProvider
from rememble.embeddings.cohere import CohereProvider
from rememble.embeddings.ollama import OllamaProvider

# -- Protocol conformance --


class TestEmbeddingProtocol:
    def test_fakeEmbedderConforms(self, fake_embedder):
        assert isinstance(fake_embedder, EmbeddingProvider)

    def test_ollamaConforms(self):
        p = OllamaProvider()
        assert isinstance(p, EmbeddingProvider)

    def test_cohereConforms(self):
        p = CohereProvider(api_key="test-key")
        assert isinstance(p, EmbeddingProvider)


# -- OllamaProvider --


class TestOllamaProvider:
    def test_properties(self):
        p = OllamaProvider(model="test-model", dimensions=512, url="http://localhost:11434")
        assert p.name == "ollama/test-model"
        assert p.dimensions == 512

    @pytest.mark.asyncio
    async def test_embed(self):
        p = OllamaProvider(model="nomic-embed-text", dimensions=768)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_resp.raise_for_status = MagicMock()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=mock_resp)

        result = await p.embed(["hello"])
        assert result == [[0.1, 0.2, 0.3]]
        p._client.post.assert_called_once_with(
            "/api/embed", json={"model": "nomic-embed-text", "input": ["hello"]}
        )

    @pytest.mark.asyncio
    async def test_embedOne(self):
        p = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.5, 0.6]]}
        mock_resp.raise_for_status = MagicMock()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=mock_resp)

        result = await p.embedOne("test")
        assert result == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_healthCheckPass(self):
        p = OllamaProvider(model="nomic-embed-text")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}
        mock_resp.raise_for_status = MagicMock()
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=mock_resp)

        assert await p.healthCheck() is True

    @pytest.mark.asyncio
    async def test_healthCheckFail(self):
        p = OllamaProvider(model="nomic-embed-text")
        p._client = AsyncMock()
        p._client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        assert await p.healthCheck() is False


# -- CohereProvider --


class TestCohereProvider:
    def test_properties(self):
        p = CohereProvider(model="embed-v4.0", api_key="key", dimensions=1024)
        assert p.name == "cohere/embed-v4.0"
        assert p.dimensions == 1024

    def test_requiresApiKey(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="API key required"),
        ):
            CohereProvider(api_key=None)

    def test_forQuerySwitchesInputType(self):
        p = CohereProvider(api_key="key")
        assert p._input_type == "search_document"
        q = p.forQuery()
        assert q._input_type == "search_query"
        # Original unchanged
        assert p._input_type == "search_document"

    @pytest.mark.asyncio
    async def test_embed(self):
        p = CohereProvider(api_key="key", model="embed-v4.0")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": {"float": [[0.1, 0.2]]}}
        mock_resp.raise_for_status = MagicMock()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=mock_resp)

        result = await p.embed(["hello"])
        assert result == [[0.1, 0.2]]
        call_json = p._client.post.call_args[1]["json"]
        assert call_json["input_type"] == "search_document"
        assert call_json["model"] == "embed-v4.0"


# -- Factory --


class TestFactory:
    @pytest.mark.asyncio
    async def test_createOllamaProvider(self):
        config = EmbeddingConfig(provider="ollama")
        with patch("rememble.embeddings.factory._tryOllama") as mock:
            mock.return_value = MagicMock(spec=EmbeddingProvider)
            from rememble.embeddings.factory import createProvider

            result = await createProvider(config)
            mock.assert_called_once_with(config)
            assert result is mock.return_value

    @pytest.mark.asyncio
    async def test_createLocalProvider(self):
        config = EmbeddingConfig(provider="local")
        with patch("rememble.embeddings.factory._createLocal") as mock:
            mock.return_value = MagicMock(spec=EmbeddingProvider)
            from rememble.embeddings.factory import createProvider

            result = await createProvider(config)
            mock.assert_called_once_with(config)
            assert result is mock.return_value

    @pytest.mark.asyncio
    async def test_autoDetectFallsBackToLocal(self):
        config = EmbeddingConfig(provider="auto")
        with (
            patch("rememble.embeddings.factory._tryOllama", side_effect=ConnectionError),
            patch("rememble.embeddings.factory._createLocal") as mock_local,
        ):
            mock_local.return_value = MagicMock(spec=EmbeddingProvider)
            from rememble.embeddings.factory import createProvider

            result = await createProvider(config)
            assert result is mock_local.return_value

"""RemembleClient tests with mocked httpx responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rememble.client import RemembleClient


def _mockResp(data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def client():
    with patch("rememble.client.httpx.Client") as MockClient:
        mock = MockClient.return_value
        c = RemembleClient("http://localhost:7707")
        c._client = mock
        yield c, mock


class TestHealth:
    def test_health(self, client):
        c, mock = client
        mock.get.return_value = _mockResp({"status": "ok"})
        assert c.health() == {"status": "ok"}
        mock.get.assert_called_once_with("/health", params=None)


class TestRemember:
    def test_remember(self, client):
        c, mock = client
        mock.post.return_value = _mockResp({"stored": True, "memory_ids": [1], "chunks": 1})
        result = c.remember("hello", source="test", tags="t")
        assert result["stored"] is True
        mock.post.assert_called_once_with(
            "/remember",
            json={
                "content": "hello", "source": "test", "tags": "t",
                "metadata": None, "project": None,
            },
        )

    def test_remember_withProject(self, client):
        c, mock = client
        mock.post.return_value = _mockResp({"stored": True, "memory_ids": [1], "chunks": 1})
        result = c.remember("hello", project="myapp")
        assert result["stored"] is True
        mock.post.assert_called_once_with(
            "/remember",
            json={
                "content": "hello", "source": None, "tags": None,
                "metadata": None, "project": "myapp",
            },
        )


class TestRecall:
    def test_recall_rag(self, client):
        c, mock = client
        mock.post.return_value = _mockResp({"query": "q", "total_tokens": 10, "items": []})
        result = c.recall("q", limit=5, use_rag=True)
        assert result["query"] == "q"

    def test_recall_raw(self, client):
        c, mock = client
        mock.post.return_value = _mockResp({"query": "q", "results": [], "graph": []})
        result = c.recall("q", use_rag=False)
        assert "results" in result

    def test_recall_withProject(self, client):
        c, mock = client
        mock.post.return_value = _mockResp({"query": "q", "total_tokens": 10, "items": []})
        c.recall("q", project="myapp")
        mock.post.assert_called_once_with(
            "/recall",
            json={"query": "q", "limit": 10, "use_rag": True, "project": "myapp"},
        )


class TestForget:
    def test_forget(self, client):
        c, mock = client
        mock.post.return_value = _mockResp({"forgotten": True, "memory_id": 1})
        assert c.forget(1)["forgotten"] is True


class TestListMemories:
    def test_list_default(self, client):
        c, mock = client
        mock.get.return_value = _mockResp({"memories": [], "count": 0, "offset": 0})
        result = c.listMemories()
        assert result["count"] == 0
        mock.get.assert_called_once_with(
            "/memories", params={"status": "active", "limit": 20, "offset": 0}
        )

    def test_list_filtered(self, client):
        c, mock = client
        mock.get.return_value = _mockResp({"memories": [], "count": 0, "offset": 0})
        c.listMemories(source="file", tags="python")
        mock.get.assert_called_once_with(
            "/memories",
            params={
                "status": "active",
                "limit": 20,
                "offset": 0,
                "source": "file",
                "tags": "python",
            },
        )


class TestStats:
    def test_stats(self, client):
        c, mock = client
        mock.get.return_value = _mockResp({"total_memories": 5})
        assert c.stats()["total_memories"] == 5


class TestListMemoriesProject:
    def test_list_withProject(self, client):
        c, mock = client
        mock.get.return_value = _mockResp({"memories": [], "count": 0, "offset": 0})
        c.listMemories(project="myapp")
        mock.get.assert_called_once_with(
            "/memories",
            params={"status": "active", "limit": 20, "offset": 0, "project": "myapp"},
        )


class TestGraph:
    def test_createEntities(self, client):
        c, mock = client
        mock.post.return_value = _mockResp({"created": [{"name": "A"}]})
        result = c.createEntities([{"name": "A", "entity_type": "test"}])
        assert len(result["created"]) == 1

    def test_createEntities_withProject(self, client):
        c, mock = client
        mock.post.return_value = _mockResp({"created": [{"name": "A"}]})
        c.createEntities([{"name": "A", "entity_type": "test"}], project="myapp")
        mock.post.assert_called_once_with(
            "/entities",
            json={"entities": [{"name": "A", "entity_type": "test"}], "project": "myapp"},
        )

    def test_createRelations(self, client):
        c, mock = client
        mock.post.return_value = _mockResp({"created": [{"type": "linked"}]})
        result = c.createRelations([{"from_name": "A", "to_name": "B", "relation_type": "linked"}])
        assert result["created"][0]["type"] == "linked"

    def test_addObservations(self, client):
        c, mock = client
        mock.post.return_value = _mockResp({"observations_added": 2})
        result = c.addObservations("Python", ["Fast", "Dynamic"])
        assert result["observations_added"] == 2

    def test_searchGraph(self, client):
        c, mock = client
        mock.get.return_value = _mockResp({"entities": [{"name": "A"}]})
        result = c.searchGraph("A")
        assert len(result["entities"]) == 1

    def test_searchGraph_withProject(self, client):
        c, mock = client
        mock.get.return_value = _mockResp({"entities": []})
        c.searchGraph("A", project="myapp")
        mock.get.assert_called_once_with(
            "/graph", params={"query": "A", "limit": 10, "project": "myapp"}
        )

    def test_deleteEntities(self, client):
        c, mock = client
        mock.request.return_value = _mockResp({"deleted": ["A"]})
        result = c.deleteEntities(["A"])
        assert "A" in result["deleted"]


class TestContextManager:
    def test_contextManager(self):
        with patch("rememble.client.httpx.Client") as MockClient:
            with RemembleClient() as c:
                assert c._client is not None
            MockClient.return_value.close.assert_called_once()

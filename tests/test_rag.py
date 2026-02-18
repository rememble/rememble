"""Tests for RAG context assembly."""

from __future__ import annotations

import sqlite3

from rememble.config import RAGConfig, SearchConfig
from rememble.db import insertMemory
from rememble.rag.context import buildContext


class TestBuildContext:
    def test_basicContext(self, db: sqlite3.Connection):
        insertMemory(db, "Python is a programming language created by Guido.", [1.0, 0.0, 0.0, 0.0])
        insertMemory(db, "Rust is a systems programming language.", [0.8, 0.2, 0.0, 0.0])
        insertMemory(db, "Cooking pasta requires water.", [0.0, 0.0, 1.0, 0.0])

        context = buildContext(
            db,
            query="programming languages",
            query_embedding=[0.9, 0.1, 0.0, 0.0],
            search_config=SearchConfig(),
            rag_config=RAGConfig(max_context_tokens=500),
        )
        assert context.total_tokens > 0
        assert len(context.items) > 0
        assert context.items[0].kind == "expanded"

    def test_tokenBudgetRespected(self, db: sqlite3.Connection):
        # Insert many memories
        for i in range(20):
            vec = [float(i % 4 == j) for j in range(4)]
            insertMemory(db, f"Memory number {i} with some content about topic {i}.", vec)

        context = buildContext(
            db,
            query="topic",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            search_config=SearchConfig(),
            rag_config=RAGConfig(max_context_tokens=50),
        )
        assert context.total_tokens <= 50

    def test_emptyDbReturnsEmptyContext(self, db: sqlite3.Connection):
        context = buildContext(
            db,
            query="anything",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            search_config=SearchConfig(),
            rag_config=RAGConfig(),
        )
        assert context.total_tokens == 0
        assert len(context.items) == 0

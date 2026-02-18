"""Config loading from ~/.rememble/config.json with defaults."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

CONFIG_DIR = Path.home() / ".rememble"
CONFIG_PATH = CONFIG_DIR / "config.json"


class EmbeddingConfig(BaseModel):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    dimensions: int = 768
    ollama_url: str = "http://localhost:11434"
    cohere_api_key: str | None = None
    cohere_model: str = "embed-v4.0"
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_backend: str = "onnx"


class SearchConfig(BaseModel):
    default_limit: int = 10
    rrf_k: int = 60
    bm25_weight: float = 0.4
    vector_weight: float = 0.5
    temporal_weight: float = 0.1
    recency_half_life_days: float = 7.0


class RAGConfig(BaseModel):
    max_context_tokens: int = 1500
    max_snippets: int = 24
    snippet_max_tokens: int = 200
    expansion_max_tokens: int = 600


class ChunkingConfig(BaseModel):
    target_tokens: int = 400
    overlap_tokens: int = 40


class RemembleConfig(BaseModel):
    db_path: str = Field(default_factory=lambda: str(CONFIG_DIR / "memory.db"))
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


def loadConfig() -> RemembleConfig:
    """Load config from ~/.rememble/config.json, creating defaults if missing."""
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text())
        return RemembleConfig.model_validate(raw)
    # Create default config
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = RemembleConfig()
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))
    return config

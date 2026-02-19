"""Config loading from ~/.rememble/config.json with env var overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path.home() / ".rememble"
CONFIG_PATH = CONFIG_DIR / "config.json"


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


class RemembleConfig(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="REMEMBLE_",
        extra="ignore",
    )
    db_path: str = Field(default_factory=lambda: str(CONFIG_DIR / "memory.db"))
    # Embedding API settings — prefixed so future API types (llm_api_*) stay distinct
    embedding_api_url: str = "http://localhost:11434/v1"
    embedding_api_key: str | None = None
    embedding_api_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768
    # HTTP server
    port: int = 7707
    pid_path: str = Field(default_factory=lambda: str(CONFIG_DIR / "rememble.pid"))
    # Sub-configs
    search: SearchConfig = Field(default_factory=SearchConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


def loadConfig() -> RemembleConfig:
    """Load config from ~/.rememble/config.json with env var overrides."""
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text())
        return RemembleConfig(**raw)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = RemembleConfig()
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))
    return config

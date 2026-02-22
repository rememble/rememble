"""Pydantic models for memories, entities, relations, and search results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Memory(BaseModel):
    id: int
    content: str
    source: str | None = None
    tags: str | None = None
    metadata_json: str | None = None
    project: str | None = None
    created_at: int
    updated_at: int
    accessed_at: int
    access_count: int = 0
    status: str = "active"


class Entity(BaseModel):
    id: int
    name: str
    entity_type: str
    project: str | None = None
    created_at: int
    updated_at: int


class Observation(BaseModel):
    id: int
    entity_id: int
    content: str
    source: str | None = None
    valid_from: int | None = None
    valid_to: int | None = None
    created_at: int


class Relation(BaseModel):
    id: int
    from_entity_id: int
    to_entity_id: int
    relation_type: str
    metadata_json: str | None = None
    created_at: int


class SearchResult(BaseModel):
    memory_id: int
    score: float
    snippet: str | None = None
    source: str | None = None  # which search lane


class GraphResult(BaseModel):
    entity: Entity
    observations: list[Observation] = Field(default_factory=list)
    relations: list[RelationWithEntity] = Field(default_factory=list)


class RelationWithEntity(BaseModel):
    relation: Relation
    entity: Entity  # the connected entity
    direction: str  # "outbound" or "inbound"


class FusedResult(BaseModel):
    memory_id: int
    score: float
    best_rank: int
    sources: list[str] = Field(default_factory=list)  # which lanes contributed
    snippet: str | None = None
    content: str | None = None


class RAGItem(BaseModel):
    kind: str  # "expanded", "snippet", "graph"
    memory_id: int | None = None
    score: float = 0.0
    text: str = ""
    tokens: int = 0


class RAGContext(BaseModel):
    query: str
    items: list[RAGItem] = Field(default_factory=list)
    total_tokens: int = 0
    entities: list[GraphResult] = Field(default_factory=list)

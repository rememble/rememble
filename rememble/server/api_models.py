"""Pydantic request models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RememberRequest(BaseModel):
    content: str
    source: str | None = None
    tags: str | None = None
    metadata: str | None = None
    project: str | None = None


class RecallRequest(BaseModel):
    query: str
    limit: int = 10
    use_rag: bool = True
    project: str | None = None


class ForgetRequest(BaseModel):
    memory_id: int


class EntityInput(BaseModel):
    name: str
    entity_type: str
    observations: list[str] = Field(default_factory=list)


class CreateEntitiesRequest(BaseModel):
    entities: list[EntityInput]
    project: str | None = None


class RelationInput(BaseModel):
    from_name: str
    to_name: str
    relation_type: str
    metadata: str | None = None


class CreateRelationsRequest(BaseModel):
    relations: list[RelationInput]


class AddObservationsRequest(BaseModel):
    entity_name: str
    observations: list[str]
    source: str | None = None


class DeleteEntitiesRequest(BaseModel):
    names: list[str]

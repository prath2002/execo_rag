"""Pinecone request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .chunk import ChunkMetadata
from .common import ExecoBaseModel


class PineconeVectorRecord(ExecoBaseModel):
    """Vector payload stored in Pinecone."""

    id: str = Field(min_length=1)
    values: list[float] = Field(min_length=1)
    metadata: ChunkMetadata
    namespace: str | None = None


class PineconeUpsertRequest(ExecoBaseModel):
    """Upsert request for one Pinecone namespace."""

    namespace: str = Field(min_length=1)
    vectors: list[PineconeVectorRecord] = Field(default_factory=list)


class PineconeQueryRequest(ExecoBaseModel):
    """Similarity query against Pinecone."""

    vector: list[float] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    namespace: str | None = None
    filter: dict[str, Any] | None = None
    include_metadata: bool = True
    include_values: bool = False


class PineconeMatch(ExecoBaseModel):
    """Single Pinecone query match."""

    id: str = Field(min_length=1)
    score: float
    metadata: ChunkMetadata | None = None
    values: list[float] | None = None


class PineconeQueryResult(ExecoBaseModel):
    """Response payload for a Pinecone query."""

    matches: list[PineconeMatch] = Field(default_factory=list)

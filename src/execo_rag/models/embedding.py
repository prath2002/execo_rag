"""Embedding provider request and response models."""

from __future__ import annotations

from pydantic import Field

from .common import ExecoBaseModel


class EmbeddingRequest(ExecoBaseModel):
    """Embedding generation request."""

    model_name: str = Field(min_length=1)
    inputs: list[str] = Field(min_length=1)
    document_id: str | None = None


class EmbeddingVector(ExecoBaseModel):
    """Single embedded vector output."""

    chunk_id: str = Field(min_length=1)
    values: list[float] = Field(min_length=1)
    dimension: int = Field(ge=1)


class EmbeddingBatchResult(ExecoBaseModel):
    """Batch embedding output from a provider."""

    model_name: str = Field(min_length=1)
    vectors: list[EmbeddingVector] = Field(default_factory=list)

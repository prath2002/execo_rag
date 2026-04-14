"""LangGraph pipeline state models."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator

from .chunk import EnrichedChunk, ValidatedChunk
from .common import ExecoBaseModel
from .document import DocumentInput
from .embedding import EmbeddingVector
from .extraction import CleanedPage, ExtractedDocument
from .metadata import DocumentMetadata
from .pinecone import PineconeVectorRecord


class PipelineStatus(str, Enum):
    """Lifecycle status for the ingestion graph."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineState(ExecoBaseModel):
    """State object shared across LangGraph ingestion nodes."""

    request_id: str = Field(min_length=1)
    pdf_path: Path
    document: DocumentInput
    status: PipelineStatus = PipelineStatus.PENDING
    extracted_document: ExtractedDocument | None = None
    cleaned_pages: list[CleanedPage] = Field(default_factory=list)
    document_metadata: DocumentMetadata | None = None
    chunks: list[EnrichedChunk] = Field(default_factory=list)
    validated_chunks: list[ValidatedChunk] = Field(default_factory=list)
    embeddings: list[EmbeddingVector] = Field(default_factory=list)
    pinecone_records: list[PineconeVectorRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @field_validator("pdf_path", mode="before")
    @classmethod
    def coerce_path(cls, value: str | Path) -> Path:
        """Coerce incoming path values into `Path` objects."""

        return Path(value)

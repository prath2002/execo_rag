"""Shared Pydantic models for the application."""

from .chunk import ChunkFlags, ChunkMetadata, EnrichedChunk, RawChunk, ValidatedChunk
from .common import ExecoBaseModel
from .document import DocumentInput, DocumentSource, DocumentType, SourceType
from .embedding import EmbeddingBatchResult, EmbeddingRequest, EmbeddingVector
from .extraction import CleanedPage, ExtractedBlock, ExtractedDocument, ExtractedPage, SectionSegment
from .metadata import DocumentMetadata, MetadataEvidence, MetadataField, SectionType
from .pinecone import (
    PineconeMatch,
    PineconeQueryRequest,
    PineconeQueryResult,
    PineconeUpsertRequest,
    PineconeVectorRecord,
)
from .pipeline_state import PipelineState, PipelineStatus

__all__ = [
    "ChunkFlags",
    "ChunkMetadata",
    "DocumentInput",
    "DocumentMetadata",
    "DocumentSource",
    "DocumentType",
    "EmbeddingBatchResult",
    "EmbeddingRequest",
    "EmbeddingVector",
    "EnrichedChunk",
    "ExecoBaseModel",
    "ExtractedBlock",
    "ExtractedDocument",
    "ExtractedPage",
    "CleanedPage",
    "MetadataEvidence",
    "MetadataField",
    "PipelineState",
    "PipelineStatus",
    "PineconeMatch",
    "PineconeQueryRequest",
    "PineconeQueryResult",
    "PineconeUpsertRequest",
    "PineconeVectorRecord",
    "RawChunk",
    "SectionSegment",
    "SectionType",
    "SourceType",
    "ValidatedChunk",
]

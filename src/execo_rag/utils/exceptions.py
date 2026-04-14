"""Application-specific exception hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AppError(Exception):
    """Base class for domain-specific application errors."""

    message: str
    error_code: str = "application_error"
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


@dataclass(slots=True)
class ConfigError(AppError):
    """Raised when the application configuration is invalid."""

    error_code: str = "config_error"


@dataclass(slots=True)
class ExtractionError(AppError):
    """Raised when PDF extraction or preprocessing fails."""

    error_code: str = "extraction_error"


@dataclass(slots=True)
class ValidationError(AppError):
    """Raised when application-level validation fails."""

    error_code: str = "validation_error"


@dataclass(slots=True)
class ExternalServiceError(AppError):
    """Raised when a remote dependency call fails."""

    error_code: str = "external_service_error"


@dataclass(slots=True)
class MetadataExtractionError(AppError):
    """Raised when document metadata cannot be extracted."""

    error_code: str = "metadata_extraction_error"


@dataclass(slots=True)
class ChunkingError(AppError):
    """Raised when chunk creation fails."""

    error_code: str = "chunking_error"


@dataclass(slots=True)
class EmbeddingError(AppError):
    """Raised when embedding generation fails."""

    error_code: str = "embedding_error"


@dataclass(slots=True)
class VectorStoreError(AppError):
    """Raised when a Pinecone vector store operation fails."""

    error_code: str = "vector_store_error"

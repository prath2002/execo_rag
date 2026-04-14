"""Document source and ingestion input models."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator

from .common import ExecoBaseModel


class DocumentType(str, Enum):
    """Supported top-level document types."""

    SHARE_PURCHASE_AGREEMENT = "share_purchase_agreement"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    """Supported document source types."""

    LOCAL_FILE = "local_file"
    API_UPLOAD = "api_upload"
    REMOTE_URL = "remote_url"


class DocumentSource(ExecoBaseModel):
    """Location and identity details for a source document."""

    source_type: SourceType
    path: Path
    file_name: str
    mime_type: str = "application/pdf"
    file_hash: str | None = None

    @field_validator("path", mode="before")
    @classmethod
    def coerce_path(cls, value: str | Path) -> Path:
        """Coerce string input into a `Path` instance."""

        return Path(value)


class DocumentInput(ExecoBaseModel):
    """Top-level input model for ingestion workflows."""

    document_id: str = Field(min_length=1)
    source: DocumentSource
    document_type: DocumentType = DocumentType.UNKNOWN

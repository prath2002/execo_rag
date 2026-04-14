"""Schemas for ingestion endpoints."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator

from execo_rag.models.common import ExecoBaseModel
from execo_rag.models.document import DocumentType


class IngestRequest(ExecoBaseModel):
    """API request payload for document ingestion."""

    document_id: str = Field(min_length=1)
    file_path: Path = Field(default=Path("src/documents/POC_TEST_SPA.pdf"))
    document_type: DocumentType = DocumentType.SHARE_PURCHASE_AGREEMENT

    @field_validator("file_path", mode="before")
    @classmethod
    def coerce_path(cls, value: str | Path) -> Path:
        """Accept string file paths in API requests."""

        return Path(value)


class IngestResponse(ExecoBaseModel):
    """API response returned after document ingestion submission."""

    request_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    message: str = Field(min_length=1)

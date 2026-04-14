"""Models for extracted and cleaned PDF text."""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from .common import ExecoBaseModel
from .metadata import SectionType


class ExtractionBlockType(str, Enum):
    """Recognized block types produced by PDF extraction."""

    TITLE = "title"
    HEADER = "header"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FOOTER = "footer"
    UNKNOWN = "unknown"


class ExtractedBlock(ExecoBaseModel):
    """A page-scoped text block extracted from a PDF."""

    block_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    block_type: ExtractionBlockType = ExtractionBlockType.UNKNOWN
    text: str = Field(min_length=1)
    bbox: tuple[float, float, float, float] | None = None


class ExtractedPage(ExecoBaseModel):
    """Raw extracted content for a single PDF page."""

    page_number: int = Field(ge=1)
    raw_text: str = ""
    blocks: list[ExtractedBlock] = Field(default_factory=list)


class CleanedPage(ExecoBaseModel):
    """Normalized text content for a single page after cleanup."""

    page_number: int = Field(ge=1)
    cleaned_text: str = ""
    removed_noise_fragments: list[str] = Field(default_factory=list)


class SectionSegment(ExecoBaseModel):
    """Section-labeled text segment used as chunking input."""

    segment_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    section: SectionType = SectionType.GENERAL
    subsection: str | None = None
    text: str = Field(min_length=1)
    order_index: int = Field(ge=0)


class ExtractedDocument(ExecoBaseModel):
    """Full extracted document payload."""

    document_id: str = Field(min_length=1)
    total_pages: int = Field(ge=1)
    pages: list[ExtractedPage] = Field(default_factory=list)
    extractor_used: str | None = None

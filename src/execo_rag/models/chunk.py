"""Chunk and chunk metadata models."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import Field, model_validator

from .common import ExecoBaseModel
from .metadata import SectionType


class ChunkFlags(ExecoBaseModel):
    """Boolean flags that simplify Pinecone metadata filtering."""

    has_escrow: bool = False
    has_indemnification: bool = False
    has_purchase_price: bool = False
    has_working_capital: bool = False


class ChunkMetadata(ExecoBaseModel):
    """Flattened metadata stored alongside each vector record."""

    document_id: str = Field(min_length=1)
    document_type: str | None = None
    effective_date: date | None = None
    buyer: str | None = None
    company_target: str | None = None
    seller: str | None = None
    shares_transacted: str | None = None
    cash_purchase_price: Decimal | None = None
    escrow_agent: str | None = None
    escrow_amount: Decimal | None = None
    target_working_capital: Decimal | None = None
    indemnification_de_minimis_amount: Decimal | None = None
    indemnification_basket_amount: Decimal | None = None
    indemnification_cap_amount: Decimal | None = None
    governing_law: str | None = None
    section: SectionType = SectionType.GENERAL
    subsection: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    has_escrow: bool = False
    has_indemnification: bool = False
    has_purchase_price: bool = False
    has_working_capital: bool = False
    chunk_text: str | None = None

    @model_validator(mode="after")
    def validate_page_range(self) -> "ChunkMetadata":
        """Ensure page ranges are internally consistent."""

        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class RawChunk(ExecoBaseModel):
    """Chunk created by the chunking service before metadata enrichment."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    section: SectionType = SectionType.GENERAL
    subsection: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    token_count: int = Field(ge=1)


class EnrichedChunk(RawChunk):
    """Chunk enriched with normalized, filterable metadata."""

    metadata: ChunkMetadata


class ValidatedChunk(EnrichedChunk):
    """Chunk that has passed schema and business validation checks."""

    is_valid: bool = True

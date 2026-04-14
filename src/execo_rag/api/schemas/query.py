"""Schemas for retrieval and metadata-filtered query endpoints."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import Field

from execo_rag.models.common import ExecoBaseModel
from execo_rag.models.metadata import SectionType


class QueryFilters(ExecoBaseModel):
    """Optional metadata filters applied to semantic retrieval."""

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
    section: SectionType | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)


class QueryRequest(ExecoBaseModel):
    """API request for semantic search with optional metadata filters."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)
    filters: QueryFilters | None = None


class QueryResultChunk(ExecoBaseModel):
    """Returned chunk snippet from retrieval."""

    chunk_id: str = Field(min_length=1)
    score: float
    text: str = Field(min_length=1)
    section: SectionType
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class QueryResponse(ExecoBaseModel):
    """API response for semantic retrieval results."""

    request_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    results: list[QueryResultChunk] = Field(default_factory=list)

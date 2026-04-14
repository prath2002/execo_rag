"""Document metadata and evidence models."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Generic, TypeVar

from pydantic import Field

from .common import ExecoBaseModel

T = TypeVar("T")


class SectionType(str, Enum):
    """Supported logical sections for chunking and filtering."""

    PREAMBLE = "preamble"
    DEFINITIONS = "definitions"
    PURCHASE_PRICE = "purchase_price"
    ESCROW = "escrow"
    WORKING_CAPITAL = "working_capital"
    INDEMNIFICATION = "indemnification"
    GOVERNING_LAW = "governing_law"
    GENERAL = "general"


class MetadataEvidence(ExecoBaseModel):
    """Evidence payload that supports traceability for an extracted field."""

    page_number: int = Field(ge=1)
    snippet: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class MetadataField(ExecoBaseModel, Generic[T]):
    """Typed extracted metadata field with optional evidence."""

    value: T | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: MetadataEvidence | None = None


class DocumentMetadata(ExecoBaseModel):
    """Structured document metadata for the supported legal schema."""

    document_id: str = Field(min_length=1)
    document_type: MetadataField[str] = Field(default_factory=MetadataField[str])
    effective_date: MetadataField[date] = Field(default_factory=MetadataField[date])
    buyer: MetadataField[str] = Field(default_factory=MetadataField[str])
    company_target: MetadataField[str] = Field(default_factory=MetadataField[str])
    seller: MetadataField[str] = Field(default_factory=MetadataField[str])
    shares_transacted: MetadataField[str] = Field(default_factory=MetadataField[str])
    cash_purchase_price: MetadataField[Decimal] = Field(default_factory=MetadataField[Decimal])
    escrow_agent: MetadataField[str] = Field(default_factory=MetadataField[str])
    escrow_amount: MetadataField[Decimal] = Field(default_factory=MetadataField[Decimal])
    target_working_capital: MetadataField[Decimal] = Field(
        default_factory=MetadataField[Decimal]
    )
    indemnification_de_minimis_amount: MetadataField[Decimal] = Field(
        default_factory=MetadataField[Decimal]
    )
    indemnification_basket_amount: MetadataField[Decimal] = Field(
        default_factory=MetadataField[Decimal]
    )
    indemnification_cap_amount: MetadataField[Decimal] = Field(
        default_factory=MetadataField[Decimal]
    )
    governing_law: MetadataField[str] = Field(default_factory=MetadataField[str])

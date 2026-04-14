"""Metadata normalization service.

Converts raw extracted :class:`DocumentMetadata` field values into
canonical, filter-safe representations before Pinecone indexing.

Normalization rules:
- String fields: NFC → control-char removal → quote/dash normalization → strip
- Date fields:   Ensure stored as :class:`~datetime.date` (parsed if still a string)
- Decimal fields: Ensure stored as :class:`~decimal.Decimal` (parsed if still a string)
- Governing law: Titlecase
- Party names:   Whitespace-collapsed, leading/trailing punctuation stripped
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal, InvalidOperation

from execo_rag.models.metadata import DocumentMetadata, MetadataField
from execo_rag.utils.dates import parse_date
from execo_rag.utils.money import parse_money
from execo_rag.utils.text import clean_field_value, normalize_to_lowercase, title_case_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field-level normalizers
# ---------------------------------------------------------------------------


def _normalize_str_field(field: MetadataField[str]) -> MetadataField[str]:
    """Normalize a string metadata field in place."""
    if field.value is None:
        return field
    normalized = clean_field_value(str(field.value))
    if not normalized:
        return MetadataField(value=None, confidence=field.confidence, evidence=field.evidence)
    return MetadataField(value=normalized, confidence=field.confidence, evidence=field.evidence)


def _normalize_party_name(field: MetadataField[str]) -> MetadataField[str]:
    """Normalize a party name (buyer, seller, company_target, escrow_agent)."""
    if field.value is None:
        return field
    normalized = clean_field_value(str(field.value))
    # Remove trailing legal suffixes that may include quotes from regex capture
    normalized = normalized.strip("\"'")
    if not normalized:
        return MetadataField(value=None, confidence=field.confidence, evidence=field.evidence)
    return MetadataField(value=normalized, confidence=field.confidence, evidence=field.evidence)


def _normalize_governing_law(field: MetadataField[str]) -> MetadataField[str]:
    """Normalize governing law to titlecase (e.g. 'delaware' → 'Delaware')."""
    if field.value is None:
        return field
    cleaned = clean_field_value(str(field.value))
    # Titlecase state/jurisdiction name
    normalized = " ".join(word.capitalize() for word in cleaned.split())
    return MetadataField(value=normalized, confidence=field.confidence, evidence=field.evidence)


def _normalize_document_type(field: MetadataField[str]) -> MetadataField[str]:
    """Normalize document type to canonical lowercase form."""
    if field.value is None:
        return field
    raw = clean_field_value(str(field.value)).lower()
    # Canonical mapping for known SPA variants
    _canonical: dict[str, str] = {
        "share purchase agreement": "share_purchase_agreement",
        "stock purchase agreement": "share_purchase_agreement",
        "equity purchase agreement": "share_purchase_agreement",
        "spa": "share_purchase_agreement",
    }
    canonical = _canonical.get(normalize_to_lowercase(raw), normalize_to_lowercase(raw))
    return MetadataField(value=canonical, confidence=field.confidence, evidence=field.evidence)


def _normalize_date_field(field: MetadataField[date]) -> MetadataField[date]:
    """Ensure date fields are stored as :class:`~datetime.date` objects."""
    if field.value is None:
        return field
    if isinstance(field.value, date):
        return field
    # Value is still a string from extraction
    parsed = parse_date(str(field.value))
    if parsed is None:
        logger.warning(
            "Could not normalize date field value",
            extra={"extra_data": {"raw_value": str(field.value)}},
        )
        return MetadataField(value=None, confidence=0.0, evidence=field.evidence)
    return MetadataField(value=parsed, confidence=field.confidence, evidence=field.evidence)


def _normalize_decimal_field(field: MetadataField[Decimal]) -> MetadataField[Decimal]:
    """Ensure decimal fields are stored as :class:`~decimal.Decimal` objects."""
    if field.value is None:
        return field
    if isinstance(field.value, Decimal):
        return field
    # Value is still a string — attempt parse
    raw = str(field.value).replace(",", "").strip()
    try:
        parsed = Decimal(raw)
    except InvalidOperation:
        # Try money parser as last resort
        parsed = parse_money(raw)  # type: ignore[assignment]
    if parsed is None:
        logger.warning(
            "Could not normalize decimal field value",
            extra={"extra_data": {"raw_value": raw}},
        )
        return MetadataField(value=None, confidence=0.0, evidence=field.evidence)
    return MetadataField(value=parsed, confidence=field.confidence, evidence=field.evidence)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_metadata(metadata: DocumentMetadata) -> DocumentMetadata:
    """Apply canonical normalization to all fields in a :class:`DocumentMetadata`.

    This service is idempotent — calling it multiple times on the same object
    is safe and produces the same result.

    Args:
        metadata: Partially or fully extracted document metadata.

    Returns:
        New :class:`DocumentMetadata` instance with all fields normalized.
    """
    logger.debug(
        "Normalizing document metadata",
        extra={"extra_data": {"document_id": metadata.document_id}},
    )

    normalized = DocumentMetadata(
        document_id=metadata.document_id,
        document_type=_normalize_document_type(metadata.document_type),
        effective_date=_normalize_date_field(metadata.effective_date),
        buyer=_normalize_party_name(metadata.buyer),
        company_target=_normalize_party_name(metadata.company_target),
        seller=_normalize_party_name(metadata.seller),
        shares_transacted=_normalize_str_field(metadata.shares_transacted),
        cash_purchase_price=_normalize_decimal_field(metadata.cash_purchase_price),
        escrow_agent=_normalize_party_name(metadata.escrow_agent),
        escrow_amount=_normalize_decimal_field(metadata.escrow_amount),
        target_working_capital=_normalize_decimal_field(metadata.target_working_capital),
        indemnification_de_minimis_amount=_normalize_decimal_field(
            metadata.indemnification_de_minimis_amount
        ),
        indemnification_basket_amount=_normalize_decimal_field(
            metadata.indemnification_basket_amount
        ),
        indemnification_cap_amount=_normalize_decimal_field(
            metadata.indemnification_cap_amount
        ),
        governing_law=_normalize_governing_law(metadata.governing_law),
    )

    # Log summary
    populated = sum(
        1
        for name in [
            "document_type", "effective_date", "buyer", "company_target", "seller",
            "shares_transacted", "cash_purchase_price", "escrow_agent", "escrow_amount",
            "target_working_capital", "indemnification_de_minimis_amount",
            "indemnification_basket_amount", "indemnification_cap_amount", "governing_law",
        ]
        if getattr(normalized, name).value is not None
    )
    logger.info(
        "Metadata normalization complete",
        extra={
            "extra_data": {
                "document_id": metadata.document_id,
                "populated_fields": populated,
            }
        },
    )

    return normalized

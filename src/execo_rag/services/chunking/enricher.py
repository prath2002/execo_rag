"""Chunk metadata enricher.

Copies document-level :class:`DocumentMetadata` onto each :class:`RawChunk`
and derives boolean flags for Pinecone metadata filtering.

Produces :class:`EnrichedChunk` objects that are ready for validation and
then embedding.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal

from execo_rag.models.chunk import ChunkMetadata, EnrichedChunk, RawChunk
from execo_rag.models.metadata import DocumentMetadata, SectionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boolean flag derivation
# ---------------------------------------------------------------------------

# Sections that trigger the has_* flags
_ESCROW_SECTIONS = {SectionType.ESCROW}
_INDEMNIFICATION_SECTIONS = {SectionType.INDEMNIFICATION}
_PURCHASE_PRICE_SECTIONS = {SectionType.PURCHASE_PRICE}
_WORKING_CAPITAL_SECTIONS = {SectionType.WORKING_CAPITAL}


def _derive_flags(
    section: SectionType,
    metadata: DocumentMetadata,
) -> tuple[bool, bool, bool, bool]:
    """Derive has_* boolean flags for Pinecone filtering.

    A flag is True if either:
    - The chunk's section corresponds to that topic, OR
    - The document has a non-null value for that field.

    Returns:
        ``(has_escrow, has_indemnification, has_purchase_price, has_working_capital)``
    """
    has_escrow = (
        section in _ESCROW_SECTIONS
        or metadata.escrow_amount.value is not None
        or metadata.escrow_agent.value is not None
    )
    has_indemnification = (
        section in _INDEMNIFICATION_SECTIONS
        or metadata.indemnification_cap_amount.value is not None
        or metadata.indemnification_basket_amount.value is not None
        or metadata.indemnification_de_minimis_amount.value is not None
    )
    has_purchase_price = (
        section in _PURCHASE_PRICE_SECTIONS
        or metadata.cash_purchase_price.value is not None
    )
    has_working_capital = (
        section in _WORKING_CAPITAL_SECTIONS
        or metadata.target_working_capital.value is not None
    )
    return has_escrow, has_indemnification, has_purchase_price, has_working_capital


# ---------------------------------------------------------------------------
# Value extraction helpers
# ---------------------------------------------------------------------------


def _str_or_none(field_value: object) -> str | None:
    if field_value is None:
        return None
    return str(field_value)


def _decimal_or_none(field_value: object) -> Decimal | None:
    if field_value is None:
        return None
    if isinstance(field_value, Decimal):
        return field_value
    try:
        return Decimal(str(field_value))
    except Exception:
        return None


def _date_or_none(field_value: object) -> date | None:
    if isinstance(field_value, date):
        return field_value
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_chunks(
    chunks: list[RawChunk],
    metadata: DocumentMetadata,
) -> list[EnrichedChunk]:
    """Attach document metadata to each raw chunk.

    Args:
        chunks: Raw chunks from the hybrid chunker.
        metadata: Normalized and verified :class:`DocumentMetadata`.

    Returns:
        List of :class:`EnrichedChunk` objects in the same order as input.
    """
    enriched: list[EnrichedChunk] = []

    for chunk in chunks:
        has_escrow, has_indemnification, has_purchase_price, has_working_capital = (
            _derive_flags(chunk.section, metadata)
        )

        chunk_meta = ChunkMetadata(
            document_id=metadata.document_id,
            # Flattened document-level fields
            document_type=_str_or_none(metadata.document_type.value),
            effective_date=_date_or_none(metadata.effective_date.value),
            buyer=_str_or_none(metadata.buyer.value),
            company_target=_str_or_none(metadata.company_target.value),
            seller=_str_or_none(metadata.seller.value),
            shares_transacted=_str_or_none(metadata.shares_transacted.value),
            cash_purchase_price=_decimal_or_none(metadata.cash_purchase_price.value),
            escrow_agent=_str_or_none(metadata.escrow_agent.value),
            escrow_amount=_decimal_or_none(metadata.escrow_amount.value),
            target_working_capital=_decimal_or_none(metadata.target_working_capital.value),
            indemnification_de_minimis_amount=_decimal_or_none(
                metadata.indemnification_de_minimis_amount.value
            ),
            indemnification_basket_amount=_decimal_or_none(
                metadata.indemnification_basket_amount.value
            ),
            indemnification_cap_amount=_decimal_or_none(
                metadata.indemnification_cap_amount.value
            ),
            governing_law=_str_or_none(metadata.governing_law.value),
            # Chunk-level positional fields
            section=chunk.section,
            subsection=chunk.subsection,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            # Boolean filter flags
            has_escrow=has_escrow,
            has_indemnification=has_indemnification,
            has_purchase_price=has_purchase_price,
            has_working_capital=has_working_capital,
            # Store the chunk text in metadata for retrieval display
            chunk_text=chunk.text,
        )

        enriched.append(
            EnrichedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                section=chunk.section,
                subsection=chunk.subsection,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                token_count=chunk.token_count,
                metadata=chunk_meta,
            )
        )

    logger.info(
        "Chunk enrichment complete",
        extra={
            "extra_data": {
                "document_id": metadata.document_id,
                "total_chunks": len(enriched),
            }
        },
    )

    return enriched

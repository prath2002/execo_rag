"""Chunk validation service.

Validates each :class:`EnrichedChunk` against required fields and
metadata consistency rules before the chunk is submitted for embedding
and Pinecone indexing.

Validation checks:
  1. ``chunk_id`` and ``document_id`` are non-empty strings.
  2. ``text`` is non-empty and meets a minimum length.
  3. ``token_count`` is positive.
  4. ``page_start`` and ``page_end`` are positive integers with page_end >= page_start.
  5. ``metadata.document_id`` matches the chunk's ``document_id``.
  6. ``metadata.section`` is a recognized :class:`SectionType`.
  7. Monetary fields are ``None`` or a positive ``Decimal``.

Returns :class:`ValidatedChunk` objects; invalid chunks are flagged with
``is_valid=False`` so the caller can decide whether to skip or fail.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from execo_rag.models.chunk import EnrichedChunk, ValidatedChunk
from execo_rag.models.metadata import SectionType
from execo_rag.utils.exceptions import ChunkingError

logger = logging.getLogger(__name__)

# Minimum meaningful chunk text length (characters)
_MIN_TEXT_CHARS = 20

# Monetary fields stored as Decimal in ChunkMetadata
_MONETARY_FIELDS = (
    "cash_purchase_price",
    "escrow_amount",
    "target_working_capital",
    "indemnification_de_minimis_amount",
    "indemnification_basket_amount",
    "indemnification_cap_amount",
)


def _validate_one(chunk: EnrichedChunk) -> tuple[bool, list[str]]:
    """Validate a single enriched chunk.

    Returns:
        ``(is_valid, list_of_failure_reasons)``
    """
    reasons: list[str] = []

    # --- Identity checks ---
    if not chunk.chunk_id or not chunk.chunk_id.strip():
        reasons.append("chunk_id is empty")

    if not chunk.document_id or not chunk.document_id.strip():
        reasons.append("document_id is empty")

    # --- Text checks ---
    if not chunk.text or len(chunk.text.strip()) < _MIN_TEXT_CHARS:
        reasons.append(f"text too short (< {_MIN_TEXT_CHARS} chars)")

    # --- Token count ---
    if chunk.token_count < 1:
        reasons.append(f"token_count={chunk.token_count} must be >= 1")

    # --- Page range ---
    if chunk.page_start < 1:
        reasons.append(f"page_start={chunk.page_start} must be >= 1")

    if chunk.page_end < 1:
        reasons.append(f"page_end={chunk.page_end} must be >= 1")

    if chunk.page_end < chunk.page_start:
        reasons.append(
            f"page_end={chunk.page_end} < page_start={chunk.page_start}"
        )

    # --- Metadata consistency ---
    if chunk.metadata.document_id != chunk.document_id:
        reasons.append(
            f"metadata.document_id '{chunk.metadata.document_id}' "
            f"!= chunk.document_id '{chunk.document_id}'"
        )

    if not isinstance(chunk.metadata.section, SectionType):
        reasons.append(f"metadata.section is not a SectionType: {chunk.metadata.section!r}")

    if chunk.section != chunk.metadata.section:
        reasons.append(
            f"chunk.section '{chunk.section}' != metadata.section '{chunk.metadata.section}'"
        )

    # --- Monetary field sanity ---
    for field_name in _MONETARY_FIELDS:
        value = getattr(chunk.metadata, field_name, None)
        if value is not None:
            if not isinstance(value, Decimal):
                reasons.append(f"{field_name} must be Decimal, got {type(value).__name__}")
            elif value <= Decimal("0"):
                reasons.append(f"{field_name}={value} must be positive")

    return len(reasons) == 0, reasons


def validate_chunks(
    enriched_chunks: list[EnrichedChunk],
    raise_on_invalid: bool = False,
) -> list[ValidatedChunk]:
    """Validate all enriched chunks and return :class:`ValidatedChunk` objects.

    Args:
        enriched_chunks: List from the enrichment service.
        raise_on_invalid: If ``True``, raise :class:`ChunkingError` when any
                          chunk fails validation (default: ``False``, flag only).

    Returns:
        List of :class:`ValidatedChunk` with ``is_valid`` set appropriately.

    Raises:
        ChunkingError: If ``raise_on_invalid=True`` and any chunk is invalid.
    """
    validated: list[ValidatedChunk] = []
    invalid_ids: list[str] = []

    for chunk in enriched_chunks:
        is_valid, reasons = _validate_one(chunk)

        if not is_valid:
            invalid_ids.append(chunk.chunk_id)
            logger.warning(
                "Chunk failed validation",
                extra={
                    "extra_data": {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "reasons": reasons,
                    }
                },
            )

        validated.append(
            ValidatedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                section=chunk.section,
                subsection=chunk.subsection,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
                is_valid=is_valid,
            )
        )

    valid_count = len(validated) - len(invalid_ids)
    logger.info(
        "Chunk validation complete",
        extra={
            "extra_data": {
                "total": len(validated),
                "valid": valid_count,
                "invalid": len(invalid_ids),
            }
        },
    )

    if raise_on_invalid and invalid_ids:
        raise ChunkingError(
            message=f"{len(invalid_ids)} chunk(s) failed validation.",
            error_code="chunk_validation_failed",
            details={"invalid_chunk_ids": invalid_ids},
        )

    return validated

"""Metadata verification service.

Validates that each extracted field:
  1. Has a non-null value.
  2. Meets the minimum confidence threshold.
  3. Has a supporting evidence snippet that can be located in the source pages.
  4. Passes field-specific sanity checks (e.g. date in plausible range, amounts > 0).

Returns a :class:`VerificationReport` documenting which fields pass, fail,
and why — without mutating the metadata object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from execo_rag.models.extraction import CleanedPage
from execo_rag.models.metadata import DocumentMetadata, MetadataField

logger = logging.getLogger(__name__)

# Minimum confidence to consider a field "verified"
_MIN_CONFIDENCE = 0.60

# Plausible date range for SPA documents
_DATE_MIN = date(1990, 1, 1)
_DATE_MAX = date(2100, 12, 31)

# Maximum reasonable monetary amount ($100 billion)
_MAX_AMOUNT = Decimal("100_000_000_000")


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass
class FieldVerificationResult:
    """Verification result for a single metadata field."""

    field_name: str
    passed: bool
    value: object = None
    confidence: float = 0.0
    failure_reason: str | None = None


@dataclass
class VerificationReport:
    """Aggregate verification report for a document's metadata."""

    document_id: str
    results: list[FieldVerificationResult] = field(default_factory=list)

    @property
    def passed_fields(self) -> list[str]:
        """Field names that passed verification."""
        return [r.field_name for r in self.results if r.passed]

    @property
    def failed_fields(self) -> list[str]:
        """Field names that failed verification."""
        return [r.field_name for r in self.results if not r.passed]

    @property
    def all_critical_passed(self) -> bool:
        """Return True if all critical fields (document_type, buyer, seller) pass."""
        critical = {"document_type", "buyer", "seller"}
        passed_set = set(self.passed_fields)
        return critical.issubset(passed_set)

    def summary(self) -> dict[str, object]:
        """Return a dict summary suitable for structured logging."""
        return {
            "document_id": self.document_id,
            "passed": len(self.passed_fields),
            "failed": len(self.failed_fields),
            "all_critical_passed": self.all_critical_passed,
            "failed_fields": self.failed_fields,
        }


# ---------------------------------------------------------------------------
# Field-level verifiers
# ---------------------------------------------------------------------------


def _verify_str_field(
    name: str,
    field: MetadataField,  # type: ignore[type-arg]
    pages_text: list[tuple[int, str]],
) -> FieldVerificationResult:
    """Verify a string field: non-null, min confidence, snippet traceable."""
    if field.value is None:
        return FieldVerificationResult(name, False, failure_reason="value is null")

    if field.confidence < _MIN_CONFIDENCE:
        return FieldVerificationResult(
            name,
            False,
            value=field.value,
            confidence=field.confidence,
            failure_reason=f"confidence {field.confidence:.2f} below threshold {_MIN_CONFIDENCE}",
        )

    value_str = str(field.value).lower().strip()
    if len(value_str) < 2:
        return FieldVerificationResult(
            name, False, value=field.value, confidence=field.confidence,
            failure_reason="value too short"
        )

    # Evidence check: look for value in any page text
    found_in_text = any(value_str in text.lower() for _, text in pages_text)
    if not found_in_text and field.evidence:
        # Check if evidence snippet itself is present
        snippet = field.evidence.snippet.lower().strip()[:60]
        found_in_text = any(snippet in text.lower() for _, text in pages_text)

    if not found_in_text:
        return FieldVerificationResult(
            name,
            False,
            value=field.value,
            confidence=field.confidence,
            failure_reason="value not traceable to source text",
        )

    return FieldVerificationResult(name, True, value=field.value, confidence=field.confidence)


def _verify_date_field(
    name: str,
    field: MetadataField,  # type: ignore[type-arg]
) -> FieldVerificationResult:
    """Verify a date field: non-null, plausible range, min confidence."""
    if field.value is None:
        return FieldVerificationResult(name, False, failure_reason="value is null")

    if field.confidence < _MIN_CONFIDENCE:
        return FieldVerificationResult(
            name, False, value=field.value, confidence=field.confidence,
            failure_reason=f"confidence {field.confidence:.2f} below threshold",
        )

    if not isinstance(field.value, date):
        return FieldVerificationResult(
            name, False, value=field.value, confidence=field.confidence,
            failure_reason="value is not a date object",
        )

    if not (_DATE_MIN <= field.value <= _DATE_MAX):
        return FieldVerificationResult(
            name, False, value=field.value, confidence=field.confidence,
            failure_reason=f"date {field.value} outside plausible range",
        )

    return FieldVerificationResult(name, True, value=field.value, confidence=field.confidence)


def _verify_decimal_field(
    name: str,
    field: MetadataField,  # type: ignore[type-arg]
) -> FieldVerificationResult:
    """Verify a monetary field: non-null, positive, within plausible range."""
    if field.value is None:
        return FieldVerificationResult(name, False, failure_reason="value is null")

    if field.confidence < _MIN_CONFIDENCE:
        return FieldVerificationResult(
            name, False, value=field.value, confidence=field.confidence,
            failure_reason=f"confidence {field.confidence:.2f} below threshold",
        )

    if not isinstance(field.value, Decimal):
        return FieldVerificationResult(
            name, False, value=field.value, confidence=field.confidence,
            failure_reason="value is not a Decimal",
        )

    if field.value <= Decimal("0"):
        return FieldVerificationResult(
            name, False, value=field.value, confidence=field.confidence,
            failure_reason="monetary amount must be positive",
        )

    if field.value > _MAX_AMOUNT:
        return FieldVerificationResult(
            name, False, value=field.value, confidence=field.confidence,
            failure_reason=f"amount {field.value} exceeds plausible maximum",
        )

    return FieldVerificationResult(name, True, value=field.value, confidence=field.confidence)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_metadata(
    metadata: DocumentMetadata,
    cleaned_pages: list[CleanedPage],
) -> VerificationReport:
    """Verify all metadata fields against confidence thresholds and source evidence.

    Args:
        metadata: Normalized :class:`DocumentMetadata` to verify.
        cleaned_pages: Source pages used for evidence traceability checks.

    Returns:
        :class:`VerificationReport` with per-field results.
    """
    pages_text = [(p.page_number, p.cleaned_text) for p in cleaned_pages]
    report = VerificationReport(document_id=metadata.document_id)

    # --- String fields ---
    for name in ("document_type", "buyer", "company_target", "seller",
                  "shares_transacted", "escrow_agent", "governing_law"):
        result = _verify_str_field(name, getattr(metadata, name), pages_text)
        report.results.append(result)

    # --- Date fields ---
    result = _verify_date_field("effective_date", metadata.effective_date)
    report.results.append(result)

    # --- Decimal / monetary fields ---
    for name in (
        "cash_purchase_price", "escrow_amount", "target_working_capital",
        "indemnification_de_minimis_amount", "indemnification_basket_amount",
        "indemnification_cap_amount",
    ):
        result = _verify_decimal_field(name, getattr(metadata, name))
        report.results.append(result)

    # Consistency check: escrow_amount should be less than purchase price
    if (
        metadata.cash_purchase_price.value is not None
        and metadata.escrow_amount.value is not None
        and isinstance(metadata.cash_purchase_price.value, Decimal)
        and isinstance(metadata.escrow_amount.value, Decimal)
    ):
        if metadata.escrow_amount.value >= metadata.cash_purchase_price.value:
            for r in report.results:
                if r.field_name == "escrow_amount":
                    r.passed = False
                    r.failure_reason = "escrow_amount >= cash_purchase_price (implausible)"

    summary = report.summary()
    logger.info(
        "Metadata verification complete",
        extra={"extra_data": summary},
    )

    if report.failed_fields:
        logger.warning(
            "Some metadata fields failed verification",
            extra={
                "extra_data": {
                    "document_id": metadata.document_id,
                    "failed_fields": report.failed_fields,
                }
            },
        )

    return report

"""Pinecone metadata filter builder.

Constructs Pinecone-compatible filter dictionaries from typed filter parameters.
Pinecone uses a MongoDB-style filter syntax:

  Single field:   {"field": {"$eq": "value"}}
  Compound AND:   {"$and": [{"f1": {...}}, {"f2": {...}}]}
  Compound OR:    {"$or": [{"f1": {...}}, {"f2": {...}}]}
  Numeric range:  {"amount": {"$gte": 1000000}}

Reference: https://docs.pinecone.io/guides/data/filter-with-metadata
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Filter = dict[str, Any]


# ---------------------------------------------------------------------------
# Primitive filter constructors
# ---------------------------------------------------------------------------


def eq(field: str, value: str | int | float | bool) -> Filter:
    """Exact-match filter: ``field == value``.

    Args:
        field: Metadata field name.
        value: Value to match.

    Returns:
        Pinecone filter dict.
    """
    return {field: {"$eq": value}}


def ne(field: str, value: str | int | float | bool) -> Filter:
    """Not-equal filter: ``field != value``."""
    return {field: {"$ne": value}}


def gte(field: str, value: int | float) -> Filter:
    """Greater-than-or-equal filter: ``field >= value``."""
    return {field: {"$gte": value}}


def lte(field: str, value: int | float) -> Filter:
    """Less-than-or-equal filter: ``field <= value``."""
    return {field: {"$lte": value}}


def in_values(field: str, values: list[str | int | float]) -> Filter:
    """Membership filter: ``field in values``."""
    return {field: {"$in": values}}


def and_(*filters: Filter) -> Filter:
    """Logical AND of multiple filters."""
    active = [f for f in filters if f]
    if not active:
        return {}
    if len(active) == 1:
        return active[0]
    return {"$and": active}


def or_(*filters: Filter) -> Filter:
    """Logical OR of multiple filters."""
    active = [f for f in filters if f]
    if not active:
        return {}
    if len(active) == 1:
        return active[0]
    return {"$or": active}


# ---------------------------------------------------------------------------
# High-level SPA filter builder
# ---------------------------------------------------------------------------


class SPAFilterBuilder:
    """Fluent builder for SPA Pinecone metadata filters.

    Usage::

        f = SPAFilterBuilder()
        f.by_document_id("doc_abc123")
        f.by_buyer("Acme Corp")
        f.by_section("indemnification")
        filter_dict = f.build()
    """

    def __init__(self) -> None:
        self._filters: list[Filter] = []

    def by_document_id(self, document_id: str) -> "SPAFilterBuilder":
        """Filter by exact document ID."""
        self._filters.append(eq("document_id", document_id))
        return self

    def by_document_type(self, document_type: str) -> "SPAFilterBuilder":
        """Filter by document type (e.g. 'share_purchase_agreement')."""
        self._filters.append(eq("document_type", document_type.lower()))
        return self

    def by_buyer(self, buyer: str) -> "SPAFilterBuilder":
        """Filter by buyer party name (case-insensitive)."""
        self._filters.append(eq("buyer", buyer.lower().strip()))
        return self

    def by_seller(self, seller: str) -> "SPAFilterBuilder":
        """Filter by seller party name."""
        self._filters.append(eq("seller", seller.lower().strip()))
        return self

    def by_company_target(self, company_target: str) -> "SPAFilterBuilder":
        """Filter by target company name."""
        self._filters.append(eq("company_target", company_target.lower().strip()))
        return self

    def by_governing_law(self, governing_law: str) -> "SPAFilterBuilder":
        """Filter by governing law jurisdiction."""
        self._filters.append(eq("governing_law", governing_law.strip()))
        return self

    def by_section(self, section: str) -> "SPAFilterBuilder":
        """Filter by SPA section (e.g. 'indemnification', 'escrow')."""
        self._filters.append(eq("section", section.lower().strip()))
        return self

    def by_effective_date(self, effective_date: date | str) -> "SPAFilterBuilder":
        """Filter by exact effective date (stored as ISO string in Pinecone)."""
        if isinstance(effective_date, date):
            date_str = effective_date.isoformat()
        else:
            date_str = str(effective_date)
        self._filters.append(eq("effective_date", date_str))
        return self

    def by_page_range(self, page_start: int, page_end: int) -> "SPAFilterBuilder":
        """Filter by overlapping page range."""
        # Chunk overlaps query range if: chunk.page_start <= page_end AND chunk.page_end >= page_start
        self._filters.append(lte("page_start", page_end))
        self._filters.append(gte("page_end", page_start))
        return self

    def has_escrow(self, value: bool = True) -> "SPAFilterBuilder":
        """Filter chunks with/without escrow content."""
        self._filters.append(eq("has_escrow", value))
        return self

    def has_indemnification(self, value: bool = True) -> "SPAFilterBuilder":
        """Filter chunks with/without indemnification content."""
        self._filters.append(eq("has_indemnification", value))
        return self

    def has_purchase_price(self, value: bool = True) -> "SPAFilterBuilder":
        """Filter chunks with/without purchase price content."""
        self._filters.append(eq("has_purchase_price", value))
        return self

    def has_working_capital(self, value: bool = True) -> "SPAFilterBuilder":
        """Filter chunks with/without working capital content."""
        self._filters.append(eq("has_working_capital", value))
        return self

    def build(self) -> Filter:
        """Build the final compound filter dict.

        Returns:
            Pinecone-compatible filter dict, or ``{}`` if no filters were added.
        """
        result = and_(*self._filters)
        logger.debug(
            "Built Pinecone filter",
            extra={"extra_data": {"filter": result}},
        )
        return result


# ---------------------------------------------------------------------------
# Convenience: build filter from API query params
# ---------------------------------------------------------------------------


def build_filter_from_params(
    document_id: str | None = None,
    document_type: str | None = None,
    buyer: str | None = None,
    seller: str | None = None,
    company_target: str | None = None,
    governing_law: str | None = None,
    section: str | None = None,
    effective_date: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    has_escrow: bool | None = None,
    has_indemnification: bool | None = None,
    has_purchase_price: bool | None = None,
    has_working_capital: bool | None = None,
) -> Filter:
    """Build a compound Pinecone filter from optional API-level query parameters.

    Only parameters that are not ``None`` are included in the filter.

    Returns:
        Pinecone-compatible filter dict (may be empty ``{}`` if no params given).
    """
    builder = SPAFilterBuilder()

    if document_id:
        builder.by_document_id(document_id)
    if document_type:
        builder.by_document_type(document_type)
    if buyer:
        builder.by_buyer(buyer)
    if seller:
        builder.by_seller(seller)
    if company_target:
        builder.by_company_target(company_target)
    if governing_law:
        builder.by_governing_law(governing_law)
    if section:
        builder.by_section(section)
    if effective_date:
        builder.by_effective_date(effective_date)
    if page_start is not None and page_end is not None:
        builder.by_page_range(page_start, page_end)
    if has_escrow is not None:
        builder.has_escrow(has_escrow)
    if has_indemnification is not None:
        builder.has_indemnification(has_indemnification)
    if has_purchase_price is not None:
        builder.has_purchase_price(has_purchase_price)
    if has_working_capital is not None:
        builder.has_working_capital(has_working_capital)

    return builder.build()

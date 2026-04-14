"""Date parsing and normalization utilities."""

from __future__ import annotations

import re
from datetime import date

# Try python-dateutil for fuzzy date parsing (installed as a dependency)
try:
    from dateutil import parser as dateutil_parser  # type: ignore[import-untyped]
    from dateutil.parser import ParserError  # type: ignore[import-untyped]

    _DATEUTIL_AVAILABLE = True
except ImportError:
    _DATEUTIL_AVAILABLE = False


# Explicit patterns for common US legal document date formats
_DATE_PATTERNS: list[tuple[str, str]] = [
    # "January 1, 2024"  /  "January 1st, 2024"
    (
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        "%B %d, %Y",
    ),
    # "Jan. 1, 2024"  /  "Jan 1, 2024"
    (
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b",
        "%b %d, %Y",
    ),
    # "1/1/2024"  /  "01/01/2024"
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "%m/%d/%Y"),
    # "2024-01-01"
    (r"\b\d{4}-\d{2}-\d{2}\b", "%Y-%m-%d"),
    # "1 January 2024"
    (
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b",
        "%d %B %Y",
    ),
]

# Ordinal suffix removal (1st → 1, 22nd → 22)
_ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)


def _strip_ordinals(text: str) -> str:
    """Remove ordinal suffixes so dateutil can parse '1st' as '1'."""
    return _ORDINAL_RE.sub(r"\1", text)


def parse_date(text: str) -> date | None:
    """Parse a date string into a Python :class:`~datetime.date`.

    Tries explicit patterns first, then falls back to ``dateutil.parser``.

    Args:
        text: Raw string that may contain a date.

    Returns:
        :class:`~datetime.date` or ``None`` if no parseable date is found.
    """
    if not text:
        return None

    text = text.strip()
    cleaned = _strip_ordinals(text)

    # Try dateutil first (most capable)
    if _DATEUTIL_AVAILABLE:
        try:
            parsed = dateutil_parser.parse(cleaned, fuzzy=False)
            return parsed.date()
        except (ParserError, ValueError, OverflowError):
            pass

    # Fallback: iterate explicit patterns
    from datetime import datetime

    for pattern_str, fmt in _DATE_PATTERNS:
        match = re.search(pattern_str, cleaned, re.IGNORECASE)
        if match:
            candidate = _strip_ordinals(match.group())
            # Remove comma before year if present and format doesn't expect it
            candidate = candidate.replace(",", "")
            try:
                return datetime.strptime(candidate, fmt.replace(",", "")).date()
            except ValueError:
                continue

    return None


def normalize_date_string(raw: str) -> str | None:
    """Parse a raw date string and return an ISO 8601 date string (YYYY-MM-DD).

    Args:
        raw: Raw date string from an extracted metadata field.

    Returns:
        ISO date string or ``None`` if parsing fails.
    """
    parsed = parse_date(raw)
    if parsed is None:
        return None
    return parsed.isoformat()


def extract_date_from_text(text: str) -> date | None:
    """Search a block of text for the first recognizable date.

    Args:
        text: Multi-sentence or paragraph text.

    Returns:
        First :class:`~datetime.date` found, or ``None``.
    """
    if _DATEUTIL_AVAILABLE:
        try:
            cleaned = _strip_ordinals(text)
            parsed = dateutil_parser.parse(cleaned, fuzzy=True)
            return parsed.date()
        except (ParserError, ValueError, OverflowError):
            pass

    # Fallback: scan explicit patterns
    from datetime import datetime

    for pattern_str, fmt in _DATE_PATTERNS:
        match = re.search(pattern_str, text, re.IGNORECASE)
        if match:
            candidate = _strip_ordinals(match.group()).replace(",", "")
            try:
                return datetime.strptime(candidate, fmt.replace(",", "")).date()
            except ValueError:
                continue

    return None

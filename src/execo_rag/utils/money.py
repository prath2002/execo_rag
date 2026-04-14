"""Money / currency parsing and normalization utilities."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


# Pattern for amounts like: $1,234,567.89  /  USD 1,234,567  /  1,234,567.00
_MONEY_RE = re.compile(
    r"""
    (?:USD\s*|US\$|\\$)?          # optional currency prefix
    (?P<amount>
        \d{1,3}                    # leading digits
        (?:,\d{3})*                # thousands groups
        (?:\.\d{1,4})?             # optional decimal
        |
        \d+(?:\.\d{1,4})?         # plain number
    )
    \s*(?:dollars?|USD)?           # optional suffix
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Words that should be converted to numeric values (for written-out amounts)
_WORD_TO_AMOUNT: dict[str, Decimal] = {
    "zero": Decimal("0"),
    "one": Decimal("1"),
    "million": Decimal("1000000"),
    "billion": Decimal("1000000000"),
}

_SCALE_WORDS: dict[str, Decimal] = {
    "thousand": Decimal("1000"),
    "million": Decimal("1_000_000"),
    "billion": Decimal("1_000_000_000"),
    "trillion": Decimal("1_000_000_000_000"),
}

_SCALE_RE = re.compile(
    r"(?P<amount>[\d,]+(?:\.\d+)?)\s*(?P<scale>thousand|million|billion|trillion)",
    re.IGNORECASE,
)


def parse_money(text: str) -> Decimal | None:
    """Parse a currency amount string into a :class:`~decimal.Decimal`.

    Handles:
    - ``$1,234,567.89``
    - ``USD 1,234,567``
    - ``1.5 million``
    - ``2.3 billion``

    Args:
        text: Raw string that may contain a monetary value.

    Returns:
        :class:`~decimal.Decimal` or ``None`` if no parseable amount found.
    """
    if not text:
        return None

    text = text.strip()

    # Try scaled words first (e.g. "1.5 million")
    scale_match = _SCALE_RE.search(text)
    if scale_match:
        raw_amount = scale_match.group("amount").replace(",", "")
        scale_word = scale_match.group("scale").lower()
        try:
            base = Decimal(raw_amount)
            multiplier = _SCALE_WORDS[scale_word]
            return base * multiplier
        except (InvalidOperation, KeyError):
            pass

    # Standard numeric pattern
    match = _MONEY_RE.search(text)
    if match:
        raw = match.group("amount").replace(",", "")
        try:
            return Decimal(raw)
        except InvalidOperation:
            return None

    return None


def format_money(amount: Decimal) -> str:
    """Format a :class:`~decimal.Decimal` as a US-style currency string.

    Args:
        amount: Monetary value.

    Returns:
        Formatted string, e.g. ``"$1,234,567.89"``.
    """
    return f"${amount:,.2f}"


def normalize_money_string(raw: str) -> str | None:
    """Parse a raw currency string and return a canonical string representation.

    Returns ``None`` if parsing fails.

    Args:
        raw: Raw monetary string from an extracted metadata field.

    Returns:
        Canonical string representation (e.g. ``"1234567.89"``) or ``None``.
    """
    value = parse_money(raw)
    if value is None:
        return None
    return str(value.normalize())

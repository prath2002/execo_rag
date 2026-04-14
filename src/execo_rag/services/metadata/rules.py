"""Regex extraction rules for the Share Purchase Agreement (SPA) metadata schema.

Each rule is a named callable that receives the full cleaned document text and
returns a tuple of (raw_value_string, page_number, snippet, confidence).
A rule returns ``None`` when no match is found.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class RuleMatch(NamedTuple):
    """Result of a single extraction rule."""

    raw_value: str
    page_number: int
    snippet: str
    confidence: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _search_pages(
    pages_text: list[tuple[int, str]],
    pattern: re.Pattern[str],
    group: int | str = 1,
    confidence: float = 0.85,
) -> RuleMatch | None:
    """Search each page for the first match of *pattern*.

    Args:
        pages_text: List of (page_number, cleaned_text) tuples.
        pattern: Compiled regex pattern.
        group: Named or numbered group to extract as the value.
        confidence: Confidence score for this rule.

    Returns:
        :class:`RuleMatch` or ``None`` if not found.
    """
    for page_num, text in pages_text:
        match = pattern.search(text)
        if match:
            try:
                raw = match.group(group).strip()
            except (IndexError, re.error):
                raw = match.group(0).strip()
            if raw:
                snippet = match.group(0).strip()[:200]
                return RuleMatch(
                    raw_value=raw,
                    page_number=page_num,
                    snippet=snippet,
                    confidence=confidence,
                )
    return None


def _search_all_pages(
    pages_text: list[tuple[int, str]],
    pattern: re.Pattern[str],
    group: int | str = 1,
    confidence: float = 0.85,
) -> list[RuleMatch]:
    """Return all matches across all pages (used for multi-occurrence fields)."""
    results: list[RuleMatch] = []
    for page_num, text in pages_text:
        for match in pattern.finditer(text):
            try:
                raw = match.group(group).strip()
            except (IndexError, re.error):
                raw = match.group(0).strip()
            if raw:
                snippet = match.group(0).strip()[:200]
                results.append(
                    RuleMatch(
                        raw_value=raw,
                        page_number=page_num,
                        snippet=snippet,
                        confidence=confidence,
                    )
                )
    return results


# ---------------------------------------------------------------------------
# Compiled patterns — SPA schema
# ---------------------------------------------------------------------------

# --- Document type ---
_DOC_TYPE_RE = re.compile(
    r"\b(Share\s+Purchase\s+Agreement|Stock\s+Purchase\s+Agreement|"
    r"Equity\s+Purchase\s+Agreement|SPA)\b",
    re.IGNORECASE,
)

# --- Effective date ---
# Matches: "dated as of January 1, 2024" / "entered into as of January 1, 2024"
_EFFECTIVE_DATE_RE = re.compile(
    r"(?:dated\s+as\s+of|entered\s+into\s+(?:as\s+of|on)|effective\s+as\s+of|"
    r"made\s+(?:and\s+entered\s+into\s+)?as\s+of)\s+"
    r"(?P<date>"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r")",
    re.IGNORECASE,
)

# --- Parties ---
# Buyer: "ABC Corp. (the 'Buyer')" / "as buyer ('Buyer')"
_BUYER_RE = re.compile(
    r"(?P<buyer>[A-Z][A-Za-z0-9\s,.'&()-]{2,80}?)"
    r"\s*[,(]\s*(?:the\s+)?[\"']?Buyer[\"']?\s*[,)]",
    re.IGNORECASE,
)

# Seller: "XYZ Inc. (the 'Seller')"
_SELLER_RE = re.compile(
    r"(?P<seller>[A-Z][A-Za-z0-9\s,.'&()-]{2,80}?)"
    r"\s*[,(]\s*(?:the\s+)?[\"']?Seller[\"']?\s*[,)]",
    re.IGNORECASE,
)

# Company target: "Company (the 'Company')" / "(the 'Target Company')"
_COMPANY_TARGET_RE = re.compile(
    r"(?P<company>[A-Z][A-Za-z0-9\s,.'&()-]{2,80}?)"
    r"\s*[,(]\s*(?:the\s+)?[\"']?(?:Company|Target Company|Target)[\"']?\s*[,)]",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Shared helpers for monetary amount matching
# ---------------------------------------------------------------------------

# Matches a dollar/currency amount: $1,234,567.89 or $50 million or 1,000,000 USD
# Used as the capture group in all monetary patterns.
_AMT = r"(?P<amount>[\d,]+(?:\.\d+)?(?:\s*(?:million|billion))?)"

# Allows up to 150 chars (no newline) between a keyword and the dollar sign.
# This handles common legal phrasing like "shall be", "equal to", "in the amount of".
_NEAR = r"[^\n]{0,150}?"

# --- Shares transacted ---
_SHARES_RE = re.compile(
    r"(?P<shares>[\d,]+(?:\.\d+)?)\s*"
    r"(?:shares|units|interests|ordinary\s+shares|common\s+shares|membership\s+interests)"
    r"(?:\s+of\s+[A-Z][\w\s]+)?",
    re.IGNORECASE,
)
_SHARES_ALL_PREAMBLE_RE = re.compile(
    r"(?:purchase\s+and\s+sale\s+of|sell\s+and\s+transfer)\s+"
    r"(?P<shares>[\d,]+(?:\.\d+)?)\s*"
    r"(?:shares|units|interests)",
    re.IGNORECASE,
)

# --- Cash purchase price ---
# Forward: "purchase price ... $X"  |  Reverse: "$X ... (the Purchase Price)"
_PURCHASE_PRICE_RE = re.compile(
    r"(?:aggregate\s+)?(?:cash\s+)?purchase\s+price" + _NEAR +
    r"\$\s*" + _AMT,
    re.IGNORECASE,
)
_CONSIDERATION_RE = re.compile(
    r"(?:total\s+)?(?:aggregate\s+)?consideration" + _NEAR +
    r"\$\s*" + _AMT,
    re.IGNORECASE,
)

# --- Escrow agent ---
_ESCROW_AGENT_RE = re.compile(
    r"(?P<agent>[A-Z][A-Za-z0-9\s,.'&()-]{2,80}?)"
    r"\s*[,(]\s*(?:the\s+)?[\"']?Escrow\s+Agent[\"']?\s*[,)]",
    re.IGNORECASE,
)

# --- Escrow amount ---
_ESCROW_AMOUNT_RE = re.compile(
    r"(?:escrow\s+(?:amount|fund|deposit|consideration|account))" + _NEAR +
    r"\$\s*" + _AMT,
    re.IGNORECASE,
)
# Reverse: "$X ... (the 'Escrow Amount')"
_ESCROW_AMOUNT_REVERSE_RE = re.compile(
    r"\$\s*(?P<amount>[\d,]+(?:\.\d+)?(?:\s*(?:million|billion))?)"
    r"[^\n]{0,150}?"
    r"""["']?(?:the\s+)?["']?Escrow\s+(?:Amount|Fund)["']?""",
    re.IGNORECASE,
)

# --- Target working capital ---
_WORKING_CAPITAL_RE = re.compile(
    r"(?:target|reference|required)\s+working\s+capital" + _NEAR +
    r"\$\s*" + _AMT,
    re.IGNORECASE,
)

# --- Indemnification thresholds ---
_INDEMNIFICATION_DE_MINIMIS_RE = re.compile(
    r"(?:de\s+minimis|minimum\s+(?:claim|loss)\s+(?:amount|threshold)|per\s+claim\s+threshold)"
    + _NEAR + r"\$\s*" + _AMT,
    re.IGNORECASE,
)
_INDEMNIFICATION_DE_MINIMIS_REVERSE_RE = re.compile(
    r"\$\s*(?P<amount>[\d,]+(?:\.\d+)?(?:\s*(?:million|billion))?)"
    r"[^\n]{0,150}?"
    r"""["']?(?:the\s+)?["']?(?:De\s+Minimis\s+(?:Amount|Threshold)|Minimum\s+Claim)["']?""",
    re.IGNORECASE,
)

_INDEMNIFICATION_BASKET_RE = re.compile(
    r"(?:basket|deductible|retention)" + _NEAR + r"\$\s*" + _AMT,
    re.IGNORECASE,
)
_INDEMNIFICATION_BASKET_REVERSE_RE = re.compile(
    r"\$\s*(?P<amount>[\d,]+(?:\.\d+)?(?:\s*(?:million|billion))?)"
    r"[^\n]{0,150}?"
    r"""["']?(?:the\s+)?["']?(?:Basket|Deductible|Retention)\s+(?:Amount)?["']?""",
    re.IGNORECASE,
)

_INDEMNIFICATION_CAP_RE = re.compile(
    r"(?:cap|ceiling|maximum\s+(?:aggregate\s+)?(?:indemnification|liability))\s*"
    + _NEAR + r"\$\s*" + _AMT,
    re.IGNORECASE,
)
_INDEMNIFICATION_CAP_REVERSE_RE = re.compile(
    r"\$\s*(?P<amount>[\d,]+(?:\.\d+)?(?:\s*(?:million|billion))?)"
    r"[^\n]{0,150}?"
    r"""["']?(?:the\s+)?["']?(?:Cap|Ceiling|Maximum\s+(?:Aggregate\s+)?Liability)["']?""",
    re.IGNORECASE,
)

# --- Governing law ---
_GOVERNING_LAW_RE = re.compile(
    r"(?:governed\s+by\s+(?:and\s+construed\s+in\s+accordance\s+with\s+)?the\s+laws\s+of\s+"
    r"(?:the\s+(?:State|Commonwealth|Province)\s+of\s+)?)"
    r"(?P<state>[A-Z][a-zA-Z\s]{2,40}?)(?=[,.\n])",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public rule functions
# ---------------------------------------------------------------------------


def extract_document_type(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the document type (e.g. 'Share Purchase Agreement')."""
    return _search_pages(pages_text, _DOC_TYPE_RE, group=0, confidence=0.95)


def extract_effective_date(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the effective/execution date."""
    return _search_pages(pages_text, _EFFECTIVE_DATE_RE, group="date", confidence=0.90)


def extract_buyer(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the buyer party name."""
    return _search_pages(pages_text[:5], _BUYER_RE, group="buyer", confidence=0.85)


def extract_seller(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the seller party name."""
    return _search_pages(pages_text[:5], _SELLER_RE, group="seller", confidence=0.85)


def extract_company_target(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the target company name."""
    return _search_pages(
        pages_text[:5], _COMPANY_TARGET_RE, group="company", confidence=0.85
    )


def extract_shares_transacted(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract number of shares transacted."""
    # Try the preamble-scoped pattern first
    result = _search_pages(
        pages_text[:3], _SHARES_ALL_PREAMBLE_RE, group="shares", confidence=0.85
    )
    if result:
        return result
    return _search_pages(pages_text[:5], _SHARES_RE, group="shares", confidence=0.70)


def extract_cash_purchase_price(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract aggregate cash purchase price."""
    result = _search_pages(pages_text, _PURCHASE_PRICE_RE, group="amount", confidence=0.88)
    if result:
        return result
    return _search_pages(pages_text, _CONSIDERATION_RE, group="amount", confidence=0.75)


def extract_escrow_agent(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the escrow agent name."""
    return _search_pages(pages_text, _ESCROW_AGENT_RE, group="agent", confidence=0.85)


def extract_escrow_amount(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the escrow amount."""
    result = _search_pages(pages_text, _ESCROW_AMOUNT_RE, group="amount", confidence=0.85)
    if result:
        return result
    return _search_pages(pages_text, _ESCROW_AMOUNT_REVERSE_RE, group="amount", confidence=0.78)


def extract_target_working_capital(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the target working capital amount."""
    return _search_pages(
        pages_text, _WORKING_CAPITAL_RE, group="amount", confidence=0.85
    )


def extract_indemnification_de_minimis(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the indemnification de minimis threshold."""
    result = _search_pages(pages_text, _INDEMNIFICATION_DE_MINIMIS_RE, group="amount", confidence=0.80)
    if result:
        return result
    return _search_pages(pages_text, _INDEMNIFICATION_DE_MINIMIS_REVERSE_RE, group="amount", confidence=0.73)


def extract_indemnification_basket(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the indemnification basket / deductible amount."""
    result = _search_pages(pages_text, _INDEMNIFICATION_BASKET_RE, group="amount", confidence=0.80)
    if result:
        return result
    return _search_pages(pages_text, _INDEMNIFICATION_BASKET_REVERSE_RE, group="amount", confidence=0.73)


def extract_indemnification_cap(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the indemnification cap / maximum aggregate liability."""
    result = _search_pages(pages_text, _INDEMNIFICATION_CAP_RE, group="amount", confidence=0.80)
    if result:
        return result
    return _search_pages(pages_text, _INDEMNIFICATION_CAP_REVERSE_RE, group="amount", confidence=0.73)


def extract_governing_law(
    pages_text: list[tuple[int, str]],
) -> RuleMatch | None:
    """Extract the governing law jurisdiction."""
    return _search_pages(
        pages_text, _GOVERNING_LAW_RE, group="state", confidence=0.90
    )

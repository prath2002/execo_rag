"""Text cleaner for raw PDF extraction output.

Removes:
- Running headers and footers (page numbers, repeated short lines)
- SEC EDGAR filing artifacts (exhibit labels, EDGAR codes)
- Unicode ligatures (ﬁ → fi, ﬂ → fl, etc.)
- Excessive whitespace and blank lines
- Form-feed and other non-printable control characters
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter

from execo_rag.models.extraction import CleanedPage, ExtractedPage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typographic quote normalization map
# Legal PDFs use curly/smart quotes that break ASCII regex patterns.
# We fold them to straight ASCII equivalents before any other processing.
# ---------------------------------------------------------------------------
_QUOTE_MAP: dict[str, str] = {
    "\u201c": '"',  # " left double quotation mark
    "\u201d": '"',  # " right double quotation mark
    "\u201e": '"',  # „ double low-9 quotation mark
    "\u2018": "'",  # ' left single quotation mark
    "\u2019": "'",  # ' right single quotation mark (also apostrophe)
    "\u201a": "'",  # ‚ single low-9 quotation mark
    "\u2032": "'",  # ′ prime (used as apostrophe in some PDFs)
    "\u00ab": '"',  # « left-pointing double angle quotation mark
    "\u00bb": '"',  # » right-pointing double angle quotation mark
    "\u2039": "'",  # ‹ single left-pointing angle quotation mark
    "\u203a": "'",  # › single right-pointing angle quotation mark
}
_QUOTE_RE = re.compile("|".join(re.escape(k) for k in _QUOTE_MAP))

# ---------------------------------------------------------------------------
# Ligature normalization map
# ---------------------------------------------------------------------------
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u00e6": "ae",  # æ
    "\u00c6": "AE",  # Æ
    "\u0153": "oe",  # œ
    "\u0152": "OE",  # Œ
}
_LIGATURE_RE = re.compile("|".join(re.escape(k) for k in _LIGATURE_MAP))

# ---------------------------------------------------------------------------
# SEC / EDGAR artifact patterns
# ---------------------------------------------------------------------------
_SEC_PATTERNS: list[re.Pattern[str]] = [
    # EDGAR exhibit cover lines  (e.g. "EXHIBIT 10.1", "Exhibit A")
    re.compile(r"^\s*EXHIBIT\s+[\w.]+\s*$", re.IGNORECASE | re.MULTILINE),
    # EDGAR filing headers (e.g. "10-K", "8-K", "S-1", "FORM 10-K")
    re.compile(r"^\s*(?:FORM\s+)?(?:10-[KQ]|8-K|S-[14]|DEF\s+14A|20-F)\s*$", re.IGNORECASE | re.MULTILINE),
    # EDGAR document start/end tags
    re.compile(r"</?(?:DOCUMENT|SEQUENCE|FILENAME|DESCRIPTION|TYPE|TEXT)>.*$", re.IGNORECASE | re.MULTILINE),
    # Page-count footers inserted by EDGAR (e.g. "Page 1 of 42")
    re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
    # Confidential / draft watermarks
    re.compile(r"^\s*(?:CONFIDENTIAL|DRAFT|EXECUTION COPY|EXECUTION VERSION)\s*$", re.IGNORECASE | re.MULTILINE),
]

# ---------------------------------------------------------------------------
# Running header / footer detection helpers
# ---------------------------------------------------------------------------
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:-\s*)?\d{1,4}(?:\s*-\s*)?\s*$"
    r"|"
    r"^\s*(?:page\s+)?\d{1,4}\s*(?:of\s+\d{1,4})?\s*$",
    re.IGNORECASE,
)

# A line is considered a candidate header/footer if it is short and repetitive
_MAX_HEADER_LINE_LEN = 80
_MIN_REPETITION_COUNT = 3  # must appear on this many pages to be auto-removed


def _normalize_quotes(text: str) -> str:
    """Fold typographic/curly quotes to ASCII straight quotes.

    Legal PDFs almost always use Unicode curly quotes.  Without this step,
    regex patterns like ``[\"']?Buyer[\"']?`` silently fail to match party
    definitions such as ``(the \u201cBuyer\u201d)``.
    """
    return _QUOTE_RE.sub(lambda m: _QUOTE_MAP[m.group()], text)


def _normalize_ligatures(text: str) -> str:
    """Replace Unicode ligatures with their ASCII equivalents."""
    return _LIGATURE_RE.sub(lambda m: _LIGATURE_MAP[m.group()], text)


def _remove_sec_artifacts(text: str) -> tuple[str, list[str]]:
    """Strip SEC/EDGAR filing artifacts.  Returns (cleaned_text, removed_fragments)."""
    removed: list[str] = []
    for pattern in _SEC_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            removed.extend(m.strip() for m in matches if m.strip())
        text = pattern.sub("", text)
    return text, removed


def _is_page_number_line(line: str) -> bool:
    """Return True if the line appears to be a standalone page number."""
    return bool(_PAGE_NUMBER_RE.match(line.strip()))


def _detect_repeated_lines(pages: list[ExtractedPage]) -> set[str]:
    """Identify lines that appear on many pages (likely running headers/footers).

    Only short lines (≤ _MAX_HEADER_LINE_LEN chars) are considered.

    Args:
        pages: Raw extracted pages.

    Returns:
        Set of line strings that should be removed.
    """
    line_counts: Counter[str] = Counter()
    for page in pages:
        seen_on_page: set[str] = set()
        for raw_line in page.raw_text.splitlines():
            line = raw_line.strip()
            if line and len(line) <= _MAX_HEADER_LINE_LEN:
                if line not in seen_on_page:
                    line_counts[line] += 1
                    seen_on_page.add(line)

    repeated = {
        line
        for line, count in line_counts.items()
        if count >= _MIN_REPETITION_COUNT
    }
    logger.debug(
        "Detected repeated header/footer lines",
        extra={"extra_data": {"count": len(repeated)}},
    )
    return repeated


def _remove_repeated_lines(text: str, repeated: set[str]) -> tuple[str, list[str]]:
    """Remove lines that are in the repeated-lines set."""
    removed: list[str] = []
    clean_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in repeated or _is_page_number_line(stripped):
            if stripped:
                removed.append(stripped)
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines), removed


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines and strip trailing whitespace."""
    # Remove form-feed and other control chars (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Collapse multiple spaces/tabs to a single space (within a line)
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    # Collapse runs of 3+ blank lines to a single blank line
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def _normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization to resolve composed forms."""
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_pages(pages: list[ExtractedPage]) -> list[CleanedPage]:
    """Clean a list of raw extracted pages.

    Processing order per page:
      1. Typographic quote normalization (curly → ASCII)
      2. Ligature normalization
      3. Unicode NFC normalization
      4. SEC/EDGAR artifact removal
      5. Running header/footer removal (corpus-level detection)
      6. Whitespace normalization

    Args:
        pages: Raw :class:`ExtractedPage` objects from the extraction service.

    Returns:
        List of :class:`CleanedPage` objects in the same page order.
    """
    if not pages:
        return []

    # Corpus-level: detect lines repeated across multiple pages
    repeated_lines = _detect_repeated_lines(pages)

    cleaned: list[CleanedPage] = []
    total_removed_fragments: int = 0

    for page in pages:
        removed_fragments: list[str] = []
        text = page.raw_text

        # Step 1: Typographic quote normalization (curly → ASCII)
        text = _normalize_quotes(text)

        # Step 2: Ligature normalization
        text = _normalize_ligatures(text)

        # Step 3: Unicode NFC normalization
        text = _normalize_unicode(text)

        # Step 4: SEC artifact removal
        text, sec_removed = _remove_sec_artifacts(text)
        removed_fragments.extend(sec_removed)

        # Step 5: Running header/footer removal
        text, header_removed = _remove_repeated_lines(text, repeated_lines)
        removed_fragments.extend(header_removed)

        # Step 6: Whitespace normalization
        text = _normalize_whitespace(text)

        total_removed_fragments += len(removed_fragments)
        cleaned.append(
            CleanedPage(
                page_number=page.page_number,
                cleaned_text=text,
                removed_noise_fragments=removed_fragments,
            )
        )

    logger.info(
        "Text cleaning complete",
        extra={
            "extra_data": {
                "total_pages": len(pages),
                "total_removed_fragments": total_removed_fragments,
            }
        },
    )
    return cleaned

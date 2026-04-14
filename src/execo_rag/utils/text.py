"""Text normalization utilities: whitespace, punctuation, and casing."""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Whitespace helpers
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Collapse all internal whitespace to a single space and strip edges.

    Args:
        text: Input string that may contain tabs, newlines, or multiple spaces.

    Returns:
        Whitespace-normalized string.
    """
    return re.sub(r"\s+", " ", text).strip()


def collapse_blank_lines(text: str, max_blank: int = 1) -> str:
    """Reduce consecutive blank lines to at most *max_blank* blank lines.

    Args:
        text: Multi-line input string.
        max_blank: Maximum number of consecutive blank lines to allow.

    Returns:
        Cleaned string with excess blank lines removed.
    """
    pattern = r"\n{" + str(max_blank + 2) + r",}"
    replacement = "\n" * (max_blank + 1)
    return re.sub(pattern, replacement, text)


# ---------------------------------------------------------------------------
# Punctuation helpers
# ---------------------------------------------------------------------------


def normalize_quotes(text: str) -> str:
    """Replace curly/smart quotes with their straight ASCII equivalents.

    Args:
        text: Input string possibly containing typographic quotes.

    Returns:
        String with ASCII-normalized quotation marks.
    """
    replacements = {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2032": "'",  # prime
        "\u2033": '"',  # double prime
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text


def normalize_dashes(text: str) -> str:
    """Replace en-dash and em-dash variants with a simple hyphen-minus.

    Args:
        text: Input string possibly containing Unicode dash characters.

    Returns:
        String with dashes normalized to ASCII hyphen ``-``.
    """
    return re.sub(r"[\u2013\u2014\u2015\u2212]", "-", text)


def strip_punctuation_edges(text: str) -> str:
    """Strip leading and trailing punctuation from a string.

    Useful for cleaning extracted party names and field values.

    Args:
        text: Input string.

    Returns:
        String with leading/trailing punctuation stripped.
    """
    return text.strip().strip(".,;:!?\"'()-")


# ---------------------------------------------------------------------------
# Casing helpers
# ---------------------------------------------------------------------------


def normalize_to_lowercase(text: str) -> str:
    """Return a lowercase, whitespace-collapsed version of *text*.

    Args:
        text: Input string.

    Returns:
        Normalized lowercase string.
    """
    return normalize_whitespace(text.lower())


def title_case_name(text: str) -> str:
    """Apply title-case to a person or company name.

    Unlike :func:`str.title`, this version preserves internal uppercase
    abbreviations (e.g. 'LLC', 'Inc.', 'II').

    Args:
        text: Raw name string.

    Returns:
        Title-cased name.
    """
    return " ".join(
        word if word.isupper() and len(word) > 1 else word.capitalize()
        for word in normalize_whitespace(text).split()
    )


# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------


def remove_control_characters(text: str) -> str:
    """Remove all Unicode control characters except newline and tab.

    Args:
        text: Input string.

    Returns:
        String with control characters stripped.
    """
    return "".join(
        ch
        for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t")
    )


def nfc_normalize(text: str) -> str:
    """Apply NFC Unicode normalization.

    Args:
        text: Input string.

    Returns:
        NFC-normalized string.
    """
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


def clean_field_value(raw: str) -> str:
    """Apply the full normalization pipeline to a raw extracted field value.

    Steps:
      1. NFC normalization
      2. Control character removal
      3. Quote normalization
      4. Dash normalization
      5. Whitespace normalization
      6. Edge punctuation stripping

    Args:
        raw: Raw extracted value string.

    Returns:
        Cleaned and normalized field value.
    """
    text = nfc_normalize(raw)
    text = remove_control_characters(text)
    text = normalize_quotes(text)
    text = normalize_dashes(text)
    text = normalize_whitespace(text)
    text = strip_punctuation_edges(text)
    return text

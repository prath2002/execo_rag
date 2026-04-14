"""Section detector for Share Purchase Agreement documents.

Assigns each paragraph to one of the recognized :class:`SectionType` values
by matching headings and contextual keywords against a priority-ordered rule set.

The detector operates on :class:`CleanedPage` objects and returns
:class:`SectionSegment` objects suitable for the hybrid chunker.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from execo_rag.models.extraction import CleanedPage, SectionSegment
from execo_rag.models.metadata import SectionType
from execo_rag.utils.ids import generate_segment_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section detection rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionRule:
    """A single heading-match rule that maps text patterns to a SectionType."""

    section: SectionType
    heading_patterns: list[re.Pattern[str]]
    keyword_patterns: list[re.Pattern[str]]
    priority: int  # Lower number = higher priority when multiple rules match


def _build_rules() -> list[SectionRule]:
    """Build the ordered list of SPA section rules."""
    return [
        SectionRule(
            section=SectionType.PREAMBLE,
            heading_patterns=[
                re.compile(r"^\s*(?:recitals?|whereas|background|preamble)\b", re.IGNORECASE),
            ],
            keyword_patterns=[
                re.compile(r"\bwhereas\b", re.IGNORECASE),
                re.compile(r"\bnow,?\s+therefore\b", re.IGNORECASE),
                re.compile(r"\bin\s+consideration\s+of\b", re.IGNORECASE),
            ],
            priority=10,
        ),
        SectionRule(
            section=SectionType.DEFINITIONS,
            heading_patterns=[
                re.compile(r"^\s*(?:article|section)?\s*\d*\.?\s*definitions?\b", re.IGNORECASE),
                re.compile(r"^\s*defined\s+terms?\b", re.IGNORECASE),
            ],
            keyword_patterns=[
                re.compile(r'"[\w\s]+"(?:\s+means?\b|\s+shall\s+mean\b)', re.IGNORECASE),
                re.compile(r"\b(?:as\s+used\s+herein|for\s+purposes\s+of\s+this\s+agreement)\b", re.IGNORECASE),
            ],
            priority=20,
        ),
        SectionRule(
            section=SectionType.PURCHASE_PRICE,
            heading_patterns=[
                re.compile(
                    r"^\s*(?:article|section)?\s*\d*\.?\s*"
                    r"(?:purchase\s+price|consideration|payment|transaction)",
                    re.IGNORECASE,
                ),
            ],
            keyword_patterns=[
                re.compile(r"\b(?:aggregate\s+)?(?:cash\s+)?purchase\s+price\b", re.IGNORECASE),
                re.compile(r"\bpurchase\s+and\s+sale\b", re.IGNORECASE),
                re.compile(r"\b(?:aggregate\s+)?consideration\b", re.IGNORECASE),
                re.compile(r"\bclose(?:ing)?\s+payment\b", re.IGNORECASE),
            ],
            priority=30,
        ),
        SectionRule(
            section=SectionType.ESCROW,
            heading_patterns=[
                re.compile(r"^\s*(?:article|section)?\s*\d*\.?\s*escrow\b", re.IGNORECASE),
            ],
            keyword_patterns=[
                re.compile(r"\bescrow\s+(?:agent|amount|fund|account|deposit)\b", re.IGNORECASE),
                re.compile(r"\bescrow\s+agreement\b", re.IGNORECASE),
                re.compile(r"\bhold\s+in\s+escrow\b", re.IGNORECASE),
            ],
            priority=40,
        ),
        SectionRule(
            section=SectionType.WORKING_CAPITAL,
            heading_patterns=[
                re.compile(
                    r"^\s*(?:article|section)?\s*\d*\.?\s*working\s+capital\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"^\s*(?:article|section)?\s*\d*\.?\s*net\s+working\s+capital\b",
                    re.IGNORECASE,
                ),
            ],
            keyword_patterns=[
                re.compile(r"\b(?:target|reference|required|estimated)\s+working\s+capital\b", re.IGNORECASE),
                re.compile(r"\bworking\s+capital\s+adjustment\b", re.IGNORECASE),
                re.compile(r"\bnet\s+working\s+capital\b", re.IGNORECASE),
            ],
            priority=50,
        ),
        SectionRule(
            section=SectionType.INDEMNIFICATION,
            heading_patterns=[
                re.compile(
                    r"^\s*(?:article|section)?\s*\d*\.?\s*indemni(?:fication|ty)\b",
                    re.IGNORECASE,
                ),
                re.compile(r"^\s*(?:article|section)?\s*\d*\.?\s*indemnities\b", re.IGNORECASE),
            ],
            keyword_patterns=[
                re.compile(r"\bindemnif(?:y|ied|ication)\b", re.IGNORECASE),
                re.compile(r"\bde\s+minimis\b", re.IGNORECASE),
                re.compile(r"\bbasket\b|\bdeductible\b", re.IGNORECASE),
                re.compile(r"\bindemnification\s+cap\b", re.IGNORECASE),
                re.compile(r"\blosses?,\s+claims?\b", re.IGNORECASE),
            ],
            priority=60,
        ),
        SectionRule(
            section=SectionType.GOVERNING_LAW,
            heading_patterns=[
                re.compile(
                    r"^\s*(?:article|section)?\s*\d*\.?\s*governing\s+law\b",
                    re.IGNORECASE,
                ),
                re.compile(r"^\s*(?:article|section)?\s*\d*\.?\s*choice\s+of\s+law\b", re.IGNORECASE),
                re.compile(r"^\s*(?:article|section)?\s*\d*\.?\s*jurisdiction\b", re.IGNORECASE),
            ],
            keyword_patterns=[
                re.compile(r"\bgoverned\s+by\s+(?:and\s+construed\s+in\s+accordance\s+with\s+)?the\s+laws\s+of\b", re.IGNORECASE),
                re.compile(r"\bsubmits?\s+to\s+(?:the\s+)?jurisdiction\b", re.IGNORECASE),
            ],
            priority=70,
        ),
    ]


_RULES = _build_rules()

# Pattern to detect a potential section heading line
_HEADING_RE = re.compile(
    r"^(?:ARTICLE|SECTION|ARTICLE\s+\d+|SECTION\s+\d+|"
    r"\d+\.\s+[A-Z]|\d+\.\d+\s+[A-Z]|[A-Z][A-Z\s]{2,40})\s*$",
    re.MULTILINE,
)

_ARTICLE_RE = re.compile(
    r"^(?:ARTICLE|SECTION)\s+(?:[IVXLCDM]+|\d+)",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


def _classify_paragraph(paragraph: str) -> SectionType:
    """Classify a paragraph into a SectionType using the rule set.

    Args:
        paragraph: Cleaned paragraph text.

    Returns:
        Best matching :class:`SectionType`, or ``SectionType.GENERAL``.
    """
    paragraph_lower = paragraph.lower()
    first_line = paragraph.splitlines()[0].strip() if paragraph else ""

    best_section: SectionType = SectionType.GENERAL
    best_priority: int = 999

    for rule in _RULES:
        if rule.priority >= best_priority:
            continue

        # Heading match (strong signal — higher confidence)
        for heading_pat in rule.heading_patterns:
            if heading_pat.search(first_line):
                best_section = rule.section
                best_priority = rule.priority
                break

        # Keyword match (weaker signal — only if no heading match yet)
        if best_section == SectionType.GENERAL or rule.priority < best_priority:
            keyword_hits = sum(
                1 for kp in rule.keyword_patterns if kp.search(paragraph_lower)
            )
            if keyword_hits >= 2 and rule.priority < best_priority:
                best_section = rule.section
                best_priority = rule.priority

    return best_section


# ---------------------------------------------------------------------------
# Paragraph splitter
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[str]:
    """Split page text into paragraphs at blank lines or article/section boundaries."""
    # First split on double newlines (blank lines)
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        stripped = para.strip()
        if stripped:
            paragraphs.append(stripped)
    return paragraphs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_sections(
    cleaned_pages: list[CleanedPage],
    document_id: str,
) -> list[SectionSegment]:
    """Detect and label logical sections from cleaned document pages.

    The detector processes pages in order, maintains a running current-section
    state, and assigns each paragraph to the most appropriate section.

    Args:
        cleaned_pages: Output from the text cleaning service.
        document_id: Document identifier used for segment ID generation.

    Returns:
        Ordered list of :class:`SectionSegment` objects, one per paragraph.
    """
    segments: list[SectionSegment] = []
    current_section: SectionType = SectionType.PREAMBLE
    order_index: int = 0

    for page in cleaned_pages:
        paragraphs = _split_paragraphs(page.cleaned_text)

        for para in paragraphs:
            if len(para.strip()) < 10:
                continue  # Skip trivially short fragments

            detected = _classify_paragraph(para)

            # Update running section state only if a specific section was detected
            if detected != SectionType.GENERAL:
                current_section = detected
            else:
                detected = current_section  # inherit from previous

            segment = SectionSegment(
                segment_id=generate_segment_id(document_id, page.page_number, order_index),
                page_number=page.page_number,
                section=detected,
                subsection=None,
                text=para,
                order_index=order_index,
            )
            segments.append(segment)
            order_index += 1

    logger.info(
        "Section detection complete",
        extra={
            "extra_data": {
                "document_id": document_id,
                "total_segments": len(segments),
                "sections_found": list(
                    {
                        s.section.value if hasattr(s.section, "value") else str(s.section)
                        for s in segments
                    }
                ),
            }
        },
    )

    return segments

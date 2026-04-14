"""PDF extraction service.

Primary parser: ``unstructured`` (partition_pdf).
Fallback parsers: ``pypdf`` then ``pdfminer.six``.

Both parsers produce a list of :class:`ExtractedPage` objects that are
assembled into an :class:`ExtractedDocument`.  The fallback activates
automatically when the primary parser raises an exception or returns no
usable text content.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from execo_rag.models.extraction import (
    ExtractionBlockType,
    ExtractedBlock,
    ExtractedDocument,
    ExtractedPage,
)
from execo_rag.utils.exceptions import ExtractionError
from execo_rag.utils.ids import generate_block_id

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Minimum characters on a page for the extraction to be considered successful
_MIN_PAGE_CHARS = 20


# ---------------------------------------------------------------------------
# Primary extractor: unstructured
# ---------------------------------------------------------------------------


def _extract_with_unstructured(
    path: Path, document_id: str
) -> list[ExtractedPage]:
    """Extract text using the ``unstructured`` library.

    Args:
        path: PDF file path.
        document_id: Parent document identifier (used for block IDs).

    Returns:
        List of :class:`ExtractedPage` objects, one per physical PDF page.

    Raises:
        ImportError: If ``unstructured`` is not installed.
        ExtractionError: If the library raises an unexpected exception.
    """
    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'unstructured' package is required for primary PDF extraction. "
            "Install it with: pip install 'unstructured[pdf]'"
        ) from exc

    logger.debug(
        "Starting unstructured extraction",
        extra={"extra_data": {"path": str(path), "document_id": document_id}},
    )

    try:
        elements = partition_pdf(
            filename=str(path),
            strategy="hi_res",
            include_page_breaks=True,
        )
    except Exception as exc:
        raise ExtractionError(
            message=f"unstructured.partition_pdf failed: {exc}",
            error_code="unstructured_parse_error",
            details={"path": str(path), "error": str(exc)},
        ) from exc

    # Group elements by page number
    pages: dict[int, list[ExtractedBlock]] = {}
    block_counters: dict[int, int] = {}

    for element in elements:
        page_num: int = getattr(element, "metadata", None) and getattr(
            element.metadata, "page_number", None
        ) or 1
        if page_num is None:
            page_num = 1

        element_type = type(element).__name__.lower()
        block_type = _map_unstructured_type(element_type)
        text = str(element).strip()
        if not text:
            continue

        idx = block_counters.get(page_num, 0)
        block = ExtractedBlock(
            block_id=generate_block_id(document_id, page_num, idx),
            page_number=page_num,
            block_type=block_type,
            text=text,
        )
        pages.setdefault(page_num, []).append(block)
        block_counters[page_num] = idx + 1

    return _assemble_pages(pages)


def _map_unstructured_type(element_type: str) -> ExtractionBlockType:
    """Map an unstructured element class name to our block type enum."""
    mapping: dict[str, ExtractionBlockType] = {
        "title": ExtractionBlockType.TITLE,
        "header": ExtractionBlockType.HEADER,
        "narrativetext": ExtractionBlockType.PARAGRAPH,
        "text": ExtractionBlockType.PARAGRAPH,
        "listitem": ExtractionBlockType.LIST_ITEM,
        "table": ExtractionBlockType.TABLE,
        "footer": ExtractionBlockType.FOOTER,
        "pagebreak": ExtractionBlockType.UNKNOWN,
    }
    return mapping.get(element_type, ExtractionBlockType.UNKNOWN)


# ---------------------------------------------------------------------------
# Fallback extractors
# ---------------------------------------------------------------------------


def _extract_with_pypdf(
    path: Path, document_id: str
) -> list[ExtractedPage]:
    """Extract text using ``pypdf`` as a reliable page-aware fallback."""

    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'pypdf' package is required for PDF fallback extraction. "
            "Install it with: pip install pypdf"
        ) from exc

    logger.debug(
        "Starting pypdf fallback extraction",
        extra={"extra_data": {"path": str(path), "document_id": document_id}},
    )

    pages: dict[int, list[ExtractedBlock]] = {}

    try:
        reader = PdfReader(str(path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            pages[page_num] = [
                ExtractedBlock(
                    block_id=generate_block_id(document_id, page_num, 0),
                    page_number=page_num,
                    block_type=ExtractionBlockType.PARAGRAPH,
                    text=text,
                )
            ]
    except Exception as exc:
        raise ExtractionError(
            message=f"pypdf extraction failed: {exc}",
            error_code="pypdf_parse_error",
            details={"path": str(path), "error": str(exc)},
        ) from exc

    return _assemble_pages(pages)


def _extract_with_pdfminer(
    path: Path, document_id: str
) -> list[ExtractedPage]:
    """Extract text using ``pdfminer.six`` as a page-aware fallback.

    Args:
        path: PDF file path.
        document_id: Parent document identifier.

    Returns:
        List of :class:`ExtractedPage` objects.

    Raises:
        ImportError: If ``pdfminer`` is not installed.
        ExtractionError: If pdfminer raises an unexpected exception.
    """
    try:
        from pdfminer.high_level import extract_pages  # type: ignore[import-untyped]
        from pdfminer.layout import LTTextBox  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'pdfminer.six' package is required for fallback PDF extraction. "
            "Install it with: pip install pdfminer.six"
        ) from exc

    logger.debug(
        "Starting pdfminer fallback extraction",
        extra={"extra_data": {"path": str(path), "document_id": document_id}},
    )

    pages: dict[int, list[ExtractedBlock]] = {}

    try:
        for page_num, page_layout in enumerate(extract_pages(str(path)), start=1):
            blocks: list[ExtractedBlock] = []
            for idx, element in enumerate(page_layout):
                if isinstance(element, LTTextBox):
                    text = element.get_text().strip()
                    if text:
                        block = ExtractedBlock(
                            block_id=generate_block_id(document_id, page_num, idx),
                            page_number=page_num,
                            block_type=ExtractionBlockType.PARAGRAPH,
                            text=text,
                        )
                        blocks.append(block)
            if blocks:
                pages[page_num] = blocks
    except Exception as exc:
        raise ExtractionError(
            message=f"pdfminer extraction failed: {exc}",
            error_code="pdfminer_parse_error",
            details={"path": str(path), "error": str(exc)},
        ) from exc

    return _assemble_pages(pages)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _assemble_pages(
    pages: dict[int, list[ExtractedBlock]],
) -> list[ExtractedPage]:
    """Convert a page-to-blocks mapping into sorted :class:`ExtractedPage` objects."""
    result: list[ExtractedPage] = []
    for page_num in sorted(pages.keys()):
        blocks = pages[page_num]
        raw_text = "\n".join(b.text for b in blocks)
        result.append(
            ExtractedPage(
                page_number=page_num,
                raw_text=raw_text,
                blocks=blocks,
            )
        )
    return result


def _is_extraction_sufficient(pages: list[ExtractedPage]) -> bool:
    """Return True if the extracted pages contain enough usable text."""
    if not pages:
        return False
    total_chars = sum(len(p.raw_text.strip()) for p in pages)
    return total_chars >= _MIN_PAGE_CHARS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_pdf(
    path: Path,
    document_id: str,
    preferred_extractor: str = "unstructured",
) -> ExtractedDocument:
    """Extract text from a PDF file and return an :class:`ExtractedDocument`.

    Tries the ``preferred_extractor`` first.  If it fails or produces no usable
    text, the service automatically falls back to ``pdfminer.six``.

    Args:
        path: Absolute path to the validated PDF file.
        document_id: Stable document identifier (used for block IDs).
        preferred_extractor: ``'unstructured'`` (default) or ``'pdfminer'``.

    Returns:
        Populated :class:`ExtractedDocument` with all pages and blocks.

    Raises:
        ExtractionError: If both parsers fail to produce usable content.
    """
    start_ts = time.perf_counter()
    logger.info(
        "PDF extraction started",
        extra={
            "extra_data": {
                "document_id": document_id,
                "path": str(path),
                "preferred_extractor": preferred_extractor,
            }
        },
    )

    pages: list[ExtractedPage] = []
    extractor_used = preferred_extractor
    primary_error: str | None = None

    # --- Primary pass ---
    if preferred_extractor == "unstructured":
        try:
            pages = _extract_with_unstructured(path, document_id)
        except (ExtractionError, ImportError, Exception) as exc:
            primary_error = str(exc)
            logger.warning(
                "Primary extraction failed; switching to fallback",
                extra={
                    "extra_data": {
                        "document_id": document_id,
                        "extractor": "unstructured",
                        "error": primary_error,
                    }
                },
            )
    else:
        try:
            pages = _extract_with_pdfminer(path, document_id)
        except (ExtractionError, ImportError, Exception) as exc:
            primary_error = str(exc)

    # --- Quality gate: trigger fallback ---
    if not _is_extraction_sufficient(pages):
        logger.info(
            "Primary extraction insufficient; running fallback extractor",
            extra={
                "extra_data": {
                    "document_id": document_id,
                    "primary_pages": len(pages),
                    "primary_error": primary_error,
                }
            },
        )
        # D3: Fallback path
        fallback_pages, fallback_extractor = _run_fallback(path, document_id, preferred_extractor)
        if _is_extraction_sufficient(fallback_pages):
            pages = fallback_pages
            extractor_used = fallback_extractor
        elif not pages:
            raise ExtractionError(
                message="Both primary and fallback PDF extractors produced no usable text.",
                error_code="extraction_empty",
                details={
                    "document_id": document_id,
                    "path": str(path),
                    "primary_error": primary_error,
                },
            )

    elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
    logger.info(
        "PDF extraction complete",
        extra={
            "extra_data": {
                "document_id": document_id,
                "extractor_used": extractor_used,
                "total_pages": len(pages),
                "duration_ms": elapsed_ms,
            }
        },
    )

    return ExtractedDocument(
        document_id=document_id,
        total_pages=len(pages),
        pages=pages,
        extractor_used=extractor_used,
    )


def _run_fallback(
    path: Path,
    document_id: str,
    failed_extractor: str,
) -> tuple[list[ExtractedPage], str]:
    """Run the fallback extractor chain and return the first usable result.

    Args:
        path: PDF file path.
        document_id: Document identifier.
        failed_extractor: The extractor that just failed ('unstructured' or 'pdfminer').

    Returns:
        Tuple of ``(pages, extractor_name)``. Returns ``([], "none")`` if every
        fallback parser fails.
    """
    fallback_order = ["pypdf", "pdfminer"]
    if failed_extractor == "pypdf":
        fallback_order = ["pdfminer", "unstructured"]
    elif failed_extractor == "pdfminer":
        fallback_order = ["pypdf", "unstructured"]

    for extractor_name in fallback_order:
        try:
            if extractor_name == "pypdf":
                pages = _extract_with_pypdf(path, document_id)
            elif extractor_name == "pdfminer":
                pages = _extract_with_pdfminer(path, document_id)
            else:
                pages = _extract_with_unstructured(path, document_id)

            if _is_extraction_sufficient(pages):
                return pages, extractor_name
        except Exception as exc:
            logger.error(
                "Fallback extraction failed",
                extra={
                    "extra_data": {
                        "document_id": document_id,
                        "extractor": extractor_name,
                        "error": str(exc),
                    }
                },
            )

    return [], "none"

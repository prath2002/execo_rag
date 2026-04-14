"""Hybrid chunker: section-first, paragraph-aware, token-bounded.

Chunking strategy:
  1. Group :class:`SectionSegment` objects by their :class:`SectionType`.
  2. Within each section group, accumulate consecutive paragraphs until the
     token budget is full.
  3. When a single paragraph exceeds the budget, split it at a word boundary
     using :func:`split_to_token_budget`.
  4. When overlap is configured (``chunk_overlap_tokens > 0``), carry the
     last N tokens of the current chunk as a prefix into the next chunk.

Returns a list of :class:`RawChunk` objects ready for metadata enrichment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from execo_rag.config.constants import DEFAULT_EMBEDDING_MODEL
from execo_rag.models.chunk import RawChunk
from execo_rag.models.extraction import SectionSegment
from execo_rag.models.metadata import SectionType
from execo_rag.utils.exceptions import ChunkingError
from execo_rag.utils.ids import generate_chunk_id

from .token_counter import count_tokens, split_to_token_budget

logger = logging.getLogger(__name__)

# Minimum meaningful chunk size (tokens)
_MIN_CHUNK_TOKENS = 10


# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------


@dataclass
class _ChunkBuffer:
    """Accumulator for in-progress chunk content."""

    paragraphs: list[str]
    page_start: int
    page_end: int
    section: SectionType
    subsection: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _emit_chunk(
    buf: _ChunkBuffer,
    document_id: str,
    model_name: str,
    chunk_index: int,
) -> RawChunk | None:
    """Convert a buffer into a :class:`RawChunk`.

    Returns ``None`` if the buffer contains no usable text.
    """
    text = buf.text.strip()
    if not text:
        return None

    token_count = count_tokens(text, model_name)
    if token_count < _MIN_CHUNK_TOKENS:
        return None

    return RawChunk(
        chunk_id=generate_chunk_id(document_id, chunk_index),
        document_id=document_id,
        text=text,
        section=buf.section,
        subsection=buf.subsection,
        page_start=buf.page_start,
        page_end=buf.page_end,
        token_count=token_count,
    )


def _extract_overlap_prefix(text: str, overlap_tokens: int, model_name: str) -> str:
    """Extract the last *overlap_tokens* worth of text as a carry-over prefix.

    Args:
        text: The text of the completed chunk.
        overlap_tokens: Maximum tokens to carry into the next chunk.
        model_name: Tokenizer model name.

    Returns:
        Overlap prefix string (may be empty if text is too short).
    """
    if overlap_tokens <= 0 or not text:
        return ""

    words = text.split()
    # Estimate: grab last 2× overlap words, then trim to exact token budget
    candidate = " ".join(words[-overlap_tokens * 2 :])
    # Binary-trim to overlap_tokens from the right
    total = count_tokens(candidate, model_name)
    if total <= overlap_tokens:
        return candidate

    # Drop words from the left until within budget
    candidate_words = candidate.split()
    while candidate_words and count_tokens(" ".join(candidate_words), model_name) > overlap_tokens:
        candidate_words.pop(0)

    return " ".join(candidate_words)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_segments(
    segments: list[SectionSegment],
    document_id: str,
    max_tokens: int = 700,
    overlap_tokens: int = 60,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[RawChunk]:
    """Produce token-bounded :class:`RawChunk` objects from section segments.

    Args:
        segments: Ordered list of :class:`SectionSegment` from section detection.
        document_id: Stable document identifier.
        max_tokens: Maximum tokens per chunk (hard ceiling).
        overlap_tokens: Tokens from the end of one chunk to carry into the next.
                        Set to 0 to disable overlap.
        model_name: Tokenizer model name (must match the embedding model).

    Returns:
        List of :class:`RawChunk` objects in document order.

    Raises:
        ChunkingError: If ``segments`` is empty or configuration is invalid.
    """
    if not segments:
        raise ChunkingError(
            message="Cannot chunk an empty segment list.",
            error_code="empty_segments",
            details={"document_id": document_id},
        )

    if max_tokens < 50:
        raise ChunkingError(
            message=f"max_tokens={max_tokens} is too small (minimum 50).",
            error_code="invalid_chunk_size",
            details={"document_id": document_id, "max_tokens": max_tokens},
        )

    chunks: list[RawChunk] = []
    chunk_index: int = 0
    overlap_prefix: str = ""

    # Determine effective budget accounting for overlap prefix
    def _effective_budget() -> int:
        if overlap_prefix:
            used = count_tokens(overlap_prefix, model_name)
            return max(_MIN_CHUNK_TOKENS, max_tokens - used)
        return max_tokens

    buf: _ChunkBuffer | None = None

    for seg in segments:
        paragraph = seg.text.strip()
        if not paragraph:
            continue

        para_tokens = count_tokens(paragraph, model_name)

        # --- Start new buffer if section changes or buffer not yet started ---
        if buf is None or seg.section != buf.section:
            # Flush existing buffer
            if buf is not None:
                chunk = _emit_chunk(buf, document_id, model_name, chunk_index)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                    overlap_prefix = _extract_overlap_prefix(
                        buf.text, overlap_tokens, model_name
                    )
                else:
                    overlap_prefix = ""

            # Start fresh buffer
            start_text = (overlap_prefix + "\n\n" + paragraph).strip() if overlap_prefix else paragraph
            buf = _ChunkBuffer(
                paragraphs=[start_text],
                page_start=seg.page_number,
                page_end=seg.page_number,
                section=seg.section,
                subsection=seg.subsection,
            )
            overlap_prefix = ""
            continue

        # --- Check if paragraph fits in the current buffer ---
        candidate_text = buf.text + "\n\n" + paragraph
        candidate_tokens = count_tokens(candidate_text, model_name)

        if candidate_tokens <= _effective_budget():
            # Fits — add to buffer
            buf.paragraphs.append(paragraph)
            buf.page_end = seg.page_number
        else:
            # Paragraph does NOT fit — flush buffer then handle paragraph
            if para_tokens <= max_tokens:
                # Paragraph fits in a fresh chunk — flush current and start new
                chunk = _emit_chunk(buf, document_id, model_name, chunk_index)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                    overlap_prefix = _extract_overlap_prefix(
                        buf.text, overlap_tokens, model_name
                    )
                else:
                    overlap_prefix = ""

                start_text = (overlap_prefix + "\n\n" + paragraph).strip() if overlap_prefix else paragraph
                buf = _ChunkBuffer(
                    paragraphs=[start_text],
                    page_start=seg.page_number,
                    page_end=seg.page_number,
                    section=seg.section,
                    subsection=seg.subsection,
                )
                overlap_prefix = ""
            else:
                # Paragraph alone exceeds max_tokens — split it
                remaining = paragraph
                first_in_split = True

                while remaining:
                    within, overflow = split_to_token_budget(remaining, max_tokens, model_name)
                    if not within:
                        # Degenerate: single token word that exceeds budget — hard break
                        within = remaining[: max_tokens * _CHARS_PER_TOKEN_APPROX]
                        overflow = remaining[max_tokens * _CHARS_PER_TOKEN_APPROX :]

                    if first_in_split:
                        # Flush existing buffer first
                        chunk = _emit_chunk(buf, document_id, model_name, chunk_index)
                        if chunk:
                            chunks.append(chunk)
                            chunk_index += 1
                            overlap_prefix = _extract_overlap_prefix(
                                buf.text, overlap_tokens, model_name
                            )
                        first_in_split = False

                    start_text = (overlap_prefix + "\n\n" + within).strip() if overlap_prefix else within
                    chunk_text = start_text.strip()
                    tok = count_tokens(chunk_text, model_name)
                    if tok >= _MIN_CHUNK_TOKENS:
                        rc = RawChunk(
                            chunk_id=generate_chunk_id(document_id, chunk_index),
                            document_id=document_id,
                            text=chunk_text,
                            section=seg.section,
                            subsection=seg.subsection,
                            page_start=seg.page_number,
                            page_end=seg.page_number,
                            token_count=tok,
                        )
                        chunks.append(rc)
                        chunk_index += 1
                        overlap_prefix = _extract_overlap_prefix(
                            chunk_text, overlap_tokens, model_name
                        )

                    remaining = overflow

                # After splitting, start a fresh buffer (empty)
                buf = _ChunkBuffer(
                    paragraphs=[],
                    page_start=seg.page_number,
                    page_end=seg.page_number,
                    section=seg.section,
                    subsection=seg.subsection,
                )

    # --- Flush remaining buffer ---
    if buf is not None:
        chunk = _emit_chunk(buf, document_id, model_name, chunk_index)
        if chunk:
            chunks.append(chunk)

    logger.info(
        "Chunking complete",
        extra={
            "extra_data": {
                "document_id": document_id,
                "total_chunks": len(chunks),
                "max_tokens": max_tokens,
                "overlap_tokens": overlap_tokens,
            }
        },
    )

    return chunks


# Approximate chars per token (used only in degenerate hard-split fallback)
_CHARS_PER_TOKEN_APPROX = 4

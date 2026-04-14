"""Answer assembly service.

Formats raw :class:`PineconeQueryResult` matches into structured
:class:`QueryResponse` API objects with evidence text, metadata, and
relevance scores.
"""

from __future__ import annotations

import logging
from typing import Any

from execo_rag.api.schemas.query import QueryResponse, QueryResultChunk
from execo_rag.models.metadata import SectionType
from execo_rag.models.pinecone import PineconeMatch, PineconeQueryResult

logger = logging.getLogger(__name__)

# Score threshold below which matches are considered too weak to include
_MIN_RELEVANCE_SCORE = 0.0  # Pinecone scores are cosine similarity [0, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(metadata: dict[str, Any] | None) -> str:
    """Extract chunk text from Pinecone metadata dict."""
    if not metadata:
        return ""
    return str(metadata.get("chunk_text") or metadata.get("text") or "")


def _extract_section(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return SectionType.GENERAL.value

    raw = metadata.get("section", SectionType.GENERAL)

    # 🔥 FIX: handle Enum properly
    if isinstance(raw, SectionType):
        return raw.value

    # handle string like "SectionType.ESCROW"
    if isinstance(raw, str) and raw.startswith("SectionType."):
        return raw.split(".")[-1].lower()

    return str(raw)


def _extract_page_start(metadata: dict[str, Any] | None) -> int:
    """Extract page_start from metadata (default 1)."""
    if not metadata:
        return 1
    try:
        return int(metadata.get("page_start", 1))
    except (TypeError, ValueError):
        return 1


def _extract_page_end(metadata: dict[str, Any] | None) -> int:
    """Extract page_end from metadata (default 1)."""
    if not metadata:
        return 1
    try:
        return int(metadata.get("page_end", 1))
    except (TypeError, ValueError):
        return 1


def _match_to_result_chunk(match: PineconeMatch) -> QueryResultChunk:
    """Convert a single Pinecone match to a :class:`QueryResultChunk`."""
    if isinstance(match.metadata, dict):
        meta = match.metadata
    elif match.metadata is not None:
        meta = match.metadata.model_dump()
    else:
        meta = {}
    return QueryResultChunk(
        chunk_id=match.id,
        score=match.score,
        text=_extract_text(meta),
        section=_extract_section(meta),
        page_start=_extract_page_start(meta),
        page_end=_extract_page_end(meta),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_response(
    query: str,
    request_id: str,
    retrieval_result: PineconeQueryResult,
    min_score: float = _MIN_RELEVANCE_SCORE,
) -> QueryResponse:
    """Build a :class:`QueryResponse` from Pinecone query results.

    Args:
        query: The original user query string.
        request_id: The request identifier for traceability.
        retrieval_result: Raw :class:`PineconeQueryResult` from the retriever.
        min_score: Minimum relevance score to include a match (default 0.0).

    Returns:
        Structured :class:`QueryResponse` ready to be returned to the API caller.
    """
    logger.debug(
        "Assembling query response",
        extra={
            "extra_data": {
                "request_id": request_id,
                "raw_match_count": len(retrieval_result.matches),
                "min_score": min_score,
            }
        },
    )

    # Filter by score threshold and convert matches to result chunks
    result_chunks: list[QueryResultChunk] = []
    for match in retrieval_result.matches:
        if match.score < min_score:
            logger.debug(
                "Match below score threshold; skipping",
                extra={
                    "extra_data": {
                        "chunk_id": match.id,
                        "score": match.score,
                        "threshold": min_score,
                    }
                },
            )
            continue
        result_chunks.append(_match_to_result_chunk(match))

    logger.info(
        "Query response assembled",
        extra={
            "extra_data": {
                "request_id": request_id,
                "results_returned": len(result_chunks),
                "results_filtered": len(retrieval_result.matches) - len(result_chunks),
            }
        },
    )

    return QueryResponse(
        request_id=request_id,
        query=query,
        results=result_chunks,
    )


def assemble_empty_response(query: str, request_id: str) -> QueryResponse:
    """Return an empty :class:`QueryResponse` for queries with no results.

    Args:
        query: Original user query.
        request_id: Request identifier.

    Returns:
        :class:`QueryResponse` with an empty results list.
    """
    logger.info(
        "No results found for query; returning empty response",
        extra={"extra_data": {"request_id": request_id, "query_preview": query[:80]}},
    )
    return QueryResponse(request_id=request_id, query=query, results=[])

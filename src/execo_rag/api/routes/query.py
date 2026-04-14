"""Query API route: POST /query.

Accepts a natural-language query and optional metadata filters, runs
semantic retrieval against Pinecone, and returns ranked result chunks.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from execo_rag.api.deps import get_app_settings, get_request_logger
from execo_rag.api.schemas.query import QueryRequest, QueryResponse
from execo_rag.config import Settings
from execo_rag.services.query.query_service import run_query
from execo_rag.utils.exceptions import EmbeddingError, VectorStoreError

router = APIRouter(prefix="/query", tags=["retrieval"])

logger = logging.getLogger(__name__)


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic search with optional metadata filters",
    description=(
        "Embeds the query, applies optional SPA metadata filters, retrieves the "
        "top-k matching chunks from Pinecone, and returns them with relevance scores."
    ),
)
async def query(
    request: Request,
    body: QueryRequest,
    req_logger: logging.Logger = Depends(get_request_logger),
    settings: Settings = Depends(get_app_settings),
) -> QueryResponse:
    """Query indexed documents using semantic search.

    - **query**: Natural-language query string.
    - **top_k**: Number of results to return (1–25, default 5).
    - **filters**: Optional metadata filters (buyer, section, governing_law, etc.).
    """
    request_id: str = getattr(request.state, "request_id", "req_unknown")

    req_logger.info(
        "Query request received",
        extra={
            "extra_data": {
                "query_preview": body.query[:80],
                "top_k": body.top_k,
                "has_filters": body.filters is not None,
            }
        },
    )

    try:
        response = run_query(
            request_id=request_id,
            query_request=body,
            settings=settings,
        )
    except EmbeddingError as exc:
        logger.error(
            "Query embedding failed",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "error_code": exc.error_code,
                    "error": exc.message,
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error_code": exc.error_code,
                "message": "Failed to generate query embedding. Check the configured embedding provider and model.",
                "request_id": request_id,
            },
        ) from exc
    except VectorStoreError as exc:
        logger.error(
            "Pinecone query failed",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "error_code": exc.error_code,
                    "error": exc.message,
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error_code": exc.error_code,
                "message": "Vector store retrieval failed. Check Pinecone configuration.",
                "request_id": request_id,
            },
        ) from exc
    except Exception as exc:
        logger.error(
            "Query failed with unexpected error",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "error": str(exc),
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "internal_error",
                "message": "An unexpected error occurred during retrieval.",
                "request_id": request_id,
            },
        ) from exc

    req_logger.info(
        "Query request completed",
        extra={
            "extra_data": {
                "results_returned": len(response.results),
            }
        },
    )

    return response

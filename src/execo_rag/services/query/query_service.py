"""Query orchestration service.

Accepts a :class:`QueryRequest`, resolves metadata filters, embeds the query,
retrieves Pinecone matches, and assembles a structured :class:`QueryResponse`.

Public entry point::

    response = run_query(request_id, query_request, settings)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from execo_rag.api.schemas.query import QueryRequest, QueryResponse
from execo_rag.config import Settings
from execo_rag.logging.metrics import record_query
from execo_rag.services.embeddings.provider import create_embedding_provider
from execo_rag.services.query.answer_builder import assemble_empty_response, assemble_response
from execo_rag.services.query.retriever import QueryRetriever
from execo_rag.services.vectorstore.filter_builder import build_filter_from_params
from execo_rag.services.vectorstore.pinecone_store import PineconeStore
from execo_rag.utils.exceptions import EmbeddingError, VectorStoreError

logger = logging.getLogger(__name__)


def _filters_to_params(query_request: QueryRequest) -> dict[str, Any]:
    """Convert ``QueryFilters`` Pydantic model to a flat params dict.

    Returns an empty dict if no filters were provided.
    """
    if not query_request.filters:
        return {}

    f = query_request.filters
    params: dict[str, Any] = {}

    if f.document_type:
        params["document_type"] = f.document_type
    if f.buyer:
        params["buyer"] = f.buyer
    if f.seller:
        params["seller"] = f.seller
    if f.company_target:
        params["company_target"] = f.company_target
    if f.governing_law:
        params["governing_law"] = f.governing_law
    if f.section:
        params["section"] = f.section.value
    if f.effective_date:
        params["effective_date"] = f.effective_date.isoformat()
    if f.page_start is not None:
        params["page_start"] = f.page_start
    if f.page_end is not None:
        params["page_end"] = f.page_end

    return params


def run_query(
    request_id: str,
    query_request: QueryRequest,
    settings: Settings,
) -> QueryResponse:
    """Execute a semantic + metadata-filtered retrieval query.

    Args:
        request_id: Request identifier for traceability.
        query_request: Validated :class:`QueryRequest` from the API layer.
        settings: Application settings (for API keys, namespace, etc.).

    Returns:
        :class:`QueryResponse` with ranked result chunks.
    """
    start_ts = time.perf_counter()

    logger.info(
        "Query request started",
        extra={
            "extra_data": {
                "request_id": request_id,
                "query_preview": query_request.query[:100],
                "top_k": query_request.top_k,
                "has_filters": query_request.filters is not None,
            }
        },
    )

    # ── Build provider and store ───────────────────────────────────────
    embedding_provider = create_embedding_provider(
        provider_name=settings.embeddings.provider,
        model_name=settings.embeddings.model,
        batch_size=settings.embeddings.batch_size,
        api_key=settings.openrouter.api_key,
        expected_dimension=settings.embeddings.dimension,
    )
    pinecone_store = PineconeStore(
        api_key=settings.pinecone.api_key,
        index_name=settings.pinecone.index_name,
        namespace=settings.pinecone.namespace,
    )
    retriever = QueryRetriever(
        embedding_provider=embedding_provider,
        pinecone_store=pinecone_store,
        default_top_k=query_request.top_k,
    )

    # ── Resolve filters ────────────────────────────────────────────────
    filter_params = _filters_to_params(query_request)
    if filter_params:
        prebuilt_filter = build_filter_from_params(**filter_params)
        logger.debug(
            "Query filters resolved",
            extra={"extra_data": {"request_id": request_id, "filter": prebuilt_filter}},
        )
    else:
        prebuilt_filter = None

    # ── Retrieve ───────────────────────────────────────────────────────
    try:
        retrieval_result = retriever.retrieve_with_filter(
            query=query_request.query,
            top_k=query_request.top_k,
            prebuilt_filter=prebuilt_filter,
            namespace=settings.pinecone.namespace,
        )
    except (EmbeddingError, VectorStoreError) as exc:
        logger.error(
            "Query retrieval failed",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            },
        )
        elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
        record_query(match_count=0, duration_ms=elapsed_ms, error=True)
        raise

    # ── Assemble response ──────────────────────────────────────────────
    if not retrieval_result.matches:
        response = assemble_empty_response(
            query=query_request.query,
            request_id=request_id,
        )
    else:
        response = assemble_response(
            query=query_request.query,
            request_id=request_id,
            retrieval_result=retrieval_result,
        )

    logger.debug(
        "Response assembled",
        extra={
            "extra_data": {
                "request_id": request_id,
                "results_count": len(response.results),
                "top_score": response.results[0].score if response.results else None,
            }
        },
    )

    elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
    record_query(match_count=len(response.results), duration_ms=elapsed_ms, error=False)

    logger.info(
        "Query request completed",
        extra={
            "extra_data": {
                "request_id": request_id,
                "results_returned": len(response.results),
                "duration_ms": elapsed_ms,
            }
        },
    )

    return response

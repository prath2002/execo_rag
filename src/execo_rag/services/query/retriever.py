"""Query retrieval service.

Combines semantic embedding of the user query with Pinecone metadata
filtering to retrieve the most relevant chunk records.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from execo_rag.models.pinecone import PineconeQueryRequest, PineconeQueryResult
from execo_rag.services.embeddings.provider import EmbeddingProvider
from execo_rag.services.vectorstore.filter_builder import Filter, build_filter_from_params
from execo_rag.services.vectorstore.pinecone_store import PineconeStore
from execo_rag.utils.exceptions import EmbeddingError, VectorStoreError

logger = logging.getLogger(__name__)


class QueryRetriever:
    """Orchestrates query embedding and Pinecone retrieval.

    Args:
        embedding_provider: An :class:`EmbeddingProvider` instance.
        pinecone_store: A connected :class:`PineconeStore` instance.
        default_top_k: Default number of results to return.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        pinecone_store: PineconeStore,
        default_top_k: int = 5,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._store = pinecone_store
        self._default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_params: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> PineconeQueryResult:
        """Embed a query and retrieve matching chunks from Pinecone.

        Args:
            query: Natural-language query string.
            top_k: Number of top results to return (default: ``default_top_k``).
            filter_params: Optional dict of filter parameters accepted by
                           :func:`~services.vectorstore.filter_builder.build_filter_from_params`.
                           Pass ``None`` or ``{}`` for no filtering.
            namespace: Pinecone namespace override.

        Returns:
            :class:`PineconeQueryResult` with ranked matches and metadata.

        Raises:
            EmbeddingError: If query embedding fails.
            VectorStoreError: If the Pinecone query fails.
        """
        if not query or not query.strip():
            raise ValueError("Query string must not be empty.")

        k = top_k or self._default_top_k
        start_ts = time.perf_counter()

        logger.info(
            "Query retrieval started",
            extra={
                "extra_data": {
                    "query_preview": query[:100],
                    "top_k": k,
                    "has_filters": bool(filter_params),
                    "namespace": namespace,
                }
            },
        )

        # --- Step 1: Embed the query ---
        try:
            query_vector = self._embedding_provider.embed_query(query)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                message=f"Query embedding failed: {exc}",
                error_code="query_embedding_failed",
                details={"query_preview": query[:100], "error": str(exc)},
            ) from exc

        logger.debug(
            "Query embedded",
            extra={"extra_data": {"vector_dim": len(query_vector)}},
        )

        # --- Step 2: Build metadata filter ---
        pinecone_filter: Filter | None = None
        if filter_params:
            pinecone_filter = build_filter_from_params(**filter_params)
            if not pinecone_filter:
                pinecone_filter = None  # Empty filter → no filter

        # --- Step 3: Query Pinecone ---
        request = PineconeQueryRequest(
            vector=query_vector,
            top_k=k,
            namespace=namespace,
            filter=pinecone_filter,
            include_metadata=True,
            include_values=False,
        )

        try:
            result = self._store.query(request, namespace=namespace)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                message=f"Pinecone retrieval failed: {exc}",
                error_code="retrieval_failed",
                details={"top_k": k, "error": str(exc)},
            ) from exc

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
        logger.info(
            "Query retrieval complete",
            extra={
                "extra_data": {
                    "matches_returned": len(result.matches),
                    "top_k_requested": k,
                    "duration_ms": elapsed_ms,
                }
            },
        )

        return result

    def retrieve_with_filter(
        self,
        query: str,
        top_k: int | None = None,
        prebuilt_filter: Filter | None = None,
        namespace: str | None = None,
    ) -> PineconeQueryResult:
        """Retrieve using a pre-built Pinecone filter dict.

        Useful when the filter has been constructed by :class:`SPAFilterBuilder`
        or :func:`build_filter_from_params` before the retrieval call.

        Args:
            query: Natural-language query string.
            top_k: Number of results.
            prebuilt_filter: Pre-built Pinecone filter dict or ``None``.
            namespace: Namespace override.

        Returns:
            :class:`PineconeQueryResult`.
        """
        if not query or not query.strip():
            raise ValueError("Query string must not be empty.")

        k = top_k or self._default_top_k
        start_ts = time.perf_counter()

        logger.info(
            "Query retrieval started",
            extra={
                "extra_data": {
                    "query_preview": query[:100],
                    "top_k": k,
                    "has_filter": prebuilt_filter is not None,
                    "namespace": namespace,
                }
            },
        )

        query_vector = self._embedding_provider.embed_query(query)

        logger.debug(
            "Query embedded",
            extra={"extra_data": {"vector_dim": len(query_vector)}},
        )

        request = PineconeQueryRequest(
            vector=query_vector,
            top_k=k,
            namespace=namespace,
            filter=prebuilt_filter,
            include_metadata=True,
            include_values=False,
        )

        result = self._store.query(request, namespace=namespace)

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
        logger.info(
            "Query retrieval complete",
            extra={
                "extra_data": {
                    "matches_returned": len(result.matches),
                    "top_k_requested": k,
                    "top_score": result.matches[0].score if result.matches else None,
                    "duration_ms": elapsed_ms,
                }
            },
        )

        return result

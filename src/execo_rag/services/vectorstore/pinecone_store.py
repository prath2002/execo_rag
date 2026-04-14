"""Pinecone vector store service.

Wraps the Pinecone Python SDK (v3+) to provide:
  - Upsert: write vector records in batches
  - Fetch: retrieve vectors by ID
  - Delete: remove vectors by ID or namespace
  - Query: semantic search with optional metadata filters

All operations are namespace-aware and include structured logging.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from decimal import Decimal
from typing import Any

from execo_rag.models.chunk import ValidatedChunk
from execo_rag.models.embedding import EmbeddingBatchResult
from execo_rag.models.pinecone import (
    PineconeMatch,
    PineconeQueryRequest,
    PineconeQueryResult,
)
from execo_rag.utils.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

# Default batch size for upsert operations
_UPSERT_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Metadata serialization helpers
# ---------------------------------------------------------------------------


def _serialize_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert non-serializable types in metadata to Pinecone-safe values.

    Pinecone metadata values must be: str, int, float, bool, or list of those.

    Args:
        raw: Raw metadata dictionary (may contain Decimal, date, etc.).

    Returns:
        Sanitized metadata dictionary.
    """
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue  # Pinecone ignores None; omit to save space
        elif isinstance(value, Decimal):
            # Store as float for Pinecone numeric range filters
            result[key] = float(value)
        elif isinstance(value, date):
            # Store ISO string for exact-match string filters
            result[key] = value.isoformat()
        elif isinstance(value, bool):
            result[key] = value
        elif isinstance(value, (int, float, str)):
            result[key] = value
        elif hasattr(value, "value"):
            # Enum-like objects (e.g. SectionType)
            result[key] = str(value.value)
        else:
            result[key] = str(value)
    return result


def _chunk_metadata_to_dict(chunk: ValidatedChunk) -> dict[str, Any]:
    """Extract Pinecone-safe metadata dict from a validated chunk."""
    meta = chunk.metadata
    raw: dict[str, Any] = {
        "document_id": meta.document_id,
        "document_type": meta.document_type,
        "effective_date": meta.effective_date,
        "buyer": meta.buyer,
        "company_target": meta.company_target,
        "seller": meta.seller,
        "shares_transacted": meta.shares_transacted,
        "cash_purchase_price": meta.cash_purchase_price,
        "escrow_agent": meta.escrow_agent,
        "escrow_amount": meta.escrow_amount,
        "target_working_capital": meta.target_working_capital,
        "indemnification_de_minimis_amount": meta.indemnification_de_minimis_amount,
        "indemnification_basket_amount": meta.indemnification_basket_amount,
        "indemnification_cap_amount": meta.indemnification_cap_amount,
        "governing_law": meta.governing_law,
        "section": meta.section.value if meta.section else None,
        "subsection": meta.subsection,
        "page_start": meta.page_start,
        "page_end": meta.page_end,
        "has_escrow": meta.has_escrow,
        "has_indemnification": meta.has_indemnification,
        "has_purchase_price": meta.has_purchase_price,
        "has_working_capital": meta.has_working_capital,
        "chunk_text": meta.chunk_text,
    }
    return _serialize_metadata(raw)


# ---------------------------------------------------------------------------
# Pinecone store
# ---------------------------------------------------------------------------


class PineconeStore:
    """Pinecone vector store client wrapping the official SDK.

    Args:
        api_key: Pinecone API key.
        index_name: Name of the target Pinecone index.
        namespace: Default namespace for all operations (can be overridden per call).
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "default",
    ) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._namespace = namespace
        self._client: object | None = None
        self._index: object | None = None
        self._index_dimension: int | None = None

    def _get_client(self) -> object:
        """Lazily initialize and return the Pinecone control-plane client."""

        if self._client is None:
            try:
                from pinecone import Pinecone  # type: ignore[import-untyped]
            except ImportError as exc:
                raise VectorStoreError(
                    message="pinecone package is required. Install with: pip install pinecone",
                    error_code="missing_pinecone_package",
                ) from exc

            self._client = Pinecone(api_key=self._api_key)

        return self._client

    def _get_index(self) -> object:
        """Lazily initialize and return the Pinecone index client."""
        if self._index is None:
            try:
                client = self._get_client()
                self._index = client.Index(self._index_name)  # type: ignore[union-attr]
                logger.info(
                    "Pinecone index client initialized",
                    extra={
                        "extra_data": {
                            "index_name": self._index_name,
                            "namespace": self._namespace,
                        }
                    },
                )
            except Exception as exc:
                raise VectorStoreError(
                    message=f"Failed to connect to Pinecone index '{self._index_name}': {exc}",
                    error_code="pinecone_connect_error",
                    details={"index_name": self._index_name, "error": str(exc)},
                ) from exc

        return self._index

    def _get_index_dimension(self) -> int | None:
        """Return the configured Pinecone index dimension when available."""

        if self._index_dimension is not None:
            return self._index_dimension

        try:
            client = self._get_client()
            description = client.describe_index(self._index_name)  # type: ignore[union-attr]
            dimension = getattr(description, "dimension", None)
            if isinstance(dimension, int):
                self._index_dimension = dimension
                return dimension
        except Exception as exc:
            logger.warning(
                "Could not determine Pinecone index dimension",
                extra={
                    "extra_data": {
                        "index_name": self._index_name,
                        "error": str(exc),
                    }
                },
            )

        return None

    def _validate_vector_dimension(self, vector: list[float]) -> None:
        """Ensure the vector matches the target Pinecone index dimension."""

        expected_dimension = self._get_index_dimension()
        if expected_dimension is None:
            return

        actual_dimension = len(vector)
        if actual_dimension != expected_dimension:
            raise VectorStoreError(
                message=(
                    f"Embedding dimension {actual_dimension} does not match Pinecone "
                    f"index dimension {expected_dimension} for index '{self._index_name}'."
                ),
                error_code="pinecone_dimension_mismatch",
                details={
                    "index_name": self._index_name,
                    "expected_dimension": expected_dimension,
                    "actual_dimension": actual_dimension,
                },
            )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[ValidatedChunk],
        embeddings: EmbeddingBatchResult,
        namespace: str | None = None,
    ) -> int:
        """Upsert chunk vectors and metadata into Pinecone.

        Builds a mapping of chunk_id → (vector, metadata) and upserts in
        batches of ``_UPSERT_BATCH_SIZE``.

        Args:
            chunks: Validated chunks (only ``is_valid=True`` chunks are indexed).
            embeddings: Embedding vectors aligned with valid chunks.
            namespace: Override the default namespace.

        Returns:
            Number of vectors successfully upserted.

        Raises:
            VectorStoreError: If the upsert fails.
        """
        ns = namespace or self._namespace
        index = self._get_index()

        # Build chunk_id → metadata mapping
        valid_chunks = [c for c in chunks if c.is_valid]
        chunk_meta_map = {c.chunk_id: _chunk_metadata_to_dict(c) for c in valid_chunks}
        vec_map = {v.chunk_id: v.values for v in embeddings.vectors}

        # Align: only include chunks that have both metadata and a vector
        records: list[dict[str, Any]] = []
        for chunk_id, meta in chunk_meta_map.items():
            values = vec_map.get(chunk_id)
            if values is None:
                logger.warning(
                    "No embedding found for chunk; skipping",
                    extra={"extra_data": {"chunk_id": chunk_id}},
                )
                continue
            records.append({"id": chunk_id, "values": values, "metadata": meta})

        if not records:
            raise VectorStoreError(
                message="No records to upsert after alignment.",
                error_code="empty_upsert",
            )

        self._validate_vector_dimension(records[0]["values"])

        start_ts = time.perf_counter()
        total_upserted = 0

        try:
            for i in range(0, len(records), _UPSERT_BATCH_SIZE):
                batch = records[i : i + _UPSERT_BATCH_SIZE]
                index.upsert(vectors=batch, namespace=ns)  # type: ignore[union-attr]
                total_upserted += len(batch)
                logger.debug(
                    "Upserted batch",
                    extra={
                        "extra_data": {
                            "batch_start": i,
                            "batch_size": len(batch),
                            "namespace": ns,
                        }
                    },
                )
        except Exception as exc:
            raise VectorStoreError(
                message=f"Pinecone upsert failed: {exc}",
                error_code="pinecone_upsert_error",
                details={"namespace": ns, "error": str(exc)},
            ) from exc

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
        logger.info(
            "Pinecone upsert complete",
            extra={
                "extra_data": {
                    "total_upserted": total_upserted,
                    "namespace": ns,
                    "duration_ms": elapsed_ms,
                }
            },
        )
        return total_upserted

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch(
        self,
        ids: list[str],
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Fetch vectors by ID from Pinecone.

        Args:
            ids: List of vector IDs to fetch.
            namespace: Override the default namespace.

        Returns:
            Raw Pinecone fetch response as a dict.

        Raises:
            VectorStoreError: If the fetch fails.
        """
        ns = namespace or self._namespace
        index = self._get_index()

        try:
            response = index.fetch(ids=ids, namespace=ns)  # type: ignore[union-attr]
            return dict(response)
        except Exception as exc:
            raise VectorStoreError(
                message=f"Pinecone fetch failed: {exc}",
                error_code="pinecone_fetch_error",
                details={"ids": ids, "namespace": ns, "error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_ids(
        self,
        ids: list[str],
        namespace: str | None = None,
    ) -> None:
        """Delete vectors by ID.

        Args:
            ids: List of vector IDs to delete.
            namespace: Override the default namespace.

        Raises:
            VectorStoreError: If the delete fails.
        """
        ns = namespace or self._namespace
        index = self._get_index()

        try:
            index.delete(ids=ids, namespace=ns)  # type: ignore[union-attr]
            logger.info(
                "Deleted vectors from Pinecone",
                extra={"extra_data": {"count": len(ids), "namespace": ns}},
            )
        except Exception as exc:
            raise VectorStoreError(
                message=f"Pinecone delete failed: {exc}",
                error_code="pinecone_delete_error",
                details={"ids": ids, "namespace": ns, "error": str(exc)},
            ) from exc

    def delete_namespace(self, namespace: str | None = None) -> None:
        """Delete all vectors in a namespace.

        Args:
            namespace: Namespace to clear (defaults to the store's namespace).

        Raises:
            VectorStoreError: If the delete fails.
        """
        ns = namespace or self._namespace
        index = self._get_index()

        try:
            index.delete(delete_all=True, namespace=ns)  # type: ignore[union-attr]
            logger.info(
                "Deleted all vectors in namespace",
                extra={"extra_data": {"namespace": ns}},
            )
        except Exception as exc:
            raise VectorStoreError(
                message=f"Pinecone namespace delete failed: {exc}",
                error_code="pinecone_namespace_delete_error",
                details={"namespace": ns, "error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        request: PineconeQueryRequest,
        namespace: str | None = None,
    ) -> PineconeQueryResult:
        """Run a semantic similarity query against Pinecone.

        Args:
            request: :class:`PineconeQueryRequest` with vector, top_k, and filters.
            namespace: Override the default namespace.

        Returns:
            :class:`PineconeQueryResult` with ranked matches.

        Raises:
            VectorStoreError: If the query fails.
        """
        ns = namespace or request.namespace or self._namespace
        index = self._get_index()

        logger.debug(
            "Querying Pinecone",
            extra={
                "extra_data": {
                    "top_k": request.top_k,
                    "namespace": ns,
                    "has_filter": request.filter is not None,
                }
            },
        )

        try:
            self._validate_vector_dimension(request.vector)
            kwargs: dict[str, Any] = {
                "vector": request.vector,
                "top_k": request.top_k,
                "namespace": ns,
                "include_metadata": request.include_metadata,
                "include_values": request.include_values,
            }
            if request.filter:
                kwargs["filter"] = request.filter

            response = index.query(**kwargs)  # type: ignore[union-attr]
        except Exception as exc:
            raise VectorStoreError(
                message=f"Pinecone query failed: {exc}",
                error_code="pinecone_query_error",
                details={"namespace": ns, "top_k": request.top_k, "error": str(exc)},
            ) from exc

        matches: list[PineconeMatch] = []
        for match in getattr(response, "matches", []):
            matches.append(
                PineconeMatch(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata if request.include_metadata else None,
                    values=match.values if request.include_values else None,
                )
            )

        logger.debug(
            "Pinecone query returned matches",
            extra={"extra_data": {"match_count": len(matches), "namespace": ns}},
        )

        return PineconeQueryResult(matches=matches)

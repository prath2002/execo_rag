"""Embedding provider abstractions and implementations.

The default embedding path is a free local sentence-transformers model:
``sentence-transformers/all-MiniLM-L6-v2`` which produces 384-dimensional
vectors suitable for a matching Pinecone index.

An OpenRouter provider remains available as an optional alternative, but the
runtime now selects providers through :func:`create_embedding_provider`.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

from execo_rag.clients import OpenRouterClient
from execo_rag.models.chunk import ValidatedChunk
from execo_rag.models.embedding import EmbeddingBatchResult, EmbeddingRequest, EmbeddingVector
from execo_rag.utils.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 32
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_TRUNCATE_CHARS = 32_000

# Module-level provider cache so the sentence-transformers model is loaded
# from disk only once per process, not once per request.
_PROVIDER_CACHE: dict[str, "SentenceTransformerEmbeddingProvider"] = {}


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for embedding generation services."""

    def embed(self, request: EmbeddingRequest) -> EmbeddingBatchResult:
        """Generate embeddings for a batch of input texts."""
        ...  # pragma: no cover

    def embed_query(self, text: str) -> list[float]:
        """Generate a single embedding vector for a query string."""
        ...  # pragma: no cover


def _truncate_inputs(texts: list[str]) -> list[str]:
    """Trim very long inputs so providers are not asked to encode huge strings."""

    return [text[:_TRUNCATE_CHARS] for text in texts]


def _build_embedding_result(
    model_name: str,
    vectors: list[list[float]],
) -> EmbeddingBatchResult:
    """Convert raw vectors into the project's embedding result model."""

    embedding_vectors = [
        EmbeddingVector(
            chunk_id=str(idx),
            values=values,
            dimension=len(values),
        )
        for idx, values in enumerate(vectors)
    ]
    return EmbeddingBatchResult(model_name=model_name, vectors=embedding_vectors)


class SentenceTransformerEmbeddingProvider:
    """Local embedding provider backed by sentence-transformers."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = _DEFAULT_BATCH_SIZE,
        expected_dimension: int | None = 384,
        normalize_embeddings: bool = True,
        device: str = "cpu",
    ) -> None:
        self._model_name = model
        self._batch_size = batch_size
        self._expected_dimension = expected_dimension
        self._normalize_embeddings = normalize_embeddings
        self._device = device
        self._model: object | None = None

    def _get_model(self) -> object:
        """Load the sentence-transformers model on first use."""

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            except ImportError as exc:
                raise EmbeddingError(
                    message=(
                        "sentence-transformers is required for local embeddings. "
                        "Install with: pip install sentence-transformers"
                    ),
                    error_code="missing_sentence_transformers_package",
                ) from exc

            try:
                self._model = SentenceTransformer(self._model_name, device=self._device)
            except Exception as exc:
                raise EmbeddingError(
                    message=f"Failed to load local embedding model '{self._model_name}': {exc}",
                    error_code="local_embedding_model_load_failed",
                    details={"model": self._model_name, "error": str(exc)},
                ) from exc

        return self._model

    def _validate_dimensions(self, vectors: list[list[float]]) -> None:
        """Ensure generated vectors match the configured embedding dimension."""

        if self._expected_dimension is None:
            return

        for idx, vector in enumerate(vectors):
            if len(vector) != self._expected_dimension:
                raise EmbeddingError(
                    message=(
                        f"Embedding vector dimension mismatch for model '{self._model_name}': "
                        f"expected {self._expected_dimension}, got {len(vector)}"
                    ),
                    error_code="embedding_dimension_mismatch",
                    details={
                        "model": self._model_name,
                        "expected_dimension": self._expected_dimension,
                        "actual_dimension": len(vector),
                        "vector_index": idx,
                    },
                )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts locally and return plain Python float lists."""

        model = self._get_model()
        safe_inputs = _truncate_inputs(texts)

        try:
            matrix = model.encode(  # type: ignore[union-attr]
                safe_inputs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(
                message=f"Local embedding generation failed: {exc}",
                error_code="local_embedding_failed",
                details={"model": self._model_name, "error": str(exc)},
            ) from exc

        vectors = matrix.tolist()
        if vectors and isinstance(vectors[0], float):
            vectors = [vectors]

        self._validate_dimensions(vectors)
        return vectors

    def embed(self, request: EmbeddingRequest) -> EmbeddingBatchResult:
        """Generate local embeddings for all inputs in the request."""

        if not request.inputs:
            raise EmbeddingError(message="Cannot embed an empty input list.", error_code="empty_inputs")

        start_ts = time.perf_counter()
        logger.info(
            "Local embedding batch started",
            extra={
                "extra_data": {
                    "provider": "sentence_transformers",
                    "model": self._model_name,
                    "total_inputs": len(request.inputs),
                    "batch_size": self._batch_size,
                }
            },
        )

        vectors = self._encode(request.inputs)

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
        logger.info(
            "Local embedding batch complete",
            extra={
                "extra_data": {
                    "provider": "sentence_transformers",
                    "model": self._model_name,
                    "total_vectors": len(vectors),
                    "dimension": len(vectors[0]) if vectors else 0,
                    "duration_ms": elapsed_ms,
                }
            },
        )

        return _build_embedding_result(self._model_name, vectors)

    def embed_query(self, text: str) -> list[float]:
        """Generate a single local embedding vector for a query string."""

        if not text.strip():
            raise EmbeddingError(
                message="Cannot embed an empty query string.",
                error_code="empty_query_embedding",
            )

        vectors = self._encode([text])
        if not vectors:
            raise EmbeddingError(
                message="Local embedding provider returned no query vector.",
                error_code="empty_query_embedding",
            )
        return vectors[0]


class OpenRouterEmbeddingProvider:
    """Optional embedding provider backed by the OpenRouter embeddings API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._batch_size = batch_size
        self._client: OpenRouterClient | None = None

    def _get_client(self) -> OpenRouterClient:
        """Lazily initialize and return the OpenRouter client."""

        if self._client is None:
            self._client = OpenRouterClient(api_key=self._api_key)

        return self._client

    def _call_api_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenRouter embeddings API with exponential-backoff retries."""

        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = client.embeddings.create(  # type: ignore[union-attr]
                    model=self._model,
                    input=_truncate_inputs(texts),
                )
                return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            except Exception as exc:
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Embedding API call failed; retrying",
                    extra={
                        "extra_data": {
                            "provider": "openrouter",
                            "attempt": attempt,
                            "max_retries": _MAX_RETRIES,
                            "delay_s": delay,
                            "error": str(exc),
                        }
                    },
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(delay)

        raise EmbeddingError(
            message=f"Embedding API failed after {_MAX_RETRIES} attempts: {last_exc}",
            error_code="embedding_api_exhausted",
            details={"model": self._model, "batch_size": len(texts), "error": str(last_exc)},
        ) from last_exc

    def embed(self, request: EmbeddingRequest) -> EmbeddingBatchResult:
        """Generate OpenRouter embeddings for all inputs in the request."""

        if not request.inputs:
            raise EmbeddingError(message="Cannot embed an empty input list.", error_code="empty_inputs")

        all_vectors: list[list[float]] = []
        for i in range(0, len(request.inputs), self._batch_size):
            batch = request.inputs[i : i + self._batch_size]
            all_vectors.extend(self._call_api_with_retry(batch))

        return _build_embedding_result(self._model, all_vectors)

    def embed_query(self, text: str) -> list[float]:
        """Generate a single OpenRouter embedding vector for a query string."""

        vectors = self._call_api_with_retry([text])
        if not vectors:
            raise EmbeddingError(
                message="OpenRouter returned no embedding for the query.",
                error_code="empty_query_embedding",
            )
        return vectors[0]


def create_embedding_provider(
    provider_name: str,
    model_name: str,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    api_key: str = "",
    expected_dimension: int | None = None,
) -> EmbeddingProvider:
    """Create (or return a cached) embedding provider implementation.

    Local (sentence-transformers) providers are cached at module level so the
    model is loaded from disk only once per process rather than once per
    request.  OpenRouter providers are stateless HTTP clients and are not
    cached.
    """
    normalized = provider_name.strip().lower()
    if normalized in {"sentence_transformers", "sentence-transformers", "local"}:
        cache_key = f"st:{model_name}:{batch_size}:{expected_dimension}"
        if cache_key not in _PROVIDER_CACHE:
            logger.debug(
                "Creating new SentenceTransformer provider (first call for this model)",
                extra={"extra_data": {"model": model_name}},
            )
            _PROVIDER_CACHE[cache_key] = SentenceTransformerEmbeddingProvider(
                model=model_name,
                batch_size=batch_size,
                expected_dimension=expected_dimension,
            )
        return _PROVIDER_CACHE[cache_key]

    if normalized == "openrouter":
        if not api_key:
            raise EmbeddingError(
                message="OPENROUTER_API_KEY is required when EMBEDDING_PROVIDER=openrouter.",
                error_code="missing_openrouter_api_key",
            )
        return OpenRouterEmbeddingProvider(
            api_key=api_key,
            model=model_name,
            batch_size=batch_size,
        )

    raise EmbeddingError(
        message=f"Unsupported embedding provider '{provider_name}'.",
        error_code="unsupported_embedding_provider",
        details={"provider": provider_name},
    )


def embed_validated_chunks(
    chunks: list[ValidatedChunk],
    provider: EmbeddingProvider,
    model_name: str,
) -> EmbeddingBatchResult:
    """Generate embeddings for all valid chunks using the given provider."""

    valid_chunks = [c for c in chunks if c.is_valid]
    if not valid_chunks:
        raise EmbeddingError(
            message="No valid chunks to embed.",
            error_code="no_valid_chunks",
        )

    request = EmbeddingRequest(
        model_name=model_name,
        inputs=[c.text for c in valid_chunks],
    )

    result = provider.embed(request)

    for i, vec in enumerate(result.vectors):
        if i < len(valid_chunks):
            result.vectors[i] = EmbeddingVector(
                chunk_id=valid_chunks[i].chunk_id,
                values=vec.values,
                dimension=vec.dimension,
            )

    return result

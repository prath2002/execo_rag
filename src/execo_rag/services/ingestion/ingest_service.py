"""Production ingestion service.

Orchestrates the full document ingestion lifecycle:
  1. Idempotency check (E8) — skip if the same file hash was already ingested
  2. Pipeline invocation via LangGraph (E2/E3)
  3. Artifact persistence (E7) — save extraction, metadata, chunks to disk
  4. Document hash registration — update the ledger for future idempotency

Public entry point::

    result = ingest_document(request_id, file_path, document_id, document_type)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from execo_rag.logging.metrics import record_ingestion
from execo_rag.models.pipeline_state import PipelineStatus
from execo_rag.repositories.artifact_repository import ArtifactRepository
from execo_rag.repositories.document_repository import DocumentRepository
from execo_rag.services.ingestion.loader import load_document_source
from execo_rag.services.orchestration.graph import run_ingestion_pipeline
from execo_rag.utils.exceptions import ExtractionError
from execo_rag.utils.ids import generate_document_id

logger = logging.getLogger(__name__)

# Shared repository instances (singletons per process)
_artifact_repo = ArtifactRepository()
_document_repo = DocumentRepository()


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class IngestResult:
    """Result returned by :func:`ingest_document`."""

    def __init__(
        self,
        document_id: str,
        request_id: str,
        status: str,
        message: str,
        chunk_count: int = 0,
        valid_chunk_count: int = 0,
        skipped: bool = False,
        errors: list[str] | None = None,
    ) -> None:
        self.document_id = document_id
        self.request_id = request_id
        self.status = status
        self.message = message
        self.chunk_count = chunk_count
        self.valid_chunk_count = valid_chunk_count
        self.skipped = skipped
        self.errors = errors or []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_document(
    request_id: str,
    file_path: str | Path,
    document_id: str | None = None,
    document_type: str = "share_purchase_agreement",
    force_reingest: bool = False,
) -> IngestResult:
    """Ingest a PDF document into the vector pipeline.

    Flow:
      1. Validate the file and compute its hash.
      2. Check idempotency — return early if already ingested (unless forced).
      3. Run the LangGraph pipeline.
      4. Persist artifacts for every pipeline stage.
      5. Register the file hash in the document ledger.

    Args:
        request_id: Tracing request identifier.
        file_path: Path to the PDF file.
        document_id: Optional stable ID (derived from file hash if not given).
        document_type: Document type slug (default ``share_purchase_agreement``).
        force_reingest: Skip idempotency check and reprocess even if known.

    Returns:
        :class:`IngestResult` with status, counts, and error details.
    """
    start_ts = time.perf_counter()
    file_path = Path(file_path)

    logger.info(
        "Ingestion request received",
        extra={
            "extra_data": {
                "request_id": request_id,
                "file_path": str(file_path),
                "force_reingest": force_reingest,
            }
        },
    )

    # ── Step 1: Validate file and compute hash ─────────────────────────
    try:
        source = load_document_source(file_path)
    except ExtractionError as exc:
        logger.error(
            "File validation failed",
            extra={"extra_data": {"request_id": request_id, "error": str(exc)}},
        )
        return IngestResult(
            document_id=document_id or "unknown",
            request_id=request_id,
            status="failed",
            message=f"File validation failed: {exc}",
            errors=[str(exc)],
        )

    file_hash = source.file_hash or ""
    # Derive document_id from hash if not provided
    if not document_id:
        document_id = generate_document_id(file_hash)

    logger.info(
        "Document accepted",
        extra={
            "extra_data": {
                "request_id": request_id,
                "document_id": document_id,
                "file_name": source.file_name,
                "sha256": (file_hash[:16] + "...") if file_hash else "unknown",
            }
        },
    )

    # ── Step 2: Idempotency check ──────────────────────────────────────
    if not force_reingest and file_hash and _document_repo.is_known(file_hash):
        existing_summary = _artifact_repo.get_run_summary(document_id)
        if existing_summary and existing_summary.get("status") == "completed":
            logger.info(
                "Document already ingested; skipping",
                extra={
                    "extra_data": {
                        "request_id": request_id,
                        "document_id": document_id,
                        "existing_status": "completed",
                    }
                },
            )
            return IngestResult(
                document_id=document_id,
                request_id=request_id,
                status="skipped",
                message="Document already ingested. Use force_reingest=True to reprocess.",
                chunk_count=existing_summary.get("chunk_count", 0),
                valid_chunk_count=existing_summary.get("valid_chunk_count", 0),
                skipped=True,
            )

    # ── Step 3: Run LangGraph ingestion pipeline ───────────────────────
    logger.info(
        "Starting ingestion pipeline",
        extra={"extra_data": {"request_id": request_id, "document_id": document_id}},
    )

    final_state = run_ingestion_pipeline(
        request_id=request_id,
        pdf_path=str(file_path),
        document_id=document_id,
        document_type=document_type,
    )

    final_status: PipelineStatus = final_state.get("status", PipelineStatus.FAILED)
    errors: list[str] = final_state.get("errors", [])

    # ── Step 4: Persist artifacts ──────────────────────────────────────
    elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
    validated_chunks = final_state.get("validated_chunks", [])
    valid_chunk_count = sum(1 for c in validated_chunks if getattr(c, "is_valid", False))

    try:
        if final_state.get("extracted_document"):
            _artifact_repo.save_extraction(document_id, final_state["extracted_document"])

        if final_state.get("cleaned_pages"):
            _artifact_repo.save_cleaned_pages(document_id, final_state["cleaned_pages"])

        if final_state.get("document_metadata"):
            _artifact_repo.save_metadata(document_id, final_state["document_metadata"])

        if validated_chunks:
            _artifact_repo.save_chunk_manifest(document_id, validated_chunks)

        _artifact_repo.save_run_summary(
            document_id=document_id,
            request_id=request_id,
            status=final_status.value if hasattr(final_status, "value") else str(final_status),
            chunk_count=len(validated_chunks),
            valid_chunk_count=valid_chunk_count,
            error_count=len(errors),
            errors=errors,
            duration_ms=elapsed_ms,
        )
    except Exception as exc:
        logger.warning(
            "Artifact persistence partially failed",
            extra={"extra_data": {"document_id": document_id, "error": str(exc)}},
        )

    # ── Step 5: Register hash in ledger ───────────────────────────────
    if final_status == PipelineStatus.COMPLETED and file_hash:
        _document_repo.register(file_hash, document_id)

    # ── Build result ───────────────────────────────────────────────────
    if final_status == PipelineStatus.COMPLETED:
        message = (
            f"Ingestion complete. {valid_chunk_count} of {len(validated_chunks)} "
            f"chunks indexed successfully."
        )
        logger.info(
            "Ingestion completed successfully",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "document_id": document_id,
                    "chunk_count": len(validated_chunks),
                    "valid_chunk_count": valid_chunk_count,
                    "duration_ms": elapsed_ms,
                }
            },
        )
    else:
        message = f"Ingestion failed with {len(errors)} error(s). Check artifacts for details."
        logger.error(
            "Ingestion failed",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "document_id": document_id,
                    "errors": errors,
                    "duration_ms": elapsed_ms,
                }
            },
        )

    status_value = final_status.value if hasattr(final_status, "value") else str(final_status)
    record_ingestion(
        status=status_value,
        chunk_count=len(validated_chunks),
        valid_chunk_count=valid_chunk_count,
        duration_ms=elapsed_ms,
        document_id=document_id,
    )

    return IngestResult(
        document_id=document_id,
        request_id=request_id,
        status=status_value,
        message=message,
        chunk_count=len(validated_chunks),
        valid_chunk_count=valid_chunk_count,
        errors=errors,
    )

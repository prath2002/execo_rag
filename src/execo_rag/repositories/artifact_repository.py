"""Artifact repository: file-based persistence for pipeline run outputs.

Saves the following artifacts per ingestion run under ``data/output/<document_id>/``:

- ``extraction.json``  — raw ``ExtractedDocument`` dump
- ``cleaned_pages.json`` — list of ``CleanedPage`` objects
- ``metadata.json``       — ``DocumentMetadata`` dump
- ``chunks_manifest.json``— list of chunk IDs, sections, page ranges, token counts
- ``run_summary.json``    — high-level run metadata (status, timestamps, counts)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base directory for all artifact output (relative to repo root at runtime)
_DEFAULT_OUTPUT_DIR = Path("data") / "output"


class ArtifactRepository:
    """Stores and retrieves pipeline run artifacts on the local filesystem.

    Args:
        output_dir: Base directory for artifact storage (default ``data/output``).
    """

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self._base = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR

    def _doc_dir(self, document_id: str) -> Path:
        """Return (and create) the per-document output directory."""
        doc_dir = self._base / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _write_json(self, path: Path, data: Any) -> None:
        """Serialize *data* to a JSON file, replacing any previous content."""
        try:
            path.write_text(
                json.dumps(data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(
                "Artifact written",
                extra={"extra_data": {"path": str(path)}},
            )
        except Exception as exc:
            logger.error(
                "Failed to write artifact",
                extra={"extra_data": {"path": str(path), "error": str(exc)}},
            )

    # ------------------------------------------------------------------
    # Public save API
    # ------------------------------------------------------------------

    def save_extraction(self, document_id: str, extracted_document: Any) -> Path:
        """Persist the raw ``ExtractedDocument`` as JSON.

        Args:
            document_id: Document identifier.
            extracted_document: ``ExtractedDocument`` Pydantic model.

        Returns:
            Path where the file was written.
        """
        path = self._doc_dir(document_id) / "extraction.json"
        self._write_json(path, extracted_document.model_dump(mode="json"))
        return path

    def save_cleaned_pages(self, document_id: str, cleaned_pages: list[Any]) -> Path:
        """Persist cleaned page text as JSON.

        Args:
            document_id: Document identifier.
            cleaned_pages: List of ``CleanedPage`` Pydantic models.

        Returns:
            Path where the file was written.
        """
        path = self._doc_dir(document_id) / "cleaned_pages.json"
        self._write_json(path, [p.model_dump(mode="json") for p in cleaned_pages])
        return path

    def save_metadata(self, document_id: str, metadata: Any) -> Path:
        """Persist ``DocumentMetadata`` as JSON.

        Args:
            document_id: Document identifier.
            metadata: ``DocumentMetadata`` Pydantic model.

        Returns:
            Path where the file was written.
        """
        path = self._doc_dir(document_id) / "metadata.json"
        self._write_json(path, metadata.model_dump(mode="json"))
        return path

    def save_chunk_manifest(
        self, document_id: str, validated_chunks: list[Any]
    ) -> Path:
        """Persist a lightweight chunk manifest (no text bodies).

        Args:
            document_id: Document identifier.
            validated_chunks: List of ``ValidatedChunk`` Pydantic models.

        Returns:
            Path where the file was written.
        """
        path = self._doc_dir(document_id) / "chunks_manifest.json"
        manifest = [
            {
                "chunk_id": c.chunk_id,
                "section": c.section.value,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "token_count": c.token_count,
                "is_valid": c.is_valid,
            }
            for c in validated_chunks
        ]
        self._write_json(path, manifest)
        return path

    def save_run_summary(
        self,
        document_id: str,
        request_id: str,
        status: str,
        chunk_count: int,
        valid_chunk_count: int,
        error_count: int,
        errors: list[str],
        duration_ms: int,
    ) -> Path:
        """Persist a high-level run summary JSON.

        Returns:
            Path where the file was written.
        """
        path = self._doc_dir(document_id) / "run_summary.json"
        summary = {
            "document_id": document_id,
            "request_id": request_id,
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,
            "valid_chunk_count": valid_chunk_count,
            "error_count": error_count,
            "errors": errors,
            "duration_ms": duration_ms,
        }
        self._write_json(path, summary)
        return path

    # ------------------------------------------------------------------
    # Idempotency helpers
    # ------------------------------------------------------------------

    def has_run_summary(self, document_id: str) -> bool:
        """Return True if a successful run summary already exists for *document_id*."""
        path = self._doc_dir(document_id) / "run_summary.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("status") == "completed"
        except Exception:
            return False

    def get_run_summary(self, document_id: str) -> dict[str, Any] | None:
        """Return the run summary dict for *document_id*, or ``None`` if absent."""
        path = self._doc_dir(document_id) / "run_summary.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

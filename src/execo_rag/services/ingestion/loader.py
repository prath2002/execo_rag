"""File loader service: validates existence, MIME type, and computes file hash."""

from __future__ import annotations

import logging
from pathlib import Path

from execo_rag.models.document import DocumentSource, SourceType
from execo_rag.utils.exceptions import ExtractionError
from execo_rag.utils.hashing import compute_sha256

logger = logging.getLogger(__name__)

# PDF magic bytes (first 4 bytes of every valid PDF file)
_PDF_MAGIC = b"%PDF"
_ALLOWED_EXTENSIONS = {".pdf"}
_ALLOWED_MIME_TYPES = {"application/pdf"}

# Maximum file size accepted for ingestion (200 MB)
_MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024


def _detect_mime_type(path: Path) -> str:
    """Detect MIME type using the filetype library with a magic-bytes fallback.

    Args:
        path: File to inspect.

    Returns:
        MIME type string (e.g. 'application/pdf').
    """
    try:
        import filetype  # type: ignore[import-untyped]

        kind = filetype.guess(str(path))
        if kind is not None:
            return kind.mime
    except ImportError:
        logger.debug("filetype library not installed; falling back to magic-byte check")

    # Fallback: read first 8 bytes and compare against known signatures
    with path.open("rb") as fh:
        header = fh.read(8)

    if header.startswith(_PDF_MAGIC):
        return "application/pdf"

    # Last resort: use Python's mimetypes (extension-based only)
    import mimetypes

    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def load_document_source(
    file_path: str | Path,
    source_type: SourceType = SourceType.LOCAL_FILE,
) -> DocumentSource:
    """Validate a PDF file and return a populated :class:`DocumentSource`.

    Performs the following checks in order:
      1. Path resolves to an existing regular file.
      2. File size is within the configured limit.
      3. MIME type is ``application/pdf``.
      4. File extension is ``.pdf``.
      5. SHA-256 hash is computed for idempotency tracking.

    Args:
        file_path: Path to the PDF file (string or :class:`Path`).
        source_type: How the file was delivered (default ``LOCAL_FILE``).

    Returns:
        Populated :class:`DocumentSource` with hash, name, and MIME type set.

    Raises:
        ExtractionError: If any validation check fails.
    """
    path = Path(file_path).resolve()

    # --- Existence check ---
    if not path.exists():
        raise ExtractionError(
            message=f"File not found: {path}",
            error_code="file_not_found",
            details={"path": str(path)},
        )

    if not path.is_file():
        raise ExtractionError(
            message=f"Path is not a regular file: {path}",
            error_code="not_a_file",
            details={"path": str(path)},
        )

    logger.debug("File exists", extra={"extra_data": {"path": str(path)}})

    # --- Size check ---
    file_size = path.stat().st_size
    if file_size == 0:
        raise ExtractionError(
            message=f"File is empty: {path}",
            error_code="empty_file",
            details={"path": str(path), "size_bytes": 0},
        )

    if file_size > _MAX_FILE_SIZE_BYTES:
        raise ExtractionError(
            message=(
                f"File exceeds maximum allowed size of "
                f"{_MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB: {path}"
            ),
            error_code="file_too_large",
            details={"path": str(path), "size_bytes": file_size},
        )

    # --- Extension check ---
    suffix = path.suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise ExtractionError(
            message=f"Unsupported file extension '{suffix}'. Expected: .pdf",
            error_code="unsupported_extension",
            details={"path": str(path), "extension": suffix},
        )

    # --- MIME type detection ---
    mime_type = _detect_mime_type(path)
    if mime_type not in _ALLOWED_MIME_TYPES:
        raise ExtractionError(
            message=f"Unsupported MIME type '{mime_type}'. Expected: application/pdf",
            error_code="unsupported_mime_type",
            details={"path": str(path), "detected_mime": mime_type},
        )

    logger.debug(
        "MIME type validated",
        extra={"extra_data": {"path": str(path), "mime_type": mime_type}},
    )

    # --- Hash computation ---
    file_hash = compute_sha256(path)
    logger.info(
        "Document source loaded",
        extra={
            "extra_data": {
                "path": str(path),
                "file_name": path.name,
                "size_bytes": file_size,
                "mime_type": mime_type,
                "sha256": file_hash,
            }
        },
    )

    return DocumentSource(
        source_type=source_type,
        path=path,
        file_name=path.name,
        mime_type=mime_type,
        file_hash=file_hash,
    )

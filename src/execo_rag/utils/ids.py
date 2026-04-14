"""Deterministic and random identifier generation utilities."""

from __future__ import annotations

import hashlib
import uuid


def generate_request_id() -> str:
    """Return a short unique request identifier prefixed with 'req_'."""
    return f"req_{uuid.uuid4().hex[:12]}"


def generate_document_id(file_hash: str) -> str:
    """Derive a stable document identifier from a file hash.

    The same file always produces the same document ID, enabling idempotent
    ingestion without external state.

    Args:
        file_hash: SHA-256 hex digest of the source file.

    Returns:
        Stable document ID prefixed with 'doc_'.
    """
    short = hashlib.sha256(file_hash.encode()).hexdigest()[:12]
    return f"doc_{short}"


def generate_chunk_id(document_id: str, index: int) -> str:
    """Generate a deterministic chunk identifier.

    Args:
        document_id: Parent document identifier.
        index: Zero-based chunk position within the document.

    Returns:
        Chunk ID string prefixed with 'chunk_'.
    """
    return f"chunk_{document_id}_{index:05d}"


def generate_segment_id(document_id: str, page: int, order: int) -> str:
    """Generate a deterministic section-segment identifier.

    Args:
        document_id: Parent document identifier.
        page: Page number where the segment starts.
        order: Segment order index within the page.

    Returns:
        Segment ID string prefixed with 'seg_'.
    """
    return f"seg_{document_id}_p{page:04d}_o{order:04d}"


def generate_block_id(document_id: str, page: int, index: int) -> str:
    """Generate a deterministic block identifier.

    Args:
        document_id: Parent document identifier.
        page: Page number where the block appears.
        index: Block index within the page.

    Returns:
        Block ID string prefixed with 'blk_'.
    """
    return f"blk_{document_id}_p{page:04d}_{index:04d}"

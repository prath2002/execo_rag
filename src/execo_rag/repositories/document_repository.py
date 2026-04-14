"""Document repository: tracks ingested document hashes for idempotency.

Persists a JSON ledger of ``{file_hash: document_id}`` at
``data/output/.document_index.json`` so repeated ingestion of the same
file can be detected without querying Pinecone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_INDEX_FILENAME = ".document_index.json"
_DEFAULT_OUTPUT_DIR = Path("data") / "output"


class DocumentRepository:
    """File-based ledger mapping SHA-256 file hashes to document IDs.

    Args:
        output_dir: Directory where the index file is stored.
    """

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self._dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / _INDEX_FILENAME
        self._index: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        """Load the index from disk, returning an empty dict on any error."""
        if not self._index_path.exists():
            return {}
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Could not load document index; starting fresh",
                extra={"extra_data": {"error": str(exc)}},
            )
            return {}

    def _save(self) -> None:
        """Persist the current in-memory index to disk."""
        try:
            self._index_path.write_text(
                json.dumps(self._index, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error(
                "Failed to save document index",
                extra={"extra_data": {"error": str(exc)}},
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_by_hash(self, file_hash: str) -> str | None:
        """Return the document_id for a previously ingested file hash, or None."""
        return self._index.get(file_hash)

    def register(self, file_hash: str, document_id: str) -> None:
        """Record that *file_hash* has been ingested as *document_id*."""
        self._index[file_hash] = document_id
        self._save()
        logger.debug(
            "Document hash registered",
            extra={"extra_data": {"file_hash": file_hash[:16] + "...", "document_id": document_id}},
        )

    def is_known(self, file_hash: str) -> bool:
        """Return True if *file_hash* is already in the ledger."""
        return file_hash in self._index

    def all_entries(self) -> dict[str, str]:
        """Return a copy of the full hash → document_id mapping."""
        return dict(self._index)

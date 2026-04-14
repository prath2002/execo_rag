"""File hashing utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_sha256(path: Path, chunk_size: int = 65_536) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: Absolute path to the file to hash.
        chunk_size: Read buffer size in bytes (default 64 KiB).

    Returns:
        Lowercase hexadecimal SHA-256 digest string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

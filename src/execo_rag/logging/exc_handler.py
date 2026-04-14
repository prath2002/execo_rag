"""Exception logging utilities.

Provides :func:`log_exception` to capture rich structured error context
matching the project's canonical error log format::

    {
      "level": "ERROR",
      "message": "...",
      "error": {
        "type": "ExtractionError",
        "message": "...",
        "file": "src/execo_rag/services/ingestion/extractor.py",
        "line": 88,
        "stack_trace": "Traceback (most recent call last): ..."
      }
    }

Usage::

    try:
        do_something()
    except Exception as exc:
        log_exception(logger, exc, "Context message", document_id="doc_abc")
        raise
"""

from __future__ import annotations

import logging
import sys
import traceback
from types import TracebackType
from typing import Any


def log_exception(
    logger: logging.Logger,
    exc: BaseException,
    message: str = "An error occurred",
    level: int = logging.ERROR,
    **extra_fields: Any,
) -> None:
    """Log an exception with full structured context.

    Args:
        logger: Logger instance to write to.
        exc: The exception to log.
        message: Human-readable context message.
        level: Log level (default ``logging.ERROR``).
        **extra_fields: Additional key-value pairs to include in ``extra_data``.
    """
    tb = exc.__traceback__
    file_path, line_number = _extract_location(tb)
    stack_trace = "".join(
        traceback.format_exception(type(exc), exc, tb)
    ).strip()

    error_payload: dict[str, Any] = {
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "file": file_path,
            "line": line_number,
            "stack_trace": stack_trace,
        },
        **extra_fields,
    }

    logger.log(level, message, extra={"extra_data": error_payload})


def _extract_location(tb: TracebackType | None) -> tuple[str, int]:
    """Walk to the innermost frame of a traceback and return (file, line)."""
    if tb is None:
        return "<unknown>", 0

    frame = tb
    while frame.tb_next is not None:
        frame = frame.tb_next

    file_path = frame.tb_frame.f_code.co_filename
    line_number = frame.tb_lineno
    return file_path, line_number


def capture_and_log(
    logger: logging.Logger,
    message: str = "Unhandled exception",
    level: int = logging.ERROR,
    reraise: bool = True,
    **extra_fields: Any,
) -> Any:
    """Context manager version of :func:`log_exception`.

    Usage::

        with capture_and_log(logger, "Extraction failed", document_id="doc_x"):
            result = extract_pdf(path, doc_id)

    Args:
        logger: Logger to write to.
        message: Context message for the error log.
        level: Log level.
        reraise: Whether to re-raise the exception (default True).
        **extra_fields: Extra structured fields for the log record.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():  # type: ignore[return]
        try:
            yield
        except Exception as exc:
            log_exception(logger, exc, message, level=level, **extra_fields)
            if reraise:
                raise

    return _ctx()

"""Structured boundary logging utilities.

Provides a context manager and decorator for consistent service-boundary
log events that include ``task_name``, ``document_id``, ``request_id``,
and ``duration_ms`` at every entry and exit point.

Usage (context manager)::

    with log_boundary("extract_pdf", document_id="doc_abc"):
        result = extract_pdf(path, doc_id)

Usage (decorator)::

    @log_task("metadata_extraction")
    def extract_metadata(document_id: str, pages: list) -> DocumentMetadata:
        ...
"""

from __future__ import annotations

import functools
import logging
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

from execo_rag.logging.context import get_log_context, set_task_name

_logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def log_boundary(
    task_name: str,
    document_id: str | None = None,
    logger: logging.Logger | None = None,
    reraise: bool = True,
) -> Generator[None, None, None]:
    """Context manager that logs task start/end with timing.

    Args:
        task_name: Short name for the task (e.g. ``'extract_pdf'``).
        document_id: Optional document identifier for log context.
        logger: Logger to use (defaults to the boundaries module logger).
        reraise: Whether to re-raise exceptions after logging (default True).

    Yields:
        Nothing — used purely for timing and logging side effects.

    Raises:
        Any exception raised within the block (after logging it).
    """
    log = logger or _logger
    set_task_name(task_name)
    start_ts = time.perf_counter()

    extra: dict[str, Any] = {"task_name": task_name}
    if document_id:
        extra["document_id"] = document_id

    log.info(
        f"[{task_name}] started",
        extra={"extra_data": extra},
    )

    try:
        yield
        elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
        log.info(
            f"[{task_name}] completed",
            extra={"extra_data": {**extra, "duration_ms": elapsed_ms}},
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
        log.error(
            f"[{task_name}] failed",
            extra={
                "extra_data": {
                    **extra,
                    "duration_ms": elapsed_ms,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            },
        )
        if reraise:
            raise


def log_task(
    task_name: str,
    logger: logging.Logger | None = None,
) -> Callable[[F], F]:
    """Decorator that wraps a function with :func:`log_boundary`.

    Extracts ``document_id`` from kwargs if present.

    Args:
        task_name: Short name for the task.
        logger: Logger to use (defaults to the decorated function's module logger).

    Returns:
        Decorator that adds boundary logging to the target function.

    Example::

        @log_task("clean_pages")
        def clean_pages(pages, document_id=None):
            ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            doc_id = kwargs.get("document_id") or (
                args[0] if args and isinstance(args[0], str) else None
            )
            log = logger or logging.getLogger(fn.__module__)
            with log_boundary(task_name, document_id=doc_id, logger=log):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator

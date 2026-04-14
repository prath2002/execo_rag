"""Shared API dependencies for settings and request-scoped logging."""

from __future__ import annotations

from uuid import uuid4

from fastapi import Request

from execo_rag.config import Settings, get_settings
from execo_rag.logging import clear_log_context, get_logger, set_request_id
from execo_rag.logging.logger import BoundLoggerAdapter


def get_app_settings() -> Settings:
    """Return the cached application settings."""

    return get_settings()


def get_request_logger(request: Request) -> BoundLoggerAdapter:
    """Return a logger bound to the current request context."""

    request_id = request.headers.get("X-Request-ID", f"req_{uuid4().hex}")
    set_request_id(request_id)
    request.state.request_id = request_id
    logger = get_logger(
        "execo_rag.api",
        path=str(request.url.path),
        method=request.method,
    )
    return logger


def reset_request_logging_context() -> None:
    """Clear request-scoped logging state after request processing."""

    clear_log_context()

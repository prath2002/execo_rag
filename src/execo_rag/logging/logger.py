"""Logger factory with structured context injection."""

from __future__ import annotations

import logging
from typing import Any

from execo_rag.config import get_settings
from execo_rag.logging.context import get_log_context
from execo_rag.logging.formatters import JsonLogFormatter, PlainTextFormatter


class ContextEnrichmentFilter(logging.Filter):
    """Inject request-scoped context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = get_log_context()
        for key, value in context.items():
            setattr(record, key, value)
        return True


def _resolve_level(level_name: str) -> int:
    """Map a level name to the stdlib logging constant."""

    return getattr(logging, level_name.upper(), logging.INFO)


def configure_logging() -> None:
    """Configure root logging exactly once for the application."""

    settings = get_settings()
    root_logger = logging.getLogger()
    if getattr(root_logger, "_execo_logging_configured", False):
        return

    root_logger.setLevel(_resolve_level(settings.logging.level))
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(_resolve_level(settings.logging.level))
    handler.addFilter(ContextEnrichmentFilter())
    if settings.logging.json_logs:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(PlainTextFormatter())

    root_logger.addHandler(handler)
    root_logger._execo_logging_configured = True  # type: ignore[attr-defined]


class BoundLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that supports a typed `extra_data` payload."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("extra_data", {})
        if self.extra:
            extra["extra_data"] = {**self.extra, **extra["extra_data"]}
        settings = get_settings()
        extra.setdefault("service", settings.app.name)
        extra.setdefault("environment", settings.app.env)
        return msg, kwargs


def get_logger(name: str, **bound_context: Any) -> BoundLoggerAdapter:
    """Return a configured logger with optional static bound context."""

    configure_logging()
    logger = logging.getLogger(name)
    return BoundLoggerAdapter(logger, bound_context)

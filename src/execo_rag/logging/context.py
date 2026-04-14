"""Context propagation helpers for structured logging."""

from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_document_id: ContextVar[str | None] = ContextVar("document_id", default=None)
_task_name: ContextVar[str | None] = ContextVar("task_name", default=None)
_extra_context: ContextVar[dict[str, Any]] = ContextVar("extra_context", default={})


def set_request_id(request_id: str | None) -> None:
    """Set the request identifier in the current context."""

    _request_id.set(request_id)


def set_document_id(document_id: str | None) -> None:
    """Set the document identifier in the current context."""

    _document_id.set(document_id)


def set_task_name(task_name: str | None) -> None:
    """Set the current pipeline task name in the current context."""

    _task_name.set(task_name)


def add_log_context(**kwargs: Any) -> None:
    """Merge additional structured keys into the active logging context."""

    current = dict(_extra_context.get())
    current.update({key: value for key, value in kwargs.items() if value is not None})
    _extra_context.set(current)


def clear_log_context() -> None:
    """Reset all logging context fields for the current execution flow."""

    _request_id.set(None)
    _document_id.set(None)
    _task_name.set(None)
    _extra_context.set({})


def get_log_context() -> dict[str, Any]:
    """Return the active structured logging context."""

    context: dict[str, Any] = {
        "request_id": _request_id.get(),
        "document_id": _document_id.get(),
        "task_name": _task_name.get(),
    }
    context.update(_extra_context.get())
    return {key: value for key, value in context.items() if value is not None}

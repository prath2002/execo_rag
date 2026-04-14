"""Logging formatters for structured and plain-text output."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Serialize log records into JSON for downstream systems."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        for key in ("request_id", "document_id", "task_name", "service", "environment"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        extra_payload = getattr(record, "extra_data", None)
        if extra_payload:
            payload["extra"] = extra_payload

        if record.exc_info:
            filename = record.pathname
            payload["error"] = {
                "message": str(record.exc_info[1]),
                "file": str(Path(filename)),
                "line": record.lineno,
                "stack_trace": self.formatException(record.exc_info),
            }

        return json.dumps(payload, default=str)


class PlainTextFormatter(logging.Formatter):
    """Readable local-development formatter with context fields."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(UTC).isoformat()
        parts = [
            timestamp,
            record.levelname,
            record.name,
            record.funcName,
            record.getMessage(),
        ]

        for key in ("request_id", "document_id", "task_name"):
            value = getattr(record, key, None)
            if value is not None:
                parts.append(f"{key}={value}")

        if record.exc_info:
            parts.append(self.formatException(record.exc_info))

        return " | ".join(parts)

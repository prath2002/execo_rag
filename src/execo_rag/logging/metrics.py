"""Lightweight in-process metrics hooks.

Tracks ingestion and query counters/latency in memory and emits a structured
``"metrics"`` log event on every record call.  The log event is the primary
integration point — a log shipper (Datadog, Loki, CloudWatch) can parse it to
populate dashboards without any additional SDK dependency.

To add Prometheus / StatsD / OTEL later, replace the ``_emit`` call inside
each ``record_*`` function with the appropriate SDK call while keeping the
log event alongside it for redundancy.

Usage::

    from execo_rag.logging.metrics import record_ingestion, record_query, get_metrics

    record_ingestion(status="completed", chunk_count=61, duration_ms=4200)
    record_query(match_count=5, duration_ms=320)
    snapshot = get_metrics()
"""

from __future__ import annotations

import logging
import threading
from typing import Any

_logger = logging.getLogger(__name__)
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Internal counters
# ---------------------------------------------------------------------------

_counters: dict[str, int] = {
    "ingestion_total": 0,
    "ingestion_completed": 0,
    "ingestion_failed": 0,
    "ingestion_skipped": 0,
    "ingestion_chunks_total": 0,
    "query_total": 0,
    "query_errors": 0,
    "query_empty_results": 0,
}

_latency: dict[str, list[float]] = {
    "ingestion_duration_ms": [],
    "query_duration_ms": [],
}

# Keep only the last N latency samples to avoid unbounded memory growth
_MAX_LATENCY_SAMPLES = 1_000


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _emit(event: str, data: dict[str, Any]) -> None:
    """Emit a structured metrics log event."""
    _logger.info(
        "metrics",
        extra={"extra_data": {"event": event, **data}},
    )


def _record_latency(key: str, value_ms: float) -> None:
    samples = _latency[key]
    samples.append(value_ms)
    if len(samples) > _MAX_LATENCY_SAMPLES:
        samples.pop(0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_ingestion(
    status: str,
    chunk_count: int = 0,
    valid_chunk_count: int = 0,
    duration_ms: float = 0.0,
    document_id: str | None = None,
) -> None:
    """Record a completed ingestion attempt.

    Args:
        status: Final status string — ``"completed"``, ``"failed"``, or ``"skipped"``.
        chunk_count: Total chunks produced.
        valid_chunk_count: Chunks that passed validation.
        duration_ms: Wall-clock duration of the ingestion in milliseconds.
        document_id: Optional document identifier for log correlation.
    """
    with _lock:
        _counters["ingestion_total"] += 1
        key = f"ingestion_{status}" if f"ingestion_{status}" in _counters else "ingestion_failed"
        _counters[key] += 1
        _counters["ingestion_chunks_total"] += chunk_count
        _record_latency("ingestion_duration_ms", duration_ms)

    payload: dict[str, Any] = {
        "status": status,
        "chunk_count": chunk_count,
        "valid_chunk_count": valid_chunk_count,
        "duration_ms": duration_ms,
        "ingestion_total": _counters["ingestion_total"],
    }
    if document_id:
        payload["document_id"] = document_id

    _emit("ingestion_recorded", payload)


def record_query(
    match_count: int = 0,
    duration_ms: float = 0.0,
    error: bool = False,
) -> None:
    """Record a completed query attempt.

    Args:
        match_count: Number of results returned to the caller.
        duration_ms: Wall-clock duration of the query in milliseconds.
        error: Whether the query ended in an error.
    """
    with _lock:
        _counters["query_total"] += 1
        if error:
            _counters["query_errors"] += 1
        if match_count == 0 and not error:
            _counters["query_empty_results"] += 1
        _record_latency("query_duration_ms", duration_ms)

    _emit(
        "query_recorded",
        {
            "match_count": match_count,
            "duration_ms": duration_ms,
            "error": error,
            "query_total": _counters["query_total"],
        },
    )


def get_metrics() -> dict[str, Any]:
    """Return a snapshot of all in-memory metrics.

    Returns:
        Dict with current counter values and p50/p95/p99 latency estimates.
    """
    with _lock:
        snapshot = dict(_counters)
        for key, samples in _latency.items():
            if samples:
                sorted_s = sorted(samples)
                n = len(sorted_s)
                snapshot[f"{key}_p50"] = sorted_s[int(n * 0.50)]
                snapshot[f"{key}_p95"] = sorted_s[int(n * 0.95)]
                snapshot[f"{key}_p99"] = sorted_s[min(int(n * 0.99), n - 1)]
                snapshot[f"{key}_count"] = n
            else:
                snapshot[f"{key}_p50"] = None
                snapshot[f"{key}_p95"] = None
                snapshot[f"{key}_p99"] = None
                snapshot[f"{key}_count"] = 0

    return snapshot

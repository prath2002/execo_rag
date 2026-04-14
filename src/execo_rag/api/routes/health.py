"""Health endpoints for runtime diagnostics."""

from typing import Any

from fastapi import APIRouter

from execo_rag.logging.metrics import get_metrics

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
def health_check() -> dict[str, str]:
    """Return a minimal application health payload."""

    return {"status": "ok"}


@router.get("/metrics", summary="In-process metrics snapshot")
def metrics_snapshot() -> dict[str, Any]:
    """Return current in-memory counter and latency percentile snapshot."""

    return {"status": "ok", "metrics": get_metrics()}

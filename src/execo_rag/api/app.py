"""FastAPI application bootstrap."""

from __future__ import annotations

import importlib.metadata
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from execo_rag.api.routes.health import router as health_router
from execo_rag.api.routes.ingest import router as ingest_router
from execo_rag.api.routes.query import router as query_router
from execo_rag.api.routes.query_agent import router as query_agent_router
from execo_rag.config import Settings, get_settings
from execo_rag.logging import clear_log_context, configure_logging, get_logger
from execo_rag.utils import ConfigError

_TRACKED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "langgraph",
    "langchain-core",
    "sentence-transformers",
    "pinecone",
    "tiktoken",
    "pypdf",
]


def _package_versions() -> dict[str, str]:
    """Return installed versions for tracked dependencies."""
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def _log_startup_diagnostics(logger: Any, settings: Settings) -> None:
    """Emit a single structured log with environment, feature flags, and dependency versions."""
    logger.info(
        "Startup diagnostics",
        extra={
            "extra_data": {
                "environment": settings.app.env,
                "app_name": settings.app.name,
                "app_version": "0.1.0",
                "log_level": settings.logging.level,
                "feature_flags": {
                    "enable_llm_fallback": settings.runtime.enable_llm_fallback,
                },
                "embedding": {
                    "provider": settings.embeddings.provider,
                    "model": settings.embeddings.model,
                    "dimension": settings.embeddings.dimension,
                    "batch_size": settings.embeddings.batch_size,
                },
                "pinecone": {
                    "index_name": settings.pinecone.index_name,
                    "namespace": settings.pinecone.namespace,
                },
                "pdf": {
                    "extractor": settings.pdf.extractor,
                    "max_chunk_tokens": settings.pdf.max_chunk_tokens,
                    "chunk_overlap_tokens": settings.pdf.chunk_overlap_tokens,
                },
                "dependency_versions": _package_versions(),
            }
        },
    )


def validate_settings(settings: Settings) -> None:
    """Validate critical startup settings before serving traffic."""

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if settings.logging.level.upper() not in valid_levels:
        raise ConfigError(
            message=f"Unsupported log level: {settings.logging.level}",
            details={"log_level": settings.logging.level},
        )

    if settings.app.port <= 0:
        raise ConfigError(
            message="APP_PORT must be a positive integer",
            details={"app_port": settings.app.port},
        )

    if settings.pdf.max_chunk_tokens <= 0:
        raise ConfigError(
            message="MAX_CHUNK_TOKENS must be a positive integer",
            details={"max_chunk_tokens": settings.pdf.max_chunk_tokens},
        )

    if settings.pdf.chunk_overlap_tokens < 0:
        raise ConfigError(
            message="CHUNK_OVERLAP_TOKENS cannot be negative",
            details={"chunk_overlap_tokens": settings.pdf.chunk_overlap_tokens},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging, validate settings, and emit startup diagnostics."""

    configure_logging()
    logger = get_logger(__name__)
    settings = get_settings()
    validate_settings(settings)
    app.state.settings = settings
    _log_startup_diagnostics(logger, settings)
    logger.info(
        "Application startup completed",
        extra={"extra_data": {"app_env": settings.app.env, "app_name": settings.app.name}},
    )
    try:
        yield
    finally:
        logger.info("Application shutdown completed")
        clear_log_context()


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application instance."""

    app = FastAPI(
        title="EXECO RAG Pipeline API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(query_agent_router)
    return app


app = create_app()

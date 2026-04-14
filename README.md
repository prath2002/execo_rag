# EXECO RAG Pipeline

This repository contains the implementation of a production-grade AI pipeline for contract digitization, metadata extraction, hybrid chunking, embedding generation, and metadata-filtered retrieval.

## Current Status

The repository is being built step-by-step from the implementation plan in `IMPLEMENTATION_PLAN.md`.

## Embedding Setup

- Default embedding provider: local `sentence-transformers`
- Default model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Pinecone index requirement: the target index must also be created with dimension `384`
- `OPENROUTER_API_KEY` is optional unless you enable metadata fallback or set `EMBEDDING_PROVIDER=openrouter`

## Package Management

This project uses `uv` as the package manager and dependency workflow tool.

### Dependency Strategy

- `pyproject.toml` is the source of truth for project metadata and dependency groups
- `uv.lock` will be the lockfile committed to the repository for reproducible builds
- local development should use `uv sync --group dev`
- CI should install dependencies from the lockfile when present

### Standard Commands

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run black --check .
uv run mypy src
```

## Planned Capabilities

- PDF ingestion and extraction
- Text cleanup and normalization
- Fixed-schema metadata extraction
- Hybrid chunking for legal documents
- Local 384-dimensional embeddings by default
- Pinecone vector storage with metadata filtering
- LangGraph-based orchestration
- Structured logging and strong validation

## Reference Files

- `IMPLEMENTATION_PLAN.md`: ordered delivery plan
- `Assignment.docx`: assignment brief
- `POC_TEST_SPA.pdf`: sample document to digitize

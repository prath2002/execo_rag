# EXECO RAG Pipeline

Production-grade AI pipeline for contract digitization and semantic retrieval. Ingests legal PDF documents (Share Purchase Agreements), extracts structured metadata, generates embeddings, indexes to Pinecone, and answers natural-language questions via a LangGraph query agent.

---

## Architecture

Two independent LangGraph pipelines:

**Ingestion** (`POST /ingest`)
```
load → extract_pdf → clean_text → extract_metadata → normalize_metadata
     → verify_metadata → detect_sections → chunk → enrich → validate
     → generate_embeddings → index_to_pinecone
```

**Query Agent** (`POST /query/agent`)
```
analyze_query (LLM: intent + filters)
    → retrieve_chunks (embed + Pinecone vector search)
    → synthesize_answer (LLM: grounded structured answer)
    → format_response
```

A simpler `POST /query` endpoint is also available for direct vector search without LLM analysis.

---

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- [Pinecone](https://www.pinecone.io/) account with an index at dimension `384`
- OpenRouter API key (optional — only needed for metadata extraction fallback and the query agent LLM nodes)

---

## Installation

```bash
# Clone and install all dependencies including dev tools
uv sync --group dev
```

---

## Configuration

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

| Env Var | Default | Description |
|---|---|---|
| `APP_NAME` | `execo-rag` | Service name |
| `APP_ENV` | `dev` | Environment (`dev` / `prod`) |
| `APP_HOST` | `0.0.0.0` | Bind address |
| `APP_PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `LOG_JSON` | `true` | Emit structured JSON logs |
| `OPENROUTER_API_KEY` | *(empty)* | Required for LLM metadata extraction and agent queries |
| `OPENROUTER_CHAT_MODEL` | `meta-llama/llama-3.2-3b-instruct:free` | OpenRouter model ID |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | `sentence_transformers` or `openrouter` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model identifier |
| `EMBEDDING_DIMENSION` | `384` | Must match your Pinecone index dimension |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding generation |
| `PINECONE_API_KEY` | *(required)* | Pinecone API key |
| `PINECONE_INDEX_NAME` | *(required)* | Pinecone index name |
| `PINECONE_NAMESPACE` | `contracts-dev` | Namespace within the index |
| `PDF_EXTRACTOR` | `pypdf` | `pypdf` (default) or `unstructured` (requires extra install) |
| `MAX_CHUNK_TOKENS` | `700` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `60` | Token overlap between adjacent chunks |
| `ENABLE_LLM_FALLBACK` | `true` | Use OpenRouter when rule-based extraction yields insufficient metadata |

> **Note:** `unstructured` is not installed by default. Keep `PDF_EXTRACTOR=pypdf` unless you have explicitly installed the `unstructured[pdf]` extra.

---

## Running the Server

```bash
uv run uvicorn execo_rag.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

For development with auto-reload:

```bash
uv run uvicorn execo_rag.api.app:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

The server emits a structured startup diagnostics log with environment, feature flags, embedding config, and dependency versions on every boot.

---

## API Reference

### `GET /health`

Service health check.

```json
{ "status": "ok" }
```

---

### `GET /metrics`

In-process metrics snapshot with ingestion/query counters and p50/p95/p99 latency percentiles.

---

### `POST /ingest`

Ingest a PDF document through the full 11-node pipeline.

**Request**
```json
{
  "document_id": "spa-acme-2024",
  "file_path": "data/contracts/acme_spa.pdf",
  "document_type": "share_purchase_agreement"
}
```

`document_type` defaults to `share_purchase_agreement`. Ingestion is idempotent — re-ingesting a file with the same SHA-256 hash is a no-op.

**Response**
```json
{
  "request_id": "req_abc123",
  "document_id": "spa-acme-2024",
  "status": "completed",
  "message": "Document ingested successfully"
}
```

**What the pipeline extracts**

The metadata extraction step attempts to identify these 14 fields from each document:

| Field | Type |
|---|---|
| `effective_date` | ISO date |
| `buyer` | string |
| `seller` | string |
| `company_target` | string |
| `shares_transacted` | string |
| `cash_purchase_price` | decimal |
| `escrow_agent` | string |
| `escrow_amount` | decimal |
| `target_working_capital` | decimal |
| `indemnification_de_minimis_amount` | decimal |
| `indemnification_basket_amount` | decimal |
| `indemnification_cap_amount` | decimal |
| `governing_law` | string |
| `document_type` | string |

Extraction uses regex rules first; if fewer than expected fields are found and `ENABLE_LLM_FALLBACK=true`, an OpenRouter LLM call is made as a second pass.

---

### `POST /query`

Direct vector search with optional metadata filters. No LLM involved.

**Request**
```json
{
  "query": "What is the indemnification cap?",
  "top_k": 5,
  "filters": {
    "section": "indemnification",
    "buyer": "Acme Corp"
  }
}
```

Available filter fields: `document_type`, `effective_date`, `buyer`, `seller`, `company_target`, `shares_transacted`, `cash_purchase_price`, `escrow_agent`, `escrow_amount`, `target_working_capital`, `indemnification_de_minimis_amount`, `indemnification_basket_amount`, `indemnification_cap_amount`, `governing_law`, `section`, `page_start`, `page_end`.

**Response**
```json
{
  "request_id": "req_abc123",
  "query": "What is the indemnification cap?",
  "results": [
    {
      "chunk_id": "chunk_xyz",
      "score": 0.91,
      "text": "...",
      "section": "indemnification",
      "page_start": 42,
      "page_end": 43
    }
  ]
}
```

---

### `POST /query/agent`

Agentic RAG query. The LLM interprets the question, extracts the appropriate Pinecone filters, retrieves relevant chunks, and synthesizes a structured grounded answer.

**Request**
```json
{
  "query": "Who are the parties and what is the purchase price?",
  "top_k": 5
}
```

**Response**
```json
{
  "request_id": "req_abc123",
  "query": "Who are the parties and what is the purchase price?",
  "refined_query": "parties buyer seller purchase price consideration",
  "intent": "Identify transaction parties and total consideration amount",
  "filter_params_used": { "section": "purchase_price" },
  "answer": "The agreement is between Acme Corp (Buyer) and Jane Doe (Seller)...",
  "confidence": "high",
  "key_findings": [
    "Buyer: Acme Corp",
    "Seller: Jane Doe",
    "Purchase price: $12,500,000"
  ],
  "references": [
    {
      "chunk_id": "chunk_xyz",
      "page_number": 3,
      "section": "purchase_price",
      "score": 0.93,
      "snippet": "The aggregate purchase price shall be $12,500,000..."
    }
  ],
  "caveats": null,
  "chunks_retrieved": 5,
  "reasoning": "Query targets parties and financial terms; applied purchase_price section filter.",
  "status": "completed"
}
```

`status` is one of `completed`, `no_results`, or `failed`.

The agent understands these section values for automatic filter extraction: `preamble`, `purchase_price`, `definitions`, `representations`, `indemnification`, `working_capital`, `governing_law`, `general`.

---

## Project Structure

```
src/execo_rag/
├── api/
│   ├── app.py               # FastAPI bootstrap & lifespan
│   ├── deps.py              # Dependency injection
│   ├── routes/              # health, ingest, query, query_agent
│   └── schemas/             # Pydantic request/response models
├── clients/
│   └── openrouter.py        # Lightweight OpenRouter HTTP client
├── config/
│   ├── settings.py          # Pydantic BaseSettings (all env vars)
│   └── constants.py         # Default values
├── logging/
│   ├── boundaries.py        # Request boundary logging
│   ├── context.py           # Structured log context (request_id, doc_id)
│   ├── exc_handler.py       # Global exception handler
│   ├── formatters.py        # JSON log formatter
│   └── metrics.py           # In-process counters & latency percentiles
├── models/                  # Pydantic data models (chunks, metadata, embeddings…)
├── prompts/                 # LLM prompt templates (txt files)
├── services/
│   ├── chunking/            # Hybrid chunker, section detector, enricher, validator
│   ├── embeddings/          # Embedding provider (sentence-transformers / OpenRouter)
│   ├── ingestion/           # PDF loader, extractor, text cleaner
│   ├── metadata/            # Extractor, normalizer, verifier, regex rules
│   ├── orchestration/       # LangGraph ingestion graph (11 nodes)
│   ├── query/               # LangGraph agent graph (4 nodes) + basic retriever
│   └── vectorstore/         # Pinecone client wrapper + filter builder
└── utils/                   # Shared helpers (dates, hashing, exceptions…)
```

---

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format check
uv run black --check .

# Type check
uv run mypy src
```

---

## Key Design Decisions

- **Embeddings are local by default.** `sentence-transformers/all-MiniLM-L6-v2` runs on CPU. The provider is cached as a singleton after first load — model weights are not reloaded per request.
- **OpenRouter is optional.** Without an API key, ingestion uses rule-based extraction only and the query agent skips LLM analysis, falling back to raw vector search.
- **Pinecone index must be created at dimension 384** to match the default embedding model. If you change `EMBEDDING_MODEL`, update `EMBEDDING_DIMENSION` and recreate the index accordingly.
- **Ingestion is idempotent.** SHA-256 hashing of the source file prevents duplicate indexing.
- **Legal PDF quirks are handled.** The text cleaner normalises Unicode smart quotes, ligatures, and SEC-style artifacts before regex metadata extraction, significantly improving field coverage.
- **Logs are structured JSON** in all environments, with request IDs threaded through every log line.

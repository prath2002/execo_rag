"""LangGraph node functions for the query agent pipeline.

Node execution order::

    analyze_query → retrieve_chunks → synthesize_answer → format_response

Each node receives the full ``QueryAgentState`` dict and returns a partial
dict with only the keys it updates.  LangGraph merges these into the
running state automatically.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from execo_rag.clients import OpenRouterClient
from execo_rag.config import get_settings
from execo_rag.services.embeddings.provider import create_embedding_provider
from execo_rag.services.vectorstore.filter_builder import build_filter_from_params
from execo_rag.services.vectorstore.pinecone_store import PineconeStore
from execo_rag.utils.exceptions import EmbeddingError, VectorStoreError

logger = logging.getLogger(__name__)

StateDict = dict[str, Any]

# Paths to prompt templates
_PROMPTS_DIR = Path(__file__).parents[3] / "prompts"
_ANALYSIS_PROMPT_PATH = _PROMPTS_DIR / "query_analysis.txt"
_SYNTHESIS_PROMPT_PATH = _PROMPTS_DIR / "answer_synthesis.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prompt(path: Path, fallback: str) -> str:
    """Load a prompt template from disk, falling back to inline text."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def _parse_llm_json(content: str) -> dict[str, Any]:
    """Strip markdown fences and parse JSON from an LLM response."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _call_openrouter(
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Call OpenRouter chat completions and return the response content string."""
    client = OpenRouterClient(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "{}").strip()


def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved Pinecone matches into a readable prompt block."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata") or {}
        if hasattr(meta, "model_dump"):
            meta = meta.model_dump()
        chunk_id = chunk.get("id", f"chunk_{i}")
        score = chunk.get("score", 0.0)
        text = (
            meta.get("chunk_text")
            or meta.get("text")
            or ""
        )
        section = meta.get("section", "general")
        page_start = meta.get("page_start", "?")
        page_end = meta.get("page_end", "?")

        lines.append(
            f"[{i}] chunk_id={chunk_id!r}  score={score:.4f}  "
            f"section={section}  pages={page_start}-{page_end}\n"
            f"{text[:1500]}"  # truncate very long chunks
        )
    return "\n\n---\n\n".join(lines)


# ---------------------------------------------------------------------------
# Node 1: Analyze query — understand intent and extract filters
# ---------------------------------------------------------------------------


def node_analyze_query(state: StateDict) -> StateDict:
    """Use an LLM to understand the query and plan retrieval filters.

    Reads:  ``query``
    Writes: ``intent``, ``filter_params``, ``refined_query``, ``reasoning``
    """
    query: str = state.get("query", "")
    request_id: str = state.get("request_id", "")

    logger.info(
        "Agent node started: analyze_query",
        extra={
            "extra_data": {
                "request_id": request_id,
                "query_preview": query[:100],
            }
        },
    )

    settings = get_settings()

    if not settings.openrouter.api_key:
        # No LLM available — pass query through unchanged with no filters
        logger.warning(
            "OPENROUTER_API_KEY not set; skipping query analysis, using raw query",
            extra={"extra_data": {"request_id": request_id}},
        )
        return {
            "intent": query,
            "filter_params": {},
            "refined_query": query,
            "reasoning": "LLM analysis skipped — no API key configured.",
        }

    system_prompt = _load_prompt(
        _ANALYSIS_PROMPT_PATH,
        fallback=(
            "You are a legal document retrieval assistant. "
            "Given the user's query about a Share Purchase Agreement, extract retrieval filters "
            "and a refined search query. Return ONLY valid JSON with keys: "
            "intent, filter_params (section/buyer/seller/company_target/governing_law), "
            "refined_query, reasoning."
        ),
    )

    try:
        content = _call_openrouter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"},
            ],
            model=settings.openrouter.chat_model,
            api_key=settings.openrouter.api_key,
            temperature=0.0,
            max_tokens=512,
        )
        parsed = _parse_llm_json(content)
    except Exception as exc:
        logger.warning(
            "Query analysis LLM call failed; using raw query",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "error": str(exc),
                }
            },
        )
        return {
            "intent": query,
            "filter_params": {},
            "refined_query": query,
            "reasoning": f"LLM analysis failed: {exc}",
        }

    # Extract only non-null filter values
    raw_filters: dict[str, Any] = parsed.get("filter_params", {}) or {}
    filter_params = {k: v for k, v in raw_filters.items() if v is not None}

    refined_query: str = parsed.get("refined_query") or query

    logger.info(
        "Query analysis complete",
        extra={
            "extra_data": {
                "request_id": request_id,
                "intent": parsed.get("intent", "")[:120],
                "filter_count": len(filter_params),
                "filters": filter_params,
                "refined_query": refined_query[:120],
            }
        },
    )

    return {
        "intent": parsed.get("intent", query),
        "filter_params": filter_params,
        "refined_query": refined_query,
        "reasoning": parsed.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Node 2: Retrieve chunks — embed + Pinecone query
# ---------------------------------------------------------------------------


def node_retrieve_chunks(state: StateDict) -> StateDict:
    """Embed the refined query and retrieve matching chunks from Pinecone.

    Reads:  ``refined_query``, ``filter_params``, ``top_k``
    Writes: ``chunks``
    """
    refined_query: str = state.get("refined_query") or state.get("query", "")
    filter_params: dict[str, Any] = state.get("filter_params", {})
    top_k: int = state.get("top_k", 5)
    request_id: str = state.get("request_id", "")

    logger.info(
        "Agent node started: retrieve_chunks",
        extra={
            "extra_data": {
                "request_id": request_id,
                "refined_query": refined_query[:100],
                "filter_params": filter_params,
                "top_k": top_k,
            }
        },
    )

    start_ts = time.perf_counter()
    settings = get_settings()

    try:
        # --- Embed the refined query ---
        provider = create_embedding_provider(
            provider_name=settings.embeddings.provider,
            model_name=settings.embeddings.model,
            batch_size=settings.embeddings.batch_size,
            api_key=settings.openrouter.api_key,
            expected_dimension=settings.embeddings.dimension,
        )
        query_vector = provider.embed_query(refined_query)

        # --- Build metadata filter ---
        pinecone_filter = build_filter_from_params(**filter_params) if filter_params else None

        # --- Query Pinecone ---
        store = PineconeStore(
            api_key=settings.pinecone.api_key,
            index_name=settings.pinecone.index_name,
            namespace=settings.pinecone.namespace,
        )

        from execo_rag.models.pinecone import PineconeQueryRequest

        request = PineconeQueryRequest(
            vector=query_vector,
            top_k=top_k,
            namespace=settings.pinecone.namespace,
            filter=pinecone_filter,
            include_metadata=True,
            include_values=False,
        )
        result = store.query(request, namespace=settings.pinecone.namespace)

    except (EmbeddingError, VectorStoreError) as exc:
        logger.error(
            "Chunk retrieval failed",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            },
        )
        return {"chunks": [], "errors": [str(exc)], "status": "failed"}
    except Exception as exc:
        logger.error(
            "Unexpected error during retrieval",
            extra={"extra_data": {"request_id": request_id, "error": str(exc)}},
        )
        return {"chunks": [], "errors": [str(exc)], "status": "failed"}

    elapsed_ms = round((time.perf_counter() - start_ts) * 1000)

    # Convert matches to plain dicts so LangGraph state stays JSON-serialisable
    chunks: list[dict[str, Any]] = []
    for match in result.matches:
        meta: dict[str, Any] = {}
        if match.metadata is not None:
            meta = (
                match.metadata.model_dump()
                if hasattr(match.metadata, "model_dump")
                else dict(match.metadata)
            )
        chunks.append(
            {
                "id": match.id,
                "score": match.score,
                "metadata": meta,
            }
        )

    logger.info(
        "Retrieval complete",
        extra={
            "extra_data": {
                "request_id": request_id,
                "chunks_found": len(chunks),
                "top_score": chunks[0]["score"] if chunks else None,
                "duration_ms": elapsed_ms,
            }
        },
    )

    return {"chunks": chunks}


# ---------------------------------------------------------------------------
# Node 3: Synthesize answer — LLM reads chunks and answers the question
# ---------------------------------------------------------------------------


def node_synthesize_answer(state: StateDict) -> StateDict:
    """Use an LLM to synthesize a structured answer from retrieved chunks.

    Reads:  ``query``, ``chunks``
    Writes: ``answer``, ``confidence``, ``key_findings``, ``references``, ``caveats``
    """
    query: str = state.get("query", "")
    chunks: list[dict[str, Any]] = state.get("chunks", [])
    request_id: str = state.get("request_id", "")

    logger.info(
        "Agent node started: synthesize_answer",
        extra={
            "extra_data": {
                "request_id": request_id,
                "chunk_count": len(chunks),
            }
        },
    )

    settings = get_settings()

    if not settings.openrouter.api_key:
        # Return chunks as plain text answer when no LLM is available
        logger.warning(
            "OPENROUTER_API_KEY not set; returning raw chunk text as answer",
            extra={"extra_data": {"request_id": request_id}},
        )
        snippets = []
        for c in chunks:
            meta = c.get("metadata") or {}
            text = meta.get("chunk_text") or meta.get("text") or ""
            snippets.append(text[:500])
        return {
            "answer": "\n\n".join(snippets) or "No results found.",
            "confidence": "low",
            "key_findings": [],
            "references": _build_references(chunks),
            "caveats": "Answer generated from raw chunk text; LLM synthesis was unavailable.",
        }

    chunks_text = _format_chunks_for_prompt(chunks)

    synthesis_template = _load_prompt(
        _SYNTHESIS_PROMPT_PATH,
        fallback=(
            "Answer the user's question based strictly on the retrieved chunks. "
            "USER QUESTION: {query}\n\nCHUNKS:\n{chunks_text}\n\n"
            "Return ONLY valid JSON: answer, confidence, key_findings, references, caveats."
        ),
    )

    user_message = (
        synthesis_template
        .replace("{query}", query)
        .replace("{chunks_text}", chunks_text)
        .replace("{chunk_count}", str(len(chunks)))
    )

    try:
        content = _call_openrouter(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert legal analyst specializing in Share Purchase Agreements. "
                        "Return ONLY valid JSON with no markdown."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            model=settings.openrouter.chat_model,
            api_key=settings.openrouter.api_key,
            temperature=0.1,
            max_tokens=2048,
        )
        parsed = _parse_llm_json(content)
    except Exception as exc:
        logger.error(
            "Answer synthesis LLM call failed",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            },
        )
        return {
            "answer": "Answer synthesis failed. Raw retrieval results are in the references.",
            "confidence": "low",
            "key_findings": [],
            "references": _build_references(chunks),
            "caveats": f"LLM synthesis failed: {exc}",
        }

    references = parsed.get("references") or _build_references(chunks)

    logger.info(
        "Answer synthesis complete",
        extra={
            "extra_data": {
                "request_id": request_id,
                "confidence": parsed.get("confidence"),
                "references_count": len(references),
            }
        },
    )

    return {
        "answer": parsed.get("answer", ""),
        "confidence": parsed.get("confidence", "low"),
        "key_findings": parsed.get("key_findings", []),
        "references": references,
        "caveats": parsed.get("caveats"),
    }


# ---------------------------------------------------------------------------
# Node 4: Format response — finalise status
# ---------------------------------------------------------------------------


def node_format_response(state: StateDict) -> StateDict:
    """Set the terminal status flag based on what was produced.

    Reads:  ``chunks``, ``answer``, ``errors``
    Writes: ``status``
    """
    errors: list[str] = state.get("errors", [])
    chunks: list[dict[str, Any]] = state.get("chunks", [])

    if errors and state.get("status") == "failed":
        return {"status": "failed"}
    if not chunks:
        return {"status": "no_results"}
    return {"status": "completed"}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_references(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a references list from raw chunk dicts (fallback when LLM fails)."""
    refs: list[dict[str, Any]] = []
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        text = meta.get("chunk_text") or meta.get("text") or ""
        refs.append(
            {
                "chunk_id": chunk.get("id", ""),
                "page_number": meta.get("page_start", 1),
                "section": meta.get("section", "general"),
                "score": chunk.get("score", 0.0),
                "snippet": text[:250],
            }
        )
    return refs


def route_after_retrieval(state: StateDict) -> str:
    """Conditional edge: skip synthesis if no chunks were found or retrieval failed."""
    if state.get("status") == "failed":
        return "format_response"
    if not state.get("chunks"):
        return "format_response"
    return "synthesize_answer"

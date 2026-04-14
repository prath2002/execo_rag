"""State schema for the LangGraph query agent pipeline."""

from __future__ import annotations

from typing import Any, TypedDict


class QueryAgentState(TypedDict, total=False):
    """Mutable state passed between query agent nodes.

    All fields are optional (``total=False``) so each node only needs to
    return the keys it actually modifies.

    Flow::

        query, top_k, request_id
            ↓  [analyze_query]
        intent, filter_params, refined_query, reasoning
            ↓  [retrieve_chunks]
        chunks
            ↓  [synthesize_answer]  (skipped when chunks is empty)
        answer, confidence, key_findings, references, caveats
            ↓  [format_response]
        status = "completed" | "no_results" | "failed"
    """

    # ── Input fields (set by the caller before graph.invoke) ─────────
    query: str
    top_k: int
    request_id: str

    # ── After analyze_query ───────────────────────────────────────────
    intent: str
    filter_params: dict[str, Any]
    refined_query: str
    reasoning: str

    # ── After retrieve_chunks ─────────────────────────────────────────
    chunks: list[dict[str, Any]]  # raw Pinecone match dicts

    # ── After synthesize_answer ───────────────────────────────────────
    answer: str
    confidence: str              # "high" | "medium" | "low"
    key_findings: list[str]
    references: list[dict[str, Any]]
    caveats: str | None

    # ── Control ───────────────────────────────────────────────────────
    errors: list[str]
    status: str                  # "running" | "completed" | "no_results" | "failed"

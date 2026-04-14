"""LangGraph query agent pipeline.

Builds and runs a 4-node directed graph that:
  1. Analyzes the natural-language query (LLM → intent + filters)
  2. Retrieves relevant chunks from Pinecone (vector + metadata filters)
  3. Synthesizes a structured answer from the chunks (LLM)
  4. Formats the final response with status

Public entry point::

    from execo_rag.services.query.agent_graph import run_query_agent

    result = run_query_agent(request_id="req_123", query="What is the purchase price?")
    print(result["answer"])
    print(result["references"])
"""

from __future__ import annotations

import logging
from typing import Any

from execo_rag.services.query.agent_nodes import (
    node_analyze_query,
    node_format_response,
    node_retrieve_chunks,
    node_synthesize_answer,
    route_after_retrieval,
)
from execo_rag.services.query.agent_state import QueryAgentState

logger = logging.getLogger(__name__)

# Node name constants
_ANALYZE = "analyze_query"
_RETRIEVE = "retrieve_chunks"
_SYNTHESIZE = "synthesize_answer"
_FORMAT = "format_response"


def build_query_agent_graph() -> Any:
    """Build and compile the LangGraph query agent pipeline.

    Graph topology::

        START → analyze_query → retrieve_chunks
                                    ↓ (conditional)
                             synthesize_answer  ←──── chunks found
                                    ↓
                             format_response    ←──── no chunks / error
                                    ↓
                                  END

    Returns:
        Compiled ``CompiledGraph`` ready to invoke.

    Raises:
        ImportError: If ``langgraph`` is not installed.
    """
    try:
        from langgraph.graph import END as LG_END  # type: ignore[import-untyped]
        from langgraph.graph import StateGraph  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "langgraph is required for the query agent. "
            "Install with: pip install langgraph"
        ) from exc

    graph = StateGraph(QueryAgentState)

    # --- Register nodes ---
    graph.add_node(_ANALYZE, node_analyze_query)
    graph.add_node(_RETRIEVE, node_retrieve_chunks)
    graph.add_node(_SYNTHESIZE, node_synthesize_answer)
    graph.add_node(_FORMAT, node_format_response)

    # --- Entry point ---
    graph.set_entry_point(_ANALYZE)

    # --- Linear edges ---
    graph.add_edge(_ANALYZE, _RETRIEVE)

    # --- Conditional edge after retrieval ---
    graph.add_conditional_edges(
        _RETRIEVE,
        route_after_retrieval,
        {
            "synthesize_answer": _SYNTHESIZE,
            "format_response": _FORMAT,
        },
    )

    # --- Terminal edges ---
    graph.add_edge(_SYNTHESIZE, _FORMAT)
    graph.add_edge(_FORMAT, LG_END)

    compiled = graph.compile()
    logger.debug("Query agent graph compiled successfully")
    return compiled


def run_query_agent(
    request_id: str,
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run the full query agent pipeline for a single natural-language question.

    Args:
        request_id: Request identifier for traceability.
        query: Natural-language question about the indexed SPA documents.
        top_k: Maximum number of chunks to retrieve from Pinecone (1–25).

    Returns:
        Final ``QueryAgentState`` dict containing:
        - ``answer``: Synthesized textual answer
        - ``confidence``: "high" | "medium" | "low"
        - ``key_findings``: List of short factual findings
        - ``references``: List of chunk reference dicts
        - ``caveats``: Any limitations or None
        - ``intent``: What the agent understood the query to mean
        - ``filter_params``: Filters applied to the Pinecone query
        - ``refined_query``: Query used for vector embedding
        - ``chunks``: Raw retrieved Pinecone matches
        - ``status``: "completed" | "no_results" | "failed"
        - ``errors``: List of error messages (empty on success)
    """
    logger.info(
        "Query agent pipeline starting",
        extra={
            "extra_data": {
                "request_id": request_id,
                "query_preview": query[:100],
                "top_k": top_k,
            }
        },
    )

    initial_state: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "request_id": request_id,
        "errors": [],
        "status": "running",
    }

    graph = build_query_agent_graph()
    final_state: dict[str, Any] = graph.invoke(initial_state)

    logger.info(
        "Query agent pipeline finished",
        extra={
            "extra_data": {
                "request_id": request_id,
                "status": final_state.get("status"),
                "chunks_retrieved": len(final_state.get("chunks", [])),
                "confidence": final_state.get("confidence"),
                "error_count": len(final_state.get("errors", [])),
            }
        },
    )

    return final_state

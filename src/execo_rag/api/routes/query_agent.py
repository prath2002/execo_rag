"""Query agent route: POST /query/agent.

Runs the 4-node LangGraph query agent:
  1. analyze_query  — LLM extracts intent and Pinecone metadata filters
  2. retrieve_chunks — embeds refined query, queries Pinecone with filters
  3. synthesize_answer — LLM synthesizes a structured answer from chunks
  4. format_response — packages final status

Returns a rich :class:`AgentQueryResponse` with the answer, confidence,
key findings, and chunk references.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from execo_rag.api.deps import get_request_logger
from execo_rag.api.schemas.agent_query import AgentQueryRequest, AgentQueryReference, AgentQueryResponse
from execo_rag.services.query.agent_graph import run_query_agent

router = APIRouter(prefix="/query", tags=["retrieval"])

logger = logging.getLogger(__name__)

def normalize_to_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return [str(value)]

def normalize_confidence(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except:
            return 0.0
    return 0.0

@router.post(
    "/agent",
    response_model=AgentQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Agentic RAG query with automatic filter extraction and answer synthesis",
    description=(
        "Accepts a natural-language question, uses an LLM agent to extract Pinecone "
        "metadata filters, retrieves relevant chunks, and returns a synthesized answer "
        "with source references and confidence rating."
    ),
)
async def query_agent(
    request: Request,
    body: AgentQueryRequest,
    req_logger: logging.Logger = Depends(get_request_logger),
) -> AgentQueryResponse:
    """Run the agentic RAG query pipeline.

    - **query**: Natural-language question (e.g. *"What is the indemnification cap?"*)
    - **top_k**: Maximum number of chunks to retrieve (1–25, default 5)

    The agent will:
    1. Interpret the question and decide which SPA metadata filters to apply.
    2. Embed the refined query and search Pinecone.
    3. Synthesize a grounded answer with references from the retrieved chunks.
    """
    request_id: str = getattr(request.state, "request_id", "req_unknown")

    req_logger.info(
        "Agent query request received",
        extra={
            "extra_data": {
                "query_preview": body.query[:100],
                "top_k": body.top_k,
            }
        },
    )

    try:
        final_state = run_query_agent(
            request_id=request_id,
            query=body.query,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.error(
            "Agent query pipeline raised an unexpected error",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "agent_pipeline_error",
                "message": "The query agent pipeline encountered an unexpected error.",
                "request_id": request_id,
            },
        ) from exc

    pipeline_status = final_state.get("status", "failed")
    errors = final_state.get("errors", [])

    if pipeline_status == "failed" and errors:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error_code": "retrieval_failed",
                "message": errors[0] if errors else "Retrieval failed.",
                "request_id": request_id,
            },
        )

    # Build typed references
    raw_refs: list[dict] = final_state.get("references", [])
    references = [
        AgentQueryReference(
            chunk_id=r.get("chunk_id", ""),
            page_number=int(r.get("page_number", 1)),
            section=str(r.get("section", "general")),
            score=float(r.get("score", 0.0)),
            snippet=str(r.get("snippet", ""))[:250],
        )
        for r in raw_refs
        if isinstance(r, dict)
    ]

    no_results = pipeline_status == "no_results"
    answer = (
        final_state.get("answer")
        or (
            "No relevant chunks were found in the database for your query. "
            "Try broadening the question or checking that documents have been ingested."
            if no_results
            else "Answer generation failed."
        )
    )

    response = AgentQueryResponse(
        request_id=request_id,
        query=body.query,
        refined_query=final_state.get("refined_query") or body.query,
        intent=final_state.get("intent") or body.query,
        filter_params_used=final_state.get("filter_params", {}),
        answer=answer,
        confidence=normalize_confidence(final_state.get("confidence")),
        key_findings=normalize_to_list(final_state.get("key_findings")) or [],
        references=references,
        caveats=normalize_to_list(final_state.get("caveats")),
        chunks_retrieved=len(final_state.get("chunks", [])),
        reasoning=final_state.get("reasoning", ""),
        status=pipeline_status,
    )

    req_logger.info(
        "Agent query request completed",
        extra={
            "extra_data": {
                "status": pipeline_status,
                "chunks_retrieved": response.chunks_retrieved,
                "confidence": response.confidence,
                "references_count": len(references),
            }
        },
    )

    return response


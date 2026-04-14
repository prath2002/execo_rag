"""Schemas for the LangGraph query agent endpoint."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from execo_rag.models.common import ExecoBaseModel


class AgentQueryRequest(ExecoBaseModel):
    """Request body for POST /query/agent."""

    query: str = Field(min_length=1, description="Natural-language question about the indexed SPAs.")
    top_k: int = Field(default=5, ge=1, le=25, description="Max chunks to retrieve from Pinecone.")


class AgentQueryReference(ExecoBaseModel):
    """A single chunk cited in the agent's answer."""

    chunk_id: str = Field(description="Unique chunk identifier in Pinecone.")
    page_number: int = Field(ge=1, description="Page number in the source document.")
    section: str = Field(description="SPA section label (e.g. 'indemnification').")
    score: float = Field(description="Cosine similarity score [0, 1].")
    snippet: str = Field(description="Most relevant excerpt from the chunk (≤ 250 chars).")


class AgentQueryResponse(ExecoBaseModel):
    """Structured response from the LangGraph query agent."""

    request_id: str = Field(description="Unique request identifier for traceability.")
    query: str = Field(description="Original user question.")
    refined_query: str = Field(description="Query rewritten by the agent for vector search.")
    intent: str = Field(description="Agent's interpretation of the question.")
    filter_params_used: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters applied to the Pinecone query.",
    )
    answer: str = Field(description="Synthesized answer grounded in the retrieved chunks.")
    confidence: float = Field(
    ge=0.0,
    le=1.0,
    description="Answer confidence score between 0 and 1."
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Short, standalone factual findings extracted from the chunks.",
    )
    references: list[AgentQueryReference] = Field(
        default_factory=list,
        description="Source chunks cited in the answer.",
    )
    caveats: list[str] | None = None
    chunks_retrieved: int = Field(description="Total chunks fetched from Pinecone.")
    reasoning: str = Field(
        default="",
        description="Agent's reasoning for the filter choices made.",
    )
    status: str = Field(description="Pipeline status: 'completed', 'no_results', or 'failed'.")

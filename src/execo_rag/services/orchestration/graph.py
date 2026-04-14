"""LangGraph ingestion pipeline graph assembly.

Builds and compiles the end-to-end ingestion graph:

    load_document → extract_pdf → clean_text → extract_metadata
    → normalize_metadata → detect_sections → chunk_document
    → enrich_chunks → validate_chunks → generate_embeddings → store_vectors

At each step a routing function checks for FAILED status and short-circuits
to END, preventing downstream nodes from running after a failure.

Usage::

    from execo_rag.services.orchestration.graph import build_ingestion_graph
    from execo_rag.models.pipeline_state import PipelineState, PipelineStatus

    graph = build_ingestion_graph()
    result = graph.invoke(initial_state)
    final_state = PipelineState(**result)
"""

from __future__ import annotations

import logging
from typing import Any

from execo_rag.models.pipeline_state import PipelineState, PipelineStatus
from execo_rag.services.orchestration.nodes import (
    node_chunk_document,
    node_clean_text,
    node_detect_sections,
    node_enrich_chunks,
    node_extract_metadata,
    node_extract_pdf,
    node_generate_embeddings,
    node_load_document,
    node_normalize_metadata,
    node_store_vectors,
    node_validate_chunks,
)
from execo_rag.services.orchestration.routing import END, route_or_fail

logger = logging.getLogger(__name__)

# Node name constants — single source of truth for edges
_LOAD = "load_document"
_EXTRACT = "extract_pdf"
_CLEAN = "clean_text"
_METADATA = "extract_metadata"
_NORMALIZE = "normalize_metadata"
_SECTIONS = "detect_sections"
_CHUNK = "chunk_document"
_ENRICH = "enrich_chunks"
_VALIDATE = "validate_chunks"
_EMBED = "generate_embeddings"
_STORE = "store_vectors"


def _build_state_schema() -> type:
    """Return a TypedDict-compatible state schema for LangGraph.

    LangGraph requires a TypedDict or Pydantic model as the state type.
    We derive a minimal schema from PipelineState to avoid strict validation
    issues during graph state merging.
    """
    from typing import TypedDict

    class IngestState(TypedDict, total=False):
        request_id: str
        pdf_path: Any
        document: Any
        status: str
        extracted_document: Any
        cleaned_pages: list
        document_metadata: Any
        sections: list
        raw_chunks: list
        chunks: list
        validated_chunks: list
        embeddings: list
        pinecone_records: list
        errors: list

    return IngestState


def build_ingestion_graph() -> Any:
    """Build and compile the LangGraph ingestion pipeline.

    Returns:
        A compiled LangGraph ``CompiledGraph`` that can be invoked with a
        state dict matching the ``IngestState`` schema.

    Raises:
        ImportError: If ``langgraph`` is not installed.
    """
    try:
        from langgraph.graph import END as LG_END  # type: ignore[import-untyped]
        from langgraph.graph import StateGraph  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "langgraph is required for pipeline orchestration. "
            "Install with: pip install langgraph"
        ) from exc

    IngestState = _build_state_schema()
    graph = StateGraph(IngestState)

    # --- Register nodes ---
    graph.add_node(_LOAD, node_load_document)
    graph.add_node(_EXTRACT, node_extract_pdf)
    graph.add_node(_CLEAN, node_clean_text)
    graph.add_node(_METADATA, node_extract_metadata)
    graph.add_node(_NORMALIZE, node_normalize_metadata)
    graph.add_node(_SECTIONS, node_detect_sections)
    graph.add_node(_CHUNK, node_chunk_document)
    graph.add_node(_ENRICH, node_enrich_chunks)
    graph.add_node(_VALIDATE, node_validate_chunks)
    graph.add_node(_EMBED, node_generate_embeddings)
    graph.add_node(_STORE, node_store_vectors)

    # --- Entry point ---
    graph.set_entry_point(_LOAD)

    # --- Conditional edges: continue or short-circuit on FAILED ---
    for current, next_node in [
        (_LOAD, _EXTRACT),
        (_EXTRACT, _CLEAN),
        (_CLEAN, _METADATA),
        (_METADATA, _NORMALIZE),
        (_NORMALIZE, _SECTIONS),
        (_SECTIONS, _CHUNK),
        (_CHUNK, _ENRICH),
        (_ENRICH, _VALIDATE),
        (_VALIDATE, _EMBED),
        (_EMBED, _STORE),
    ]:
        router = route_or_fail(next_node)
        graph.add_conditional_edges(
            current,
            router,
            {next_node: next_node, END: LG_END},
        )

    # --- Terminal edge ---
    graph.add_edge(_STORE, LG_END)

    compiled = graph.compile()
    logger.debug("Ingestion pipeline graph compiled successfully")
    return compiled


def run_ingestion_pipeline(
    request_id: str,
    pdf_path: str,
    document_id: str,
    document_type: str = "share_purchase_agreement",
) -> dict[str, Any]:
    """Run the full ingestion pipeline for a single PDF document.

    Args:
        request_id: Request identifier for traceability.
        pdf_path: Absolute path to the PDF file.
        document_id: Stable document identifier.
        document_type: Document type enum value (default SPA).

    Returns:
        Final state dict with ``status``, ``validated_chunks``,
        ``embeddings``, and ``errors``.
    """
    from pathlib import Path

    from execo_rag.models.document import (
        DocumentInput,
        DocumentSource,
        DocumentType,
        SourceType,
    )

    # Build a minimal source for the initial state (will be re-validated in node 1)
    source = DocumentSource(
        source_type=SourceType.LOCAL_FILE,
        path=Path(pdf_path),
        file_name=Path(pdf_path).name,
        mime_type="application/pdf",
    )

    try:
        doc_type = DocumentType(document_type)
    except ValueError:
        doc_type = DocumentType.UNKNOWN

    document = DocumentInput(
        document_id=document_id,
        source=source,
        document_type=doc_type,
    )

    initial_state: dict[str, Any] = {
        "request_id": request_id,
        "pdf_path": str(pdf_path),
        "document": document,
        "status": PipelineStatus.PENDING,
        "errors": [],
        "cleaned_pages": [],
        "sections": [],
        "raw_chunks": [],
        "chunks": [],
        "validated_chunks": [],
        "embeddings": [],
        "pinecone_records": [],
    }

    logger.info(
        "Ingestion pipeline starting",
        extra={
            "extra_data": {
                "request_id": request_id,
                "document_id": document_id,
                "pdf_path": str(pdf_path),
            }
        },
    )

    graph = build_ingestion_graph()
    final_state: dict[str, Any] = graph.invoke(initial_state)

    final_status = final_state.get("status", PipelineStatus.FAILED)
    logger.info(
        "Ingestion pipeline finished",
        extra={
            "extra_data": {
                "request_id": request_id,
                "document_id": document_id,
                "final_status": final_status.value if hasattr(final_status, "value") else str(final_status),
                "error_count": len(final_state.get("errors", [])),
            }
        },
    )

    return final_state

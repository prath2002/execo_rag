"""LangGraph pipeline node functions.

Each node is a pure function that receives the current graph state as a dict
and returns a dict containing only the fields it updates.  LangGraph merges
these updates into the running state automatically.

Node execution order (defined in graph.py):
  load_document → extract_pdf → clean_text → extract_metadata
  → normalize_metadata → detect_sections → chunk_document
  → enrich_chunks → validate_chunks → generate_embeddings
  → store_vectors → [DONE]

All nodes catch exceptions and record them in ``state["errors"]`` while
transitioning to ``PipelineStatus.FAILED`` so the router can short-circuit.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any

from execo_rag.config import get_settings
from execo_rag.logging.context import set_task_name
from execo_rag.models.pipeline_state import PipelineStatus
from execo_rag.services.chunking.enricher import enrich_chunks
from execo_rag.services.chunking.hybrid_chunker import chunk_segments
from execo_rag.services.chunking.section_detector import detect_sections
from execo_rag.services.chunking.validator import validate_chunks
from execo_rag.services.embeddings.provider import create_embedding_provider, embed_validated_chunks
from execo_rag.services.ingestion.cleaner import clean_pages
from execo_rag.services.ingestion.extractor import extract_pdf
from execo_rag.services.ingestion.loader import load_document_source
from execo_rag.services.metadata.extractor import extract_document_metadata
from execo_rag.services.metadata.normalizer import normalize_metadata
from execo_rag.services.vectorstore.pinecone_store import PineconeStore

logger = logging.getLogger(__name__)

# Type alias for the state dict passed between LangGraph nodes
StateDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fail(state: StateDict, node_name: str, exc: Exception) -> StateDict:
    """Record a failure in the pipeline state and return updated state."""
    tb = traceback.format_exc()
    error_msg = f"[{node_name}] {type(exc).__name__}: {exc}"
    logger.error(
        "Pipeline node failed",
        extra={
            "extra_data": {
                "node": node_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": tb,
            }
        },
    )
    current_errors: list[str] = list(state.get("errors", []))
    current_errors.append(error_msg)
    return {"status": PipelineStatus.FAILED, "errors": current_errors}


def _start_node(name: str) -> float:
    """Log node start and return start timestamp."""
    set_task_name(name)
    logger.info(
        "Pipeline node started",
        extra={"extra_data": {"node": name}},
    )
    return time.perf_counter()


def _end_node(name: str, start: float) -> None:
    """Log node completion with duration."""
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    logger.info(
        "Pipeline node completed",
        extra={"extra_data": {"node": name, "duration_ms": elapsed_ms}},
    )


# ---------------------------------------------------------------------------
# Node 1: Load document source
# ---------------------------------------------------------------------------


def node_load_document(state: StateDict) -> StateDict:
    """Validate the PDF file and compute its hash.

    Reads: ``pdf_path``, ``document``
    Writes: ``document`` (with populated source.file_hash)
    """
    start = _start_node("load_document")
    try:
        from execo_rag.models.document import DocumentInput, SourceType

        pdf_path = Path(state["pdf_path"])
        document: DocumentInput = state["document"]

        source = load_document_source(pdf_path, source_type=SourceType.LOCAL_FILE)

        # Rebuild document with the validated source
        updated_document = DocumentInput(
            document_id=document.document_id,
            source=source,
            document_type=document.document_type,
        )

        _end_node("load_document", start)
        return {"document": updated_document, "status": PipelineStatus.RUNNING}
    except Exception as exc:
        return _fail(state, "load_document", exc)


# ---------------------------------------------------------------------------
# Node 2: Extract PDF text
# ---------------------------------------------------------------------------


def node_extract_pdf(state: StateDict) -> StateDict:
    """Run the primary (and fallback) PDF extractor.

    Reads: ``document``
    Writes: ``extracted_document``
    """
    start = _start_node("extract_pdf")
    try:
        settings = get_settings()
        document = state["document"]
        pdf_path = Path(state["pdf_path"])

        extracted = extract_pdf(
            path=pdf_path,
            document_id=document.document_id,
            preferred_extractor=settings.pdf.extractor,
        )

        logger.info(
            "Text extraction complete",
            extra={
                "extra_data": {
                    "document_id": document.document_id,
                    "page_count": len(extracted.pages),
                    "extractor_used": extracted.extractor_used,
                }
            },
        )
        _end_node("extract_pdf", start)
        return {"extracted_document": extracted}
    except Exception as exc:
        return _fail(state, "extract_pdf", exc)


# ---------------------------------------------------------------------------
# Node 3: Clean extracted text
# ---------------------------------------------------------------------------


def node_clean_text(state: StateDict) -> StateDict:
    """Clean raw extracted page text.

    Reads: ``extracted_document``
    Writes: ``cleaned_pages``
    """
    start = _start_node("clean_text")
    try:
        extracted = state["extracted_document"]
        cleaned = clean_pages(extracted.pages)

        _end_node("clean_text", start)
        return {"cleaned_pages": cleaned}
    except Exception as exc:
        return _fail(state, "clean_text", exc)


# ---------------------------------------------------------------------------
# Node 4: Extract document metadata
# ---------------------------------------------------------------------------


def node_extract_metadata(state: StateDict) -> StateDict:
    """Extract structured metadata from cleaned pages.

    Reads: ``document``, ``cleaned_pages``
    Writes: ``document_metadata``
    """
    start = _start_node("extract_metadata")
    try:
        settings = get_settings()
        document = state["document"]
        cleaned_pages = state["cleaned_pages"]

        metadata = extract_document_metadata(
            document_id=document.document_id,
            cleaned_pages=cleaned_pages,
            enable_llm_fallback=settings.runtime.enable_llm_fallback,
            openrouter_api_key=settings.openrouter.api_key,
            llm_model=settings.openrouter.chat_model,
        )

        non_null_fields = sum(
                1 for v in metadata.model_dump().values() if v is not None
        )
        logger.info(
            "Metadata extracted",
            extra={
                "extra_data": {
                    "document_id": document.document_id,
                    "fields_populated": non_null_fields,
                    "total_fields": 8,
                    "buyer": metadata.buyer,
                    "seller": metadata.seller,
                    "governing_law": metadata.governing_law,
                    "effective_date": str(metadata.effective_date) if metadata.effective_date else None,
                }
            },
        )
        _end_node("extract_metadata", start)
        return {"document_metadata": metadata}
    except Exception as exc:
        return _fail(state, "extract_metadata", exc)


# ---------------------------------------------------------------------------
# Node 5: Normalize metadata
# ---------------------------------------------------------------------------


def node_normalize_metadata(state: StateDict) -> StateDict:
    """Normalize extracted metadata values to canonical forms.

    Reads: ``document_metadata``
    Writes: ``document_metadata`` (normalized)
    """
    start = _start_node("normalize_metadata")
    try:
        metadata = state["document_metadata"]
        normalized = normalize_metadata(metadata)

        _end_node("normalize_metadata", start)
        return {"document_metadata": normalized}
    except Exception as exc:
        return _fail(state, "normalize_metadata", exc)


# ---------------------------------------------------------------------------
# Node 6: Detect sections
# ---------------------------------------------------------------------------


def node_detect_sections(state: StateDict) -> StateDict:
    """Detect logical SPA sections from cleaned page text.

    Reads: ``document``, ``cleaned_pages``
    Writes: ``sections`` (list[SectionSegment] stored in state as raw list)

    Note: SectionSegments are passed through the state as a transient
    field used only by the chunking node.
    """
    start = _start_node("detect_sections")
    try:
        document = state["document"]
        cleaned_pages = state["cleaned_pages"]

        sections = detect_sections(
            cleaned_pages=cleaned_pages,
            document_id=document.document_id,
        )

        _end_node("detect_sections", start)
        return {"sections": sections}
    except Exception as exc:
        return _fail(state, "detect_sections", exc)


# ---------------------------------------------------------------------------
# Node 7: Chunk document
# ---------------------------------------------------------------------------


def node_chunk_document(state: StateDict) -> StateDict:
    """Produce token-bounded chunks from section segments.

    Reads: ``sections``, ``document``
    Writes: ``raw_chunks`` (transient)
    """
    start = _start_node("chunk_document")
    try:
        settings = get_settings()
        sections = state.get("sections", [])
        document = state["document"]

        raw_chunks = chunk_segments(
            segments=sections,
            document_id=document.document_id,
            max_tokens=settings.pdf.max_chunk_tokens,
            overlap_tokens=settings.pdf.chunk_overlap_tokens,
            model_name=settings.embeddings.model,
        )

        logger.info(
            "Document chunked",
            extra={
                "extra_data": {
                    "document_id": document.document_id,
                    "section_count": len(sections),
                    "chunk_count": len(raw_chunks),
                    "max_chunk_tokens": settings.pdf.max_chunk_tokens,
                }
            },
        )
        _end_node("chunk_document", start)
        return {"raw_chunks": raw_chunks}
    except Exception as exc:
        return _fail(state, "chunk_document", exc)


# ---------------------------------------------------------------------------
# Node 8: Enrich chunks
# ---------------------------------------------------------------------------


def node_enrich_chunks(state: StateDict) -> StateDict:
    """Enrich raw chunks with document-level metadata and filter flags.

    Reads: ``raw_chunks``, ``document_metadata``
    Writes: ``chunks``
    """
    start = _start_node("enrich_chunks")
    try:
        raw_chunks = state.get("raw_chunks", [])
        metadata = state["document_metadata"]

        enriched = enrich_chunks(chunks=raw_chunks, metadata=metadata)

        _end_node("enrich_chunks", start)
        return {"chunks": enriched}
    except Exception as exc:
        return _fail(state, "enrich_chunks", exc)


# ---------------------------------------------------------------------------
# Node 9: Validate chunks
# ---------------------------------------------------------------------------


def node_validate_chunks(state: StateDict) -> StateDict:
    """Validate enriched chunks for schema and consistency.

    Reads: ``chunks``
    Writes: ``validated_chunks``
    """
    start = _start_node("validate_chunks")
    try:
        chunks = state.get("chunks", [])
        validated = validate_chunks(enriched_chunks=chunks, raise_on_invalid=False)

        valid_count = sum(1 for c in validated if getattr(c, "is_valid", False))
        logger.info(
            "Chunks validated",
            extra={
                "extra_data": {
                    "total_chunks": len(validated),
                    "valid_chunks": valid_count,
                    "invalid_chunks": len(validated) - valid_count,
                }
            },
        )
        _end_node("validate_chunks", start)
        return {"validated_chunks": validated}
    except Exception as exc:
        return _fail(state, "validate_chunks", exc)


# ---------------------------------------------------------------------------
# Node 10: Generate embeddings
# ---------------------------------------------------------------------------


def node_generate_embeddings(state: StateDict) -> StateDict:
    """Generate embeddings for all valid chunks using the configured provider.

    Reads: ``validated_chunks``
    Writes: ``embeddings``
    """
    start = _start_node("generate_embeddings")
    try:
        settings = get_settings()
        validated_chunks = state.get("validated_chunks", [])

        provider = create_embedding_provider(
            provider_name=settings.embeddings.provider,
            model_name=settings.embeddings.model,
            batch_size=settings.embeddings.batch_size,
            api_key=settings.openrouter.api_key,
            expected_dimension=settings.embeddings.dimension,
        )

        embedding_result = embed_validated_chunks(
            chunks=validated_chunks,
            provider=provider,
            model_name=settings.embeddings.model,
        )

        logger.info(
            "Embeddings generated",
            extra={
                "extra_data": {
                    "vector_count": len(embedding_result.vectors),
                    "dimension": embedding_result.vectors[0].dimension if embedding_result.vectors else 0,
                    "model": settings.embeddings.model,
                }
            },
        )
        _end_node("generate_embeddings", start)
        return {"embeddings": embedding_result.vectors}
    except Exception as exc:
        return _fail(state, "generate_embeddings", exc)


# ---------------------------------------------------------------------------
# Node 11: Store vectors in Pinecone
# ---------------------------------------------------------------------------


def node_store_vectors(state: StateDict) -> StateDict:
    """Upsert validated chunks and their embeddings to Pinecone.

    Reads: ``validated_chunks``, ``embeddings``, ``document``
    Writes: ``pinecone_records`` (lightweight record refs), ``status``
    """
    start = _start_node("store_vectors")
    try:
        from execo_rag.models.embedding import EmbeddingBatchResult, EmbeddingVector

        settings = get_settings()
        validated_chunks = state.get("validated_chunks", [])
        embeddings = state.get("embeddings", [])

        store = PineconeStore(
            api_key=settings.pinecone.api_key,
            index_name=settings.pinecone.index_name,
            namespace=settings.pinecone.namespace,
        )

        # Re-wrap embeddings into EmbeddingBatchResult
        embedding_result = EmbeddingBatchResult(
            model_name=settings.embeddings.model,
            vectors=[
                EmbeddingVector(
                    chunk_id=v.chunk_id,
                    values=v.values,
                    dimension=v.dimension,
                )
                for v in embeddings
            ],
        )

        upserted = store.upsert_chunks(
            chunks=validated_chunks,
            embeddings=embedding_result,
            namespace=settings.pinecone.namespace,
        )

        logger.info(
            "Vectors stored in Pinecone",
            extra={"extra_data": {"upserted_count": upserted}},
        )

        _end_node("store_vectors", start)
        return {"status": PipelineStatus.COMPLETED}
    except Exception as exc:
        return _fail(state, "store_vectors", exc)

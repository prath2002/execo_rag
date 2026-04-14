"""Ingestion API route: POST /ingest.

Accepts a PDF file path and document metadata, runs the full ingestion
pipeline, and returns a structured response with status and chunk counts.

The route runs the pipeline synchronously within the request.  For very
large documents, switch the service call to a BackgroundTask and return
202 Accepted — the scaffolding is included as a commented block.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from execo_rag.api.deps import get_app_settings, get_request_logger
from execo_rag.api.schemas.ingest import IngestRequest, IngestResponse
from execo_rag.config import Settings
from execo_rag.logging.context import set_document_id
from execo_rag.services.ingestion.ingest_service import ingest_document
from execo_rag.utils.exceptions import AppError

router = APIRouter(prefix="/ingest", tags=["ingestion"])

logger = logging.getLogger(__name__)


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a PDF document into the vector pipeline",
    description=(
        "Validates the PDF, extracts text, identifies metadata, chunks the document, "
        "generates embeddings, and indexes into Pinecone.  Idempotent by file hash."
    ),
)
async def ingest(
    request: Request,
    body: IngestRequest,
    req_logger: logging.Logger = Depends(get_request_logger),
    settings: Settings = Depends(get_app_settings),
) -> IngestResponse:
    """Run the full ingestion pipeline for a single PDF document.

    - **document_id**: Caller-supplied stable identifier for the document.
    - **file_path**: Absolute or relative path to the PDF on the server.
    - **document_type**: One of ``share_purchase_agreement`` or ``unknown``.

    Returns ingestion ``status``, ``request_id``, and a human-readable ``message``.
    """
    request_id: str = getattr(request.state, "request_id", "req_unknown")
    set_document_id(body.document_id)

    req_logger.info(
        "Ingest request received",
        extra={
            "extra_data": {
                "document_id": body.document_id,
                "file_path": str(body.file_path),
                "document_type": body.document_type.value,
            }
        },
    )

    try:
        result = ingest_document(
            request_id=request_id,
            file_path=str(body.file_path),
            document_id=body.document_id,
            document_type=body.document_type.value,
        )
    except AppError as exc:
        logger.error(
            "Ingest failed with application error",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "document_id": body.document_id,
                    "error_code": exc.error_code,
                    "error": exc.message,
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": exc.error_code,
                "message": exc.message,
                "request_id": request_id,
            },
        ) from exc
    except Exception as exc:
        logger.error(
            "Ingest failed with unexpected error",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "document_id": body.document_id,
                    "error": str(exc),
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "internal_error",
                "message": "An unexpected error occurred during ingestion.",
                "request_id": request_id,
            },
        ) from exc

    req_logger.info(
        "Ingest request completed",
        extra={
            "extra_data": {
                "document_id": result.document_id,
                "status": result.status,
                "chunk_count": result.chunk_count,
                "valid_chunk_count": result.valid_chunk_count,
                "skipped": result.skipped,
            }
        },
    )

    return IngestResponse(
        request_id=request_id,
        document_id=result.document_id,
        status=result.status,
        message=result.message,
    )

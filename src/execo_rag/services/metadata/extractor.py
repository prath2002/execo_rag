"""Document metadata extraction service.

Phase D6: Rule-based extraction for the fixed SPA schema.
Phase D7: Optional LLM fallback for missing or low-confidence fields.

Extraction flow:
  1. Run all regex rules against cleaned page text.
  2. For each missing or low-confidence field, attempt the LLM fallback
     (only if ``enable_llm_fallback=True`` in settings).
  3. Return a :class:`DocumentMetadata` with :class:`MetadataField` wrappers
     carrying evidence and confidence for every field.
"""

from __future__ import annotations

import json
import logging
import re
import time
from decimal import Decimal
from pathlib import Path

from execo_rag.clients import OpenRouterClient
from execo_rag.config.constants import DEFAULT_OPENROUTER_CHAT_MODEL
from execo_rag.models.extraction import CleanedPage
from execo_rag.models.metadata import DocumentMetadata, MetadataEvidence, MetadataField
from execo_rag.utils.dates import extract_date_from_text, parse_date
from execo_rag.utils.exceptions import MetadataExtractionError
from execo_rag.utils.money import parse_money
from execo_rag.utils.text import clean_field_value

from . import rules as R

logger = logging.getLogger(__name__)

# Fields whose confidence is considered "low" and triggers the LLM fallback
_LOW_CONFIDENCE_THRESHOLD = 0.60

# Path to the metadata extraction prompt template
_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "metadata_extraction.txt"
_METADATA_FIELD_NAMES = [
    "document_type",
    "effective_date",
    "buyer",
    "company_target",
    "seller",
    "shares_transacted",
    "cash_purchase_price",
    "escrow_agent",
    "escrow_amount",
    "target_working_capital",
    "indemnification_de_minimis_amount",
    "indemnification_basket_amount",
    "indemnification_cap_amount",
    "governing_law",
]


# ---------------------------------------------------------------------------
# Helper: build pages_text list
# ---------------------------------------------------------------------------


def _build_pages_text(cleaned_pages: list[CleanedPage]) -> list[tuple[int, str]]:
    """Convert CleanedPage objects to (page_number, text) tuples."""
    return [(p.page_number, p.cleaned_text) for p in cleaned_pages]


def _full_text(pages_text: list[tuple[int, str]]) -> str:
    """Concatenate all page texts into a single string."""
    return "\n\n".join(text for _, text in pages_text)


# ---------------------------------------------------------------------------
# Rule-based extraction (D6)
# ---------------------------------------------------------------------------


def _run_rules(
    document_id: str,
    pages_text: list[tuple[int, str]],
) -> DocumentMetadata:
    """Run all regex rules and populate a :class:`DocumentMetadata` object.

    Args:
        document_id: Stable document identifier.
        pages_text: List of (page_number, cleaned_text) tuples.

    Returns:
        :class:`DocumentMetadata` with fields populated from rule matches.
    """
    meta = DocumentMetadata(document_id=document_id)

    def _apply(
        field_name: str,
        match: R.RuleMatch | None,
        converter: object = None,
    ) -> None:
        """Apply a rule match to the metadata object."""
        if match is None:
            return
        raw = match.raw_value
        value: object = raw
        if converter is not None:
            try:
                value = converter(raw)  # type: ignore[operator]
            except Exception:
                value = raw

        evidence = MetadataEvidence(
            page_number=match.page_number,
            snippet=match.snippet,
            confidence=match.confidence,
        )
        mf = MetadataField(value=value, confidence=match.confidence, evidence=evidence)
        setattr(meta, field_name, mf)

    _apply("document_type", R.extract_document_type(pages_text), clean_field_value)
    _apply(
        "effective_date",
        R.extract_effective_date(pages_text),
        parse_date,
    )
    _apply("buyer", R.extract_buyer(pages_text), clean_field_value)
    _apply("company_target", R.extract_company_target(pages_text), clean_field_value)
    _apply("seller", R.extract_seller(pages_text), clean_field_value)
    _apply(
        "shares_transacted",
        R.extract_shares_transacted(pages_text),
        lambda v: clean_field_value(v),
    )
    _apply(
        "cash_purchase_price",
        R.extract_cash_purchase_price(pages_text),
        parse_money,
    )
    _apply("escrow_agent", R.extract_escrow_agent(pages_text), clean_field_value)
    _apply("escrow_amount", R.extract_escrow_amount(pages_text), parse_money)
    _apply(
        "target_working_capital",
        R.extract_target_working_capital(pages_text),
        parse_money,
    )
    _apply(
        "indemnification_de_minimis_amount",
        R.extract_indemnification_de_minimis(pages_text),
        parse_money,
    )
    _apply(
        "indemnification_basket_amount",
        R.extract_indemnification_basket(pages_text),
        parse_money,
    )
    _apply(
        "indemnification_cap_amount",
        R.extract_indemnification_cap(pages_text),
        parse_money,
    )
    _apply("governing_law", R.extract_governing_law(pages_text), clean_field_value)

    return meta


# ---------------------------------------------------------------------------
# LLM-based fallback extraction (D7)
# ---------------------------------------------------------------------------


def _load_prompt_template() -> str:
    """Load the metadata extraction prompt from the prompts directory."""
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    # Inline fallback if file not found
    return """You are a legal document analysis assistant. Extract the following metadata fields
from the provided Share Purchase Agreement text.

Return ONLY valid JSON with this exact schema:
{
  "document_type": "string or null",
  "effective_date": "YYYY-MM-DD string or null",
  "buyer": "string or null",
  "company_target": "string or null",
  "seller": "string or null",
  "shares_transacted": "string or null",
  "cash_purchase_price": "numeric string or null",
  "escrow_agent": "string or null",
  "escrow_amount": "numeric string or null",
  "target_working_capital": "numeric string or null",
  "indemnification_de_minimis_amount": "numeric string or null",
  "indemnification_basket_amount": "numeric string or null",
  "indemnification_cap_amount": "numeric string or null",
  "governing_law": "string or null"
}

Fields to extract from:
{document_text}

Only include fields present in the document. Return null for missing fields.
"""


def _missing_fields(meta: DocumentMetadata) -> list[str]:
    """Return field names where value is None or confidence is below threshold."""
    missing = []
    for name in _METADATA_FIELD_NAMES:
        field: MetadataField = getattr(meta, name)  # type: ignore[type-arg]
        if field.value is None or field.confidence < _LOW_CONFIDENCE_THRESHOLD:
            missing.append(name)
    return missing


def _resolve_openrouter_model(model: str) -> str:
    """Normalize the requested model to a free OpenRouter model id."""
    candidate = (model or "").strip()
    if not candidate or candidate.startswith("gpt-"):
        return DEFAULT_OPENROUTER_CHAT_MODEL
    if ":" not in candidate:
        return f"{candidate}:free"
    return candidate


def _parse_llm_json_content(content: str) -> dict[str, object]:
    """Parse strict JSON output and keep only supported metadata fields."""
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    return {
        field_name: payload[field_name]
        for field_name in _METADATA_FIELD_NAMES
        if field_name in payload
    }


def _apply_llm_results(
    meta: DocumentMetadata,
    llm_data: dict[str, object],
    missing_fields: list[str],
    page_number: int = 1,
) -> DocumentMetadata:
    """Merge LLM-extracted fields into *meta* for fields that were missing.

    Args:
        meta: Existing metadata object.
        llm_data: Parsed JSON dict from LLM response.
        missing_fields: Field names to consider for update.
        page_number: Representative page number for LLM evidence.

    Returns:
        Updated :class:`DocumentMetadata`.
    """
    converters: dict[str, object] = {
        "effective_date": parse_date,
        "cash_purchase_price": parse_money,
        "escrow_amount": parse_money,
        "target_working_capital": parse_money,
        "indemnification_de_minimis_amount": parse_money,
        "indemnification_basket_amount": parse_money,
        "indemnification_cap_amount": parse_money,
    }

    for field_name in missing_fields:
        raw = llm_data.get(field_name)
        if raw is None:
            continue
        raw_str = str(raw)
        converter = converters.get(field_name)
        value: object = raw_str
        if converter is not None:
            try:
                value = converter(raw_str)  # type: ignore[operator]
            except Exception:
                value = raw_str

        if value is not None:
            evidence = MetadataEvidence(
                page_number=page_number,
                snippet=raw_str[:200],
                confidence=0.70,
            )
            mf = MetadataField(value=value, confidence=0.70, evidence=evidence)
            setattr(meta, field_name, mf)
            logger.debug(
                "LLM fallback applied field",
                extra={"extra_data": {"field": field_name, "value": str(value)[:80]}},
            )

    return meta


def _run_llm_fallback(
    meta: DocumentMetadata,
    full_text: str,
    missing_fields: list[str],
    document_id: str,
    openrouter_api_key: str,
    model: str = DEFAULT_OPENROUTER_CHAT_MODEL,
) -> DocumentMetadata:
    """Use OpenRouter's chat API to fill missing metadata fields.

    Args:
        meta: Partially populated metadata object.
        full_text: Full cleaned document text (may be truncated).
        missing_fields: Field names that need values.
        document_id: Document identifier for logging.
        openrouter_api_key: OpenRouter API key.
        model: OpenRouter model id or base model name.

    Returns:
        Updated :class:`DocumentMetadata`.
    """
    if not openrouter_api_key:
        logger.warning(
            "OPENROUTER_API_KEY not configured; skipping LLM fallback",
            extra={"extra_data": {"document_id": document_id}},
        )
        return meta

    # Truncate text to ~12 000 chars to stay within context limits
    truncated_text = full_text[:12_000]

    prompt_template = _load_prompt_template()
    prompt = prompt_template.replace("{document_text}", truncated_text)
    resolved_model = _resolve_openrouter_model(model)

    logger.info(
        "Running LLM metadata fallback",
        extra={
            "extra_data": {
                "document_id": document_id,
                "missing_fields": missing_fields,
                "model": resolved_model,
                "provider": "openrouter",
            }
        },
    )

    try:
        client = OpenRouterClient(api_key=openrouter_api_key)
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured legal metadata. "
                        "Return ONLY a valid JSON object — no markdown, no code fences, "
                        "no explanation. Start your response with '{' and end with '}'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
            # NOTE: response_format={"type":"json_object"} is NOT used here because
            # most free models on OpenRouter do not support that parameter and return 400.
            # The system prompt enforces JSON output instead.
        )
        content = (response.choices[0].message.content or "{}").strip()
        # Strip markdown code fences if the model emits them despite instructions
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        llm_data = _parse_llm_json_content(content)
    except Exception as exc:
        logger.error(
            "LLM fallback call failed",
            extra={
                "extra_data": {
                    "document_id": document_id,
                    "model": resolved_model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            },
        )
        return meta

    # Derive a representative page number (middle of document)
    return _apply_llm_results(meta, llm_data, missing_fields, page_number=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_document_metadata(
    document_id: str,
    cleaned_pages: list[CleanedPage],
    enable_llm_fallback: bool = False,
    openrouter_api_key: str = "",
    llm_model: str = DEFAULT_OPENROUTER_CHAT_MODEL,
) -> DocumentMetadata:
    """Extract structured metadata from cleaned PDF pages.

    Args:
        document_id: Stable document identifier.
        cleaned_pages: List of :class:`CleanedPage` from the cleaning service.
        enable_llm_fallback: Whether to call an LLM for missing/low-confidence fields.
        openrouter_api_key: OpenRouter API key for the fallback.
        llm_model: OpenRouter model name for fallback extraction.

    Returns:
        Populated :class:`DocumentMetadata` with evidence and confidence for each field.

    Raises:
        MetadataExtractionError: If the input is empty or a critical error occurs.
    """
    if not cleaned_pages:
        raise MetadataExtractionError(
            message="Cannot extract metadata from empty page list.",
            error_code="empty_pages",
            details={"document_id": document_id},
        )

    start_ts = time.perf_counter()
    logger.info(
        "Metadata extraction started",
        extra={
            "extra_data": {
                "document_id": document_id,
                "total_pages": len(cleaned_pages),
                "llm_fallback_enabled": enable_llm_fallback,
            }
        },
    )

    pages_text = _build_pages_text(cleaned_pages)

    # --- D6: Rule-based extraction ---
    try:
        metadata = _run_rules(document_id, pages_text)
    except Exception as exc:
        raise MetadataExtractionError(
            message=f"Rule-based metadata extraction failed: {exc}",
            error_code="rule_extraction_failed",
            details={"document_id": document_id, "error": str(exc)},
        ) from exc

    # Count extracted fields
    extracted_count = sum(
        1 for n in _METADATA_FIELD_NAMES if getattr(metadata, n).value is not None
    )

    logger.info(
        "Rule-based extraction complete",
        extra={
            "extra_data": {
                "document_id": document_id,
                "fields_extracted": extracted_count,
                "fields_total": len(_METADATA_FIELD_NAMES),
            }
        },
    )

    # --- D7: LLM fallback ---
    if enable_llm_fallback and openrouter_api_key:
        missing = _missing_fields(metadata)
        if missing:
            full_text = _full_text(pages_text)
            metadata = _run_llm_fallback(
                metadata,
                full_text,
                missing,
                document_id,
                openrouter_api_key,
                llm_model,
            )
        else:
            logger.debug(
                "All fields extracted by rules; skipping LLM fallback",
                extra={"extra_data": {"document_id": document_id}},
            )
    elif enable_llm_fallback:
        logger.warning(
            "LLM fallback enabled but OPENROUTER_API_KEY is missing",
            extra={"extra_data": {"document_id": document_id}},
        )

    elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
    logger.info(
        "Metadata extraction complete",
        extra={
            "extra_data": {
                "document_id": document_id,
                "duration_ms": elapsed_ms,
            }
        },
    )

    return metadata

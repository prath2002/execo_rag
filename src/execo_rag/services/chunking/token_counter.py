"""Token counter utility for embedding-model-compatible tokenization.

Primary: tiktoken (exact token counts for supported tokenizer mappings).
Fallback: character-based approximation (~4 chars/token) when tiktoken
          is not installed or the model encoding is unavailable.
"""

from __future__ import annotations

import logging

from execo_rag.config.constants import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Characters per token approximation (conservative estimate for legal text)
_CHARS_PER_TOKEN = 4

# Cache for tiktoken encoders so we don't re-load on every call
_ENCODER_CACHE: dict[str, object] = {}

# Default encoding used for general-purpose token counting fallbacks
_DEFAULT_ENCODING = "cl100k_base"


def _get_encoder(model_name: str) -> object | None:
    """Return a cached tiktoken encoder for the given model.

    Args:
        model_name: Embedding model name (e.g. ``'sentence-transformers/all-MiniLM-L6-v2'``).

    Returns:
        tiktoken Encoding object or ``None`` if tiktoken is unavailable.
    """
    if model_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[model_name]

    try:
        import tiktoken  # type: ignore[import-untyped]

        try:
            encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Unknown model: fall back to cl100k_base for broad compatibility.
            logger.debug(
                "Unknown tiktoken model; using cl100k_base",
                extra={"extra_data": {"model": model_name}},
            )
            encoder = tiktoken.get_encoding(_DEFAULT_ENCODING)

        _ENCODER_CACHE[model_name] = encoder
        return encoder

    except ImportError:
        logger.debug(
            "tiktoken not installed; using character-based approximation",
            extra={"extra_data": {"model": model_name}},
        )
        _ENCODER_CACHE[model_name] = None
        return None


def count_tokens(text: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """Count the number of tokens in *text* for the given embedding model.

    Args:
        text: Text string to tokenize.
        model_name: Embedding model name used to select the tokenizer.

    Returns:
        Integer token count. Always >= 0.
    """
    if not text:
        return 0

    encoder = _get_encoder(model_name)

    if encoder is not None:
        return len(encoder.encode(text))  # type: ignore[union-attr]

    # Fallback: character-based approximation
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_tokens(text: str) -> int:
    """Fast character-based token estimate (no external dependency).

    Use when exact counts are not critical (e.g. rough sizing before chunking).

    Args:
        text: Text string.

    Returns:
        Approximate token count.
    """
    return max(0, len(text) // _CHARS_PER_TOKEN)


def tokens_fit(
    text: str,
    max_tokens: int,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> bool:
    """Return True if *text* fits within *max_tokens* for the given model.

    Args:
        text: Text string.
        max_tokens: Maximum allowed token count.
        model_name: Embedding model name.

    Returns:
        ``True`` if ``count_tokens(text) <= max_tokens``.
    """
    return count_tokens(text, model_name) <= max_tokens


def split_to_token_budget(
    text: str,
    max_tokens: int,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[str, str]:
    """Split *text* into two parts: within the token budget and the remainder.

    Splits on sentence/word boundaries to avoid cutting mid-word.

    Args:
        text: Full text to split.
        max_tokens: Maximum tokens allowed in the first part.
        model_name: Embedding model name.

    Returns:
        Tuple of (within_budget_text, overflow_text).
    """
    if not text:
        return "", ""

    if count_tokens(text, model_name) <= max_tokens:
        return text, ""

    # Binary search for the split point (in characters)
    lo, hi = 0, len(text)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if count_tokens(text[:mid], model_name) <= max_tokens:
            lo = mid
        else:
            hi = mid

    # Snap back to a word boundary
    split_pos = lo
    while split_pos > 0 and not text[split_pos - 1].isspace():
        split_pos -= 1

    if split_pos == 0:
        split_pos = lo  # no word boundary found; hard split

    return text[:split_pos].rstrip(), text[split_pos:].lstrip()

"""Minimal OpenRouter client with familiar nested namespaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_HEADERS = {
    "HTTP-Referer": "execo-rag",
    "X-Title": "execo-rag",
}


@dataclass(slots=True)
class ChatCompletionMessage:
    """Chat completion message payload."""

    role: str | None = None
    content: str | None = None


@dataclass(slots=True)
class ChatCompletionChoice:
    """Single chat completion choice."""

    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None


@dataclass(slots=True)
class ChatCompletionResponse:
    """Parsed chat completion response."""

    id: str | None
    model: str | None
    choices: list[ChatCompletionChoice]
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class EmbeddingItem:
    """Single embedding vector response item."""

    index: int
    embedding: list[float]


@dataclass(slots=True)
class EmbeddingResponse:
    """Parsed embeddings response."""

    data: list[EmbeddingItem]
    model: str | None
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


class _ChatCompletionsAPI:
    """Namespace for chat completion calls."""

    def __init__(self, client: "OpenRouterClient") -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        **extra: Any,
    ) -> ChatCompletionResponse:
        """Create a chat completion."""

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        payload.update(extra)

        body = self._client._post("/chat/completions", payload)
        choices = [
            ChatCompletionChoice(
                index=choice.get("index", idx),
                message=ChatCompletionMessage(
                    role=choice.get("message", {}).get("role"),
                    content=choice.get("message", {}).get("content"),
                ),
                finish_reason=choice.get("finish_reason"),
            )
            for idx, choice in enumerate(body.get("choices", []))
        ]
        return ChatCompletionResponse(
            id=body.get("id"),
            model=body.get("model"),
            choices=choices,
            usage=body.get("usage"),
            raw=body,
        )


class _ChatAPI:
    """Namespace for chat APIs."""

    def __init__(self, client: "OpenRouterClient") -> None:
        self.completions = _ChatCompletionsAPI(client)


class _EmbeddingsAPI:
    """Namespace for embedding calls."""

    def __init__(self, client: "OpenRouterClient") -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: list[str] | str,
        **extra: Any,
    ) -> EmbeddingResponse:
        """Create embedding vectors."""

        payload: dict[str, Any] = {
            "model": model,
            "input": input,
        }
        payload.update(extra)

        body = self._client._post("/embeddings", payload)
        data = [
            EmbeddingItem(
                index=item.get("index", idx),
                embedding=item.get("embedding", []),
            )
            for idx, item in enumerate(body.get("data", []))
        ]
        return EmbeddingResponse(
            data=data,
            model=body.get("model"),
            usage=body.get("usage"),
            raw=body,
        )


class OpenRouterClient:
    """Lightweight OpenRouter client with familiar nested APIs."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: float = 60.0,
        referer: str = DEFAULT_OPENROUTER_HEADERS["HTTP-Referer"],
        title: str = DEFAULT_OPENROUTER_HEADERS["X-Title"],
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": referer,
                "X-Title": title,
            },
        )
        self.chat = _ChatAPI(self)
        self.embeddings = _EmbeddingsAPI(self)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a POST request and decode the JSON response."""

        response = self._http.post(path, json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("OpenRouter response must be a JSON object.")
        return data

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._http.close()

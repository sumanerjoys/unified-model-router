"""OpenAI-compatible adapter.

The unified schema is modeled on OpenAI's, so this adapter is a near
pass-through. It still owns the vendor-specific details: how to shape the
request body and how to parse OpenAI-style SSE chunks into unified chunks.
Groq, Together, OpenRouter, etc. are all OpenAI-compatible and reuse this.
"""

from __future__ import annotations

import json
from typing import Any

from app.adapters.base import Adapter
from app.models.unified import ChatCompletionChunk, Delta, StreamChoice


class OpenAIAdapter(Adapter):
    """Adapter for OpenAI-compatible providers."""

    name = "openai"

    def translate_request(self, unified: dict[str, Any]) -> dict[str, Any]:
        """OpenAI accepts the unified shape directly; forward known fields only."""
        body: dict[str, Any] = {
            "model": unified["model"],
            "messages": unified["messages"],
            "stream": unified.get("stream", False),
        }
        if unified.get("temperature") is not None:
            body["temperature"] = unified["temperature"]
        if unified.get("max_tokens") is not None:
            body["max_tokens"] = unified["max_tokens"]
        return body

    def parse_chunk(self, raw_data: str, model: str) -> ChatCompletionChunk | None:
        """Parse an OpenAI SSE chunk into a unified chunk."""
        if self.is_done(raw_data):
            return None
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            return None

        choices_out: list[StreamChoice] = []
        for choice in payload.get("choices", []):
            delta = choice.get("delta", {}) or {}
            choices_out.append(
                StreamChoice(
                    index=choice.get("index", 0),
                    delta=Delta(
                        role=delta.get("role"),
                        content=delta.get("content"),
                        reasoning_content=delta.get("reasoning_content"),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            )
        # Chunks with an empty choices array (e.g. the trailing usage-only chunk
        # some providers emit) carry no client-visible delta -> skip them.
        if not choices_out:
            return None

        return ChatCompletionChunk(
            id=payload.get("id") or f"chatcmpl-{model}",
            model=model,
            choices=choices_out,
        )

"""Mock-vendor adapter with a DELIBERATELY different schema.

This exists to prove the Adapter pattern does real translation work (not just
pass-through). The mock vendor intentionally differs from the unified/OpenAI
shape in several ways, mirroring the kind of differences a real non-OpenAI
provider (e.g. Anthropic) imposes:

  - The system prompt is a separate top-level `system` field, NOT a message.
  - Messages use `sender` + `text` instead of `role` + `content`.
  - `max_tokens` is named `max_output_tokens`.
  - Streaming events wrap the token as `{"type": "token", "text": "..."}` and
    signal completion with `{"type": "end", "stop": "..."}`.

The adapter maps all of this to/from the unified schema.
"""

from __future__ import annotations

import json
from typing import Any

from app.adapters.base import Adapter
from app.models.unified import ChatCompletionChunk, Delta, StreamChoice


class MockAdapter(Adapter):
    """Adapter for the local mock provider (non-OpenAI-shaped)."""

    name = "mock"

    def translate_request(self, unified: dict[str, Any]) -> dict[str, Any]:
        """Split out the system prompt and rename message/token fields."""
        system_parts: list[str] = []
        turns: list[dict[str, str]] = []
        for msg in unified["messages"]:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                turns.append({"sender": msg["role"], "text": msg["content"]})

        body: dict[str, Any] = {
            "model": unified["model"],
            "system": "\n".join(system_parts),
            "turns": turns,
            "stream": unified.get("stream", False),
        }
        if unified.get("temperature") is not None:
            body["temperature"] = unified["temperature"]
        if unified.get("max_tokens") is not None:
            body["max_output_tokens"] = unified["max_tokens"]
        return body

    def parse_chunk(self, raw_data: str, model: str) -> ChatCompletionChunk | None:
        """Translate the mock vendor's event shape into a unified chunk."""
        if self.is_done(raw_data):
            return None
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            return None

        event_type = payload.get("type")
        if event_type == "token":
            return ChatCompletionChunk(
                model=model,
                choices=[StreamChoice(delta=Delta(content=payload.get("text", "")))],
            )
        if event_type == "end":
            return ChatCompletionChunk(
                model=model,
                choices=[
                    StreamChoice(delta=Delta(), finish_reason=payload.get("stop", "stop"))
                ],
            )
        return None

    def chat_completions_path(self) -> str:
        """The mock provider exposes generation under /generate."""
        return "/generate"

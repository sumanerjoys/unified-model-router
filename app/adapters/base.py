"""Adapter contract — PURE schema translation, zero I/O.

An adapter converts:
  - a unified request  -> a vendor-native request body, and
  - a vendor SSE chunk -> a unified ChatCompletionChunk.

Adapters must not perform any network calls or hold mutable state. Keeping them
pure means every vendor's translation quirks are unit-testable with zero mocking
(dict in -> dict out). Transport (HTTP, streaming, timeouts) lives in the
ProviderClient layer, not here.
"""

from __future__ import annotations

import abc
from typing import Any

from app.models.unified import ChatCompletionChunk


class Adapter(abc.ABC):
    """Abstract base for a vendor adapter."""

    #: Stable adapter name (used in logs and the registry).
    name: str = "base"

    @abc.abstractmethod
    def translate_request(self, unified: dict[str, Any]) -> dict[str, Any]:
        """Convert a unified request dict into the vendor's request body.

        Args:
            unified: The unified request as a plain dict (already validated).

        Returns:
            The vendor-native request body to send upstream.
        """

    @abc.abstractmethod
    def parse_chunk(self, raw_data: str, model: str) -> ChatCompletionChunk | None:
        """Convert one vendor SSE `data:` payload into a unified chunk.

        Args:
            raw_data: The content after the `data: ` prefix from the upstream
                SSE stream (not including the prefix or trailing newlines).
            model: The unified model name to stamp onto the emitted chunk.

        Returns:
            A unified ChatCompletionChunk, or None if this payload carries no
            client-visible delta (e.g. a keep-alive or an unrecognized event).
        """

    @staticmethod
    def is_done(raw_data: str) -> bool:
        """Return True if this SSE payload is the terminal sentinel."""
        return raw_data.strip() == "[DONE]"

    def chat_completions_path(self) -> str:
        """Relative path for the chat completions endpoint on this vendor."""
        return "/chat/completions"

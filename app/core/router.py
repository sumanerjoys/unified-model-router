"""Router — provider selection and fallback orchestration.

The Router iterates the provider chain and enforces the fallback policy:

  - Fallback is only attempted for retryable errors (TRANSIENT / TIMEOUT).
  - Fallback is only safe BEFORE the first chunk is flushed to the client
    (the "point of no return"). Once any chunk has been yielded, a mid-stream
    failure terminates the stream instead of silently switching providers.
  - Attempts are bounded by a hop budget AND an overall wall-clock deadline so
    a single outage cannot cause a retry storm or unbounded latency.

The Router speaks only in unified chunks; it knows nothing about HTTP or vendor
payloads (that is the ProviderClient + Adapter concern).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app.adapters.registry import Provider
from app.config import Settings
from app.core.errors import ErrorClass, ProviderError
from app.core.provider_client import ProviderClient
from app.models.unified import ChatCompletionChunk

logger = logging.getLogger("router")


class Router:
    """Orchestrates provider attempts with bounded, silent fallback."""

    def __init__(
        self,
        provider_client: ProviderClient,
        chain: list[Provider],
        settings: Settings,
    ) -> None:
        self._client = provider_client
        self._chain = chain
        self._settings = settings

    async def stream(
        self,
        unified_request: dict[str, Any],
        model: str,
        request_id: str,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Yield unified chunks, falling back across providers as needed.

        Args:
            unified_request: The validated unified request as a dict.
            model: Unified model name to stamp on emitted chunks.
            request_id: Correlation id for structured logging.

        Yields:
            Unified ChatCompletionChunk objects.

        Raises:
            ProviderError: if all eligible providers fail before any chunk is
                emitted, the last error is raised for the API layer to surface.
        """
        deadline = time.monotonic() + self._settings.request_deadline_seconds
        max_attempts = min(self._settings.max_fallback_hops + 1, len(self._chain))
        first_chunk_sent = False
        last_error: ProviderError | None = None

        for attempt, provider in enumerate(self._chain[:max_attempts]):
            if time.monotonic() >= deadline:
                logger.warning(
                    "deadline exceeded before attempt",
                    extra={"request_id": request_id, "attempt": attempt},
                )
                break

            logger.info(
                "provider attempt",
                extra={
                    "request_id": request_id,
                    "attempt": attempt,
                    "provider": provider.name,
                },
            )
            try:
                async for chunk in self._client.stream_chat(
                    provider, unified_request, model
                ):
                    first_chunk_sent = True
                    yield chunk
                # Stream finished cleanly.
                logger.info(
                    "provider success",
                    extra={"request_id": request_id, "provider": provider.name},
                )
                return
            except ProviderError as exc:
                last_error = exc
                logger.warning(
                    "provider failed",
                    extra={
                        "request_id": request_id,
                        "provider": provider.name,
                        "error_class": exc.error_class.value,
                        "status_code": exc.status_code,
                    },
                )
                # Point of no return: cannot silently switch once bytes are out.
                if first_chunk_sent:
                    logger.error(
                        "mid-stream failure after first chunk; terminating",
                        extra={"request_id": request_id, "provider": provider.name},
                    )
                    raise
                # Non-retryable errors surface immediately.
                if not exc.is_retryable:
                    raise
                # Otherwise loop continues to the next provider (silent fallback).

        # Exhausted the chain (or deadline) with only retryable failures.
        if last_error is not None:
            raise last_error
        raise ProviderError(  # pragma: no cover - only if chain is empty
            ErrorClass.FATAL,
            "router",
            message="no providers configured",
        )

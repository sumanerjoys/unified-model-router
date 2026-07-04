"""ProviderClient — the transport layer.

Owns everything about *talking* to an upstream provider:
  - issuing the request via the shared httpx.AsyncClient,
  - opening a streaming connection and yielding unified chunks as they arrive
    (never buffering the full body -> a "water pipe, not a bucket"),
  - applying timeouts, and
  - classifying failures into the error taxonomy.

Adapters (pure translation) are invoked here but perform no I/O themselves. The
critical resilience detail lives in `stream_chat`: a try/finally guarantees the
upstream response is closed when the client disconnects (CancelledError) or the
generator is otherwise torn down, preventing socket/token leaks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.adapters.registry import Provider
from app.core.errors import (
    ProviderError,
    classify_exception,
    classify_status,
    parse_retry_after,
)
from app.models.unified import ChatCompletionChunk


class ProviderClient:
    """Transport wrapper around a shared httpx.AsyncClient for one call."""

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    def _headers(self, provider: Provider) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    async def stream_chat(
        self,
        provider: Provider,
        unified_request: dict[str, Any],
        model: str,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Open a streaming request to `provider` and yield unified chunks.

        Raises:
            ProviderError: if the connection fails or the upstream returns a
                non-2xx status *before* streaming begins. Errors are classified
                so the Router can decide whether to fall back.
        """
        body = provider.adapter.translate_request(unified_request)
        url = f"{provider.base_url}{provider.adapter.chat_completions_path()}"

        try:
            async with self._http.stream(
                "POST", url, json=body, headers=self._headers(provider)
            ) as response:
                if response.status_code >= 400:
                    # Drain the (small) error body so classification can use it,
                    # then raise before any client-visible chunk is emitted.
                    await response.aread()
                    raise ProviderError(
                        classify_status(response.status_code),
                        provider.name,
                        status_code=response.status_code,
                        retry_after=parse_retry_after(response.headers),
                        message=f"{provider.name} returned {response.status_code}",
                    )

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if provider.adapter.is_done(raw):
                        return
                    chunk = provider.adapter.parse_chunk(raw, model)
                    if chunk is not None:
                        yield chunk
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified and re-raised
            raise ProviderError(
                classify_exception(exc),
                provider.name,
                message=f"{provider.name} transport error: {exc!r}",
            ) from exc

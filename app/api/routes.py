"""API layer — the HTTP surface for the gateway.

Defines POST /v1/chat/completions. Responsibilities kept here (and nowhere else):
validation, request-id assignment, building the Router, and returning either a
streaming SSE response or an aggregated non-streaming JSON body. Routing and
transport concerns live in the core layer.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.adapters.registry import build_provider_chain
from app.config import get_settings
from app.core.errors import ProviderError
from app.core.provider_client import ProviderClient
from app.core.router import Router
from app.models.unified import (
    SSE_DONE,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ResponseMessage,
)

logger = logging.getLogger("api")

router = APIRouter()


def get_router(request: Request) -> Router:
    """Provide a Router bound to the shared HTTP client and provider chain.

    This is a FastAPI dependency so tests can override it via
    `app.dependency_overrides[get_router]` to inject failure-injecting or
    fully in-process providers without patching module internals.
    """
    settings = get_settings()
    provider_client = ProviderClient(request.app.state.http_client)
    chain = build_provider_chain(settings)
    return Router(provider_client, chain, settings)


@router.post("/v1/chat/completions", tags=["inference"], response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    x_request_id: str | None = Header(default=None),
    app_router: Router = Depends(get_router),
) -> StreamingResponse | JSONResponse:
    """Unified chat completions endpoint with SSE streaming and fallback."""
    request_id = x_request_id or f"req-{uuid.uuid4().hex}"
    unified = body.model_dump()
    model = body.model
    headers = {"X-Request-ID": request_id}

    if body.stream:

        async def sse() -> AsyncIterator[bytes]:
            try:
                async for chunk in app_router.stream(unified, model, request_id):
                    yield chunk.to_sse().encode()
                yield SSE_DONE.encode()
            except ProviderError as exc:
                # All providers failed before first byte -> emit an SSE error event.
                logger.error(
                    "all providers failed",
                    extra={"request_id": request_id, "error": exc.message},
                )
                err_payload = json.dumps(
                    {"error": {"type": "upstream_error", "message": exc.message}}
                )
                yield f"data: {err_payload}\n\n".encode()
                yield SSE_DONE.encode()

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={
                **headers,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: aggregate upstream chunks into one body. We still request a
    # stream upstream (single transport code path) and collect it server-side.
    unified["stream"] = True
    try:
        parts: list[str] = []
        finish_reason = "stop"
        async for chunk in app_router.stream(unified, model, request_id):
            for choice in chunk.choices:
                if choice.delta.content:
                    parts.append(choice.delta.content)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
    except ProviderError as exc:
        return JSONResponse(
            {"error": {"type": "upstream_error", "message": exc.message}},
            status_code=502,
            headers=headers,
        )

    response = ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                message=ResponseMessage(content="".join(parts)),
                finish_reason=finish_reason,
            )
        ],
    )
    return JSONResponse(response.model_dump(), headers=headers)

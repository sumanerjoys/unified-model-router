"""Local mock LLM provider.

A standalone FastAPI app that speaks the mock vendor's (non-OpenAI) schema so we
can demonstrate real adapter translation AND deterministic, controllable
behavior for the fallback demo:

  - It exposes POST /v1/generate (the path MockAdapter targets).
  - It streams a fixed, deterministic response as `{"type":"token"}` events
    followed by an `{"type":"end"}` event (SSE).
  - Failure modes can be forced via the `fail` query param or FORCE_FAIL env
    (`429`, `503`, `timeout`) so we can trigger the gateway's fallback on demand.

Run it on its own port, e.g.:
    uvicorn mock_provider.server:app --port 9100
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Mock LLM Provider")

DEFAULT_TOKENS = ["Hello", " from", " the", " mock", " provider", "!"]


def _forced_failure(request: Request) -> str | None:
    """Return a forced failure mode from query param or env, if any."""
    return request.query_params.get("fail") or os.getenv("FORCE_FAIL") or None


@app.post("/v1/generate", response_model=None)
async def generate(request: Request) -> StreamingResponse | JSONResponse:
    """Mock generation endpoint using the non-OpenAI vendor schema."""
    fail = _forced_failure(request)
    if fail in {"429", "503", "500", "502", "504"}:
        return JSONResponse(
            {"error": {"type": "rate_limit", "message": f"forced {fail}"}},
            status_code=int(fail),
            headers={"Retry-After": "1"} if fail == "429" else None,
        )
    if fail == "timeout":
        await asyncio.sleep(30)  # exceed the gateway read timeout

    body = await request.json()
    tokens = DEFAULT_TOKENS

    async def event_stream() -> AsyncIterator[bytes]:
        for tok in tokens:
            yield f"data: {json.dumps({'type': 'token', 'text': tok})}\n\n".encode()
            await asyncio.sleep(0.02)
        yield f"data: {json.dumps({'type': 'end', 'stop': 'stop'})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    if not body.get("stream", False):
        text = "".join(tokens)
        return JSONResponse({"output": text, "stop": "stop"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness for the mock provider."""
    return {"status": "ok"}

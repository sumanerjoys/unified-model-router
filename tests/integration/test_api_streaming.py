"""Integration tests for the HTTP surface (POST /v1/chat/completions).

Uses the real FastAPI app with a dependency override on `get_router` so the
production request path is exercised end to end (validation, SSE framing,
headers, [DONE] sentinel), while upstream stays in-process and deterministic.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from app.api.routes import get_router
from app.core.router import Router
from app.main import create_app

EXPECTED = "Hello from the mock provider!"


def _reassemble(sse_body: str) -> str:
    content = ""
    for line in sse_body.splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            payload = json.loads(line[len("data:"):].strip())
            for choice in payload.get("choices", []):
                content += choice.get("delta", {}).get("content") or ""
    return content


@pytest.fixture
def client_factory(
    make_router: Callable[..., Router],
) -> Callable[[dict[str, str | None]], httpx.AsyncClient]:
    """Build an ASGI test client whose Router is overridden per provider modes."""

    def _factory(provider_modes: dict[str, str | None]) -> httpx.AsyncClient:
        app = create_app()
        app.dependency_overrides[get_router] = lambda: make_router(provider_modes)
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gw"
        )

    return _factory


async def test_streaming_happy_path(client_factory) -> None:
    async with client_factory({"primary": None}) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gpt-x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            assert "x-request-id" in resp.headers
            body = "".join([line + "\n" async for line in resp.aiter_lines()])
    assert "[DONE]" in body
    assert _reassemble(body) == EXPECTED


async def test_streaming_silent_fallback_on_429(client_factory) -> None:
    async with client_factory({"primary": "429", "fallback": None}) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gpt-x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            # Client sees a normal 200 stream; the upstream 429 is invisible.
            assert resp.status_code == 200
            body = "".join([line + "\n" async for line in resp.aiter_lines()])
    assert "error" not in body
    assert _reassemble(body) == EXPECTED


async def test_streaming_all_fail_emits_error_event(client_factory) -> None:
    async with client_factory({"primary": "503", "fallback": "503"}) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gpt-x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200  # stream already opened
            body = "".join([line + "\n" async for line in resp.aiter_lines()])
    assert "upstream_error" in body
    assert "[DONE]" in body


async def test_non_streaming_aggregates_response(client_factory) -> None:
    async with client_factory({"primary": None}) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == EXPECTED
    assert data["object"] == "chat.completion"


async def test_non_streaming_all_fail_returns_502(client_factory) -> None:
    async with client_factory({"primary": "503", "fallback": "503"}) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_error"


async def test_invalid_request_is_rejected_422(client_factory) -> None:
    async with client_factory({"primary": None}) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-x", "messages": []},  # empty messages -> invalid
        )
    assert resp.status_code == 422


async def test_client_disconnect_closes_upstream(
    make_router: Callable[..., Router],
) -> None:
    """Dropping the consumer mid-stream must not leak the upstream connection.

    We consume one chunk from the Router's async generator, then close it early
    (simulating a client disconnect / cancellation). The generator's finally
    block must run and close the upstream response without raising.
    """
    router = make_router({"primary": None})
    unified = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    gen: AsyncIterator = router.stream(unified, "test-model", "disc")
    first = await gen.__anext__()
    assert first.choices[0].delta.content  # got a real chunk

    # Early close simulates the client going away; must clean up quietly.
    await gen.aclose()

"""Shared test fixtures and helpers.

The central idea: upstream providers are served by the in-process mock provider
ASGI app via httpx's ASGITransport, so tests are fully hermetic (no sockets, no
real LLM, deterministic). Failure injection is done with a thin transport
wrapper that appends `?fail=<mode>` to every upstream request, which the mock
provider honors. Nothing here patches application module internals.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from app.adapters.mock import MockAdapter
from app.adapters.registry import Provider
from app.config import Settings
from app.core.provider_client import ProviderClient
from app.core.router import Router
from mock_provider.server import app as mock_app


class _FailInjectingASGITransport(httpx.ASGITransport):
    """ASGITransport that forces the mock provider into a failure mode.

    The special mode ``"timeout"`` raises ``httpx.ReadTimeout`` directly, which
    is exactly what a real socket read timeout surfaces as. This is used instead
    of a wall-clock sleep because ASGITransport does not enforce httpx timeouts
    (there is no real socket in-process), so a sleep would stall tests without
    actually exercising the timeout path.
    """

    def __init__(self, app: object, fail: str) -> None:
        super().__init__(app=app)
        self._fail = fail

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._fail == "timeout":
            raise httpx.ReadTimeout("simulated read timeout", request=request)
        request.url = request.url.copy_merge_params({"fail": self._fail})
        return await super().handle_async_request(request)


def make_upstream_client(fail: str | None = None) -> httpx.AsyncClient:
    """Build an httpx client whose requests are served by the mock provider.

    Args:
        fail: Optional forced failure mode ('429', '503', 'timeout', ...).

    Returns:
        An AsyncClient wired to the in-process mock provider ASGI app.
    """
    transport: httpx.ASGITransport = (
        _FailInjectingASGITransport(mock_app, fail)
        if fail is not None
        else httpx.ASGITransport(app=mock_app)
    )
    return httpx.AsyncClient(transport=transport, base_url="http://mock")


class _PerProviderClient(ProviderClient):
    """A ProviderClient that selects a distinct upstream client per provider name.

    Lets a single Router attempt use, e.g., a failing 'primary' client and a
    healthy 'fallback' client — exactly the shape needed to test fallback.
    """

    def __init__(self, clients_by_provider: dict[str, httpx.AsyncClient]) -> None:
        self._clients = clients_by_provider

    async def stream_chat(self, provider, unified_request, model):  # type: ignore[override]
        self._http = self._clients[provider.name]
        async for chunk in ProviderClient.stream_chat(
            self, provider, unified_request, model
        ):
            yield chunk


@pytest.fixture
def settings() -> Settings:
    """Fast, deterministic settings for tests."""
    return Settings(
        max_fallback_hops=2,
        request_deadline_seconds=10,
        upstream_connect_timeout=2,
        upstream_read_timeout=2,
    )


@pytest.fixture
def make_router(settings: Settings) -> Callable[..., Router]:
    """Factory that builds a Router over per-provider in-process clients.

    Usage:
        router = make_router({"primary": "429", "fallback": None})
    where each value is the failure mode for that provider (None = healthy).
    """
    created: list[httpx.AsyncClient] = []

    def _factory(provider_modes: dict[str, str | None]) -> Router:
        clients: dict[str, httpx.AsyncClient] = {}
        chain: list[Provider] = []
        for name, mode in provider_modes.items():
            client = make_upstream_client(mode)
            created.append(client)
            clients[name] = client
            chain.append(Provider(name, MockAdapter(), "http://mock/v1", "k"))
        return Router(_PerProviderClient(clients), chain, settings)

    return _factory


async def collect_content(router: Router, request_id: str = "test") -> str:
    """Drive a Router stream and concatenate all delta content."""
    unified = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    parts: list[str] = []
    async for chunk in router.stream(unified, "test-model", request_id):
        for choice in chunk.choices:
            if choice.delta.content:
                parts.append(choice.delta.content)
    return "".join(parts)

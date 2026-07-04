"""Integration tests for the Router's fallback behavior.

Upstream is the in-process mock provider (via ASGITransport). Failures are
injected through the fixture without patching application internals.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.core.errors import ErrorClass, ProviderError
from app.core.router import Router
from tests.conftest import collect_content

EXPECTED = "Hello from the mock provider!"


async def test_happy_path_streams_full_content(
    make_router: Callable[..., Router],
) -> None:
    router = make_router({"primary": None})
    assert await collect_content(router) == EXPECTED


async def test_primary_429_silently_falls_back(
    make_router: Callable[..., Router],
) -> None:
    router = make_router({"primary": "429", "fallback": None})
    # Client receives the complete answer with no error surfaced.
    assert await collect_content(router) == EXPECTED


async def test_primary_503_silently_falls_back(
    make_router: Callable[..., Router],
) -> None:
    router = make_router({"primary": "503", "fallback": None})
    assert await collect_content(router) == EXPECTED


async def test_all_providers_fail_surfaces_transient_error(
    make_router: Callable[..., Router],
) -> None:
    router = make_router({"primary": "503", "fallback": "503"})
    with pytest.raises(ProviderError) as excinfo:
        await collect_content(router)
    assert excinfo.value.error_class is ErrorClass.TRANSIENT


async def test_fatal_error_does_not_fall_back(
    make_router: Callable[..., Router],
) -> None:
    # 404 is FATAL; the fallback must NOT be attempted (would mask a real bug).
    router = make_router({"primary": "404", "fallback": None})
    with pytest.raises(ProviderError) as excinfo:
        await collect_content(router)
    assert excinfo.value.error_class is ErrorClass.FATAL


async def test_hop_budget_is_bounded(
    make_router: Callable[..., Router],
) -> None:
    # With 3 failing providers but max_fallback_hops=2 (=> 3 attempts allowed),
    # exhaustion still surfaces a transient error rather than looping forever.
    router = make_router(
        {"primary": "503", "backup1": "503", "backup2": "503"}
    )
    with pytest.raises(ProviderError):
        await collect_content(router)


async def test_timeout_triggers_fallback(
    make_router: Callable[..., Router],
) -> None:
    # Primary hangs beyond the read timeout -> TIMEOUT -> fall back to healthy.
    router = make_router({"primary": "timeout", "fallback": None})
    assert await collect_content(router) == EXPECTED

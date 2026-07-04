"""Tests for the API Authorization Guard extension.

Covers the guard both as a unit (the dependency in isolation) and end-to-end
through the app, including that it rejects *before* any upstream call.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from fastapi import HTTPException

from app.api import auth
from app.api.auth import require_api_key
from app.api.routes import get_router
from app.config import Settings
from app.core.router import Router
from app.main import create_app


def _patch_keys(monkeypatch, keys: str) -> None:
    """Point the guard's settings lookup at a controlled key list."""
    monkeypatch.setattr(auth, "get_settings", lambda: Settings(gateway_api_keys=keys))


# --- Unit: the dependency in isolation ---


async def test_guard_disabled_when_no_keys_configured(monkeypatch) -> None:
    _patch_keys(monkeypatch, "")
    # Should not raise even with no credentials.
    await require_api_key(authorization=None, x_api_key=None)


async def test_guard_accepts_valid_bearer(monkeypatch) -> None:
    _patch_keys(monkeypatch, "k1,k2")
    await require_api_key(authorization="Bearer k1", x_api_key=None)
    await require_api_key(authorization=None, x_api_key="k2")


async def test_guard_rejects_invalid_key(monkeypatch) -> None:
    _patch_keys(monkeypatch, "k1")
    with pytest.raises(HTTPException) as exc:
        await require_api_key(authorization="Bearer wrong", x_api_key=None)
    assert exc.value.status_code == 401


async def test_guard_rejects_missing_key(monkeypatch) -> None:
    _patch_keys(monkeypatch, "k1")
    with pytest.raises(HTTPException) as exc:
        await require_api_key(authorization=None, x_api_key=None)
    assert exc.value.status_code == 401


# --- Integration: through the app, guard runs before upstream ---


@pytest.fixture
def guarded_client(
    make_router: Callable[..., Router], monkeypatch
) -> httpx.AsyncClient:
    """App with the guard enabled (one allowed key) and a mock router."""
    _patch_keys(monkeypatch, "secret-123")
    app = create_app()
    app.dependency_overrides[get_router] = lambda: make_router({"primary": None})
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    )


async def test_request_without_key_is_rejected_401(guarded_client) -> None:
    async with guarded_client as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert resp.status_code == 401


async def test_request_with_valid_key_succeeds(guarded_client) -> None:
    async with guarded_client as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret-123"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert resp.status_code == 200

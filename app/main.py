"""FastAPI application factory and lifespan (shared HTTP client)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI

from app import __version__
from app.config import get_settings


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create a single shared AsyncClient reused across all requests.

    Using one pooled client (instead of one per request) avoids TCP/TLS
    handshake churn and is the main throughput lever for the gateway.
    """
    settings = get_settings()
    timeout = httpx.Timeout(
        connect=settings.upstream_connect_timeout,
        read=settings.upstream_read_timeout,
        write=settings.upstream_read_timeout,
        pool=settings.upstream_connect_timeout,
    )
    app.state.http_client = httpx.AsyncClient(timeout=timeout)
    try:
        yield
    finally:
        await app.state.http_client.aclose()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Unified Model Router",
        version=__version__,
        summary="OpenRouter-style LLM API gateway with SSE streaming and resilient fallback.",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["ops"])
    async def ready() -> dict[str, str]:
        """Readiness probe (shared client initialized)."""
        client_ready = getattr(app.state, "http_client", None) is not None
        return {"status": "ready" if client_ready else "starting"}

    return app


app = create_app()

"""Provider registry — maps a logical model to an ordered provider chain.

Each provider entry pairs an Adapter (how to translate) with the connection
target (where to send). The Router consumes the ordered chain to drive fallback:
it tries entry 0, then entry 1, and so on within the configured hop budget.

The ordering here is the seam where cost-aware routing (an extension) can later
reorder the chain cheapest-responsive-first, without touching the Router.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.adapters.base import Adapter
from app.adapters.mock import MockAdapter
from app.adapters.openai import OpenAIAdapter
from app.config import Settings


@dataclass(frozen=True)
class Provider:
    """A concrete upstream target: an adapter plus where/how to reach it."""

    name: str
    adapter: Adapter
    base_url: str
    api_key: str


def build_provider_chain(settings: Settings) -> list[Provider]:
    """Build the default ordered provider chain (primary first, then fallback).

    Args:
        settings: Application settings carrying provider URLs and keys.

    Returns:
        Ordered list of providers; the Router attempts them in order.
    """
    return [
        Provider(
            name="primary",
            adapter=OpenAIAdapter(),
            base_url=settings.primary_base_url.rstrip("/"),
            api_key=settings.primary_api_key,
        ),
        Provider(
            name="fallback",
            adapter=MockAdapter(),
            base_url=settings.fallback_base_url.rstrip("/"),
            api_key=settings.fallback_api_key,
        ),
    ]

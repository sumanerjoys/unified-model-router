"""Provider registry — maps a logical model to an ordered provider chain.

Each provider entry pairs an Adapter (how to translate) with the connection
target (where to send). The Router consumes the ordered chain to drive fallback:
it tries entry 0, then entry 1, and so on within the configured hop budget.

`build_provider_chain` is the seam for cost-aware routing (an extension): when
enabled, the chain is reordered cheapest-responsive-first before the Router sees
it, so no Router change is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.adapters.base import Adapter
from app.adapters.mock import MockAdapter
from app.adapters.openai import OpenAIAdapter
from app.config import Settings
from app.models.manifest import cost_for


@dataclass(frozen=True)
class Provider:
    """A concrete upstream target: an adapter plus where/how to reach it."""

    name: str
    adapter: Adapter
    base_url: str
    api_key: str
    model: str = ""
    cost_per_1k: float = 0.0


def order_by_cost(chain: list[Provider]) -> list[Provider]:
    """Return the chain sorted cheapest-first (stable for equal costs)."""
    return sorted(chain, key=lambda p: p.cost_per_1k)


def build_provider_chain(settings: Settings) -> list[Provider]:
    """Build the ordered provider chain.

    Providers are attempted in list order by the Router. When cost-aware routing
    is enabled (`settings.cost_aware_routing`), the chain is reordered
    cheapest-responsive-first via the per-model cost manifest, so the router
    prefers the cheapest variant before defaulting to pricier fallbacks.

    Args:
        settings: Application settings carrying provider URLs, keys, and flags.

    Returns:
        Ordered list of providers; the Router attempts them in order.
    """
    primary_model = settings.primary_model
    fallback_model = settings.fallback_model
    chain = [
        Provider(
            name="primary",
            adapter=OpenAIAdapter(),
            base_url=settings.primary_base_url.rstrip("/"),
            api_key=settings.primary_api_key,
            model=primary_model,
            cost_per_1k=cost_for(primary_model),
        ),
        Provider(
            name="fallback",
            adapter=MockAdapter(),
            base_url=settings.fallback_base_url.rstrip("/"),
            api_key=settings.fallback_api_key,
            model=fallback_model,
            cost_per_1k=cost_for(fallback_model),
        ),
    ]
    if settings.cost_aware_routing:
        return order_by_cost(chain)
    return chain

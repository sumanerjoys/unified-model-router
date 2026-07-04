"""Tests for cost-aware routing (extension)."""

from __future__ import annotations

from app.adapters.mock import MockAdapter
from app.adapters.registry import Provider, build_provider_chain, order_by_cost
from app.config import Settings
from app.models.manifest import DEFAULT_COST_PER_1K, cost_for


def _p(name: str, cost: float) -> Provider:
    return Provider(name, MockAdapter(), "http://x/v1", "k", model=name, cost_per_1k=cost)


class TestManifest:
    def test_known_model_cost(self) -> None:
        assert cost_for("openai-gpt-oss-120b") == 0.0006

    def test_unknown_model_uses_default(self) -> None:
        assert cost_for("some-unknown-model") == DEFAULT_COST_PER_1K


class TestOrderByCost:
    def test_orders_cheapest_first(self) -> None:
        chain = [_p("expensive", 0.02), _p("cheap", 0.0002), _p("mid", 0.005)]
        ordered = order_by_cost(chain)
        assert [p.name for p in ordered] == ["cheap", "mid", "expensive"]

    def test_stable_for_equal_costs(self) -> None:
        chain = [_p("a", 0.001), _p("b", 0.001)]
        assert [p.name for p in order_by_cost(chain)] == ["a", "b"]


class TestBuildChain:
    def test_default_order_when_disabled(self) -> None:
        # Primary is pricier than fallback, but without cost-aware routing the
        # declared order (primary first) is preserved.
        settings = Settings(
            cost_aware_routing=False,
            primary_model="openai-o3",  # expensive
            fallback_model="openai-gpt-oss-20b",  # cheap
        )
        chain = build_provider_chain(settings)
        assert [p.name for p in chain] == ["primary", "fallback"]

    def test_cost_aware_reorders_cheapest_first(self) -> None:
        settings = Settings(
            cost_aware_routing=True,
            primary_model="openai-o3",  # expensive
            fallback_model="openai-gpt-oss-20b",  # cheap
        )
        chain = build_provider_chain(settings)
        # Cheapest (the fallback's cheap model) should now be attempted first.
        assert chain[0].name == "fallback"
        assert chain[0].cost_per_1k < chain[1].cost_per_1k

    def test_fallback_still_present_after_reorder(self) -> None:
        settings = Settings(cost_aware_routing=True)
        chain = build_provider_chain(settings)
        assert len(chain) == 2
        assert {p.name for p in chain} == {"primary", "fallback"}

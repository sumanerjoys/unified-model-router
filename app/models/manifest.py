"""Cost manifest for cost-aware routing (extension).

Maps a model id to an approximate blended cost per 1K tokens (USD). Used to order
the provider chain cheapest-first, so the router prefers the cheapest responsive
variant before defaulting to pricier fallbacks. Values are indicative and can be
refreshed at runtime; unknown models fall back to a neutral default so they are
neither unfairly preferred nor penalized.
"""

from __future__ import annotations

# Approximate blended (input+output) USD per 1K tokens. Indicative values.
COST_PER_1K_TOKENS: dict[str, float] = {
    "openai-gpt-oss-20b": 0.0002,
    "openai-gpt-oss-120b": 0.0006,
    "openai-gpt-4o-mini": 0.0006,
    "openai-gpt-4o": 0.0075,
    "openai-o3-mini": 0.0044,
    "openai-o3": 0.02,
    "llama3.3-70b-instruct": 0.0006,
}

#: Used when a model is not present in the manifest.
DEFAULT_COST_PER_1K: float = 0.005


def cost_for(model: str) -> float:
    """Return the per-1K-token cost for a model, or the neutral default."""
    return COST_PER_1K_TOKENS.get(model, DEFAULT_COST_PER_1K)

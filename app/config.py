"""Application configuration loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseSettings):
    """Connection settings for a single upstream provider."""

    base_url: str
    api_key: str


class Settings(BaseSettings):
    """Gateway settings. Values come from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gateway auth (extension). Comma-separated allowed keys; empty disables the guard.
    gateway_api_keys: str = ""

    # Primary provider (real, OpenAI-compatible).
    primary_base_url: str = "https://api.openai.com/v1"
    primary_api_key: str = ""

    # Fallback provider (defaults to the local mock).
    fallback_base_url: str = "http://localhost:9100/v1"
    fallback_api_key: str = "mock-key"

    # Routing / resilience tunables.
    max_fallback_hops: int = Field(default=2, ge=1)
    request_deadline_seconds: float = Field(default=60.0, gt=0)
    upstream_connect_timeout: float = Field(default=5.0, gt=0)
    upstream_read_timeout: float = Field(default=60.0, gt=0)

    @property
    def allowed_gateway_keys(self) -> set[str]:
        """Parsed set of allowed gateway keys (empty set means the guard is off)."""
        return {k.strip() for k in self.gateway_api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

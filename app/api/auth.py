"""API Authorization Guard (extension).

A FastAPI dependency that validates the caller's gateway API key against the
configured allow-list *before* any upstream provider connection is initialized.
Invalid or missing credentials are rejected immediately with 401, so bad traffic
never triggers an upstream call.

The guard is opt-in: if no gateway keys are configured (`GATEWAY_API_KEYS` empty),
it is disabled and all traffic is allowed (useful for local/dev).
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import get_settings


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """Pull the presented key from either an Authorization or X-API-Key header."""
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        value = authorization.strip()
        if value.lower().startswith("bearer "):
            return value[len("bearer ") :].strip()
        return value
    return None


async def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject the request unless a valid gateway key is presented.

    No-op when the guard is disabled (no keys configured). Raises 401 otherwise
    if the presented key is missing or not in the allow-list.
    """
    allowed = get_settings().allowed_gateway_keys
    if not allowed:
        return  # guard disabled

    presented = _extract_key(authorization, x_api_key)
    if presented is None or presented not in allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

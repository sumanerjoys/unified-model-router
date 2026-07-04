"""Error taxonomy that drives routing decisions.

Upstream failures are classified into three classes. Only TRANSIENT and TIMEOUT
are retryable (they trigger a fallback to the next provider); FATAL errors are
surfaced to the client because retrying would not help and could mask a real
client-side bug.
"""

from __future__ import annotations

import enum

import httpx


class ErrorClass(enum.Enum):
    """Classification of an upstream failure."""

    TRANSIENT = "transient"  # 429, 502, 503, connection reset -> fall back
    TIMEOUT = "timeout"      # connect/read timeout -> fall back
    FATAL = "fatal"          # 400/401/403/404, malformed -> surface to client

    @property
    def is_retryable(self) -> bool:
        """Whether this error class should trigger a provider switch."""
        return self in (ErrorClass.TRANSIENT, ErrorClass.TIMEOUT)


#: Upstream status codes treated as transient (safe to fall back).
TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class ProviderError(Exception):
    """An upstream provider failure with a classification and context.

    Attributes:
        error_class: The taxonomy class driving retry/fallback behavior.
        provider: Name of the provider that failed.
        status_code: Upstream HTTP status, if any.
        retry_after: Parsed Retry-After seconds, if the upstream supplied it.
        message: Human-readable detail.
    """

    def __init__(
        self,
        error_class: ErrorClass,
        provider: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        message: str = "",
    ) -> None:
        self.error_class = error_class
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        self.message = message or f"{provider} failed ({error_class.value})"
        super().__init__(self.message)

    @property
    def is_retryable(self) -> bool:
        """Whether the failure should trigger a fallback."""
        return self.error_class.is_retryable


def classify_status(status_code: int) -> ErrorClass:
    """Classify an HTTP status code into an ErrorClass."""
    if status_code in TRANSIENT_STATUS_CODES:
        return ErrorClass.TRANSIENT
    return ErrorClass.FATAL


def classify_exception(exc: Exception) -> ErrorClass:
    """Classify a raised httpx exception into an ErrorClass."""
    if isinstance(exc, httpx.TimeoutException):
        return ErrorClass.TIMEOUT
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, httpx.RemoteProtocolError)):
        return ErrorClass.TRANSIENT
    return ErrorClass.FATAL


def parse_retry_after(headers: httpx.Headers) -> float | None:
    """Parse a Retry-After header (delta-seconds form) if present and numeric."""
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None

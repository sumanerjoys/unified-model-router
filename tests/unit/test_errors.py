"""Unit tests for the error taxonomy and classifiers."""

from __future__ import annotations

import httpx
import pytest

from app.core.errors import (
    ErrorClass,
    ProviderError,
    classify_exception,
    classify_status,
    parse_retry_after,
)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_status_codes(status: int) -> None:
    assert classify_status(status) is ErrorClass.TRANSIENT


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_fatal_status_codes(status: int) -> None:
    assert classify_status(status) is ErrorClass.FATAL


def test_timeout_exception_is_timeout() -> None:
    assert classify_exception(httpx.ConnectTimeout("x")) is ErrorClass.TIMEOUT
    assert classify_exception(httpx.ReadTimeout("x")) is ErrorClass.TIMEOUT


def test_network_errors_are_transient() -> None:
    assert classify_exception(httpx.ConnectError("x")) is ErrorClass.TRANSIENT
    assert classify_exception(httpx.RemoteProtocolError("x")) is ErrorClass.TRANSIENT


def test_unknown_exception_is_fatal() -> None:
    assert classify_exception(ValueError("x")) is ErrorClass.FATAL


def test_retryable_property() -> None:
    assert ErrorClass.TRANSIENT.is_retryable
    assert ErrorClass.TIMEOUT.is_retryable
    assert not ErrorClass.FATAL.is_retryable


def test_provider_error_carries_context() -> None:
    err = ProviderError(
        ErrorClass.TRANSIENT, "primary", status_code=429, retry_after=3.0
    )
    assert err.is_retryable
    assert err.status_code == 429
    assert err.retry_after == 3.0
    assert "primary" in str(err)


def test_parse_retry_after() -> None:
    assert parse_retry_after(httpx.Headers({"retry-after": "5"})) == 5.0
    assert parse_retry_after(httpx.Headers({})) is None
    assert parse_retry_after(httpx.Headers({"retry-after": "soon"})) is None

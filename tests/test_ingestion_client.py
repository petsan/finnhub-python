"""Tests for the FinnhubGateway wrapper (rate-limiter + retry behaviour)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from finnhub.exceptions import FinnhubAPIException

import requests

from finn_predictor.ingestion.client import (
    FinnhubGateway,
    IngestionError,
    RateLimiter,
    REDACTED,
    scrub_token,
)


# ---------------- RateLimiter ----------------


class FakeClock:
    """Drop-in monotonic clock; advances only when ``advance`` is called."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.t += secs


def test_rate_limiter_under_cap_does_not_sleep() -> None:
    clk = FakeClock()
    rl = RateLimiter(3, now=clk.now, sleep=clk.sleep)
    for _ in range(3):
        rl.acquire()
    assert clk.sleeps == []


def test_rate_limiter_blocks_when_cap_reached() -> None:
    clk = FakeClock()
    rl = RateLimiter(2, now=clk.now, sleep=clk.sleep)
    rl.acquire()           # t=0
    clk.t = 10.0
    rl.acquire()           # t=10
    clk.t = 20.0
    rl.acquire()           # 3rd call inside 60s -> must wait 60 - (20 - 0) = 40s
    assert clk.sleeps, "expected a sleep when cap is reached"
    assert clk.sleeps[0] == pytest.approx(40.0)


def test_rate_limiter_window_slides() -> None:
    clk = FakeClock()
    rl = RateLimiter(2, now=clk.now, sleep=clk.sleep)
    rl.acquire()
    clk.t = 61.0  # first timestamp now older than 60s
    rl.acquire()
    rl.acquire()  # second slot still fresh, so still room — no sleep
    assert clk.sleeps == []


def test_rate_limiter_rejects_non_positive_cap() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0)


# ---------------- FinnhubGateway ----------------


def _bad_response(status: int) -> MagicMock:
    """Build a minimal mock requests.Response for FinnhubAPIException."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = "boom"
    resp.json.return_value = {"error": "boom"}
    return resp


def _gateway(client: MagicMock, *, max_retries: int = 2) -> FinnhubGateway:
    sleeps: list[float] = []
    rl = RateLimiter(60, now=lambda: 0.0, sleep=lambda s: None)
    return FinnhubGateway(
        client=client,
        rate_limiter=rl,
        max_retries=max_retries,
        backoff_base=0.01,
        sleep=lambda s: sleeps.append(s),
    )


def test_gateway_general_news_passthrough() -> None:
    client = MagicMock()
    client.general_news.return_value = [{"id": 1}]
    gw = _gateway(client)
    out = gw.general_news("general", 0)
    assert out == [{"id": 1}]
    client.general_news.assert_called_once_with("general", 0)


def test_gateway_retries_on_retryable_status() -> None:
    client = MagicMock()
    client.general_news.side_effect = [
        FinnhubAPIException(_bad_response(429)),
        FinnhubAPIException(_bad_response(503)),
        [{"id": 7}],
    ]
    gw = _gateway(client, max_retries=3)
    assert gw.general_news() == [{"id": 7}]
    assert client.general_news.call_count == 3


def test_gateway_does_not_retry_on_non_retryable_status() -> None:
    client = MagicMock()
    client.general_news.side_effect = FinnhubAPIException(_bad_response(400))
    gw = _gateway(client)
    with pytest.raises(IngestionError):
        gw.general_news()
    assert client.general_news.call_count == 1


def test_gateway_exhausts_retries_then_raises() -> None:
    client = MagicMock()
    client.general_news.side_effect = FinnhubAPIException(_bad_response(429))
    gw = _gateway(client, max_retries=2)
    with pytest.raises(IngestionError):
        gw.general_news()
    assert client.general_news.call_count == 3  # initial + 2 retries


def test_gateway_company_news_and_candles() -> None:
    client = MagicMock()
    client.company_news.return_value = [{"id": 9}]
    client.stock_candles.return_value = {"s": "ok"}
    gw = _gateway(client)
    assert gw.company_news("AAPL", "2026-05-01", "2026-05-19") == [{"id": 9}]
    assert gw.stock_candles("AAPL", "D", 1, 2) == {"s": "ok"}


# ---------------- Token scrubbing ----------------


CANARY_TOKEN = "sk-test-9f7c-CANARY"


def _gateway_with_real_token(client: MagicMock) -> FinnhubGateway:
    # Make the gateway report `CANARY_TOKEN` as the current token so _scrub
    # can find it inside exception strings.
    client._session.params = {"token": CANARY_TOKEN}
    rl = RateLimiter(60, now=lambda: 0.0, sleep=lambda s: None)
    return FinnhubGateway(
        client=client,
        rate_limiter=rl,
        max_retries=0,
        backoff_base=0.0,
        sleep=lambda s: None,
    )


def test_scrub_token_helper() -> None:
    assert scrub_token("hello tok", "tok") == f"hello {REDACTED}"
    assert scrub_token("hello", "") == "hello"
    assert scrub_token("hello", None) == "hello"
    # multiple occurrences (Finnhub URLs sometimes repeat the token in path+query)
    assert scrub_token(f"a {CANARY_TOKEN} b {CANARY_TOKEN}", CANARY_TOKEN) \
        == f"a {REDACTED} b {REDACTED}"


def test_gateway_scrubs_token_from_request_exception() -> None:
    """SSL / connection errors include the URL+token; the gateway must redact."""
    client = MagicMock()
    leaky_url = f"https://api.finnhub.io/api/v1/news?token={CANARY_TOKEN}&category=general"
    client.general_news.side_effect = requests.exceptions.SSLError(
        f"HTTPSConnectionPool(host='api.finnhub.io'): SSL error on {leaky_url}"
    )
    gw = _gateway_with_real_token(client)
    with pytest.raises(IngestionError) as excinfo:
        gw.general_news()
    msg = str(excinfo.value)
    assert CANARY_TOKEN not in msg
    assert REDACTED in msg
    # The chain should be suppressed so default tracebacks don't re-leak.
    assert excinfo.value.__suppress_context__ is True


def test_gateway_scrubs_token_from_finnhub_api_exception() -> None:
    client = MagicMock()
    resp = MagicMock()
    resp.status_code = 401
    resp.text = "unauthorized"
    resp.json.return_value = {"error": f"bad token={CANARY_TOKEN}"}
    client.general_news.side_effect = FinnhubAPIException(resp)
    gw = _gateway_with_real_token(client)
    with pytest.raises(IngestionError) as excinfo:
        gw.general_news()
    assert CANARY_TOKEN not in str(excinfo.value)


def test_gateway_scrubs_token_from_connection_error() -> None:
    """ConnectionError is the parent of many transport failures we care about."""
    client = MagicMock()
    client.company_news.side_effect = requests.exceptions.ConnectionError(
        f"Failed to establish a new connection: ?token={CANARY_TOKEN}"
    )
    gw = _gateway_with_real_token(client)
    with pytest.raises(IngestionError) as excinfo:
        gw.company_news("AAPL", "2026-05-01", "2026-05-19")
    assert CANARY_TOKEN not in str(excinfo.value)


def test_gateway_does_not_scrub_when_token_is_unset() -> None:
    """If we never set a token, scrubbing is a no-op (avoid masking unrelated text)."""
    client = MagicMock()
    client._session.params = {}  # no token
    client.general_news.side_effect = requests.exceptions.SSLError("plain message")
    rl = RateLimiter(60, now=lambda: 0.0, sleep=lambda s: None)
    gw = FinnhubGateway(client=client, rate_limiter=rl, max_retries=0,
                        backoff_base=0.0, sleep=lambda s: None)
    with pytest.raises(IngestionError) as excinfo:
        gw.general_news()
    assert "plain message" in str(excinfo.value)
    assert REDACTED not in str(excinfo.value)

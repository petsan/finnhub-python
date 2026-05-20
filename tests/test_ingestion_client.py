"""Tests for the FinnhubGateway wrapper (rate-limiter + retry behaviour)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from finnhub.exceptions import FinnhubAPIException

from finn_predictor.ingestion.client import FinnhubGateway, RateLimiter


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
    with pytest.raises(FinnhubAPIException):
        gw.general_news()
    assert client.general_news.call_count == 1


def test_gateway_exhausts_retries_then_raises() -> None:
    client = MagicMock()
    client.general_news.side_effect = FinnhubAPIException(_bad_response(429))
    gw = _gateway(client, max_retries=2)
    with pytest.raises(FinnhubAPIException):
        gw.general_news()
    assert client.general_news.call_count == 3  # initial + 2 retries


def test_gateway_company_news_and_candles() -> None:
    client = MagicMock()
    client.company_news.return_value = [{"id": 9}]
    client.stock_candles.return_value = {"s": "ok"}
    gw = _gateway(client)
    assert gw.company_news("AAPL", "2026-05-01", "2026-05-19") == [{"id": 9}]
    assert gw.stock_candles("AAPL", "D", 1, 2) == {"s": "ok"}

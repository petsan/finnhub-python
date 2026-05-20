"""Throttled / retrying wrapper around :class:`finnhub.Client`.

Why a wrapper? The upstream client raises :class:`FinnhubAPIException` on any
non-2xx and has no built-in rate limiting or retry. For our scheduled
ingestion job we want:

  * deterministic call-pacing so we don't blow the free-tier 60 req/min quota
  * a couple of bounded retries on transient errors (429, 5xx)
  * a uniform place to scrub the API key out of logs

Anything fancier (circuit breaker, async fan-out) can replace this module
without callers noticing.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque

import requests
from finnhub import Client
from finnhub.exceptions import FinnhubAPIException


logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
REDACTED = "<REDACTED>"


class IngestionError(Exception):
    """Domain error raised after API-key scrubbing.

    Always raised with the API key replaced by :data:`REDACTED`. Created with
    ``raise IngestionError(msg) from None`` so the original exception chain
    (which may contain the raw URL) is suppressed from default tracebacks.
    """


def scrub_token(text: str, token: str | None) -> str:
    """Replace ``token`` everywhere it appears in ``text``.

    Robust to ``token`` being None or empty (returns ``text`` unchanged) so
    callers don't have to special-case the "no key yet" path.
    """
    if not token:
        return text
    return text.replace(token, REDACTED)


class RateLimiter:
    """Sliding-window rate limiter, ``calls_per_minute`` cap.

    Not thread-safe — callers should hold their own lock if sharing a single
    limiter across threads. The APScheduler in-process job runs one ingestion
    at a time so we don't need synchronization for our default deployment.
    """

    def __init__(
        self,
        calls_per_minute: int,
        *,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive")
        self._cap = calls_per_minute
        self._window: Deque[float] = deque()
        self._now = now
        self._sleep = sleep

    def acquire(self) -> None:
        """Block (via ``sleep``) until issuing one more call is in budget."""
        now = self._now()
        # Discard timestamps older than 60s.
        while self._window and now - self._window[0] >= 60.0:
            self._window.popleft()

        if len(self._window) >= self._cap:
            wait_for = 60.0 - (now - self._window[0])
            if wait_for > 0:
                self._sleep(wait_for)
            # Re-tick after sleeping.
            now = self._now()
            while self._window and now - self._window[0] >= 60.0:
                self._window.popleft()

        self._window.append(now)


@dataclass
class FinnhubGateway:
    """Adapter exposing the slice of Finnhub we care about, with retries."""

    client: Client
    rate_limiter: RateLimiter
    max_retries: int = 3
    backoff_base: float = 0.5
    sleep: Callable[[float], None] = field(default=time.sleep)

    def _current_token(self) -> str | None:
        """Best-effort lookup of the active API token on the underlying client.

        Returns None unless the token looks like a real non-empty string —
        this keeps :func:`scrub_token` safe when the client is a MagicMock
        or otherwise unconfigured.
        """
        try:
            token = self.client._session.params.get("token")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(token, str) and token:
            return token
        return None

    def _scrub(self, text: str) -> str:
        return scrub_token(text, self._current_token())

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke ``fn`` through the limiter, retrying transient failures.

        Any exception raised (Finnhub-level or transport-level) is scrubbed
        of the API token before being re-raised as :class:`IngestionError`.
        ``FinnhubAPIException`` is intercepted *after* the retry loop because
        only its ``status_code`` participates in retry decisions, not the
        leaky URL.
        """
        attempt = 0
        while True:
            self.rate_limiter.acquire()
            try:
                return fn(*args, **kwargs)
            except FinnhubAPIException as exc:
                if exc.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    delay = self.backoff_base * (2 ** attempt)
                    logger.warning(
                        "Finnhub %s -> %s, retrying in %.2fs (attempt %d/%d)",
                        getattr(fn, "__name__", repr(fn)),
                        exc.status_code,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    self.sleep(delay)
                    attempt += 1
                    continue
                # FinnhubAPIException.__str__ doesn't include the URL but its
                # .response object does (response.url). Re-raise with a
                # scrubbed string so callers can safely log it.
                raise IngestionError(
                    self._scrub(f"FinnhubAPI {exc.status_code}: {exc.message}")
                ) from None
            except requests.exceptions.RequestException as exc:
                # Transport-level failures (SSL, DNS, timeout, connection
                # reset) embed the full URL — including ?token=... — in their
                # message. Scrub before re-raising.
                raise IngestionError(self._scrub(str(exc))) from None

    # --- News --------------------------------------------------------------

    def general_news(self, category: str = "general", min_id: int = 0) -> list[dict]:
        return self._call(self.client.general_news, category, min_id)

    def company_news(self, symbol: str, _from: str, to: str) -> list[dict]:
        return self._call(self.client.company_news, symbol, _from, to)

    # --- Prices ------------------------------------------------------------

    def stock_candles(
        self, symbol: str, resolution: str, _from: int, to: int
    ) -> dict[str, Any]:
        return self._call(self.client.stock_candles, symbol, resolution, _from, to)

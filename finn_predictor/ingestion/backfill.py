"""Historical company-news backfill.

Pages :func:`finnhub.Client.company_news` across an arbitrary date range
in fixed-size chunks. Each chunk is funnelled through the existing
:func:`finn_predictor.ingestion.news.ingest_company_news`, so:

  * the :func:`storage.repo.upsert_articles` dedupe applies (re-running a
    backfill is safe — already-seen ``finnhub_id`` values are skipped);
  * the :class:`FinnhubGateway`'s rate limiter + retry already pace each
    chunk request;
  * a per-chunk failure (e.g. 429 / 5xx that exceeds retries, or a 403
    on a region the user's plan doesn't cover) is captured into a
    ``failures`` list and the remaining chunks still run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from finn_predictor.ingestion.client import FinnhubGateway, IngestionError
from finn_predictor.ingestion.news import ingest_company_news


logger = logging.getLogger(__name__)


# Finnhub's /company-news endpoint accepts a date range but caps each
# response at ~250–300 items in practice. 30 days is a safe chunk for
# even high-volume tickers (Apple/Nvidia easily clear 5–10 articles a
# day across sources, so a one-month window stays well under the cap).
DEFAULT_CHUNK_DAYS = 30


@dataclass(frozen=True)
class BackfillResult:
    """Counts and failures for a single-ticker backfill run."""

    symbol: str
    inserted: int
    chunks_attempted: int
    chunks_failed: int
    failures: list[dict[str, object]]

    @property
    def succeeded_chunks(self) -> int:
        return self.chunks_attempted - self.chunks_failed


def _as_utc(d: datetime) -> datetime:
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)


def backfill_company_news(
    session: Session,
    gateway: FinnhubGateway,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
) -> BackfillResult:
    """Backfill ``/company-news`` for ``symbol`` over ``[start, end]``.

    ``end`` is treated inclusively (so a 30-day backfill covers 30 calendar
    days). Returns a :class:`BackfillResult` describing what landed.

    A per-chunk :class:`IngestionError` (already token-scrubbed by the
    gateway) is appended to ``failures`` rather than aborting the run —
    a single 5xx on month 7 should not lose months 1–6.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")

    start = _as_utc(start)
    end = _as_utc(end)
    if end < start:
        raise ValueError("end must be on or after start")

    inserted = 0
    chunks_attempted = 0
    chunks_failed = 0
    failures: list[dict[str, object]] = []

    cur = start
    while True:
        # Clip the chunk to the user-supplied end. Using `<` (not `<=`)
        # lets a single-day range [D, D] still produce one call.
        chunk_end = min(cur + timedelta(days=chunk_days), end)
        if chunk_end < cur:
            break

        chunks_attempted += 1
        try:
            inserted += ingest_company_news(
                session,
                gateway,
                symbol=symbol,
                start=cur,
                end=chunk_end,
            )
        except IngestionError as exc:
            chunks_failed += 1
            failures.append(
                {
                    "op": f"backfill_company_news:{symbol}",
                    "chunk_start": cur,
                    "chunk_end": chunk_end,
                    "error": str(exc),
                }
            )
            logger.warning(
                "backfill chunk failed for %s [%s, %s]: %s",
                symbol,
                cur.date(),
                chunk_end.date(),
                exc,
            )

        if chunk_end >= end:
            break
        cur = chunk_end

    return BackfillResult(
        symbol=symbol,
        inserted=inserted,
        chunks_attempted=chunks_attempted,
        chunks_failed=chunks_failed,
        failures=failures,
    )


def backfill_many(
    session: Session,
    gateway: FinnhubGateway,
    *,
    symbols: list[str],
    start: datetime,
    end: datetime,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
) -> dict[str, BackfillResult]:
    """Backfill multiple tickers; one :class:`BackfillResult` per symbol.

    Ignores empty / whitespace-only entries in ``symbols``. The order of
    returned items matches the input order.
    """
    out: dict[str, BackfillResult] = {}
    for sym in symbols:
        sym = sym.strip()
        if not sym:
            continue
        out[sym] = backfill_company_news(
            session,
            gateway,
            symbol=sym,
            start=start,
            end=end,
            chunk_days=chunk_days,
        )
    return out

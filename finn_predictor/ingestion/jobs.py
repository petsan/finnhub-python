"""APScheduler glue.

Two callables are exposed:

  * :func:`run_daily_ingest` — pulls today's news + prices + scores articles
    + writes a fresh prediction for ^GSPC (iter 1) and every sector ETF
    (iter 2 once sectors are seeded).
  * :func:`build_scheduler` — returns an :class:`BackgroundScheduler` already
    wired to ``run_daily_ingest`` on a daily cron.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session

from finn_predictor.ingestion.client import FinnhubGateway, IngestionError
from finn_predictor.ingestion.news import ingest_company_news, ingest_general_news
from finn_predictor.ingestion.prices import ingest_market_caps, ingest_price_history
from finn_predictor.learning.config import active_weights
from finn_predictor.predictor.classifier import (
    load_calibration,
    resolve_classifier_mode,
)
from finn_predictor.predictor.market import predict_market
from finn_predictor.predictor.sectors import predict_all_sectors
from finn_predictor.predictor.stocks import predict_all_stocks
from finn_predictor.sentiment.base import Scorer
from finn_predictor.storage.models import SentimentScore
from finn_predictor.storage.repo import (
    ensure_default_sectors,
    save_scores,
    unscored_articles,
)


logger = logging.getLogger(__name__)


def score_pending_articles(session: Session, scorer: Scorer, limit: int = 500) -> int:
    """Score any articles missing a row for ``scorer.model_version``."""
    pending = unscored_articles(session, scorer.model_version, limit=limit)
    if not pending:
        return 0
    fresh = [
        SentimentScore(
            article_id=art.id,
            score=scorer.score(f"{art.headline}. {art.summary}".strip()),
            model_version=scorer.model_version,
        )
        for art in pending
    ]
    return save_scores(session, fresh)


def run_daily_ingest(
    *,
    session: Session,
    gateway: FinnhubGateway,
    scorer: Scorer,
    market_symbol: str = "^GSPC",
    company_symbols: Iterable[str] = (),
    today: datetime | None = None,
) -> dict[str, object]:
    """Run one end-to-end ingestion + prediction cycle.

    Resilient by design: every Finnhub call is wrapped so a single failing
    endpoint (e.g. a 403 because the user's plan doesn't include
    ``/stock/candle``) does not abort the rest of the run. Errors are
    collected into ``counts["failures"]`` and surfaced to the caller —
    already token-scrubbed by :class:`FinnhubGateway`.

    Returns a dict with integer counts per category plus a ``"failures"``
    key holding ``[{"op": str, "error": str}, ...]``.
    """
    today = today or datetime.now(timezone.utc)
    yesterday = today - timedelta(days=7)  # generous window catches weekends

    counts: dict[str, object] = {
        "general_news": 0,
        "company_news": 0,
        "market_prices": 0,
        "sector_prices": 0,
        "company_prices": 0,
        "company_caps": 0,
        "scored": 0,
        "predictions": 0,
    }
    failures: list[dict[str, str]] = []

    def _try(op_name: str, fn) -> int:
        try:
            return int(fn())
        except IngestionError as exc:
            failures.append({"op": op_name, "error": str(exc)})
            return 0

    counts["general_news"] = _try(
        "general_news",
        lambda: ingest_general_news(session, gateway, category="general"),
    )
    counts["market_prices"] = _try(
        f"market_prices:{market_symbol}",
        lambda: ingest_price_history(
            session, gateway, symbol=market_symbol, start=yesterday, end=today
        ),
    )

    # Iter-2 prerequisites: per-sector ETF prices.
    sectors = ensure_default_sectors(session)
    for sector in sectors:
        counts["sector_prices"] = int(counts["sector_prices"]) + _try(
            f"sector_prices:{sector.etf_symbol}",
            lambda s=sector: ingest_price_history(
                session, gateway, symbol=s.etf_symbol, start=yesterday, end=today
            ),
        )

    for symbol in company_symbols:
        counts["company_news"] = int(counts["company_news"]) + _try(
            f"company_news:{symbol}",
            lambda sym=symbol: ingest_company_news(
                session, gateway, symbol=sym, start=yesterday, end=today
            ),
        )
        counts["company_prices"] = int(counts["company_prices"]) + _try(
            f"company_prices:{symbol}",
            lambda sym=symbol: ingest_price_history(
                session, gateway, symbol=sym, start=yesterday, end=today
            ),
        )
        # Market caps populate the cap-weighting lookup used by
        # `predict_sector`. A 403 here (often the case on free tier)
        # is non-fatal — sectors silently fall back to uniform
        # weighting for any ticker missing a cap row.
        counts["company_caps"] = int(counts["company_caps"]) + _try(
            f"company_caps:{symbol}",
            lambda sym=symbol: ingest_market_caps(
                session, gateway, symbol=sym, start=yesterday, end=today
            ),
        )

    # Scoring + predictions are DB-only operations — they always run, even if
    # every Finnhub fetch above failed, because there may be older articles
    # left over from a previous successful ingestion.
    counts["scored"] = score_pending_articles(session, scorer)

    # Load active learned weights once and pass through to every predictor.
    # All four knobs the learning loop fits — threshold, σ floor,
    # half-life, per-source multipliers — are honoured here. Without
    # this plumbing, "self-improvement" would persist new weights and
    # then never actually use the half-life or source weights at live
    # scoring time.
    weights = active_weights(session)
    learned_kwargs = dict(
        threshold_sigma=weights.threshold_sigma,
        min_baseline_sigma=weights.min_baseline_sigma,
        half_life_hours=weights.half_life_hours,
        source_weights=weights.source_weights,
    )

    # When FINN_PREDICTOR_CLASSIFIER=logreg and a calibration has been
    # fitted (via the `fit-classifier` CLI subcommand), pass it through
    # so the final classify step swaps z-distance for calibrated
    # probability. Falls back to the rule classifier on missing data.
    calibration = None
    if resolve_classifier_mode() == "logreg":
        calibration = load_calibration(session)
    learned_kwargs["calibration"] = calibration

    market_pred = predict_market(
        session, scorer=scorer, on_date=today, symbol=market_symbol,
        **learned_kwargs,
    )
    if market_pred is not None:
        counts["predictions"] = int(counts["predictions"]) + 1

    # sector_universe=None means "read cached ETF_HOLDING rows from the
    # DB". Sectors without cached constituents (i.e. nobody has clicked
    # *Refresh constituents* in the Focus tab) are skipped silently.
    sector_preds = predict_all_sectors(
        session, scorer=scorer, on_date=today, **learned_kwargs,
    )
    counts["predictions"] = int(counts["predictions"]) + len(sector_preds)

    # Per-stock predictions for every ticker the user supplied to
    # company_symbols. Each ticker gets its own Prediction row with
    # target_symbol=<ticker> so the existing upsert (one row per
    # ticker per day per model) applies.
    stock_preds = predict_all_stocks(
        session, scorer=scorer, symbols=list(company_symbols), on_date=today,
        **learned_kwargs,
    )
    counts["predictions"] = int(counts["predictions"]) + len(stock_preds)

    counts["failures"] = failures
    return counts


def build_scheduler(
    job: Callable[[], None],
    *,
    hour: int = 21,
    minute: int = 30,
) -> BackgroundScheduler:
    """Build a BackgroundScheduler that runs ``job`` daily at the given UTC time.

    Default: 21:30 UTC (~17:30 ET, half an hour after the NYSE close). The
    caller owns starting/stopping the scheduler.
    """
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        job,
        trigger=CronTrigger(hour=hour, minute=minute, timezone="UTC"),
        id="daily_ingest",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    return scheduler

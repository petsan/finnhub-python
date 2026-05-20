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

from finn_predictor.ingestion.client import FinnhubGateway
from finn_predictor.ingestion.news import ingest_company_news, ingest_general_news
from finn_predictor.ingestion.prices import ingest_price_history
from finn_predictor.predictor.market import predict_market
from finn_predictor.predictor.sectors import predict_all_sectors
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
) -> dict[str, int]:
    """Run one end-to-end ingestion + prediction cycle.

    Returns a small dict of counts for the calling job to log.
    """
    today = today or datetime.now(timezone.utc)
    yesterday = today - timedelta(days=7)  # generous window catches weekends

    counts = {
        "general_news": ingest_general_news(session, gateway, category="general"),
        "company_news": 0,
        "market_prices": ingest_price_history(
            session, gateway, symbol=market_symbol, start=yesterday, end=today
        ),
        "sector_prices": 0,
        "company_prices": 0,
        "scored": 0,
        "predictions": 0,
    }

    # Iter-2 prerequisites: per-sector ETF prices + per-company news.
    sectors = ensure_default_sectors(session)
    for sector in sectors:
        counts["sector_prices"] += ingest_price_history(
            session,
            gateway,
            symbol=sector.etf_symbol,
            start=yesterday,
            end=today,
        )

    for symbol in company_symbols:
        counts["company_news"] += ingest_company_news(
            session, gateway, symbol=symbol, start=yesterday, end=today
        )
        counts["company_prices"] += ingest_price_history(
            session, gateway, symbol=symbol, start=yesterday, end=today
        )

    counts["scored"] = score_pending_articles(session, scorer)

    market_pred = predict_market(session, scorer=scorer, on_date=today, symbol=market_symbol)
    if market_pred is not None:
        counts["predictions"] += 1

    sector_preds = predict_all_sectors(session, scorer=scorer, on_date=today)
    counts["predictions"] += len(sector_preds)
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

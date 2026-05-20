"""Repository helpers. All DB access funnels through this module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from finn_predictor.storage.models import (
    NewsArticle,
    PriceBar,
    Prediction,
    PredictionOutcome,
    Sector,
    SentimentScore,
)


def _utcnow() -> datetime:
    """UTC-aware now() — duplicated from models.py so callers don't import a private helper."""
    return datetime.now(timezone.utc)


# Default sector universe used by iteration 2. The codes mirror the Sector
# Select SPDR ETFs which we use as proxy targets.
DEFAULT_SECTORS: tuple[tuple[str, str, str], ...] = (
    ("TECH", "Information Technology", "XLK"),
    ("ENERGY", "Energy", "XLE"),
    ("FIN", "Financials", "XLF"),
    ("HEALTH", "Health Care", "XLV"),
    ("DISCRETIONARY", "Consumer Discretionary", "XLY"),
    ("STAPLES", "Consumer Staples", "XLP"),
    ("INDUSTRIAL", "Industrials", "XLI"),
    ("MATERIAL", "Materials", "XLB"),
    ("UTILITIES", "Utilities", "XLU"),
    ("REAL_ESTATE", "Real Estate", "XLRE"),
    ("COMM", "Communication Services", "XLC"),
)


# -- News -------------------------------------------------------------------


def upsert_articles(session: Session, articles: Iterable[NewsArticle]) -> int:
    """Insert articles, skipping any whose ``finnhub_id`` is already present.

    Returns the number of new rows inserted. We use SQLite's ``ON CONFLICT
    DO NOTHING`` to keep this idempotent under retried ingestion runs.
    """
    inserted = 0
    for art in articles:
        # Core-level inserts don't fire SQLAlchemy column defaults; fall back
        # to a wall-clock timestamp if the caller didn't set one explicitly.
        ingested_at = art.ingested_at or _utcnow()
        stmt = (
            sqlite_insert(NewsArticle)
            .values(
                finnhub_id=art.finnhub_id,
                category=art.category,
                symbol=art.symbol,
                headline=art.headline,
                summary=art.summary or "",
                source=art.source or "",
                url=art.url or "",
                published_at=art.published_at,
                ingested_at=ingested_at,
            )
            .on_conflict_do_nothing(index_elements=["finnhub_id"])
        )
        result = session.execute(stmt)
        inserted += result.rowcount or 0
    session.commit()
    return inserted


def articles_in_window(
    session: Session,
    start: datetime,
    end: datetime,
    *,
    category: Optional[str] = None,
    symbol: Optional[str] = None,
) -> list[NewsArticle]:
    """Articles published in ``[start, end)`` (UTC).

    ``category`` and ``symbol`` are optional filters; pass ``symbol=None`` to
    include both whole-market and company news.
    """
    stmt = select(NewsArticle).where(
        NewsArticle.published_at >= start,
        NewsArticle.published_at < end,
    )
    if category is not None:
        stmt = stmt.where(NewsArticle.category == category)
    if symbol is not None:
        stmt = stmt.where(NewsArticle.symbol == symbol)
    stmt = stmt.order_by(NewsArticle.published_at)
    return list(session.scalars(stmt))


def unscored_articles(
    session: Session, model_version: str, *, limit: int = 500
) -> list[NewsArticle]:
    """Articles that have no SentimentScore for ``model_version`` yet."""
    scored_subq = (
        select(SentimentScore.article_id)
        .where(SentimentScore.model_version == model_version)
        .subquery()
    )
    stmt = (
        select(NewsArticle)
        .where(NewsArticle.id.not_in(select(scored_subq.c.article_id)))
        .order_by(NewsArticle.published_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


# -- Sentiment --------------------------------------------------------------


def save_scores(session: Session, scores: Iterable[SentimentScore]) -> int:
    """Persist sentiment scores. ``(article_id, model_version)`` is unique."""
    n = 0
    for s in scores:
        session.add(s)
        n += 1
    session.commit()
    return n


# -- Prices -----------------------------------------------------------------


def upsert_price_bars(session: Session, bars: Iterable[PriceBar]) -> int:
    """Insert daily price bars, skipping ``(symbol, trade_date)`` duplicates."""
    inserted = 0
    for bar in bars:
        stmt = (
            sqlite_insert(PriceBar)
            .values(
                symbol=bar.symbol,
                trade_date=bar.trade_date,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume if bar.volume is not None else 0.0,
            )
            .on_conflict_do_nothing(index_elements=["symbol", "trade_date"])
        )
        result = session.execute(stmt)
        inserted += result.rowcount or 0
    session.commit()
    return inserted


def price_bars(
    session: Session,
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[PriceBar]:
    """Ordered list of bars for ``symbol`` in ``[start, end)``."""
    stmt = (
        select(PriceBar)
        .where(
            PriceBar.symbol == symbol,
            PriceBar.trade_date >= start,
            PriceBar.trade_date < end,
        )
        .order_by(PriceBar.trade_date)
    )
    return list(session.scalars(stmt))


def latest_price_bar(session: Session, symbol: str) -> Optional[PriceBar]:
    stmt = (
        select(PriceBar)
        .where(PriceBar.symbol == symbol)
        .order_by(PriceBar.trade_date.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


# -- Predictions ------------------------------------------------------------


def save_prediction(session: Session, prediction: Prediction) -> Prediction:
    """Upsert a Prediction. Returns the persisted instance.

    If a prediction already exists for the same
    ``(target_symbol, prediction_date, model_version)`` triple we update the
    label/confidence rather than create a duplicate. This makes the predictor
    job safely re-runnable.
    """
    existing = session.scalar(
        select(Prediction).where(
            Prediction.target_symbol == prediction.target_symbol,
            Prediction.prediction_date == prediction.prediction_date,
            Prediction.model_version == prediction.model_version,
        )
    )
    if existing is not None:
        existing.label = prediction.label
        existing.confidence = prediction.confidence
        existing.sentiment_index = prediction.sentiment_index
        existing.article_count = prediction.article_count
        session.commit()
        return existing

    session.add(prediction)
    session.commit()
    return prediction


def predictions_for(
    session: Session,
    target_symbol: str,
    *,
    since: Optional[datetime] = None,
) -> list[Prediction]:
    stmt = select(Prediction).where(Prediction.target_symbol == target_symbol)
    if since is not None:
        stmt = stmt.where(Prediction.prediction_date >= since)
    stmt = stmt.order_by(Prediction.prediction_date)
    return list(session.scalars(stmt))


def save_outcome(session: Session, outcome: PredictionOutcome) -> PredictionOutcome:
    """Persist (or refresh) the realised outcome for a prediction."""
    realised_at = outcome.realised_at or _utcnow()
    existing = session.scalar(
        select(PredictionOutcome).where(
            PredictionOutcome.prediction_id == outcome.prediction_id
        )
    )
    if existing is not None:
        existing.realised_return = outcome.realised_return
        existing.hit = outcome.hit
        existing.realised_at = realised_at
        session.commit()
        return existing

    if outcome.realised_at is None:
        outcome.realised_at = realised_at
    session.add(outcome)
    session.commit()
    return outcome


# -- Sectors ----------------------------------------------------------------


def ensure_default_sectors(session: Session) -> list[Sector]:
    """Idempotently insert :data:`DEFAULT_SECTORS` and return them all."""
    for code, name, etf in DEFAULT_SECTORS:
        existing = session.scalar(select(Sector).where(Sector.code == code))
        if existing is None:
            session.add(Sector(code=code, name=name, etf_symbol=etf))
    session.commit()
    return list(session.scalars(select(Sector).order_by(Sector.code)))


def all_sectors(session: Session) -> Sequence[Sector]:
    return list(session.scalars(select(Sector).order_by(Sector.code)))


# -- Date helpers used widely ----------------------------------------------


def utc_day_window(day: datetime) -> tuple[datetime, datetime]:
    """Return ``[start_of_day, start_of_next_day)`` in UTC for ``day``."""
    day = day.astimezone(timezone.utc) if day.tzinfo else day.replace(tzinfo=timezone.utc)
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)

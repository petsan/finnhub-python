"""Shared pytest fixtures.

Every test gets a fresh in-memory SQLite database — fast and fully isolated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

import pytest
from sqlalchemy.orm import Session

from finn_predictor.storage import create_engine_and_session, init_db
from finn_predictor.storage.models import (
    NewsArticle,
    PriceBar,
    Prediction,
    SentimentScore,
)


@pytest.fixture
def session() -> Iterator[Session]:
    engine, SessionLocal = create_engine_and_session("sqlite:///:memory:")
    init_db(engine)
    with SessionLocal() as s:
        yield s
    engine.dispose()


@pytest.fixture
def utc() -> type[timezone]:
    return timezone


def make_article(
    *,
    finnhub_id: int,
    headline: str = "headline",
    summary: str = "summary",
    category: str = "general",
    symbol: str | None = None,
    published_at: datetime | None = None,
    source: str = "Reuters",
    url: str = "https://example.com",
) -> NewsArticle:
    """Factory used across multiple test modules."""
    return NewsArticle(
        finnhub_id=finnhub_id,
        category=category,
        symbol=symbol,
        headline=headline,
        summary=summary,
        source=source,
        url=url,
        published_at=published_at or datetime(2026, 5, 19, 12, tzinfo=timezone.utc),
    )


def make_score(article_id: int, score: float, *, model_version: str = "vader-test") -> SentimentScore:
    return SentimentScore(
        article_id=article_id, score=score, model_version=model_version
    )


def make_price_bar(
    symbol: str,
    trade_date: datetime,
    close: float,
    *,
    open_: float | None = None,
) -> PriceBar:
    return PriceBar(
        symbol=symbol,
        trade_date=trade_date,
        open=open_ if open_ is not None else close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def make_prediction(
    *,
    target_symbol: str,
    prediction_date: datetime,
    label: str = "UP",
    confidence: float = 0.5,
    sentiment_index: float = 0.1,
    article_count: int = 5,
    model_version: str = "vader-test",
) -> Prediction:
    return Prediction(
        target_symbol=target_symbol,
        prediction_date=prediction_date,
        label=label,
        confidence=confidence,
        sentiment_index=sentiment_index,
        article_count=article_count,
        model_version=model_version,
    )

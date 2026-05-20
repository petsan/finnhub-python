"""SQLAlchemy ORM models.

Design notes:
    * Times are stored as UTC-aware `DateTime`. We pass `timezone=True` on the
      column so PostgreSQL would store with tz and SQLite stores ISO strings
      that round-trip.
    * `NewsArticle.finnhub_id` is the upstream `id` field — used to dedupe
      across ingestion runs.
    * `Prediction.target_symbol` is "^GSPC" for the iter-1 whole-market call
      and a sector-ETF ticker (XLK, XLE, ...) for iter 2.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _utcnow() -> datetime:
    """Timezone-aware UTC now, suitable as a SQLAlchemy default factory."""
    return datetime.now(timezone.utc)


class Sector(Base):
    """A coarse market sector (e.g. Technology -> XLK)."""

    __tablename__ = "sectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    etf_symbol: Mapped[str] = mapped_column(String(8), nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"Sector(code={self.code!r}, etf={self.etf_symbol!r})"


class NewsArticle(Base):
    """A single news article ingested from Finnhub."""

    __tablename__ = "news_articles"
    __table_args__ = (
        UniqueConstraint("finnhub_id", name="uq_news_articles_finnhub_id"),
        Index("ix_news_articles_published_at", "published_at"),
        Index("ix_news_articles_symbol", "symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    finnhub_id: Mapped[int] = mapped_column(Integer, nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    # symbol is null for general (whole-market) news, set for company news.
    symbol: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    sentiment_scores: Mapped[list["SentimentScore"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )


class SentimentScore(Base):
    """A per-article sentiment score from a named model.

    A given article may be scored by multiple models (e.g. ``vader`` and
    ``finbert``); the ``(article_id, model_version)`` pair is unique.
    """

    __tablename__ = "sentiment_scores"
    __table_args__ = (
        UniqueConstraint(
            "article_id", "model_version", name="uq_sentiment_scores_article_model"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    article: Mapped[NewsArticle] = relationship(back_populates="sentiment_scores")


class PriceBar(Base):
    """A daily OHLCV bar for a target symbol (index or ETF)."""

    __tablename__ = "price_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", name="uq_price_bars_symbol_date"),
        Index("ix_price_bars_symbol_date", "symbol", "trade_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    trade_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class HistoricalMarketCap(Base):
    """A snapshot of a ticker's market capitalisation on a given date.

    Sourced from Finnhub's ``/stock/historical-market-cap`` endpoint and
    used to cap-weight sector-level sentiment aggregates (a tech-sector
    headline about Apple should outweigh one about a $500M small-cap by
    several orders of magnitude). We persist the raw value Finnhub
    returns (millions of USD) — downstream code normalises rather than
    relies on the absolute unit so currency / scale changes don't break
    the predictor.
    """

    __tablename__ = "historical_market_caps"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "as_of_date", name="uq_historical_market_caps_symbol_date"
        ),
        Index("ix_historical_market_caps_symbol_date", "symbol", "as_of_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    as_of_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    market_cap: Mapped[float] = mapped_column(Float, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class Prediction(Base):
    """A directional forecast for a target symbol on a given date."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "target_symbol",
            "prediction_date",
            "model_version",
            name="uq_predictions_symbol_date_model",
        ),
        Index("ix_predictions_target_date", "target_symbol", "prediction_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    prediction_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    label: Mapped[str] = mapped_column(String(8), nullable=False)  # UP/DOWN/FLAT
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_index: Mapped[float] = mapped_column(Float, nullable=False)
    article_count: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    outcome: Mapped[Optional["PredictionOutcome"]] = relationship(
        back_populates="prediction",
        uselist=False,
        cascade="all, delete-orphan",
    )


class AppSetting(Base):
    """Tiny key/value table for cross-session UI settings.

    Used today for ``activation_policy`` ∈ {AUTO, MANUAL}. Survives
    Streamlit restarts (unlike ``st.session_state``) and is the
    canonical source the predictor + training loop read from.
    """

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class LearnedWeight(Base):
    """One row per (version, dimension, key) learned during a training run.

    A "version" is the unit of weight activation — exactly one version is
    ``is_active = True`` at any time, and we never mutate historical
    versions so old predictions remain reproducible by referencing the
    version that generated them.

    ``dimension`` ∈ {``THRESHOLD_SIGMA``, ``MIN_BASELINE_SIGMA``,
    ``HALF_LIFE_HOURS``, ``SOURCE_WEIGHT``}. ``key`` is None for the
    scalar dimensions and the source name (e.g. ``"Reuters"``) for
    SOURCE_WEIGHT rows.
    """

    __tablename__ = "learned_weights"
    __table_args__ = (
        UniqueConstraint(
            "version", "dimension", "key", name="uq_learned_weights"
        ),
        Index("ix_learned_weights_version", "version"),
        Index("ix_learned_weights_active", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    dimension: Mapped[str] = mapped_column(String(32), nullable=False)
    key: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    fitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    training_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    holdout_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=False)


class RelatedEntity(Base):
    """Cached relationship from a ticker to another entity.

    ``relationship`` ∈ {PEER, SUPPLIER, CUSTOMER, ETF_HOLDING}.
    ``related_symbol`` is a ticker for everything except where Finnhub
    returns a non-listed entity (then it's the human name).
    """

    __tablename__ = "related_entities"
    __table_args__ = (
        UniqueConstraint(
            "source_symbol",
            "related_symbol",
            "relationship",
            name="uq_related_entities",
        ),
        Index("ix_related_entities_source", "source_symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    related_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    relationship: Mapped[str] = mapped_column(String(32), nullable=False)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class PredictionOutcome(Base):
    """The realised next-session return paired with a Prediction."""

    __tablename__ = "prediction_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    realised_return: Mapped[float] = mapped_column(Float, nullable=False)
    hit: Mapped[bool] = mapped_column(nullable=False)
    realised_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    prediction: Mapped[Prediction] = relationship(back_populates="outcome")

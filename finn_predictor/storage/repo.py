"""Repository helpers. All DB access funnels through this module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from finn_predictor.storage.models import (
    AppSetting,
    LearnedWeight,
    NewsArticle,
    PriceBar,
    Prediction,
    PredictionOutcome,
    RelatedEntity,
    Sector,
    SentimentScore,
)


def _utcnow() -> datetime:
    """UTC-aware now() — duplicated from models.py so callers don't import a private helper."""
    return datetime.now(timezone.utc)


def _dialect_insert(session: Session):
    """Return the right INSERT builder for the session's bound dialect.

    SQLAlchemy's ``dialects.sqlite.insert`` and
    ``dialects.postgresql.insert`` both expose ``on_conflict_do_nothing``
    but live in separate modules. Picking at runtime lets the same
    repository functions work against either store without changing
    call sites — set ``FINN_PREDICTOR_DB_URL=postgresql://...`` and
    everything else just works.
    """
    bind = session.get_bind()
    name = getattr(getattr(bind, "dialect", None), "name", "sqlite")
    if name == "postgresql":
        return postgres_insert
    # SQLite is the historical default; anything else falls back to the
    # SQLite builder (works on most file-backed SQL dialects with the
    # same syntax). This isn't strict — callers that point at MySQL
    # or another dialect that doesn't support ON CONFLICT DO NOTHING
    # will hit a clear error from the DB driver.
    return sqlite_insert


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
    insert_builder = _dialect_insert(session)
    for art in articles:
        # Core-level inserts don't fire SQLAlchemy column defaults; fall back
        # to a wall-clock timestamp if the caller didn't set one explicitly.
        ingested_at = art.ingested_at or _utcnow()
        stmt = (
            insert_builder(NewsArticle)
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
    insert_builder = _dialect_insert(session)
    for bar in bars:
        stmt = (
            insert_builder(PriceBar)
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


def migrate_predictions_to_daily(session: Session) -> int:
    """Collapse duplicate predictions to one row per (symbol, day, model).

    Before this migration, ``Prediction.prediction_date`` was stored at
    seconds precision, so a user clicking *Run ingestion now* repeatedly
    on a single calendar day would mint multiple rows that differed only
    by timestamp. Going forward the predictors normalise to start-of-UTC-
    day; this helper cleans up the existing rows.

    Strategy:
      * group predictions by ``(target_symbol, utc-day(prediction_date), model_version)``
      * keep the row with the latest ``created_at`` per group, normalise
        its ``prediction_date`` to the day's midnight UTC
      * delete the other rows in the group (cascade removes any attached
        ``PredictionOutcome``)

    Returns the number of rows deleted (zero on a clean DB → idempotent).
    """
    all_preds = list(
        session.scalars(
            select(Prediction).order_by(Prediction.created_at.desc())
        )
    )
    by_group: dict[tuple[str, datetime, str], list[Prediction]] = {}
    for p in all_preds:
        day_start, _ = utc_day_window(p.prediction_date)
        by_group.setdefault((p.target_symbol, day_start, p.model_version), []).append(p)

    deleted = 0
    for (sym, day, mv), group in by_group.items():
        # group is sorted DESC by created_at because the source query was;
        # take group[0] as the survivor.
        survivor = group[0]
        survivor.prediction_date = day  # normalise even when no extras exist
        for extra in group[1:]:
            session.delete(extra)
            deleted += 1
    session.commit()
    return deleted


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


# -- Related entities ------------------------------------------------------

# The fixed vocabulary for RelatedEntity.relationship. Anything else
# is a programming error.
RELATIONSHIPS = frozenset(
    {"PEER", "SUPPLIER", "CUSTOMER", "ETF_HOLDING"}
)


def upsert_related_entity(
    session: Session,
    *,
    source_symbol: str,
    related_symbol: str,
    relationship: str,
    rank: Optional[int] = None,
    metadata_text: Optional[str] = None,
    fetched_at: Optional[datetime] = None,
) -> RelatedEntity:
    """Idempotently insert or refresh a (source, related, relationship) row.

    On conflict the ``rank``, ``metadata_text``, and ``fetched_at`` columns
    are updated; the older fields are overwritten with whatever the caller
    just observed. Returns the persisted instance.
    """
    if relationship not in RELATIONSHIPS:
        raise ValueError(
            f"relationship must be one of {sorted(RELATIONSHIPS)}, got {relationship!r}"
        )
    fetched = fetched_at or _utcnow()

    existing = session.scalar(
        select(RelatedEntity).where(
            RelatedEntity.source_symbol == source_symbol,
            RelatedEntity.related_symbol == related_symbol,
            RelatedEntity.relationship == relationship,
        )
    )
    if existing is not None:
        existing.rank = rank
        existing.metadata_text = metadata_text
        existing.fetched_at = fetched
        session.commit()
        return existing

    row = RelatedEntity(
        source_symbol=source_symbol,
        related_symbol=related_symbol,
        relationship=relationship,
        rank=rank,
        metadata_text=metadata_text,
        fetched_at=fetched,
    )
    session.add(row)
    session.commit()
    return row


def related_entities_for(
    session: Session,
    source_symbol: str,
    *,
    relationship: Optional[str] = None,
) -> list[RelatedEntity]:
    """Cached related entities for ``source_symbol``.

    Order: by ``rank`` ascending (None last), then by ``related_symbol``.
    """
    stmt = select(RelatedEntity).where(
        RelatedEntity.source_symbol == source_symbol
    )
    if relationship is not None:
        stmt = stmt.where(RelatedEntity.relationship == relationship)
    rows = list(session.scalars(stmt))
    rows.sort(
        key=lambda r: (r.rank is None, r.rank if r.rank is not None else 0, r.related_symbol)
    )
    return rows


# -- App settings (cross-session UI preferences) ---------------------------

# Canonical key for the activation policy. Values are "AUTO" or "MANUAL".
SETTING_ACTIVATION_POLICY = "activation_policy"
POLICY_AUTO = "AUTO"
POLICY_MANUAL = "MANUAL"
VALID_POLICIES = frozenset({POLICY_AUTO, POLICY_MANUAL})

# Tolerance (in objective units) for the holdout-improvement gate. A
# new candidate must score within (current - tolerance) on the same
# holdout window to be eligible for auto-activation.
SETTING_HOLDOUT_TOLERANCE = "holdout_tolerance"
DEFAULT_HOLDOUT_TOLERANCE = 0.01


def get_setting(
    session: Session, key: str, *, default: Optional[str] = None
) -> Optional[str]:
    """Read a row from app_settings; returns ``default`` when missing."""
    row = session.scalar(select(AppSetting).where(AppSetting.key == key))
    return row.value if row is not None else default


def set_setting(session: Session, key: str, value: str) -> AppSetting:
    """Upsert an app_settings row. Stamps ``updated_at`` to now."""
    now = _utcnow()
    existing = session.scalar(select(AppSetting).where(AppSetting.key == key))
    if existing is not None:
        existing.value = value
        existing.updated_at = now
        session.commit()
        return existing
    row = AppSetting(key=key, value=value, updated_at=now)
    session.add(row)
    session.commit()
    return row


def get_activation_policy(session: Session) -> str:
    """Read activation_policy with AUTO as the safe default."""
    val = get_setting(session, SETTING_ACTIVATION_POLICY, default=POLICY_AUTO)
    if val not in VALID_POLICIES:
        return POLICY_AUTO
    return val


def set_activation_policy(session: Session, value: str) -> None:
    """Set activation_policy. Rejects unknown values."""
    if value not in VALID_POLICIES:
        raise ValueError(
            f"activation_policy must be one of {sorted(VALID_POLICIES)}, got {value!r}"
        )
    set_setting(session, SETTING_ACTIVATION_POLICY, value)


def get_holdout_tolerance(session: Session) -> float:
    """Read the holdout-gate tolerance. Defaults to :data:`DEFAULT_HOLDOUT_TOLERANCE`.

    A junk value persisted in the DB falls back to the default.
    """
    raw = get_setting(session, SETTING_HOLDOUT_TOLERANCE)
    if raw is None:
        return DEFAULT_HOLDOUT_TOLERANCE
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_HOLDOUT_TOLERANCE
    if v < 0:
        return DEFAULT_HOLDOUT_TOLERANCE
    return v


def set_holdout_tolerance(session: Session, value: float) -> None:
    """Set holdout-gate tolerance. Must be a non-negative float."""
    v = float(value)
    if v < 0 or v != v:  # reject NaN + negatives
        raise ValueError("holdout_tolerance must be a non-negative float")
    set_setting(session, SETTING_HOLDOUT_TOLERANCE, repr(v))


# -- LearnedWeight activation ---------------------------------------------


def activate_learned_version(session: Session, version: int) -> int:
    """Flip is_active=True on rows with ``version`` and False on the rest.

    Returns the number of rows updated to active. Raises if no rows exist
    for that version (a no-op activate would be silent breakage).
    """
    rows = list(
        session.scalars(select(LearnedWeight).where(LearnedWeight.version == version))
    )
    if not rows:
        raise ValueError(f"no LearnedWeight rows exist for version {version!r}")
    n_activated = 0
    for r in session.scalars(select(LearnedWeight)):
        target = r.version == version
        if r.is_active != target:
            r.is_active = target
        if target:
            n_activated += 1
    session.commit()
    return n_activated


# -- Date helpers used widely ----------------------------------------------


def utc_day_window(day: datetime) -> tuple[datetime, datetime]:
    """Return ``[start_of_day, start_of_next_day)`` in UTC for ``day``."""
    day = day.astimezone(timezone.utc) if day.tzinfo else day.replace(tzinfo=timezone.utc)
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)

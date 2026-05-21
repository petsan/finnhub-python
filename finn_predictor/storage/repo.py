"""Repository helpers. All DB access funnels through this module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from finn_predictor.storage.models import (
    AppSetting,
    HistoricalMarketCap,
    LearnedWeight,
    NewsArticle,
    PriceBar,
    Prediction,
    PredictionOutcome,
    RelatedEntity,
    Sector,
    SentimentScore,
    Watchlist,
    WatchlistMember,
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


# -- Market caps ------------------------------------------------------------


def upsert_market_caps(
    session: Session, caps: Iterable[HistoricalMarketCap]
) -> int:
    """Insert market-cap rows, skipping ``(symbol, as_of_date)`` duplicates.

    Mirrors :func:`upsert_price_bars`: dialect-aware insert builder, no-op
    on conflict, returns the number of newly inserted rows.
    """
    inserted = 0
    insert_builder = _dialect_insert(session)
    for cap in caps:
        stmt = (
            insert_builder(HistoricalMarketCap)
            .values(
                symbol=cap.symbol,
                as_of_date=cap.as_of_date,
                market_cap=float(cap.market_cap),
            )
            .on_conflict_do_nothing(index_elements=["symbol", "as_of_date"])
        )
        result = session.execute(stmt)
        inserted += result.rowcount or 0
    session.commit()
    return inserted


def latest_market_caps(
    session: Session,
    symbols: Iterable[str],
    *,
    on_or_before: Optional[datetime] = None,
) -> dict[str, float]:
    """Return ``{symbol: market_cap}`` using the most recent known snapshot.

    Symbols with no cap rows are simply absent from the result — callers
    decide the fallback (typically a uniform 1.0 weight). When
    ``on_or_before`` is given, only snapshots dated at-or-before that
    point are considered, which keeps the historical backtester
    look-ahead-free.
    """
    out: dict[str, float] = {}
    for sym in symbols:
        stmt = (
            select(HistoricalMarketCap)
            .where(HistoricalMarketCap.symbol == sym)
            .order_by(HistoricalMarketCap.as_of_date.desc())
            .limit(1)
        )
        if on_or_before is not None:
            stmt = (
                select(HistoricalMarketCap)
                .where(
                    HistoricalMarketCap.symbol == sym,
                    HistoricalMarketCap.as_of_date <= on_or_before,
                )
                .order_by(HistoricalMarketCap.as_of_date.desc())
                .limit(1)
            )
        row = session.scalars(stmt).first()
        if row is not None and row.market_cap > 0:
            out[sym] = float(row.market_cap)
    return out


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
        # Magnitude columns: only copy over when the caller supplied
        # them. Leaving the existing values in place when the caller
        # passes Nones means a re-run without the magnitude calibration
        # doesn't erase a band that was written during a prior run
        # with the calibration on.
        if prediction.expected_return_p10 is not None:
            existing.expected_return_p10 = prediction.expected_return_p10
        if prediction.expected_return_p50 is not None:
            existing.expected_return_p50 = prediction.expected_return_p50
        if prediction.expected_return_p90 is not None:
            existing.expected_return_p90 = prediction.expected_return_p90
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
    {
        "PEER",
        "SUPPLIER",
        "CUSTOMER",
        "ETF_HOLDING",
        # Affinity additions (PR-4..PR-6):
        # COMPETITOR — auto-seeded from PEER rows that share Finnhub's
        #   ``finnhubIndustry`` with the target. Distinct from PEER so
        #   the operator can curate which peers are *really* competitors
        #   without losing the underlying peer list. Both rows can
        #   coexist for the same (source, related) pair — the unique
        #   constraint is on the (source, related, relationship) triple.
        "COMPETITOR",
        # INSTITUTIONAL_HOLDER — 13-F filers holding the target stock.
        # ``related_symbol`` is the institution name (uppercased, truncated
        # to fit the 64-char column); ``rank`` is the position in Finnhub's
        # response (0 = biggest holder by share count); ``metadata_text``
        # carries the JSON-encoded ownership %, share count, and filing
        # date so callers can sort by % without re-parsing.
        "INSTITUTIONAL_HOLDER",
        # THEME_MEMBER — a ticker that's a member of an
        # :class:`InvestmentTheme`. ``source_symbol`` is the theme code
        # (NOT a ticker — intentional inversion so the "members of theme
        # X" query is one index hit); ``related_symbol`` is the
        # constituent ticker. ``rank`` is the position in Finnhub's
        # response (no semantic meaning beyond response order today).
        "THEME_MEMBER",
    }
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


# -- Streak / flipped analytics for the per-stock table -------------------


@dataclass(frozen=True)
class StreakInfo:
    """Current-label streak + previous label for a target symbol.

    A ``streak`` of N means the latest N consecutive predictions for
    ``symbol`` carry the same ``current_label``. ``previous_label`` is
    the label of the prediction immediately before the streak started
    (i.e. the first row, scanning back from latest, whose label differs
    from ``current_label``); ``None`` when ``symbol`` has only ever
    carried one label. ``flipped`` is True iff
    ``previous_label is not None and previous_label != current_label``
    — which is always the same as ``previous_label is not None``, but
    the explicit field is cheaper to read at the call site.

    "Yesterday's label" semantics: we use the predictions' own ordering
    by ``prediction_date desc``, not literal "yesterday." When ingestion
    misses a day or the operator backfills only weekdays, the previous
    label is the next-most-recent row, not the calendar predecessor.
    Documented here so the UI caption matches.
    """

    symbol: str
    current_label: str
    streak: int                   # ≥ 1 whenever the symbol is present in the dict
    previous_label: Optional[str] = None
    flipped: bool = False


def streaks_for(
    session: Session,
    symbols: Iterable[str],
    *,
    model_version: Optional[str] = None,
) -> dict[str, StreakInfo]:
    """Compute current-label streak + flipped-vs-previous in one SQL pass.

    Returns ``{symbol: StreakInfo}`` only for symbols that have at
    least one prediction row; symbols with no history are omitted (so
    a caller can use ``streaks.get(symbol)`` and fall back cleanly).

    ``model_version`` (optional) filters predictions to a single model
    so a mixed live VADER + back-fitted logreg history doesn't conflate
    streaks across model boundaries. The UI passes the active model's
    version when it has one.

    The query is selective on ``target_symbol`` (indexed) and orders by
    ``(target_symbol, prediction_date desc)`` so we can walk the rows
    in a single pass, grouped by symbol, and compute everything in
    Python — no recursive CTE, no per-symbol round-trip.
    """
    symbol_set = {s for s in symbols if s}
    if not symbol_set:
        return {}

    stmt = (
        select(
            Prediction.target_symbol,
            Prediction.label,
            Prediction.prediction_date,
        )
        .where(Prediction.target_symbol.in_(symbol_set))
        .order_by(Prediction.target_symbol, Prediction.prediction_date.desc())
    )
    if model_version is not None:
        stmt = stmt.where(Prediction.model_version == model_version)

    grouped: dict[str, list[str]] = {}
    for sym, label, _dt in session.execute(stmt):
        grouped.setdefault(sym, []).append(label)

    out: dict[str, StreakInfo] = {}
    for sym, labels in grouped.items():
        # ``grouped`` is built via ``setdefault(...).append(label)`` so
        # every value list is guaranteed non-empty — no defensive
        # ``if not labels`` branch needed.
        current_label = labels[0]
        streak = 0
        previous_label: Optional[str] = None
        for label in labels:
            if previous_label is None and label == current_label:
                streak += 1
                continue
            # First disagreement marks the end of the streak; record
            # the prior label and stop scanning. Anything older isn't
            # useful for the "flipped today vs yesterday" caption.
            previous_label = label
            break
        out[sym] = StreakInfo(
            symbol=sym,
            current_label=current_label,
            streak=streak,
            previous_label=previous_label,
            flipped=previous_label is not None,
        )
    return out


# -- Watchlists -----------------------------------------------------------

# Watchlist name shape: keep human-friendly but pinned down enough that
# the UI dropdown / URL slugging works. Allowed: letters, digits, space,
# underscore, hyphen, dot. 1–64 chars (matches the DB column).
import re as _re

_WATCHLIST_NAME_RE = _re.compile(r"[A-Za-z0-9 _.\-]{1,64}")


class WatchlistError(ValueError):
    """Raised on bad watchlist input (name shape, missing list, etc.).

    A ValueError subclass so callers can catch either type. Carrying its
    own class lets the UI distinguish "the operator made a typo" from
    "unrelated ValueError" without grovelling through the message.
    """


def _normalise_watchlist_name(name: str) -> str:
    """Strip surrounding whitespace and reject the empty / malformed case.

    Returns the cleaned name. Raises :class:`WatchlistError` for empty,
    too-long, or charset-violating inputs. Case is preserved — two
    lists ``Tech`` and ``tech`` are considered the same by the DB's
    case-sensitive UNIQUE index, but we don't normalise to one case
    so the operator sees what they typed.
    """
    if not isinstance(name, str):
        raise WatchlistError("watchlist name must be a string")
    cleaned = name.strip()
    if not cleaned:
        raise WatchlistError("watchlist name must be non-empty")
    if _WATCHLIST_NAME_RE.fullmatch(cleaned) is None:
        raise WatchlistError(
            f"watchlist name {name!r} contains invalid characters; "
            "allowed: letters, digits, space, underscore, hyphen, dot"
        )
    return cleaned


def _validate_watchlist_symbol(symbol: str) -> str:
    """Uppercase + shape-check a symbol via the ingestion-layer validator.

    Centralises the per-ticker validation so a Watchlist member can
    never be created with a symbol that the rest of the pipeline would
    reject downstream. Imported lazily to avoid a top-level dependency
    cycle (``ingestion`` already imports ``storage``).
    """
    from finn_predictor.ingestion.symbols import valid_ticker

    if not isinstance(symbol, str):
        raise WatchlistError("symbol must be a string")
    cleaned = symbol.strip().upper()
    if not valid_ticker(cleaned):
        raise WatchlistError(f"symbol {symbol!r} is not a valid ticker")
    return cleaned


def create_watchlist(
    session: Session,
    *,
    name: str,
    description: Optional[str] = None,
) -> Watchlist:
    """Insert a new watchlist.

    Raises :class:`WatchlistError` if the name is malformed or a list
    with the same name already exists. Returns the persisted instance
    so the caller can read ``id`` / ``created_at`` without re-querying.
    """
    cleaned = _normalise_watchlist_name(name)
    if session.scalar(select(Watchlist).where(Watchlist.name == cleaned)):
        raise WatchlistError(f"watchlist {cleaned!r} already exists")

    now = _utcnow()
    row = Watchlist(
        name=cleaned,
        description=(description.strip() if description else None) or None,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.commit()
    return row


def get_watchlist(session: Session, name: str) -> Optional[Watchlist]:
    """Return the watchlist with ``name``, or None if missing.

    Lenient on input: strips outer whitespace but raises on a malformed
    name (so callers can't paper over typos by getting None back).
    """
    cleaned = _normalise_watchlist_name(name)
    return session.scalar(select(Watchlist).where(Watchlist.name == cleaned))


def list_watchlists(session: Session) -> list[Watchlist]:
    """All watchlists, ordered by name."""
    return list(session.scalars(select(Watchlist).order_by(Watchlist.name)))


def rename_watchlist(
    session: Session, *, old_name: str, new_name: str
) -> Watchlist:
    """Rename an existing watchlist.

    Raises if ``old_name`` is missing or ``new_name`` collides with an
    existing list. No-op (returns the existing row) when ``old_name``
    and ``new_name`` resolve to the same string after normalisation.
    """
    old_clean = _normalise_watchlist_name(old_name)
    new_clean = _normalise_watchlist_name(new_name)

    existing = session.scalar(select(Watchlist).where(Watchlist.name == old_clean))
    if existing is None:
        raise WatchlistError(f"watchlist {old_clean!r} not found")

    if new_clean == old_clean:
        return existing

    clash = session.scalar(select(Watchlist).where(Watchlist.name == new_clean))
    if clash is not None:
        raise WatchlistError(f"watchlist {new_clean!r} already exists")

    existing.name = new_clean
    existing.updated_at = _utcnow()
    session.commit()
    return existing


def update_watchlist_description(
    session: Session, *, name: str, description: Optional[str]
) -> Watchlist:
    """Replace the description (or clear it with ``None`` / empty string)."""
    cleaned = _normalise_watchlist_name(name)
    existing = session.scalar(select(Watchlist).where(Watchlist.name == cleaned))
    if existing is None:
        raise WatchlistError(f"watchlist {cleaned!r} not found")
    new_desc = (description.strip() if description else None) or None
    existing.description = new_desc
    existing.updated_at = _utcnow()
    session.commit()
    return existing


def delete_watchlist(session: Session, *, name: str) -> bool:
    """Delete a watchlist and all its members (cascade).

    Returns True when a row was deleted, False when no such list
    existed — lets the UI's "Delete <name>" button be idempotent
    without raising on a double-click.
    """
    cleaned = _normalise_watchlist_name(name)
    existing = session.scalar(select(Watchlist).where(Watchlist.name == cleaned))
    if existing is None:
        return False
    session.delete(existing)
    session.commit()
    return True


def add_to_watchlist(
    session: Session,
    *,
    name: str,
    symbol: str,
    notes: Optional[str] = None,
) -> WatchlistMember:
    """Add ``symbol`` to ``name``. Idempotent — re-adding refreshes notes.

    The symbol passes through :func:`_validate_watchlist_symbol` so
    nothing the upstream Finnhub client would reject can land in the
    DB. Re-adding an existing (list, symbol) pair updates only
    ``notes`` and bumps the parent list's ``updated_at`` — useful when
    the operator annotates an existing pick.
    """
    cleaned_name = _normalise_watchlist_name(name)
    cleaned_symbol = _validate_watchlist_symbol(symbol)

    parent = session.scalar(select(Watchlist).where(Watchlist.name == cleaned_name))
    if parent is None:
        raise WatchlistError(f"watchlist {cleaned_name!r} not found")

    existing = session.scalar(
        select(WatchlistMember).where(
            WatchlistMember.watchlist_id == parent.id,
            WatchlistMember.symbol == cleaned_symbol,
        )
    )
    cleaned_notes = (notes.strip() if notes else None) or None
    now = _utcnow()
    if existing is not None:
        existing.notes = cleaned_notes
        parent.updated_at = now
        session.commit()
        return existing

    row = WatchlistMember(
        watchlist_id=parent.id,
        symbol=cleaned_symbol,
        notes=cleaned_notes,
        added_at=now,
    )
    session.add(row)
    parent.updated_at = now
    session.commit()
    return row


def remove_from_watchlist(
    session: Session, *, name: str, symbol: str
) -> bool:
    """Remove ``symbol`` from ``name``. Returns True iff a row was removed."""
    cleaned_name = _normalise_watchlist_name(name)
    cleaned_symbol = _validate_watchlist_symbol(symbol)

    parent = session.scalar(select(Watchlist).where(Watchlist.name == cleaned_name))
    if parent is None:
        raise WatchlistError(f"watchlist {cleaned_name!r} not found")

    member = session.scalar(
        select(WatchlistMember).where(
            WatchlistMember.watchlist_id == parent.id,
            WatchlistMember.symbol == cleaned_symbol,
        )
    )
    if member is None:
        return False
    session.delete(member)
    parent.updated_at = _utcnow()
    session.commit()
    return True


def watchlist_members(session: Session, *, name: str) -> list[WatchlistMember]:
    """All members of ``name``, ordered by symbol."""
    cleaned = _normalise_watchlist_name(name)
    parent = session.scalar(select(Watchlist).where(Watchlist.name == cleaned))
    if parent is None:
        raise WatchlistError(f"watchlist {cleaned!r} not found")
    return list(
        session.scalars(
            select(WatchlistMember)
            .where(WatchlistMember.watchlist_id == parent.id)
            .order_by(WatchlistMember.symbol)
        )
    )


def watchlist_symbols(
    session: Session, *, name: Optional[str] = None
) -> list[str]:
    """Symbols belonging to one or all watchlists.

    When ``name`` is None, returns the deduped union across every
    watchlist, sorted alphabetically — the natural default for
    "ingest everything I've ever bookmarked." When ``name`` is set,
    returns just that list's symbols in sort order. Missing list
    raises :class:`WatchlistError` so the caller doesn't silently
    treat a typo as an empty universe.
    """
    if name is None:
        rows = session.scalars(select(WatchlistMember.symbol).distinct())
        return sorted({s for s in rows})

    cleaned = _normalise_watchlist_name(name)
    parent = session.scalar(select(Watchlist).where(Watchlist.name == cleaned))
    if parent is None:
        raise WatchlistError(f"watchlist {cleaned!r} not found")
    rows = session.scalars(
        select(WatchlistMember.symbol).where(WatchlistMember.watchlist_id == parent.id)
    )
    return sorted({s for s in rows})


def replace_watchlist_symbols(
    session: Session, *, name: str, symbols: Iterable[str]
) -> int:
    """Atomically replace ``name``'s members with ``symbols``.

    Designed for the sidebar's "paste tickers + save" flow — the UI
    parses the textbox, validates via :mod:`finn_predictor.ingestion.symbols`,
    then hands the result to this helper which wipes the old set and
    inserts the new one in a single transaction. Returns the number of
    symbols in the resulting list.

    Each symbol is re-validated here so a caller that bypassed the UI
    parser still can't corrupt the table.
    """
    cleaned_name = _normalise_watchlist_name(name)
    parent = session.scalar(select(Watchlist).where(Watchlist.name == cleaned_name))
    if parent is None:
        raise WatchlistError(f"watchlist {cleaned_name!r} not found")

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = _validate_watchlist_symbol(raw)
        if sym in seen:
            continue
        seen.add(sym)
        cleaned.append(sym)

    now = _utcnow()
    # Drop the old set first. We have to ``flush`` between the deletes
    # and the new inserts: SQLAlchemy's unit-of-work otherwise batches
    # INSERTs ahead of DELETEs on the same table, which trips the
    # ``(watchlist_id, symbol)`` UNIQUE constraint when the same symbol
    # appears in both the old and the new set.
    existing_rows = list(
        session.scalars(
            select(WatchlistMember).where(WatchlistMember.watchlist_id == parent.id)
        )
    )
    for row in existing_rows:
        session.delete(row)
    session.flush()

    for sym in cleaned:
        session.add(
            WatchlistMember(
                watchlist_id=parent.id,
                symbol=sym,
                notes=None,
                added_at=now,
            )
        )
    parent.updated_at = now
    session.commit()
    return len(cleaned)


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

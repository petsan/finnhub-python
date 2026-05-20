"""Streamlit dashboard entry point.

Run with::

    streamlit run finn_predictor/ui/app.py

The UI itself never calls Finnhub directly — it renders whatever the
ingestion job has persisted. Users supply a Finnhub API key through a
sidebar input; that key lives **only** in :data:`streamlit.session_state`
(server-side, in-memory, cleared on tab/server close) and is never written
to disk or to the database.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

import altair as alt
import pandas as pd
import streamlit as st
from finnhub import Client as FinnhubClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.config import load_settings
from finn_predictor.security import (
    auth_enabled,
    current_password_hash,
    verify_password,
)
from finn_predictor.ingestion.backfill import BackfillResult, backfill_many
from finn_predictor.ingestion.client import (
    FinnhubGateway,
    IngestionError,
    RateLimiter,
    scrub_token,
)
from finn_predictor.ingestion.jobs import run_daily_ingest, score_pending_articles
from finn_predictor.ingestion.prices_yf import (
    PriceBackfillResult,
    backfill_prices_yf,
)
from finn_predictor.predictor.backtest import score_outcomes
from finn_predictor.predictor.explain import (
    ArticleContribution,
    article_contributions,
    explain_prediction,
)
from finn_predictor.predictor.focus import (
    CompanyFocus,
    EventFocus,
    RefreshResult,
    RelatedPrediction,
    SectorFocus,
    compose_company_focus,
    compose_event_focus,
    compose_sector_focus,
    refresh_company_relationships,
    refresh_sector_constituents,
)
from finn_predictor.learning import (
    LearnedConfig,
    TrainingReport,
    active_weights,
    train_weights,
)
from finn_predictor.learning.config import (
    DIM_HALF_LIFE,
    DIM_MIN_SIGMA,
    DIM_SOURCE_WEIGHT,
    DIM_THRESHOLD,
)
from finn_predictor.learning.train import (
    MIN_TRADES_FOR_TRAINING,
    NotEnoughDataError,
)
from finn_predictor.storage.models import LearnedWeight
from finn_predictor.storage.repo import (
    POLICY_AUTO,
    POLICY_MANUAL,
    activate_learned_version,
    get_activation_policy,
    get_holdout_tolerance,
    set_activation_policy,
    set_holdout_tolerance,
)
from finn_predictor.predictor.stocks import retroactive_predict_many
from finn_predictor.predictor.trades import (
    PerformanceSummary,
    TradeRecord,
    cumulative_pnl_series,
    hit_rate_by_label,
    hit_rate_by_target_kind,
    hypothetical_trades,
    performance_summary,
    rolling_hit_rate,
    trades_dataframe,
)
from finn_predictor.sentiment import resolve_active_scorer, warn_if_scorer_mismatch
from finn_predictor.storage import create_engine_and_session, init_db
from finn_predictor.storage.models import (
    NewsArticle,
    Prediction,
    PredictionOutcome,
    Sector,
    SentimentScore,
)
from finn_predictor.storage.repo import all_sectors, predictions_for, price_bars
from finn_predictor.storage.clustering import resolve_active_clusterer
from finn_predictor.storage.stories import earliest_story_times  # noqa: F401  (kept for tests / external imports)
from finn_predictor.storage.symbol_names import expand_symbol, expand_symbol_short


API_KEY_SESSION_KEY = "finnhub_api_key"
AUTHED_SESSION_KEY = "_finn_authenticated"

# localStorage key the sidebar writes the Finnhub token under when the
# user submits one. Persisted client-side only — the token still never
# touches the server's filesystem or the SQLite DB. Cleared atomically
# by the sidebar's *Clear key* button via _clear_browser_api_key().
BROWSER_STORAGE_API_KEY = "finn_predictor_finnhub_api_key"
_BROWSER_STORAGE_HYDRATED_FLAG = "_finn_local_storage_hydrated"
_BROWSER_STORAGE_LAST_WRITTEN = "_finn_local_storage_last_written"


def _get_local_storage():  # pragma: no cover - depends on Streamlit runtime
    """Return a streamlit_local_storage handle, or ``None`` outside Streamlit.

    The package's ``LocalStorage()`` constructor polls
    ``st.session_state`` until the frontend custom component posts
    back. That works under ``streamlit run`` but **hangs forever**
    under ``streamlit.testing.v1.AppTest`` (which never executes the
    component JS) and would block any other non-interactive runtime
    the same way.

    Two escape hatches keep that out of our way:

    1. ``FINN_PREDICTOR_DISABLE_LOCAL_STORAGE=1`` — explicit opt-out
       used by the smoke-test fixture and by anyone who wants the
       legacy "session-only key" behaviour back without touching the
       code.
    2. ``ImportError`` — the package isn't installed at all (older
       deploys / minimal envs).

    On either, this returns ``None`` and the sidebar runs in
    no-persistence mode without re-raising. The text-input + Clear
    button still work; only the localStorage bridge is missing.
    """
    if os.environ.get("FINN_PREDICTOR_DISABLE_LOCAL_STORAGE", "").strip() in {"1", "true", "yes"}:
        return None
    try:
        from streamlit_local_storage import LocalStorage
    except ImportError:
        return None
    try:
        return LocalStorage()
    except Exception:
        return None


def _hydrate_api_key_from_browser() -> None:  # pragma: no cover - Streamlit UI
    """Pre-fill ``st.session_state[API_KEY_SESSION_KEY]`` from localStorage.

    Runs at most once per browser session (gated by a session_state
    flag) so the user can still clear the field mid-session without
    immediately repopulating it from the cache. Silent no-op when
    localStorage isn't reachable.
    """
    if st.session_state.get(_BROWSER_STORAGE_HYDRATED_FLAG):
        return
    if st.session_state.get(API_KEY_SESSION_KEY):
        # Already populated for this session (e.g. user typed it before
        # the hydrate ran). Nothing to do; flag and move on.
        st.session_state[_BROWSER_STORAGE_HYDRATED_FLAG] = True
        return

    ls = _get_local_storage()
    if ls is None:
        st.session_state[_BROWSER_STORAGE_HYDRATED_FLAG] = True
        return

    try:
        stored = ls.getItem(BROWSER_STORAGE_API_KEY)
    except Exception:
        stored = None
    if isinstance(stored, str) and stored.strip():
        st.session_state[API_KEY_SESSION_KEY] = stored.strip()
        st.session_state[_BROWSER_STORAGE_LAST_WRITTEN] = stored.strip()
    st.session_state[_BROWSER_STORAGE_HYDRATED_FLAG] = True


def _persist_api_key_to_browser(api_key: str) -> None:  # pragma: no cover - Streamlit UI
    """Write the current key to localStorage if it changed since last write.

    Idempotent: each render doesn't re-call setItem unless the key
    actually changed. Storing the last-written value in session_state
    avoids spamming the component bridge on every rerun.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return
    if st.session_state.get(_BROWSER_STORAGE_LAST_WRITTEN) == api_key:
        return
    ls = _get_local_storage()
    if ls is None:
        return
    try:
        ls.setItem(BROWSER_STORAGE_API_KEY, api_key)
        st.session_state[_BROWSER_STORAGE_LAST_WRITTEN] = api_key
    except Exception:
        # localStorage write failed — leave session_state untouched so
        # the next render will retry.
        pass


def _clear_browser_api_key() -> None:  # pragma: no cover - Streamlit UI
    """Remove the persisted key from localStorage.

    Called by the *Clear key* button so the wipe is atomic across
    server session_state and browser cache. Survives the case where
    localStorage isn't reachable (no-op).
    """
    st.session_state.pop(_BROWSER_STORAGE_LAST_WRITTEN, None)
    ls = _get_local_storage()
    if ls is None:
        return
    try:
        ls.deleteItem(BROWSER_STORAGE_API_KEY)
    except Exception:
        pass


def _enforce_auth_gate() -> bool:  # pragma: no cover - Streamlit UI
    """Show a password prompt until the session is authenticated.

    Returns True iff the rest of the page should render. When auth is
    disabled (no ``FINN_PREDICTOR_PASSWORD_HASH`` env var) this is a
    no-op that returns True immediately — preserves the default
    localhost-dev experience.
    """
    if not auth_enabled():
        return True
    if st.session_state.get(AUTHED_SESSION_KEY):
        return True

    st.set_page_config(page_title="Finn-Predictor — locked")
    st.title("🔒 Finn-Predictor")
    st.caption(
        "This instance has a password gate enabled "
        "(`FINN_PREDICTOR_PASSWORD_HASH` is set)."
    )
    pw = st.text_input(
        "Password",
        type="password",
        key="_finn_password_prompt",
        placeholder="enter the deploy password",
    )
    if st.button("Unlock", type="primary", disabled=not pw):
        if verify_password(pw, current_password_hash()):
            st.session_state[AUTHED_SESSION_KEY] = True
            # Discard the typed password from session_state so it
            # doesn't sit in memory between reruns.
            st.session_state.pop("_finn_password_prompt", None)
            st.rerun()
        else:
            st.error("Wrong password.")
    st.caption(
        "Forgot the password? Stop the server, unset "
        "`FINN_PREDICTOR_PASSWORD_HASH` (or set a new one with "
        "`python -m finn_predictor.cli hash-password`), and restart."
    )
    return False


# -- Data helpers (pure, testable) ----------------------------------------


def latest_market_prediction(session: Session, symbol: str = "^GSPC") -> Prediction | None:
    """Most recent prediction for the whole-market target."""
    stmt = (
        select(Prediction)
        .where(Prediction.target_symbol == symbol)
        .order_by(Prediction.prediction_date.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def latest_predictions(session: Session) -> list[Prediction]:
    """One latest Prediction per target_symbol — across market + every sector + every stock.

    Sorted with ``^GSPC`` first (the headline call) then alphabetical by
    target symbol. Returns an empty list when no predictions exist.
    """
    distinct_symbols = session.scalars(
        select(Prediction.target_symbol).distinct()
    ).all()
    out: list[Prediction] = []
    for sym in distinct_symbols:
        latest = session.scalar(
            select(Prediction)
            .where(Prediction.target_symbol == sym)
            .order_by(Prediction.prediction_date.desc())
            .limit(1)
        )
        if latest is not None:
            out.append(latest)
    # Market call first, then sectors alphabetically.
    out.sort(key=lambda p: (p.target_symbol != "^GSPC", p.target_symbol))
    return out


def format_expected_move(pred: Prediction) -> str | None:
    """Render the magnitude band as ``-1.2% to +1.5% (median +0.4%)``.

    Returns ``None`` when the band hasn't been populated (any of the
    three columns null) — the caller decides whether to skip rendering
    the row entirely. Pure / testable; the Streamlit wrapper below is
    the side-effecting display helper.
    """
    p10 = pred.expected_return_p10
    p50 = pred.expected_return_p50
    p90 = pred.expected_return_p90
    if p10 is None or p50 is None or p90 is None:
        return None
    return (
        f"{p10 * 100:+.2f}% to {p90 * 100:+.2f}% "
        f"(median {p50 * 100:+.2f}%)"
    )


def _render_expected_move(pred: Prediction) -> None:  # pragma: no cover - Streamlit
    """Side-effecting wrapper: shows the expected-move line if present."""
    band = format_expected_move(pred)
    if band is None:
        return
    st.caption(
        f"**Expected next-bar move (10th–90th pctile):** {band}"
    )


def partition_predictions(
    session: Session, preds: Iterable[Prediction]
) -> tuple[Prediction | None, list[Prediction], list[Prediction]]:
    """Split ``preds`` into (market, sectors, stocks).

    Uses the Sector table to recognise sector-ETF targets. Anything
    that isn't ``^GSPC`` and isn't a registered sector ETF is treated
    as an individual stock.
    """
    sector_etfs = {s.etf_symbol for s in all_sectors(session)}
    market: Prediction | None = None
    sectors: list[Prediction] = []
    stocks: list[Prediction] = []
    for p in preds:
        if p.target_symbol == "^GSPC":
            market = p
        elif p.target_symbol in sector_etfs:
            sectors.append(p)
        else:
            stocks.append(p)
    return market, sectors, stocks


def build_market_price_chart(
    bars: Iterable["object"], *, symbol: str = "^GSPC", days: int = 30
) -> alt.Chart | None:
    """Altair line chart of recent closes for ``symbol``.

    Pan + mouse-wheel zoom come from ``.interactive()`` — the default
    Altair / Vega-Lite interaction set fits the "look at recent
    history" use case without bringing in a brush-selection widget.

    Returns ``None`` when there are no bars to draw, so the caller
    can hide the chart entirely instead of rendering an empty axis
    (which Altair otherwise still renders with placeholder ticks).
    """
    rows = [
        {"date": b.trade_date, "close": float(b.close), "symbol": symbol}
        for b in bars
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("date").tail(days)
    if df.empty:
        return None
    # Keep the y-axis tight: pad ±2% around the visible window's range
    # so daily candles don't get crushed when the close range is
    # narrow but the absolute level is high.
    y_lo = float(df["close"].min())
    y_hi = float(df["close"].max())
    pad = max((y_hi - y_lo) * 0.05, abs(y_hi) * 0.005)
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "close:Q",
                title=f"{symbol} close",
                scale=alt.Scale(domain=[y_lo - pad, y_hi + pad]),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("close:Q", title="Close", format=",.2f"),
            ],
        )
        .properties(height=240)
        .interactive()  # mouse-wheel zoom, click-drag pan
    )


def group_stock_predictions_by_sector(
    session: Session, stock_preds: Iterable[Prediction]
) -> tuple[list[tuple["object", list[Prediction]]], list[Prediction]]:
    """Bucket ``stock_preds`` by their curated sector.

    Returns ``([(Sector, [pred, …]), …], [unmapped_pred, …])``:

    * The first element is sectors that have at least one mapped
      stock, ordered by ``Sector.code`` so the UI renders in a
      stable order across reruns. Each sector's predictions inside
      keep the caller's input order.
    * The second element is per-stock predictions whose
      ``target_symbol`` isn't in the curated map — surfaced as
      an "Other / unmapped" section by the UI so the user notices
      and can either add the ticker to the curated map or accept
      that the stock won't roll up into a sector prediction.

    The Sector objects come from the live DB so the UI gets the
    user's localised ``name`` and the canonical ``etf_symbol``,
    not a hard-coded label.
    """
    sectors_by_code = {s.code: s for s in all_sectors(session)}
    from finn_predictor.storage.sector_membership import sector_for_ticker

    by_code: dict[str, list[Prediction]] = {}
    unmapped: list[Prediction] = []
    for p in stock_preds:
        code = sector_for_ticker(p.target_symbol)
        if code is None or code not in sectors_by_code:
            unmapped.append(p)
            continue
        by_code.setdefault(code, []).append(p)
    grouped = [
        (sectors_by_code[code], by_code[code])
        for code in sorted(by_code.keys())
    ]
    return grouped, unmapped


def stock_predictions_table(
    session: Session, stock_preds: Iterable[Prediction]
) -> pd.DataFrame:
    """DataFrame of per-stock predictions sorted by descending confidence."""
    rows = []
    for p in stock_preds:
        rows.append(
            {
                "Company": expand_symbol_short(session, p.target_symbol),
                "Ticker": p.target_symbol,
                "Call": p.label,
                "Confidence": round(p.confidence, 2),
                "Articles": p.article_count,
                "Sentiment": round(p.sentiment_index, 3),
                "As of": p.prediction_date,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["Company", "Ticker", "Call", "Confidence", "Articles", "Sentiment", "As of"]
        )
    df = pd.DataFrame(rows)
    # Sort by absolute confidence descending so the strongest calls float up.
    return df.sort_values(["Confidence", "Articles"], ascending=False).reset_index(drop=True)


def headlines_from_contributions(
    contributions: Iterable[ArticleContribution],
    *,
    limit: int = 10,
) -> list[dict]:
    """Convert contribution objects to UI-row dicts (top ``limit`` by |contribution|)."""
    rows: list[dict] = []
    for c in list(contributions)[:limit]:
        a = c.article
        rows.append(
            {
                "published_at": a.published_at,
                "headline": a.headline,
                "source": a.source,
                "symbol": a.symbol or "*",
                "url": a.url or "",
                "sentiment": c.score,
                "contribution": c.contribution,
                "supports_call": c.supports_call,
            }
        )
    return rows


# Articles whose absolute VADER sentiment is at or below this threshold
# are "neutral" — the model saw them but didn't extract enough polarity
# to move the weighted index. Matches FLAT_SUPPORT_BAND in
# predictor/explain.py: a contribution of that size doesn't tip a FLAT
# call into UP or DOWN either, so the cutoff is internally consistent.
NEUTRAL_SENTIMENT_THRESHOLD = 0.05


def neutral_headlines_from_contributions(
    contributions: Iterable[ArticleContribution],
    *,
    sentiment_threshold: float = NEUTRAL_SENTIMENT_THRESHOLD,
    limit: int = 10,
) -> list[dict]:
    """Articles in the prediction's window that didn't move the needle.

    Filters the contribution list down to rows whose absolute sentiment
    score sits at or below ``sentiment_threshold`` — i.e. the scorer
    saw them but rated them neutral. Sorted by published time
    descending so the freshest neutral headlines surface first; the
    main "Recent headlines" section already orders by
    |contribution| desc, so this complement orders by recency.

    The point is editorial: a market call backed by 30 scored articles
    will typically have ~half scored neutrally. Hiding those entirely
    overstates how decisive the model thought the day was. Surfacing
    them in their own section keeps the reader honest about base
    rates while keeping the headline contribution view free of noise.
    """
    threshold = abs(float(sentiment_threshold))
    rows: list[dict] = []
    for c in contributions:
        if abs(c.score) > threshold:
            continue
        a = c.article
        rows.append(
            {
                "published_at": a.published_at,
                "headline": a.headline,
                "source": a.source,
                "symbol": a.symbol or "*",
                "url": a.url or "",
                "sentiment": c.score,
            }
        )
    # Recency wins for the neutral pile — they have nothing else to
    # rank by, and the user is most likely scanning for "did anything
    # interesting happen that the model under-weighted."
    rows.sort(key=lambda r: r["published_at"], reverse=True)
    return rows[:limit]


def recent_headlines(
    session: Session, *, limit: int = 10, model_version: str | None = None
) -> list[dict]:
    """Most recent headlines paired with their sentiment score (or NaN)."""
    stmt = (
        select(NewsArticle)
        .order_by(NewsArticle.published_at.desc())
        .limit(limit)
    )
    articles = list(session.scalars(stmt))
    if not articles:
        return []

    score_map: dict[int, float] = {}
    if model_version is not None:
        rows = session.execute(
            select(SentimentScore.article_id, SentimentScore.score).where(
                SentimentScore.article_id.in_([a.id for a in articles]),
                SentimentScore.model_version == model_version,
            )
        ).all()
        score_map = {aid: float(sc) for aid, sc in rows}

    return [
        {
            "published_at": a.published_at,
            "headline": a.headline,
            "source": a.source,
            "symbol": a.symbol or "*",
            "url": a.url or "",
            "sentiment": score_map.get(a.id, float("nan")),
        }
        for a in articles
    ]


def prediction_history(
    session: Session,
    target_symbol: str,
    *,
    days: int = 60,
) -> pd.DataFrame:
    """A DataFrame with one row per prediction + outcome (if known)."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    preds = predictions_for(session, target_symbol, since=since)
    if not preds:
        return pd.DataFrame(
            columns=[
                "prediction_date",
                "label",
                "confidence",
                "sentiment_index",
                "realised_return",
                "hit",
            ]
        )

    outcome_map: dict[int, PredictionOutcome] = {
        o.prediction_id: o
        for o in session.scalars(
            select(PredictionOutcome).where(
                PredictionOutcome.prediction_id.in_([p.id for p in preds])
            )
        )
    }

    return pd.DataFrame(
        [
            {
                "prediction_date": p.prediction_date,
                "label": p.label,
                "confidence": p.confidence,
                "sentiment_index": p.sentiment_index,
                "realised_return": outcome_map[p.id].realised_return
                if p.id in outcome_map
                else None,
                "hit": outcome_map[p.id].hit if p.id in outcome_map else None,
            }
            for p in preds
        ]
    )


# Articles within this many seconds of the earliest cluster member are
# considered "the same wire flash" — for those the UI suppresses a
# separate "first reported" annotation because it would just echo the
# article's own publish time.
_FIRST_SEEN_DELTA_SECONDS = 300


def attach_first_seen(
    session: Session, rows: list[dict]
) -> list[dict]:
    """Mutate ``rows`` in place to add a ``first_seen_at`` datetime per row.

    The value is the earliest published_at of any article with the same
    :func:`finn_predictor.storage.stories.story_key`. ``None`` when no
    match was found in the lookback window.
    """
    if not rows:
        return rows
    headlines = [r.get("headline", "") for r in rows]
    # Pluggable: prefix matcher by default; FINN_PREDICTOR_CLUSTERER=embedding
    # switches to the sentence-transformers cosine clusterer.
    clusterer = resolve_active_clusterer()
    earliest = clusterer.earliest_times(session, headlines)
    for r in rows:
        r["first_seen_at"] = earliest.get(r.get("headline", ""))
    return rows


def contribution_chart_data(
    contributions: Sequence[ArticleContribution],
    *,
    session: Session | None = None,
) -> pd.DataFrame:
    """Build a DataFrame ready for an Altair contribution chart.

    Columns:

    * ``x`` — integer rank 0..N-1, ordered most-negative → most-positive
      (i.e. left-to-right along the chart's x-axis goes from −1 toward
      +1 in contribution space).
    * ``contribution`` — the signed contribution (y-axis value).
    * ``sentiment`` — raw score for the tooltip.
    * ``headline``, ``source``, ``symbol`` — tooltip text.
    * ``company`` — expanded ticker (if a session is provided).
    * ``direction`` — ``"positive"`` / ``"negative"`` / ``"neutral"``,
      used by the chart's colour scale.
    * ``supports_call`` — for the tooltip text only.
    """
    if not contributions:
        return pd.DataFrame(
            columns=[
                "x",
                "contribution",
                "sentiment",
                "headline",
                "source",
                "symbol",
                "company",
                "published_at",
                "first_seen_at",
                "direction",
                "supports_call",
            ]
        )

    # Sort ascending by contribution so x-axis runs left=most-negative,
    # right=most-positive (matches the spec).
    ordered = sorted(contributions, key=lambda c: c.contribution)

    # One-shot lookup of cluster-earliest timestamps for every headline in
    # the dataset. We only run the query when a session is supplied —
    # tests that pass session=None still get a usable frame.
    first_seen_map: dict[str, datetime | None] = {}
    if session is not None:
        first_seen_map = resolve_active_clusterer().earliest_times(
            session, [c.article.headline for c in ordered]
        )

    rows: list[dict[str, object]] = []
    for i, c in enumerate(ordered):
        if c.contribution > 0:
            direction = "positive"
        elif c.contribution < 0:
            direction = "negative"
        else:
            direction = "neutral"
        sym = c.article.symbol or ""
        rows.append(
            {
                "x": i,
                "contribution": c.contribution,
                "sentiment": c.score,
                "headline": c.article.headline,
                "source": c.article.source or "",
                "symbol": sym,
                "company": expand_symbol(session, sym) if (session and sym) else "",
                "published_at": c.article.published_at,
                "first_seen_at": first_seen_map.get(c.article.headline),
                "direction": direction,
                "supports_call": c.supports_call,
            }
        )
    return pd.DataFrame(rows)


def build_contribution_chart(df: pd.DataFrame, *, title: str = "") -> alt.Chart:
    """Divergent vertical-bar chart of per-article contributions.

    Positive contributions extend upward from the 0-line, negative ones
    downward. Bars are coloured by sign (green/red, with a neutral grey
    for FLAT-band contributions). Tooltips show headline + source +
    sentiment + signed contribution.
    """
    color_scale = alt.Scale(
        domain=["negative", "neutral", "positive"],
        range=["#d62728", "#9aa0a6", "#2ca02c"],
    )
    return (
        alt.Chart(df)
        .mark_bar(size=14)
        .encode(
            x=alt.X(
                "x:O",
                axis=alt.Axis(labels=False, ticks=False, title="articles ordered −1 → 0 → +1"),
                sort=None,
            ),
            y=alt.Y(
                "contribution:Q",
                axis=alt.Axis(title="signed contribution"),
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "direction:N",
                scale=color_scale,
                legend=alt.Legend(title="direction"),
            ),
            tooltip=[
                alt.Tooltip("headline:N", title="Headline"),
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("symbol:N", title="Ticker"),
                alt.Tooltip("company:N", title="Company"),
                alt.Tooltip("published_at:T", title="Published"),
                alt.Tooltip("first_seen_at:T", title="First reported"),
                alt.Tooltip("sentiment:Q", title="Sentiment", format="+.3f"),
                alt.Tooltip("contribution:Q", title="Contribution", format="+.4f"),
                alt.Tooltip("supports_call:N", title="Supports call"),
            ],
        )
        .properties(height=260, title=title)
    )


def _escape_markdown(text: str) -> str:
    """Escape characters that would break a Markdown link's display text.

    Headlines occasionally contain ``[``, ``]``, ``*``, or backticks. We
    don't try to fully escape Markdown — just enough that a link like
    ``[<text>](<url>)`` always renders the headline verbatim.
    """
    return (
        text.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("`", "\\`")
    )


def _format_headline_markdown(row: dict) -> str:
    """One Markdown line per headline; clickable if the URL is present.

    When the row carries a ``contribution`` field (set by
    :func:`headlines_from_contributions`), the line also shows that signed
    number and a leading dot indicating whether the article supported the
    Call (🟢) or opposed it (🔴). When neither field is present we render
    the legacy headline-only layout.
    """
    headline = _escape_markdown(str(row.get("headline") or "").strip()) or "(no title)"
    url = (row.get("url") or "").strip()
    # Only link when the URL looks remotely usable. Anything else is plain text.
    title_md = f"[{headline}]({url})" if url.startswith(("http://", "https://")) else f"**{headline}**"

    marker = ""
    supports = row.get("supports_call")
    if supports is True:
        marker = "🟢 "
    elif supports is False:
        marker = "🔴 "

    bits = [title_md]
    source = (row.get("source") or "").strip()
    if source:
        bits.append(f"*{_escape_markdown(source)}*")
    symbol = (row.get("symbol") or "").strip()
    if symbol and symbol != "*":
        company = (row.get("company") or "").strip()
        if company and company != symbol:
            # "Apple Inc. (`AAPL`)" — company name spelled out, ticker in code font.
            bits.append(f"{_escape_markdown(company)} (`{symbol}`)")
        else:
            bits.append(f"`{symbol}`")
    published = row.get("published_at")
    if published is not None:
        bits.append(published.strftime("%Y-%m-%d %H:%M UTC"))
    first_seen = row.get("first_seen_at")
    if first_seen is not None and published is not None:
        # Only annotate when the story was reported earlier elsewhere by
        # more than a wire-flash window — same-minute republishes would
        # just clutter the line.
        try:
            delta = (published - first_seen).total_seconds()
        except TypeError:
            # Mixed naive/aware datetimes from SQLite roundtrip — coerce
            # both to naive UTC for the diff.
            p = published.replace(tzinfo=None) if published.tzinfo else published
            f = first_seen.replace(tzinfo=None) if first_seen.tzinfo else first_seen
            delta = (p - f).total_seconds()
        if delta > _FIRST_SEEN_DELTA_SECONDS:
            bits.append(
                f"first reported {first_seen.strftime('%Y-%m-%d %H:%M UTC')}"
            )
    sentiment = row.get("sentiment")
    # NaN sentiment shows as "—"; otherwise show signed score.
    if isinstance(sentiment, float) and sentiment == sentiment:  # NaN check
        bits.append(f"sentiment {sentiment:+.2f}")
    else:
        bits.append("sentiment —")
    contribution = row.get("contribution")
    if isinstance(contribution, (int, float)) and contribution == contribution:
        bits.append(f"contrib {contribution:+.3f}")
    return marker + " · ".join(bits)


def sector_grid(session: Session, sectors: Iterable[Sector]) -> pd.DataFrame:
    """Latest prediction per sector, returned as a one-row-per-sector frame."""
    rows: list[dict] = []
    for sector in sectors:
        latest = latest_market_prediction(session, symbol=sector.etf_symbol)
        rows.append(
            {
                "sector": sector.name,
                "etf": sector.etf_symbol,
                "label": latest.label if latest else "—",
                "confidence": latest.confidence if latest else None,
                "sentiment_index": latest.sentiment_index if latest else None,
                "articles": latest.article_count if latest else 0,
            }
        )
    return pd.DataFrame(rows)


# -- Ingestion helper (pure; testable without Streamlit) ------------------


def run_ingestion_with_key(
    session: Session,
    *,
    api_key: str,
    rate_limit_per_minute: int = 55,
    company_symbols: Iterable[str] = (),
) -> dict[str, int]:
    """Construct a Finnhub gateway with the given key and run one ingestion.

    The key is held only inside the locally-scoped :class:`FinnhubClient` for
    the duration of this call. It is never persisted, logged, or returned in
    the result dict.

    Any exception (including transport-level ``requests`` errors whose
    messages embed the full URL with the token) is re-raised as
    :class:`IngestionError` with the token scrubbed from the message. Use
    ``from None`` to suppress the original chain so default tracebacks don't
    re-leak the URL via ``__cause__``.
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")

    client = FinnhubClient(api_key=api_key)
    # Bypass HTTPS_PROXY / HTTP_PROXY env vars so an intercepting dev proxy
    # (Burp, mitmproxy, Charles, ...) doesn't break TLS verification.
    # The proxy serves its own self-signed cert which certifi rejects.
    # The user opted into this behaviour explicitly during deploy setup.
    try:
        client._session.trust_env = False
    except AttributeError:  # pragma: no cover - upstream session always exists today
        pass
    try:
        gateway = FinnhubGateway(
            client=client,
            rate_limiter=RateLimiter(rate_limit_per_minute),
        )
        try:
            scorer = resolve_active_scorer()
            # One-shot WARNING if the env-configured scorer no longer
            # matches the model_version on the most recent prediction.
            warn_if_scorer_mismatch(session, active_scorer=scorer)
            counts = run_daily_ingest(
                session=session,
                gateway=gateway,
                scorer=scorer,
                company_symbols=list(company_symbols),
            )
        except IngestionError:
            # Already scrubbed by the gateway — let it propagate as-is.
            raise
        except Exception as exc:
            # Belt-and-braces: anything the gateway missed gets scrubbed here.
            raise IngestionError(scrub_token(str(exc), api_key)) from None
    finally:
        # Explicitly close the requests.Session so the key-bearing connection
        # pool doesn't sit around in memory longer than needed.
        client.close()
    return counts


def run_price_backfill(
    session: Session,
    *,
    days: int = 90,
    history_fn=None,
) -> dict[str, object]:
    """Backfill yfinance prices for every prediction target + score outcomes.

    Pulls daily OHLC for ``^GSPC`` and every distinct ``target_symbol``
    already in the ``predictions`` table, then runs the backtester so
    ``prediction_outcomes`` populates for any predictions whose next
    bar is now available.

    No API key needed — yfinance hits Yahoo's public endpoints. The
    ``history_fn`` injection point is for tests; production callers
    leave it None.

    Returns ``{"backfill": PriceBackfillResult, "scored": BacktestReport}``.
    """
    targets = set(
        s for (s,) in session.execute(
            select(Prediction.target_symbol).distinct()
        ).all()
    )
    # Always include ^GSPC even if we haven't predicted it yet — it's
    # the headline target.
    targets.add("^GSPC")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    backfill_kwargs: dict[str, object] = {
        "symbols": sorted(targets),
        "start": start,
        "end": end,
    }
    if history_fn is not None:
        backfill_kwargs["history_fn"] = history_fn

    bf = backfill_prices_yf(session, **backfill_kwargs)
    scored = score_outcomes(session)
    return {"backfill": bf, "scored": scored}


def build_cum_pnl_chart(df: pd.DataFrame) -> alt.Chart:
    """Cumulative-PnL line chart with target_kind colour breakdown."""
    return (
        alt.Chart(df)
        .mark_line(point=False)
        .encode(
            x=alt.X("entry_date:T", axis=alt.Axis(title="prediction date")),
            y=alt.Y(
                "cum_pnl_pct:Q",
                axis=alt.Axis(title="cumulative PnL %", format="+.1%"),
            ),
            tooltip=[
                alt.Tooltip("entry_date:T", title="Date"),
                alt.Tooltip("target_symbol:N", title="Target"),
                alt.Tooltip("pnl_pct:Q", title="Trade PnL", format="+.2%"),
                alt.Tooltip("cum_pnl_pct:Q", title="Cumulative", format="+.2%"),
            ],
        )
        .properties(height=260)
    )


def build_rolling_hit_chart(df: pd.DataFrame, window: int) -> alt.Chart:
    """Rolling hit-rate line chart with a 50% reference rule."""
    base = alt.Chart(df)
    line = base.mark_line().encode(
        x=alt.X("entry_date:T", axis=alt.Axis(title="prediction date")),
        y=alt.Y(
            "hit_rate:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(title=f"{window}-trade rolling hit-rate", format=".0%"),
        ),
        tooltip=[
            alt.Tooltip("entry_date:T", title="Date"),
            alt.Tooltip("hit_rate:Q", title="Hit rate", format=".1%"),
        ],
    )
    # Coin-flip reference line at 50%
    ref = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        strokeDash=[4, 4], color="#9aa0a6"
    ).encode(y="y:Q")
    return (line + ref).properties(height=260)


def build_hit_rate_by_kind_chart(df: pd.DataFrame) -> alt.Chart:
    """Bar chart of hit-rate per target_kind (MARKET / SECTOR / STOCK)."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("target_kind:N", title="target type"),
            y=alt.Y(
                "hit_rate:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", title="hit rate"),
            ),
            color=alt.Color(
                "target_kind:N",
                legend=None,
                scale=alt.Scale(
                    domain=["MARKET", "SECTOR", "STOCK"],
                    range=["#1f77b4", "#2ca02c", "#ff7f0e"],
                ),
            ),
            tooltip=[
                alt.Tooltip("target_kind:N", title="Target type"),
                alt.Tooltip("trades:Q", title="Closed trades"),
                alt.Tooltip("hits:Q", title="Wins"),
                alt.Tooltip("hit_rate:Q", title="Hit rate", format=".1%"),
            ],
        )
        .properties(height=220)
    )


def build_hit_rate_by_label_chart(df: pd.DataFrame) -> alt.Chart:
    """Bar chart of hit-rate per UP/DOWN/FLAT label."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=["UP", "FLAT", "DOWN"], title="call"),
            y=alt.Y(
                "hit_rate:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", title="hit rate"),
            ),
            color=alt.Color(
                "label:N",
                legend=None,
                scale=alt.Scale(
                    domain=["UP", "FLAT", "DOWN"],
                    range=["#2ca02c", "#9aa0a6", "#d62728"],
                ),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Call"),
                alt.Tooltip("trades:Q", title="Total"),
                alt.Tooltip("hits:Q", title="Hits"),
                alt.Tooltip("hit_rate:Q", title="Hit rate", format=".1%"),
            ],
        )
        .properties(height=220)
    )


def run_relationships_refresh(
    session: Session,
    *,
    api_key: str,
    company_symbols: Iterable[str] = (),
    sector_etfs: Iterable[str] = (),
    rate_limit_per_minute: int = 55,
) -> dict[str, list[RefreshResult]]:
    """Refresh peer / supply-chain / ETF-holding caches for the given symbols.

    Same safety guarantees as run_ingestion_with_key:
    short-lived FinnhubClient, ``trust_env=False``, scrubbed
    IngestionError on any leak path.
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")

    client = FinnhubClient(api_key=api_key)
    try:
        client._session.trust_env = False
    except AttributeError:  # pragma: no cover
        pass

    company_results: list[RefreshResult] = []
    sector_results: list[RefreshResult] = []
    try:
        gateway = FinnhubGateway(
            client=client,
            rate_limiter=RateLimiter(rate_limit_per_minute),
        )
        try:
            for sym in company_symbols:
                sym = sym.strip()
                if not sym:
                    continue
                company_results.append(
                    refresh_company_relationships(session, gateway, symbol=sym)
                )
            for etf in sector_etfs:
                etf = etf.strip()
                if not etf:
                    continue
                sector_results.append(
                    refresh_sector_constituents(session, gateway, etf_symbol=etf)
                )
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(scrub_token(str(exc), api_key)) from None
    finally:
        client.close()

    return {"companies": company_results, "sectors": sector_results}


def run_backfill_with_key(
    session: Session,
    *,
    api_key: str,
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
    chunk_days: int = 30,
    rate_limit_per_minute: int = 55,
) -> dict[str, object]:
    """Backfill historical company news for each ticker, score, retro-predict.

    Mirrors :func:`run_ingestion_with_key`'s safety guarantees: the key
    lives only inside the locally-scoped client, ``trust_env=False`` to
    bypass dev proxies, and any leak in exception messages is scrubbed
    via :class:`IngestionError`.

    Returns a dict with:
      * ``backfill``: ``{symbol: BackfillResult}`` from
        :func:`backfill_many`.
      * ``scored``: how many new sentiment rows were written.
      * ``predictions``: ``{symbol: n}`` from :func:`retroactive_predict_many`.
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")
    syms = [s.strip() for s in symbols if s and s.strip()]
    if not syms:
        raise ValueError("symbols must contain at least one non-empty ticker")

    client = FinnhubClient(api_key=api_key)
    try:
        client._session.trust_env = False
    except AttributeError:  # pragma: no cover
        pass

    try:
        gateway = FinnhubGateway(
            client=client,
            rate_limiter=RateLimiter(rate_limit_per_minute),
        )
        scorer = resolve_active_scorer()
        try:
            results = backfill_many(
                session,
                gateway,
                symbols=syms,
                start=start,
                end=end,
                chunk_days=chunk_days,
            )
            scored = score_pending_articles(session, scorer)
            predictions = retroactive_predict_many(
                session,
                scorer=scorer,
                symbols=syms,
                start=start,
                end=end,
            )
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(scrub_token(str(exc), api_key)) from None
    finally:
        client.close()

    return {
        "backfill": results,
        "scored": scored,
        "predictions": predictions,
    }


# -- Streamlit page (thin glue, exercised only via the dev server) --------


@dataclass(frozen=True)
class _SidebarState:
    """What the sidebar surfaced to ``main()`` on this rerun."""

    triggered_key: str | None
    symbols_csv: str
    backfill_triggered_key: str | None
    backfill_tickers_csv: str
    backfill_lookback_days: int
    price_backfill_triggered: bool
    price_backfill_days: int


def _render_sidebar() -> _SidebarState:  # pragma: no cover - Streamlit UI
    """Render the API-key + ingestion + backfill sidebar."""
    # Hydrate the key from browser localStorage on the first render
    # of this session. Must run BEFORE the text_input below, because
    # the input's value comes from st.session_state[API_KEY_SESSION_KEY].
    _hydrate_api_key_from_browser()

    with st.sidebar:
        st.header("Finnhub API key")
        st.caption(
            "Required to fetch news and prices. Persisted in this "
            "browser's **localStorage** so it survives page reloads, "
            "but **never** written to the server's disk or the SQLite "
            "database. *Clear key* wipes both the session and the "
            "browser cache."
        )
        st.text_input(
            "API key",
            type="password",
            key=API_KEY_SESSION_KEY,
            placeholder="paste your key here",
            help="Your free key from https://finnhub.io/dashboard",
        )

        current = (st.session_state.get(API_KEY_SESSION_KEY) or "").strip()
        if current:
            # Mirror the in-memory value into the browser cache the
            # first time we see it (no-op if unchanged since last write).
            _persist_api_key_to_browser(current)
            st.success("✓ key set (cached in browser)")
        else:
            st.warning("no key set — ingestion disabled")

        symbols_csv = st.text_input(
            "Company tickers (optional, comma-separated)",
            value="",
            placeholder="AAPL, MSFT, NVDA",
        )

        col_clear, col_run = st.columns(2)
        with col_clear:
            if st.button("Clear key", disabled=not current):
                # Atomic wipe: server session + browser cache. Both
                # have to go in the same click, otherwise the next
                # rerun would re-hydrate from cache.
                st.session_state.pop(API_KEY_SESSION_KEY, None)
                _clear_browser_api_key()
                # Reset the hydrate flag so a manual page refresh after
                # this click doesn't auto-pull from cache either (the
                # delete should already have made that a no-op, but
                # cheap defence in depth).
                st.session_state.pop(_BROWSER_STORAGE_HYDRATED_FLAG, None)
                st.rerun()
        with col_run:
            run_clicked = st.button("Run ingestion now", disabled=not current)

        st.divider()
        st.subheader("Backfill historical news")
        st.caption(
            "Pulls `/company-news` for each ticker over the lookback "
            "window, dedupes against the existing DB, scores new "
            "articles, and writes one Prediction per UTC day so the "
            "*History* tab and rolling baseline populate immediately. "
            "Free-tier Finnhub keys typically allow ~1 year back."
        )
        backfill_tickers_csv = st.text_input(
            "Tickers to backfill (CSV)",
            value="",
            placeholder="AAPL, MSFT, NVDA",
            key="backfill_tickers_input",
        )
        backfill_lookback_days = st.slider(
            "Lookback (days)",
            min_value=7,
            max_value=365,
            value=30,
            step=1,
            help="Window: today minus N days through today (UTC).",
        )
        backfill_clicked = st.button(
            "Backfill",
            disabled=not (current and backfill_tickers_csv.strip()),
            help="Requires a key + at least one ticker.",
        )

        st.divider()
        st.subheader("Backfill prices (yfinance)")
        st.caption(
            "Pulls daily OHLC from Yahoo Finance for every prediction "
            "target in the DB and scores any predictions whose next-"
            "session close is now available. No API key needed — fills "
            "the gap left by Finnhub's gated `/stock/candle`. Drives the "
            "*Performance* tab's hit-rate and cumulative-PnL curves."
        )
        price_backfill_days = st.slider(
            "Price lookback (days)",
            min_value=14,
            max_value=730,
            value=180,
            step=1,
            key="price_lookback_slider",
        )
        price_backfill_clicked = st.button("Backfill prices")

    triggered = current if (current and run_clicked) else None
    backfill_key = current if (current and backfill_clicked) else None
    return _SidebarState(
        triggered_key=triggered,
        symbols_csv=symbols_csv,
        backfill_triggered_key=backfill_key,
        backfill_tickers_csv=backfill_tickers_csv,
        backfill_lookback_days=backfill_lookback_days,
        price_backfill_triggered=bool(price_backfill_clicked),
        price_backfill_days=price_backfill_days,
    )


def _parse_symbols(csv: str) -> list[str]:
    """Split a comma-separated ticker string. Empty → empty list. Public for tests."""
    return [s.strip().upper() for s in csv.split(",") if s.strip()]


def main() -> None:  # pragma: no cover - thin glue exercised by the dev server
    # Auth gate first — locks the page until the env-var password
    # matches. No-op when FINN_PREDICTOR_PASSWORD_HASH is unset.
    if not _enforce_auth_gate():
        return

    st.set_page_config(page_title="Finn-Predictor", layout="wide")

    # UI must boot even without a key — load_settings(require_api_key=False)
    # returns Settings with an empty finnhub_api_key, which the UI never
    # reads. The actual key comes from st.session_state set by the sidebar.
    settings = load_settings(require_api_key=False)
    engine, SessionLocal = create_engine_and_session(settings.database_url)
    init_db(engine)

    sidebar = _render_sidebar()
    triggered_key = sidebar.triggered_key
    symbols_csv = sidebar.symbols_csv

    with SessionLocal() as session:
        if sidebar.price_backfill_triggered:
            with st.spinner(
                f"Backfilling prices for the last "
                f"{sidebar.price_backfill_days} days…"
            ):
                try:
                    out = run_price_backfill(
                        session, days=sidebar.price_backfill_days
                    )
                except Exception as exc:
                    st.sidebar.error(f"Price backfill failed: {exc}")
                else:
                    bf: PriceBackfillResult = out["backfill"]
                    bt = out["scored"]
                    st.sidebar.success(
                        f"Prices done — bars +{bf.bars_inserted} "
                        f"({bf.symbols_with_data}/{bf.symbols_requested} symbols), "
                        f"outcomes +{bt.scored} (hits {bt.hits})"
                    )
                    if bf.failures:
                        with st.sidebar.expander(
                            f"⚠ {len(bf.failures)} price failure(s)",
                            expanded=False,
                        ):
                            for f in bf.failures:
                                st.write(f"**`{f.get('symbol', '*')}`** — {f.get('error', '')}")

        if sidebar.backfill_triggered_key:
            backfill_syms = _parse_symbols(sidebar.backfill_tickers_csv)
            today_utc = datetime.now(timezone.utc)
            start = today_utc - timedelta(days=sidebar.backfill_lookback_days)
            label = (
                f"Backfilling {len(backfill_syms)} ticker(s) over "
                f"{sidebar.backfill_lookback_days} days…"
            )
            with st.spinner(label):
                try:
                    backfill_out = run_backfill_with_key(
                        session,
                        api_key=sidebar.backfill_triggered_key,
                        symbols=backfill_syms,
                        start=start,
                        end=today_utc,
                        rate_limit_per_minute=settings.rate_limit_per_minute,
                    )
                except Exception as exc:
                    msg = scrub_token(str(exc), sidebar.backfill_triggered_key)
                    st.sidebar.error(f"Backfill failed: {msg}")
                else:
                    total_inserted = sum(
                        r.inserted for r in backfill_out["backfill"].values()
                    )
                    total_failures = sum(
                        r.chunks_failed for r in backfill_out["backfill"].values()
                    )
                    total_preds = sum(backfill_out["predictions"].values())
                    st.sidebar.success(
                        f"Backfill done — articles +{total_inserted}, "
                        f"scored {backfill_out['scored']}, "
                        f"predictions {total_preds}"
                    )
                    with st.sidebar.expander(
                        "Backfill detail",
                        expanded=False,
                    ):
                        for sym, res in backfill_out["backfill"].items():
                            n_pred = backfill_out["predictions"].get(sym, 0)
                            st.write(
                                f"**`{sym}`** — +{res.inserted} articles, "
                                f"{res.succeeded_chunks}/{res.chunks_attempted} "
                                f"chunk(s) ok, {n_pred} prediction(s)"
                            )
                            for f in res.failures:
                                st.caption(f"  · {f['error']}")
                    if total_failures:
                        st.sidebar.caption(
                            f"⚠ {total_failures} chunk(s) failed across "
                            "tickers; see Backfill detail."
                        )

        if triggered_key:
            symbols = _parse_symbols(symbols_csv)
            with st.spinner("Ingesting news & prices…"):
                try:
                    counts = run_ingestion_with_key(
                        session,
                        api_key=triggered_key,
                        rate_limit_per_minute=settings.rate_limit_per_minute,
                        company_symbols=symbols,
                    )
                    failures = counts.get("failures") or []
                    news_failure = next(
                        (f for f in failures if f["op"] == "general_news"),
                        None,
                    )

                    if news_failure is not None:
                        # The /news endpoint itself rejected the request —
                        # usually means the key lacks access, the plan was
                        # downgraded, or a daily quota tripped. The candles
                        # would have failed anyway on free-tier; this is the
                        # one to act on.
                        st.sidebar.error(
                            f"News fetch (`{news_failure['op']}`) failed: "
                            f"{news_failure['error']}\n\n"
                            "Likely causes: invalid or rotated key, plan "
                            "doesn't include `/news`, daily quota exhausted."
                        )
                        if len(failures) > 1:
                            with st.sidebar.expander(
                                f"… plus {len(failures) - 1} other failure(s)",
                                expanded=False,
                            ):
                                for f in failures:
                                    if f is news_failure:
                                        continue
                                    st.write(f"**`{f['op']}`** — {f['error']}")
                    else:
                        st.sidebar.success(
                            f"Done — articles +{counts['general_news']}, "
                            f"scored {counts['scored']}, "
                            f"predictions {counts['predictions']}"
                        )
                        if counts.get("general_news", 0) == 0 and not failures:
                            st.sidebar.caption(
                                "Note: `/news` returned no new articles this "
                                "run. The feed is unchanged since the last "
                                "ingestion."
                            )
                        if failures:
                            with st.sidebar.expander(
                                f"⚠ {len(failures)} endpoint(s) failed "
                                "(expand for detail)",
                                expanded=False,
                            ):
                                for f in failures:
                                    st.write(f"**`{f['op']}`** — {f['error']}")
                except Exception as exc:
                    # Triple-layer scrub: gateway + helper already replaced
                    # the token, but we run one more pass at the display
                    # boundary in case any future code path adds a new leak.
                    msg = scrub_token(str(exc), triggered_key)
                    st.sidebar.error(f"Ingestion failed: {msg}")

        st.title("Finn-Predictor")
        (
            tab_today,
            tab_history,
            tab_sectors,
            tab_performance,
            tab_focus,
            tab_learning,
        ) = st.tabs(
            ["Today", "History", "Sectors", "Performance", "Focus", "Learning"]
        )

        with tab_today:
            preds = latest_predictions(session)
            market_pred, sector_preds, stock_preds = partition_predictions(
                session, preds
            )

            # ^GSPC price chart at the top — same lookback window as the
            # rolling baseline (30 days). Pan + mouse-wheel zoom via
            # Altair's interactive(). Hidden when no bars exist (free-tier
            # /stock/candle 403 path); yfinance backfill populates them.
            _gspc_window_end = datetime.now(timezone.utc) + timedelta(days=1)
            _gspc_window_start = _gspc_window_end - timedelta(days=45)
            _gspc_bars = price_bars(
                session, "^GSPC",
                start=_gspc_window_start, end=_gspc_window_end,
            )
            _gspc_chart = build_market_price_chart(
                _gspc_bars, symbol="^GSPC", days=30,
            )
            if _gspc_chart is not None:
                st.caption(
                    "^GSPC daily close, last 30 days — scroll to zoom, "
                    "click-drag to pan."
                )
                st.altair_chart(_gspc_chart, use_container_width=True)

            if market_pred is None:
                st.info(
                    "No predictions yet. Paste an API key in the sidebar and "
                    "click *Run ingestion now*."
                )
            else:
                cols = st.columns(3)
                cols[0].metric("Call", market_pred.label)
                cols[1].metric("Confidence", f"{market_pred.confidence:.2f}")
                cols[2].metric("Articles", market_pred.article_count)

                # --- Expected move band -------------------------------
                # Populated only when FINN_PREDICTOR_MAGNITUDE=quantile
                # and a calibration has been fitted. Null columns mean
                # the band is off — render nothing rather than zeros so
                # the user isn't misled about whether a forecast exists.
                _render_expected_move(market_pred)

            # --- Why this Call? ---------------------------------------
            # The explanation block is scoped to the market + sector
            # predictions. Per-stock predictions get their own compact
            # table below — explaining each of N stocks in 5 paragraphs
            # would drown the page.
            explanation_preds: list[Prediction] = []
            if market_pred is not None:
                explanation_preds.append(market_pred)
            explanation_preds.extend(sector_preds)

            if explanation_preds:
                if len(explanation_preds) == 1:
                    st.subheader("Why this Call?")
                else:
                    st.subheader(
                        f"Why these {len(explanation_preds)} Calls? "
                        f"({len(sector_preds)} sector(s) plus the market)"
                    )

                # Fixed-height container makes the explanation scrollable
                # when the per-prediction text gets long.
                with st.container(height=480):
                    for i, p in enumerate(explanation_preds):
                        long_name = expand_symbol(session, p.target_symbol)
                        st.markdown(
                            f"### {long_name} — **{p.label}** "
                            f"(confidence {p.confidence:.2f}, "
                            f"{p.article_count} article(s))"
                        )
                        st.markdown(explain_prediction(session, prediction=p))
                        if i < len(explanation_preds) - 1:
                            st.divider()

            # --- Per-stock predictions, grouped by sector ------------
            if stock_preds:
                st.subheader("Per-stock predictions, grouped by sector")
                st.caption(
                    "Directional call per ticker, computed from that "
                    "stock's own company-news sentiment. Same classifier "
                    "as the market call. Stocks are bucketed by their "
                    "curated sector membership; each sector header shows "
                    "that sector's synthesized prediction (when the "
                    "ingest job had constituents to aggregate). Tickers "
                    "not in the curated map land in *Other / unmapped*."
                )

                # Side-channel lookup so each sector header can show
                # the synthesized sector call alongside the stocks.
                sector_pred_by_etf = {
                    sp.target_symbol: sp for sp in sector_preds
                }

                grouped, unmapped = group_stock_predictions_by_sector(
                    session, stock_preds
                )

                _stock_column_config = {
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                    "Sentiment": st.column_config.NumberColumn(
                        "Sentiment",
                        format="%+.3f",
                        help="Recency-weighted mean of today's "
                             "scored articles, on a [-1, +1] scale.",
                    ),
                    "As of": st.column_config.DatetimeColumn(
                        "As of",
                        format="YYYY-MM-DD",
                    ),
                }

                for sector, preds_in_sector in grouped:
                    synth = sector_pred_by_etf.get(sector.etf_symbol)
                    if synth is not None:
                        sector_header = (
                            f"#### {sector.name} (`{sector.etf_symbol}`) "
                            f"— **{synth.label}** · "
                            f"conf {synth.confidence:.2f} · "
                            f"{synth.article_count} article(s) "
                            f"across {len(preds_in_sector)} stock(s)"
                        )
                    else:
                        sector_header = (
                            f"#### {sector.name} (`{sector.etf_symbol}`) "
                            f"— *no synthesized prediction yet* · "
                            f"{len(preds_in_sector)} stock(s)"
                        )
                    st.markdown(sector_header)
                    df = stock_predictions_table(session, preds_in_sector)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=_stock_column_config,
                    )

                if unmapped:
                    st.markdown(
                        "#### Other / unmapped — "
                        f"{len(unmapped)} stock(s) not in the curated "
                        "sector map"
                    )
                    df = stock_predictions_table(session, unmapped)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=_stock_column_config,
                    )

            # --- Contribution chart (per-article, divergent bars) ----
            market_contribs: list[ArticleContribution] = []
            if market_pred is not None:
                market_contribs = article_contributions(session, prediction=market_pred)

            if market_contribs:
                st.subheader("Per-article contribution chart")
                st.caption(
                    "One bar per article in today's window. Positive contributions "
                    "extend above the 0-line, negative below. Bars are sorted "
                    "left-to-right from most-negative to most-positive."
                )
                df = contribution_chart_data(market_contribs, session=session)
                chart = build_contribution_chart(
                    df,
                    title=f"Contributions to {expand_symbol(session, market_pred.target_symbol)} call",
                )
                st.altair_chart(chart, use_container_width=True)

            # --- Recent headlines (sorted by contribution to the market call) ---
            st.subheader("Recent headlines")
            if market_pred is not None:
                if market_contribs:
                    st.caption(
                        "Sorted by signed contribution to the Call. "
                        "🟢 = supports the Call, 🔴 = opposes."
                    )
                    rows = headlines_from_contributions(market_contribs, limit=10)
                else:
                    # Prediction exists but no scored articles — fall back to recency.
                    rows = recent_headlines(
                        session, limit=10, model_version=market_pred.model_version
                    )
            else:
                rows = recent_headlines(session, limit=10)

            # Expand each row's ticker → company name once, so the headline
            # display can show "Apple Inc. (AAPL)" instead of just AAPL.
            # expand_symbol_short strips any trailing " (SYM)" so we don't
            # render "(XLK) (XLK)" for sector-ETF tickers.
            for r in rows:
                sym = (r.get("symbol") or "").strip()
                if sym and sym != "*":
                    r["company"] = expand_symbol_short(session, sym)

            # Attach the earliest "first reported" timestamp per headline.
            attach_first_seen(session, rows)

            if rows:
                for row in rows:
                    st.markdown("- " + _format_headline_markdown(row))
            else:
                st.write("No headlines yet.")

            # --- Neutral headlines (scored but ~zero, didn't move the index) ---
            # These articles were in the prediction's window and got
            # through the scorer, but the polarity score was at the
            # noise floor (|score| ≤ NEUTRAL_SENTIMENT_THRESHOLD), so
            # they contributed nothing material to the weighted mean.
            # Surfacing them separately keeps the reader honest about
            # how much of the day's news flow the model actually used
            # versus shrugged at.
            if market_pred is not None and market_contribs:
                neutral_rows = neutral_headlines_from_contributions(
                    market_contribs, limit=10
                )
                if neutral_rows:
                    st.subheader("Neutral headlines")
                    st.caption(
                        f"In the prediction's window but with "
                        f"|sentiment| ≤ {NEUTRAL_SENTIMENT_THRESHOLD} — "
                        f"the scorer saw them but didn't extract enough "
                        f"polarity to move the Call. Sorted by recency."
                    )
                    for r in neutral_rows:
                        sym = (r.get("symbol") or "").strip()
                        if sym and sym != "*":
                            r["company"] = expand_symbol_short(session, sym)
                    attach_first_seen(session, neutral_rows)
                    for row in neutral_rows:
                        st.markdown("- " + _format_headline_markdown(row))

        with tab_history:
            hist = prediction_history(session, target_symbol="^GSPC")
            if hist.empty:
                st.write("No history yet.")
            else:
                st.dataframe(hist, use_container_width=True)
                if hist["realised_return"].notna().any():
                    st.line_chart(
                        hist.set_index("prediction_date")[["sentiment_index"]]
                    )

        with tab_sectors:
            sectors_present = all_sectors(session)
            grid = sector_grid(session, sectors_present)
            if grid.empty:
                # Distinguish the three empty-state shapes so the user
                # gets an actionable nudge instead of a wrong diagnosis.
                if not sectors_present:
                    st.write(
                        "Sectors not seeded yet — run an ingestion cycle "
                        "(the default 11 Sector SPDR ETFs are seeded on "
                        "first ingest)."
                    )
                else:
                    # Sectors ARE seeded; the predictions table is empty
                    # because predict_all_sectors had no constituents to
                    # aggregate. Two ways to get this populated.
                    from finn_predictor.storage.repo import related_entities_for

                    any_cached = any(
                        related_entities_for(
                            session, s.etf_symbol, relationship="ETF_HOLDING"
                        )
                        for s in sectors_present
                    )
                    if not any_cached:
                        st.info(
                            "**No sector predictions yet.** Sector "
                            "aggregation needs cached constituents per ETF. "
                            "Open the **Focus** tab → **Sector** mode → pick "
                            "an ETF → click **Refresh constituents**. Then "
                            "run ingestion again — each refreshed sector "
                            "starts producing predictions on the next cycle."
                        )
                    else:
                        st.info(
                            "**No sector predictions yet.** Constituents "
                            "are cached, but the per-constituent "
                            "`company-news` feed has no articles in the "
                            "current window. Run ingestion (or backfill) "
                            "to populate `company`-category articles."
                        )
            else:
                st.dataframe(grid, use_container_width=True)

        # --- Performance tab (hypothetical-trade accuracy + PnL) ---
        with tab_performance:
            trades = hypothetical_trades(session)
            summary = performance_summary(trades)

            st.caption(
                "Each prediction is treated as a paper trade: **UP → long** "
                "the target at the prediction-day close, **DOWN → short**, "
                "**FLAT → no trade**. Exit is the next session's close. "
                "Hit rate uses the backtester's sign-of-direction rule "
                "(|return| < 0.25% counts as a FLAT hit)."
            )

            if summary.closed_trades == 0 and summary.open_trades == 0:
                st.info(
                    "No predictions yet. Run an ingestion or a backfill "
                    "from the sidebar to populate this tab."
                )
            elif summary.closed_trades == 0:
                st.warning(
                    f"{summary.open_trades} open prediction(s) — no "
                    "outcomes yet. Click *Backfill prices* in the sidebar "
                    "to pull historical OHLC and score them."
                )

            # --- Summary metrics row ---
            cols = st.columns(5)
            cols[0].metric("Predictions", summary.total_predictions)
            cols[1].metric("Closed trades", summary.closed_trades)
            cols[2].metric(
                "Hit rate",
                f"{summary.hit_rate:.1%}" if summary.closed_trades else "—",
            )
            cols[3].metric(
                "Cumulative PnL",
                f"{summary.cumulative_pnl_pct:+.2%}" if summary.closed_trades else "—",
            )
            cols[4].metric(
                "Avg PnL / trade",
                f"{summary.avg_pnl_per_trade_pct:+.3%}" if summary.closed_trades else "—",
            )

            if summary.closed_trades:
                cols2 = st.columns(3)
                cols2[0].metric(
                    "Best trade",
                    f"{summary.best_trade_pnl_pct:+.2%}" if summary.best_trade_pnl_pct is not None else "—",
                )
                cols2[1].metric(
                    "Worst trade",
                    f"{summary.worst_trade_pnl_pct:+.2%}" if summary.worst_trade_pnl_pct is not None else "—",
                )
                cols2[2].metric(
                    "Open positions", summary.open_trades
                )

            # --- Cumulative PnL chart ---
            cum_df = cumulative_pnl_series(trades)
            if not cum_df.empty:
                st.subheader("Cumulative PnL over time")
                st.altair_chart(
                    build_cum_pnl_chart(cum_df), use_container_width=True
                )

            # --- Rolling hit-rate ---
            rolling_df = rolling_hit_rate(trades, window=14)
            if not rolling_df.empty:
                st.subheader("Rolling 14-trade hit-rate")
                st.altair_chart(
                    build_rolling_hit_chart(rolling_df, window=14),
                    use_container_width=True,
                )

            # --- Per-target-kind + per-label breakdowns ---
            kind_df = hit_rate_by_target_kind(trades)
            label_df = hit_rate_by_label(trades)
            if not kind_df.empty or not label_df.empty:
                cols3 = st.columns(2)
                with cols3[0]:
                    st.subheader("Hit rate by target type")
                    if kind_df.empty:
                        st.write("No closed directional trades yet.")
                    else:
                        st.altair_chart(
                            build_hit_rate_by_kind_chart(kind_df),
                            use_container_width=True,
                        )
                with cols3[1]:
                    st.subheader("Hit rate by Call")
                    if label_df.empty:
                        st.write("No closed trades yet.")
                    else:
                        st.altair_chart(
                            build_hit_rate_by_label_chart(label_df),
                            use_container_width=True,
                        )

            # --- Trade ledger ---
            if trades:
                st.subheader("Trade ledger")
                df = trades_dataframe(trades).sort_values(
                    "entry_date", ascending=False
                )
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence", min_value=0.0, max_value=1.0,
                            format="%.2f",
                        ),
                        "pnl_pct": st.column_config.NumberColumn(
                            "PnL %", format="%+.2f%%",
                        ),
                        "realised_return": st.column_config.NumberColumn(
                            "Next-day move", format="%+.2f%%",
                        ),
                    },
                )

        # --- Focus tab ----------------------------------------------------
        with tab_focus:
            st.caption(
                "Drill into a single company, sector, or current event. "
                "For companies the view pulls in peers (and supply-chain "
                "when your Finnhub plan supports `/stock/supply-chain`); "
                "for sectors it lists the cached ETF constituents; for "
                "free-text events it searches the news DB and emits an "
                "implied Call from the aggregated sentiment."
            )

            mode = st.radio(
                "Focus on:",
                options=["Company", "Sector", "Event"],
                horizontal=True,
                key="focus_mode",
            )
            # Use whichever scorer the live pipeline is configured for —
            # falls back to VADER when FINN_PREDICTOR_SCORER is unset, so
            # the Focus tab keeps working unchanged for the default deploy.
            vader_version = resolve_active_scorer().model_version

            if mode == "Company":
                tickers_known = sorted(
                    {p.target_symbol for p in latest_predictions(session)
                     if p.target_symbol != "^GSPC"}
                )
                col1, col2 = st.columns([3, 1])
                ticker_input = col1.text_input(
                    "Ticker", value=tickers_known[0] if tickers_known else "",
                    placeholder="e.g. AAPL",
                )
                with col2:
                    refresh_clicked = st.button(
                        "Refresh peers + supply chain",
                        disabled=not (sidebar.triggered_key or sidebar.backfill_triggered_key) and not (st.session_state.get(API_KEY_SESSION_KEY) or ""),
                        help="Calls Finnhub to update the peer/supply-chain "
                             "cache for this ticker. Requires a key in the sidebar.",
                    )
                if refresh_clicked and ticker_input:
                    key = (st.session_state.get(API_KEY_SESSION_KEY) or "").strip()
                    if not key:
                        st.warning("Set the API key in the sidebar first.")
                    else:
                        with st.spinner(f"Refreshing {ticker_input.upper()} relationships…"):
                            try:
                                refresh_out = run_relationships_refresh(
                                    session,
                                    api_key=key,
                                    company_symbols=[ticker_input],
                                    rate_limit_per_minute=settings.rate_limit_per_minute,
                                )
                            except Exception as exc:
                                msg = scrub_token(str(exc), key)
                                st.error(f"Refresh failed: {msg}")
                            else:
                                r = refresh_out["companies"][0] if refresh_out["companies"] else None
                                if r is not None:
                                    parts = []
                                    if r.peers_added:
                                        parts.append(f"{r.peers_added} peers")
                                    if r.suppliers_added:
                                        parts.append(f"{r.suppliers_added} suppliers")
                                    if r.customers_added:
                                        parts.append(f"{r.customers_added} customers")
                                    if parts:
                                        st.success("Refreshed: " + ", ".join(parts))
                                    if r.failures:
                                        with st.expander(
                                            f"⚠ {len(r.failures)} call(s) failed",
                                            expanded=False,
                                        ):
                                            for f in r.failures:
                                                st.write(
                                                    f"**`{f['op']}`** — {f['error']}"
                                                )

                if ticker_input:
                    bundle = compose_company_focus(
                        session, ticker_input, model_version=vader_version
                    )
                    _render_company_focus(session, bundle)
                else:
                    st.info("Type a ticker to focus on.")

            elif mode == "Sector":
                sectors = all_sectors(session)
                if not sectors:
                    st.info("Sectors not seeded yet — run an ingestion cycle first.")
                else:
                    sector_labels = {f"{s.name} ({s.etf_symbol})": s for s in sectors}
                    chosen_label = st.selectbox(
                        "Sector", options=list(sector_labels.keys())
                    )
                    chosen = sector_labels[chosen_label]
                    refresh_clicked = st.button(
                        "Refresh constituents",
                        help=(
                            "Calls Finnhub's /etf/holdings to update the "
                            "cached top constituents. Requires a key."
                        ),
                    )
                    if refresh_clicked:
                        key = (st.session_state.get(API_KEY_SESSION_KEY) or "").strip()
                        if not key:
                            st.warning("Set the API key in the sidebar first.")
                        else:
                            with st.spinner(f"Refreshing {chosen.etf_symbol} constituents…"):
                                try:
                                    out = run_relationships_refresh(
                                        session,
                                        api_key=key,
                                        sector_etfs=[chosen.etf_symbol],
                                    )
                                except Exception as exc:
                                    msg = scrub_token(str(exc), key)
                                    st.error(f"Refresh failed: {msg}")
                                else:
                                    sr = out["sectors"][0] if out["sectors"] else None
                                    if sr is not None:
                                        st.success(
                                            f"Refreshed: {sr.holdings_added} constituents"
                                        )
                                        if sr.failures:
                                            with st.expander(
                                                "⚠ failures", expanded=False,
                                            ):
                                                for f in sr.failures:
                                                    st.write(
                                                        f"**`{f['op']}`** — {f['error']}"
                                                    )

                    bundle = compose_sector_focus(
                        session, chosen.code, model_version=vader_version,
                    )
                    if bundle is None:
                        st.info("Sector not found in DB.")
                    else:
                        _render_sector_focus(session, bundle)

            else:  # Event
                col1, col2 = st.columns([3, 1])
                query = col1.text_input(
                    "Search news for:",
                    placeholder="e.g. Iran war, Fed rate cut, layoffs",
                )
                lookback = col2.slider(
                    "Lookback days",
                    min_value=1, max_value=90, value=14, step=1,
                    key="event_lookback",
                )
                if query:
                    bundle = compose_event_focus(
                        session, query,
                        lookback_days=lookback,
                        model_version=vader_version,
                    )
                    _render_event_focus(session, bundle)
                else:
                    st.info("Type a query to search.")

        with tab_learning:
            _render_learning_tab_wrapped(session)


def _render_related_grid(
    session: Session,
    rows: list[RelatedPrediction],
    *,
    relationship_label: str,
) -> None:  # pragma: no cover - rendering glue
    """Compact table of related entities + their predictions."""
    if not rows:
        return
    df_rows = []
    for r in rows:
        df_rows.append(
            {
                relationship_label: r.related_symbol,
                "Company": expand_symbol_short(session, r.related_symbol),
                "Call": r.prediction.label if r.prediction else "—",
                "Confidence": r.prediction.confidence if r.prediction else None,
                "Articles": r.prediction.article_count if r.prediction else 0,
                "Sentiment": r.prediction.sentiment_index if r.prediction else None,
                "Note": r.metadata_text or "",
            }
        )
    df = pd.DataFrame(df_rows)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f",
            ),
            "Sentiment": st.column_config.NumberColumn(
                "Sentiment", format="%+.3f",
            ),
        },
    )


def _render_company_focus(
    session: Session, bundle: CompanyFocus
) -> None:  # pragma: no cover - rendering glue
    long_name = expand_symbol(session, bundle.symbol)
    st.subheader(long_name)

    cols = st.columns(3)
    if bundle.own_prediction is not None:
        cols[0].metric("Call", bundle.own_prediction.label)
        cols[1].metric("Confidence", f"{bundle.own_prediction.confidence:.2f}")
        cols[2].metric("Articles", bundle.own_prediction.article_count)
    else:
        cols[0].metric("Call", "—")
        cols[1].metric("Confidence", "—")
        cols[2].metric("Articles", 0)

    if bundle.sector_etf:
        st.caption(
            f"Sector: **{bundle.sector_code or '?'}** "
            f"(`{bundle.sector_etf}`)" + (
                f" — sector call **{bundle.sector_prediction.label}** "
                f"(conf {bundle.sector_prediction.confidence:.2f})"
                if bundle.sector_prediction else " — no sector call yet"
            )
        )

    if bundle.peers:
        st.markdown("**Peers**")
        _render_related_grid(session, bundle.peers, relationship_label="Peer")
    if bundle.suppliers:
        st.markdown("**Suppliers**")
        _render_related_grid(session, bundle.suppliers, relationship_label="Supplier")
    if bundle.customers:
        st.markdown("**Customers**")
        _render_related_grid(session, bundle.customers, relationship_label="Customer")

    if not (bundle.peers or bundle.suppliers or bundle.customers):
        st.caption(
            "No cached relationships yet. Click *Refresh peers + supply "
            "chain* above to pull them from Finnhub."
        )

    st.markdown("**Recent articles (subject + peers)**")
    if bundle.recent_articles:
        for row in bundle.recent_articles:
            sym = (row.get("symbol") or "").strip()
            if sym and sym != "*":
                row["company"] = expand_symbol_short(session, sym)
        attach_first_seen(session, bundle.recent_articles)
        for row in bundle.recent_articles:
            st.markdown("- " + _format_headline_markdown(row))
    else:
        st.write("No matching articles in the last 30 days.")


def _render_sector_focus(
    session: Session, bundle: SectorFocus
) -> None:  # pragma: no cover - rendering glue
    st.subheader(f"{bundle.sector_name} ({bundle.etf_symbol})")
    cols = st.columns(3)
    if bundle.own_prediction is not None:
        cols[0].metric("Call", bundle.own_prediction.label)
        cols[1].metric("Confidence", f"{bundle.own_prediction.confidence:.2f}")
        cols[2].metric("Articles", bundle.own_prediction.article_count)
    else:
        cols[0].metric("Call", "—")
        cols[1].metric("Confidence", "—")
        cols[2].metric("Articles", 0)

    if bundle.constituents:
        st.markdown("**Top constituents**")
        _render_related_grid(
            session, bundle.constituents, relationship_label="Ticker"
        )
    else:
        st.caption(
            "No constituents cached yet. Click *Refresh constituents* to pull "
            "the top holdings from Finnhub's /etf/holdings."
        )

    st.markdown("**Recent articles (ETF + constituents)**")
    if bundle.recent_articles:
        for row in bundle.recent_articles:
            sym = (row.get("symbol") or "").strip()
            if sym and sym != "*":
                row["company"] = expand_symbol_short(session, sym)
        attach_first_seen(session, bundle.recent_articles)
        for row in bundle.recent_articles:
            st.markdown("- " + _format_headline_markdown(row))
    else:
        st.write("No matching articles in the last 30 days.")


def _render_event_focus(
    session: Session, bundle: EventFocus
) -> None:  # pragma: no cover - rendering glue
    cols = st.columns(4)
    cols[0].metric("Implied Call", bundle.implied_label)
    cols[1].metric("Confidence", f"{bundle.implied_confidence:.2f}")
    cols[2].metric("Articles", bundle.article_count)
    cols[3].metric(
        "Aggregate sentiment",
        f"{bundle.aggregate_sentiment:+.3f}" if bundle.article_count else "—",
    )
    st.caption(
        f"Searched the news DB for `{bundle.query}` over the last "
        f"{bundle.lookback_days} day(s). Implied Call uses sign-of-mean "
        "(±0.1 dead-band); confidence is |mean|/0.5 clipped to 1."
    )

    if bundle.target_breakdown:
        st.markdown("**Per-ticker breakdown**")
        rows = []
        for b in bundle.target_breakdown:
            sym = b["symbol"]
            rows.append(
                {
                    "Ticker": sym,
                    "Company": expand_symbol_short(session, sym) if sym != "*" else "(general)",
                    "Articles": b["articles"],
                    "Scored": b["scored"],
                    "Mean sentiment": (
                        b["mean_sentiment"]
                        if b["mean_sentiment"] == b["mean_sentiment"]
                        else None
                    ),
                }
            )
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True,
            column_config={
                "Mean sentiment": st.column_config.NumberColumn(
                    "Mean sentiment", format="%+.3f",
                ),
            },
        )

    st.markdown("**Matching articles**")
    if bundle.matched_articles:
        for row in bundle.matched_articles:
            sym = (row.get("symbol") or "").strip()
            if sym and sym != "*":
                row["company"] = expand_symbol_short(session, sym)
        attach_first_seen(session, bundle.matched_articles)
        for row in bundle.matched_articles:
            st.markdown("- " + _format_headline_markdown(row))
    else:
        st.write("No matches.")


# --- Learning tab ----------------------------------------------------


def _render_learning_tab(session: Session) -> None:  # pragma: no cover
    """Self-improvement: train new weights from the trade ledger."""
    st.caption(
        "Train new weights from the hypothetical-trade ledger. The "
        "objective is `hit_rate + 0.5 × cumulative_PnL` over a "
        "time-split training window; the holdout is the last 14 days. "
        "After training the new version is **auto-activated** "
        "(newest wins). Older versions stay in the table so you can "
        "inspect them."
    )

    # --- Activation policy toggle (persisted across sessions) -------
    st.subheader("Activation policy")
    current_policy = get_activation_policy(session)
    policy_idx = 0 if current_policy == POLICY_AUTO else 1
    chosen = st.radio(
        "When training finishes:",
        options=[POLICY_AUTO, POLICY_MANUAL],
        index=policy_idx,
        format_func=lambda v: (
            "Auto-activate — newest version wins" if v == POLICY_AUTO
            else "Manual approval — leave new version inactive; you click Activate"
        ),
        horizontal=False,
        key="activation_policy_radio",
    )
    if chosen != current_policy:
        set_activation_policy(session, chosen)
        st.success(f"Activation policy set to **{chosen}**.")
        # Re-read so subsequent renders this frame reflect the new value.
        current_policy = chosen

    # Holdout gate tolerance — only meaningful under AUTO; render under
    # both for discoverability but greyed-out semantics are explained.
    current_tol = get_holdout_tolerance(session)
    new_tol = st.slider(
        "Holdout-gate tolerance (objective units)",
        min_value=0.0, max_value=0.10, value=current_tol, step=0.005,
        help=(
            "Under AUTO policy, a newly trained version is activated "
            "only if its holdout score is within (active − tolerance) "
            "of the currently live config (re-scored on the same "
            "window). 0.00 = strict (no regressions allowed); 0.10 = "
            "very permissive. Has no effect under MANUAL policy."
        ),
    )
    if abs(new_tol - current_tol) > 1e-9:
        set_holdout_tolerance(session, new_tol)

    cfg = active_weights(session)
    st.subheader("Active weights")
    cols = st.columns(4)
    cols[0].metric(
        "Version", str(cfg.version) if cfg.version is not None else "defaults",
    )
    cols[1].metric("Threshold σ", f"{cfg.threshold_sigma:.3f}")
    cols[2].metric("Min baseline σ", f"{cfg.min_baseline_sigma:.3f}")
    cols[3].metric("Half-life (h)", f"{cfg.half_life_hours:.2f}")

    if cfg.source_weights:
        with st.expander(
            f"Source weights ({len(cfg.source_weights)} sources)",
            expanded=False,
        ):
            df = pd.DataFrame(
                [
                    {"Source": k, "Weight": v}
                    for k, v in sorted(
                        cfg.source_weights.items(),
                        key=lambda kv: -kv[1],
                    )
                ]
            )
            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "Weight": st.column_config.NumberColumn(
                        "Weight", format="%.3f",
                    ),
                },
            )

    st.subheader("Train a new version")
    n_calls = st.slider(
        "Bayesian-optimisation iterations",
        min_value=10, max_value=80, value=30, step=5,
        help="More iterations = better optima but longer training. "
             "30 finishes in seconds on hundreds of trades.",
    )
    train_clicked = st.button("Retrain now")

    if train_clicked:
        with st.spinner(
            f"Training: {n_calls} Bayesian-optimisation iterations…"
        ):
            try:
                report: TrainingReport = train_weights(
                    session, n_calls=n_calls
                )
            except NotEnoughDataError as exc:
                st.warning(
                    f"Not enough closed predictions to train ({exc}). "
                    f"Need at least {MIN_TRADES_FOR_TRAINING} closed "
                    "trades — run more ingestion + price backfill first."
                )
                report = None
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                report = None

        if report is not None:
            header = (
                f"Trained version {report.version} "
                f"(train n={report.n_train}, holdout n={report.n_holdout})"
            )
            if report.gate_blocked:
                st.warning(header + " — gate **blocked** activation")
            elif report.activated:
                st.success(header + " — activated")
            else:
                st.info(header + " — saved but not activated")

            cols = st.columns(2)
            cols[0].metric(
                "Training score",
                f"{report.training_score:+.4f}",
                delta=f"{report.training_score - report.baseline_training_score:+.4f} vs. baseline",
            )
            if report.holdout_score is not None:
                delta = (
                    f"{report.holdout_score - (report.baseline_holdout_score or 0.0):+.4f} vs. baseline"
                    if report.baseline_holdout_score is not None
                    else None
                )
                cols[1].metric(
                    "Holdout score",
                    f"{report.holdout_score:+.4f}",
                    delta=delta,
                )
            else:
                cols[1].metric("Holdout score", "—")

            if report.gate_reason:
                st.caption(report.gate_reason)
                if not report.activated:
                    if st.button(
                        f"Activate v{report.version} anyway",
                        key=f"override_gate_v{report.version}",
                    ):
                        try:
                            activate_learned_version(session, report.version)
                            st.success(
                                f"Activated version {report.version} "
                                "despite the gate."
                            )
                            st.rerun()
                        except ValueError as exc:
                            st.error(f"Activation failed: {exc}")

            st.json(
                {
                    "fitted": report.fitted,
                    "source_weight_count": len(report.source_weights),
                    "active_holdout_at_decision": report.active_holdout_score_at_decision,
                    "tolerance": report.holdout_tolerance,
                }
            )

    # History of versions + per-row Activate buttons.
    st.subheader("Version history")
    hist_rows = list(
        session.scalars(
            select(LearnedWeight)
            .where(LearnedWeight.dimension == DIM_THRESHOLD)
            .order_by(LearnedWeight.version.desc())
        )
    )
    if not hist_rows:
        st.caption("No trained versions yet.")
    else:
        hist_df = pd.DataFrame(
            [
                {
                    "Version": r.version,
                    "Active": "✓" if r.is_active else "",
                    "Threshold σ": float(r.value),
                    "Training score": r.training_score,
                    "Holdout score": r.holdout_score,
                    "Fitted at": r.fitted_at,
                }
                for r in hist_rows
            ]
        )
        st.dataframe(
            hist_df, use_container_width=True, hide_index=True,
            column_config={
                "Training score": st.column_config.NumberColumn(
                    "Training score", format="%+.4f",
                ),
                "Holdout score": st.column_config.NumberColumn(
                    "Holdout score", format="%+.4f",
                ),
            },
        )

        # Per-row Activate buttons. Skip the row already active; one
        # row at a time so the rerun is clean.
        st.caption(
            "Click *Activate* on a non-active version to flip the live "
            "weights. Useful when activation policy is set to Manual, "
            "or for reverting after an auto-activated retrain."
        )
        inactive_rows = [r for r in hist_rows if not r.is_active]
        if inactive_rows:
            cols = st.columns(min(len(inactive_rows), 4))
            for i, r in enumerate(inactive_rows[: len(cols)]):
                clicked = cols[i].button(
                    f"Activate v{r.version}",
                    key=f"activate_v{r.version}",
                )
                if clicked:
                    try:
                        activate_learned_version(session, r.version)
                        st.success(f"Activated version {r.version}.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(f"Activation failed: {exc}")
            if len(inactive_rows) > len(cols):
                st.caption(
                    f"…{len(inactive_rows) - len(cols)} older inactive "
                    "version(s) not shown."
                )


def _render_learning_tab_wrapped(session: Session) -> None:
    """Outer wrapper so import-time failures don't crash the tab."""
    try:
        _render_learning_tab(session)
    except Exception as exc:  # pragma: no cover - last-resort UI guard
        st.error(f"Learning tab error: {exc}")


# Install the tab body via a small monkey-patch into main(): we already
# referenced ``tab_learning`` in the st.tabs(...) call above, so we just
# need to render its body when the tab is selected.


if __name__ == "__main__":  # pragma: no cover
    main()

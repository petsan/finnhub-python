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
from finn_predictor.sentiment.vader import VaderScorer
from finn_predictor.storage import create_engine_and_session, init_db
from finn_predictor.storage.models import (
    NewsArticle,
    Prediction,
    PredictionOutcome,
    Sector,
    SentimentScore,
)
from finn_predictor.storage.repo import all_sectors, predictions_for
from finn_predictor.storage.stories import earliest_story_times
from finn_predictor.storage.symbol_names import expand_symbol, expand_symbol_short


API_KEY_SESSION_KEY = "finnhub_api_key"


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
    earliest = earliest_story_times(session, headlines)
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
        first_seen_map = earliest_story_times(
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
            counts = run_daily_ingest(
                session=session,
                gateway=gateway,
                scorer=VaderScorer(),
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
        scorer = VaderScorer()
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
    with st.sidebar:
        st.header("Finnhub API key")
        st.caption(
            "Required to fetch news and prices. Stored **only** in this "
            "browser tab's server-side session — never written to disk or "
            "the database. Closing the tab or restarting the server "
            "discards it."
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
            st.success("✓ key set for this session")
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
                st.session_state.pop(API_KEY_SESSION_KEY, None)
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
        tab_today, tab_history, tab_sectors, tab_performance = st.tabs(
            ["Today", "History", "Sectors", "Performance"]
        )

        with tab_today:
            preds = latest_predictions(session)
            market_pred, sector_preds, stock_preds = partition_predictions(
                session, preds
            )

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

            # --- Per-stock predictions --------------------------------
            if stock_preds:
                st.subheader("Per-stock predictions")
                st.caption(
                    "Directional call per ticker, computed from that "
                    "stock's own company-news sentiment. Same classifier "
                    "as the market call — UP/DOWN/FLAT by z-score against "
                    "the ticker's 30-day baseline. Sorted by confidence."
                )
                stocks_df = stock_predictions_table(session, stock_preds)
                st.dataframe(
                    stocks_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
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
                    },
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
            grid = sector_grid(session, all_sectors(session))
            if grid.empty:
                st.write("Sectors not seeded yet — run an ingestion cycle.")
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


if __name__ == "__main__":  # pragma: no cover
    main()

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

from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

import altair as alt
import pandas as pd
import streamlit as st
from finnhub import Client as FinnhubClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.config import load_settings
from finn_predictor.ingestion.client import (
    FinnhubGateway,
    IngestionError,
    RateLimiter,
    scrub_token,
)
from finn_predictor.ingestion.jobs import run_daily_ingest
from finn_predictor.predictor.explain import (
    ArticleContribution,
    article_contributions,
    explain_prediction,
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
    """One latest Prediction per target_symbol — across market + every sector.

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


# -- Streamlit page (thin glue, exercised only via the dev server) --------


def _render_sidebar() -> tuple[str | None, str]:  # pragma: no cover - Streamlit UI
    """Render the API-key sidebar.

    Returns ``(triggered_key, symbols_csv)`` — ``triggered_key`` is the
    current in-session key iff the user clicked "Run ingestion now" this
    render, otherwise ``None``.
    """
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

    triggered = current if (current and run_clicked) else None
    return triggered, symbols_csv


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

    triggered_key, symbols_csv = _render_sidebar()

    with SessionLocal() as session:
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
                    if failures and counts.get("general_news", 0) == 0 and counts.get("scored", 0) == 0:
                        # Everything failed — most likely a bad key or a fully
                        # blocked plan. Show the first failure prominently.
                        st.sidebar.error(
                            "Every Finnhub call failed — "
                            f"first error: {failures[0]['error']}"
                        )
                    else:
                        st.sidebar.success(
                            f"Done — articles +{counts['general_news']}, "
                            f"scored {counts['scored']}, "
                            f"predictions {counts['predictions']}"
                        )
                        if failures:
                            with st.sidebar.expander(
                                f"⚠ {len(failures)} endpoint(s) failed (expand for detail)",
                                expanded=False,
                            ):
                                for f in failures:
                                    st.write(f"**{f['op']}** — {f['error']}")
                except Exception as exc:
                    # Triple-layer scrub: gateway + helper already replaced
                    # the token, but we run one more pass at the display
                    # boundary in case any future code path adds a new leak.
                    msg = scrub_token(str(exc), triggered_key)
                    st.sidebar.error(f"Ingestion failed: {msg}")

        st.title("Finn-Predictor")
        tab_today, tab_history, tab_sectors = st.tabs(
            ["Today", "History", "Sectors"]
        )

        with tab_today:
            preds = latest_predictions(session)
            market_pred = next(
                (p for p in preds if p.target_symbol == "^GSPC"), None
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
            if preds:
                if len(preds) == 1:
                    st.subheader("Why this Call?")
                else:
                    st.subheader(
                        f"Why these {len(preds)} Calls? "
                        f"({len(preds) - 1} sector(s) plus the market)"
                    )

                # Fixed-height container makes the explanation scrollable
                # when the per-prediction text gets long.
                with st.container(height=480):
                    for i, p in enumerate(preds):
                        long_name = expand_symbol(session, p.target_symbol)
                        st.markdown(
                            f"### {long_name} — **{p.label}** "
                            f"(confidence {p.confidence:.2f}, "
                            f"{p.article_count} article(s))"
                        )
                        st.markdown(explain_prediction(session, prediction=p))
                        if i < len(preds) - 1:
                            st.divider()

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


if __name__ == "__main__":  # pragma: no cover
    main()

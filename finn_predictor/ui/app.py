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
from typing import Iterable

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
                    st.sidebar.success(
                        f"Done — articles +{counts['general_news']}, "
                        f"scored {counts['scored']}, "
                        f"predictions {counts['predictions']}"
                    )
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
            pred = latest_market_prediction(session)
            if pred is None:
                st.info("No predictions yet. Paste an API key in the sidebar and click *Run ingestion now*.")
            else:
                cols = st.columns(3)
                cols[0].metric("Call", pred.label)
                cols[1].metric("Confidence", f"{pred.confidence:.2f}")
                cols[2].metric("Articles", pred.article_count)

            st.subheader("Recent headlines")
            headlines = recent_headlines(session, limit=10)
            if headlines:
                st.dataframe(pd.DataFrame(headlines), use_container_width=True)
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

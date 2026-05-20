"""Streamlit dashboard entry point.

Run with::

    streamlit run finn_predictor/ui/app.py

The UI is deliberately read-only — it never calls Finnhub directly; it
renders whatever the ingestion job has persisted.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.config import load_settings
from finn_predictor.storage import create_engine_and_session, init_db
from finn_predictor.storage.models import (
    NewsArticle,
    Prediction,
    PredictionOutcome,
    Sector,
    SentimentScore,
)
from finn_predictor.storage.repo import all_sectors, predictions_for


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


# -- Streamlit page (best-effort smoke-tested) ----------------------------


def main() -> None:  # pragma: no cover - thin glue exercised by AppTest
    st.set_page_config(page_title="Finn-Predictor", layout="wide")

    settings = load_settings()
    engine, SessionLocal = create_engine_and_session(settings.database_url)
    init_db(engine)

    with SessionLocal() as session:
        st.title("Finn-Predictor")
        tab_today, tab_history, tab_sectors = st.tabs(
            ["Today", "History", "Sectors"]
        )

        with tab_today:
            pred = latest_market_prediction(session)
            if pred is None:
                st.info("No predictions yet. Run the ingestion job to populate.")
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

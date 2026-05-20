"""APScheduler / daily-ingest job tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from finn_predictor.ingestion.jobs import (
    build_scheduler,
    run_daily_ingest,
    score_pending_articles,
)
from finn_predictor.sentiment.vader import VaderScorer
from finn_predictor.storage.models import SentimentScore
from finn_predictor.storage.repo import upsert_articles
from tests.conftest import make_article


D = datetime(2026, 5, 19, 12, tzinfo=timezone.utc)


def test_score_pending_articles_writes_one_row_per_article(session) -> None:
    upsert_articles(
        session,
        [
            make_article(finnhub_id=1, headline="Markets surge", published_at=D),
            make_article(finnhub_id=2, headline="Banks in turmoil", published_at=D),
        ],
    )
    scorer = VaderScorer()
    n = score_pending_articles(session, scorer)
    assert n == 2
    # Re-run -> nothing more to score.
    assert score_pending_articles(session, scorer) == 0
    assert session.query(SentimentScore).count() == 2


def test_run_daily_ingest_aggregates_counts(session) -> None:
    gw = MagicMock()
    gw.general_news.return_value = [
        {"id": 1, "datetime": int(D.timestamp()), "headline": "Markets surge"},
        {"id": 2, "datetime": int(D.timestamp()), "headline": "Strong earnings"},
        {"id": 3, "datetime": int(D.timestamp()), "headline": "Record close"},
    ]
    gw.company_news.return_value = []
    gw.stock_candles.return_value = {
        "s": "ok",
        "t": [int(D.timestamp())],
        "o": [100.0],
        "h": [101.0],
        "l": [99.0],
        "c": [100.5],
        "v": [1.0],
    }

    counts = run_daily_ingest(
        session=session,
        gateway=gw,
        scorer=VaderScorer(),
        market_symbol="^GSPC",
        company_symbols=[],
        today=D,
    )
    assert counts["general_news"] == 3
    assert counts["market_prices"] == 1
    # ensure_default_sectors seeds 11 sectors -> stock_candles called for each
    assert counts["sector_prices"] >= 1
    assert counts["scored"] >= 3
    # We may or may not write a prediction depending on the article cutoff;
    # but the key returned to callers should always be present.
    assert "predictions" in counts


def test_run_daily_ingest_processes_company_symbols(session) -> None:
    gw = MagicMock()
    gw.general_news.return_value = []
    gw.company_news.return_value = [
        {"id": 99, "datetime": int(D.timestamp()), "headline": "AAPL beats"}
    ]
    gw.stock_candles.return_value = {
        "s": "ok",
        "t": [int(D.timestamp())],
        "o": [100.0],
        "h": [101.0],
        "l": [99.0],
        "c": [100.5],
        "v": [1.0],
    }

    counts = run_daily_ingest(
        session=session,
        gateway=gw,
        scorer=VaderScorer(),
        market_symbol="^GSPC",
        company_symbols=["AAPL"],
        today=D,
    )
    assert counts["company_news"] == 1
    assert counts["company_prices"] == 1


def test_build_scheduler_registers_job() -> None:
    called: list[int] = []

    def job() -> None:
        called.append(1)

    sched = build_scheduler(job, hour=22, minute=15)
    jobs = sched.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "daily_ingest"
    # CronTrigger field check — we don't actually start the scheduler.
    trig = jobs[0].trigger
    fields = {f.name: str(f) for f in trig.fields}
    assert fields["hour"] == "22"
    assert fields["minute"] == "15"
    # No shutdown call needed — APScheduler's add_job works pre-start.

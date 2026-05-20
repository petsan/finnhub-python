"""APScheduler / daily-ingest job tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from finn_predictor.ingestion.client import IngestionError
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
    # No failures on a clean run.
    assert counts["failures"] == []


def test_run_daily_ingest_continues_when_one_endpoint_403s(session) -> None:
    """A 403 on /stock/candle (common on Finnhub free tier) must NOT prevent
    /news from ingesting. The failure surfaces in counts['failures']."""
    gw = MagicMock()
    gw.general_news.return_value = [
        {"id": 11, "datetime": int(D.timestamp()), "headline": "OK"}
    ]
    gw.company_news.return_value = []
    # Every stock_candles call raises the scrubbed 403.
    gw.stock_candles.side_effect = IngestionError(
        "FinnhubAPI 403: You don't have access to this resource."
    )

    counts = run_daily_ingest(
        session=session,
        gateway=gw,
        scorer=VaderScorer(),
        market_symbol="^GSPC",
        company_symbols=[],
        today=D,
    )

    # News still ingested + scored despite the price 403s.
    assert counts["general_news"] == 1
    assert counts["market_prices"] == 0
    assert counts["sector_prices"] == 0
    assert counts["scored"] >= 1

    failures = counts["failures"]
    # 1 market + 11 sector ETFs = 12 candle failures
    assert len(failures) == 12
    ops = {f["op"] for f in failures}
    assert "market_prices:^GSPC" in ops
    assert any(op.startswith("sector_prices:") for op in ops)
    # Errors are already scrubbed (carry no token).
    assert all("403" in f["error"] for f in failures)


def test_run_daily_ingest_isolates_per_company_failures(session) -> None:
    """Per-company failures must not affect peers — AAPL fails, MSFT proceeds."""
    gw = MagicMock()
    gw.general_news.return_value = []
    gw.stock_candles.return_value = {"s": "no_data"}

    def _company_news(symbol, _from, to):  # match the gateway signature
        if symbol == "AAPL":
            raise IngestionError("FinnhubAPI 403: blocked")
        return [{"id": 99, "datetime": int(D.timestamp()), "headline": "MSFT beats"}]

    gw.company_news.side_effect = _company_news

    counts = run_daily_ingest(
        session=session,
        gateway=gw,
        scorer=VaderScorer(),
        company_symbols=["AAPL", "MSFT"],
        today=D,
    )
    assert counts["company_news"] == 1  # MSFT's article landed
    aapl_failures = [f for f in counts["failures"] if "AAPL" in f["op"]]
    msft_failures = [f for f in counts["failures"] if "MSFT" in f["op"]]
    assert len(aapl_failures) == 1
    assert msft_failures == []


def test_run_daily_ingest_processes_company_symbols(session) -> None:
    gw = MagicMock()
    gw.general_news.return_value = []
    # Returns enough articles per company that the per-stock predictor
    # passes the MIN_ARTICLES_FOR_CALL=3 threshold.
    gw.company_news.return_value = [
        {
            "id": 90 + i,
            "datetime": int(D.timestamp()) + i,
            "headline": f"AAPL ships amazing product {i}",
        }
        for i in range(5)
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
    assert counts["company_news"] == 5
    assert counts["company_prices"] == 1
    # Per-stock prediction was produced and counted.
    from finn_predictor.storage.repo import predictions_for
    assert len(predictions_for(session, "AAPL")) == 1


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

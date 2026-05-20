"""CLI smoke tests — covers the Docker entrypoint code path."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import patch

import pytest

from finn_predictor.cli import _build_parser, main
from finn_predictor.storage import create_engine_and_session, init_db
from finn_predictor.storage.models import NewsArticle


def test_parser_recognises_known_commands() -> None:
    parser = _build_parser()
    args = parser.parse_args(["reset-db", "--yes"])
    assert args.command == "reset-db"
    assert args.yes is True

    args = parser.parse_args(["retrain", "--n-calls", "5", "--activate", "no"])
    assert args.command == "retrain"
    assert args.n_calls == 5
    assert args.activate == "no"

    args = parser.parse_args(["serve"])
    assert args.command == "serve"


def test_serve_subcommand_prints_hint(capsys) -> None:
    rc = main(["serve"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "streamlit run" in out


def test_reset_db_requires_yes_flag(capsys, tmp_path, monkeypatch) -> None:
    """Without --yes, reset-db refuses to touch the DB."""
    db_path = tmp_path / "f.db"
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{db_path}")
    rc = main(["reset-db"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "refusing" in err.lower()


def test_reset_db_drops_and_recreates(tmp_path, monkeypatch) -> None:
    """reset-db --yes nukes the schema; existing rows are gone afterwards."""
    db_path = tmp_path / "f.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", db_url)

    # Seed: add one news row, confirm it's there, then reset.
    engine, SL = create_engine_and_session(db_url)
    init_db(engine)
    with SL() as s:
        s.add(
            NewsArticle(
                finnhub_id=1,
                category="general",
                headline="will be wiped",
                published_at=datetime.now(timezone.utc),
            )
        )
        s.commit()
        assert s.query(NewsArticle).count() == 1
    engine.dispose()

    rc = main(["reset-db", "--yes"])
    assert rc == 0

    engine, SL = create_engine_and_session(db_url)
    init_db(engine)
    with SL() as s:
        assert s.query(NewsArticle).count() == 0
    engine.dispose()


def test_retrain_subcommand_reports_not_enough_data(capsys, tmp_path, monkeypatch) -> None:
    """retrain returns rc=3 + a friendly message when the ledger is empty."""
    db_path = tmp_path / "f.db"
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{db_path}")
    rc = main(["retrain", "--n-calls", "5"])
    err = capsys.readouterr().err
    assert rc == 3
    assert "not enough" in err.lower()


def test_hash_password_with_explicit_arg(capsys) -> None:
    """`cli hash-password <plaintext>` prints a bcrypt hash + a hint."""
    rc = main(["hash-password", "hunter2-2026"])
    captured = capsys.readouterr()
    assert rc == 0
    # The hash goes to stdout; the export hint to stderr.
    assert captured.out.strip().startswith("$2")
    assert "FINN_PREDICTOR_PASSWORD_HASH" in captured.err


def test_hash_password_rejects_too_short(capsys) -> None:
    rc = main(["hash-password", "ab"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "at least" in err.lower()


def test_ingest_subcommand_requires_api_key(capsys, tmp_path, monkeypatch) -> None:
    """`cli ingest` without FINNHUB_API_KEY returns exit 2 with a friendly message."""
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db")
    rc = main(["ingest"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "FINNHUB_API_KEY" in err


def test_ingest_subcommand_happy_path(capsys, tmp_path, monkeypatch) -> None:
    """`cli ingest` with a key prints JSON counts and returns 0.

    The Finnhub client is real but the network is mocked via
    ``run_daily_ingest`` so this stays hermetic.
    """
    monkeypatch.setenv("FINNHUB_API_KEY", "stub-key")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db")

    fake_counts: dict[str, object] = {
        "general_news": 3,
        "company_news": 0,
        "market_prices": 5,
        "sector_prices": 0,
        "company_prices": 0,
        "scored": 3,
        "predictions": 1,
        "failures": [],
    }

    with patch(
        "finn_predictor.ingestion.jobs.run_daily_ingest",
        return_value=fake_counts,
    ) as ingest_mock:
        rc = main(["ingest"])

    assert rc == 0
    out = capsys.readouterr().out
    assert json.loads(out) == fake_counts
    # And the lazy import + dispatch actually reached run_daily_ingest.
    assert ingest_mock.call_count == 1
    call_kwargs = ingest_mock.call_args.kwargs
    assert {"session", "gateway", "scorer"} <= set(call_kwargs)


def test_hash_password_reads_from_stdin(capsys, monkeypatch) -> None:
    """Without an argument, hash-password prompts via getpass.getpass."""
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "hunter2-2026")
    rc = main(["hash-password"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip().startswith("$2")


def test_hash_password_stdin_cancelled(capsys, monkeypatch) -> None:
    """Ctrl-C / EOF at the prompt exits 2 with a cancelled message."""
    def _raise_eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("getpass.getpass", _raise_eof)
    rc = main(["hash-password"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "cancelled" in err.lower()


def test_hash_password_rejects_too_long(capsys) -> None:
    """A password longer than MAX_PASSWORD_LEN is rejected via the ValueError branch."""
    too_long = "a" * 300
    rc = main(["hash-password", too_long])
    err = capsys.readouterr().err
    assert rc == 2
    assert "at most" in err.lower()


def test_retrain_subcommand_honours_activate_yes(capsys, tmp_path, monkeypatch) -> None:
    """--activate yes is parsed and threaded into train_weights even when it fails."""
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db")
    rc = main(["retrain", "--n-calls", "1", "--activate", "yes"])
    # Empty DB → NotEnoughDataError → rc=3. The branch we care about
    # (activate=="yes" → activate_arg=True) is exercised before the raise.
    assert rc == 3
    assert "not enough" in capsys.readouterr().err.lower()


def test_retrain_subcommand_honours_activate_no(capsys, tmp_path, monkeypatch) -> None:
    """--activate no is parsed and threaded into train_weights even when it fails."""
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db")
    rc = main(["retrain", "--n-calls", "1", "--activate", "no"])
    assert rc == 3


def test_fit_classifier_subcommand_needs_data(capsys, tmp_path, monkeypatch) -> None:
    """Empty DB → rc=3 + 'need at least' message."""
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db")
    rc = main(["fit-classifier"])
    err = capsys.readouterr().err
    assert rc == 3
    assert "need at least" in err.lower()


def test_fit_classifier_subcommand_success(tmp_path, monkeypatch, capsys) -> None:
    """With a seeded dataset, fit-classifier prints calibration JSON and rc=0."""
    db_url = f"sqlite:///{tmp_path}/f.db"
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", db_url)

    from finn_predictor.sentiment.vader import VaderScorer
    from finn_predictor.storage import create_engine_and_session, init_db
    from finn_predictor.storage.models import Prediction, PredictionOutcome
    from finn_predictor.storage.repo import save_outcome, save_prediction

    mv = VaderScorer().model_version
    engine, SL = create_engine_and_session(db_url)
    init_db(engine)
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    with SL() as s:
        for i in range(20):
            sentiment = 0.5 if i < 10 else -0.5
            realised = 0.01 if i < 10 else -0.01
            p = save_prediction(
                s,
                Prediction(
                    target_symbol="^GSPC",
                    prediction_date=base + timedelta(days=i),
                    label="UP" if sentiment > 0 else "DOWN",
                    confidence=0.5,
                    sentiment_index=sentiment,
                    article_count=5,
                    model_version=mv,
                ),
            )
            save_outcome(
                s,
                PredictionOutcome(
                    prediction_id=p.id, realised_return=realised, hit=True
                ),
            )
    engine.dispose()

    rc = main(["fit-classifier"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["beta"] > 0
    assert payload["n_samples"] == 20
    assert payload["model_version"] == mv


def test_fit_magnitude_subcommand_needs_data(capsys, tmp_path, monkeypatch) -> None:
    """Empty DB → rc=3 + 'need at least' message."""
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db")
    rc = main(["fit-magnitude"])
    err = capsys.readouterr().err
    assert rc == 3
    assert "need at least" in err.lower()


def test_fit_magnitude_subcommand_success(tmp_path, monkeypatch, capsys) -> None:
    """With seeded outcomes, fit-magnitude prints fit JSON and rc=0."""
    import random

    db_url = f"sqlite:///{tmp_path}/f.db"
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", db_url)

    from finn_predictor.sentiment.vader import VaderScorer
    from finn_predictor.storage import create_engine_and_session, init_db
    from finn_predictor.storage.models import Prediction, PredictionOutcome
    from finn_predictor.storage.repo import save_outcome, save_prediction

    mv = VaderScorer().model_version
    engine, SL = create_engine_and_session(db_url)
    init_db(engine)
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    random.seed(7)
    with SL() as s:
        for i in range(40):
            sentiment = random.uniform(-0.8, 0.8)
            realised = 0.01 * sentiment + random.gauss(0.0, 0.015)
            p = save_prediction(
                s,
                Prediction(
                    target_symbol="^GSPC",
                    prediction_date=base + timedelta(days=i),
                    label="UP" if sentiment > 0 else "DOWN",
                    confidence=0.5,
                    sentiment_index=sentiment,
                    article_count=5,
                    model_version=mv,
                ),
            )
            save_outcome(
                s,
                PredictionOutcome(
                    prediction_id=p.id, realised_return=realised, hit=True
                ),
            )
    engine.dispose()

    rc = main(["fit-magnitude"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["n_samples"] == 40
    assert payload["model_version"] == mv
    taus = [f["tau"] for f in payload["fits"]]
    assert taus == [0.10, 0.50, 0.90]


def test_refresh_constituents_requires_api_key(capsys, tmp_path, monkeypatch) -> None:
    """Without FINNHUB_API_KEY → rc=2 with a friendly message."""
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db")
    rc = main(["refresh-constituents"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "FINNHUB_API_KEY" in err


def test_refresh_constituents_rejects_unseeded_db(capsys, tmp_path, monkeypatch) -> None:
    """No sectors in the DB → rc=2 (we don't auto-seed here)."""
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db")
    rc = main(["refresh-constituents"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no sectors seeded" in err.lower()


def _seed_sectors_db(db_url: str) -> None:
    """Seed the default 11 sectors so the refresh CLI has something to iterate."""
    from finn_predictor.storage import create_engine_and_session, init_db
    from finn_predictor.storage.repo import ensure_default_sectors

    engine, SL = create_engine_and_session(db_url)
    init_db(engine)
    with SL() as s:
        ensure_default_sectors(s)
    engine.dispose()


def test_refresh_constituents_happy_path(tmp_path, monkeypatch, capsys) -> None:
    """Mocked /etf/holdings → every sector's constituents land as ETF_HOLDING rows."""
    db_url = f"sqlite:///{tmp_path}/f.db"
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", db_url)
    _seed_sectors_db(db_url)

    # Patch the gateway's etfs_holdings to a deterministic payload per ETF.
    fake_payload = {
        "holdings": [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corp."},
            {"symbol": "NVDA", "name": "NVIDIA Corp."},
        ]
    }
    with patch(
        "finn_predictor.ingestion.client.FinnhubGateway.etfs_holdings",
        return_value=fake_payload,
    ):
        rc = main(["refresh-constituents", "--limit", "3"])

    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    refreshed = {entry["etf"] for entry in parsed["refreshed"]}
    # All 11 default sectors should appear.
    assert {"XLK", "XLE", "XLF", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"} <= refreshed
    # 3 holdings per sector × 11 sectors = 33 row-inserts confirmed via the
    # summary's per-sector counts.
    assert all(entry["holdings"] == 3 for entry in parsed["refreshed"])

    # And the DB now actually has the rows.
    from finn_predictor.storage import create_engine_and_session, init_db
    from finn_predictor.storage.models import RelatedEntity
    engine, SL = create_engine_and_session(db_url)
    init_db(engine)
    with SL() as s:
        n = s.query(RelatedEntity).filter(
            RelatedEntity.relationship == "ETF_HOLDING"
        ).count()
    engine.dispose()
    assert n == 33


def test_refresh_constituents_isolates_per_sector_failures(
    tmp_path, monkeypatch, capsys
) -> None:
    """A 403 on one ETF doesn't block the others — failure goes in summary."""
    db_url = f"sqlite:///{tmp_path}/f.db"
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", db_url)
    _seed_sectors_db(db_url)

    from finn_predictor.ingestion.client import IngestionError

    def _holdings(symbol):
        if symbol == "XLE":
            raise IngestionError("FinnhubAPI 403: gated on free tier")
        return {
            "holdings": [{"symbol": "GENERIC", "name": "Generic Inc."}],
        }

    with patch(
        "finn_predictor.ingestion.client.FinnhubGateway.etfs_holdings",
        side_effect=_holdings,
    ):
        rc = main(["refresh-constituents"])

    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    failures = [f for f in parsed["failures"] if f["etf"] == "XLE"]
    assert len(failures) == 1
    # And the other 10 still landed in `refreshed`.
    refreshed = {entry["etf"] for entry in parsed["refreshed"]}
    assert "XLE" not in refreshed
    assert len(refreshed) == 10


def test_refresh_constituents_filter_by_etf(tmp_path, monkeypatch, capsys) -> None:
    """--etf XLK --etf XLV restricts the run to those two sectors."""
    db_url = f"sqlite:///{tmp_path}/f.db"
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", db_url)
    _seed_sectors_db(db_url)

    with patch(
        "finn_predictor.ingestion.client.FinnhubGateway.etfs_holdings",
        return_value={"holdings": [{"symbol": "X", "name": "X"}]},
    ):
        rc = main(["refresh-constituents", "--etf", "XLK", "--etf", "XLV"])

    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    refreshed = {entry["etf"] for entry in parsed["refreshed"]}
    assert refreshed == {"XLK", "XLV"}
    skipped = set(parsed["skipped"])
    # Everything else got skipped — confirms the filter, not just an empty fan-out.
    assert "XLE" in skipped


def test_unknown_subcommand_returns_2(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db")
    with pytest.raises(SystemExit) as info:
        # argparse exits with SystemExit on unknown subcommands.
        main(["nope"])
    assert info.value.code == 2

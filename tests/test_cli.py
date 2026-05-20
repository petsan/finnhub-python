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


def test_unknown_subcommand_returns_2(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db")
    with pytest.raises(SystemExit) as info:
        # argparse exits with SystemExit on unknown subcommands.
        main(["nope"])
    assert info.value.code == 2

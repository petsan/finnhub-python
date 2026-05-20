"""Tests for additive schema migrations applied by ``init_db``.

When a column lands on the Prediction table after a deploy is already
running, an older DB has the older schema. ``init_db`` issues
``ALTER TABLE ADD COLUMN`` for the missing ones so the deploy upgrades
in place without a manual migration step. The tests below lock that
behaviour in.
"""

from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from finn_predictor.storage import create_engine_and_session, init_db


def _make_old_prediction_table(url: str) -> None:
    """Create the predictions table without the magnitude columns."""
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE predictions ("
                "  id INTEGER PRIMARY KEY,"
                "  target_symbol TEXT,"
                "  prediction_date DATETIME,"
                "  label TEXT,"
                "  confidence FLOAT,"
                "  sentiment_index FLOAT,"
                "  article_count INTEGER,"
                "  model_version TEXT,"
                "  created_at DATETIME"
                ")"
            )
        )
    engine.dispose()


def _columns(engine, table: str) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns(table)}


def test_init_db_adds_magnitude_columns_to_old_predictions(tmp_path) -> None:
    """An old-schema predictions table picks up the three magnitude columns."""
    db = tmp_path / "old.db"
    url = f"sqlite:///{db}"
    _make_old_prediction_table(url)

    engine, _ = create_engine_and_session(url)
    init_db(engine)
    cols = _columns(engine, "predictions")
    engine.dispose()

    for new_col in ("expected_return_p10", "expected_return_p50", "expected_return_p90"):
        assert new_col in cols, f"{new_col} missing after migration; got {cols}"


def test_init_db_migration_is_idempotent(tmp_path) -> None:
    """Running init_db a second time doesn't re-add columns or raise."""
    db = tmp_path / "old.db"
    url = f"sqlite:///{db}"
    _make_old_prediction_table(url)

    engine, _ = create_engine_and_session(url)
    init_db(engine)
    init_db(engine)  # second call — no exception, no duplicate columns
    cols = _columns(engine, "predictions")
    engine.dispose()

    assert sum(c.startswith("expected_return_") for c in cols) == 3


def test_init_db_noop_on_fresh_install(tmp_path) -> None:
    """A fresh DB already has the columns; the migration step is a no-op."""
    db = tmp_path / "fresh.db"
    url = f"sqlite:///{db}"
    engine, _ = create_engine_and_session(url)
    init_db(engine)
    cols = _columns(engine, "predictions")
    engine.dispose()

    # Fresh install: every Prediction column is present via create_all,
    # so no ALTER TABLE was needed. Just confirm the surface looks right.
    expected_subset = {
        "expected_return_p10",
        "expected_return_p50",
        "expected_return_p90",
        "label",
        "confidence",
        "sentiment_index",
    }
    assert expected_subset <= cols

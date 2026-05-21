"""Engine + session factory.

Kept tiny on purpose: callers get back a (engine, session_factory) pair and
own the session lifecycle. We do not stash anything in module globals so the
test suite can spin up isolated in-memory databases per test.
"""

from __future__ import annotations

from typing import Tuple

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from finn_predictor.storage.models import Base


# Columns that landed on the Prediction table after the original
# schema was cut. ``init_db`` ALTERs them onto an existing table when
# they're missing, so a DB written by an older Finn-Predictor version
# upgrades in place on the next start without a manual migration.
_PREDICTION_LATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("expected_return_p10", "FLOAT"),
    ("expected_return_p50", "FLOAT"),
    ("expected_return_p90", "FLOAT"),
)


def _add_missing_columns(engine: Engine) -> None:
    """Idempotently ALTER TABLE for columns that landed in later versions.

    SQLAlchemy's ``create_all`` only adds *tables*, never columns, so a
    DB populated by an earlier Finn-Predictor would silently miss the
    new ones and crash later in the predictor. We inspect the live
    schema and issue ``ALTER TABLE ADD COLUMN`` only for the ones that
    aren't already there — both SQLite and Postgres support this
    syntax. New / fresh installs are no-ops because ``create_all``
    already laid down the full schema.
    """
    inspector = inspect(engine)
    if "predictions" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("predictions")}
    missing = [(c, t) for c, t in _PREDICTION_LATE_COLUMNS if c not in existing]
    if not missing:
        return

    with engine.begin() as conn:
        for col, sql_type in missing:
            conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {col} {sql_type}"))


def create_engine_and_session(
    database_url: str, echo: bool = False
) -> Tuple[Engine, sessionmaker[Session]]:
    """Create a SQLAlchemy engine and a bound sessionmaker.

    For SQLite URLs we set ``check_same_thread=False`` so the APScheduler
    background thread can share the engine with the Streamlit main thread.
    """
    connect_args: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(database_url, echo=echo, connect_args=connect_args, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return engine, SessionLocal


def init_db(engine: Engine) -> None:
    """Create all tables + apply additive column migrations. Idempotent."""
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)

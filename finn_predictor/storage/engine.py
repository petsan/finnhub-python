"""Engine + session factory.

Kept tiny on purpose: callers get back a (engine, session_factory) pair and
own the session lifecycle. We do not stash anything in module globals so the
test suite can spin up isolated in-memory databases per test.
"""

from __future__ import annotations

from typing import Tuple

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from finn_predictor.storage.models import Base


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
    """Create all tables. Idempotent."""
    Base.metadata.create_all(engine)

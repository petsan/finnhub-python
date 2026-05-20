"""Tiny argv → operation CLI used by the Docker entrypoint.

Three operations:

* ``serve`` (default): print the Streamlit command users should run.
  The Docker entrypoint runs Streamlit itself; this branch exists so
  ``python -m finn_predictor.cli`` is useful even outside Docker.
* ``reset-db``: drop every Finn-Predictor table and recreate the schema
  (idempotent: safe even when the file doesn't exist yet).
* ``retrain``: run one ``train_weights`` cycle and print the report
  (so cron / docker exec can trigger it without the UI).

The DB URL is read from ``FINN_PREDICTOR_DB_URL`` (defaults to
``sqlite:///finn_predictor.db``), matching the Streamlit app.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from typing import Sequence

from finn_predictor.config import DEFAULT_DB_URL, load_settings
from finn_predictor.storage import create_engine_and_session, init_db
from finn_predictor.storage.models import Base


logger = logging.getLogger("finn_predictor.cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="finn-predictor",
        description="Finn-Predictor admin / Docker-entrypoint helper.",
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("serve", help="Print the Streamlit launch command.")

    reset = sub.add_parser(
        "reset-db",
        help="Drop and recreate every Finn-Predictor table. Destroys data.",
    )
    reset.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt. Required in non-interactive shells.",
    )

    retrain = sub.add_parser(
        "retrain",
        help="Run train_weights once and print the report as JSON.",
    )
    retrain.add_argument(
        "--n-calls", type=int, default=30,
        help="Bayesian-optimisation iterations (default 30).",
    )
    retrain.add_argument(
        "--activate", choices=["auto", "yes", "no"], default="auto",
        help="Activation override. 'auto' uses the persisted policy.",
    )

    hashp = sub.add_parser(
        "hash-password",
        help=(
            "Print a bcrypt hash suitable for "
            "FINN_PREDICTOR_PASSWORD_HASH. Reads the password "
            "from stdin if not given on the command line."
        ),
    )
    hashp.add_argument(
        "password",
        nargs="?",
        default=None,
        help="The plaintext password. Omit to read from stdin.",
    )

    sub.add_parser(
        "ingest",
        help=(
            "Run one daily-ingest cycle headlessly. Reads "
            "FINNHUB_API_KEY from env. Suitable for cron / "
            "scheduled-task sidecar."
        ),
    )

    return p


def _resolve_db_url() -> str:
    """DB URL from env, falling back to the same default the UI uses."""
    try:
        return load_settings(require_api_key=False).database_url
    except Exception:  # pragma: no cover
        return DEFAULT_DB_URL


def cmd_reset_db(*, yes: bool) -> int:
    """Drop and recreate every table. Returns shell exit code."""
    url = _resolve_db_url()
    if not yes:
        print(
            f"refusing to reset {url} without --yes "
            "(this destroys data)",
            file=sys.stderr,
        )
        return 2
    engine, _ = create_engine_and_session(url)
    Base.metadata.drop_all(engine)
    init_db(engine)
    print(f"reset-db: schema recreated at {url}")
    return 0


def cmd_retrain(*, n_calls: int, activate: str) -> int:
    """Run one training cycle. Returns shell exit code."""
    # Lazy import so 'serve' / 'reset-db' don't drag skopt in.
    from finn_predictor.learning import train_weights
    from finn_predictor.learning.train import NotEnoughDataError

    url = _resolve_db_url()
    engine, SessionLocal = create_engine_and_session(url)
    init_db(engine)

    activate_arg: bool | None
    if activate == "yes":
        activate_arg = True
    elif activate == "no":
        activate_arg = False
    else:
        activate_arg = None

    with SessionLocal() as session:
        try:
            report = train_weights(
                session, n_calls=n_calls, activate=activate_arg,
            )
        except NotEnoughDataError as exc:
            print(f"retrain: not enough data — {exc}", file=sys.stderr)
            return 3

    out = asdict(report)
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_hash_password(plaintext: str | None) -> int:
    """Print a bcrypt hash for FINN_PREDICTOR_PASSWORD_HASH."""
    from finn_predictor.security import (
        MAX_PASSWORD_LEN,
        MIN_PASSWORD_LEN,
        hash_password,
    )

    if plaintext is None:
        # Read from stdin so the password doesn't show up in shell history.
        import getpass

        try:
            plaintext = getpass.getpass("Password: ")
        except (EOFError, KeyboardInterrupt):
            print("hash-password: cancelled", file=sys.stderr)
            return 2

    if not plaintext or len(plaintext) < MIN_PASSWORD_LEN:
        print(
            f"hash-password: password must be at least "
            f"{MIN_PASSWORD_LEN} characters",
            file=sys.stderr,
        )
        return 2

    try:
        h = hash_password(plaintext)
    except ValueError as exc:
        print(f"hash-password: {exc}", file=sys.stderr)
        return 2
    print(h)
    print(
        "\nSet this as FINN_PREDICTOR_PASSWORD_HASH on the server "
        "(quote it because the hash contains $):\n"
        f"  export FINN_PREDICTOR_PASSWORD_HASH='{h}'",
        file=sys.stderr,
    )
    return 0


def cmd_ingest() -> int:
    """Run one daily-ingest cycle headlessly."""
    # Lazy import — keeps CLI import-time cheap for the lighter
    # subcommands.
    from finn_predictor.config import load_settings
    from finn_predictor.ingestion.client import FinnhubGateway, RateLimiter
    from finn_predictor.ingestion.jobs import run_daily_ingest
    from finn_predictor.sentiment.vader import VaderScorer
    from finnhub import Client as FinnhubClient

    try:
        settings = load_settings()  # requires FINNHUB_API_KEY
    except RuntimeError as exc:
        print(f"ingest: {exc}", file=sys.stderr)
        return 2

    url = _resolve_db_url()
    engine, SessionLocal = create_engine_and_session(url)
    init_db(engine)

    client = FinnhubClient(api_key=settings.finnhub_api_key)
    # Same proxy bypass + close pattern as the UI's run_ingestion_with_key.
    try:
        client._session.trust_env = False
    except AttributeError:  # pragma: no cover
        pass

    try:
        gateway = FinnhubGateway(
            client=client,
            rate_limiter=RateLimiter(settings.rate_limit_per_minute),
        )
        with SessionLocal() as session:
            counts = run_daily_ingest(
                session=session,
                gateway=gateway,
                scorer=VaderScorer(),
            )
    finally:
        client.close()

    print(json.dumps(counts, indent=2, default=str))
    return 0


def cmd_serve() -> int:
    """Print the Streamlit launch command; never starts it itself.

    The Docker entrypoint runs ``streamlit run …`` directly. This
    subcommand exists so a user invoking the CLI by hand sees an
    actionable hint instead of hanging on an exec.
    """
    print(
        "Run the UI with:\n"
        "  streamlit run finn_predictor/ui/app.py --server.port 8501\n"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Honour FINN_PREDICTOR_LOG_FORMAT / _LOG_LEVEL. The setup_logging
    # call also attaches the SecretScrubFilter to every handler so log
    # records can't accidentally leak Finnhub tokens or bcrypt hashes.
    from finn_predictor.logging_config import setup_logging

    setup_logging(force=True)

    cmd = args.command or "serve"
    if cmd == "serve":
        return cmd_serve()
    if cmd == "reset-db":
        return cmd_reset_db(yes=bool(args.yes))
    if cmd == "retrain":
        return cmd_retrain(n_calls=int(args.n_calls), activate=args.activate)
    if cmd == "hash-password":
        return cmd_hash_password(args.password)
    if cmd == "ingest":
        return cmd_ingest()
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

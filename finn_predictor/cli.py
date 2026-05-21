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

from sqlalchemy import select

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

    fit = sub.add_parser(
        "fit-classifier",
        help=(
            "Fit the logistic-regression calibration on closed predictions "
            "and persist it. Active once FINN_PREDICTOR_CLASSIFIER=logreg."
        ),
    )
    fit.add_argument(
        "--target-symbol", default=None,
        help=(
            "Restrict the training set to one target symbol "
            "(e.g. '^GSPC' or 'AAPL'). Default fits across all."
        ),
    )

    fit_mag = sub.add_parser(
        "fit-magnitude",
        help=(
            "Fit the quantile-band magnitude calibration on closed predictions "
            "and persist it. Active once FINN_PREDICTOR_MAGNITUDE=quantile."
        ),
    )
    fit_mag.add_argument(
        "--target-symbol", default=None,
        help=(
            "Restrict the training set to one target symbol "
            "(e.g. '^GSPC' or 'AAPL'). Default fits across all."
        ),
    )

    refresh = sub.add_parser(
        "refresh-constituents",
        help=(
            "Headless equivalent of clicking 'Refresh constituents' in the "
            "Focus → Sector mode for every seeded sector. Calls Finnhub's "
            "/etf/holdings for each ETF and caches the result as "
            "RelatedEntity(ETF_HOLDING) rows. Required prerequisite for "
            "sector-level predictions and the cap-weighting fan-out."
        ),
    )
    refresh.add_argument(
        "--etf",
        action="append",
        default=None,
        help=(
            "Restrict to one or more specific ETF symbols (repeatable). "
            "Default refreshes every sector in the DB."
        ),
    )
    refresh.add_argument(
        "--limit", type=int, default=25,
        help="Maximum constituents to cache per sector (default 25).",
    )

    # PR-4: affinity / competitor curation subcommands.
    promote = sub.add_parser(
        "promote-competitor",
        help=(
            "Mark a (symbol, peer_symbol) pair as COMPETITOR in the "
            "relationship cache. Operator-curated; bypasses the "
            "industry-match check used by refresh-competitors."
        ),
    )
    promote.add_argument("symbol", help="The target ticker (e.g. AAPL).")
    promote.add_argument("peer_symbol", help="The competitor's ticker (e.g. MSFT).")

    demote = sub.add_parser(
        "demote-competitor",
        help=(
            "Remove a COMPETITOR row from the relationship cache. "
            "The matching PEER row (if any) is preserved."
        ),
    )
    demote.add_argument("symbol", help="The target ticker (e.g. AAPL).")
    demote.add_argument("peer_symbol", help="The peer ticker to demote.")

    refresh_comp = sub.add_parser(
        "refresh-competitors",
        help=(
            "Auto-seed COMPETITOR rows for one or more tickers by "
            "matching their PEER rows on Finnhub finnhubIndustry. "
            "Requires FINNHUB_API_KEY. Per-ticker failure isolation."
        ),
    )
    refresh_comp.add_argument(
        "--symbol",
        action="append",
        required=True,
        help=(
            "Ticker to seed competitors for (repeatable). At least "
            "one is required."
        ),
    )

    # PR-6: investment themes.
    add_theme = sub.add_parser(
        "add-theme",
        help=(
            "Register an InvestmentTheme without an immediate Finnhub "
            "fetch. Use refresh-themes afterward to populate "
            "constituents."
        ),
    )
    add_theme.add_argument("theme_code", help="Finnhub theme code (e.g. cyberSecurity).")
    add_theme.add_argument(
        "--name", default=None,
        help="Display name (default: title-case render of theme_code).",
    )
    add_theme.add_argument(
        "--description", default=None,
        help="Optional human-readable description.",
    )

    refresh_themes = sub.add_parser(
        "refresh-themes",
        help=(
            "Pull constituents for every registered InvestmentTheme "
            "via Finnhub /stock/investment-theme. Requires "
            "FINNHUB_API_KEY. Falls back to the curated default theme "
            "list when no themes are registered."
        ),
    )
    refresh_themes.add_argument(
        "--theme",
        action="append",
        default=None,
        help=(
            "Restrict to specific theme codes (repeatable). Default "
            "refreshes every theme already in the DB; if none, uses "
            "the curated DEFAULT_THEME_CODES list."
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
    from finn_predictor.sentiment import get_scorer, warn_if_scorer_mismatch
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

    # FINN_PREDICTOR_SCORER picks the live scorer (default 'vader').
    # FinBERT pulls torch/transformers lazily — surface a friendly
    # message if it's not installed rather than a raw ImportError.
    try:
        scorer = get_scorer(settings.scorer_name)
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        print(
            f"ingest: scorer {settings.scorer_name!r} requested but its "
            f"dependencies aren't installed ({exc}). Install with "
            f"`pip install torch transformers` or unset "
            f"FINN_PREDICTOR_SCORER.",
            file=sys.stderr,
        )
        client.close()
        return 2

    try:
        gateway = FinnhubGateway(
            client=client,
            rate_limiter=RateLimiter(settings.rate_limit_per_minute),
        )
        with SessionLocal() as session:
            # Surface a one-line mismatch warning when the live scorer
            # doesn't match the most-recent prediction's model_version.
            warn_if_scorer_mismatch(session, active_scorer=scorer)
            counts = run_daily_ingest(
                session=session,
                gateway=gateway,
                scorer=scorer,
            )
    finally:
        client.close()

    print(json.dumps(counts, indent=2, default=str))
    return 0


def cmd_fit_classifier(*, target_symbol: str | None) -> int:
    """Fit + persist the logreg calibration. Returns shell exit code."""
    from finn_predictor.predictor.classifier import (
        NotEnoughCalibrationDataError,
        fit_logreg_calibration,
        save_calibration,
    )
    from finn_predictor.sentiment import resolve_active_scorer

    url = _resolve_db_url()
    engine, SessionLocal = create_engine_and_session(url)
    init_db(engine)

    # The training set is filtered by the active scorer's model_version
    # so predictions written by VADER and FinBERT don't get mixed.
    model_version = resolve_active_scorer().model_version

    with SessionLocal() as session:
        try:
            calibration = fit_logreg_calibration(
                session,
                model_version=model_version,
                target_symbol=target_symbol,
            )
        except NotEnoughCalibrationDataError as exc:
            print(f"fit-classifier: {exc}", file=sys.stderr)
            return 3
        save_calibration(session, calibration)

    print(
        json.dumps(
            {
                "beta": calibration.beta,
                "intercept": calibration.intercept,
                "n_samples": calibration.n_samples,
                "model_version": model_version,
                "target_symbol": target_symbol,
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_fit_magnitude(*, target_symbol: str | None) -> int:
    """Fit + persist the quantile-band magnitude calibration."""
    from finn_predictor.predictor.magnitude import (
        NotEnoughMagnitudeDataError,
        fit_quantile_calibration,
        save_calibration,
    )
    from finn_predictor.sentiment import resolve_active_scorer

    url = _resolve_db_url()
    engine, SessionLocal = create_engine_and_session(url)
    init_db(engine)

    # Same model_version filter as fit-classifier so VADER + FinBERT
    # predictions never get mixed in one calibration.
    model_version = resolve_active_scorer().model_version

    with SessionLocal() as session:
        try:
            calibration = fit_quantile_calibration(
                session,
                model_version=model_version,
                target_symbol=target_symbol,
            )
        except NotEnoughMagnitudeDataError as exc:
            print(f"fit-magnitude: {exc}", file=sys.stderr)
            return 3
        save_calibration(session, calibration)

    print(
        json.dumps(
            {
                "fits": [
                    {"tau": f.tau, "intercept": f.intercept, "beta": f.beta}
                    for f in calibration.fits
                ],
                "n_samples": calibration.n_samples,
                "feature_name": calibration.feature_name,
                "model_version": model_version,
                "target_symbol": target_symbol,
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_refresh_constituents(
    *, etfs: list[str] | None, limit: int
) -> int:
    """Cache ETF_HOLDING rows for every (or selected) sector.

    Headless mirror of the Focus tab's *Refresh constituents* button.
    Iterates the Sector table, calls Finnhub /etf/holdings per ETF,
    upserts the result as RelatedEntity(ETF_HOLDING) rows so subsequent
    daily ingests can produce sector predictions and the cap-weighting
    fan-out has constituents to walk.

    Exits 2 when FINNHUB_API_KEY is missing or no matching sectors
    exist; 0 otherwise (per-sector failures land in the JSON output
    rather than the rc, matching the resilient-ingest pattern).
    """
    from finn_predictor.config import load_settings
    from finn_predictor.ingestion.client import FinnhubGateway, RateLimiter
    from finn_predictor.predictor.focus import refresh_sector_constituents
    from finn_predictor.storage.repo import all_sectors
    from finnhub import Client as FinnhubClient

    try:
        settings = load_settings()  # requires FINNHUB_API_KEY
    except RuntimeError as exc:
        print(f"refresh-constituents: {exc}", file=sys.stderr)
        return 2

    url = _resolve_db_url()
    engine, SessionLocal = create_engine_and_session(url)
    init_db(engine)

    # Normalise the ETF filter once. None/empty list means "everything".
    wanted = {e.strip().upper() for e in (etfs or []) if e.strip()}

    client = FinnhubClient(api_key=settings.finnhub_api_key)
    try:
        client._session.trust_env = False
    except AttributeError:  # pragma: no cover
        pass

    summary: dict[str, object] = {
        "refreshed": [],
        "failures": [],
        "skipped": [],
    }

    try:
        gateway = FinnhubGateway(
            client=client,
            rate_limiter=RateLimiter(settings.rate_limit_per_minute),
        )
        with SessionLocal() as session:
            sectors = list(all_sectors(session))
            if not sectors:
                print(
                    "refresh-constituents: no sectors seeded; run "
                    "`ingest` first (or call `ensure_default_sectors`)",
                    file=sys.stderr,
                )
                return 2

            for sector in sectors:
                etf = sector.etf_symbol.strip().upper()
                if wanted and etf not in wanted:
                    summary["skipped"].append(etf)
                    continue

                # Resilient: a per-sector failure (free-tier 403 on
                # /etf/holdings, network blip, etc.) doesn't block its
                # peers — same pattern as run_daily_ingest's _try.
                result = refresh_sector_constituents(
                    session, gateway, etf_symbol=etf, limit=limit
                )
                if result.holdings_added > 0:
                    summary["refreshed"].append(
                        {"etf": etf, "holdings": result.holdings_added}
                    )
                # refresh_sector_constituents collects IngestionError
                # into result.failures rather than raising — drain it
                # into the summary so the operator sees it.
                for f in result.failures:
                    summary["failures"].append({"etf": etf, **f})
    finally:
        client.close()

    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_promote_competitor(*, symbol: str, peer_symbol: str) -> int:
    """Manually mark ``peer_symbol`` as a COMPETITOR of ``symbol``.

    No API call — pure DB mutation. Exits 2 on validation failure.
    """
    from finn_predictor.predictor.affinity import promote_peer_to_competitor

    url = _resolve_db_url()
    engine, SessionLocal = create_engine_and_session(url)
    init_db(engine)
    try:
        with SessionLocal() as session:
            row = promote_peer_to_competitor(
                session, symbol=symbol, peer_symbol=peer_symbol
            )
    except ValueError as exc:
        print(f"promote-competitor: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "action": "promoted",
                "source_symbol": row.source_symbol,
                "related_symbol": row.related_symbol,
                "relationship": row.relationship,
            }
        )
    )
    return 0


def cmd_demote_competitor(*, symbol: str, peer_symbol: str) -> int:
    """Remove a COMPETITOR row. Exit 0 whether the row existed or not."""
    from finn_predictor.predictor.affinity import demote_competitor

    url = _resolve_db_url()
    engine, SessionLocal = create_engine_and_session(url)
    init_db(engine)
    try:
        with SessionLocal() as session:
            removed = demote_competitor(
                session, symbol=symbol, peer_symbol=peer_symbol
            )
    except ValueError as exc:
        print(f"demote-competitor: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"action": "demoted", "removed": removed}))
    return 0


def cmd_refresh_competitors(*, symbols: list[str]) -> int:
    """Auto-seed COMPETITOR rows for one or more tickers.

    Requires FINNHUB_API_KEY. Iterates ``symbols``, calling
    :func:`refresh_competitors` for each. Per-ticker failure isolation
    — a missing industry or network blip on one symbol doesn't block
    the rest. Prints a JSON summary; exits 2 only when the API key
    is missing.
    """
    from finn_predictor.config import load_settings
    from finn_predictor.ingestion.client import FinnhubGateway, RateLimiter
    from finn_predictor.predictor.affinity import refresh_competitors
    from finnhub import Client as FinnhubClient

    try:
        settings = load_settings()  # requires FINNHUB_API_KEY
    except RuntimeError as exc:
        print(f"refresh-competitors: {exc}", file=sys.stderr)
        return 2

    url = _resolve_db_url()
    engine, SessionLocal = create_engine_and_session(url)
    init_db(engine)

    client = FinnhubClient(api_key=settings.finnhub_api_key)
    try:
        client._session.trust_env = False
    except AttributeError:  # pragma: no cover
        pass

    summary: dict[str, object] = {"refreshed": [], "failures": []}

    try:
        gateway = FinnhubGateway(
            client=client,
            rate_limiter=RateLimiter(settings.rate_limit_per_minute),
        )
        with SessionLocal() as session:
            for sym in symbols:
                result = refresh_competitors(session, gateway, symbol=sym)
                summary["refreshed"].append(
                    {
                        "symbol": result.symbol,
                        "competitors_added": result.competitors_added,
                        "peers_considered": result.peers_considered,
                        "skipped_no_industry": result.skipped_no_industry,
                    }
                )
                for f in result.failures:
                    summary["failures"].append({"symbol": result.symbol, **f})
    finally:
        client.close()

    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_add_theme(
    *, theme_code: str, name: str | None, description: str | None
) -> int:
    """Register an InvestmentTheme row. No API call. Exit 2 on validation."""
    from finn_predictor.predictor.affinity import add_investment_theme

    url = _resolve_db_url()
    engine, SessionLocal = create_engine_and_session(url)
    init_db(engine)
    try:
        with SessionLocal() as session:
            row = add_investment_theme(
                session,
                theme_code=theme_code,
                name=name,
                description=description,
            )
    except ValueError as exc:
        print(f"add-theme: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps({
            "action": "registered",
            "theme_code": row.theme_code,
            "name": row.name,
            "description": row.description,
        })
    )
    return 0


def cmd_refresh_themes(*, theme_codes: list[str] | None) -> int:
    """Pull THEME_MEMBER constituents for registered (or curated) themes."""
    from finn_predictor.config import load_settings
    from finn_predictor.ingestion.client import FinnhubGateway, RateLimiter
    from finn_predictor.predictor.affinity import refresh_investment_themes
    from finn_predictor.predictor.themes import DEFAULT_THEME_CODES
    from finn_predictor.storage.models import InvestmentTheme
    from finnhub import Client as FinnhubClient

    try:
        settings = load_settings()
    except RuntimeError as exc:
        print(f"refresh-themes: {exc}", file=sys.stderr)
        return 2

    url = _resolve_db_url()
    engine, SessionLocal = create_engine_and_session(url)
    init_db(engine)

    client = FinnhubClient(api_key=settings.finnhub_api_key)
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
            if theme_codes:
                codes = list(theme_codes)
            else:
                registered = list(session.scalars(select(InvestmentTheme)))
                codes = (
                    [t.theme_code for t in registered]
                    if registered else list(DEFAULT_THEME_CODES)
                )
            result = refresh_investment_themes(
                session, gateway, theme_codes=codes
            )
    finally:
        client.close()

    print(json.dumps({
        "themes_processed": result.themes_processed,
        "themes_added": result.themes_added,
        "members_added": result.members_added,
        "failures": result.failures,
    }, indent=2, default=str))
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
    if cmd == "fit-classifier":
        return cmd_fit_classifier(target_symbol=args.target_symbol)
    if cmd == "fit-magnitude":
        return cmd_fit_magnitude(target_symbol=args.target_symbol)
    if cmd == "refresh-constituents":
        return cmd_refresh_constituents(etfs=args.etf, limit=int(args.limit))
    if cmd == "promote-competitor":
        return cmd_promote_competitor(
            symbol=args.symbol, peer_symbol=args.peer_symbol
        )
    if cmd == "demote-competitor":
        return cmd_demote_competitor(
            symbol=args.symbol, peer_symbol=args.peer_symbol
        )
    if cmd == "refresh-competitors":
        return cmd_refresh_competitors(symbols=args.symbol)
    if cmd == "add-theme":
        return cmd_add_theme(
            theme_code=args.theme_code,
            name=args.name,
            description=args.description,
        )
    if cmd == "refresh-themes":
        return cmd_refresh_themes(theme_codes=args.theme)
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Runtime configuration loaded from environment variables.

We deliberately keep this module dependency-free so it can be imported from
both the Streamlit UI process and short-lived scripts (jobs, backtests, tests)
without surprises. The Settings dataclass is frozen so it cannot be mutated
after construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_DB_URL = "sqlite:///finn_predictor.db"


@dataclass(frozen=True)
class Settings:
    """Application configuration snapshot.

    Attributes:
        finnhub_api_key: API token sent to api.finnhub.io. Never logged.
        database_url:    SQLAlchemy URL. Defaults to a file-backed SQLite db.
        request_timeout: HTTP timeout in seconds for Finnhub calls.
        rate_limit_per_minute: Soft cap on calls per minute (Finnhub free tier
            is ~60/min; we leave headroom).
    """

    finnhub_api_key: str
    database_url: str = DEFAULT_DB_URL
    request_timeout: float = 15.0
    rate_limit_per_minute: int = 55


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build a Settings from an env mapping (defaults to ``os.environ``).

    Raises:
        RuntimeError: if ``FINNHUB_API_KEY`` is not set. We refuse to start
            up silently with a placeholder because Finnhub would just 401 and
            we'd write nothing to the DB.
    """
    source = os.environ if env is None else env
    api_key = source.get("FINNHUB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "FINNHUB_API_KEY is not set. Export it before starting "
            "finn-predictor (e.g. `export FINNHUB_API_KEY=...`)."
        )

    return Settings(
        finnhub_api_key=api_key,
        database_url=source.get("FINN_PREDICTOR_DB_URL", DEFAULT_DB_URL),
        request_timeout=float(source.get("FINN_PREDICTOR_TIMEOUT", "15")),
        rate_limit_per_minute=int(source.get("FINN_PREDICTOR_RATE_LIMIT", "55")),
    )

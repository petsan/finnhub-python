"""Centralised logging configuration.

One ``setup_logging()`` call configures the root logger so every module
(library + UI + CLI) emits records the same way. Two formats:

* **text** (default) — human-friendly stdout output, matches the
  previous behaviour. Suitable for local dev + ``docker compose logs``.
* **json** — one record per line, machine-parseable. Suitable for
  shipping to log aggregators (Loki / CloudWatch / Datadog).

The format is selected via ``FINN_PREDICTOR_LOG_FORMAT`` (``text`` or
``json``), with ``text`` as the default. The log level uses
``FINN_PREDICTOR_LOG_LEVEL`` (default ``INFO``).

A :class:`SecretScrubFilter` is attached to the root logger so any log
record containing what looks like a Finnhub token or a bcrypt hash gets
masked — defence in depth behind the explicit scrubbing on the
ingestion path.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

from finn_predictor.security import SecretScrubFilter


VALID_FORMATS = ("text", "json")
DEFAULT_FORMAT = "text"
DEFAULT_LEVEL = "INFO"


class _JsonFormatter(logging.Formatter):
    """Single-line JSON formatter — no nested objects, no exception traces.

    Exception traces are rendered as a single string under ``exc_info``
    so downstream log parsers don't have to deal with multi-line
    records.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _resolve_format(env: Optional[dict] = None) -> str:
    source = os.environ if env is None else env
    val = (source.get("FINN_PREDICTOR_LOG_FORMAT") or "").strip().lower()
    if val in VALID_FORMATS:
        return val
    return DEFAULT_FORMAT


def _resolve_level(env: Optional[dict] = None) -> int:
    source = os.environ if env is None else env
    name = (source.get("FINN_PREDICTOR_LOG_LEVEL") or DEFAULT_LEVEL).strip().upper()
    return getattr(logging, name, logging.INFO)


def setup_logging(
    *,
    fmt: Optional[str] = None,
    level: Optional[int] = None,
    env: Optional[dict] = None,
    force: bool = False,
) -> None:
    """Configure the root logger. Idempotent under most call patterns.

    Calling twice is safe — it replaces existing handlers when
    ``force=True`` (default behaviour in CLI / Streamlit boot), or no-ops
    when handlers already exist and ``force=False`` (library use).

    The :class:`SecretScrubFilter` is attached to **every** handler we
    add so it sees records before they're emitted regardless of which
    sink. Other handlers added later won't have the filter — callers
    that attach extras should add :class:`SecretScrubFilter` themselves.
    """
    chosen_fmt = fmt or _resolve_format(env)
    chosen_level = level if level is not None else _resolve_level(env)

    root = logging.getLogger()
    if root.handlers and not force:
        # Library mode — don't fight the host application's config.
        return

    # Tear down any existing handlers (only when force=True or empty).
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    if chosen_fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    handler.addFilter(SecretScrubFilter())
    root.addHandler(handler)
    root.setLevel(chosen_level)

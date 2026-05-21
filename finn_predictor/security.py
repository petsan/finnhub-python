"""Auth + scrub helpers used by the UI and CLI.

Two responsibilities:

* :func:`hash_password` / :func:`verify_password` — bcrypt wrappers used
  by the UI's password gate. Hashes are stored only in the
  ``FINN_PREDICTOR_PASSWORD_HASH`` environment variable, never in the
  database (so wiping the volume doesn't lock you out — just unset the
  env var).
* :class:`SecretScrubFilter` — a logging filter that masks anything
  that looks like a long opaque token, as a last-resort net behind the
  three explicit scrubbing layers we already have on the ingestion
  path. Best paired with :func:`finn_predictor.logging_config.setup_logging`.

Bcrypt is the only choice we make here on purpose: it's the safest
sane default, has a single Python wheel, and is widely understood. If
you want argon2 down the road, this module is the only place to swap.
"""

from __future__ import annotations

import logging
import os
import re

import bcrypt


# How long a password the gate accepts. 4 is short enough for "let me
# in"; 256 is more than anyone should ever set.
MIN_PASSWORD_LEN = 4
MAX_PASSWORD_LEN = 256


def hash_password(plaintext: str) -> str:
    """Return a bcrypt hash suitable for ``FINN_PREDICTOR_PASSWORD_HASH``.

    Raises :class:`ValueError` for empty / too-long input.
    """
    if not plaintext:
        raise ValueError("password must be non-empty")
    if len(plaintext) > MAX_PASSWORD_LEN:
        raise ValueError(
            f"password must be at most {MAX_PASSWORD_LEN} characters"
        )
    # bcrypt operates on bytes; the cost factor of 12 is the modern
    # default — slow enough to brute-force, fast enough to feel snappy.
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plaintext.encode("utf-8"), salt).decode("ascii")


def verify_password(plaintext: str, stored_hash: str) -> bool:
    """Constant-time check that ``plaintext`` matches ``stored_hash``.

    Returns ``False`` rather than raising on malformed hashes — the UI
    treats every wrong path identically to "wrong password" to avoid
    leaking which env var is broken.
    """
    if not plaintext or not stored_hash:
        return False
    try:
        return bcrypt.checkpw(
            plaintext.encode("utf-8"),
            stored_hash.encode("ascii"),
        )
    except (ValueError, TypeError):
        return False


def auth_enabled(env: dict[str, str] | None = None) -> bool:
    """True when ``FINN_PREDICTOR_PASSWORD_HASH`` is set to a non-empty value."""
    source = os.environ if env is None else env
    return bool((source.get("FINN_PREDICTOR_PASSWORD_HASH") or "").strip())


def current_password_hash(env: dict[str, str] | None = None) -> str:
    """Return the configured bcrypt hash (empty string when unset)."""
    source = os.environ if env is None else env
    return (source.get("FINN_PREDICTOR_PASSWORD_HASH") or "").strip()


# ---------------------------------------------------------------------------
# Defence-in-depth logging filter.
# ---------------------------------------------------------------------------


# Anything that looks like a Finnhub API token: 40 lowercase-alphanumeric
# chars. Their tokens are 40 chars of [0-9a-z] (e.g. "d8...h0"). The
# regex is intentionally narrow so legitimate alphanumeric strings of
# different lengths don't get masked.
_FINNHUB_TOKEN_RE = re.compile(r"\b[0-9a-z]{40}\b")
# Anything that smells like a bcrypt hash (the prefix is well-defined).
_BCRYPT_HASH_RE = re.compile(r"\$2[aby]\$\d{2}\$[A-Za-z0-9./]{53}")


class SecretScrubFilter(logging.Filter):
    """Mask long opaque tokens in log records.

    Applied to the root logger by :func:`logging_config.setup_logging`.
    Catches anything the three explicit scrubbing layers
    (gateway → run_ingestion_with_key → UI display) missed. Cheap
    enough to run on every record: two compiled regexes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Don't mutate the original args structure; build the final
        # message and replace .msg + .args so downstream formatters use
        # the scrubbed text.
        try:
            text = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        scrubbed = _FINNHUB_TOKEN_RE.sub("<REDACTED-TOKEN>", text)
        scrubbed = _BCRYPT_HASH_RE.sub("<REDACTED-HASH>", scrubbed)
        if scrubbed != text:
            record.msg = scrubbed
            record.args = ()
        return True

"""Ticker-symbol validation + parsing for user-supplied input.

Two responsibilities:

* :func:`valid_ticker` — shape check (``^[A-Z0-9.^-]{1,16}$``). Cheap,
  pure, regex-backed. Defends the rest of the codebase against
  surprises like ``"; DROP TABLE; --"`` or 200-character user input
  reaching the upstream Finnhub URL builder.

* :func:`parse_ticker_list` — the comma-separated entry point used by
  the sidebar text inputs (live ticker list + backfill ticker list).
  Returns a :class:`ParseResult` with the canonicalised valid list,
  what was rejected and why, and a ``truncated`` flag when the input
  blew past :data:`MAX_TICKERS` or :data:`MAX_INPUT_LEN`.

Defence-in-depth rationale (security findings F-08, F-13):

* The upstream ``finnhub.Client`` accepts whatever string we hand it
  and URL-encodes it into the query string. That's safe today but the
  library may grow path-interpolating endpoints in the future. Better
  to refuse the call here than to find out the hard way.

* Streamlit happily accepts arbitrary user input via ``text_input``.
  Without a cap, an operator with auth-gate access can paste a 10 MiB
  blob and our ingestion loop will dutifully iterate it.

Outputs are stable across re-parses: deduped, uppercased, order
preserved (first-seen wins). Tests live in :mod:`tests.test_symbols`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Maximum number of tickers we'll accept in a single user-supplied list.
# Sized to comfortably exceed any reasonable watchlist while still
# keeping the per-cycle Finnhub call budget bounded (every ticker
# triggers a separate /company-news request, so 50 already means 50+
# API calls per ingest plus the general feed).
MAX_TICKERS = 50

# Maximum raw-string length we'll even parse. 8 KiB is way more than
# a 50-ticker comma-separated list needs (worst case ~17 chars/symbol
# × 50 = 850 chars) but small enough to refuse anyone pasting a log
# file by accident.
MAX_INPUT_LEN = 8192

# Allowed-character set for tickers. Letters and digits cover the
# common case (AAPL, GOOG, BRK.B). ``.`` covers class-share notation
# (BRK.B, RDS.A). ``^`` is the prefix Finnhub uses for indices
# (^GSPC, ^DJI). ``-`` covers exchange-suffixed tickers (e.g. some
# international listings like RY.TO and dual-class shares like BF-B).
# We deliberately exclude underscore, slash, colon, and quotes —
# nothing legitimate uses them and they're the first thing an injection
# attempt reaches for.
#
# Note: we use ``re.fullmatch`` at every callsite (not ``re.match``)
# because Python's ``$`` anchor also matches **before** a trailing
# newline by default. ``re.fullmatch`` is unambiguous: the whole string
# must match, end-to-end, with no slop. The bracket class itself uses
# no anchors for that reason.
_TICKER_RE = re.compile(r"[A-Z0-9.^-]{1,16}")


# Reasons we may reject a token. Strings (not enums) so they round-trip
# through JSON logs cleanly without an import dance.
REJECT_EMPTY = "empty"
REJECT_TOO_LONG = "too_long"
REJECT_BAD_CHARS = "bad_chars"
REJECT_DUPLICATE = "duplicate"


@dataclass(frozen=True)
class ParseResult:
    """Outcome of :func:`parse_ticker_list`.

    Attributes:
        valid: Deduped, uppercased ticker list in first-seen order.
            Capped at :data:`MAX_TICKERS`.
        rejected: ``[(raw_token, reason), ...]`` for everything the
            parser refused. ``raw_token`` is exactly what the user
            typed (minus surrounding whitespace) so an operator can
            spot their own typo.
        truncated: True when the input would have produced more than
            :data:`MAX_TICKERS` valid entries, or when it exceeded
            :data:`MAX_INPUT_LEN`. The UI can then surface "we kept
            the first N — paste fewer or split into watchlists."
    """

    valid: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    truncated: bool = False


def valid_ticker(symbol: str) -> bool:
    """Return True iff ``symbol`` matches the shape rules for a ticker.

    The check is case-sensitive and expects already-uppercased input.
    :func:`parse_ticker_list` uppercases for you; callers operating on
    individual symbols (e.g. CLI args, repo lookups) should upper()
    first if they want to be lenient.
    """
    if not isinstance(symbol, str):
        return False
    return _TICKER_RE.fullmatch(symbol) is not None


def parse_ticker_list(
    raw: str,
    *,
    max_count: int = MAX_TICKERS,
    max_input_len: int = MAX_INPUT_LEN,
) -> ParseResult:
    """Parse a comma-separated ticker string into a validated, capped list.

    Behaviour:
        * Strips outer whitespace from each token.
        * Uppercases each token before validating (so ``"aapl"`` becomes
          ``"AAPL"`` and is accepted).
        * Drops empty tokens silently — they're noise from trailing
          commas (``"AAPL,, MSFT"``), not user error worth surfacing.
        * Drops tokens that don't match :data:`_TICKER_RE`, reporting
          them via :attr:`ParseResult.rejected` with reason
          :data:`REJECT_BAD_CHARS` or :data:`REJECT_TOO_LONG`.
        * Dedupes (first-seen wins); duplicates appear in
          ``rejected`` with reason :data:`REJECT_DUPLICATE`.
        * Caps the output at ``max_count`` and sets
          :attr:`ParseResult.truncated` if it had to. Surplus tokens
          are *not* added to ``rejected`` — that's a UI hint, not a
          per-token validation failure.

    Defensive:
        * ``raw`` is None or non-string → empty ParseResult.
        * ``raw`` longer than ``max_input_len`` → empty valid + truncated.
    """
    if not isinstance(raw, str):
        return ParseResult(truncated=False)
    if len(raw) > max_input_len:
        return ParseResult(truncated=True)

    valid: list[str] = []
    rejected: list[tuple[str, str]] = []
    seen: set[str] = set()
    truncated = False

    for token in raw.split(","):
        candidate = token.strip()
        if not candidate:
            continue  # silent: trailing comma / blank entry

        upper = candidate.upper()

        # Length guard pre-regex so the error message is specific.
        # (The regex also rejects 0-length / 17+-length but we surface
        # the actionable reason.)
        if len(upper) > 16:
            rejected.append((candidate, REJECT_TOO_LONG))
            continue

        if _TICKER_RE.fullmatch(upper) is None:
            rejected.append((candidate, REJECT_BAD_CHARS))
            continue

        if upper in seen:
            rejected.append((candidate, REJECT_DUPLICATE))
            continue

        if len(valid) >= max_count:
            truncated = True
            # Don't keep iterating once we've blown the cap — the
            # rest are "would-have-been-valid" but we drop silently.
            break

        seen.add(upper)
        valid.append(upper)

    return ParseResult(valid=valid, rejected=rejected, truncated=truncated)

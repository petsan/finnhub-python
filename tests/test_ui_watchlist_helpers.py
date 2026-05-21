"""Tests for the PR-2 UI helpers — parser-warning formatter + active-ticker resolver.

The Streamlit sidebar function itself is ``# pragma: no cover`` because it
needs a live Streamlit runtime; instead, we test the two pure helpers it
delegates to.

The ``session`` fixture is the in-memory SQLite from ``conftest.py``.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from finn_predictor.ingestion.symbols import ParseResult, parse_ticker_list
from finn_predictor.storage.repo import (
    add_to_watchlist,
    create_watchlist,
)
from finn_predictor.ui.app import (
    _format_parser_warning,
    _resolve_active_tickers,
)


# ---------------------------------------------------------------------------
# _format_parser_warning — caption text for the sidebar
# ---------------------------------------------------------------------------

def test_warning_clean_input_returns_none() -> None:
    """No rejected tokens and no truncation → no caption shown."""
    result = parse_ticker_list("AAPL, MSFT, NVDA")
    assert _format_parser_warning(result) is None


def test_warning_empty_input_returns_none() -> None:
    result = parse_ticker_list("")
    assert _format_parser_warning(result) is None


def test_warning_lists_rejected_tokens() -> None:
    result = parse_ticker_list("AAPL, A_B_C, MSFT")
    msg = _format_parser_warning(result)
    assert msg is not None
    assert "Dropped" in msg
    assert "A_B_C" in msg
    assert "invalid characters" in msg


def test_warning_pluralises_reasons() -> None:
    # Three distinct reasons in one input — the caption should call
    # each one out by its human-friendly noun.
    result = parse_ticker_list("AAPL, AAPL, BAD;CHAR, " + "X" * 17)
    msg = _format_parser_warning(result)
    assert msg is not None
    assert "duplicate" in msg
    assert "invalid characters" in msg
    assert "too long" in msg


def test_warning_caps_at_five_tokens_with_more_indicator() -> None:
    # Make a list with 7 rejected tokens.
    raw = ", ".join([f"BAD;SYM{i}" for i in range(7)] + ["AAPL"])
    result = parse_ticker_list(raw)
    assert len(result.rejected) == 7
    msg = _format_parser_warning(result)
    assert msg is not None
    # Only the first 5 should be named; the rest collapses to "+N more".
    assert "+ 2 more" in msg


def test_warning_truncated_flag() -> None:
    raw = ", ".join([f"SYM{i}" for i in range(60)])
    result = parse_ticker_list(raw)
    assert result.truncated is True
    msg = _format_parser_warning(result)
    assert msg is not None
    assert "truncated" in msg.lower()
    assert "50 tickers" in msg


def test_warning_rejected_plus_truncated_combined() -> None:
    raw = ", ".join(["BAD;CHAR"] + [f"SYM{i}" for i in range(60)])
    result = parse_ticker_list(raw)
    msg = _format_parser_warning(result)
    assert msg is not None
    assert "Dropped" in msg
    assert "truncated" in msg.lower()


def test_warning_handles_none_input() -> None:
    """Defensive — the sidebar wraps the call in try/except but the
    helper itself must not crash on None."""
    assert _format_parser_warning(None) is None


def test_warning_handles_minimal_duck() -> None:
    """The formatter only reads ``rejected`` and ``truncated`` — any
    object exposing those attributes works."""
    class Stub:
        rejected = [("foo", "bad_chars")]
        truncated = True

    msg = _format_parser_warning(Stub())
    assert msg is not None
    assert "foo" in msg
    assert "truncated" in msg.lower()


def test_warning_unknown_reason_falls_through() -> None:
    """A reason string the formatter doesn't know about should still
    appear in the caption verbatim (forward-compat with new reasons)."""
    class Stub:
        rejected = [("foo", "mystery_reason")]
        truncated = False

    msg = _format_parser_warning(Stub())
    assert msg is not None
    assert "mystery_reason" in msg


def test_warning_empty_rejected_list_no_caption() -> None:
    """Edge case: ``rejected=[]`` and ``truncated=False`` is a clean parse."""
    pr = ParseResult(valid=["AAPL"], rejected=[], truncated=False)
    assert _format_parser_warning(pr) is None


# ---------------------------------------------------------------------------
# _resolve_active_tickers — sidebar's "which list?" precedence
# ---------------------------------------------------------------------------

def test_resolver_no_active_list_returns_parsed_textbox(session: Session) -> None:
    out = _resolve_active_tickers(
        session, symbols_csv="aapl, msft", active_watchlist=None
    )
    assert out == ["AAPL", "MSFT"]


def test_resolver_empty_textbox_with_no_list_returns_empty(session: Session) -> None:
    out = _resolve_active_tickers(
        session, symbols_csv="", active_watchlist=None
    )
    assert out == []


def test_resolver_active_list_overrides_textbox(session: Session) -> None:
    create_watchlist(session, name="Tech")
    add_to_watchlist(session, name="Tech", symbol="AAPL")
    add_to_watchlist(session, name="Tech", symbol="NVDA")

    out = _resolve_active_tickers(
        session,
        symbols_csv="THIS, IS, IGNORED",   # textbox content is irrelevant
        active_watchlist="Tech",
    )
    # Comes from the watchlist, alphabetically sorted (per watchlist_symbols).
    assert out == ["AAPL", "NVDA"]


def test_resolver_falls_back_to_textbox_when_list_missing(session: Session) -> None:
    """If a list is selected but no longer exists, fall back to textbox.

    UX > correctness here: a dropped watchlist should not break the
    Run-ingestion button. The next rerun will reset the dropdown.
    """
    out = _resolve_active_tickers(
        session,
        symbols_csv="AAPL, MSFT",
        active_watchlist="MISSING_LIST",
    )
    assert out == ["AAPL", "MSFT"]


def test_resolver_falls_back_on_invalid_list_name(session: Session) -> None:
    """A malformed watchlist name (e.g. injection-like) should also
    fall back cleanly rather than raising."""
    out = _resolve_active_tickers(
        session,
        symbols_csv="AAPL",
        active_watchlist="with/slash",
    )
    assert out == ["AAPL"]


def test_resolver_empty_watchlist_returns_empty(session: Session) -> None:
    """A watchlist with no members returns an empty list, not falling
    through to the textbox. Empty is an intentional state — the
    operator may have deliberately cleared the list."""
    create_watchlist(session, name="Empty")
    out = _resolve_active_tickers(
        session,
        symbols_csv="AAPL, MSFT",
        active_watchlist="Empty",
    )
    assert out == []


def test_resolver_active_list_dedupes_across_lists_internally(
    session: Session,
) -> None:
    """A watchlist's own ``watchlist_symbols`` view returns deduped
    symbols. Resolver passes that through unchanged."""
    create_watchlist(session, name="Tech")
    add_to_watchlist(session, name="Tech", symbol="AAPL")
    out = _resolve_active_tickers(
        session, symbols_csv="", active_watchlist="Tech"
    )
    assert out == ["AAPL"]


def test_resolver_blank_active_watchlist_uses_textbox(session: Session) -> None:
    """An empty-string ``active_watchlist`` should be treated as None
    (truthiness check). Mirrors the UI's "manual textbox" choice
    coming through as None."""
    out = _resolve_active_tickers(
        session, symbols_csv="AAPL", active_watchlist=""
    )
    assert out == ["AAPL"]

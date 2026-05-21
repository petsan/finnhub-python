"""Unit tests for :mod:`finn_predictor.ingestion.symbols`.

Coverage targets:

* Every branch in :func:`parse_ticker_list` and :func:`valid_ticker`.
* The two security-relevant caps (count + input length) actually trip.
* Real-world ticker shapes accepted (BRK.B, ^GSPC, RDS.A).
* Common injection-style payloads rejected with the expected reason.
"""

from __future__ import annotations

import pytest

from finn_predictor.ingestion.symbols import (
    MAX_INPUT_LEN,
    MAX_TICKERS,
    REJECT_BAD_CHARS,
    REJECT_DUPLICATE,
    REJECT_TOO_LONG,
    ParseResult,
    parse_ticker_list,
    valid_ticker,
)


# ---------------------------------------------------------------------------
# valid_ticker — shape check on already-uppercased input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "good",
    [
        "AAPL",
        "MSFT",
        "BRK.B",       # class-share dot notation
        "BF-B",        # dash notation
        "^GSPC",       # index prefix
        "RDS.A",       # international class share
        "XLE",         # 3-letter ETF
        "Z",           # single letter (Zillow on some exchanges)
        "1",           # purely numeric (rare but allowed)
        "ABCD1234EFGH",  # 12 chars — within cap
        "ABCDEFGHIJKLMNOP",  # exact 16-char cap
    ],
)
def test_valid_ticker_accepts_known_shapes(good: str) -> None:
    assert valid_ticker(good), f"expected {good!r} to be accepted"


@pytest.mark.parametrize(
    "bad",
    [
        "",                                # empty
        "aapl",                            # lowercase — caller should upper() first
        "AAPL ",                           # trailing space
        " AAPL",                           # leading space
        "AAPL,MSFT",                       # multi-symbol blob
        "DROP TABLE",                      # spaces
        "AAPL;DROP",                       # semicolon
        "AAPL'",                           # quote
        '"AAPL"',                          # quoted
        "AAPL/USD",                        # slash
        "AAPL:NASDAQ",                     # colon (exchange suffix using colon)
        "AAPL_X",                          # underscore (not in our charset)
        "ABCDEFGHIJKLMNOPQ",               # 17 chars — over cap
        "../../etc/passwd",                # path traversal
        "<script>",                        # angle brackets
        "AAPL\n",                          # newline
        "AAPL\x00",                        # null byte
    ],
)
def test_valid_ticker_rejects_bad_shapes(bad: str) -> None:
    assert not valid_ticker(bad), f"expected {bad!r} to be rejected"


def test_valid_ticker_rejects_non_string() -> None:
    # Non-string types must return False, not raise — the parser is
    # the only public entry point and we want it impossible to crash.
    assert valid_ticker(None) is False  # type: ignore[arg-type]
    assert valid_ticker(123) is False   # type: ignore[arg-type]
    assert valid_ticker(["AAPL"]) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_ticker_list — the public entry point used by the UI
# ---------------------------------------------------------------------------

def test_parse_empty_string_is_empty_result() -> None:
    r = parse_ticker_list("")
    assert isinstance(r, ParseResult)
    assert r.valid == []
    assert r.rejected == []
    assert r.truncated is False


def test_parse_whitespace_only_is_empty_result() -> None:
    r = parse_ticker_list("   \t  ,  ,   ")
    # All tokens are blank-after-strip — silently dropped, no rejects.
    assert r.valid == []
    assert r.rejected == []


def test_parse_lowercases_get_uppercased() -> None:
    r = parse_ticker_list("aapl, msft")
    assert r.valid == ["AAPL", "MSFT"]
    assert r.rejected == []


def test_parse_preserves_first_seen_order() -> None:
    r = parse_ticker_list("NVDA, AAPL, MSFT, GOOG")
    assert r.valid == ["NVDA", "AAPL", "MSFT", "GOOG"]


def test_parse_dedupe_keeps_first_drops_rest() -> None:
    r = parse_ticker_list("AAPL, aapl, MSFT, AAPL")
    assert r.valid == ["AAPL", "MSFT"]
    # Both later duplicates surface as REJECT_DUPLICATE — operator
    # can see they typed it twice.
    assert ("aapl", REJECT_DUPLICATE) in r.rejected
    assert ("AAPL", REJECT_DUPLICATE) in r.rejected


def test_parse_real_world_mixed_input() -> None:
    """Realistic sidebar paste — mix of good, bad, dupes, blank cells."""
    raw = "AAPL, msft , , NVDA, BRK.B, ^GSPC, AAPL, INVALID;NAME, "
    r = parse_ticker_list(raw)
    assert r.valid == ["AAPL", "MSFT", "NVDA", "BRK.B", "^GSPC"]
    # Trailing duplicate
    assert ("AAPL", REJECT_DUPLICATE) in r.rejected
    # The injection-style token
    assert ("INVALID;NAME", REJECT_BAD_CHARS) in r.rejected
    assert r.truncated is False


def test_parse_rejects_too_long_token_with_specific_reason() -> None:
    # 17 chars — over the per-ticker cap. We surface TOO_LONG so the
    # operator knows the issue isn't a typo.
    long_token = "A" * 17
    r = parse_ticker_list(f"AAPL, {long_token}")
    assert r.valid == ["AAPL"]
    assert (long_token, REJECT_TOO_LONG) in r.rejected


def test_parse_rejects_bad_chars_with_specific_reason() -> None:
    r = parse_ticker_list("AAPL, A_B_C, MSFT")
    assert r.valid == ["AAPL", "MSFT"]
    assert ("A_B_C", REJECT_BAD_CHARS) in r.rejected


def test_parse_count_cap_truncates_silently() -> None:
    # Generate MAX_TICKERS + 5 unique tickers.
    tickers = [f"SYM{i}" for i in range(MAX_TICKERS + 5)]
    raw = ", ".join(tickers)
    r = parse_ticker_list(raw)
    assert len(r.valid) == MAX_TICKERS
    assert r.truncated is True
    # Surplus tickers do NOT appear in rejected — truncation is a
    # UI hint, not a per-token validation issue.
    assert all("SYM" not in token for token, _ in r.rejected)


def test_parse_custom_max_count() -> None:
    r = parse_ticker_list("AAPL, MSFT, NVDA, GOOG, AMZN", max_count=3)
    assert r.valid == ["AAPL", "MSFT", "NVDA"]
    assert r.truncated is True


def test_parse_input_length_cap_blocks_pathological_payload() -> None:
    # Generate a string longer than MAX_INPUT_LEN. Even if every
    # token is well-formed, we refuse to parse it — defence against
    # paste-the-log-file accidents and DoS.
    raw = ("AAPL," * (MAX_INPUT_LEN // 5)) + "ZZZZ"
    assert len(raw) > MAX_INPUT_LEN
    r = parse_ticker_list(raw)
    assert r.valid == []
    assert r.truncated is True
    assert r.rejected == []


def test_parse_custom_max_input_len() -> None:
    r = parse_ticker_list("AAPL, MSFT", max_input_len=5)
    assert r.valid == []
    assert r.truncated is True


def test_parse_none_input_is_empty_result() -> None:
    # Defensive — Streamlit text_input never returns None today, but
    # the parser shouldn't crash if a future Streamlit version does.
    r = parse_ticker_list(None)  # type: ignore[arg-type]
    assert r.valid == []
    assert r.rejected == []
    assert r.truncated is False


def test_parse_non_string_input_is_empty_result() -> None:
    r = parse_ticker_list(12345)  # type: ignore[arg-type]
    assert r.valid == []


def test_parse_result_is_immutable() -> None:
    r = parse_ticker_list("AAPL")
    with pytest.raises(Exception):
        # frozen=True dataclass — direct assignment must raise.
        r.valid = ["MSFT"]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration: the existing _parse_symbols UI helper now delegates here
# ---------------------------------------------------------------------------

def test_ui_parse_symbols_delegates_to_parser() -> None:
    """Pinning test: _parse_symbols must keep returning a plain list.

    The UI's sidebar passes the result around as a list of strings; we
    want to delegate to parse_ticker_list internally without changing
    that public shape. Anything that touches this test should also
    update the UI sidebar.
    """
    from finn_predictor.ui.app import _parse_symbols

    # Old-style API: just a list of cleaned ticker strings.
    assert _parse_symbols("") == []
    assert _parse_symbols("aapl") == ["AAPL"]
    assert _parse_symbols("AAPL, msft , , NVDA") == ["AAPL", "MSFT", "NVDA"]
    # Dedupe behaviour comes from the new validator.
    assert _parse_symbols("AAPL, AAPL") == ["AAPL"]
    # Garbage drops out silently — the validator owns the surface.
    assert _parse_symbols("AAPL, ;DROP TABLE") == ["AAPL"]

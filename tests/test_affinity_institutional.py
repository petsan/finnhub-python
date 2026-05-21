"""Tests for :func:`finn_predictor.predictor.affinity.refresh_institutional_holders` (PR-5).

Covers:
* Happy-path ingest of a multi-holder payload — order, percentage, share
  count, value, filing date all land in ``metadata_text`` as JSON.
* Name normalisation across filings (case + whitespace variants resolve
  to the same DB key).
* Idempotent re-runs (``holders_added`` resets to 0; ``holders_refreshed``
  reports the touched rows).
* ``limit`` truncation.
* Empty / malformed payload paths.
* ``IngestionError`` captured as a failure (no exception propagation).
* CompanyFocus exposes the new ``institutional_holders`` field.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from finn_predictor.ingestion.client import IngestionError
from finn_predictor.predictor.affinity import (
    INSTITUTION_NAME_MAX_LEN,
    InstitutionalHoldersResult,
    refresh_institutional_holders,
)
from finn_predictor.predictor.focus import compose_company_focus
from finn_predictor.storage.repo import related_entities_for


_FROZEN_NOW = datetime(2026, 5, 21, tzinfo=timezone.utc)


def _payload(*holders: dict) -> dict:
    """Shape Finnhub's /institutional/ownership response from holder dicts."""
    return {"symbol": "AAPL", "data": list(holders)}


def _holder(
    *,
    name: str,
    percentage: float = 1.0,
    share: float = 1000.0,
    value: float = 100_000.0,
    filingDate: str = "2026-03-31",
) -> dict:
    return {
        "name": name,
        "percentage": percentage,
        "share": share,
        "value": value,
        "filingDate": filingDate,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_refresh_persists_holders_in_order(session: Session) -> None:
    gw = MagicMock()
    gw.institutional_ownership.return_value = _payload(
        _holder(name="Vanguard Group Inc",  percentage=7.45, share=1_200_000_000),
        _holder(name="BlackRock Inc",       percentage=6.30, share=1_000_000_000),
        _holder(name="State Street Corp",   percentage=4.10, share=  600_000_000),
    )
    result = refresh_institutional_holders(
        session, gw, symbol="AAPL", today=_FROZEN_NOW,
    )
    assert isinstance(result, InstitutionalHoldersResult)
    assert result.symbol == "AAPL"
    assert result.holders_added == 3
    assert result.holders_refreshed == 0
    assert result.failures == []

    rows = related_entities_for(
        session, "AAPL", relationship="INSTITUTIONAL_HOLDER"
    )
    # related_entities_for sorts by rank asc — so position 0 (biggest
    # holder by share) leads.
    names = [r.related_symbol for r in rows]
    assert names == [
        "VANGUARD GROUP INC",
        "BLACKROCK INC",
        "STATE STREET CORP",
    ]
    # metadata_text round-trips the numeric fields the UI cares about.
    top = rows[0]
    meta = json.loads(top.metadata_text)
    assert meta["percentage"] == 7.45
    assert meta["share"] == 1_200_000_000
    assert meta["filing_date"] == "2026-03-31"


def test_refresh_uses_uppercase_normalised_name_as_key(session: Session) -> None:
    """Two filings that disagree on case + whitespace should resolve to
    the same DB row, not duplicate it."""
    gw = MagicMock()

    gw.institutional_ownership.return_value = _payload(
        _holder(name="Vanguard Group, Inc.", percentage=7.0),
    )
    refresh_institutional_holders(session, gw, symbol="AAPL", today=_FROZEN_NOW)
    rows1 = related_entities_for(session, "AAPL", relationship="INSTITUTIONAL_HOLDER")
    assert len(rows1) == 1

    # Same holder, different formatting on the next filing.
    gw.institutional_ownership.return_value = _payload(
        _holder(name="  VANGUARD GROUP, INC.  ", percentage=8.0),
    )
    result = refresh_institutional_holders(
        session, gw, symbol="AAPL", today=_FROZEN_NOW
    )
    rows2 = related_entities_for(session, "AAPL", relationship="INSTITUTIONAL_HOLDER")
    assert len(rows2) == 1  # still just one row
    assert result.holders_added == 0
    assert result.holders_refreshed == 1
    # Refreshed percentage made it through.
    assert json.loads(rows2[0].metadata_text)["percentage"] == 8.0


def test_refresh_truncates_long_names(session: Session) -> None:
    """A pathologically long holder name doesn't crash the column."""
    long_name = "GIGANTOR OVER-LEVERAGED MULTINATIONAL ASSET MANAGEMENT GROUP HOLDING COMPANY LIMITED"
    assert len(long_name) > INSTITUTION_NAME_MAX_LEN

    gw = MagicMock()
    gw.institutional_ownership.return_value = _payload(
        _holder(name=long_name, percentage=2.5),
    )
    refresh_institutional_holders(session, gw, symbol="AAPL", today=_FROZEN_NOW)
    rows = related_entities_for(session, "AAPL", relationship="INSTITUTIONAL_HOLDER")
    assert len(rows[0].related_symbol) == INSTITUTION_NAME_MAX_LEN


def test_refresh_limit_truncates_response(session: Session) -> None:
    gw = MagicMock()
    gw.institutional_ownership.return_value = _payload(
        *[_holder(name=f"Holder {i}", percentage=10.0 - i) for i in range(10)]
    )
    result = refresh_institutional_holders(
        session, gw, symbol="AAPL", limit=4, today=_FROZEN_NOW,
    )
    assert result.holders_added == 4
    rows = related_entities_for(session, "AAPL", relationship="INSTITUTIONAL_HOLDER")
    assert len(rows) == 4


def test_refresh_skips_duplicate_names_within_one_response(session: Session) -> None:
    """Some Finnhub responses repeat a filer across filing types — keep
    the first (which is the biggest share) and skip the rest."""
    gw = MagicMock()
    gw.institutional_ownership.return_value = _payload(
        _holder(name="Vanguard", percentage=7.5, share=1_000),
        _holder(name="Vanguard", percentage=0.1, share=10),     # dup
        _holder(name="BlackRock", percentage=6.0, share=900),
    )
    refresh_institutional_holders(session, gw, symbol="AAPL", today=_FROZEN_NOW)
    rows = related_entities_for(session, "AAPL", relationship="INSTITUTIONAL_HOLDER")
    assert {r.related_symbol for r in rows} == {"VANGUARD", "BLACKROCK"}
    vanguard = next(r for r in rows if r.related_symbol == "VANGUARD")
    assert json.loads(vanguard.metadata_text)["percentage"] == 7.5


# ---------------------------------------------------------------------------
# Empty / malformed responses
# ---------------------------------------------------------------------------

def test_empty_data_array_is_no_op(session: Session) -> None:
    gw = MagicMock()
    gw.institutional_ownership.return_value = {"symbol": "AAPL", "data": []}
    result = refresh_institutional_holders(
        session, gw, symbol="AAPL", today=_FROZEN_NOW
    )
    assert result.holders_added == 0
    assert result.failures == []


def test_non_dict_response_is_no_op(session: Session) -> None:
    gw = MagicMock()
    gw.institutional_ownership.return_value = ["not", "a", "dict"]
    result = refresh_institutional_holders(
        session, gw, symbol="AAPL", today=_FROZEN_NOW
    )
    assert result.holders_added == 0


def test_data_field_missing_is_no_op(session: Session) -> None:
    gw = MagicMock()
    gw.institutional_ownership.return_value = {"symbol": "AAPL"}
    result = refresh_institutional_holders(
        session, gw, symbol="AAPL", today=_FROZEN_NOW
    )
    assert result.holders_added == 0


def test_holder_without_name_dropped(session: Session) -> None:
    gw = MagicMock()
    gw.institutional_ownership.return_value = _payload(
        {"percentage": 5.0},  # missing name
        _holder(name="Real Holder", percentage=3.0),
    )
    result = refresh_institutional_holders(
        session, gw, symbol="AAPL", today=_FROZEN_NOW
    )
    assert result.holders_added == 1


def test_holder_with_non_string_name_dropped(session: Session) -> None:
    gw = MagicMock()
    gw.institutional_ownership.return_value = _payload(
        {"name": 12345, "percentage": 5.0},  # non-string name
        _holder(name="Real Holder", percentage=3.0),
    )
    result = refresh_institutional_holders(
        session, gw, symbol="AAPL", today=_FROZEN_NOW
    )
    assert result.holders_added == 1


def test_holder_with_string_percentage_is_coerced(session: Session) -> None:
    """Finnhub sometimes returns numeric fields as strings; coerce
    rather than persist 'nan' or skip."""
    gw = MagicMock()
    gw.institutional_ownership.return_value = _payload(
        {"name": "X", "percentage": "5.5", "share": "1000",
         "value": "100000.0", "filingDate": "2026-03-31"},
    )
    refresh_institutional_holders(session, gw, symbol="AAPL", today=_FROZEN_NOW)
    rows = related_entities_for(session, "AAPL", relationship="INSTITUTIONAL_HOLDER")
    meta = json.loads(rows[0].metadata_text)
    assert meta["percentage"] == 5.5
    assert meta["share"] == 1000.0


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------

def test_ingestion_error_captured_not_raised(session: Session) -> None:
    gw = MagicMock()
    gw.institutional_ownership.side_effect = IngestionError(
        "FinnhubAPI 403: gated on free tier"
    )
    result = refresh_institutional_holders(
        session, gw, symbol="AAPL", today=_FROZEN_NOW
    )
    assert result.holders_added == 0
    assert any("institutional_ownership:AAPL" in f["op"]
               for f in result.failures)


def test_empty_symbol_raises(session: Session) -> None:
    with pytest.raises(ValueError):
        refresh_institutional_holders(
            session, MagicMock(), symbol="", today=_FROZEN_NOW
        )


def test_zero_limit_is_no_op(session: Session) -> None:
    """A limit of 0 returns immediately without calling the gateway —
    saves quota when the caller is intentionally just clearing the
    UI state."""
    gw = MagicMock()
    result = refresh_institutional_holders(
        session, gw, symbol="AAPL", limit=0, today=_FROZEN_NOW
    )
    assert result.holders_added == 0
    assert gw.institutional_ownership.call_count == 0


def test_negative_lookback_clamped_to_one_day(session: Session) -> None:
    """A nonsensical negative lookback shouldn't compute a bogus
    from-date; we clamp to at least 1 day so the call still proceeds
    with the legitimate today value."""
    gw = MagicMock()
    gw.institutional_ownership.return_value = _payload()
    refresh_institutional_holders(
        session, gw, symbol="AAPL", lookback_days=-10, today=_FROZEN_NOW,
    )
    args, kwargs = gw.institutional_ownership.call_args
    # Either positional or kw — work with both.
    if args:
        _, from_iso, to_iso = args
    else:
        from_iso = kwargs.get("_from")
        to_iso = kwargs.get("to")
    # to_iso == today; from_iso == today - 1 day.
    assert to_iso == "2026-05-21"
    assert from_iso == "2026-05-20"


# ---------------------------------------------------------------------------
# CompanyFocus integration
# ---------------------------------------------------------------------------

def test_compose_company_focus_returns_institutional_holders(
    session: Session,
) -> None:
    gw = MagicMock()
    gw.institutional_ownership.return_value = _payload(
        _holder(name="Vanguard", percentage=7.5),
        _holder(name="BlackRock", percentage=6.0),
    )
    refresh_institutional_holders(session, gw, symbol="AAPL", today=_FROZEN_NOW)

    bundle = compose_company_focus(session, "AAPL")
    holder_names = sorted(h.related_symbol for h in bundle.institutional_holders)
    assert holder_names == ["BLACKROCK", "VANGUARD"]


def test_compose_company_focus_no_holders_is_empty_list(session: Session) -> None:
    bundle = compose_company_focus(session, "AAPL")
    assert bundle.institutional_holders == []


# ---------------------------------------------------------------------------
# RELATIONSHIPS allowed-set
# ---------------------------------------------------------------------------

def test_relationships_allowed_set_includes_institutional_holder() -> None:
    from finn_predictor.storage.repo import RELATIONSHIPS
    assert "INSTITUTIONAL_HOLDER" in RELATIONSHIPS

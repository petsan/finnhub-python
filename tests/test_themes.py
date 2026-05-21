"""Tests for PR-6: InvestmentTheme model + ingest + predictor.

Covers:
* `add_investment_theme` registers + idempotently updates rows.
* `refresh_investment_themes` walks codes, upserts theme + member rows,
  isolates per-theme failures.
* `_humanise_theme_code` produces sensible display names.
* `predict_theme` returns None when no members or no signal; computes
  a `ThemePrediction` when the signal exists.
* `predict_all_themes` iterates registered themes + sorts by confidence.
* RELATIONSHIPS allowed-set includes `THEME_MEMBER`.
* CLI `add-theme` and `refresh-themes` (happy path + API-key gate).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from finn_predictor.cli import _build_parser, main
from finn_predictor.ingestion.client import IngestionError
from finn_predictor.predictor.affinity import (
    ThemeRefreshResult,
    _humanise_theme_code,
    add_investment_theme,
    refresh_investment_themes,
)
from finn_predictor.predictor.themes import (
    DEFAULT_THEME_CODES,
    ThemePrediction,
    predict_all_themes,
    predict_theme,
)
from finn_predictor.sentiment.base import Scorer
from finn_predictor.storage.models import (
    InvestmentTheme,
    NewsArticle,
    SentimentScore,
)
from finn_predictor.storage.repo import (
    related_entities_for,
    save_scores,
    upsert_articles,
    upsert_related_entity,
)
from tests.conftest import make_article, make_score


class _StubScorer(Scorer):
    """Scorer that emits a fixed score for any article — used for
    deterministic predict_theme tests."""

    def __init__(self, *, score: float, model_version: str = "stub-test"):
        self._score = score
        self._mv = model_version

    @property
    def model_version(self) -> str:
        return self._mv

    def score(self, text: str) -> float:
        return self._score


# ---------------------------------------------------------------------------
# _humanise_theme_code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "code,expected",
    [
        ("cyberSecurity", "Cyber Security"),
        ("aiSemis", "Ai Semis"),
        ("financialExchangesData", "Financial Exchanges Data"),
        ("ev", "Ev"),
        ("ROBOTICS", "Robotics"),
    ],
)
def test_humanise_theme_code(code: str, expected: str) -> None:
    assert _humanise_theme_code(code) == expected


# ---------------------------------------------------------------------------
# add_investment_theme
# ---------------------------------------------------------------------------

def test_add_investment_theme_with_defaults(session: Session) -> None:
    row = add_investment_theme(session, theme_code="cyberSecurity")
    assert row.theme_code == "cyberSecurity"
    assert row.name == "Cyber Security"
    assert row.description is None


def test_add_investment_theme_with_explicit_name(session: Session) -> None:
    row = add_investment_theme(
        session, theme_code="cyberSecurity",
        name="Cyber & Endpoint Security",
        description="  vendors of cyber + endpoint  ",
    )
    assert row.name == "Cyber & Endpoint Security"
    assert row.description == "vendors of cyber + endpoint"


def test_add_investment_theme_idempotent_updates(session: Session) -> None:
    a = add_investment_theme(session, theme_code="cyberSecurity")
    b = add_investment_theme(
        session, theme_code="cyberSecurity", name="Updated"
    )
    assert a.id == b.id
    assert b.name == "Updated"


def test_add_investment_theme_empty_raises(session: Session) -> None:
    with pytest.raises(ValueError):
        add_investment_theme(session, theme_code="")
    with pytest.raises(ValueError):
        add_investment_theme(session, theme_code="   ")


# ---------------------------------------------------------------------------
# refresh_investment_themes
# ---------------------------------------------------------------------------

def test_refresh_investment_themes_happy_path(session: Session) -> None:
    gw = MagicMock()
    gw.stock_investment_theme.side_effect = lambda code: {
        "cyberSecurity": {"symbols": ["PANW", "CRWD", "FTNT"]},
        "aiSemis":       {"symbols": ["NVDA", "AMD"]},
    }[code]

    result = refresh_investment_themes(
        session, gw, theme_codes=["cyberSecurity", "aiSemis"]
    )
    assert isinstance(result, ThemeRefreshResult)
    assert result.themes_processed == ["cyberSecurity", "aiSemis"]
    assert result.themes_added == 2
    assert result.members_added == 5

    cyber = related_entities_for(session, "cyberSecurity",
                                 relationship="THEME_MEMBER")
    assert sorted(m.related_symbol for m in cyber) == ["CRWD", "FTNT", "PANW"]


def test_refresh_investment_themes_uses_defaults(session: Session) -> None:
    """No theme_codes arg → falls back to DEFAULT_THEME_CODES."""
    gw = MagicMock()
    gw.stock_investment_theme.return_value = {"symbols": []}

    result = refresh_investment_themes(session, gw)
    # Every default code was attempted.
    assert result.themes_processed == list(DEFAULT_THEME_CODES)
    assert gw.stock_investment_theme.call_count == len(DEFAULT_THEME_CODES)


def test_refresh_investment_themes_per_theme_failure_isolated(
    session: Session,
) -> None:
    """A 403 on one theme shouldn't kill the rest."""
    gw = MagicMock()

    def lookup(code):
        if code == "cyberSecurity":
            raise IngestionError("FinnhubAPI 403: gated")
        return {"symbols": ["X"]}

    gw.stock_investment_theme.side_effect = lookup
    result = refresh_investment_themes(
        session, gw, theme_codes=["cyberSecurity", "aiSemis"]
    )
    # cyberSecurity still got an InvestmentTheme row (registered before
    # the constituent fetch), just no members.
    assert result.themes_added == 2
    assert result.members_added == 1
    assert any("stock_investment_theme:cyberSecurity" in f["op"]
               for f in result.failures)


def test_refresh_investment_themes_idempotent(session: Session) -> None:
    gw = MagicMock()
    gw.stock_investment_theme.return_value = {"symbols": ["NVDA", "AMD"]}

    r1 = refresh_investment_themes(session, gw, theme_codes=["aiSemis"])
    r2 = refresh_investment_themes(session, gw, theme_codes=["aiSemis"])

    assert r1.themes_added == 1
    assert r2.themes_added == 0
    assert r1.members_added == 2
    assert r2.members_added == 0


def test_refresh_investment_themes_dedupes_within_response(session: Session) -> None:
    gw = MagicMock()
    gw.stock_investment_theme.return_value = {"symbols": ["NVDA", "nvda", "AMD"]}
    result = refresh_investment_themes(session, gw, theme_codes=["aiSemis"])
    # Lowercase variant collapses to NVDA via uppercase normalization.
    assert result.members_added == 2


def test_refresh_investment_themes_accepts_dict_symbols(session: Session) -> None:
    """Some Finnhub variants return [{"symbol": "AAPL"}, ...]."""
    gw = MagicMock()
    gw.stock_investment_theme.return_value = {
        "symbols": [{"symbol": "NVDA"}, {"symbol": "AMD"}]
    }
    result = refresh_investment_themes(session, gw, theme_codes=["aiSemis"])
    assert result.members_added == 2


def test_refresh_investment_themes_empty_response(session: Session) -> None:
    gw = MagicMock()
    gw.stock_investment_theme.return_value = {"symbols": []}
    result = refresh_investment_themes(session, gw, theme_codes=["aiSemis"])
    assert result.themes_added == 1
    assert result.members_added == 0
    assert result.failures == []


def test_refresh_investment_themes_skips_blank_codes(session: Session) -> None:
    """Blank entries in the input list shouldn't be processed."""
    gw = MagicMock()
    gw.stock_investment_theme.return_value = {"symbols": []}
    result = refresh_investment_themes(
        session, gw, theme_codes=["", "  ", "aiSemis"]
    )
    assert result.themes_processed == ["aiSemis"]


def test_refresh_investment_themes_malformed_payload(session: Session) -> None:
    """Non-dict / wrong-shape payload → zero members, no crash."""
    gw = MagicMock()
    gw.stock_investment_theme.side_effect = [
        ["not", "a", "dict"],   # array
        {"data": ["A"]},        # right key wrong shape
    ]
    result = refresh_investment_themes(
        session, gw, theme_codes=["one", "two"]
    )
    assert result.members_added == 0
    assert result.themes_added == 2  # both registered


# ---------------------------------------------------------------------------
# predict_theme
# ---------------------------------------------------------------------------

def _seed_articles_with_scores(
    session: Session,
    *,
    symbol: str,
    scores: list[float],
    finnhub_id_start: int = 1,
    day: datetime | None = None,
) -> None:
    """Helper: seed N scored company-news articles for ``symbol`` on a
    specific UTC day."""
    day = day or datetime(2026, 5, 21, tzinfo=timezone.utc)
    arts = [
        make_article(
            finnhub_id=finnhub_id_start + i,
            category="company",
            symbol=symbol,
            published_at=day.replace(hour=12),
        )
        for i in range(len(scores))
    ]
    upsert_articles(session, arts)
    persisted = session.query(NewsArticle).filter(NewsArticle.symbol == symbol).all()
    save_scores(
        session,
        [make_score(a.id, sc, model_version="stub-test")
         for a, sc in zip(persisted, scores)],
    )


def test_predict_theme_returns_none_when_no_members(session: Session) -> None:
    scorer = _StubScorer(score=0.5)
    out = predict_theme(
        session, scorer=scorer, theme_code="aiSemis",
        on_date=datetime(2026, 5, 21, tzinfo=timezone.utc),
    )
    assert out is None


def test_predict_theme_returns_none_when_no_articles(session: Session) -> None:
    upsert_related_entity(
        session, source_symbol="aiSemis", related_symbol="NVDA",
        relationship="THEME_MEMBER", rank=0,
    )
    scorer = _StubScorer(score=0.5)
    out = predict_theme(
        session, scorer=scorer, theme_code="aiSemis",
        on_date=datetime(2026, 5, 21, tzinfo=timezone.utc),
    )
    assert out is None


def test_predict_theme_emits_prediction_when_signal_present(session: Session) -> None:
    # Membership.
    for i, sym in enumerate(["NVDA", "AMD", "INTC"]):
        upsert_related_entity(
            session, source_symbol="aiSemis", related_symbol=sym,
            relationship="THEME_MEMBER", rank=i,
        )
    # Articles + scores.
    day = datetime(2026, 5, 21, tzinfo=timezone.utc)
    _seed_articles_with_scores(
        session, symbol="NVDA", scores=[0.5, 0.6, 0.7], day=day,
        finnhub_id_start=1,
    )
    _seed_articles_with_scores(
        session, symbol="AMD", scores=[0.4, 0.5], day=day,
        finnhub_id_start=100,
    )

    scorer = _StubScorer(score=0.5)
    out = predict_theme(
        session, scorer=scorer, theme_code="aiSemis", on_date=day,
        min_baseline_sigma=0.01,
    )
    assert isinstance(out, ThemePrediction)
    assert out.theme_code == "aiSemis"
    assert out.constituent_count == 3
    assert out.article_count == 5
    # Strongly positive sentiment → UP call.
    assert out.label == "UP"
    assert 0.0 <= out.confidence <= 1.0


def test_predict_theme_empty_string_raises(session: Session) -> None:
    scorer = _StubScorer(score=0.0)
    with pytest.raises(ValueError):
        predict_theme(session, scorer=scorer, theme_code="")


# ---------------------------------------------------------------------------
# predict_all_themes
# ---------------------------------------------------------------------------

def test_predict_all_themes_iterates_registered(session: Session) -> None:
    """Registers two themes + members + scored articles for one,
    asserts only the one with signal makes it into the output."""
    add_investment_theme(session, theme_code="aiSemis")
    add_investment_theme(session, theme_code="cyberSecurity")
    upsert_related_entity(
        session, source_symbol="aiSemis", related_symbol="NVDA",
        relationship="THEME_MEMBER", rank=0,
    )
    # cyberSecurity has no members → predict_theme returns None
    day = datetime(2026, 5, 21, tzinfo=timezone.utc)
    _seed_articles_with_scores(
        session, symbol="NVDA", scores=[0.5, 0.6, 0.7], day=day,
    )

    scorer = _StubScorer(score=0.5)
    out = predict_all_themes(
        session, scorer=scorer, on_date=day,
    )
    codes = [t.theme_code for t in out]
    assert "aiSemis" in codes
    assert "cyberSecurity" not in codes  # no members
    # Operator-friendly name is patched in from the InvestmentTheme row.
    aisemis = next(t for t in out if t.theme_code == "aiSemis")
    assert aisemis.theme_name == "Ai Semis"


def test_predict_all_themes_supports_explicit_codes(session: Session) -> None:
    """When codes are passed, registered themes are still used for name lookup."""
    add_investment_theme(session, theme_code="aiSemis", name="AI Semiconductors")
    upsert_related_entity(
        session, source_symbol="aiSemis", related_symbol="NVDA",
        relationship="THEME_MEMBER", rank=0,
    )
    day = datetime(2026, 5, 21, tzinfo=timezone.utc)
    _seed_articles_with_scores(
        session, symbol="NVDA", scores=[0.5, 0.6, 0.7], day=day,
    )

    out = predict_all_themes(
        session, scorer=_StubScorer(score=0.5),
        on_date=day, theme_codes=["aiSemis"],
    )
    assert len(out) == 1
    assert out[0].theme_name == "AI Semiconductors"


def test_predict_all_themes_sorted_by_confidence(session: Session) -> None:
    """Two themes with signal — output is sorted by descending confidence."""
    for code in ["aiSemis", "cyberSecurity"]:
        add_investment_theme(session, theme_code=code)
    for sym in ["NVDA", "AMD"]:
        upsert_related_entity(
            session, source_symbol="aiSemis", related_symbol=sym,
            relationship="THEME_MEMBER",
        )
    upsert_related_entity(
        session, source_symbol="cyberSecurity", related_symbol="PANW",
        relationship="THEME_MEMBER",
    )

    day = datetime(2026, 5, 21, tzinfo=timezone.utc)
    # Strong signal for aiSemis (many articles, big positive).
    _seed_articles_with_scores(
        session, symbol="NVDA", scores=[0.8, 0.9, 0.7, 0.85],
        day=day, finnhub_id_start=1,
    )
    _seed_articles_with_scores(
        session, symbol="AMD", scores=[0.6, 0.7], day=day, finnhub_id_start=100,
    )
    # Weaker signal for cyberSecurity.
    _seed_articles_with_scores(
        session, symbol="PANW", scores=[0.1, 0.05, 0.0], day=day,
        finnhub_id_start=200,
    )

    out = predict_all_themes(
        session, scorer=_StubScorer(score=0.5), on_date=day,
    )
    # Both themes have signal — output sorted by descending confidence.
    confidences = [t.confidence for t in out]
    assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# RELATIONSHIPS allowed-set
# ---------------------------------------------------------------------------

def test_relationships_includes_theme_member() -> None:
    from finn_predictor.storage.repo import RELATIONSHIPS
    assert "THEME_MEMBER" in RELATIONSHIPS


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_add_theme(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(
        "FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db"
    )
    rc = main(["add-theme", "cyberSecurity", "--name", "Cyber"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["theme_code"] == "cyberSecurity"
    assert parsed["name"] == "Cyber"


def test_cli_add_theme_empty_raises(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(
        "FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db"
    )
    rc = main(["add-theme", "  "])
    assert rc == 2
    assert "non-empty" in capsys.readouterr().err


def test_cli_refresh_themes_requires_api_key(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv(
        "FINN_PREDICTOR_DB_URL", f"sqlite:///{tmp_path}/f.db"
    )
    rc = main(["refresh-themes"])
    assert rc == 2
    assert "FINNHUB_API_KEY" in capsys.readouterr().err


def test_cli_refresh_themes_happy_path(tmp_path, monkeypatch, capsys) -> None:
    db_url = f"sqlite:///{tmp_path}/f.db"
    monkeypatch.setenv("FINNHUB_API_KEY", "stub")
    monkeypatch.setenv("FINN_PREDICTOR_DB_URL", db_url)

    with patch(
        "finn_predictor.ingestion.client.FinnhubGateway.stock_investment_theme",
        return_value={"symbols": ["NVDA", "AMD"]},
    ):
        # No registered themes → uses DEFAULT_THEME_CODES list.
        # Override via --theme to keep the test small/fast.
        rc = main(["refresh-themes", "--theme", "aiSemis"])

    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["themes_added"] == 1
    assert parsed["members_added"] == 2


def test_parser_recognises_theme_commands() -> None:
    parser = _build_parser()
    args = parser.parse_args(["add-theme", "aiSemis"])
    assert args.command == "add-theme"
    assert args.theme_code == "aiSemis"

    args = parser.parse_args(
        ["refresh-themes", "--theme", "aiSemis", "--theme", "cyberSecurity"]
    )
    assert args.command == "refresh-themes"
    assert args.theme == ["aiSemis", "cyberSecurity"]

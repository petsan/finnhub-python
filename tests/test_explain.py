"""Tests for finn_predictor.predictor.explain."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.predictor.explain import (
    FLAT_SUPPORT_BAND,
    article_contributions,
    explain_prediction,
)
from finn_predictor.storage.repo import (
    save_prediction,
    save_scores,
    upsert_articles,
)
from tests.conftest import make_article, make_prediction, make_score


D = datetime(2026, 5, 19, 20, tzinfo=timezone.utc)


def _seed(session, *, ids_scores, day, category="general", symbol=None):
    """Insert one article per (id, score) pair and persist scores."""
    arts = [
        make_article(
            finnhub_id=i, category=category, symbol=symbol, published_at=day
        )
        for i, _ in ids_scores
    ]
    upsert_articles(session, arts)
    persisted = (
        session.query(type(arts[0]))
        .filter(type(arts[0]).finnhub_id.in_([i for i, _ in ids_scores]))
        .all()
    )
    by_fh = {a.finnhub_id: a for a in persisted}
    save_scores(
        session,
        [
            make_score(by_fh[i].id, s, model_version="vader-test")
            for i, s in ids_scores
        ],
    )
    return [by_fh[i] for i, _ in ids_scores]


# ---------------- article_contributions ----------------


def test_article_contributions_empty_when_no_articles(session) -> None:
    pred = save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D),
    )
    assert article_contributions(session, prediction=pred) == []


def test_article_contributions_skips_unscored(session) -> None:
    arts = [make_article(finnhub_id=i, published_at=D) for i in range(3)]
    upsert_articles(session, arts)
    persisted = session.query(type(arts[0])).all()
    # Score only the first one.
    save_scores(
        session,
        [make_score(persisted[0].id, 0.7, model_version="vader-test")],
    )
    pred = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D,
            model_version="vader-test",
        ),
    )
    contribs = article_contributions(session, prediction=pred)
    assert len(contribs) == 1
    assert contribs[0].score == pytest.approx(0.7)


def test_article_contributions_signed_and_sorted(session) -> None:
    """Top by |contribution|, signs match scores."""
    _seed(session, ids_scores=[(1, 0.9), (2, -0.8), (3, 0.1)], day=D)
    pred = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D,
            label="UP",
            model_version="vader-test",
        ),
    )
    contribs = article_contributions(session, prediction=pred)
    assert [c.article.finnhub_id for c in contribs] == [1, 2, 3]
    # Contributions must sum to the weighted mean (roughly):
    total = sum(c.contribution for c in contribs)
    assert -1.0 <= total <= 1.0


def test_article_contributions_supports_call_up(session) -> None:
    _seed(session, ids_scores=[(1, 0.9), (2, -0.5)], day=D)
    pred = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D,
            label="UP",
            model_version="vader-test",
        ),
    )
    contribs = article_contributions(session, prediction=pred)
    by_id = {c.article.finnhub_id: c for c in contribs}
    assert by_id[1].supports_call is True   # positive supports UP
    assert by_id[2].supports_call is False  # negative opposes UP


def test_article_contributions_supports_call_down(session) -> None:
    _seed(session, ids_scores=[(1, 0.6), (2, -0.7)], day=D)
    pred = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D,
            label="DOWN",
            model_version="vader-test",
        ),
    )
    by_id = {c.article.finnhub_id: c for c in
             article_contributions(session, prediction=pred)}
    assert by_id[1].supports_call is False  # positive opposes DOWN
    assert by_id[2].supports_call is True   # negative supports DOWN


def test_article_contributions_supports_call_flat(session) -> None:
    """For FLAT, only near-zero contributions count as 'supporting'."""
    _seed(session, ids_scores=[(1, 0.05), (2, 0.9)], day=D)
    pred = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D,
            label="FLAT",
            model_version="vader-test",
        ),
    )
    by_id = {c.article.finnhub_id: c for c in
             article_contributions(session, prediction=pred)}
    # Small contribution → supports FLAT; large → does not.
    assert abs(by_id[1].contribution) < FLAT_SUPPORT_BAND
    assert by_id[1].supports_call is True
    assert by_id[2].supports_call is False


def test_article_contributions_sector_uses_company_category(session) -> None:
    """Non-^GSPC targets filter on company-category articles."""
    from finn_predictor.storage.repo import ensure_default_sectors

    # Seed sectors so XLK is recognised as a sector ETF (and over-includes
    # all company news instead of filtering to symbol='XLK').
    ensure_default_sectors(session)

    _seed(session, ids_scores=[(11, 0.8)], day=D, category="general")
    _seed(session, ids_scores=[(12, 0.6)], day=D, category="company", symbol="AAPL")
    pred = save_prediction(
        session,
        make_prediction(
            target_symbol="XLK",
            prediction_date=D,
            label="UP",
            model_version="vader-test",
        ),
    )
    contribs = article_contributions(session, prediction=pred)
    finnhub_ids = {c.article.finnhub_id for c in contribs}
    # Only the company article counts; the general one is filtered out.
    assert finnhub_ids == {12}


def test_article_contributions_stock_scopes_to_ticker_symbol(session) -> None:
    """For an individual-stock target the article filter must be exact:
    only that ticker's company news contributes — not other tickers'."""
    _seed(session, ids_scores=[(20, 0.7)], day=D, category="company", symbol="AAPL")
    _seed(session, ids_scores=[(21, 0.8)], day=D, category="company", symbol="MSFT")
    pred = save_prediction(
        session,
        make_prediction(
            target_symbol="AAPL",   # individual stock, NOT in Sector table
            prediction_date=D,
            label="UP",
            model_version="vader-test",
        ),
    )
    contribs = article_contributions(session, prediction=pred)
    finnhub_ids = {c.article.finnhub_id for c in contribs}
    assert finnhub_ids == {20}  # AAPL only; MSFT excluded


# ---------------- explain_prediction ----------------


def _explain_with_seed(session, *, label, ids_scores):
    _seed(session, ids_scores=ids_scores, day=D)
    pred = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D,
            label=label,
            confidence=0.61,
            sentiment_index=0.25,
            article_count=len(ids_scores),
            model_version="vader-test",
        ),
    )
    return explain_prediction(session, prediction=pred)


def test_explain_prediction_has_five_paragraphs(session) -> None:
    text = _explain_with_seed(session, label="UP", ids_scores=[(1, 0.8), (2, -0.4)])
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    # 3 fixed + bullet list (1 paragraph) + counter-signal + caveat = at least 5.
    assert len(paragraphs) >= 5


def test_explain_prediction_mentions_symbol_label_confidence(session) -> None:
    text = _explain_with_seed(session, label="UP", ids_scores=[(1, 0.9), (2, 0.8)])
    assert "^GSPC" in text
    assert "**UP**" in text
    assert "0.61" in text  # confidence value


def test_explain_prediction_lists_top_contributors(session) -> None:
    text = _explain_with_seed(
        session, label="UP", ids_scores=[(1, 0.9), (2, 0.7), (3, -0.6)]
    )
    # Should mention "Positive movers" header
    assert "Positive movers" in text
    assert "Negative movers" in text


def test_explain_prediction_handles_empty_contributions(session) -> None:
    """No articles at all → graceful fallback paragraph instead of bullet list."""
    pred = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D,
            label="FLAT",
            sentiment_index=0.0,
            article_count=0,
            model_version="vader-test",
        ),
    )
    text = explain_prediction(session, prediction=pred)
    assert "couldn't be reconstructed" in text


def test_explain_prediction_caveats_paragraph_present(session) -> None:
    text = _explain_with_seed(session, label="UP", ids_scores=[(1, 0.9), (2, 0.8)])
    assert "Caveats" in text
    assert "VADER" in text


def test_explain_prediction_down_counter_signal(session) -> None:
    text = _explain_with_seed(session, label="DOWN", ids_scores=[(1, 0.6), (2, -0.8)])
    assert "Counter-signal" in text
    assert "🔴" in text


def test_explain_prediction_flat_explains_cancellation(session) -> None:
    text = _explain_with_seed(session, label="FLAT", ids_scores=[(1, 0.4), (2, -0.4)])
    assert "FLAT" in text
    assert "cancel" in text.lower() or "inside" in text.lower()

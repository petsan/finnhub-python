"""Explain a Prediction in plain English.

Given a :class:`Prediction` row, reconstruct the article set that fed into
it, rank each article by its signed contribution to the day's weighted
sentiment mean, and render a multi-paragraph Markdown explanation.

Why this lives in its own module: the explanation depends on private
weighting details of :mod:`aggregate` and on the classifier constants in
:mod:`market`. Keeping it separate lets the UI import a high-level
``explain_prediction(...)`` without dragging Streamlit into the predictor
package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.predictor.aggregate import _recency_weight, rolling_baseline
from finn_predictor.predictor.market import MIN_BASELINE_SIGMA, THRESHOLD_SIGMA
from finn_predictor.storage.models import (
    NewsArticle,
    Prediction,
    Sector,
    SentimentScore,
)
from finn_predictor.storage.repo import articles_in_window, utc_day_window


# A "supports FLAT" article is one whose individual contribution is near
# zero. We use this small slop to decide which dots to colour green when
# the prediction is FLAT.
FLAT_SUPPORT_BAND = 0.05


@dataclass(frozen=True)
class ArticleContribution:
    """One article's signed contribution to a Prediction's weighted index."""

    article: NewsArticle
    score: float       # raw sentiment in [-1, +1]
    weight: float      # recency weight, unit-free
    contribution: float  # score * weight / Σweights; signed
    supports_call: bool  # sign of contribution agrees with Prediction.label


def _filter_for_prediction(
    session: Session, target_symbol: str
) -> tuple[Optional[str], Optional[str]]:
    """Map a target symbol to the (category, symbol) used at predict time.

    - ``^GSPC`` → ``("general", None)``: whole-market predictor reads
      general news.
    - Sector ETFs (rows in the ``Sector`` table) → ``("company", None)``:
      the predictor passes a ticker universe at call time which we
      don't persist; we over-include here.
    - Anything else (individual stock target) → ``("company", target_symbol)``:
      scope strictly to articles tagged with that ticker.
    """
    if target_symbol == "^GSPC":
        return ("general", None)
    sector = session.scalar(
        select(Sector).where(Sector.etf_symbol == target_symbol)
    )
    if sector is not None:
        return ("company", None)
    return ("company", target_symbol)


def article_contributions(
    session: Session,
    *,
    prediction: Prediction,
    half_life_hours: float = 12.0,
) -> list[ArticleContribution]:
    """Rank articles by absolute contribution to ``prediction``.

    Articles without a SentimentScore row for ``prediction.model_version``
    are dropped — they didn't actually feed the index. Returned list is
    sorted by |contribution| descending so the UI can pull a top-N.
    """
    category, symbol = _filter_for_prediction(session, prediction.target_symbol)
    start, end = utc_day_window(prediction.prediction_date)
    articles = articles_in_window(
        session, start, end, category=category, symbol=symbol
    )
    if not articles:
        return []

    score_rows = session.execute(
        select(SentimentScore.article_id, SentimentScore.score).where(
            SentimentScore.article_id.in_([a.id for a in articles]),
            SentimentScore.model_version == prediction.model_version,
        )
    ).all()
    score_map = {aid: float(sc) for aid, sc in score_rows}

    weighted: list[tuple[NewsArticle, float, float]] = []
    for a in articles:
        if a.id not in score_map:
            continue
        w = _recency_weight(a.published_at, end, half_life_hours)
        weighted.append((a, score_map[a.id], w))

    if not weighted:
        return []

    total_w = sum(w for _, _, w in weighted) or 1.0
    label = prediction.label
    out: list[ArticleContribution] = []
    for a, s, w in weighted:
        c = s * w / total_w
        if label == "UP":
            supports = c > 0
        elif label == "DOWN":
            supports = c < 0
        else:  # FLAT
            supports = abs(c) < FLAT_SUPPORT_BAND
        out.append(
            ArticleContribution(
                article=a, score=s, weight=w, contribution=c, supports_call=supports
            )
        )

    out.sort(key=lambda c: abs(c.contribution), reverse=True)
    return out


def _bullet(c: ArticleContribution) -> str:
    """One Markdown bullet describing one contribution."""
    src = f" *({c.article.source})*" if c.article.source else ""
    return f"  - **{c.contribution:+.3f}** — \"{c.article.headline}\"{src}"


def explain_prediction(
    session: Session,
    *,
    prediction: Prediction,
    contributions: Optional[Sequence[ArticleContribution]] = None,
) -> str:
    """Five-paragraph Markdown explanation of why ``prediction`` was made.

    Layout:
        1. Call summary — what the model is calling and how strongly.
        2. Mechanics — how the index compares to the rolling baseline.
        3. Top contributors — biggest individual movers (signed).
        4. Counter-signals — what was pushing the other way.
        5. Caveats — known limitations of iteration 1.
    """
    if contributions is None:
        contributions = article_contributions(session, prediction=prediction)

    sym = prediction.target_symbol
    label = prediction.label
    conf = prediction.confidence
    n = prediction.article_count
    idx = prediction.sentiment_index
    plural = "" if n == 1 else "s"
    date_str = prediction.prediction_date.strftime("%Y-%m-%d")

    # Paragraph 1 — what we're calling.
    p1 = (
        f"The model is calling **{label}** on `{sym}` for **{date_str}** with "
        f"confidence **{conf:.2f}** (where 0 means no signal and 1 means the "
        f"classifier's maximum). The call rests on **{n}** scored article{plural} "
        f"with an aggregate recency-weighted sentiment index of "
        f"**{idx:+.3f}** on the `{prediction.model_version}` scale, which runs "
        f"from −1 (most negative) to +1 (most positive)."
    )

    # Paragraph 2 — how the index becomes a label.
    category, _ = _filter_for_prediction(session, sym)
    baseline = rolling_baseline(
        session,
        model_version=prediction.model_version,
        end_day=prediction.prediction_date,
        category=category,
    )
    sigma = max(baseline.stddev, MIN_BASELINE_SIGMA)
    z = (idx - baseline.mean) / sigma
    p2 = (
        f"To turn that index into a directional call, the classifier compares "
        f"today against a 30-day rolling baseline. Right now the baseline mean "
        f"is **{baseline.mean:+.3f}** with σ = **{baseline.stddev:.3f}** "
        f"(clamped to a floor of {MIN_BASELINE_SIGMA:.2f} on quiet weeks to "
        f"avoid divide-by-zero). That puts today at **z = {z:+.2f}** relative "
        f"to the baseline. The rule is: emit **UP** when z > +{THRESHOLD_SIGMA}, "
        f"**DOWN** when z < −{THRESHOLD_SIGMA}, **FLAT** otherwise — which is "
        f"why this lands on **{label}**. Confidence is |z| ⁄ 2, clipped to 1."
    )

    # Paragraph 3 — biggest movers.
    pos = [c for c in contributions if c.contribution > 0]
    neg = [c for c in contributions if c.contribution < 0]
    if pos or neg:
        lines: list[str] = []
        if pos:
            lines.append("Positive movers (pushed toward UP):")
            lines.extend(_bullet(c) for c in pos[:3])
        if neg:
            lines.append("Negative movers (pushed toward DOWN):")
            lines.extend(_bullet(c) for c in neg[:3])
        p3 = (
            "The biggest individual movers (signed contribution = score × weight ⁄ "
            "Σweights) inside today's article window:\n\n" + "\n".join(lines)
        )
    else:
        p3 = (
            "Per-article contribution detail couldn't be reconstructed for this "
            f"prediction — either no scored articles exist in today's window, "
            f"or no score rows for `{prediction.model_version}` have been "
            "written yet."
        )

    # Paragraph 4 — counter-signal narrative.
    if label == "UP":
        if neg:
            p4 = (
                f"Not every article in the window agreed: **{len(neg)} of "
                f"{len(pos) + len(neg)}** scored articles were net-negative and "
                f"dampened the call. If they had outweighed the {len(pos)} "
                f"positive article{'' if len(pos) == 1 else 's'}, the call "
                f"would have flipped to DOWN or FLAT. Those articles are "
                f"flagged 🔴 in the headline list below."
            )
        else:
            p4 = (
                "Every scored article in today's window pushed in the same "
                "direction — no internal counter-signal. Worth noting because "
                "uniformity often means a single news event is dominating the feed."
            )
    elif label == "DOWN":
        if pos:
            p4 = (
                f"Counter-signal: **{len(pos)} of {len(pos) + len(neg)}** "
                f"scored articles in today's window were positive, fighting "
                f"against the **DOWN** call. The model still tilts negative "
                f"because the negative articles carried more weight — the "
                f"weighted sum sits {abs(z):.2f}σ below baseline. Positive "
                f"articles are flagged 🔴 in the headline list (they oppose "
                f"the call)."
            )
        else:
            p4 = (
                "No positive articles in today's window — the negative reading "
                "is uncontested by the feed itself."
            )
    else:  # FLAT
        p4 = (
            f"Positive contributors: **{len(pos)}**; negative contributors: "
            f"**{len(neg)}**. They roughly cancel, leaving today's z-score "
            f"inside ±{THRESHOLD_SIGMA}σ of the baseline — hence **FLAT**. A "
            f"FLAT call doesn't mean 'no news'; it means the net direction "
            f"isn't different enough from the recent norm to bet on."
        )

    # Paragraph 5 — caveats.
    p5 = (
        "**Caveats.** Iteration 1's classifier is a hand-coded threshold rule, "
        "not a fitted statistical model — the confidence number is a "
        "normalised z-distance, not a probability. VADER scores English affect "
        "well but is weak on finance-specific jargon (\"beat estimates\", "
        "\"guidance lowered\", \"misses on top line\"); iteration 2 plans to "
        "swap in FinBERT under the same `Scorer` interface so historical calls "
        "remain comparable across model versions. The call is sentiment-only: "
        "it ignores price action, the earnings calendar, macro data releases, "
        "and intraday microstructure. Backtest accuracy across past predictions "
        "is on the *History* tab — treat this as a market-mood gauge, not a "
        "trade signal."
    )

    return "\n\n".join([p1, p2, p3, p4, p5])

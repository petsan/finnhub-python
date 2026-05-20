# Finn-Predictor — Summary

A sentiment-driven, crude market & per-stock direction predictor built on top of
the upstream `finnhub-python` client. Built interactively on the `finn-predictor`
branch; left the upstream library untouched.

## What it does

* Pulls news + (when the user's plan allows) daily price candles from Finnhub.
* Scores every article with VADER (lexicon-based, lightweight); a FinBERT
  scorer is wired behind the same `Scorer` interface for a drop-in upgrade.
* Aggregates the day's recency-weighted sentiment, compares it to a 30-day
  rolling baseline, and emits an **UP / DOWN / FLAT** call with a confidence
  number for each of:
  * **the whole market** (`^GSPC`, from `/news?category=general`),
  * **each sector** (the 11 Sector SPDR ETFs, from per-constituent
    `/company-news`),
  * **each user-listed stock** (per-ticker `/company-news` for the symbols
    the user supplies in the sidebar).
* Stores everything in SQLite. The full prediction history is queryable;
  re-running on the same UTC day upserts the same row (one prediction per
  target per day per model).
* Backfills up to ~1 year of `/company-news` per ticker in one click, then
  walks each historical UTC day to populate the prediction history and the
  rolling baseline.

## Architecture (three layers, plus cross-cutting)

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1 — Web UI (Streamlit)                            │
│    Today  · per-stock table · contribution chart ·       │
│    "Why this Call?" 5-¶ explanation · linked headlines   │
│    History · Sectors                                     │
│    Sidebar: session-only API key, daily ingest button,   │
│             "Backfill historical news" with lookback     │
└──────────────▲────────────────────────────────────▲──────┘
               │ reads only                          │ reads
┌──────────────┴────────────────────┐  ┌────────────┴──────┐
│  Layer 3 — Recommendation engine  │  │  Backtester       │
│    z-score classifier             │  │    pairs each pred│
│    market / sector / stock        │  │    with realised  │
│    retroactive replay             │  │    next-bar return│
└──────────────▲────────────────────┘  └─────────▲─────────┘
               │ reads/writes                    │
┌──────────────┴─────────────────────────────────┴─────────┐
│  Layer 2 — Storage (SQLite via SQLAlchemy)               │
│    news_articles · sentiment_scores · price_bars ·       │
│    predictions · prediction_outcomes · sectors           │
└──────────────▲───────────────────────────────────────────┘
               │ writes
┌──────────────┴───────────────────────────────────────────┐
│  Ingestion (FinnhubGateway → finnhub.Client)             │
│    daily ingest · historical backfill · rate-limit       │
│    + retry + token scrubbing + proxy bypass              │
└──────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
# from repo root
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install sqlalchemy apscheduler vaderSentiment streamlit \
                     pandas altair pytest pytest-cov pytest-mock \
                     requests-mock freezegun

# run the dashboard (no key required to boot; UI surfaces a key input)
.venv/bin/streamlit run finn_predictor/ui/app.py

# tests + coverage
.venv/bin/python -m pytest --cov
```

Then open <http://127.0.0.1:8501>, paste a Finnhub key in the sidebar (it
lives only in `st.session_state` — never on disk, never in the DB, never in
logs), and click **Run ingestion now**. To bootstrap the rolling baseline,
use **Backfill historical news** with a ticker list and a lookback window
of up to a year.

## Key user-facing features

| Feature | Notes |
|---|---|
| Session-only API key | Masked input, lives in server memory, cleared on tab close. |
| Triple-layer key scrubbing | Errors (including `requests.SSLError` URL leaks) get `<REDACTED>` before display, logging, or persistence. Defence-in-depth at gateway, helper, and UI display layers. |
| HTTPS_PROXY bypass | The ingestion client sets `trust_env=False` so a dev proxy (Burp / mitmproxy) doesn't break TLS verification. |
| Resilient ingest | A 403 on `/stock/candle` doesn't abort the run; `/news` still ingests. Per-op failures surface in a collapsed expander. |
| News-feed empty vs. failed | Sidebar distinguishes "`/news` 403 — key/plan issue" from "`/news` returned 0 today, candles gated as usual". |
| Per-stock predictions table | Sorted by confidence with Streamlit `ProgressColumn`. Company names expanded from a curated map (`^GSPC` → S&P 500 Index, `XLK` → Information Technology, mega-cap tickers → company names). |
| "Why this Call?" explanation | 5 paragraphs per market/sector call, in a 480-px scrollable container. Paragraph topics: call summary · mechanics · top movers · counter-signal · caveats. |
| Per-article contribution chart | Divergent vertical Altair bars: x-axis sorted -1→0→+1 by signed contribution, y-axis is signed contribution centered on a 0-line. Tooltip includes headline, source, ticker, expanded company name, published, first-reported, sentiment, contribution, supports-call. |
| Story clustering | Each article's "first reported" timestamp comes from the earliest member of its story cluster (prefix-matched `story_key`), with a 5-minute wire-flash window to suppress same-minute republishes from cluttering the line. |
| Predictions deduped | `prediction_date` normalised to start-of-UTC-day so multiple ingests on the same day upsert one row. One-shot migration cleans up older duplicates. |
| Historical backfill | Pages `/company-news` in 30-day chunks per ticker, dedupes against the DB, scores new articles, then writes one Prediction per UTC day in the window. Builds up the rolling baseline immediately. |

## Limitations

* **General market news has no history** — Finnhub's `/news` only paginates
  forward. The market call's baseline can only build up over wall-clock time.
* **`/stock/candle` is gated** on the free tier, so backtest accuracy
  (`prediction_outcomes`) stays empty until you upgrade. Predictions still
  write fine; only the realised-return validation is missing.
* **Classifier is a rule, not a fitted model.** Confidence is normalised
  z-distance, not probability. Day-1 of a new ticker can pin at conf=1.0
  before the rolling baseline accumulates 30 days of history.
* **VADER is general-purpose.** Financial jargon ("beat estimates", "guidance
  lowered") is under-weighted. FinBERT swap-in is wired but not active by
  default to keep the dependency tree small.

## Test posture

Every external HTTP call goes through `FinnhubGateway`, which is mocked
in tests — **no test makes a real Finnhub call**. SQLite-backed tests run
against `:memory:` per-test, giving fast and isolated coverage. As of the
latest commit: **213 tests passing at 98% line+branch coverage** across the
`finn_predictor` package.

See `progress.md` for the design doc and `diff.md` for the per-commit
change log.

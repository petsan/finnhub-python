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

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Web UI (Streamlit)                                      │
│    Today      · per-stock table · contribution chart · 5-¶         │
│               "Why this Call?" · linked headlines                  │
│    History    · prediction timeline                                │
│    Sectors    · per-sector grid                                    │
│    Performance · hit-rate + cumulative PnL charts + trade ledger   │
│    Focus      · Company / Sector / Event drill-downs               │
│    Learning   · activation policy + holdout gate + Bayesian        │
│               retrain + version history                            │
│    Sidebar    · session-only API key (optional bcrypt password    │
│               gate around the whole UI) · daily-ingest +          │
│               historical-news + yfinance-prices backfills          │
└──────────────▲──────────────────────────▲──────────────────▲──────┘
               │ reads only                │ reads             │ writes
┌──────────────┴────────┐  ┌───────────────┴─────────┐  ┌─────┴─────────────────┐
│  Recommendation       │  │  Backtester             │  │  Self-improvement     │
│  engine               │  │    pairs each pred with │  │    Bayesian opt over  │
│    z-score classifier │  │    realised next-bar    │  │    threshold / σ floor│
│    market / sector /  │  │    return; signs PnL by │  │    / half-life /      │
│    stock              │  │    Call direction       │  │    per-source weights │
│    explanation block  │  │                         │  │    versioned active   │
│    focus composer     │  │                         │  │    set in DB          │
│    retroactive replay │  │                         │  │    holdout gate       │
└──────────────▲────────┘  └─────────────▲───────────┘  └─────────▲─────────────┘
               │                          │                         │
┌──────────────┴──────────────────────────┴─────────────────────────┴───────────┐
│  Layer 2 — Storage (SQLAlchemy; SQLite by default, PostgreSQL supported)      │
│    news_articles · sentiment_scores · price_bars · predictions ·              │
│    prediction_outcomes · sectors · related_entities · learned_weights ·       │
│    app_settings                                                               │
└──────────────▲────────────────────────────────────────────────────────────────┘
               │ writes
┌──────────────┴────────────────────────────────────────────────────────────────┐
│  Ingestion                                                                    │
│    FinnhubGateway → finnhub.Client                                            │
│      news (general + per-company) · per-target-and-sector-ETF candles ·       │
│      peers · supply chain · ETF holdings · profile2                           │
│    yfinance (no key required) → daily OHLC for every prediction target        │
│    Three-layer API-token scrubbing · proxy bypass · per-endpoint failure      │
│    isolation · rate-limit + retry                                             │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
# from repo root
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

# run the dashboard (no key required to boot; UI surfaces a key input)
.venv/bin/streamlit run finn_predictor/ui/app.py

# tests + coverage
.venv/bin/pip install pytest pytest-cov pytest-mock requests-mock freezegun
.venv/bin/python -m pytest --cov
```

### Docker

```bash
docker compose up --build                  # build + serve on http://localhost:8501
                                           # reuses any data in the finn_data volume

RESET_DB=1 docker compose up --build       # wipe DB on the way up

docker compose run --rm app reset-db --yes # wipe DB, exit
docker compose run --rm app retrain        # run one training cycle, print JSON
docker compose run --rm app shell          # interactive bash inside the container
```

The SQLite file lives in a named volume (`finn_data`), so `docker compose
down` keeps your articles, predictions, outcomes, and learned weights.
`docker compose down -v` deletes the volume.

Then open <http://127.0.0.1:8501>, paste a Finnhub key in the sidebar (it
lives only in `st.session_state` — never on disk, never in the DB, never in
logs), and click **Run ingestion now**. To bootstrap the rolling baseline,
use **Backfill historical news** with a ticker list and a lookback window
of up to a year.

## Key user-facing features

| Feature | Notes |
|---|---|
| Session-only API key | Masked input, lives in server memory, cleared on tab close. |
| Triple-layer key scrubbing | Errors (including `requests.SSLError` URL leaks) get `<REDACTED>` before display, logging, or persistence. Defence-in-depth at gateway, helper, and UI display layers. A regex-based `SecretScrubFilter` on every logging handler is the fourth safety net. |
| HTTPS_PROXY bypass | The ingestion client sets `trust_env=False` so a dev proxy (Burp / mitmproxy) doesn't break TLS verification. |
| Resilient ingest | A 403 on `/stock/candle` doesn't abort the run; `/news` still ingests. Per-op failures surface in a collapsed expander. |
| News-feed empty vs. failed | Sidebar distinguishes "`/news` 403 — key/plan issue" from "`/news` returned 0 today, candles gated as usual". |
| Per-stock predictions table | Sorted by confidence with Streamlit `ProgressColumn`. Company names expanded from a curated map (`^GSPC` → S&P 500 Index, `XLK` → Information Technology, mega-cap tickers → company names). |
| "Why this Call?" explanation | 5 paragraphs per market/sector call, in a 480-px scrollable container. Paragraph topics: call summary · mechanics · top movers · counter-signal · caveats. |
| Per-article contribution chart | Divergent vertical Altair bars: x-axis sorted -1→0→+1 by signed contribution, y-axis is signed contribution centered on a 0-line. Tooltip includes headline, source, ticker, expanded company name, published, first-reported, sentiment, contribution, supports-call. |
| Story clustering | Each article's "first reported" timestamp comes from the earliest member of its story cluster (prefix-matched `story_key`), with a 5-minute wire-flash window to suppress same-minute republishes from cluttering the line. |
| Predictions deduped | `prediction_date` normalised to start-of-UTC-day so multiple ingests on the same day upsert one row. One-shot migration cleans up older duplicates. |
| Historical news backfill | Pages `/company-news` in 30-day chunks per ticker, dedupes against the DB, scores new articles, then writes one Prediction per UTC day in the window. Builds up the rolling baseline immediately. |
| yfinance price backfill | Pulls daily OHLC for every prediction target — fills the gap left by Finnhub's gated `/stock/candle`. No API key needed. Runs the backtester after, closing newly-paired predictions. |
| Performance tab | 5 summary metrics, 4 Altair charts (cumulative PnL · rolling 14-trade hit-rate with 50% reference line · hit-rate by target kind · hit-rate by Call), sortable trade ledger. |
| Focus tab | Three modes — Company (subject + peers + supply chain + recent articles), Sector (ETF + cached constituents), Event (free-text news search with implied Call). |
| Self-improvement | Bayesian-optimisation retrain over `threshold_sigma`, `min_baseline_sigma`, `half_life_hours`, plus closed-form per-source weights. All four knobs applied at **live** scoring time (not just inside the simulator). Versioned `LearnedWeight` rows; old versions kept for revert. |
| Activation policy | `AUTO` (newest version wins) or `MANUAL` (user clicks Activate). Persisted across Streamlit restarts via an `app_settings` table. |
| Holdout-improvement gate | Under AUTO policy, refuses to activate a new version whose holdout score drops by more than the configured tolerance vs. the live config. UI shows the gap + an "Activate anyway" override. |
| Auth gate | Optional bcrypt password gate around the entire UI. Generate the hash via `python -m finn_predictor.cli hash-password`, set `FINN_PREDICTOR_PASSWORD_HASH`. Unset → no auth (safe default for localhost). |
| Postgres support | Set `FINN_PREDICTOR_DB_URL="postgresql+psycopg://..."` — the upserts are dialect-aware. `psycopg[binary]` ships in default requirements. |
| Structured logging | `FINN_PREDICTOR_LOG_FORMAT=json` switches to one-record-per-line JSON output suitable for log aggregators. |
| Docker | `docker compose up --build` builds and starts the app on host 8501. Named volume keeps the SQLite DB across `down`. `RESET_DB=1` wipes on next start. `docker compose run --rm app ingest|retrain|reset-db|shell|hash-password` for headless ops. Runs as non-root. |

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

Every external HTTP call goes through `FinnhubGateway` (Finnhub) or an
injectable `history_fn` (yfinance), both mocked in tests — **no test
makes a real network call**. SQLite-backed tests run against `:memory:`
per-test, giving fast and isolated coverage. As of the latest commit:
**337 tests passing at 95% line+branch coverage** across the
`finn_predictor` package.

See `progress.md` for the design doc, `diff.md` for the per-commit
change log, `user-manual.md` for how to drive the dashboard, and
`installation-manual.md` for how to install / deploy / harden it.

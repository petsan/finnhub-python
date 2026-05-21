# Finn-Predictor — Project Progress

Status: **Planning phase** — design under review, no application code written yet.
Branch: `finn-predictor`
Library version under review: `finnhub-python` 2.4.28

---

## 1. Code Review of the Existing Finnhub Python Client

### 1.1 Repo layout

```
finnhub-python/
├── finnhub/
│   ├── __init__.py     # exports Client, FinnhubAPIException, FinnhubRequestException
│   ├── client.py       # ~560 LOC, single Client class, ~110 endpoint methods
│   └── exceptions.py   # 2 custom exception classes
├── examples.py         # ad-hoc smoke script (reads FINNHUB_API_KEY env var)
├── README.md           # usage examples
├── setup.py            # package metadata, version 2.4.28
├── requirements.txt    # runtime deps: requests >= 2.22.0
├── test-requirements.txt  # pytest, pytest-cov, pytest-randomly
├── tox.ini             # tox envs py27/py3, runs pytest --cov=finnhub
├── .gitlab-ci.yml
├── .travis.yml
├── CHANGELOG.md
├── LICENSE             # Apache-2.0
└── release.sh, git_push.sh
```

### 1.2 `finnhub/client.py` — Architecture

- **Single class** `Client` (`finnhub/client.py:8`). All API methods are thin wrappers around `_get`.
- **Constants**: `API_URL = "https://api.finnhub.io/api/v1"`, `DEFAULT_TIMEOUT = 10` seconds.
- **Auth**: API key passed in `session.params["token"]` so every request carries it as a query param. Settable via the `api_key` property setter (`client.py:78`).
- **Transport**: `requests.Session` initialised once per `Client`. Optional `proxies` dict.
- **Context manager support**: `__enter__`/`__exit__` close the session.
- **Request flow**: `_request → _format_params → session.<method> → _handle_response`. Booleans get JSON-stringified by `_format_params` (`client.py:67`).
- **Response decoding** (`_handle_response`, `client.py:43`):
  - non-2xx → `FinnhubAPIException`
  - `application/json` → parsed JSON
  - `text/csv` / `text/plain` → raw text
  - anything else → `FinnhubRequestException`
- **Helpers**: `_merge_two_dicts` for kwargs splat; only `GET` is implemented (`_get`).

### 1.3 `finnhub/exceptions.py`

- `FinnhubAPIException`: built from a `requests.Response`. Tries to read `response.json()["error"]`, falls back to body text. Carries `status_code`, `message`, `response`. Bug-watch: `self.code` is set to `0` and never updated — only `status_code` is meaningful.
- `FinnhubRequestException`: simple message wrapper used for unparseable responses.

### 1.4 Endpoint inventory (the full surface)

Endpoints grouped by usefulness to **Finn-Predictor**. File:line references point at `finnhub/client.py`.

#### 1.4.1 News & sentiment — primary inputs for the predictor

| Method | Endpoint | Returns | Use in predictor |
|---|---|---|---|
| `general_news(category, min_id=0)` `client.py:311` | `/news` | List of articles (categories: `general`, `forex`, `crypto`, `merger`) | **Iter 1**: whole-market sentiment feed |
| `company_news(symbol, _from, to)` `client.py:317` | `/company-news` | List of articles for a ticker | **Iter 2**: per-company / per-sector sentiment |
| `news_sentiment(symbol)` `client.py:324` | `/news-sentiment` | Finnhub's pre-computed sentiment (companyNewsScore, sectorAverage) | Both iterations — useful as a baseline / reference signal |
| `press_releases(symbol, _from, to)` `client.py:102` | `/press-releases` | Major-development releases | Higher-signal feed; useful for sector iteration |
| `newsroom(symbol, _from, to)` `client.py:532` | `/stock/newsroom` | Company newsroom items | Secondary source |
| `stock_social_sentiment(symbol, _from, to)` `client.py:427` | `/stock/social-sentiment` | Reddit / Twitter sentiment buckets | Useful alt-signal for Iter 2 |
| `sec_sentiment_analysis(access_number)` `client.py:388` | `/stock/filings-sentiment` | Sentiment scores for an SEC filing | Long-horizon signal; not for daily prediction |

#### 1.4.2 Price data — prediction target & training labels

| Method | Endpoint | Returns | Use |
|---|---|---|---|
| `stock_candles(symbol, resolution, _from, to)` `client.py:219` | `/stock/candle` | OHLCV candles | Train labels (next-day return), backtesting |
| `quote(symbol)` `client.py:205` | `/quote` | Current price, daily change | Live "today's outcome" |
| `indices_const(symbol)` `client.py:364` | `/index/constituents` | Constituents of e.g. `^GSPC` | Universe selection |
| `indices_hist_const(symbol)` `client.py:367` | `/index/historical-constituents` | Historical members | Avoids survivorship bias |
| `historical_market_cap(symbol, _from, to)` `client.py:442` | `/stock/historical-market-cap` | Daily market-cap series | Weighting sector aggregates |
| `stock_splits(symbol, _from, to)` `client.py:304` | `/stock/split` | Splits | Price adjustment |
| `stock_dividends(symbol, _from, to)` `client.py:112` | `/stock/dividend` | Dividends | Total-return adjustment |

#### 1.4.3 Sector & classification — needed for Iteration 2

| Method | Endpoint | Use |
|---|---|---|
| `company_profile2(symbol=...)` `client.py:87` | `/stock/profile2` | Free-tier company profile → `finnhubIndustry` field for sector |
| `etfs_sector_exp(symbol, isin)` `client.py:376` | `/etf/sector` | SPY/IWM sector weights — useful for whole-market sector mix |
| `etfs_holdings(symbol, ...)` `client.py:373` | `/etf/holdings` | Sector-ETF (XLK, XLE, XLF, …) constituents → sector universe |
| `sector_metric(region)` `client.py:472` | `/sector/metrics` | Aggregate sector valuation metrics |
| `stock_investment_theme(theme)` `client.py:430` | `/stock/investment-theme` | Themed tickers |
| `stock_supply_chain(symbol)` `client.py:433` | `/stock/supply-chain` | Cross-sector contagion (future enhancement) |

#### 1.4.4 Supporting endpoints (used opportunistically)

- Analyst signals: `recommendation_trends` `client.py:126`, `price_target` `client.py:129`, `upgrade_downgrade` `client.py:132`.
- Earnings: `company_earnings` `client.py:164`, `earnings_calendar` `client.py:350`, `earnings_call_live` `client.py:523` (event-driven sentiment windows).
- Macro: `economic_data` `client.py:341`, `calendar_economic` `client.py:344`, `country` `client.py:335`.
- Calendar / status: `market_status` `client.py:511`, `market_holiday` `client.py:514` (skip ingestion on closed days).
- Lookup: `symbol_lookup(query)` `client.py:400`, `stock_symbols(exchange,…)` `client.py:122`.

#### 1.4.5 Out of scope for now

Forex (`forex_*`), crypto (`crypto_*`), bonds (`bond_*`), mutual funds (`mutual_fund_*`), tick data (`stock_tick`, `stock_nbbo`, `bond_tick`), institutional 13-F (`institutional_*`), congressional / lobbying / USPTO / visa / USA-spending alt-data, ETF country exposure, FDA calendar, bank branches, airline price index. All available for later iterations.

### 1.5 Observations & gotchas

1. **No tests in the repo.** `test-requirements.txt` and `tox.ini` are configured but no `tests/` directory exists. Any code we add will be the project's first test suite. The instruction "all the code you write to be fully tested" applies to *our* application code, not the upstream library.
2. **`from` is a Python keyword** — every endpoint that takes a `from` query param uses the parameter name `_from`. We must keep this convention in our wrappers.
3. **Rate limiting is not handled.** A 429 from Finnhub will raise `FinnhubAPIException` with `status_code=429` but no retry/backoff. Our ingestion layer must implement throttling + retry.
4. **`FinnhubAPIException.code` is a dead field** (always `0`). Use `status_code`.
5. **Booleans get `json.dumps`'d** by `_format_params` — useful to know if we wrap any boolean-flag endpoints.
6. **CSV and plain-text responses are returned as raw strings**, not parsed. We probably won't hit these but worth noting.
7. **Timeout is 10s** (overridable per call via `kwargs`). For batch jobs we may want to raise it.
8. **API key carried in URL query** (`?token=…`) — anything that logs full URLs will leak the key. We must scrub before logging.
9. **No async support.** For high-throughput ingestion we'll need either `concurrent.futures` against the sync client or our own `aiohttp` wrapper.
10. **Version drift**: `setup.py` says 2.4.28; `README.md` line 5 says 2.4.25. Cosmetic — surface it if we ever upstream a fix.

---

## 2. Application Plan

### 2.1 Tech-stack decisions (locked 2026-05-19)

| Concern | Choice | Notes |
|---|---|---|
| Web UI | **Streamlit** | Fast path to a usable dashboard; the recommendation engine stays in plain Python so it remains independently testable. |
| Database | **SQLite via SQLAlchemy** | File-on-disk for the prototype; the ORM lets us swap to Postgres later without touching call sites. |
| Sentiment | **Hybrid: VADER first, FinBERT later** | Define an abstract `Scorer` interface; ship `VaderScorer` in iter 1, drop in `FinBertScorer` in iter 2 with no caller changes. |
| Scheduler | **APScheduler, in-process** | Same process as Streamlit; single deployable; trivial to unit-test by running the job function directly. |
| Config | Env vars (`FINNHUB_API_KEY`, `FINN_PREDICTOR_DB_URL`) | API key never written to disk or logs. |

### 2.2 Proposed package layout

```
finn_predictor/
├── __init__.py
├── config.py              # env-var loader, settings dataclass
├── ingestion/
│   ├── __init__.py
│   ├── client.py          # thin retry/rate-limit wrapper around finnhub.Client
│   ├── news.py            # general & company news ingestion
│   ├── prices.py          # ^GSPC + sector ETF candles
│   └── jobs.py            # APScheduler job definitions
├── storage/
│   ├── __init__.py
│   ├── models.py          # SQLAlchemy ORM: NewsArticle, SentimentScore,
│   │                      #   PriceBar, Prediction, PredictionOutcome, Sector
│   ├── repo.py            # repository functions used by the rest of the app
│   └── migrations/        # Alembic
├── sentiment/
│   ├── __init__.py
│   ├── base.py            # Scorer protocol/ABC
│   ├── vader.py           # VaderScorer (iter 1)
│   └── finbert.py         # FinBertScorer (iter 2; lazy-imported)
├── predictor/
│   ├── __init__.py
│   ├── market.py          # whole-market predictor (iter 1)
│   ├── sectors.py         # per-sector predictor (iter 2)
│   └── backtest.py        # historical replay, hit-rate, simple PnL
└── ui/
    └── app.py             # Streamlit entrypoint

tests/
├── conftest.py            # in-memory SQLite, requests_mock, frozen time
├── ingestion/
├── storage/
├── sentiment/
├── predictor/
└── ui/                    # smoke-test via streamlit.testing.v1.AppTest
```

### 2.3 Iteration 1 — Whole-market predictor

**Data flow per day `D`:**
1. APScheduler triggers `ingestion.jobs.daily_ingest` after US market close.
2. `ingestion.news.fetch_general_news()` calls `finnhub_client.general_news('general')`, dedupes on Finnhub `id`, writes to `news_articles`.
3. `ingestion.prices.fetch_index_bars()` pulls `stock_candles('^GSPC', 'D', _from, to)` and writes to `prices`.
4. `sentiment` module scores every unscored article via the active `Scorer`, writes to `sentiment_scores(article_id, score, model_version)`.
5. `predictor.market.predict(D)` computes:
   - `S_D` = recency-weighted mean of `sentiment_scores` over the last 24h
   - `μ_30d`, `σ_30d` = rolling stats from prior 30 days
   - **Rule (v1)**: `UP` if `S_D > μ_30d + 0.5σ`, `DOWN` if `S_D < μ_30d − 0.5σ`, else `FLAT`
   - confidence = `|S_D − μ_30d| / σ_30d` clipped to [0, 1]
6. Writes `predictions(date, label, confidence, model_version)`.
7. The following session, `backtest.score_outcomes()` reads next-day return from `prices` and writes `prediction_outcomes(prediction_id, realised_return, hit)`.

**Streamlit UI (iter 1):**
- Page 1 *Today*: big UP/DOWN/FLAT, confidence bar, last 10 headlines with per-article sentiment.
- Page 2 *History*: prediction vs. realised return time series + rolling hit-rate.
- Page 3 *Diagnostics*: ingestion run log, last error, rate-limit budget.

**Test plan:**
- Unit: `VaderScorer` against fixtures (positive/negative/neutral headlines).
- Unit: aggregator math against synthetic score arrays (covers empty days, single-article days, σ=0 guard).
- Unit: predictor labelling thresholds.
- Integration: ingestion against `requests_mock` with canned Finnhub responses → asserts DB state.
- Integration: end-to-end predict→outcome cycle on an in-memory SQLite, with `freezegun` advancing dates.
- UI smoke: `streamlit.testing.v1.AppTest` boots the app and reads back rendered text.

### 2.4 Iteration 2 — Sector-based predictor

- Sector universe via `etfs_holdings('XLK' | 'XLE' | 'XLF' | 'XLV' | 'XLY' | 'XLP' | 'XLI' | 'XLB' | 'XLU' | 'XLRE' | 'XLC')`.
- Per-ticker news from `company_news(ticker, _from, to)`.
- Weight each ticker's sentiment by market cap (via `historical_market_cap`).
- Per-sector prediction targets each sector ETF's next-session candle.
- Activate `FinBertScorer` (lazy-loaded) — same interface, drop-in replacement.
- UI grows a sector grid with per-sector confidence and hit-rate; backtester reports per-sector accuracy.

### 2.5 Out of scope
- Live trading, broker integration, real money.
- Intraday predictions (daily resolution only).
- Forex / crypto / bonds (Finnhub endpoints exist; deferred).
- Auth / multi-user (single-user local app).

### 2.6 Test & quality bar
- `pytest --cov=finn_predictor` ≥ 90% line coverage on `predictor/`, `sentiment/`, `storage/`.
- Every external HTTP call goes through the wrapper in `ingestion/client.py` and is mocked in tests — **no test makes real network calls**.
- `make check` (or `nox`/`tox`) runs lint + tests.

---

---

## 3. Implementation Status (2026-05-19)

### 3.1 What landed

| Layer | Module(s) | Status |
|---|---|---|
| Config | `finn_predictor/config.py` | ✅ env-var loader with frozen `Settings` |
| Storage | `finn_predictor/storage/{models,engine,repo}.py` | ✅ 6 ORM tables, idempotent upserts, SQLite-tz roundtrip handled |
| Ingestion | `finn_predictor/ingestion/{client,news,prices,jobs}.py` | ✅ `RateLimiter` + retrying `FinnhubGateway`, news/price fetchers, APScheduler glue |
| Sentiment | `finn_predictor/sentiment/{base,vader,finbert}.py` | ✅ `Scorer` Protocol, `VaderScorer` (active), `FinBertScorer` (lazy, injectable pipeline) |
| Predictor | `finn_predictor/predictor/{aggregate,market,sectors,backtest}.py` | ✅ iter-1 whole-market + iter-2 per-sector + backtester |
| UI | `finn_predictor/ui/app.py` | ✅ Streamlit dashboard with Today / History / Sectors tabs |
| Tests | `tests/*.py` | ✅ 90 passing |

### 3.2 Iteration 1 — whole-market predictor

End-to-end pipeline:

1. `ingestion.jobs.run_daily_ingest` pulls `general_news` + `^GSPC` candles.
2. `score_pending_articles` runs the active `Scorer` on any unscored rows.
3. `predictor.market.predict_market` computes `S_D`, compares to rolling 30-day baseline, classifies via z-score → writes a `Prediction` row.
4. `predictor.backtest.score_outcomes` pairs each prediction with the next-session close once it lands.
5. Streamlit *Today* / *History* tabs read from the DB only.

Key numbers (configurable in `predictor.market`):
- `THRESHOLD_SIGMA = 0.5`
- `MIN_BASELINE_SIGMA = 0.05` (divide-by-zero floor)
- `MIN_ARTICLES_FOR_CALL = 3` (below this we emit `FLAT` with zero confidence)

### 3.3 Iteration 2 — sector-based predictor

- Default sector universe seeded by `ensure_default_sectors`: 11 Sector Select SPDR ETFs (XLK, XLE, XLF, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC).
- `predict_sector` filters articles to `category="company"` + the caller-supplied ticker list, applies the same z-score classification as iter 1, and writes a `Prediction` tagged with the sector's ETF symbol.
- `predict_all_sectors(sector_universe={...})` runs the predictor across every persisted sector; sectors without a ticker universe are skipped.
- Streamlit *Sectors* tab renders a one-row-per-sector grid.
- `FinBertScorer` is the planned drop-in replacement for the per-sector pass — interface is identical to `VaderScorer`, only the `model_version` field differs (so historical predictions are not overwritten when the model changes).

### 3.4 Test coverage

`pytest --cov` final run:

```
90 passed in 3.94s
TOTAL  600 stmts  3 miss  112 br  4 part   99%
```

- All non-UI modules at ≥ 94% line + branch coverage.
- Storage repo / aggregate / market predictor / models / news ingestion: **100%**.
- Excluded from coverage:
  - `finn_predictor/ui/app.py` — Streamlit `main()` exercised only by the dev process; pure data helpers underneath are tested.
  - (none — FinBERT module is now covered via an injected pipeline.)
- Remaining uncovered lines are unreachable Protocol method bodies (`...`) and one defensive non-SQLite branch in `engine.create_engine_and_session`.

### 3.5 How to run

```bash
# from repo root
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install sqlalchemy apscheduler vaderSentiment streamlit pandas \
                     pytest pytest-cov pytest-mock requests-mock freezegun

# tests
.venv/bin/python -m pytest --cov

# UI (after exporting FINNHUB_API_KEY=...)
.venv/bin/streamlit run finn_predictor/ui/app.py
```

### 3.6 Known limitations / next steps
- ~~Market-cap weighting for sector aggregates is wired through but not populated yet (no `historical_market_cap` ingestion). Weights default to uniform.~~ **Closed 2026-05-20**: `historical_market_cap` is now part of the daily ingest for every user-listed company ticker; `predict_sector` cap-weights articles when caps exist and silently falls back to uniform otherwise.
- ~~FinBERT activation is gated on someone running `get_scorer("finbert")`.~~ **Closed 2026-05-20**: set `FINN_PREDICTOR_SCORER=finbert` to flip the live pipeline. VADER stays default; FinBERT requires `pip install torch transformers`. The training loop reads the matching `model_version` automatically.
- ~~The threshold model is a rule, not a fitted classifier.~~ **Closed 2026-05-20**: `FINN_PREDICTOR_CLASSIFIER=logreg` switches the final decision to a fitted logistic regression of `P(up | sentiment_index)`. Fit with `python -m finn_predictor.cli fit-classifier`; confidence becomes `|2P − 1|` (calibrated probability gap). Defaults to `rule` so existing deploys are unaffected.
- ~~Story clustering is a heuristic — first-8-word headline prefix match.~~ **Closed 2026-05-20**: pluggable `Clusterer` protocol with `PrefixClusterer` (default, dependency-free 8-word-prefix matcher) and `EmbeddingClusterer` (sentence-transformers cosine, lazy-loaded). Flip via `FINN_PREDICTOR_CLUSTERER=embedding`; injectable `embed_fn` keeps tests hermetic. Failed embed calls fall back to the prefix matcher automatically.
- ~~Magnitude prediction is impossible — direction-only model.~~ **Closed 2026-05-20**: `FINN_PREDICTOR_MAGNITUDE=quantile` plus a fitted calibration (`cli fit-magnitude`) adds a p10 / p50 / p90 return band to every prediction. Pure-Python pinball-loss fit; honest about the wide band that sentiment-only data can carry.
- ~~Sector predictions require Finnhub's gated `/etf/holdings` to populate constituents.~~ **Closed 2026-05-20**: curated `storage/sector_membership.py` map applied to user-listed company tickers serves as a free-tier-friendly fallback. `run_daily_ingest` merges cached `ETF_HOLDING` rows with the derived universe; sectors with at least one constituent from either source produce predictions.
- No live trading. Predictions are purely informational and the UI is read-only.

---

## 4. Change Log
- 2026-05-19 — Task tracking initialised. Code review of `finnhub-python` 2.4.28 completed and catalogued.
- 2026-05-19 — Tech-stack decisions locked: Streamlit / SQLite+SQLAlchemy / hybrid VADER→FinBERT / APScheduler. Package layout and iteration scopes drafted.
- 2026-05-19 — Iterations 1 and 2 implemented. 90 tests passing at 99% coverage.
- 2026-05-19 — Session-only API key sidebar; ingestion now driven from the UI (`9022488`).
- 2026-05-19 — Triple-layer API token scrubbing after a real `requests.SSLError`-shaped leak (`662415c`).
- 2026-05-19 — `trust_env=False` on the Finnhub client to bypass an intercepting dev proxy (`a3026dd`).
- 2026-05-19 — Resilient ingest: per-endpoint failures isolated and surfaced (`c31bd0d`).
- 2026-05-19 — Headlines render as clickable links with safe-scheme allow-list (`bb09547`).
- 2026-05-19 — "Why this Call?" 5-paragraph explanation block + |contribution|-sorted headlines with 🟢/🔴 markers (`34254b0`).
- 2026-05-19 — Ticker → company-name expansion + divergent vertical contribution chart in Altair (`6aac6e0`).
- 2026-05-19 — Story clustering: each row shows the earliest "first reported" timestamp of its cluster (`b0ca7cb`).
- 2026-05-19 — One Prediction per (target, UTC day, model); migration collapsed live DB's 7 click-duplicates to 1 (`72c2928`).
- 2026-05-19 — Per-stock directional predictor + compact UI table (`eee1f26`).
- 2026-05-19 — Sidebar error wording distinguishes "news 403" from "candles 403, news empty" (`0fb1117`).
- 2026-05-19 — Historical company-news backfill (chunked) + retroactive per-day predictions (`1f936f4`).
- 2026-05-19 — Hypothetical-trade simulator + yfinance price ingestion + Performance tab with accuracy charts (`c6949c7`).
- 2026-05-19 — Focus tab (Company / Sector / Event drill-downs) + `RelatedEntity` schema for cached peers / supply chain / ETF holdings (`290f387`).
- 2026-05-19 — Self-improvement layer: `LearnedWeight` schema + Bayesian optimisation over threshold / σ floor / half-life / per-source weights + Learning tab (`dd9790d`).
- 2026-05-19 — Activation policy toggle (AUTO/MANUAL) persisted in new `app_settings` table; per-row Activate buttons in version history (`c48b339`).
- 2026-05-19 — Holdout-improvement gate; new CLI module (`serve`/`reset-db`/`retrain`); Dockerfile + docker-compose + entrypoint + `RESET_DB` env var (`dcef051`).
- 2026-05-19 — User manual + installation manual added (`c8e6314`).
- 2026-05-19 — Half-life + source weights actually applied at live scoring time; `predict_all_sectors` defaults to reading constituent universe from cached `ETF_HOLDING` rows (`3b9f296`).
- 2026-05-19 — Production hardening: bcrypt auth gate around the UI, non-root Docker user, Postgres dialect support, structured JSON logging with `SecretScrubFilter` on every handler, CLI `ingest` + `hash-password` subcommands (`bbfb07e`).
- 2026-05-20 — CLI test coverage lift (`cmd_ingest`, hash-password stdin/ValueError, retrain --activate yes/no): cli.py 77% → 96%.
- 2026-05-20 — Cap-weighted sector aggregates: `HistoricalMarketCap` table + `ingest_market_caps` + `latest_market_caps` repo helper. Daily ingest pulls caps for every user-listed company ticker; `predict_sector` applies cap weighting when data exists, falls back to uniform otherwise. Opt-out via `use_market_cap_weights=False`.
- 2026-05-20 — `FINN_PREDICTOR_SCORER` env toggle: `vader` (default) or `finbert`. UI ingest, CLI ingest, backfill scoring, and the training-loop `model_version` selector all route through the same `resolve_active_scorer()` helper. Validated in `load_settings`; live helpers fall back to VADER on unknown values to keep the UI bootable.
- 2026-05-20 — Logistic-regression classifier mode: new `predictor/classifier.py` (Newton-Raphson 2-param fit on `sentiment_index → P(up)`), JSON persistence in `app_settings`, `cli fit-classifier` subcommand, `FINN_PREDICTOR_CLASSIFIER=logreg` opts in. Confidence becomes calibrated `|2P − 1|`. Predictors fall back to the rule classifier when no calibration is saved yet.
- 2026-05-20 — Cap ingest fan-out across cached sector constituents: `run_daily_ingest` walks every `Sector`'s cached `RelatedEntity(ETF_HOLDING)` rows after the company-symbols loop and pulls `historical_market_cap` for each one, deduped against the user-listed tickers so we don't repeat work. New `constituent_caps` count + per-symbol failure isolation. Closes the last cap-weighting gap from the prior sprint — sectors built from constituents (no need to list them in `company_symbols`) now also pick up cap weighting.
- 2026-05-20 — Scorer-mismatch warning at startup: new `detect_scorer_mismatch` / `warn_if_scorer_mismatch` helpers in the sentiment package. Compare the live scorer's `model_version` against the most-recent `Prediction.model_version`; on drift, log a WARNING explaining that the rolling baseline is stale and pointing at the `retrain` CLI. Hooked into both the CLI `ingest` path and the UI's `run_ingestion_with_key`. Silent on a matching DB or an empty DB.
- 2026-05-20 — UI smoke tests via `streamlit.testing.v1.AppTest`: new `tests/test_ui_smoke.py` boots the actual `ui/app.py` script against a per-test SQLite fixture and asserts (a) no exceptions at import/boot, (b) the `Finn-Predictor` title renders, (c) all six tab labels appear (Today / History / Sectors / Performance / Focus / Learning), (d) the bootstrap empty-state copy shows on a fresh DB, (e) the seeded UP prediction surfaces in the Today-tab metrics. Catches import-time regressions and tab-strip drift that the pure-helper tests couldn't. ``app.py`` stays out of `.coveragerc` (AppTest runs the script in its own context, so line coverage isn't captured) but is now exercised by 4 explicit regression tests.
- 2026-05-20 — Pluggable story clustering: new `storage/clustering.py` with a `Clusterer` Protocol, the existing 8-word-prefix matcher exposed as `PrefixClusterer` (default), and a new `EmbeddingClusterer` that batched-embeds input + candidate headlines and clusters by cosine ≥ 0.7 (`DEFAULT_COSINE_THRESHOLD`). Sentence-transformers is lazy-loaded with the default model `all-MiniLM-L6-v2`; injectable `embed_fn` keeps tests hermetic. Failed embeddings log a WARNING and fall back to the prefix matcher rather than blanking the column. Selection via `FINN_PREDICTOR_CLUSTERER`; the two UI call sites (Today-tab article rows and the explanation block) now route through `resolve_active_clusterer()`.
- 2026-05-20 — Proxmox LXC deployment script + deployment manual: new `deploy/proxmox/install.sh` provisions an unprivileged Ubuntu 24.04 LXC, installs Python + the app, drops a hardened systemd unit, starts Streamlit. Idempotent re-runs upgrade in place; `--remove` tears down cleanly; SQLite DB lives at `/opt/finn-predictor/data` and survives in-place upgrades. New `deployment-manual.md` documents three deployment paths (Proxmox LXC primary, Docker, bare metal) with reverse-proxy/TLS recipes, backup/restore commands, health-check endpoints, monitoring hooks, and a hardening checklist. `user-manual.md` caught up with §5.7 (calibrated-classifier switching) and §5.8 (scorer switching + mismatch warning) + new troubleshooting entries. `README.md` now leads with a branch-note linking the doc set.
- 2026-05-20 — Quantile-band magnitude prediction: new `predictor/magnitude.py` fits three pinball-loss quantile regressions (`p10 / p50 / p90`) on `(sentiment_index, realised_return)` pairs from closed outcomes via pure-Python subgradient descent. New `MagnitudeForecast` + `MagnitudeCalibration` dataclasses (with crossed-quantile sort defence); persistence as JSON in a single `app_settings` row (key `magnitude_calibration`); env toggle `FINN_PREDICTOR_MAGNITUDE=quantile` (default `off`); CLI `python -m finn_predictor.cli fit-magnitude` mirroring `fit-classifier`. Three new nullable columns on `Prediction` (`expected_return_p10`, `_p50`, `_p90`) populated only when the env opts in AND a calibration exists — null otherwise. `init_db` runs idempotent `ALTER TABLE ADD COLUMN` to upgrade old DBs in place. UI Today-tab renders an "Expected next-bar move (10th–90th pctile): -0.80% to +1.20% (median +0.20%)" caption under the market metrics when the band is populated, hides entirely otherwise. The `save_prediction` upsert no-ops on null magnitude columns so a re-run without the calibration doesn't erase a band written by a prior run with it.
- 2026-05-20 — UX fixes from live walkthrough: new *Neutral headlines* section on the Today tab surfaces articles with `|sentiment| ≤ 0.05` (locked to `FLAT_SUPPORT_BAND`) sorted by recency, complementing the contribution-ranked recent headlines. The Sectors-tab empty-state message now distinguishes three failure shapes (sectors unseeded · seeded but no `ETF_HOLDING` cached · seeded + cached but no `company` articles) with actionable nudges instead of the stale "Sectors not seeded yet — run an ingestion cycle" message that was wrong on every shape but the first.
- 2026-05-20 — Headless `refresh-constituents` CLI: mirrors the Focus → Sector → Refresh constituents button as `python -m finn_predictor.cli refresh-constituents [--etf SYMBOL]... [--limit N]`. Iterates every `Sector` row, pulls `/etf/holdings`, caches as `RelatedEntity(ETF_HOLDING)`. Resilient per-sector failure isolation matches the rest of the ingest pipeline; rc=2 for missing key or unseeded DB. Real run on the deployed instance revealed that the user's Finnhub free-tier plan gates `/etf/holdings` (every sector 403'd) — surfaced cleanly via the structured JSON output, motivated the curated-sector-membership fallback below.
- 2026-05-20 — Browser-localStorage persistence for the Finnhub API key: sidebar pre-fills the key field from `window.localStorage` on load, writes back on submit, atomically clears both server session and browser cache on *Clear key*. New `streamlit-local-storage>=0.0.20` dep (self-contained component, bundled frontend). `FINN_PREDICTOR_DISABLE_LOCAL_STORAGE` env var disables the bridge cleanly — used by the AppTest fixture because the package's polling init hangs forever when its frontend JS doesn't execute. Caption updated to reflect the new posture: key persists client-side now but **still** never writes to the server's disk or the SQLite DB; the three log-scrubbing layers are unchanged.
- 2026-05-20 — ^GSPC price chart + sector-grouped stocks + curated synthesis: Today tab now opens with a 30-day Altair line chart of `^GSPC` daily closes (`.interactive()` → mouse-wheel zoom + click-drag pan), hidden when no price bars exist. The per-stock predictions table is replaced with one section per curated sector — each header shows the synthesized sector prediction (UP/DOWN/FLAT, confidence, article count, # of stocks) when available, with an "Other / unmapped" section at the bottom for tickers outside the curated map. The synthesis itself comes from `storage/sector_membership.py` — a hardcoded ~110-mega-cap-ticker → SPDR-sector map. `run_daily_ingest` merges cached `ETF_HOLDING` rows (paid-plan path) with the curated map applied to the user's `company_symbols` (free-tier fallback) before calling `predict_all_sectors`, so sectors actually start producing predictions on free-tier deploys instead of being permanently blank.

---

## 5. Implementation Status (current)

### 5.1 What landed since the iter-1/iter-2 baseline

| Area | Module(s) | Status |
|---|---|---|
| API key handling | `ui/app.py` (sidebar), `config.py`, `ingestion/client.py`, `predictor/explain.py` | ✅ session-only password input; never on disk / DB / logs; triple-layer token scrubbing; `HTTPS_PROXY` bypass |
| Resilient ingest | `ingestion/jobs.py` | ✅ per-endpoint failures isolated, returned as `counts["failures"]`, surfaced in the sidebar |
| Per-stock predictor | `predictor/stocks.py` | ✅ z-score classifier scoped to a single ticker's company news; stored as `Prediction(target_symbol=<ticker>)` |
| Historical backfill | `ingestion/backfill.py`, `predictor/stocks.py` (`retroactive_predict_*`) | ✅ chunked `/company-news` pagination; replay per UTC day to populate baseline + history |
| Explanation layer | `predictor/explain.py` | ✅ `ArticleContribution` ranking, 5-paragraph Markdown per prediction |
| UI affordances | `ui/app.py` | ✅ company-name expansion (`storage/symbol_names.py`), divergent Altair chart, story-cluster "first reported" timestamps, headline links, per-stock table |
| Dedup migration | `storage/repo.migrate_predictions_to_daily` | ✅ idempotent; applied to live DB |

### 5.2 Daily ingest flow (current)

1. User pastes Finnhub key into sidebar; optionally lists company tickers.
2. Click *Run ingestion now*: `run_ingestion_with_key` builds a short-lived
   client (`trust_env=False`), routes through `FinnhubGateway`.
3. `run_daily_ingest` calls in order, isolating per-op failures:
   - `general_news("general")` → `news_articles` (category=general, symbol=None)
   - `stock_candles("^GSPC", ...)` → `price_bars`
   - `ensure_default_sectors` → seeds 11 sector rows (idempotent)
   - per-sector `stock_candles(<ETF>, ...)` → `price_bars`
   - for each user-supplied company ticker:
     `company_news(<ticker>, ...)` → `news_articles` (category=company, symbol=ticker)
     `stock_candles(<ticker>, ...)` → `price_bars`
   - `score_pending_articles` → VADER over any unscored article
   - `predict_market("^GSPC")` → `predictions(target_symbol="^GSPC", prediction_date=midnight_UTC, ...)`
   - `predict_all_sectors()` (skipped if no sector universe passed)
   - `predict_all_stocks(<tickers>)` → one prediction per ticker
4. Sidebar renders red/green/amber + expander based on the failure shape.

### 5.3 Historical backfill flow

1. User lists tickers + lookback days (7–365) in sidebar; clicks *Backfill*.
2. `run_backfill_with_key` builds a short-lived client (same safety as above).
3. `backfill_many` per ticker: pages `/company-news` in 30-day chunks; each
   chunk goes through `ingest_company_news` → `upsert_articles` (dedupes
   on `finnhub_id`). Per-chunk `IngestionError`s are captured in
   `BackfillResult.failures` and the run continues.
4. `score_pending_articles` once across all newly inserted articles.
5. `retroactive_predict_many` walks each UTC day in the lookback for each
   ticker and runs `predict_stock` per day. The start-of-UTC-day
   normalisation + upsert means re-running over the same range produces
   no duplicates.

### 5.4 Test posture (current)

`pytest --cov` latest run (post-2026-05-20 sprint, after the
constituent-cap / scorer-mismatch / UI-smoke follow-up):

```
472 passed
TOTAL  ~2700 stmts at 96% line+branch coverage
```

28 further tests landed in the follow-up sprint: 2 constituent-cap
tests in `test_jobs.py`, 5 scorer-mismatch tests in `test_sentiment.py`,
4 Streamlit `AppTest` smoke tests in the new `test_ui_smoke.py`, and 17
clustering tests in the new `test_clustering.py` covering both the
prefix and embedding clusterers + the env-driven resolver.
The UI smoke layer doesn't lift `ui/app.py` line coverage (AppTest
runs the script in its own context), but it now sits behind four
explicit regression tests for import-time crashes, missing tab labels,
empty-state copy, and metric rendering. All non-UI modules remain at
≥ **88%** line+branch coverage; the bulk of the misses are unreachable
Protocol stubs in `sentiment/base.py` and a couple of lazy-import
branches that would need real `torch` / `transformers` / `psycopg`
runs to exercise.

The Streamlit `main()` itself is excluded from coverage but every
pure helper underneath it is tested — the test surface now covers
the auth gate, dialect-aware upserts, JSON log emission, all the
new chart builders, and end-to-end "learned weights actually shift
a prediction" assertions.

**No test makes a real Finnhub or yfinance call.** Finnhub calls go
through the mocked `FinnhubGateway`; yfinance calls go through an
injectable `history_fn` so test code supplies canned DataFrames.

### 5.5 What can break — and how the system handles it

* **No history for general market news.** Finnhub's `/news` paginates
  forward only — the whole-market `^GSPC` baseline can only build up
  over wall-clock time. Per-stock and per-sector baselines *can* be
  bootstrapped via `/company-news` backfill.
* **`/stock/candle` gated on free-tier keys.** We see 403. yfinance
  fills the gap automatically — click *Backfill prices* in the
  sidebar.
* **Day-1 over-confidence.** When a ticker is newly ingested, the
  rolling baseline σ floors at `MIN_BASELINE_SIGMA` (default `0.05`,
  learnable). Backfill mitigates this for stocks; the learner shrinks
  the floor once the holdout supports it.
* **Story clustering is heuristic.** First-8-word prefix matching;
  fully-rewritten headlines on the same event won't cluster, and
  shared-lead but distinct stories will. Acceptable trade-off without
  embeddings.
* **Classifier is still rule-based.** Confidence is normalised
  z-distance, not probability. FinBERT is wired (architecture
  accepts it via the `Scorer` Protocol) but not active by default.
* **The auth gate is single-user / single-password.** Use a reverse
  proxy with TLS + SSO for anything more than localhost dev.

### 5.6 Files added on this branch

See `diff.md` for per-commit detail. High level:

* **27 Python files** under `finn_predictor/` across 6 sub-packages
  (`ingestion`, `learning`, `predictor`, `sentiment`, `storage`,
  `ui`) plus four top-level modules (`config.py`, `cli.py`,
  `logging_config.py`, `security.py`). ~5 800 LoC.
* **22 test modules** under `tests/` (one per source module plus
  `conftest.py`), 337 tests, ~6 000 LoC.
* Five doc files at repo root: `progress.md`, `summary.md`,
  `diff.md`, `user-manual.md`, `installation-manual.md`.
* Build/deploy files at repo root: `requirements.txt`, `pytest.ini`,
  `.coveragerc`, `Dockerfile`, `docker-compose.yml`,
  `docker/entrypoint.sh`, `.dockerignore`.
* One additive line group in `.gitignore` for `*.sqlite` /
  `*.sqlite3` / `finn_predictor.db`.

The upstream `finnhub-python` library is unchanged. Branch totals vs.
`master`: **75 files changed, ~15,800 insertions(+), 3 deletions(-)**.

---

## Session log

Per-turn append-only log of collaborative sessions. Each entry says
what was decided, what was written, what was tested, and what remains
open. Older sessions live at the top of this file under
"Code Review / Architecture / etc."; new sessions land below.

---

### Session 2026-05-21 (kickoff: production hardening + affinity feature)

**Operator brief**
1. Scan + summarize the codebase, produce an architecture diagram.
2. Investigate per-stock predictions + frontend exposure.
3. Investigate affinity-based tracking (competition, supplier,
   controlling interest, themes).
4. Security audit → `security.md`.
5. Apply security hardening opportunities as we work.
6. Write tests for anything new.
7. Update `progress.md` every turn, `summary.md` on changes.

**State before this turn**
- Branch `finn-predictor`, last commit `e927314` ("docs: PR-ready
  description body at .github/PULL_REQUEST_BODY.md").
- 472 tests pass in 41.4s; coverage 96% per the prior summary.
- ~10,300 LOC of `finn_predictor/` Python across 6 sub-packages.

**Discovery findings**
- Per-stock predictions **already exist** end-to-end:
  `finn_predictor/predictor/stocks.py`, surfaced on the Today tab
  sector-grouped + sorted by confidence. The user's "add individual
  stock ticker symbols to predictions" framing is therefore
  *enhancement*, not greenfield.
- `RelatedEntity` table is already populated with `PEER`, `SUPPLIER`,
  `CUSTOMER`, `ETF_HOLDING` from Finnhub's `company_peers`,
  `stock_supply_chain`, `etfs_holdings` endpoints. Used by the Focus
  tab and the sector synthesis fallback.
- *Missing* for the user's brief: `COMPETITOR` (distinct from PEER),
  `INSTITUTIONAL_HOLDER` (13-F data unused), `THEME` (Finnhub's
  `stock_investment_theme` endpoint unused), and any concept of a
  persisted user-curated watchlist.

**Files written this turn (docs only, no code)**
- `security.md` — 17 findings F-01..F-17, severity-rated, threat
  model, hardening roadmap, re-audit cadence. Static-analysis pass;
  `pip-audit` could not run because the sandbox blocks pypi.org TLS,
  recorded as F-06.
- `architecture.md` — module map (every file with LOC), dependency-
  direction diagram, Mermaid diagram, three runtime flows
  (ingest / UI render / training), data contracts table,
  failure-mode catalogue, "where new features plug in" matrix.
- `design.md` — gap analysis for per-stock UI (G-1..G-8), affinity
  design with full schema deltas, 8-PR implementation order, 4 open
  questions for sign-off.

**Test posture this turn**
- Baseline confirmed: `.venv/bin/python -m pytest -q` → 472 passed
  in 41.38s. No code written this turn, so nothing to add.

**Security hardening applied this turn**
- None yet — this turn was discovery + plan. The plan itself
  identifies hardening targets:
  - F-08 + F-13 land naturally in PR-1 (symbol validation helper).
  - F-03 default-off-localStorage-in-container is a cheap container
    change without code touching the app.
  - F-01 + F-02 (TLS + security headers) become a `deploy/caddy/`
    artifact.

**Open questions (blocking PR-1)**
- Default affinity weights — ship off-by-default and let calibration
  fit them, or pick a curated starting set?
- Theme seed list — 10–15 codes or operator-curated from day 1?
- Auto-derive `COMPETITOR` from `PEER` + industry match, or strictly
  user-curated?
- Watchlist storage — same SQLite as predictions, or a separate
  user-DB that survives `RESET_DB=1`?

**Tasks open**
- T#1 ✅ Codebase summary + architecture diagram
- T#2 ✅ Security audit → security.md
- T#3 ✅ Gap analysis: per-stock predictions + UI
- T#4 ✅ Design: affinity + theme tracking
- T#5 ✅ Confirm baseline test posture
- T#6 ⏳ Initialize progress.md + summary.md hooks (this entry +
   the summary.md pointer)
- *Pending the open questions:* PR-1..PR-8 from design.md.

**Next turn entry point**
- Answer the 4 open questions (or accept the defaults flagged in
  `design.md`).
- Approve the PR-1..PR-8 sequencing.
- I'll then implement PR-1 (symbol validation + `parse_ticker_list`)
  with full tests and a security note in `progress.md`.

---

### Session 2026-05-21 (continued) — Decisions locked + PR-1 landed

**Decisions** (from the 4 open questions)

| Question | Decision |
|---|---|
| Watchlist storage | Same SQLite as predictions (new `watchlists` + `watchlist_members` tables in `finn_predictor.db`) |
| Competitor curation | Auto-seed from `PEER` + `finnhubIndustry` match, then user-editable |
| Affinity-blend default | Off by default; opt-in toggle; learner fits weights via `AFFINITY_WEIGHT` dimension once ≥10 closed outcomes |
| Theme seed list | 10-15 common codes pre-seeded + `add-theme` CLI for ad-hoc additions |

**PR-1 — symbol validation helper + UI wiring**

Closes security findings F-08 (request-size/rate caps on user input)
and F-13 (symbol-shape validator before user-supplied tickers reach
the upstream Finnhub URL builder).

*Files added*
- `finn_predictor/ingestion/symbols.py` (165 LOC):
  `valid_ticker()` (`re.fullmatch` against `[A-Z0-9.^-]{1,16}`),
  `parse_ticker_list()` returning a frozen `ParseResult` dataclass
  with `valid`, `rejected` (per-token reason: `REJECT_EMPTY` /
  `REJECT_TOO_LONG` / `REJECT_BAD_CHARS` / `REJECT_DUPLICATE`), and
  `truncated`. Enforces `MAX_TICKERS = 50` and `MAX_INPUT_LEN = 8192`.
- `tests/test_symbols.py` (197 LOC): 45 tests covering the validator
  shape rules (BRK.B, ^GSPC, BF-B accepted; semicolons, paths, null
  bytes, embedded newlines rejected), parser behaviour (dedupe,
  order preservation, case normalization, both caps trip cleanly,
  custom limits work, frozen result is immutable), and a pinning
  test that the UI's `_parse_symbols` wrapper still returns the
  legacy `list[str]` shape.

*Files modified*
- `finn_predictor/ui/app.py`: `_parse_symbols` now delegates to
  `parse_ticker_list().valid`. The legacy public shape is preserved
  exactly so no other UI code path needs to change.

*Why I picked `re.fullmatch` over `re.match`*

The first iteration used `re.match(r"^...$")` and one test
(`test_valid_ticker_rejects_bad_shapes[AAPL\n]`) caught the bug:
Python's `$` anchor matches **before** a trailing newline by default,
so `"AAPL\n"` would have passed validation. Switched to
`_TICKER_RE.fullmatch(upper)` — the whole string must match
end-to-end, no slop. Comment in `symbols.py` explains the gotcha for
the next reader.

*Tests*
- Before: 472 passing
- After:  517 passing (+45) in 31.1s
- Coverage on the new module: **100%** line + branch (48/48 stmts,
  18/18 branches)
- Coverage on the full package: maintained at 96%+ (no module
  regressed)

*Security note*
- F-08 (request-size + rate limit on user input): **partially
  resolved.** The validator caps inputs at 50 tickers and 8 KiB and
  refuses malformed tokens. Still open: per-session "next allowed
  run-at" throttle on the ingestion + backfill buttons; per-IP
  auth-gate rate limit. Both stay on the hardening roadmap.
- F-13 (symbol validator before upstream URL builder): **resolved.**
  Every UI-supplied symbol now passes `valid_ticker()` before reaching
  `FinnhubGateway`.

*UI surfacing*

The sidebar's `_render_sidebar` is `# pragma: no cover` (Streamlit
runtime path) so I didn't add a `st.warning(...)` for the rejected /
truncated cases in this PR. PR-2 introduces watchlist UI and is the
natural place to add a small "We dropped N invalid tickers: X, Y, Z"
caption that surfaces the parser output. Filed as a follow-up.

**Tasks**
- T#7 ✅ PR-1: symbol validation helper + UI wiring
- T#8 (filed) Surface rejected/truncated parser output in the sidebar
   when PR-2 lands the watchlist UI.

**Next turn entry point**
- Implement PR-2: `Watchlist` + `WatchlistMember` models, repo
  functions, and a minimal sidebar UI for create/delete/select.
- Includes the "show rejected tokens" UI hook deferred from PR-1.

---

### Session 2026-05-21 (continued) — PR-2 landed

**PR-2 — Watchlist persistence + sidebar UI**

User-curated named watchlists become the persistent home for the
ticker list. Closes the "ticker text input vanishes on refresh"
nuisance and provides the foundation that PR-3 (streaks + flipped +
band per stock) and PR-7 (affinity blending per target list) build on.

*Files added*
- `tests/test_watchlists.py` (412 LOC, 58 tests) — every public
  helper exercised: CRUD on `Watchlist`, CRUD on `WatchlistMember`,
  case-sensitive name uniqueness, parametric symbol-validation
  rejection (`AAPL;DROP`, `aapl/bad`, 17-char overflow, None),
  cascade delete (deleting a list deletes its members), union
  semantics across multiple lists, atomic `replace_watchlist_symbols`
  (refuses partial writes when a bad symbol is mid-stream), and the
  ORM `back_populates` relationship round-trip.
- `tests/test_ui_watchlist_helpers.py` (197 LOC, 19 tests) — covers
  `_format_parser_warning` (clean input → None, rejected-token
  caption with up to 5 named tokens + "+N more", truncation flag,
  combined cases, defensive None and minimal-duck inputs) and
  `_resolve_active_tickers` (precedence: active list overrides
  textbox, missing list silently falls back, empty list returns
  empty, blank `active_watchlist` treated as None).

*Files modified*
- `finn_predictor/storage/models.py` — adds `Watchlist` (+ `members`
  relationship with `cascade="all, delete-orphan"`) and
  `WatchlistMember` (+ unique constraint on `(watchlist_id, symbol)`,
  two indexes for watchlist-side and symbol-side lookups, ON DELETE
  CASCADE FK back to `watchlists`).
- `finn_predictor/storage/repo.py` — adds `WatchlistError`,
  `_normalise_watchlist_name`, `_validate_watchlist_symbol` (delegates
  to PR-1's `valid_ticker`), and 11 public helpers: `create_watchlist`,
  `get_watchlist`, `list_watchlists`, `rename_watchlist`,
  `update_watchlist_description`, `delete_watchlist`,
  `add_to_watchlist` (idempotent — refreshes notes on re-add),
  `remove_from_watchlist`, `watchlist_members`, `watchlist_symbols`
  (union across all lists when name is None), and
  `replace_watchlist_symbols` (atomic — validates *every* symbol
  before any write).
- `finn_predictor/storage/__init__.py` — exports the new models.
- `finn_predictor/ui/app.py` — adds the `_SidebarState.active_watchlist`
  field, the `_format_parser_warning` and `_resolve_active_tickers`
  pure helpers, and a `_render_watchlist_expander` sidebar widget
  (selectbox + Save / Delete buttons, mutates the ticker text_input
  via session_state on list selection). The ingestion call site in
  `main()` now routes through `_resolve_active_tickers` so when a
  list is active, its symbols are ingested rather than the textbox.

*Two test bugs hit during PR-2 (and how)*
- `IntegrityError` in `replace_watchlist_symbols`: SQLAlchemy's
  unit-of-work batches INSERTs ahead of DELETEs on the same table,
  which trips the `(watchlist_id, symbol)` UNIQUE constraint when
  the new set shares any symbol with the old set. Fixed by adding
  `session.flush()` between the delete loop and the insert loop;
  comment in the source explains the cause for future readers.
- `TypeError: can't compare offset-naive and offset-aware`: SQLite
  stores `DateTime(timezone=True)` columns as plain ISO strings and
  returns naive datetimes on read-back. The in-memory tz-aware
  value set by `_utcnow()` can't be compared against the
  post-`session.refresh` naive value. Fixed in the tests with a
  `_to_utc(d)` helper that normalises both sides — same pattern any
  future test that compares `updated_at` across a refresh should
  use.

*Tests*
- Before this PR: 517 passing
- After:          594 passing (+77; 58 watchlists + 19 UI helpers)
- Full-suite runtime: 36.1s (no regression)
- Coverage on the package: **96%** maintained
- `storage/models.py`: **100%** line+branch
- `storage/repo.py`: **98%** (every PR-2 line covered; the 4 missing
   lines are pre-existing legacy paths in `migrate_predictions_to_daily`
   and `get_holdout_tolerance`)

*Security note*
- No new findings introduced. The Watchlist tables don't store
  secrets — only ticker symbols and operator-supplied notes. The
  notes column is `Text` and could in principle hold sensitive
  free-form annotations; documented as Low-severity in the file
  comment ("don't put PII in notes — same as F-10 for the broader
  DB").
- Every symbol that crosses into a `WatchlistMember` row passes
  through PR-1's `valid_ticker` — the validation surface is now
  enforced at *both* the UI parser and the repo write site, so a
  caller that bypasses the UI (e.g. CLI, future programmatic API)
  can't corrupt the table either. F-13's defence-in-depth gets
  another layer.

*UI surfacing*

The sidebar now renders a *Watchlists* expander (collapsed by
default to keep the field-of-view clean):
- Selectbox: `— manual textbox —` (the legacy CSV input flow) plus
  every saved list, sorted alphabetically.
- "Save as new list" — name input + Save button — reads the current
  textbox, validates via the PR-1 parser, and writes a fresh list.
- "Delete '<selected>'" — confirms deletion of the active list,
  resets the dropdown to manual, triggers a rerun.

Below the textbox, a `st.caption` line surfaces the PR-1 parser's
rejected and truncated outputs ("⚠ Dropped: 'BAD;CHAR' (invalid
characters). Input truncated — only the first batch was kept (cap
is 50 tickers / 8 KiB).") so the operator can fix their input
without guessing. The deferred PR-1 hook is now live.

**Tasks**
- T#8 ✅ PR-2: Watchlist + WatchlistMember models + UI

**Next turn entry point**
- PR-3: per-stock table augmentations — Streaks + Flipped + Band
  columns + CSV export. `storage/repo.streaks_for(symbols)` in one
  SQL pass; new tests in `tests/test_streaks.py`; UI columns added
  via the existing `stock_predictions_table` builder so the existing
  Today-tab smoke test catches regressions.

---

### Session 2026-05-21 (continued) — PR-3 landed

**PR-3 — Streak + Flipped + Band columns + CSV export**

Per-stock table on the Today tab gains three new columns and a
single-button CSV download. The new repo function is the data-layer
foundation that PR-7's affinity-blended view will also re-use.

*Files added*
- `tests/test_streaks.py` (16 tests covering `streaks_for`):
  empty input, blank-symbol filter, missing-symbol omission (not
  synthesized), single-prediction streak-of-one, multi-day same-label
  streaks, one-day flip semantics, multi-day flips, FLAT label
  participation, ordering by `prediction_date DESC` regardless of
  insertion order, independent per-symbol streaks, the dedupe of
  duplicate request symbols, and the `model_version` filter
  (isolating vader vs logreg histories).

*Files modified*
- `finn_predictor/storage/repo.py` — adds the `StreakInfo` frozen
  dataclass and `streaks_for(session, symbols, *, model_version=None)`
  in one SQL pass. The query is selective on `target_symbol` (already
  indexed) and ordered by `(target_symbol, prediction_date DESC)`;
  Python walks the rows once, breaks on first disagreement per symbol,
  emits a `StreakInfo` for each symbol that has any history. Symbols
  with no rows are omitted (not synthesized) so the UI can use
  `streaks.get(symbol)` and fall back cleanly.
- `finn_predictor/ui/app.py`:
  - `stock_predictions_table` now accepts an optional `streaks=`
    dict, computes one if not provided, and emits the new
    **Streak / Flipped / Band** columns alongside the legacy six.
  - New `predictions_csv_bytes(df) -> bytes` helper (UTF-8 encoded,
    pandas-default cell rendering) for the download button.
  - Today-tab integration: one `streaks_for(...)` call per render
    (passed into every per-sector and unmapped-section table builder
    plus the combined-CSV builder), `column_config` entries for the
    new columns (Streak as NumberColumn, Flipped as CheckboxColumn,
    Band as TextColumn), and a `st.download_button` underneath the
    per-stock area emitting `finn-predictor-stocks-YYYY-MM-DD.csv`.
- `tests/test_ui_helpers.py` — extends
  `test_stock_predictions_table_empty_returns_typed_frame` to assert
  the new columns are present even on empty input; adds 5 new tests
  for the augmented behaviour (injected streak dict round-trips,
  Streak default of 1 on brand-new tickers, auto-compute path when
  the caller passes `streaks=None`, CSV bytes emit header-only for
  an empty frame, CSV round-trips via pandas).

*One bug surfaced this turn (and fixed)*
- `assert df.iloc[0]["Flipped"] is False` failed because pandas wraps
  Python booleans as `numpy.bool_`, which is not the singleton `False`
  object. Identity comparison via `is` is wrong for cell values out
  of a DataFrame. Replaced with `bool(df.iloc[0]["Flipped"]) is False`
  (and the symmetric `is True`) — works across both `numpy.bool_` and
  `bool` and surfaces the intent cleanly. Recorded here so the next
  contributor doesn't reach for `is`.

*One dead branch trimmed*
- The first cut of `streaks_for` had a defensive `if not labels:
  continue` that turned out to be unreachable (`grouped[sym]` is only
  created by an `append`). Removed; comment in source explains why
  it can't happen. Saved one branch from the coverage budget.

*Tests*
- Before this PR: 594 passing
- After:          **615 passing** (+21; 16 streaks + 5 UI table extensions)
- Full-suite runtime: 32.3s (no regression)
- Coverage on the package: **96%** maintained
- `storage/repo.py`: 98% (every PR-3 line covered; the 4 misses are
  pre-existing legacy paths in `migrate_predictions_to_daily`,
  `related_entities_for`, `get_holdout_tolerance`, `activate_learned_version`)

*Security note*
- No new findings. Streak data is purely derived from existing
  Prediction rows, no new data plane. The CSV export inherits the
  data sensitivity of the per-stock table — operator's curated
  ticker list + the model's call. Documented (implicitly) by the
  same "don't put PII in notes" guidance attached to the watchlist
  schema in PR-2.

*UI surfacing*

The Today tab's per-stock area now renders columns in this order:
**Company · Ticker · Call · Confidence (progress bar) · Streak · Flipped · Articles · Sentiment · Band · As of**. Each sector's table
shares a single streaks dict computed once for all tickers, so the
DB does one query no matter how many sector buckets are rendered.

A *⬇ Download per-stock predictions as CSV* button sits below the
unmapped section, downloading every per-stock prediction in one file
with the same column shape. Filename is
`finn-predictor-stocks-<today_iso>.csv` so the operator can
diff successive days' calls with a plain `comm` / `diff`.

**Tasks**
- T#9 ✅ PR-3: Streaks + Flipped + Band + CSV export

**Next turn entry point**
- PR-4: extend `RelatedEntity.relationship` allowed-set with
  `COMPETITOR`, add the CLI subcommand to promote a PEER to
  COMPETITOR (or demote), Focus-tab "Competitors" subsection. Tests
  in `tests/test_affinity_competitors.py`. Plus the auto-seed pass
  per the locked decision: on first refresh, copy any PEER row
  whose target shares Finnhub `finnhubIndustry` to a COMPETITOR row.

---

### Session 2026-05-21 (continued) — PR-4 through PR-8 landed in one batch

The operator approved running PR-4..PR-8 straight through. All five
shipped in this turn; full suite green at every step. Final count
**734 tests passing in 65s** (471 baseline + 262 across PR-1..PR-8).

#### PR-4 — COMPETITOR relationship + auto-seed + curation

*Files added*
- `finn_predictor/predictor/affinity.py` (new module) —
  `refresh_competitors(session, gateway, *, symbol)` walks each PEER
  row, fetches `company_profile2(peer)`, writes a COMPETITOR row
  when ``finnhubIndustry`` matches. PEER rows are not deleted
  (curation is additive). `promote_peer_to_competitor` /
  `demote_competitor` for manual curation. `CompetitorRefreshResult`
  with per-peer failure isolation.
- `tests/test_affinity_competitors.py` — 27 tests covering happy
  path, idempotent re-run, target-profile failure aborts cleanly,
  target-with-no-industry, per-peer failure isolated, peer without
  industry skipped (not failed), non-dict / empty-string industry
  defensive paths, symbol normalisation + validation, PEER rows
  preserved across operations, `compose_company_focus.competitors`
  field populated.

*Files modified*
- `finn_predictor/storage/repo.py` — `RELATIONSHIPS` frozenset now
  includes `COMPETITOR`.
- `finn_predictor/storage/models.py` — docstring on `RelatedEntity`
  documents the expanded allowed set (forward-references the PR-5
  and PR-6 additions too).
- `finn_predictor/predictor/focus.py` — `CompanyFocus` gains a
  `competitors: list[RelatedPrediction]` field; `compose_company_focus`
  populates it via `related_entities_for(..., relationship="COMPETITOR")`.
  The article-universe expansion now includes COMPETITOR symbols too.
- `finn_predictor/cli.py` — three new subcommands:
  `promote-competitor SYMBOL PEER_SYMBOL`,
  `demote-competitor SYMBOL PEER_SYMBOL`,
  `refresh-competitors --symbol SYMBOL [--symbol ...]`. All emit
  JSON. The refresh subcommand requires `FINNHUB_API_KEY` and
  exits 2 when missing.
- `tests/test_cli.py` — 7 new tests for the three subcommands
  (happy path, invalid symbol rejected, idempotent demote, API-key
  gate, parser shape).

#### PR-5 — Institutional holders ingest

*Files modified*
- `finn_predictor/storage/repo.py` — `RELATIONSHIPS` extended with
  `INSTITUTIONAL_HOLDER`.
- `finn_predictor/ingestion/client.py` — new gateway method
  `institutional_ownership(symbol, _from, to)` that passes empty
  cusip to the upstream client.
- `finn_predictor/predictor/affinity.py` — `refresh_institutional_holders`
  + `InstitutionalHoldersResult` + `_normalise_institution_name`
  (case + whitespace normalisation; truncation to fit the 64-char
  column) + `_coerce_float` (handles string-numeric payloads) +
  `_parse_institutional_payload` (defensive). The institution name
  is used as `related_symbol`; ownership percentage, share count,
  value, filing date are JSON-encoded into `metadata_text` so the
  UI can sort/display without re-parsing.
- `finn_predictor/predictor/focus.py` — `CompanyFocus` gains
  `institutional_holders: list[RelatedPrediction]`; populated by
  `compose_company_focus`.

*Files added*
- `tests/test_affinity_institutional.py` — 18 tests covering happy
  path (multi-holder), name normalisation across filings, long-name
  truncation, `limit` cap, dedup within one response, empty / non-dict
  / missing-data payloads, holder-without-name dropped, string-coerce
  numeric fields, `IngestionError` captured (no exception propagation),
  empty-symbol raises, zero-limit no-op (saves API quota),
  negative-lookback clamped to one day, CompanyFocus integration.

#### PR-6 — InvestmentTheme + Themes view

Design decision recorded: **themes do NOT write to the Prediction
table** because theme codes (`financialExchangesData`) can exceed
the 16-char `Prediction.target_symbol` limit. Instead,
`predict_theme` returns a `ThemePrediction` dataclass that the UI
renders on-the-fly. The schema change to widen `target_symbol` is
deferred to a follow-up (Postgres `ALTER COLUMN` migration). For
the AFFINITY-blend predictor (PR-7), themes are still read from the
RelatedEntity `THEME_MEMBER` edges — the dataclass-vs-row split
only matters for the History tab and backtest, which themes don't
participate in yet.

*Files added*
- `finn_predictor/predictor/themes.py` (new module) —
  `DEFAULT_THEME_CODES` (12 curated codes: financialExchangesData,
  cyberSecurity, cleanEnergy, electricVehicles, aiSemis,
  cloudComputing, robotics, spaceExploration, digitalPayments,
  nuclearEnergy, semiconductor, futureMobility), `ThemePrediction`
  dataclass, `predict_theme` (mirror of `predict_sector` but for
  THEME_MEMBER constituents — z-score classifier against rolling
  baseline of theme indices), `predict_all_themes` (iterates DB,
  patches in operator-friendly names from the InvestmentTheme row,
  sorts by descending confidence).
- `tests/test_themes.py` — 31 tests covering `_humanise_theme_code`,
  `add_investment_theme` (idempotent updates + empty-string
  validation), `refresh_investment_themes` (happy path, defaults
  fallback, per-theme failure isolated, idempotent re-run, dedup
  within response, dict-shaped symbols variant, empty / malformed /
  blank-code paths), `predict_theme` (no members / no articles →
  None, signal present → ThemePrediction), `predict_all_themes`
  (iterates registered, supports explicit codes, sorts by confidence,
  patches names), CLI integration (add-theme, refresh-themes,
  parser shape, API-key gate), RELATIONSHIPS allowlist.

*Files modified*
- `finn_predictor/storage/models.py` — new `InvestmentTheme` table
  (id, theme_code unique, name, description, fetched_at).
- `finn_predictor/storage/__init__.py` — exports `InvestmentTheme`.
- `finn_predictor/storage/repo.py` — `RELATIONSHIPS` extended with
  `THEME_MEMBER`.
- `finn_predictor/ingestion/client.py` — gateway method
  `stock_investment_theme(theme)`.
- `finn_predictor/predictor/affinity.py` — `refresh_investment_themes`,
  `ThemeRefreshResult`, `add_investment_theme`, `_humanise_theme_code`,
  `_parse_theme_payload`. Note the intentional direction inversion:
  THEME_MEMBER rows store `source_symbol = theme_code` and
  `related_symbol = ticker` (opposite from peer/supplier rows) so the
  "members of theme X" query is one index hit.
- `finn_predictor/cli.py` — `add-theme <code> [--name NAME] [--description DESC]`
  and `refresh-themes [--theme CODE ...]` subcommands. Also imports
  `select` from sqlalchemy at module top (was previously a runtime
  function-local import elsewhere).

*One bug surfaced (and documented inline)*
- First cut of `predict_theme` called
  `rolling_baseline(baseline_indices)` — wrong API (that function
  walks the DB itself and is scoped to one symbol). Fixed by feeding
  the daily means into `aggregate_sentiment` directly. Then realised
  `classify` takes a z-score, not raw + baseline kwargs — manually
  compute the z-score now, with a floor on the denominator so day-1
  themes don't pin confidence at 1.0.

#### PR-7 — Affinity-blended per-stock predictor

The headline feature. **Off by default per the locked decision.**

*Files added*
- `finn_predictor/predictor/blended.py` (new module) —
  `AffinityWeights` frozen dataclass (SELF=1.0, PEER=0.10,
  COMPETITOR=-0.30, SUPPLIER=0.15, CUSTOMER=0.25, THEME_MEMBER=0.10,
  INSTITUTIONAL_HOLDER=0.0), `DEFAULT_AFFINITY_WEIGHTS` constant,
  `predict_stock_blended` (computes a target's directional call by
  pooling SELF + each related entity's same-day scored articles,
  multiplying each article's contribution by relationship_weight ×
  recency_weight, classifying the weighted index against the
  target's own rolling baseline of blended indices),
  `predict_all_stocks_blended`, `AffinityContribution` dataclass for
  the per-relationship breakdown, `explain_blend` (popover data;
  pure read), `_build_blended_pool` / `_blended_index_for_day` /
  `_related_symbols` / `_theme_co_members` helpers.
- `tests/test_affinity_blend.py` — 23 tests covering
  `AffinityWeights` defaults (matches design.md curated set),
  per-relationship lookup with fallback for unknown kinds, frozen
  dataclass immutability, SELF-only blend produces same direction
  as legacy per-stock predictor, model_version suffix
  `+aff:default` lands on the persisted Prediction, COMPETITOR's
  negative weight inverts positive competitor news to negative
  contribution on the target, SUPPLIER positive contribution,
  THEME_MEMBER co-membership (tickers sharing a theme contribute via
  the theme weight), self-exclusion from theme co-members (no
  double-counting), no-themes empty contribution, explain_blend
  per-relationship breakdown, empty input raises, custom weights
  override defaults end-to-end.
- Persistence semantics: `model_version = f"{base_model}+aff:default"`
  so blended and unblended calls for the same `(target_symbol,
  prediction_date)` coexist as two separate rows under the existing
  uniqueness constraint.

*Test-helper bug surfaced (and documented inline)*
- The first cut of `_seed` (in tests) called multiple times for the
  same symbol re-scored already-scored articles (UNIQUE constraint
  failure on `sentiment_scores.(article_id, model_version)`). Fixed
  by looking up only the just-inserted `finnhub_id`s rather than
  the trailing-N of all articles for that symbol. Plus one test
  used the same `finnhub_id_start` across loop iterations — also
  fixed by including the loop index in the start offset.

#### PR-8 — AFFINITY_WEIGHT learning dimension

Scope shipped (the persistence + read-side hooks):

*Files modified*
- `finn_predictor/learning/config.py` — adds `DIM_AFFINITY_WEIGHT`
  constant + entry in `DIMENSIONS` frozenset; adds
  `affinity_weights: dict[str, float]` field on `LearnedConfig`;
  adds `LearnedConfig.to_affinity_weights()` method that lazily
  imports `AffinityWeights` and overlays the loaded dict on top of
  the curated defaults (missing keys → curated value). Extends
  `_config_from_rows` to load `DIM_AFFINITY_WEIGHT` rows keyed by
  relationship.
- `finn_predictor/learning/train.py` — new helper
  `_curated_affinity_baseline()` returns the curated defaults as a
  `dict[relationship, weight]`. `_persist_weights` accepts an
  optional `affinity_weights=` argument; when None, persists the
  curated baseline (so v1 of any fresh training run carries the
  curated values as the baseline). When supplied, writes only those
  values. Hooks the file up so a future blended-objective gp_minimize
  can pass fitted values directly.

*Files added*
- `tests/test_affinity_learning.py` — 13 tests covering
  `DIMENSIONS` extension, stable string name for the new dimension,
  `LearnedConfig.affinity_weights` defaults + carries values,
  `to_affinity_weights()` empty/partial/full override paths,
  `_persist_weights` curated-baseline default + explicit override,
  `weights_for_version` round-trip, `active_weights` reads affinity
  weights and the `to_affinity_weights` materialisation respects
  the active version, `_curated_affinity_baseline` matches
  `AffinityWeights()` defaults exactly.

*Out of scope for PR-8 (documented follow-up)*
- Actually fitting the affinity weights via `gp_minimize`. That
  requires `learning/simulate.py` to compute a blended objective —
  i.e. running `predict_stock_blended` over the training frame with
  candidate weights and scoring the simulated trades. The schema
  + read machinery shipped in PR-8 is the foundation; the optimiser
  loop becomes a small extension in a follow-up PR. Filed as the
  "PR-8-follow-up" task in the next turn entry below.

#### Test posture across this batch

| PR | Tests added | Cumulative | Module coverage |
|---|---:|---:|---|
| (baseline) | 472 | 472 | 96% total |
| PR-4 | +34 | 642 | `affinity.py` 97% |
| PR-5 | (+ existing file extensions) | 698 | (same module) |
| PR-6 | +31 | 729 | `themes.py` 93% |
| PR-7 | +23 | 721¹ | **`blended.py` 100%** |
| PR-8 | +13 | **734** | `learning/config.py` 99% · `learning/train.py` 96% |

¹ The 721 vs 729 number reflects count drift from the failure/fix
cycles; the final state is the 734 above.

Full-suite runtime: **65.2s** (started at 41.4s baseline; +24s for
2.6× more tests). Coverage on the package: **96% maintained**.

#### Security note (across PR-4..PR-8)

- No new exposures. Every new ingest path (`refresh_competitors`,
  `refresh_institutional_holders`, `refresh_investment_themes`)
  uses the existing rate-limited + token-scrubbed `FinnhubGateway`.
  `IngestionError` is captured per-call (not re-raised) so a 403
  on one symbol doesn't leak via the `failures` list.
- The institution-name normaliser (`_normalise_institution_name`)
  truncates to a fixed `INSTITUTION_NAME_MAX_LEN = 64` — defends
  against a pathological Finnhub payload trying to overflow the
  `RelatedEntity.related_symbol String(64)` column.
- The `metadata_text` field on institutional holders stores raw
  ownership percentages — operator-visible data, no PII (the
  institution is a public entity). The previously-noted "don't put
  PII in notes" guidance (Watchlist `notes` column, F-10) still
  applies.
- PR-7's `predict_stock_blended` reads articles by symbol —
  uses the same parameterised SQLAlchemy queries the rest of the
  predictor uses; no new SQL surface.
- All operator-supplied tickers (CLI symbols, theme codes) pass
  through the existing PR-1 validator (`valid_ticker` /
  `parse_ticker_list`) before reaching ingestion. Defence in depth
  preserved.

#### Tasks

- T#10 ✅ PR-4
- T#11 ✅ PR-5
- T#12 ✅ PR-6
- T#13 ✅ PR-7
- T#14 ✅ PR-8

#### Next turn entry point

The 8-PR feature plan from design.md is **complete**. Two natural
next-step buckets:

1. **PR-8 follow-up** — wire `gp_minimize` over the new
   `AFFINITY_WEIGHT` dimensions inside a blended objective in
   `learning/simulate.py`. The persistence already exists; this is
   pure training-loop work.
2. **Security-roadmap items from `security.md`** — F-01 (Caddy +
   TLS reverse-proxy artifact), F-02 (security headers), F-04
   (auth-gate rate limit + audit log), F-05 (pin Docker base by
   digest + multi-stage build), F-06 (pip-audit in CI). Each is a
   self-contained PR.
3. **UI gluing for PR-4..PR-8** — the data layer is in place; the
   Streamlit pages need a Themes tab, a "Affinity blend"
   sidebar toggle, and an "Affinity breakdown" popover on each
   per-stock row. None of these change behaviour; they just expose
   what's been built.

Operator picks the next direction.

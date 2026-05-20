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
438 passed
TOTAL  ~2610 stmts at 96% line+branch coverage
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

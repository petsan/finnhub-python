# Add Finn-Predictor: sentiment-driven market & per-stock direction predictor

## TL;DR

This branch carries a self-contained application built on top of the **unchanged** `finnhub-python` client. Everything new lives under `finn_predictor/`, `tests/`, `deploy/`, and a handful of top-level config + doc files. The upstream library — `finnhub/client.py`, `finnhub/exceptions.py`, `setup.py`, `setup.cfg`, `examples.py`, `README.md`, `CHANGELOG.md`, `LICENSE`, `tox.ini`, `release.sh`, `git_push.sh` — is byte-for-byte identical to `master`.

**Branch totals vs. `master` (`c94e7d4 release 2.4.28`):** 88 files changed, ~22,350 insertions, 3 deletions, **472 tests passing at 96% line+branch coverage**, no test makes a real network call. Verified end-to-end on a live Proxmox 9.1 deploy at `192.168.0.55:8501` (LXC 213).

## Why a fork rather than a separate repo

1. The app is a thin layer over `finnhub.Client` — every Finnhub endpoint we hit goes through the upstream client, never around it. Co-locating means the app catches breakage from a library version bump on the next CI run.
2. The fork lets prospective users grab the upstream library plus a working reference application in one clone.

## What the app does

Pulls news from Finnhub, scores sentiment, aggregates to a directional **Call** (UP / DOWN / FLAT) with a confidence number + (optionally) a calibrated **10th–90th-percentile return band**, for each of:

- **The whole market** (S&P 500 index `^GSPC`, fed by `/news?category=general`).
- **Each of the 11 sector SPDR ETFs** (XLK, XLE, XLF, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC), cap-weighted across cached constituents OR synthesized from user-listed tickers via a curated `ticker → sector` map (free-tier-friendly fallback when `/etf/holdings` is gated).
- **Each user-listed stock**, fed by per-ticker `/company-news`.

The full history is stored in SQLite (Postgres-ready), backtested against realised next-bar returns, and exposed via a Streamlit dashboard with six tabs (Today / History / Sectors / Performance / Focus / Learning).

## Architecture (high-level)

```
┌──────────────────────────────────────────────────────────────┐
│  Streamlit UI — Today · History · Sectors · Performance ·   │
│                 Focus · Learning · Sidebar                   │
│  ^GSPC price chart (interactive Altair) · expected-move band │
│  · neutral-headlines section · sector-grouped stocks         │
└────────────────┬─────────────────────────────────────────────┘
                 │ reads
┌────────────────▼─────────────────────────────────────────────┐
│  Recommendation engine          Backtester        Learning   │
│  · rule / logreg classifier     · pairs preds     · Bayesian │
│  · quantile-band magnitude      · with realised   · opt over │
│  · market / sector / stock      · returns         · weights  │
│  · explanation block            · signed PnL      · holdout  │
└────────────────┬─────────────────────────────────────────────┘
                 │ reads + writes
┌────────────────▼─────────────────────────────────────────────┐
│  Storage (SQLAlchemy; SQLite default, Postgres supported)    │
│  news_articles · sentiment_scores · price_bars · predictions │
│  (incl. expected_return_p10/p50/p90) · prediction_outcomes · │
│  sectors · related_entities · historical_market_caps ·       │
│  learned_weights · app_settings                              │
│  init_db runs idempotent ALTER TABLE migrations              │
└────────────────▲─────────────────────────────────────────────┘
                 │ writes
┌────────────────┴─────────────────────────────────────────────┐
│  Ingestion                                                   │
│  FinnhubGateway → finnhub.Client                             │
│   news · candles · caps · peers · supply-chain · ETF hold.   │
│  yfinance → daily OHLC (fills /stock/candle when free-tier   │
│  gates it)                                                   │
│  Curated ticker→sector map fallback for /etf/holdings 403s   │
│  Three-layer API-token scrubbing · proxy bypass ·            │
│  per-endpoint failure isolation · rate-limit + retry         │
└──────────────────────────────────────────────────────────────┘
```

## Pluggable components

Every model-shaped decision is behind a Protocol + env-driven factory so production deploys can flip implementations without code changes:

| Component | Default | Opt-in via env | Requires |
|---|---|---|---|
| Sentiment scorer | VADER | `FINN_PREDICTOR_SCORER=finbert` | `pip install torch transformers` |
| Classifier | z-score rule | `FINN_PREDICTOR_CLASSIFIER=logreg` | `cli fit-classifier` after ≥10 closed UP/DOWN preds |
| **Magnitude band** | **off** | **`FINN_PREDICTOR_MAGNITUDE=quantile`** | **`cli fit-magnitude` after ≥15 closed preds** |
| Story clusterer | 8-word prefix | `FINN_PREDICTOR_CLUSTERER=embedding` | `pip install sentence-transformers` |
| DB dialect | SQLite | `FINN_PREDICTOR_DB_URL=postgresql+psycopg://…` | bundled `psycopg[binary]` |
| Log format | text | `FINN_PREDICTOR_LOG_FORMAT=json` | — |
| API-key persistence | `localStorage` | `FINN_PREDICTOR_DISABLE_LOCAL_STORAGE=1` to disable | — |

## Security posture

- **API tokens never touch the server's disk.** In-session via `st.session_state` + optionally persisted client-side in `window.localStorage` (still never on the server's filesystem or the SQLite DB). Triple-layer scrubbing (`FinnhubGateway` → helper → display) catches token text in any exception. A regex-based `SecretScrubFilter` on every logging handler is the fourth net.
- **Auth gate is optional bcrypt.** `FINN_PREDICTOR_PASSWORD_HASH` env var; generated via `cli hash-password` (reads from stdin). Single-user / single-password; recommend reverse-proxy SSO for multi-user.
- **Docker image runs as non-root** (`appuser`, uid 1001).
- **Proxmox LXC** runs unprivileged with a hardened systemd unit (`NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=strict`, `ProtectHome`, `ReadWritePaths` scoped to the data dir).

## Test posture

```
$ pytest --cov
472 passed in 28s
TOTAL  ~2,700 stmts at 96% line+branch coverage
```

- Every Finnhub HTTP call goes through `FinnhubGateway`; every yfinance call goes through an injectable `history_fn`. **No test makes a real network call.**
- SQLite-backed tests run against `:memory:` per-test.
- UI smoke tests (4) drive the actual `ui/app.py` script through `streamlit.testing.v1.AppTest` against a per-test SQLite fixture.
- Schema-migration tests cover idempotent `ALTER TABLE ADD COLUMN` for late-arriving columns (the magnitude band landed this way) so older DBs upgrade in place.

## Deployment

Three paths documented in `deployment-manual.md`:

1. **Proxmox LXC (primary)** — `deploy/proxmox/install.sh` provisions an unprivileged Ubuntu 24.04 LXC, installs Python + the app, drops a hardened systemd unit, starts Streamlit on :8501. Idempotent re-runs upgrade in place; `--remove` tears down cleanly; `--local-source` packages a local checkout when the branch isn't on a reachable remote.
2. **Docker / Docker Compose** — `docker compose up --build` against the included `Dockerfile` + `docker-compose.yml`. Named volume persists the DB across `down`.
3. **Bare metal / VM** — venv + systemd unit recipe.

Reverse-proxy + TLS examples for Caddy and nginx (including the websocket-upgrade headers Streamlit needs) are in `deployment-manual.md §1.10`.

## Per-commit changelog

Full per-commit narrative lives in `diff.md` (~1,030 lines). Twenty-seven commits, oldest → newest:

<details>
<summary>Click for the full list</summary>

| Commit | Subject |
|---|---|
| `c4864df` | Initial Finn-Predictor scaffold |
| `9022488` | Session-only API key sidebar |
| `662415c` | Triple-layer API token scrubbing |
| `a3026dd` | `trust_env=False` to bypass intercepting dev proxy |
| `c31bd0d` | Resilient ingest — per-endpoint failure isolation |
| `bb09547` | Headlines as clickable links |
| `34254b0` | "Why this Call?" 5-paragraph explanation |
| `6aac6e0` | Ticker → company-name expansion + Altair contribution chart |
| `b0ca7cb` | Story clustering "first reported" timestamps |
| `72c2928` | One Prediction per `(target, UTC day, model)` |
| `eee1f26` | Per-stock directional predictor |
| `0fb1117` | Distinguish "news 403" from "candles 403, news empty" |
| `1f936f4` | Historical company-news backfill |
| `c6949c7` | Hypothetical-trade simulator + Performance tab |
| `290f387` | Focus tab — Company / Sector / Event drill-downs |
| `dd9790d` | Self-improvement — Bayesian opt over weights |
| `c48b339` | Activation policy AUTO/MANUAL |
| `dcef051` | Holdout-improvement gate + Docker |
| `c8e6314` | user-manual.md + installation-manual.md |
| `3b9f296` | Apply learned half-life + source weights live |
| `bbfb07e` | Production hardening — auth, non-root, Postgres, JSON logs |
| `61a9958` | Cap-weighted sectors + `FINN_PREDICTOR_SCORER` + logreg classifier |
| `24314ff` | Constituent caps + scorer-mismatch warning + UI smoke tests |
| `dcaa18c` | Pluggable story clustering (embedding opt-in) |
| `f355ee4` | Proxmox LXC installer + `deployment-manual.md` |
| `729da7a` | Quantile-band magnitude prediction (opt-in) |
| `7ec83b1` | Neutral-headlines section + actionable Sectors empty state |
| `c39b7ca` | Headless `refresh-constituents` CLI |
| `ac5551d` | Browser-localStorage API-key persistence |
| `fdf4133` | ^GSPC price chart + sector-grouped stocks + curated synthesis |
| `8467a42` | Docs sweep |

</details>

## How to review

```bash
# Read order if you've got 30 minutes:
1. summary.md            # 1-page architecture overview
2. progress.md §1-§2     # design decisions
3. finn_predictor/predictor/{market,sectors,stocks,classifier,magnitude}.py
4. finn_predictor/ingestion/{client,jobs}.py
5. finn_predictor/storage/{sector_membership,clustering}.py
6. tests/conftest.py + one tab-aligned test module

# Read order if you've got 5 minutes:
1. summary.md
2. progress.md §5 (current status)
```

To run it locally:

```bash
git clone -b finn-predictor https://github.com/petsan/finnhub-python
cd finnhub-python
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/pip install pytest pytest-cov pytest-mock requests-mock freezegun
.venv/bin/python -m pytest                       # 472 passing
.venv/bin/streamlit run finn_predictor/ui/app.py # http://localhost:8501
```

To deploy on Proxmox in one command:

```bash
# On the Proxmox host, as root:
git clone -b finn-predictor https://github.com/petsan/finnhub-python /tmp/finn
bash /tmp/finn/deploy/proxmox/install.sh
```

## Out of scope (intentional)

- **Live trading / broker integration.** The UI is read-only; nothing places orders.
- **Intraday predictions.** Daily resolution only.
- **Forex / crypto / bonds / mutual funds.** Finnhub endpoints exist; deferred.
- **Multi-user auth.** Single-user / single-password — recommend reverse-proxy SSO.
- **General-market news history.** Finnhub's `/news` only paginates forward; the `^GSPC` baseline can only build up over wall-clock time. Per-stock and per-sector baselines *can* be backfilled.

## Disclaimer

Predictions are informational only. **Not investment advice.** The classifier is sentiment-only — no price action, no earnings calendar, no macro, no microstructure. The expected-move band is intentionally wide because sentiment alone explains only single-digit-percent variance of next-day returns; treat it as range awareness, not point forecasting. Backtest PnL on the Performance tab models no friction; real-world results will be strictly lower.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

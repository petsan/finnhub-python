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
| Pluggable scorer | `FINN_PREDICTOR_SCORER=finbert` swaps VADER for FinBERT (requires `pip install torch transformers`); the live pipeline, backfill scoring, and the learning loop's `model_version` selector all route through one helper. |
| Cap-weighted sectors | Per-sector aggregation multiplies each article's recency × source weight by its company's most-recent market cap (Finnhub `/stock/historical-market-cap`). Tickers without a cap row keep weight 1.0; sectors fall back to uniform when nothing has been ingested. |
| Calibrated classifier | `FINN_PREDICTOR_CLASSIFIER=logreg` plus a fitted calibration replaces the z-score rule with `P(up) = sigmoid(intercept + beta * sentiment_index)`. Fit via `python -m finn_predictor.cli fit-classifier` once enough outcomes have closed; confidence becomes `|2P − 1|`. |
| Pluggable story clustering | Default 8-word-prefix matcher catches identical-lead reposts; `FINN_PREDICTOR_CLUSTERER=embedding` switches to sentence-transformers cosine clustering (lazy-loaded, ~80 MB on first use, default model `all-MiniLM-L6-v2`). The embedding path catches paraphrased rewrites the prefix matcher misses. Falls back to prefix on embed failure. |
| Proxmox LXC installer | `deploy/proxmox/install.sh` provisions an unprivileged Ubuntu 24.04 LXC, installs Python + the app, drops a hardened systemd unit, starts Streamlit on port 8501. Idempotent re-runs upgrade in place; `--remove` tears down cleanly. `deployment-manual.md` covers reverse proxy + TLS, backups, monitoring, hardening. |
| Magnitude band (opt-in) | `FINN_PREDICTOR_MAGNITUDE=quantile` plus a fitted calibration adds a 10th–90th-percentile return band to each prediction. Three new nullable columns on `Prediction`; pinball-loss quantile regression on closed outcomes; fit via `python -m finn_predictor.cli fit-magnitude`. UI Today-tab renders the band as `-0.8% to +1.2% (median +0.2%)`. Honest about uncertainty by design — the band is wide because sentiment-only signal can't claim more. |
| Neutral headlines | Today tab surfaces articles with `|sentiment| ≤ 0.05` (locked to the same `FLAT_SUPPORT_BAND` the classifier uses) in their own section under *Recent headlines*, sorted by recency. Keeps the reader honest about how much of the day's news flow the model actively used versus shrugged at. |
| ^GSPC price chart | Today tab opens with a 30-day Altair line chart of S&P 500 daily closes — scroll wheel zooms, click-drag pans. Hidden when no price bars exist. |
| Sector grouping + curated synthesis | Per-stock predictions on the Today tab are bucketed by curated `ticker → SPDR-sector` membership, with each section header showing a *synthesized* sector prediction aggregated from the user-listed stocks in that sector. Lets free-tier deploys produce sector predictions without Finnhub's gated `/etf/holdings`; cached `ETF_HOLDING` rows (paid plan) merge cleanly with the curated fallback. |
| Headless `refresh-constituents` | `python -m finn_predictor.cli refresh-constituents [--etf SYMBOL]... [--limit N]` mirrors the Focus → Sector → Refresh constituents button so cron / containers / batch ops can populate `ETF_HOLDING` without going through Streamlit. Per-sector resilient failure isolation; JSON output. |
| Browser-localStorage API-key persistence | Sidebar pre-fills the Finnhub key from `window.localStorage` on load; survives `Ctrl+R`. *Clear key* atomically wipes both server session and browser cache. Server-side scrubbing layers unchanged — the key still never writes to disk or the DB. Disable via `FINN_PREDICTOR_DISABLE_LOCAL_STORAGE=1` for the legacy session-only behaviour. |
| Postgres support | Set `FINN_PREDICTOR_DB_URL="postgresql+psycopg://..."` — the upserts are dialect-aware. `psycopg[binary]` ships in default requirements. |
| Structured logging | `FINN_PREDICTOR_LOG_FORMAT=json` switches to one-record-per-line JSON output suitable for log aggregators. |
| Docker | `docker compose up --build` builds and starts the app on host 8501. Named volume keeps the SQLite DB across `down`. `RESET_DB=1` wipes on next start. `docker compose run --rm app ingest|retrain|reset-db|shell|hash-password` for headless ops. Runs as non-root. |

## Limitations

* **General market news has no history** — Finnhub's `/news` only paginates
  forward. The market call's baseline can only build up over wall-clock time.
* **`/stock/candle` is gated** on the free tier, so backtest accuracy
  (`prediction_outcomes`) stays empty until you upgrade. Predictions still
  write fine; only the realised-return validation is missing.
* **Default classifier is still a rule.** Confidence is normalised
  z-distance, not probability. Day-1 of a new ticker can pin at conf=1.0
  before the rolling baseline accumulates 30 days of history. A calibrated
  logistic regression is available via `FINN_PREDICTOR_CLASSIFIER=logreg`
  once you've accumulated ≥10 closed UP/DOWN predictions and run
  `python -m finn_predictor.cli fit-classifier` — confidence then becomes
  `|2P − 1|`.
* **VADER is the default scorer.** Financial jargon ("beat estimates",
  "guidance lowered") is under-weighted. FinBERT is a one-env-var flip
  (`FINN_PREDICTOR_SCORER=finbert`) but requires `pip install torch
  transformers` because we keep the heavy deps out of the default tree.
* **Sector cap weights default uniform when caps haven't ingested yet.**
  Daily ingest now pulls `historical_market_cap` for every user-listed
  ticker, so the cap weighting kicks in automatically as soon as those
  symbols are added; sectors built from constituents that haven't been
  fetched still aggregate uniformly.

## Test posture

Every external HTTP call goes through `FinnhubGateway` (Finnhub) or an
injectable `history_fn` (yfinance), both mocked in tests — **no test
makes a real network call**. SQLite-backed tests run against `:memory:`
per-test, giving fast and isolated coverage. As of the latest commit:
**734 tests passing at 96% line+branch coverage** across the
`finn_predictor` package (472 baseline + 262 across the eight PRs
of the 2026-05-21 feature wave). Highlights:
`finn_predictor/ingestion/symbols.py`, PR-2's watchlist code in
`storage/models.py`, and PR-7's `finn_predictor/predictor/blended.py`
all ship at 100% line+branch coverage. `storage/repo.py` is at 98%
(every line added through PR-8 covered; remaining misses are
pre-existing legacy paths).

See `progress.md` for the design doc, `diff.md` for the per-commit
change log, `user-manual.md` for how to drive the dashboard, and
`installation-manual.md` for how to install / deploy / harden it.

## Additional reference docs (added 2026-05-21)

- **`architecture.md`** — engineer's reference: module-by-module LOC
  map, dependency direction, Mermaid diagram, runtime flows
  (ingest / UI render / training), data contracts, failure modes,
  and a "where new features plug in" matrix.
- **`security.md`** — first-pass security audit. 17 findings rated
  Critical→Info, threat model (assets, actors, surfaces), and a
  hardening roadmap. Re-run on every release. **F-13 resolved by
  PR-1; F-08 partially resolved.**
- **`design.md`** — proposal for the next feature wave: persisted
  watchlists, affinity-blended predictions (COMPETITOR /
  INSTITUTIONAL_HOLDER), and investment-theme tracking. Decisions
  on all four open questions are locked (see `progress.md` 2026-05-21
  entry); PR-1..PR-8 implementation order is the working plan.

## Changes since 2026-05-21 baseline (e927314)

### PR-1 — symbol-validation helper

- **New:** `finn_predictor/ingestion/symbols.py` (`valid_ticker`,
  `parse_ticker_list`, `ParseResult`). Caps user input at 50 tickers
  and 8 KiB; rejects everything that isn't `^[A-Z0-9.^-]{1,16}$`
  (via `re.fullmatch` — see the file's comment on the `$`-anchor
  trailing-newline gotcha).
- **New:** `tests/test_symbols.py` — 45 tests, 100% coverage.
- **Modified:** `finn_predictor/ui/app.py` `_parse_symbols` delegates
  to the new parser; legacy `list[str]` return shape preserved.
- **Security:** closes F-13 (symbol validator before upstream URL
  builder); partial close on F-08 (input caps; still need per-session
  ingest throttle + per-IP auth rate limit).

### PR-2 — Watchlist persistence + sidebar UI

- **New tables:** `Watchlist` (id, name unique, description, timestamps)
  + `WatchlistMember` (FK to watchlist with ON DELETE CASCADE, symbol,
  notes, added_at; `(watchlist_id, symbol)` unique). Migration is
  free — `init_db()`'s `create_all` picks them up on next start.
- **New repo helpers** in `storage/repo.py`: `create_watchlist`,
  `get_watchlist`, `list_watchlists`, `rename_watchlist`,
  `update_watchlist_description`, `delete_watchlist`, `add_to_watchlist`
  (idempotent), `remove_from_watchlist`, `watchlist_members`,
  `watchlist_symbols` (union across all lists when name is None),
  `replace_watchlist_symbols` (atomic; validates every symbol before
  any write). Plus a `WatchlistError(ValueError)` exception type and
  shape-checked name / symbol normalisers.
- **New UI helpers** in `ui/app.py`: `_format_parser_warning(ParseResult)`
  builds the sidebar caption that surfaces PR-1's rejected /
  truncated outputs; `_resolve_active_tickers(session, ...)` decides
  whether to ingest a saved list or the textbox content.
- **New sidebar widget:** *Watchlists* expander — selectbox of saved
  lists + manual-textbox option, "Save as new list" form, "Delete"
  button.
- **Tests:** `tests/test_watchlists.py` (58 tests covering CRUD,
  cascade delete, atomicity, ORM relationship) + `tests/test_ui_watchlist_helpers.py`
  (19 tests covering both new UI helpers).
- **Two SQLAlchemy / SQLite gotchas surfaced** (and documented in
  source comments for the next reader): (1) UoW batches INSERTs ahead
  of DELETEs on the same table → fixed with an explicit
  `session.flush()` between the two loops in
  `replace_watchlist_symbols`; (2) SQLite strips tz from
  `DateTime(timezone=True)` on read-back → tests use a
  `_to_utc(d)` helper that normalises both sides before comparison.
- **Security:** PR-1's `valid_ticker` is now also enforced inside
  every repo write path, so a future CLI / programmatic caller that
  bypasses the UI parser still can't corrupt the table.

### PR-3 — Streaks + Flipped + Band columns + CSV export

- **New repo helper:** `streaks_for(session, symbols, *, model_version=None) -> dict[str, StreakInfo]`
  in `storage/repo.py`. One SQL pass selecting
  `(target_symbol, label, prediction_date)` for the requested
  symbols, ordered by `(target_symbol, prediction_date DESC)`;
  Python walks the rows once and emits one `StreakInfo` per symbol
  with any history (omits symbols with no rows so callers can use
  `streaks.get(sym)` safely).
- **`StreakInfo` dataclass** (frozen): `symbol`, `current_label`,
  `streak (≥1)`, `previous_label (Optional[str])`,
  `flipped (bool)`. Documented "yesterday" semantics: it's the prior
  prediction by date, not literally calendar-yesterday.
- **`stock_predictions_table` extended:** new optional `streaks=`
  parameter (computed via the DB if absent), and three new columns
  — **Streak** (int, ≥1), **Flipped** (bool), **Band** (pre-formatted
  string from `format_expected_move` or empty).
- **`predictions_csv_bytes(df) -> bytes`** new pure helper: UTF-8
  CSV serialisation for the `st.download_button` data argument.
  Empty frame → header-only CSV. Round-trips via pandas.
- **UI integration:** the Today tab computes streaks once per render
  (passed to every per-section table builder + the combined-CSV
  builder), Streamlit `column_config` entries for the new columns
  (NumberColumn / CheckboxColumn / TextColumn), and a
  *⬇ Download per-stock predictions as CSV* button under the
  per-stock area. Filename:
  `finn-predictor-stocks-<today_iso>.csv`.
- **Tests:** `tests/test_streaks.py` (16 tests covering every shape
  of streak history) + 5 new tests in `tests/test_ui_helpers.py`
  (injected streak dict, defaults, auto-compute path, CSV header-only,
  pandas round-trip).
- **One bug surfaced** (and documented inline): pandas wraps Python
  bools as `numpy.bool_`, so `cell is False` is always False. Switched
  test assertions to `bool(cell) is False`. The next contributor
  reading the test file gets the lesson without having to debug it.

### PR-4 — COMPETITOR relationship + auto-seed

- **New module** `finn_predictor/predictor/affinity.py`. Auto-seed
  via `refresh_competitors` (PEER + finnhubIndustry match →
  COMPETITOR row; PEER rows preserved). Manual curation via
  `promote_peer_to_competitor` / `demote_competitor`. CLI subcommands
  in three flavours. `CompetitorRefreshResult` with per-peer
  failure isolation.
- **Schema:** `RELATIONSHIPS` frozenset extended with `COMPETITOR`.
- **Focus tab:** `CompanyFocus.competitors` field populated by
  `compose_company_focus`. Article universe expanded to include
  competitor symbols.
- **CLI:** `promote-competitor SYMBOL PEER_SYMBOL`,
  `demote-competitor SYMBOL PEER_SYMBOL`,
  `refresh-competitors --symbol SYMBOL [--symbol ...]`.
- **Tests:** `tests/test_affinity_competitors.py` (27) +
  `tests/test_cli.py` extensions (7).

### PR-5 — Institutional holders ingest (13-F)

- **New ingest** `refresh_institutional_holders(session, gateway, *,
  symbol, lookback_days, limit, today)` in
  `predictor/affinity.py`. Resilient to gating (403 captured as a
  failure, never raised). Pulls top N holders by Finnhub's natural
  sort order; ownership % + share + value + filing date live in
  `metadata_text` as JSON.
- **Helpers:** `_normalise_institution_name` (case + whitespace +
  truncate to 64 chars), `_coerce_float` (string→float for Finnhub's
  occasional string-numeric fields), `_parse_institutional_payload`
  (defensive — non-dict / missing data → empty).
- **Schema:** `RELATIONSHIPS` extended with `INSTITUTIONAL_HOLDER`.
  `Gateway.institutional_ownership(symbol, _from, to)` added.
- **Focus tab:** `CompanyFocus.institutional_holders` field
  populated by `compose_company_focus`.
- **Tests:** `tests/test_affinity_institutional.py` (18 tests).

### PR-6 — InvestmentTheme + ingest + themes predictor

- **Design note:** theme codes can exceed `Prediction.target_symbol
  String(16)` so themes do **not** write to the Prediction table.
  `predict_theme` returns a `ThemePrediction` dataclass — the
  UI/Today tab renders it on the fly. THEME_MEMBER edges remain the
  canonical lookup for PR-7's blender. Schema widening of
  `target_symbol` to `String(64)` is the documented follow-up.
- **New table:** `InvestmentTheme(id, theme_code unique, name,
  description, fetched_at)`.
- **New module** `finn_predictor/predictor/themes.py`:
  `DEFAULT_THEME_CODES` (12 curated codes), `ThemePrediction`
  dataclass, `predict_theme` (z-score rule against rolling baseline
  of theme indices; THEME_MEMBER → constituents lookup),
  `predict_all_themes` (iterates registered themes, sorted by
  confidence, patches in operator-friendly names).
- **Ingest:** `refresh_investment_themes(session, gateway, *,
  theme_codes=None)` in `predictor/affinity.py`. Per-theme failure
  isolation; idempotent re-run. Adds `_humanise_theme_code` and
  `add_investment_theme` for the operator-curated path.
- **Schema:** `RELATIONSHIPS` extended with `THEME_MEMBER`.
  `Gateway.stock_investment_theme(theme)` added.
- **Direction inversion documented:** THEME_MEMBER edges use
  `source_symbol = theme_code` and `related_symbol = ticker` — the
  reverse of the other relationship kinds — so that "constituents
  of theme X" is a single indexed lookup.
- **CLI:** `add-theme <code> [--name NAME] [--description DESC]`,
  `refresh-themes [--theme CODE ...]`.
- **Tests:** `tests/test_themes.py` (31 tests).
- **One bug surfaced** (and documented inline): `predict_theme`
  first cut called `rolling_baseline(daily_means)` — wrong API
  (that function walks the DB itself). Replaced with manual
  `aggregate_sentiment(daily_means)` + z-score computation against
  baseline mean/stddev.

### PR-7 — Affinity-blended per-stock predictor

The headline feature. **Off by default per the locked decision.**

- **New module** `finn_predictor/predictor/blended.py`:
  `AffinityWeights` frozen dataclass with curated defaults
  (SELF=1.0, PEER=0.10, COMPETITOR=-0.30, SUPPLIER=0.15,
  CUSTOMER=0.25, THEME_MEMBER=0.10, INSTITUTIONAL_HOLDER=0.0),
  `DEFAULT_AFFINITY_WEIGHTS`, `predict_stock_blended`,
  `predict_all_stocks_blended`, `AffinityContribution` dataclass,
  `explain_blend` (per-relationship breakdown for the UI popover).
- **Aggregation math:** for each related entity kind, articles
  contribute `(score × |weight| × recency_weight)` to the numerator
  and `|weight| × recency_weight` to the denominator; negative
  weights (COMPETITOR) preserve sign on contribution but not on the
  magnitude denominator. The result is a signed weighted mean
  bounded in [-1, +1]-ish, classified against the target's own
  rolling baseline of *blended* indices.
- **Persistence:** Prediction.model_version becomes
  `f"{base_model}+aff:default"` so blended and unblended calls
  coexist under the existing `(target, day, model)` uniqueness
  constraint.
- **THEME_MEMBER direction:** the blender uses
  `_theme_co_members(symbol)` (inverse lookup — themes containing
  the target → all other constituents) to find blend partners via
  themes, then excludes the target itself to prevent double-counting.
- **Tests:** `tests/test_affinity_blend.py` (23 tests, **100%
  module coverage on `blended.py`**).
- **Test-helper bug surfaced** (and documented inline): the
  `_seed` factory was re-scoring already-scored articles when
  called multiple times for the same symbol (UNIQUE constraint on
  sentiment_scores). Fixed by looking up only the just-inserted
  finnhub_ids rather than a tail-slice of all articles for that
  symbol.

### PR-8 — AFFINITY_WEIGHT learning dimension

**Hooks shipped, gp_minimize integration is a documented follow-up.**

- **New dimension:** `DIM_AFFINITY_WEIGHT = "AFFINITY_WEIGHT"`
  registered in `learning.config.DIMENSIONS`. One LearnedWeight row
  per relationship (key = relationship string, value = float).
- **`LearnedConfig` extended** with `affinity_weights: dict[str,
  float]` field and `to_affinity_weights() -> AffinityWeights`
  method (lazy import to avoid cycle; missing keys fall back to
  the curated default per-field).
- **`_persist_weights` extended** with an optional
  `affinity_weights=` argument; when None, persists the curated
  baseline from `_curated_affinity_baseline()` so every fresh
  training run carries the curated values as v1 baseline. When
  supplied, persists just those values.
- **Read path:** `weights_for_version` and `active_weights`
  round-trip affinity weights. The blended predictor can call
  `active_weights(session).to_affinity_weights()` to materialise an
  `AffinityWeights` for use at call time.
- **Tests:** `tests/test_affinity_learning.py` (13 tests).
- **Out of scope for PR-8 (filed as PR-8-follow-up):** actually
  fitting affinity weights via `gp_minimize`. That requires
  `learning/simulate.py` to compute a blended objective — i.e.
  running `predict_stock_blended` over the training frame with
  candidate weights and scoring the simulated trades. The schema
  + read machinery shipped in PR-8 is the foundation.

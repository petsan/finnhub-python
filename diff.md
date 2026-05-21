# Finn-Predictor — Change Log (branch `finn-predictor` vs. `master`)

Every commit on the `finn-predictor` branch is described below in
chronological (oldest → newest) order. The upstream `finnhub-python`
library itself (`finnhub/`, `setup.py`, etc.) is **untouched**; everything
new lives under `finn_predictor/`, `tests/`, and a few config files.

Totals across the branch: **101 files changed, ~26,400 insertions(+),
3 deletions(-)** vs. `master` (commit `c94e7d4 release 2.4.28`). The
2026-05-21 collaborative session added 13 new source/test files +
3 new docs (`architecture.md`, `security.md`, `design.md`) on top of
the previous 88-file total; see the PR-1 → PR-8 entries below for
the per-PR breakdown.

**Status (2026-05-21):** the branch was merged into `petsan/master`
as PR #1 (merge commit `e1d56e4`). All subsequent commits land
directly on `master`. The `finn-predictor` branch has been
deleted locally — its history is preserved through the merge.

---

## c4864df — Add finn-predictor: sentiment-driven market & sector forecaster

*38 files, +3759 / −0*

The initial commit. Three-layer application on top of `finnhub.Client`:

* **storage** — SQLAlchemy ORM (`models.py`), engine/session factory
  (`engine.py`), repository helpers (`repo.py`). Six tables:
  `news_articles`, `sentiment_scores`, `price_bars`, `predictions`,
  `prediction_outcomes`, `sectors`. SQLite-tz-roundtrip handled by
  normalising naive datetimes back to UTC at read time.
* **ingestion** — `RateLimiter` (sliding-window 55/min cap), retrying
  `FinnhubGateway` over the upstream client, news/price fetchers, and an
  APScheduler-backed daily-ingest job.
* **sentiment** — `Scorer` Protocol with `VaderScorer` (active) and
  `FinBertScorer` (lazy import; injectable pipeline so tests don't pull
  `torch`/`transformers`).
* **predictor** — z-score classifier with whole-market (`predict_market`,
  iter 1) and per-sector (`predict_sector`, iter 2) predictors, plus
  `score_outcomes` backtester.
* **ui** — Streamlit `Today` / `History` / `Sectors` dashboard.

86 tests at 98% line+branch coverage.

## 9022488 — ui: accept Finnhub API key via sidebar, session-only

*5 files, +265 / −13*

The UI now boots without `FINNHUB_API_KEY` set. A new sidebar exposes a
masked password input whose value lives only in `st.session_state` —
server-side, in-memory, per-tab. It is never written to disk, the DB, or
logs. Closing the tab or restarting the server discards it.

* `config.load_settings(require_api_key=True)` default preserved for CLI
  callers; UI bootstraps with `require_api_key=False`.
* `run_ingestion_with_key(session, api_key, ...)` builds a short-lived
  `finnhub.Client`, runs daily ingest, closes the client in a `finally`
  so the key-bearing connection pool doesn't linger.
* Defensive leak test: a canary key is run through the full helper, then
  every text column in the DB is scanned to confirm the canary never
  appears.

## 662415c — ingestion: scrub API token from all error paths

*5 files, +218 / −17*

A real-world failure surfaced a leak: `requests`-level transport errors
(`SSLError`, `ConnectionError`) include the full request URL — and
therefore `?token=…` — in their `__str__`. The earlier
`FinnhubAPIException`-only catch missed this path.

Defence-in-depth scrubbing at three layers:

1. `FinnhubGateway._call` now catches both `FinnhubAPIException` and
   `requests.exceptions.RequestException`, re-raising as a new
   `IngestionError` whose message has the live token redacted (looked up
   dynamically from `client._session.params`). The chain is suppressed
   via `raise IngestionError(...) from None`.
2. `run_ingestion_with_key` wraps any remaining `Exception` with the same
   scrub.
3. The Streamlit error display does one final scrub before writing to
   `st.sidebar.error`.

## a3026dd — ui: bypass HTTPS_PROXY env var in ingestion connection

*2 files, +31 / −0*

The user's dev environment had `HTTPS_PROXY=127.0.0.1:8080` pointing at a
local intercepting proxy (Burp/mitmproxy), which served its own self-
signed cert. `requests` honoured the env proxy by default → TLS
verification failed on every Finnhub call.

Per deploy decision, set `Session.trust_env = False` on the Finnhub
client built inside `run_ingestion_with_key` so the ingestion ignores
`HTTPS_PROXY` / `HTTP_PROXY` and connects directly. Other processes in
the same shell are unaffected.

## c31bd0d — ingestion: continue past per-endpoint failures, surface them in UI

*3 files, +144 / −26*

A 403 on `/stock/candle` (common on Finnhub free tier — the endpoint is
gated) used to abort the entire ingestion run, so news that the key
*does* have access to also failed to land. Now each Finnhub call is
wrapped: failures are collected into `counts["failures"]` and the run
continues.

Score-pending and predict steps are DB-only and always execute, so
articles ingested in earlier runs still get scored even if today's fetch
is fully blocked. UI shows a green "Done — articles +N, scored M,
predictions P" with an expander listing per-op failures.

## bb09547 — ui: link headlines to source URLs

*2 files, +160 / −2*

`recent_headlines()` now returns the article's `url`. New
`_format_headline_markdown()` builds one Markdown line per row with a
clickable headline link, falling back to bold text when no URL is
present. URL allow-list: only `http://` and `https://` become clickable;
unsafe schemes (`javascript:`, `data:`) render as plain bold text.
Headline brackets/backticks are escaped so they can't break the
Markdown.

The *Today* tab also now passes `VaderScorer().model_version` into
`recent_headlines` so the sentiment column actually populates
(previously always NaN).

## 34254b0 — ui: explain each Call in 5 paragraphs; sort headlines by contribution

*5 files, +771 / −20*

New module `finn_predictor/predictor/explain.py`:

* `ArticleContribution` dataclass: per-article signed contribution to a
  Prediction's weighted index (score × weight ÷ Σweights).
* `article_contributions(session, prediction)` ranks articles by
  `|contribution|` descending; `supports_call` flag matches the call sign.
* `explain_prediction(...)` renders 5-paragraph Markdown:
  call summary · mechanics · top movers · counter-signal · caveats.

UI: a "Why this Call?" `st.container(height=480)` scrollable block above
*Recent headlines*. Recent headlines are now sorted by
`|contribution|` descending, with 🟢 / 🔴 markers and a `contrib ±0.XXX`
suffix when a market prediction exists.

## 6aac6e0 — ui: expand tickers to company names + divergent contribution chart

*5 files, +522 / −6*

Two additions:

1. **Ticker → display-name expansion** via a new
   `finn_predictor/storage/symbol_names.py`. `WELL_KNOWN_NAMES` covers
   `^GSPC`, the 11 sector SPDR ETFs, and ~35 mega-cap tickers.
   `expand_symbol(session, sym)` resolves via that dict, then the
   `Sector` table, then falls back to the raw symbol.
   `expand_symbol_short()` strips any trailing `" (SYM)"`.
2. **Per-article contribution chart** via
   `contribution_chart_data()` + `build_contribution_chart()` using
   Altair. Divergent vertical bars with x-axis sorted ascending by
   contribution (left = most-negative, right = most-positive), y-axis
   centred on a 0-line. Tooltip includes headline, source, ticker,
   expanded company name, sentiment, contribution, supports-call flag.

## b0ca7cb — ui: show earliest "first reported" timestamp per story

*5 files, +520 / −0*

New module `finn_predictor/storage/stories.py`:

* `story_key(headline, n_words=8)` normalises a headline to a comparison
  key (lowercase, alphanumeric+space, leading definite/indefinite
  article dropped, first 8 tokens kept).
* `earliest_story_times(session, headlines, lookback_days=14)` returns
  the earliest `published_at` of any article whose `story_key` matches.
  **Prefix-aware**: a short headline ("Fed cuts rates") clusters with a
  longer variant ("Fed cuts rates by 25 bps in surprise move overnight")
  when one's key is a word-prefix of the other's.

UI: `attach_first_seen()` populates `row["first_seen_at"]`; the headline
formatter shows "first reported YYYY-MM-DD HH:MM UTC" only when it's
more than 5 minutes earlier than the article's own `published_at`
(`_FIRST_SEEN_DELTA_SECONDS`). Also added to the chart tooltip.

## 72c2928 — predictor: one row per (target, day, model) — no more click-duplicates

*6 files, +186 / −3*

Before: clicking *Run ingestion now* repeatedly on a single calendar day
produced N distinct `Prediction` rows that differed only in the
seconds-precision `prediction_date`. *History* over-counted and the
rolling baseline would double-count once same-day clicks landed in the
30-day window.

After: `predict_market` and `predict_sector` both normalise the input
`on_date` to start-of-UTC-day via `utc_day_window(on_date)[0]` before
constructing the `Prediction`. Two clicks at 03:48 and 04:06 UTC on the
same day now collide on `save_prediction`'s upsert key and update the
same row.

`migrate_predictions_to_daily(session)` in `storage/repo.py` collapses
existing duplicates: groups by `(target_symbol, utc-day, model_version)`,
keeps the row with the most-recent `created_at`, normalises its
`prediction_date` to midnight UTC, deletes the rest. Idempotent.
Migration was applied to the live `finn_predictor.db` (7 rows → 1).

## eee1f26 — predictor: per-stock directional call + UI table

*9 files, +456 / −21*

New module `finn_predictor/predictor/stocks.py`:

* `predict_stock(symbol)` — a thin specialisation of `predict_market`
  scoped to the ticker's own company news. Stored as a `Prediction` row
  with `target_symbol=<ticker>`, inheriting the one-row-per-day upsert
  and the click-duplicate fix.
* `predict_all_stocks(symbols)` — fans out across a list, skipping
  empty strings and tickers with no scored articles.

Daily ingest now fires per-stock predictions after sector predictions.
`_filter_for_prediction` in `explain.py` was widened to take a Session
and use the `Sector` table to distinguish sector ETFs from individual
stock targets — so the per-stock explanation block scopes article
windows to exactly that ticker.

UI: a new "Per-stock predictions" section between the explanation block
and the contribution chart. `st.dataframe` + `ProgressColumn` for
confidence, `NumberColumn` for sentiment, `DatetimeColumn` for "As of".
Sorted by Confidence desc, Articles desc.

## 0fb1117 — ui: distinguish "news 403" from "stock_candles 403, news empty"

*1 file, +33 / −7*

The previous "Every Finnhub call failed" red banner used the wrong
condition (`counts.general_news == 0 AND counts.scored == 0`), which
also held when `/news` succeeded with an empty response and only
`/stock/candle` returned 403.

Now: if `general_news` is in the failures list, show a clear red error
naming the op and likely causes (invalid/rotated key, plan without
`/news`, daily quota exhausted), with other failures in a secondary
expander. Otherwise (typical free-tier shape) keep the green success
banner. Empty-feed special case: success + a caption noting the feed
was unchanged.

## 1f936f4 — ingestion+ui: historical company-news backfill with retroactive predictions

*8 files, +851 / −12*

New module `finn_predictor/ingestion/backfill.py`:

* `backfill_company_news(symbol, start, end, chunk_days=30)` pages
  `/company-news` across the date range. Returns a `BackfillResult`
  dataclass (inserted/chunks_attempted/chunks_failed/failures). Each
  chunk routes through the existing `ingest_company_news` →
  `upsert_articles`, so re-running is safe (dedupe on `finnhub_id`). A
  per-chunk `IngestionError` is captured and remaining chunks still run.
* `backfill_many(symbols, ...)` runs the above across a ticker list,
  keyed by symbol.

New helpers in `finn_predictor/predictor/stocks.py`:

* `retroactive_predict_stock(symbol, start, end)` walks each UTC day in
  `[start, end]` and runs `predict_stock` per day. Inherits the
  one-row-per-day upsert. Re-running over the same range writes no
  extras.
* `retroactive_predict_many(symbols, ...)` fans out across tickers.

UI:

* `_SidebarState` dataclass carries both the daily-ingest click and the
  new backfill click + tickers CSV + 7–365-day lookback slider.
* `run_backfill_with_key()` mirrors the existing helper's safety
  guarantees: short-lived client, `trust_env=False`, scrubbed
  `IngestionError` on any leak path.
* Sidebar gains a "Backfill historical news" section under the existing
  daily-ingest controls. On click: spinner, then a green success block
  plus a per-ticker expander with chunk counts and any errors.

213 tests passing at 98% coverage by this commit.

## c6949c7 — predictor+ui: hypothetical trades + Performance tab

*8 files, +1547 / −2*

Three new layers stacked together so we can measure how often the
predictor is right.

New module `finn_predictor/ingestion/prices_yf.py`: yfinance-backed
daily-OHLC ingestion. Fills the gap left by Finnhub's gated
`/stock/candle` — no API key needed; writes to the same `PriceBar`
table the rest of the backtester reads. Per-symbol failures isolated.

New module `finn_predictor/predictor/trades.py`: `TradeRecord` dataclass
derived from each `Prediction` + its `PredictionOutcome` (UP → long
at prediction-day close, DOWN → short, FLAT → no trade; exit at next
session close). Aggregate helpers: `cumulative_pnl_series`,
`rolling_hit_rate`, `hit_rate_by_target_kind`, `hit_rate_by_label`,
`performance_summary`. All pure functions; no schema change.

UI: fourth tab **Performance** with 5 summary metrics, 4 Altair charts
(cumulative PnL, rolling 14-trade hit-rate with 50% reference rule,
hit-rate by target kind, hit-rate by Call), and a sortable trade
ledger. Sidebar gains *Backfill prices (yfinance)* — collects every
prediction target in the DB, pulls OHLC, then runs `score_outcomes()`
to close any newly-paired predictions.

## 290f387 — predictor+ui: Focus tab — company / sector / event drill-downs

*8 files, +1528 / −2*

New entity graph + a fifth tab.

Schema: `RelatedEntity` (source_symbol, related_symbol, relationship
∈ {PEER, SUPPLIER, CUSTOMER, ETF_HOLDING}, rank, metadata_text,
fetched_at) with `storage/repo.upsert_related_entity` /
`related_entities_for`.

Gateway: `company_peers`, `company_profile2`, `stock_supply_chain`,
`etfs_holdings` added to `FinnhubGateway`.

New module `finn_predictor/predictor/focus.py`:
* `compose_company_focus(symbol)` → CompanyFocus (own prediction +
  peers' predictions + supply chain when the plan supports it +
  recent articles spanning the subject and its peers).
* `compose_sector_focus(sector_code)` → SectorFocus (ETF's prediction
  + cached top constituents).
* `compose_event_focus(query, lookback_days)` → EventFocus (free-text
  search across headline + summary, per-ticker breakdown, implied
  Call via sign-of-mean classifier with ±0.1 dead-band).
* `refresh_company_relationships` and `refresh_sector_constituents`
  pull from Finnhub and persist via the new repo.

UI: fifth tab **Focus** with mode picker (Company / Sector / Event)
and per-mode renderers. Each renders the subject's metrics, related-
entities grids (with company-name expansion + ProgressColumn for
confidence), and the existing linked-headline list.

## dd9790d — learning: train + apply weights from hypothetical-trade outcomes

*10 files, +1422 / −4*

Self-improvement layer.

Schema: `LearnedWeight` (version, dimension, key, value, fitted_at,
training_score, holdout_score, is_active). Exactly one version is
active; old versions stay for inspection / revert.

New package `finn_predictor/learning/`:
* `config.py` — LearnedConfig dataclass + `active_weights(session)` +
  `weights_for_version`. Falls back to hand-tuned constants when no
  version is active.
* `simulate.py` — `build_training_frame` loads articles + scores +
  outcomes once; `simulate(frame, config)` replays the predictor
  pipeline (recency-weighted aggregate → rolling baseline → z-score
  → classify with the candidate's threshold) for every closed
  prediction. `blended_objective = hit_rate + 0.5 × cum_pnl`.
* `train.py` — `train_weights(session, n_calls=30)` time-splits the
  frame (last 14 days = holdout), fits per-source weights via a
  closed-form hit-rate-ratio pass, then runs skopt's `gp_minimize`
  over (threshold_sigma, min_baseline_sigma, half_life_hours).
  Persists a new LearnedWeight version + auto-activates.

`predict_market` gains `threshold_sigma` + `min_baseline_sigma`
overrides. Daily ingest reads `active_weights(session)` and passes
them through. UI: sixth tab **Learning** with current weights +
*Retrain now* button + Bayesian-iteration slider + version history.

New dep: scikit-optimize 0.10.2 (lazy-imported).

## c48b339 — learning: toggle between auto-activate and manual approval

*7 files, +319 / −7*

Cross-session app_settings table backs an `activation_policy`
preference (AUTO ⇔ MANUAL). UI radio in the Learning tab; `train_weights`
consults it via a new sentinel default `activate=None`.

Schema: `AppSetting` (key, value, updated_at) — tiny key/value table
for cross-session UI preferences. Survives Streamlit restarts.

Repo: `get_setting` / `set_setting` (generic), plus typed
`get_activation_policy` / `set_activation_policy` (defaults AUTO;
junk values coerce back to AUTO). New `activate_learned_version`
flips `is_active` flags atomically.

UI: radio under active-weights ("Auto-activate — newest wins" vs.
"Manual approval — leave new version inactive; you click Activate").
Each inactive version in the history table gets an "Activate v\<n\>"
button. Max 4 buttons rendered to avoid layout blow-up.

## dcef051 — learning + docker: holdout gate + container deploy

*12 files, +776 / −8*

Holdout-improvement gate + Docker stack.

Holdout gate: new `holdout_tolerance` setting (default 0.01). Under
AUTO policy, `train_weights` re-scores the currently-active config on
the same holdout window via `blended_objective`; if the candidate
scores worse than `(active - tolerance)`, save the version but leave
inactive and report the gap. `TrainingReport` carries `activated`,
`gate_blocked`, `gate_reason`, `active_holdout_score_at_decision`,
`holdout_tolerance` so the UI can explain what happened. Explicit
`activate=True` still bypasses the gate.

New CLI module `finn_predictor/cli.py` with `serve` / `reset-db
[--yes]` / `retrain [--n-calls N] [--activate auto|yes|no]`.
Argparse-driven; exit codes 0/2/3.

Docker: Dockerfile (python:3.12-slim + build-essential), docker-
compose.yml (single `app` service, `finn_data` named volume,
healthcheck), docker/entrypoint.sh (routes serve|reset-db|retrain|
shell + RESET_DB env), requirements.txt (every app dep pinned to a
lower bound), .dockerignore.

UI: Learning tab gains tolerance slider, three-state post-training
feedback (Activated / Saved-not-active / Gate-blocked) with
"Activate v\<n\> anyway" override button.

## c8e6314 — docs: add user-manual.md and installation-manual.md

*2 files, +796 / −0*

Two long-form guides separating "how do I use the dashboard?" from
"how do I install / deploy / maintain it?". Cover sidebar contract,
every tab, recommended first-run flow, dashboard limits,
troubleshooting matrix, Docker workflow, CLI cheat-sheet, env vars,
backup/restore, upgrade procedure, install troubleshooting,
production-hardening notes.

## 3b9f296 — predictor: apply learned half-life + source weights live; sectors auto-load from DB

*8 files, +368 / −30*

Closes two gaps where the system claimed behavior it didn't deliver.

`aggregate.daily_sentiment_index` gains a `source_weights` dict arg;
per-article weight becomes `recency_weight × source_weights.get(a.source, 1.0)`.
`rolling_baseline` forwards `half_life_hours` + `source_weights` to
every inner daily call so the baseline tracks the live config.

`predict_market`, `predict_sector`, `predict_stock`,
`predict_all_sectors`, `predict_all_stocks` all gain the four
learnable params and forward them. `predict_sector` refactored:
`_sector_articles_scores` became `_sector_scored_articles` returning
(article, score) pairs so per-source weights can be applied.

`predict_all_sectors(sector_universe=None)` now reads from cached
`RelatedEntity(relationship="ETF_HOLDING")` rows via the new
`_sector_universe_from_db(session)` helper. One click in
*Focus → Sector → Refresh constituents* now makes every subsequent
daily ingest produce a sector Call automatically.

`ingestion.jobs.run_daily_ingest` builds a single `learned_kwargs`
dict from `active_weights(session)` and threads it through every
predictor — half-life and source weights now actually take effect at
live scoring time.

## bbfb07e — production hardening: auth gate, non-root Docker, Postgres, JSON logs, ingest CLI

*12 files, +762 / −9*

Six hardening items from `installation-manual §9` in one pass.

New module `finn_predictor/security.py`:
* `hash_password(plaintext)` / `verify_password(plaintext, hash)` —
  bcrypt wrappers. Cost factor 12. Constant-time check.
* `auth_enabled()` / `current_password_hash()` — env helpers.
* `SecretScrubFilter` — logging filter that masks anything matching
  the Finnhub-token regex (40-char [0-9a-z]) or the bcrypt-hash
  prefix. Defence in depth behind the three explicit scrubbing layers
  on the ingestion path.

UI: `_enforce_auth_gate()` runs before `main()` does anything else.
When `FINN_PREDICTOR_PASSWORD_HASH` is set, the page renders a
password prompt until the right plaintext unlocks the
session_state authentication flag. Unset = no auth = same default-
localhost behaviour.

New module `finn_predictor/logging_config.py`: `setup_logging(fmt=)`
reads `FINN_PREDICTOR_LOG_FORMAT` (`text`|`json`, default `text`)
and `FINN_PREDICTOR_LOG_LEVEL` (default `INFO`). JSON path emits
one-record-per-line `{ts, level, logger, message, exc_info?}`. Both
formats attach `SecretScrubFilter` to every handler.

Postgres dialect: new `_dialect_insert(session)` helper picks
`sqlalchemy.dialects.postgresql.insert` vs `sqlite.insert` at runtime
based on the bound dialect. Both upsert call sites switched to it.
`psycopg[binary]>=3.1` in `requirements.txt`.

Docker: Dockerfile now adds a system user `finn` (uid 1001, no
shell, no home), `chown`s /app + /data, `USER finn`. Runtime
process never has root.

CLI: two new subcommands.
* `hash-password [plaintext]` — prints a bcrypt hash + the exact
  `export FINN_PREDICTOR_PASSWORD_HASH=...` line on stderr.
  Reads from stdin via `getpass.getpass` when called without args so
  the plaintext stays out of shell history.
* `ingest` — runs one `run_daily_ingest` cycle headlessly. Reads
  `FINNHUB_API_KEY` from env. Suitable for cron / scheduled-task
  sidecar. Crontab example in `installation-manual §5.1`.

New deps:
* `bcrypt>=4.0` (auth gate)
* `psycopg[binary]>=3.1` (Postgres driver; in default
  `requirements.txt` so the image works against either store
  out of the box)

337 tests passing at 95% coverage by this commit.

---

## 61a9958 — sprint: cap-weighted sectors, pluggable scorer, calibrated classifier

*28 files, +1529 / −38*

Closes the three open "Known limitations / next steps" items from
`progress.md §3.6` in one pass, plus a CLI test coverage lift.

**Cap-weighted sector aggregates.** New
`HistoricalMarketCap(symbol, as_of_date, market_cap)` model with a
unique `(symbol, as_of_date)` constraint. Companion repo helpers:

* `upsert_market_caps(session, caps)` — dialect-aware insert with
  no-op-on-conflict (matches `upsert_price_bars`).
* `latest_market_caps(session, symbols, *, on_or_before=None)` —
  returns `{symbol: cap}` from the most recent snapshot per ticker;
  `on_or_before` guards historical backtests against look-ahead.

New `FinnhubGateway.historical_market_cap(symbol, _from, to)` (Finnhub
takes ISO dates here, not epoch seconds — gateway docstring spells
that out). New `ingest_market_caps(session, gateway, symbol, start,
end)` in `ingestion/prices.py` parses Finnhub's `{symbol, data:
[{atDate, marketCapitalization}, ...]}` shape, skipping malformed
rows. Daily ingest now calls it for every user-listed company
ticker; failures isolated per-symbol via the same `_try` wrapper.

`predict_sector` accepts `use_market_cap_weights=True` (default). When
on and the DB has caps for any of `sector_symbols`, each article's
per-ticker weight is multiplied by its company's most-recent cap
(normalised to mean 1.0 across the sector so the weighted_mean stays
in the same numeric range as the cap-less path — important because
the rolling baseline doesn't see cap weights).

**`FINN_PREDICTOR_SCORER` env toggle.** New `scorer_name` field on
`Settings` (validated against `vader`|`finbert`). New
`resolve_active_scorer()` helper in `sentiment/__init__.py` reads
the env directly (so short-lived callbacks don't need a Settings
instance) and falls back to VADER on unknown / blank values — the
UI must not crash at startup over a typo. Routed through every live
scoring site: `cmd_ingest` (CLI), `run_ingestion_with_key` and
`run_backfill_with_key` (UI), Focus tab's version display, and the
training loop's `model_version` selector.

**Logistic-regression classifier mode.** New
`predictor/classifier.py` with:

* `LogisticCalibration(beta, intercept, n_samples)` dataclass.
* `_fit_logreg_newton(xs, ys)` — pure-Python 2-param Newton-Raphson
  MLE with a small `eps=1e-3` L2 prior (otherwise a separable
  training set sends |beta| → ∞). Iterates ≤ 50 times to ≤ 1e-7
  step norm.
* `fit_logreg_calibration(session, model_version, target_symbol=None)`
  — walks closed UP/DOWN predictions, raises
  `NotEnoughCalibrationDataError` when < 10 samples or all on one
  class, returns the fitted calibration.
* `apply_logreg_classification(sentiment_index, calibration,
  decision_band=0.05)` → `(label, confidence)`. Probability above
  `0.5 + band` → UP, below `0.5 - band` → DOWN, else FLAT.
  Confidence is `|2P − 1|` — a calibrated probability gap.
* `save_calibration` / `load_calibration` — JSON blob in a single
  `AppSetting` row (`logreg_calibration`).
* `resolve_classifier_mode()` reads `FINN_PREDICTOR_CLASSIFIER`
  (default `rule`, alt `logreg`).

Predictor signatures grew an optional `calibration=` parameter
(market, sectors, stocks, plus `predict_all_sectors` /
`predict_all_stocks` for the fan-out). `run_daily_ingest` loads
the calibration once per call when the env is `logreg`.

New CLI subcommand `python -m finn_predictor.cli fit-classifier
[--target-symbol SYM]`. Filters the training set by the active
scorer's `model_version` so VADER + FinBERT predictions don't get
mixed.

**CLI test coverage lift.** New tests for `cmd_ingest` happy path
(via `run_daily_ingest` mock), `hash-password` stdin / EOF /
too-long paths, `retrain --activate yes|no`. `cli.py` 77% → 96%.

**Net test count:** 337 → 381 passing. Overall coverage: 95% → 96%.
`classifier.py` lands at 95% straight out of the gate. Docs:
`progress.md §3.6` strikes three items, §4 changelog appends six
2026-05-20 entries, §5.4 refreshes test posture. `summary.md`
Limitations + Features aligned. `installation-manual.md` documents
the two new env vars.

---

## 24314ff — follow-up: constituent caps, scorer-mismatch warning, UI smoke tests

*8 files, +446 / −19*

Closes the last cap-weighting gap from the prior commit, plus two
adjacent items.

**Cap ingest fan-out across cached sector constituents.** After the
per-company loop, `run_daily_ingest` now walks every `Sector`'s
cached `RelatedEntity(ETF_HOLDING)` rows and pulls
`historical_market_cap` for each constituent — deduped against the
user-listed tickers via a per-run `caps_ingested` set so we never
call `/historical-market-cap` twice for the same symbol. New
`constituent_caps` count and per-symbol failure isolation. Sectors
built purely from constituents now also pick up cap weighting,
without listing each ticker explicitly.

**Scorer-mismatch warning.** New `detect_scorer_mismatch` /
`warn_if_scorer_mismatch` helpers in the sentiment package compare
the live scorer's `model_version` against the most recent
`Prediction.model_version`. On drift, the WARNING explains that the
rolling baseline is now stale and points at the `retrain` CLI.
Hooked into both the CLI `ingest` path and the UI's
`run_ingestion_with_key`. Silent on a matching DB or an empty DB.

**UI smoke tests via `streamlit.testing.v1.AppTest`.** New
`tests/test_ui_smoke.py` boots `ui/app.py` against a per-test
SQLite fixture and asserts (a) no exceptions at import/boot, (b) the
`Finn-Predictor` title renders, (c) all six tab labels appear, (d)
the bootstrap copy shows on a fresh DB, (e) a seeded UP prediction
surfaces in the Today-tab metrics. AppTest runs the script in its
own context so line coverage isn't captured; `ui/app.py` stays in
`.coveragerc` `omit` but is now behind four explicit regression
tests.

**Net:** 381 → 392 passing, 96% coverage maintained.

---

## dcaa18c — storage+ui: pluggable story clustering (prefix default, embedding opt-in)

*8 files, +564 / −12*

Closes the last documented heuristic in `progress.md §3.6`:
paraphrased reposts that share an event but rewrite the lead now
cluster correctly when the embedding clusterer is active, while the
default deploy keeps the dependency-free behaviour unchanged.

New module `finn_predictor/storage/clustering.py`:

* `Clusterer` Protocol — one method, `earliest_times(session,
  headlines, *, lookback_days, now) -> dict[str, datetime|None]`.
* `PrefixClusterer` (default) — thin wrapper around the existing
  `earliest_story_times` so the call sites can hold a single
  `Clusterer` instance regardless of mode.
* `EmbeddingClusterer` — pulls every article in the lookback window,
  embeds the input + candidate headlines in one batched call,
  clusters by cosine similarity ≥ threshold (default 0.7), returns
  the earliest `published_at` per input. Sentence-transformers is
  lazy-imported on first use (default model `all-MiniLM-L6-v2`,
  ~80 MB on first download); an injectable `embed_fn` keeps tests
  hermetic.
* `resolve_active_clusterer()` — env-driven picker; default
  `prefix`, `embedding` opts in, unknown values fall back to prefix.

Resilience: a failing embed call (model load fails, OOM, network
flake) logs a WARNING and falls back to the prefix matcher rather
than blanking the *first reported* column.

Pure-Python cosine math (no numpy dependency at the type boundary).
UI wiring: the two callers in `finn_predictor/ui/app.py` that
previously imported `earliest_story_times` directly now route
through `resolve_active_clusterer().earliest_times(...)`.

**Tests** (17 new, in `tests/test_clustering.py`): pure cosine math,
PrefixClusterer protocol compliance + parity with the legacy helper,
EmbeddingClusterer with injected `embed_fn` (paraphrase clustering,
threshold enforcement, empty inputs, empty window, batch dedupe,
fallback-on-failure), env-driven resolver paths.

**Net:** 392 → 409 passing, 96% coverage maintained. No new runtime
dependencies — `sentence-transformers` is opt-in.

---

## f355ee4 — deploy: Proxmox LXC installer + deployment manual + doc refresh

*7 files, +1277 / −5*

Deployment story rounded out:

* **`deploy/proxmox/install.sh`** — one-shot Proxmox LXC installer.
  Provisions an unprivileged Ubuntu 24.04 LXC (1 GB / 2 cores / 8 GB
  default; overridable via env or `--memory` / `--cores` / `--disk`
  flags), installs Python 3.12 + the app, drops a hardened systemd
  unit (`NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=strict`,
  `ProtectHome`, scoped `ReadWritePaths`), starts the service.
  Idempotent: re-runs against an existing CTID refresh the app and
  restart the service without touching the SQLite DB under
  `/opt/finn-predictor/data`. `--remove` for teardown. Optional
  flags for FinBERT (`--with-finbert`) and embedding clusterer
  (`--with-embeddings`).
* **`deployment-manual.md`** — new top-level doc with three
  deployment paths (Proxmox LXC primary, Docker, bare metal),
  reverse-proxy + TLS recipes (Caddy and nginx), backup/restore
  commands, health-check endpoints, monitoring hooks, hardening
  checklist.
* **`user-manual.md`** — caught up with the recent sprints: §1
  mentions both classifier modes, §3.3 documents cap-weighted sector
  aggregation, §5.7/§5.8 add walkthroughs for switching the
  classifier and scorer, §6 picks up new troubleshooting cases
  (scorer-mismatch warning, fit-classifier rc=3 paths). §7 points
  at the new deployment manual.
* **`README.md`** — top-of-file branch note explaining what's in this
  fork and linking the doc set.
* **`summary.md` / `progress.md` / `diff.md`** — refreshed test
  posture (409 passing, 96% coverage), changelog entries, this diff
  block.

No production code changed; 409 tests still pass.

Verified end-to-end on a live Proxmox 9.1 node:
* Cluster-wide free-CTID lookup picked the right ID via
  `pvesh get /cluster/nextid`.
* `--local-source` path packaged and pushed cleanly (.git excluded).
* Inner script detected the pre-seeded tree (no `.git` → skip clone).
* venv + pip install + systemd unit installed, service started.
* `GET /_stcore/health` returned 200 from both inside and outside
  the container.

---

## 729da7a — predictor: quantile-band magnitude prediction (opt-in)

*19 files, +1217 / −6*

Adds the magnitude mode discussed in the chat exchange about "can it
predict how much the market will move". Honest about uncertainty by
design: the band is wide because the underlying signal carries only
single-digit-percent variance explanation, and a single-number
forecast would imply precision the data can't support.

New module `finn_predictor/predictor/magnitude.py`:

* `MagnitudeForecast(p10, p50, p90)` dataclass with crossed-quantile
  sort defence (a fit that crosses on small data silently reorders
  so the UI never shows lower > upper).
* `MagnitudeCalibration`: three `QuantileFit` rows (one per tau in
  `DEFAULT_QUANTILES = (0.10, 0.50, 0.90)`), plus `n_samples` and
  `feature_name` for future-version mismatch detection.
* `_fit_quantile_regression_1d`: pure-Python pinball-loss
  minimisation via subgradient descent. Warm-starts intercept at the
  y-median and tracks the best-objective parameters seen.
* `fit_quantile_calibration`: walks closed predictions, raises
  `NotEnoughMagnitudeDataError` when n < `MIN_FIT_SAMPLES=15` or
  when realised_return has zero variance.
* `save_calibration` / `load_calibration`: JSON in a single
  `AppSetting` row (key `magnitude_calibration`), mirroring the
  logreg classifier pattern.
* `resolve_magnitude_mode`: env-driven picker, defaults to `off`,
  silently falls back from unknown values.

Schema: three nullable Float columns on Prediction —
`expected_return_p10`, `_p50`, `_p90`. New `_add_missing_columns`
helper in `storage/engine.py` runs idempotent ALTER TABLE ADD COLUMN
on `init_db`, so DBs written by older deploys upgrade in place when
the service restarts. Verified via `tests/test_engine_migrations.py`.

Predictor wiring: `predict_market` / `predict_sector` /
`predict_stock` take an optional `magnitude_calibration` kwarg.
`save_prediction`'s upsert no-ops on null magnitude columns
specifically, so a re-run without the calibration doesn't erase a
band written by a prior run with it.

CLI: new `python -m finn_predictor.cli fit-magnitude` mirrors
`fit-classifier`. Filters the training set by the active scorer's
`model_version` so VADER + FinBERT predictions never mix.

UI: `format_expected_move(prediction)` renders the band as
`-0.80% to +1.20% (median +0.20%)`. Today tab calls
`_render_expected_move` under the market metrics; the row is hidden
when the band isn't populated.

Tests: +29 (21 in `test_magnitude.py`, 3 in
`test_engine_migrations.py`, +2 in `test_cli.py`, +3 in
`test_ui_helpers.py`). Net: 409 → 438 passing.

---

## 7ec83b1 — ui: neutral-headlines section + actionable Sectors-tab empty state

*3 files, +237 / −2*

Two UX fixes prompted by the live deploy walkthrough.

**Neutral headlines** (new section under Recent headlines on the
Today tab): articles in the prediction's window whose scorer-extracted
sentiment sits at or below `NEUTRAL_SENTIMENT_THRESHOLD=0.05` — i.e.
the model saw them but couldn't extract polarity. Surfacing them
keeps the reader honest about base rates while keeping the
contribution-ranked view free of noise. The threshold is locked to
`predictor.explain.FLAT_SUPPORT_BAND` (tested) so the
"this didn't move the Call" cutoff stays consistent across UI
sections.

**Sectors-tab empty state**: previously displayed "Sectors not
seeded yet — run an ingestion cycle" whenever the latest-prediction-
per-sector grid was empty, which was misleading because the 11
default sectors ARE seeded on first ingest. Now distinguishes three
shapes:

* No sectors in the DB → original message.
* Sectors seeded but no `ETF_HOLDING` cached → actionable nudge
  pointing at Focus → Sector → Refresh constituents.
* Sectors + ETF_HOLDING cached but no `company` articles in the
  window → nudge to run ingestion / backfill.

Tests: +6 in `test_ui_helpers.py` (neutral-filter cutoff, recency
ordering, limit, threshold override, empty input,
FLAT_SUPPORT_BAND parity). Net: 438 → 444 passing.

---

## c39b7ca — cli: refresh-constituents — headless equivalent of UI Refresh button

*4 files, +264 / −0*

Mirrors the Focus → Sector → Refresh constituents button as a CLI
subcommand so cron / containers / headless ops can populate
`RelatedEntity(ETF_HOLDING)` without going through Streamlit.

```
python -m finn_predictor.cli refresh-constituents
python -m finn_predictor.cli refresh-constituents --etf XLK --etf XLV
python -m finn_predictor.cli refresh-constituents --limit 50
```

Behaviour:

* Reads `FINNHUB_API_KEY` from env (rc=2 + friendly message when
  unset, same shape as `cli ingest` / `fit-classifier`).
* Rejects an unseeded DB (rc=2) rather than silently no-op — keeps
  the failure mode legible.
* Iterates every `Sector` row by default; `--etf <SYMBOL>`
  (repeatable) scopes the run, `--limit` caps constituents per
  sector (default 25).
* Resilient: per-sector `IngestionError` (free-tier 403 on
  `/etf/holdings`, rate-limit) lands in `summary["failures"]`
  without blocking peers.
* Prints a JSON summary listing refreshed sectors, per-sector
  failures, and skipped sectors (when `--etf` is used).

Tests: +5 in `test_cli.py` (requires-api-key, rejects-unseeded-db,
happy-path, isolates-per-sector-failures, filter-by-etf). Net:
444 → 449 passing.

**Live note**: a real run on the deployed instance surfaced that
the user's Finnhub free-tier plan gates `/etf/holdings` (every
sector got 403'd). The CLI worked exactly as designed — per-sector
resilient failure isolation, structured JSON output — but exposed
that sector synthesis needs another path on free tier. That
becomes the curated map in `fdf4133`.

---

## ac5551d — ui: persist Finnhub API key in browser localStorage

*6 files, +202 / −11*

Adds the persistence the sidebar lacked: the API token now survives
page reloads via `window.localStorage`, and the *Clear key* button
wipes both the server session and the browser cache in one atomic
click.

Implementation:

* New constants: `BROWSER_STORAGE_API_KEY` (the localStorage key,
  prefixed with `finn_predictor_` so we don't collide with other
  Streamlit apps on the same origin) plus two `session_state`
  markers for change detection and one-shot hydration.
* `_get_local_storage()` — defensive constructor wrapper around
  `streamlit-local-storage`'s `LocalStorage()`. Returns `None` instead
  of hanging when:
    1. The package isn't installed (ImportError).
    2. `FINN_PREDICTOR_DISABLE_LOCAL_STORAGE` is truthy — explicit
       opt-out for tests / scripts / users who want the legacy
       session-only behaviour.
    3. The constructor raises for any other reason.
* `_hydrate_api_key_from_browser()` — runs at most once per session
  BEFORE the text_input renders. Reads localStorage, pre-fills
  `st.session_state[API_KEY_SESSION_KEY]`. Bypassed cleanly when
  localStorage isn't reachable.
* `_persist_api_key_to_browser(api_key)` — change-detected
  writeback. Avoids spamming the component bridge on every rerun.
* `_clear_browser_api_key()` — wired into the *Clear key* button.
  Atomic wipe: server session + browser cache + hydrate flag in
  one click.

Why the escape hatch: `streamlit-local-storage`'s `__init__` polls
`st.session_state` until the frontend custom component reports back —
fine under `streamlit run`, but it never exits under
`streamlit.testing.v1.AppTest` (component JS doesn't execute),
hanging the entire smoke-test suite. The DISABLE env var lets the
test fixture skip the bridge while preserving the production path.

Sidebar caption updated to reflect the new posture: the key now
"persists in this browser's localStorage" but "never written to the
server's disk or the SQLite database" — both still true,
distinguishing client-side from server-side storage.

Dependency: `streamlit-local-storage>=0.0.20` added to
`requirements.txt`. Self-contained Streamlit component with a
bundled frontend bundle; no transitive heavyweight deps.

Tests: +4 in `test_ui_helpers.py`. Net: 449 → 453 passing.

---

## fdf4133 — ui+jobs: ^GSPC price chart + sector-grouped stocks + curated synthesis

*7 files, +648 / −37*

Two features in one commit because they share the goal of making
the Today tab useful on a free-tier Finnhub deploy where
`/etf/holdings` returns 403 and the Sectors tab would otherwise be
blank.

**Curated ticker → sector membership** (new module):

* `finn_predictor/storage/sector_membership.py` — ~110 mega-cap
  tickers across the 11 SPDR sectors, hardcoded. Public knowledge
  (top holdings of each Sector Select SPDR ETF as of mid-2025).
* `sector_for_ticker(symbol)` — normalises case + whitespace,
  returns the sector code or None for the long tail.
* `sector_universe_from_tickers(tickers)` — builds the
  `{sector_code: [ticker, …]}` shape `predict_all_sectors`
  consumes. Dedupes, preserves user-input order, silently drops
  unmapped symbols.
* `merge_sector_universes(primary, secondary)` — unions two
  universe dicts; primary wins on ordering. Used by the ingest
  job to combine cached ETF_HOLDING with the curated fallback.

**Daily-ingest wiring** (`jobs.py`):

`run_daily_ingest` now builds a merged sector universe before
calling `predict_all_sectors`. Primary: cached
`RelatedEntity(ETF_HOLDING)` rows (paid-plan path). Secondary:
curated map applied to the user's `company_symbols` (free-tier
fallback). Sectors with at least one constituent from either
source produce predictions.

**UI Today-tab — ^GSPC price chart** (new):

`build_market_price_chart(bars, *, symbol, days)` — pure helper
returning an Altair line chart with `.interactive()` for mouse-
wheel zoom + click-drag pan. Y-axis padded ±5% so daily candles
don't get crushed. Hidden when no bars exist.

**UI Today-tab — sector-grouped stock predictions** (changed):

`group_stock_predictions_by_sector` buckets predictions by curated
sector code, returns `([(Sector, [pred, …]), …], [unmapped, …])`.
The Today tab replaces the single flat dataframe with one section
per sector, each header showing the synthesized sector prediction
when available. "Other / unmapped" section appears below for
tickers not in the curated map.

Tests: +18 (13 in `test_sector_membership.py`, +5 in
`test_ui_helpers.py`). Net: 453 → 472 passing.

---

## HEAD — docs: bring summary.md / progress.md / diff.md current (this commit)

*~3 files, ~+400 / −few*

Catch-up docs sweep capturing the five-commit run between
`f355ee4` and `fdf4133`:

* `progress.md` §3.6: all originally-open limitations are now
  struck through (magnitude, scorer toggle, classifier mode,
  clusterer, cap weighting, constituent caps, free-tier sector
  fallback). §5.4 test posture refreshed to 472 passing / 96%
  coverage. §4 changelog gets four new dated entries.
* `summary.md`: Features table gains four rows (Magnitude band,
  Neutral headlines, refresh-constituents CLI, localStorage
  persistence, curated sector synthesis, price chart). Test
  posture line bumped to 472.
* `diff.md`: per-commit blocks for `729da7a`, `7ec83b1`,
  `c39b7ca`, `ac5551d`, `fdf4133`, each at the same level of
  detail as the prior entries.

No application code changed; 472 tests still pass.

---

## 2026-05-21 — kickoff: codebase summary + security audit + architecture diagram

*3 docs added, no source code changed.*

Operator brief: scan the codebase, produce a software-architecture
diagram, investigate how to add individual-stock predictions + the
front-end, design how to track stocks/themes by affinity (competition,
supplier, controlling interest, etc.), run a security audit, write
tests for everything, and keep `progress.md` + `summary.md` current.

Discovery findings (no code changes yet):

* Per-stock predictions **already exist** end-to-end
  (`predictor/stocks.py`, surfaced on the Today tab sector-grouped +
  sorted by confidence). The brief's "add individual stock ticker
  symbols" framing is *enhancement*, not greenfield.
* `RelatedEntity` table already populated with `PEER`, `SUPPLIER`,
  `CUSTOMER`, `ETF_HOLDING` from Finnhub `company_peers`,
  `stock_supply_chain`, `etfs_holdings`. Used by the Focus tab + sector
  synthesis fallback.
* **Missing** for the brief: `COMPETITOR` (distinct from PEER),
  `INSTITUTIONAL_HOLDER` (13-F data unused), `THEME` (Finnhub's
  `stock_investment_theme` unused), persisted user-curated watchlist.

Docs landed:

* **`security.md`** — first-pass security audit. 17 findings F-01..F-17
  with severity ratings (Critical / High / Medium / Low / Info), threat
  model (assets, actors, surfaces), hardening roadmap. `pip-audit`
  could not run in the sandbox (TLS verify blocked) — recorded as F-06.
* **`architecture.md`** — engineer's reference: module map with LOC
  per file, dependency-direction diagram, Mermaid diagram, three
  runtime flows (ingest / UI render / training), data contracts table,
  failure-mode catalogue, "where new features plug in" matrix.
* **`design.md`** — gap analysis (per-stock UI G-1..G-8) + affinity
  schema deltas + 8-PR implementation plan (PR-1 .. PR-8) + four
  open questions for sign-off.

Baseline test count confirmed: 472 passing in 41.4s. Operator approved
all four defaults (watchlists in same SQLite; auto-seed COMPETITOR
from PEER+industry; affinity blend off by default with learner fitting
weights; 10–15 themes pre-seeded).

---

## PR-1 — symbol-validation helper + UI wiring

*3 files, +377 / −2*

Closes security findings F-08 (input caps on user-supplied tickers)
and F-13 (symbol-shape validator before user-supplied tickers reach
the upstream Finnhub URL builder).

**New module** `finn_predictor/ingestion/symbols.py`:

* `valid_ticker(symbol)` — `re.fullmatch` against `[A-Z0-9.^-]{1,16}`.
  The `fullmatch` is deliberate: Python's `$` anchor matches before a
  trailing newline by default, so `"AAPL\n"` would have slipped past a
  naive `re.match(r"^...$")`. The comment in source warns the next
  reader.
* `parse_ticker_list(raw, *, max_count=50, max_input_len=8192)`
  returning a frozen `ParseResult` dataclass with `valid`, `rejected`
  (per-token reason: `REJECT_EMPTY` / `REJECT_TOO_LONG` /
  `REJECT_BAD_CHARS` / `REJECT_DUPLICATE`), and `truncated`. Caps
  the input at 50 tickers and 8 KiB.

**UI wiring**: `_parse_symbols` in `ui/app.py` now delegates to
`parse_ticker_list().valid` — legacy `list[str]` return shape
preserved so all existing callsites keep working.

Tests: +45 (new `tests/test_symbols.py`), 100% line+branch coverage on
the new module. Net: 472 → 517 passing.

---

## PR-2 — Watchlist + WatchlistMember models + sidebar UI

*5 files, +811 / −5*

Persisted, named ticker lists become the canonical source of truth
(replacing the comma-separated sidebar textbox). Foundation that PR-3
(streaks) and PR-7 (affinity blending per target list) build on.

**Schema** (new tables, additive — `init_db()`'s `create_all` picks
them up on next start, no migration needed):

* `Watchlist(id, name unique, description, created_at, updated_at)`.
* `WatchlistMember(id, watchlist_id FK ON DELETE CASCADE, symbol,
  notes, added_at)` with `(watchlist_id, symbol)` UNIQUE and indexes
  for both watchlist-side and symbol-side lookups.

**Repo helpers** in `storage/repo.py`: `create_watchlist`,
`get_watchlist`, `list_watchlists`, `rename_watchlist`,
`update_watchlist_description`, `delete_watchlist`, `add_to_watchlist`
(idempotent — refreshes notes on re-add), `remove_from_watchlist`,
`watchlist_members`, `watchlist_symbols` (union across all lists when
name is None), `replace_watchlist_symbols` (atomic — validates every
symbol before any write). Plus `WatchlistError(ValueError)` and
shape-checked name / symbol normalisers.

**UI helpers** in `ui/app.py`: `_format_parser_warning(ParseResult)`
builds the sidebar caption surfacing PR-1's rejected / truncated
output; `_resolve_active_tickers(session, ...)` decides whether the
ingestion pipeline acts on a saved list or the textbox content.

**Sidebar widget**: *Watchlists* expander — selectbox of saved lists
+ manual-textbox option, "Save as new list" form, "Delete <name>"
button. When a list is selected, the ticker text input pre-populates
with its symbols via `st.session_state`.

Tests: +77 (`tests/test_watchlists.py` 58 + `tests/test_ui_watchlist_helpers.py`
19). Net: 517 → 594 passing.

**Two SQLAlchemy / SQLite gotchas surfaced** (and documented inline
for the next reader):

1. SQLAlchemy's unit-of-work batches INSERTs ahead of DELETEs on the
   same table — `replace_watchlist_symbols` tripped the
   `(watchlist_id, symbol)` UNIQUE constraint when old and new sets
   overlapped. Fix: explicit `session.flush()` between the delete loop
   and the insert loop.
2. SQLite strips tz from `DateTime(timezone=True)` columns on read-back,
   so comparing the in-memory tz-aware value against the
   post-`session.refresh` naive value raises `TypeError`. Tests use a
   `_to_utc(d)` helper that normalises both sides.

Security: PR-1's `valid_ticker` is now also enforced inside the repo
write paths — a future CLI / programmatic caller bypassing the UI
parser still can't corrupt the table.

---

## PR-3 — Streak + Flipped + Band columns + CSV export

*4 files, +468 / −10*

Per-stock table on the Today tab gains three new columns + a
single-button CSV download.

**New repo helper** in `storage/repo.py`:
`streaks_for(session, symbols, *, model_version=None) -> dict[str, StreakInfo]`.
One SQL pass selecting `(target_symbol, label, prediction_date)` for
the requested symbols, ordered by `(target_symbol, prediction_date
DESC)`; Python walks the rows once, breaks on first disagreement per
symbol, emits one `StreakInfo` per symbol with history. Symbols with
no rows are *omitted* (callers can use `streaks.get(sym)` safely).

**`StreakInfo` frozen dataclass**: `symbol`, `current_label`,
`streak (≥1)`, `previous_label (Optional[str])`, `flipped (bool)`.
Documented "yesterday" semantics: it's the prior prediction by date,
not literally calendar-yesterday.

**`stock_predictions_table` extended** with an optional `streaks=`
parameter (computed via the DB if absent) + three new columns:
**Streak** (int, ≥1), **Flipped** (bool), **Band** (pre-formatted
string from `format_expected_move` or empty).

**`predictions_csv_bytes(df) -> bytes`** new pure helper for the
`st.download_button` data argument. Empty frame → header-only CSV.
Round-trips via pandas.

**UI integration**: Today tab computes streaks once per render
(passed to every per-section table builder + the combined-CSV
builder), Streamlit `column_config` entries for the new columns
(NumberColumn / CheckboxColumn / TextColumn), and a
*⬇ Download per-stock predictions as CSV* button under the
per-stock area. Filename: `finn-predictor-stocks-<today_iso>.csv`.

Tests: +21 (`tests/test_streaks.py` 16 + 5 in `tests/test_ui_helpers.py`).
Net: 594 → 615 passing.

**One bug surfaced** (and documented inline): pandas wraps Python
bools as `numpy.bool_`, so `df.iloc[0]["Flipped"] is False` is always
False. Switched test assertions to `bool(cell) is False`.

**One dead branch trimmed**: the first cut of `streaks_for` had a
defensive `if not labels: continue` that turned out to be unreachable
(`grouped[sym]` only exists when at least one append ran). Removed
with a comment explaining why.

---

## PR-4 — COMPETITOR relationship + auto-seed + curation

*7 files, +1004 / −15*

**New module** `finn_predictor/predictor/affinity.py` — this PR adds
the COMPETITOR machinery; PR-5 and PR-6 will extend this module with
institutional-holder and theme ingest respectively.

* `refresh_competitors(session, gateway, *, symbol)` — walks each
  PEER row for the target, fetches `company_profile2(peer)`, writes a
  COMPETITOR row when ``finnhubIndustry`` matches. PEER rows are
  preserved (curation is additive). Idempotent re-run; per-peer
  failure isolated; target without industry returns a no-op result
  with an informational failure entry.
* `promote_peer_to_competitor(session, *, symbol, peer_symbol)` and
  `demote_competitor(session, *, symbol, peer_symbol)` for manual
  curation. Both validate symbols via PR-1's `valid_ticker`.
* `CompetitorRefreshResult` dataclass with `peers_considered`,
  `competitors_added`, `skipped_no_industry`, `failures`.

**Schema**: `RELATIONSHIPS` frozenset in `storage/repo.py` extended
with `COMPETITOR`. `RelatedEntity` model docstring updated to document
the expanded allowed set (forward-references PR-5 + PR-6 additions
too).

**Focus tab**: `CompanyFocus` gains `competitors: list[RelatedPrediction]`;
`compose_company_focus` populates it via
`related_entities_for(..., relationship="COMPETITOR")`. The article
universe expansion now includes competitor symbols too so the
recent-articles list isn't blind to competitor news.

**CLI**: three new subcommands —
`promote-competitor SYMBOL PEER_SYMBOL`,
`demote-competitor SYMBOL PEER_SYMBOL` (idempotent — exit 0 with
`removed=false` when the row didn't exist),
`refresh-competitors --symbol SYMBOL [--symbol ...]`. The refresh
subcommand requires `FINNHUB_API_KEY` and exits 2 when missing.

Tests: +34 (`tests/test_affinity_competitors.py` 27 + 7 new in
`tests/test_cli.py`). Net: 615 → 649 passing.

---

## PR-5 — Institutional holders ingest (13-F)

*4 files, +454 / −1*

**Gateway**: new method `institutional_ownership(symbol, _from, to)`
in `ingestion/client.py` — passes empty cusip to the upstream client
(symbol-only lookup; cusip is optional on `/institutional/ownership`).

**Ingest**: `refresh_institutional_holders(session, gateway, *, symbol,
lookback_days=180, limit=25, today=None)` in `predictor/affinity.py`.
Resilient to gating (403 captured into `failures`, never raised).
Pulls top N holders by Finnhub's natural sort order (descending by
share value); persists one `RelatedEntity(relationship="INSTITUTIONAL_HOLDER")`
per holder.

Persistence design:

* `related_symbol` is the institution name (uppercased + collapsed
  whitespace + truncated to fit the 64-char column). The same filer
  reporting under "Vanguard Group, Inc." one quarter and "VANGUARD
  GROUP INC" the next collapses to one row.
* `rank` is the position in Finnhub's response (0 = biggest holder)
  so `related_entities_for`'s ascending-by-rank sort naturally surfaces
  the largest holder first.
* `metadata_text` carries the JSON-encoded ownership %, share count,
  value, and filing date so the UI can sort/display without re-parsing.

Helpers: `_normalise_institution_name`, `_coerce_float` (Finnhub
occasionally returns numeric fields as strings),
`_parse_institutional_payload` (non-dict / missing-data → empty
defensive path).

**Schema**: `RELATIONSHIPS` extended with `INSTITUTIONAL_HOLDER`.
`InstitutionalHoldersResult` dataclass with `holders_added`,
`holders_refreshed`, `failures`.

**Focus tab**: `CompanyFocus.institutional_holders` field populated by
`compose_company_focus`.

Tests: +18 (new `tests/test_affinity_institutional.py`).
Net: 649 → 667 passing.

---

## PR-6 — InvestmentTheme + ingest + themes predictor

*7 files, +1185 / −5*

**Design decision recorded**: themes do **not** write to the
`Prediction` table because theme codes can exceed the 16-char
`Prediction.target_symbol` limit (`financialExchangesData` is 22
chars). Instead, `predict_theme` returns a `ThemePrediction` dataclass;
the UI renders it from a fresh aggregation each pageload. Schema
widening to `String(64)` is filed as a follow-up.

**New table** `InvestmentTheme(id, theme_code unique, name, description,
fetched_at)` — mirrors `Sector` semantically.

**New module** `finn_predictor/predictor/themes.py`:

* `DEFAULT_THEME_CODES` — 12 curated Finnhub theme codes
  (financialExchangesData, cyberSecurity, cleanEnergy, electricVehicles,
  aiSemis, cloudComputing, robotics, spaceExploration, digitalPayments,
  nuclearEnergy, semiconductor, futureMobility).
* `ThemePrediction` frozen dataclass: theme_code, theme_name,
  constituent_count, label, confidence, sentiment_index, article_count,
  model_version, prediction_date.
* `predict_theme(session, *, scorer, theme_code, ...)` — mirror of
  `predict_sector` but for THEME_MEMBER constituents. Z-score classifier
  against a rolling baseline of theme indices. Returns None when no
  members or no scored articles.
* `predict_all_themes(session, *, scorer, ...)` — iterates registered
  themes, patches in operator-friendly names from `InvestmentTheme`,
  sorts by descending confidence. Per-theme failure isolation in the
  loop.

**Ingest extension** in `predictor/affinity.py`:

* `refresh_investment_themes(session, gateway, *, theme_codes=None)`
  — defaults to `DEFAULT_THEME_CODES` when no codes supplied. Per-theme
  failure isolation; idempotent re-run; dedupes within one response.
* `add_investment_theme` for the operator-curated path (no API call).
* `_humanise_theme_code` — `cyberSecurity` → `Cyber Security` for display.
* `_parse_theme_payload` — defensive against non-dict / wrong-shape
  payloads.

**Direction inversion documented**: THEME_MEMBER rows store
`source_symbol = theme_code` and `related_symbol = ticker` (opposite
from peer/supplier rows) so the "constituents of theme X" query is one
index hit.

**Schema**: `RELATIONSHIPS` extended with `THEME_MEMBER`.
`Gateway.stock_investment_theme(theme)` added.

**CLI**: `add-theme <code> [--name NAME] [--description DESC]` and
`refresh-themes [--theme CODE ...]`. Also added `from sqlalchemy import
select` at module top of `cli.py` (was previously runtime function-local
elsewhere).

Tests: +31 (new `tests/test_themes.py`). Net: 667 → 698 passing.

**One bug surfaced** (and documented inline): the first cut of
`predict_theme` called `rolling_baseline(daily_means)` — wrong API
(that function walks the DB itself and is scoped to one symbol).
Replaced with `aggregate_sentiment(daily_means)` + manual z-score
computation with a floor on the denominator so day-1 themes don't pin
confidence at 1.0.

---

## PR-7 — Affinity-blended per-stock predictor

*2 files, +759 / 0*

The headline feature of the wave. **Off by default per the locked
decision.**

**New module** `finn_predictor/predictor/blended.py`:

* `AffinityWeights` frozen dataclass with curated defaults — SELF=1.0,
  PEER=0.10, COMPETITOR=−0.30, SUPPLIER=0.15, CUSTOMER=0.25,
  THEME_MEMBER=0.10, INSTITUTIONAL_HOLDER=0.0 (no article corpus
  matches an institution name, but the field exists for PR-8 fitting).
* `DEFAULT_AFFINITY_WEIGHTS` constant.
* `predict_stock_blended(session, *, scorer, symbol, on_date,
  affinity_weights, ...)` — pools SELF + each related entity's
  same-day scored articles, multiplies each article's contribution by
  (relationship_weight × recency_weight), classifies the weighted
  index against the target's own rolling baseline of *blended* indices.
* `predict_all_stocks_blended` — list iteration with skip-on-blank.
* `AffinityContribution` dataclass for the per-relationship breakdown
  (surfaces in the UI popover).
* `explain_blend` — pure read; computes contributions without
  persisting. Suitable for popover handlers.
* `_build_blended_pool` / `_blended_index_for_day` /
  `_related_symbols` / `_theme_co_members` private helpers.

**Aggregation math**: for each relationship kind, articles contribute
`(score × |weight| × recency_weight)` to the numerator and
`|weight| × recency_weight` to the denominator. Negative weights
(COMPETITOR) preserve sign on contribution but not on the magnitude
denominator. The result is a signed weighted mean bounded roughly in
[−1, +1].

**Persistence**: `Prediction.model_version = f"{base_model}+aff:default"`
so blended and unblended calls coexist as two separate rows under the
existing `(target_symbol, prediction_date, model_version)` uniqueness
constraint.

**THEME_MEMBER co-membership**: `_theme_co_members(symbol)` does the
inverse lookup (themes containing the target → all other constituents)
and excludes the target itself so its own articles aren't
double-counted via the theme path.

Tests: +23 (new `tests/test_affinity_blend.py`), **100% line+branch
coverage on `blended.py`**. Net: 698 → 721 passing.

**Test-helper bug surfaced** (documented inline): the `_seed` factory
in the new test file re-scored already-scored articles when called
multiple times for the same symbol — UNIQUE constraint on
`sentiment_scores.(article_id, model_version)`. Fixed by looking up
only the just-inserted finnhub_ids rather than a tail-slice of all
articles for that symbol. Plus one test used the same
`finnhub_id_start` across loop iterations — also fixed by including
the loop index in the start offset.

---

## PR-8 — AFFINITY_WEIGHT learning dimension (persistence + read hooks)

*3 files, +298 / −7*

**Hooks shipped; `gp_minimize` integration deferred** to a follow-up
since a blended objective requires substantial refactoring of
`learning/simulate.py`. What landed:

**New dimension** `DIM_AFFINITY_WEIGHT = "AFFINITY_WEIGHT"` registered
in `learning.config.DIMENSIONS`. One `LearnedWeight` row per
relationship (key = relationship string, value = float).

**`LearnedConfig` extended** in `learning/config.py`:

* New field `affinity_weights: dict[str, float]`.
* New method `to_affinity_weights() -> AffinityWeights` — lazily
  imports `AffinityWeights` to avoid an import cycle; missing keys
  fall back to the curated default per-field rather than zeroing out.
* `_config_from_rows` extended to load `DIM_AFFINITY_WEIGHT` rows keyed
  by relationship.

**`_persist_weights` extended** in `learning/train.py`:

* New helper `_curated_affinity_baseline()` returns the curated
  defaults as a `dict[relationship, weight]`.
* New optional `affinity_weights=` argument; when None, persists the
  curated baseline (so every fresh training run carries the curated
  values as v1). When supplied, persists only those values.

**Read path**: `weights_for_version` and `active_weights` already
walk `_config_from_rows` so they automatically round-trip the new
dimension. The blended predictor can call
`active_weights(session).to_affinity_weights()` to materialise an
`AffinityWeights` for use at call time.

Tests: +13 (new `tests/test_affinity_learning.py`). Net: 721 → 734
passing.

**Out of scope** (filed as PR-8-follow-up): wiring `gp_minimize` over
the new dimensions inside a blended objective in
`learning/simulate.py`. The persistence + read machinery shipped here
is the foundation; the optimiser loop becomes a small extension once
the blended objective is in place.

---

## e1d56e4 — Merge pull request #1 from petsan/finn-predictor

*Merge commit on `petsan/master`, 2026-05-21T15:25:36Z.*

The full Finn-Predictor history (35 commits, c4864df → f37ea4c)
landed on `petsan/master` via the GitHub UI merge-commit strategy
(not squash — the per-commit history is preserved for archaeology).
Base: `petsan/finnhub-python:master`. **Not** merged to the upstream
`Finnhub-Stock-API/finnhub-python` — that would require a separate
paring-down pass (upstream maintainers may not want the whole
application living in the client repo).

Post-merge local sync:

* Fast-forwarded local `master` from `c94e7d4` (release 2.4.28 tip)
  to `e1d56e4`. 105 files / 31 937 insertions in one ff — no
  conflicts. Local `finn-predictor` branch deleted.
* Switched local `master`'s upstream from `origin/master` (upstream
  Finnhub-Stock-API) to `petsan/master` (this fork) so `git pull` /
  `git push` default to the fork.

GitHub flagged one moderate Dependabot vulnerability on
`petsan/master` at push time (alert #1). Not introduced by this
session; worth inspecting alongside F-06 (`pip-audit` in CI).

No application code changed beyond what was already in the merged
PR; this entry exists to record the merge itself + the local clone
state transition.

---

## 2026-05-21 — docs sweep: bring progress.md / summary.md / diff.md current

*3 files updated, no code changed.*

* `progress.md` gains a `## Session log` section with one dated entry
  per turn of the collaborative session — discovery, decisions locked,
  then one entry per PR landed.
* `summary.md` test count bumped to 734; full PR-1 → PR-8 changelog
  blocks added; pointers to the three new reference docs
  (`architecture.md`, `security.md`, `design.md`).
* `diff.md` (this file) gains a kickoff entry + eight PR entries
  (PR-1 → PR-8) + this docs-sweep entry; branch totals at the top
  updated.

Coverage at the close of the session: **96% line+branch maintained**
across the package, **734 tests passing in 65.2s**.

---

## Files added (by directory)

```
finn_predictor/
  __init__.py
  config.py · cli.py · logging_config.py · security.py
  ingestion/
    __init__.py · backfill.py · client.py · jobs.py · news.py ·
    prices.py · prices_yf.py ·
    symbols.py            (NEW — PR-1)
  learning/
    __init__.py · config.py · simulate.py · train.py
  predictor/
    __init__.py · aggregate.py · backtest.py · classifier.py ·
    explain.py · focus.py · magnitude.py · market.py · sectors.py ·
    stocks.py · trades.py ·
    affinity.py           (NEW — PR-4 + extended in PR-5, PR-6)
    themes.py             (NEW — PR-6)
    blended.py            (NEW — PR-7)
  sentiment/
    __init__.py · base.py · finbert.py · vader.py
  storage/
    __init__.py · clustering.py · engine.py · models.py · repo.py ·
    sector_membership.py · stories.py · symbol_names.py
  ui/
    __init__.py · app.py
deploy/
  proxmox/install.sh

tests/
  __init__.py · conftest.py · test_aggregate.py · test_backfill.py ·
  test_backtest.py · test_classifier.py · test_cli.py ·
  test_clustering.py · test_config.py · test_engine_migrations.py ·
  test_explain.py · test_focus.py · test_ingestion_client.py ·
  test_ingestion_news_prices.py · test_jobs.py · test_learning.py ·
  test_logging_config.py · test_magnitude.py · test_predictor_market.py ·
  test_predictor_sectors.py · test_predictor_stocks.py ·
  test_prices_yf.py · test_sector_membership.py · test_security.py ·
  test_sentiment.py · test_storage_repo.py · test_stories.py ·
  test_symbol_names.py · test_trades.py · test_ui_helpers.py ·
  test_ui_smoke.py ·
  test_symbols.py                  (NEW — PR-1)
  test_watchlists.py               (NEW — PR-2)
  test_ui_watchlist_helpers.py     (NEW — PR-2)
  test_streaks.py                  (NEW — PR-3)
  test_affinity_competitors.py     (NEW — PR-4)
  test_affinity_institutional.py   (NEW — PR-5)
  test_themes.py                   (NEW — PR-6)
  test_affinity_blend.py           (NEW — PR-7)
  test_affinity_learning.py        (NEW — PR-8)

progress.md · summary.md · diff.md · user-manual.md ·
installation-manual.md · deployment-manual.md · pytest.ini ·
.coveragerc · requirements.txt · Dockerfile · docker-compose.yml ·
docker/entrypoint.sh · .dockerignore ·
architecture.md       (NEW — 2026-05-21 kickoff)
security.md           (NEW — 2026-05-21 kickoff)
design.md             (NEW — 2026-05-21 kickoff)
```

## Files in the upstream library that were NOT touched

The original `finnhub-python` client is unchanged. We catalogued it in
§1 of `progress.md` and import it as a dependency.

```
finnhub/__init__.py · finnhub/client.py · finnhub/exceptions.py
setup.py · setup.cfg · test-requirements.txt · tox.ini · examples.py ·
README.md · CHANGELOG.md · LICENSE
```

`requirements.txt` at the repo root is **new** — it lists this app's
deps, separate from the upstream library's pure-library
`setup.py:REQUIRES`. The only other edit outside `finn_predictor/` was
an addition to `.gitignore` (`finn_predictor.db`, `*.sqlite`,
`*.sqlite3`).

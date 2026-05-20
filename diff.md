# Finn-Predictor — Change Log (branch `finn-predictor` vs. `master`)

Every commit on the `finn-predictor` branch is described below in
chronological (oldest → newest) order. The upstream `finnhub-python`
library itself (`finnhub/`, `setup.py`, etc.) is **untouched**; everything
new lives under `finn_predictor/`, `tests/`, and a few config files.

Totals across the branch: **75 files changed, ~15,800 insertions(+),
3 deletions(-)** vs. `master` (commit `c94e7d4 release 2.4.28`).

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

## Files added (by directory)

```
finn_predictor/
  __init__.py
  config.py · cli.py · logging_config.py · security.py
  ingestion/
    __init__.py · backfill.py · client.py · jobs.py · news.py ·
    prices.py · prices_yf.py
  learning/
    __init__.py · config.py · simulate.py · train.py
  predictor/
    __init__.py · aggregate.py · backtest.py · explain.py · focus.py ·
    market.py · sectors.py · stocks.py · trades.py
  sentiment/
    __init__.py · base.py · finbert.py · vader.py
  storage/
    __init__.py · engine.py · models.py · repo.py · stories.py ·
    symbol_names.py
  ui/
    __init__.py · app.py

tests/
  __init__.py · conftest.py · test_aggregate.py · test_backfill.py ·
  test_backtest.py · test_cli.py · test_config.py · test_explain.py ·
  test_focus.py · test_ingestion_client.py · test_ingestion_news_prices.py ·
  test_jobs.py · test_learning.py · test_logging_config.py ·
  test_predictor_market.py · test_predictor_sectors.py ·
  test_predictor_stocks.py · test_prices_yf.py · test_security.py ·
  test_sentiment.py · test_storage_repo.py · test_stories.py ·
  test_symbol_names.py · test_trades.py · test_ui_helpers.py

progress.md · summary.md · diff.md · user-manual.md ·
installation-manual.md · pytest.ini · .coveragerc · requirements.txt ·
Dockerfile · docker-compose.yml · docker/entrypoint.sh · .dockerignore
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

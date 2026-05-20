# Finn-Predictor — User Manual

A walkthrough of how to use the dashboard, in roughly the order a new
user encounters it. Pairs with `installation-manual.md` (which covers
getting the app running in the first place).

---

## 1. What the app does

Finn-Predictor pulls news from Finnhub, scores each article's sentiment
with VADER, and aggregates the day's signal into a directional **Call**
— **UP**, **DOWN**, or **FLAT** — for each of:

* the whole market (the S&P 500 index `^GSPC`),
* the eleven Sector SPDR ETFs (`XLK`, `XLE`, `XLF`, `XLV`, `XLY`,
  `XLP`, `XLI`, `XLB`, `XLU`, `XLRE`, `XLC`),
* every individual stock ticker you ask it to follow.

Each Call carries a **confidence** number (0 = no signal, 1 = the
classifier's maximum). The classifier is a rule, not a fitted model —
read paragraph 5 of any "Why this Call?" explanation for the full
list of caveats.

**This is not investment advice.** The dashboard is a market-mood gauge
that you can train against its own historical hit-rate. Predictions are
informational; the UI never places trades and there is no broker
integration.

---

## 2. The sidebar

Everything that talks to Finnhub starts in the sidebar.

### 2.1 Finnhub API key

Paste your key (get one at <https://finnhub.io/dashboard>) into the
**API key** field. The key is masked while typing.

**It lives only in this browser tab's server-side session.** It is
never written to disk, never written to the SQLite database, never
written to logs, and never returned in error messages — three
independent scrubbing layers strip it from any exception text before
display. Closing the tab or restarting the server discards it. There
is a *Clear key* button if you want to nuke it explicitly.

The application boots and renders the UI **without** a key — every tab
is read-only against the existing data. You only need a key to fetch
new data.

### 2.2 Run ingestion now

Fetches today's general market news + market & sector ETF prices, plus
per-company news for any tickers you list in the *Company tickers*
field (comma-separated). Scores any new articles, then writes one
Prediction per UTC day per target.

What happens depends on what worked:

| Sidebar status | Meaning |
|---|---|
| 🟢 Green "Done — articles +N, scored M, predictions P" | Everything that should have worked, did. |
| 🟢 Green + amber expander "⚠ N endpoint(s) failed" | News fetched fine; some price calls were blocked. Almost always means your Finnhub plan doesn't include `/stock/candle` — news still ingests, predictions still write, you just won't have prices to backtest against. |
| 🔴 Red "News fetch (`general_news`) failed: …" | The `/news` call itself was rejected. Likely an invalid/rotated key, a plan that doesn't include `/news`, or a daily quota that's been hit. |

### 2.3 Backfill historical news

Pulls `/company-news` for the tickers in *Tickers to backfill (CSV)*
over the lookback window (slider, 7–365 days). Free-tier Finnhub keys
typically allow ~1 year. Useful for bootstrapping the rolling baseline:
without history, the per-stock predictor's confidence number can pin at
1.00 on day 1.

For each ticker the run reports articles inserted, chunks ok, and the
number of historical predictions written.

### 2.4 Backfill prices (yfinance)

Because Finnhub's `/stock/candle` is paid-tier, the dashboard talks to
Yahoo Finance directly for price data. **No API key needed.** This is
what fills the *Performance* tab.

The button collects every prediction target already in the DB (the
market, sectors, and your individual stocks), pulls daily OHLC over the
chosen lookback window, then runs the backtester so any open
predictions whose next-session close is now available get scored.

---

## 3. The five tabs

### 3.1 Today

The headline view.

* **Top tiles** — the market Call, its confidence, and how many
  articles fed it.
* **Why this Call?** — a scrollable block with five paragraphs per
  prediction (the market call plus any sector calls): call summary,
  mechanics (z-score against the 30-day rolling baseline), top movers,
  counter-signal, and caveats.
* **Per-stock predictions** — a compact table of every individual
  ticker that has a prediction today, with company name expanded, a
  confidence progress bar, articles count, and signed sentiment. Sorted
  by confidence.
* **Per-article contribution chart** — divergent vertical bars
  (Altair). The X-axis is article index sorted ascending by signed
  contribution (left = most-negative, right = most-positive), Y-axis
  is signed contribution centered on a 0-line, colour is sign. Hover
  for headline, source, ticker, expanded company name, published time,
  first-reported time, sentiment, and signed contribution.
* **Recent headlines** — clickable bullet list. When the market call
  exists, headlines are sorted by `|contribution|` descending; each
  row leads with 🟢 if the article supports the Call and 🔴 if it
  opposes. The row also shows source, expanded company, published
  time, "first reported" timestamp if the same story appeared earlier
  elsewhere, sentiment score, and signed contribution.

### 3.2 History

Each row is one prediction with its outcome (when available). Use it to
audit how the model behaved over time. The X-axis of any plot here is
the prediction date.

### 3.3 Sectors

One row per sector ETF, with its current Call and the underlying ticker.
Click into a sector via *Focus* if you want details.

### 3.4 Performance

The "is it any good?" view, driven by the hypothetical-trade ledger.

How a paper trade is built from a Prediction:

| Call | Direction | Entry | Exit | PnL formula |
|---|---|---|---|---|
| **UP** | LONG | prediction-day close | next session close | `+realised_return` |
| **DOWN** | SHORT | prediction-day close | next session close | `-realised_return` |
| **FLAT** | none | — | — | 0 (still counted in hit-rate via the `|return|<0.25%` rule) |

What you see:

* **Summary metrics** — Predictions, Closed trades, Hit rate,
  Cumulative PnL, Avg PnL/trade. Plus a second row with Best trade,
  Worst trade, Open positions.
* **Cumulative PnL over time** — line chart of the running sum of
  signed returns.
* **Rolling 14-trade hit-rate** — line chart with a dashed 50%
  reference rule.
* **Hit rate by target type** — bar chart split by MARKET / SECTOR /
  STOCK.
* **Hit rate by Call** — bar chart split by UP / FLAT / DOWN.
* **Trade ledger** — every paper trade, sortable.

Caveat: no friction is modelled. No commissions, no spread, no
slippage. Real-world PnL would be strictly lower.

### 3.5 Focus

Drill into a single subject. Pick the mode at the top:

**Company** — pick a ticker. Shows the subject's Call, the sector it
belongs to (and that sector's Call), and tables for:

* **Peers** — Finnhub's `company_peers` list, with each peer's own
  Call if we have one.
* **Suppliers / Customers** — only populated if your Finnhub plan
  supports `/stock/supply-chain`. A 403 here is captured silently;
  the peer view still works.
* **Recent articles** — across the subject and its peers.

The **Refresh peers + supply chain** button calls Finnhub once to
update the cached relationships. Needs the API key.

**Sector** — pick a sector ETF. Shows the ETF's Call, top
constituents from `/etf/holdings`, and recent articles across the
constituent universe.

**Event** — free-text search. Type something like `Iran war` or
`Fed rate cut`, set the lookback in days, and the dashboard finds
every article whose headline or summary matches (case-insensitive,
all terms must appear). The aggregate sentiment yields an *implied
Call* using a sign-of-mean classifier (±0.1 dead-band; confidence is
`|mean|/0.5` clipped to 1). The view includes a per-ticker breakdown
of the matches.

### 3.6 Learning

This is where the model self-improves.

**Activation policy** (radio at the top):

* **Auto-activate — newest version wins** — after training, the new
  version goes live automatically, gated by holdout improvement (see
  next).
* **Manual approval — leave new version inactive** — training writes
  the new version but it stays inactive. You decide whether to flip
  it.

**Holdout-gate tolerance** (slider, 0.0–0.10) — only meaningful under
AUTO. After fitting a candidate, the system re-scores the *currently
active* config on the same 14-day holdout window. If the new
candidate's holdout score is below `(active - tolerance)`, the gate
**blocks** auto-activation. The version is still saved; there's an
"Activate v\<n\> anyway" override button so you can flip it manually
if you disagree. Default tolerance is `0.01` objective units (≈ a
1-percentage-point regression on the blended `hit_rate + 0.5 *
cum_pnl` objective).

**Active weights** — four tiles: version, threshold σ, min baseline
σ, half-life hours. Plus a collapsed expander listing per-source
weight multipliers (Reuters / Bloomberg / etc., learned by
hit-rate ratio).

**Train a new version** — click *Retrain now* and Bayesian
optimisation searches over `threshold_sigma`, `min_baseline_sigma`,
and `half_life_hours` for the parameter set that maximises
`hit_rate + 0.5 * cumulative_PnL` on the historical trade ledger.
You can change the iteration budget (10–80, default 30).

The post-training feedback states whether the new version was
**activated**, **saved but not activated**, or **gate blocked**, and
includes the gap to the previous holdout score.

**Version history** — every trained version with its scores. Inactive
rows get an "Activate v\<n\>" button so you can revert at any time.

---

## 4. Recommended first-run flow

1. Paste your Finnhub key in the sidebar.
2. Type a small list of company tickers (e.g. `AAPL, MSFT, NVDA, JPM,
   XOM`) in the *Company tickers* field.
3. Click **Run ingestion now**. Confirm the green success block.
4. Click **Backfill historical news** with the same tickers and a
   30–90 day lookback. Watch the per-ticker counts.
5. Click **Backfill prices (yfinance)**. The Performance tab starts
   populating.
6. Open the **Today** tab. Read the explanation paragraph for the
   market call. Hover the contribution chart.
7. Open the **Focus** tab → Company mode. Pick `AAPL`, click
   *Refresh peers + supply chain*. See peers' Calls appear.
8. Open the **Performance** tab. Inspect the rolling hit-rate.
9. Open the **Learning** tab. Once you have ≥10 closed trades, click
   *Retrain now*. Watch whether the gate blocks or activates the new
   version.

---

## 5. Limits of what the dashboard can know

* **No history for the general market feed.** Finnhub's `/news`
  endpoint only paginates forward. The market call's 30-day rolling
  baseline can only build up over wall-clock time. Per-stock and
  per-sector baselines *can* be backfilled via `/company-news`.
* **Per-stock day-1 over-confidence.** Without backfill, a newly added
  ticker has σ at the floor (`min_baseline_sigma`), so modest
  sentiment can z-score above the threshold and pin confidence at 1.
* **VADER is general-purpose.** Finance-specific jargon ("beat
  estimates", "guidance lowered", "misses on top line") is
  under-weighted. The architecture supports swapping in FinBERT under
  the same `Scorer` interface, but the FinBERT path is gated on
  installing `torch` + `transformers` (kept out of the default
  dependency tree). To activate it: `pip install torch transformers`
  then set `FINN_PREDICTOR_SCORER=finbert` before launching the UI or
  the `ingest` CLI. The training loop picks up the new
  `model_version` automatically — retrain once to fit weights against
  the new scorer.
* **Calls are sentiment-only.** The model ignores price action, the
  earnings calendar, macro releases, and microstructure.
* **Backtest accuracy is the model's, not yours.** No friction
  modelled. Hit-rate and Cumulative PnL on the Performance tab are
  the upper bound of "what a frictionless system would have done";
  add any realistic spread/commission/slippage and the curve sags.
* **Story clustering defaults to a heuristic.** The "first reported"
  timestamp is computed via an 8-word headline prefix match. Wires that
  paraphrase heavily won't cluster; unrelated stories with the same
  lead may cluster. For paraphrase-aware clustering, install
  `sentence-transformers` and set `FINN_PREDICTOR_CLUSTERER=embedding`
  — the column is then driven by cosine similarity on
  `all-MiniLM-L6-v2` embeddings (~80 MB on first download).

---

## 6. Troubleshooting

### Sidebar shows red "News fetch failed: FinnhubAPI 403"

Your key isn't permitted on the `/news` endpoint. Check
<https://finnhub.io/dashboard> — recently rotated key on a free-tier
account is the usual cause. Free tier *does* normally include `/news`;
if it doesn't, raise a support ticket with Finnhub.

### Sidebar shows green "Done — articles +0"

Finnhub had no new news in the feed window since your last run. Not an
error. Wait, then ingest again.

### Many sidebar amber entries `market_prices:^GSPC … 403`

You're on a Finnhub plan without `/stock/candle`. The dashboard now
gets prices via **yfinance** instead — click **Backfill prices
(yfinance)** to populate the Performance tab. The Finnhub price
attempts will keep 403'ing on every ingestion run; that's fine, they
don't block anything.

### Performance tab shows "No predictions yet" or "0 open trades"

You need predictions *with* outcomes. Predictions write whenever
articles ingest; outcomes write when *prices* land in the next
session. If you've never run *Backfill prices*, do that now.

### Trade ledger has many open trades, never closed

The next-session price bar hasn't been ingested yet. Re-run
*Backfill prices* with the same lookback. If the bar still doesn't
land, yfinance may have a hole for that specific date (rare; usually
ETFs/futures specific).

### Learning tab refuses to retrain — "not enough closed predictions"

You need at least 10 closed trades (Predictions with Outcomes) for
training to be meaningful. Run more ingestion + price backfill first.

### Learning tab shows "gate blocked"

The new candidate scored below the active version on the holdout
window. The new version is saved, but inactive. Three options:
- Click **Activate v\<n\> anyway** to override.
- Raise the **Holdout-gate tolerance** slider (above) and retrain.
- Leave the gate blocking — the existing weights will keep producing
  predictions until a better candidate beats the holdout.

### `Ingestion failed: HTTPSConnectionPool … SSL: CERTIFICATE_VERIFY_FAILED`

You have an intercepting HTTPS proxy set in the environment (Burp,
mitmproxy, Charles, Fiddler). The dashboard *already* bypasses
`HTTPS_PROXY`/`HTTP_PROXY` via `Session.trust_env = False` for its own
ingestion calls. If you still see this, your dev proxy's CA bundle is
the only intercepting layer that could cause it — check what's running
on port 8080 of your machine.

### "Every Finnhub call failed — first error: …"

Misleading if it shows up. The dashboard distinguishes two cases:

- A real total failure where `/news` itself returned 403 — red error
  with the op named.
- A partial-failure shape where `/news` returned an empty body and
  only the price calls 403'd — green success + amber expander.

If you see the red text, your key really lost access.

---

## 7. Where to go next

- **`installation-manual.md`** — getting a fresh instance running
  (venv or Docker), upgrading, resetting, backing up.
- **`summary.md`** — top-level architecture overview.
- **`progress.md`** — design decisions and implementation status.
- **`diff.md`** — per-commit change log on the `finn-predictor` branch.

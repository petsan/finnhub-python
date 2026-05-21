# Finn-Predictor — Design Doc: Per-Stock Tracking + Affinity & Themes

This is the design proposal that needs sign-off before implementation
begins. Two threads:

1. **Per-stock predictions + UI** (mostly built; small gaps).
2. **Affinity + theme tracking** (partly built as `RelatedEntity`;
   needs extension + a new tracked-entity user model).

For each section: **what exists today**, **what's missing**, and **the
proposed change** with file:line landings, schema diffs, and test
sketches.

---

## 1. Per-stock predictions + UI

### 1.1 What exists today

- `finn_predictor/predictor/stocks.py` — `predict_stock`,
  `predict_all_stocks`, `retroactive_predict_stock`,
  `retroactive_predict_many`. The per-stock call uses the same z-score
  classifier as the market call but scoped to `category="company"`
  + `article_symbol=<ticker>`.
- One `Prediction` row per `(target_symbol, prediction_date, model_version)`.
- UI sidebar accepts a comma-separated ticker list and feeds it to
  `predict_all_stocks` during ingestion.
- Today tab renders per-stock predictions sorted by confidence, with a
  `ProgressColumn` for the confidence bar.
- Tickers are sector-grouped on the Today tab via curated
  `sector_membership.TICKER_TO_SECTOR` overlaid with cached
  `ETF_HOLDING` rows.
- The Focus tab → Company mode pulls peers + supply chain for any
  ticker.

### 1.2 Gaps

| # | Gap | Why it matters |
|---|---|---|
| G-1 | Ticker list is not persisted across sessions | Operators retype it every time. localStorage hydrates the API key but not the ticker list. |
| G-2 | No first-class "watchlist" concept | The ticker list lives in a text box. There's no "named watchlist", no shared lists, no per-list ingestion cadence. |
| G-3 | No per-stock detail page | The Today tab summarises; the Focus tab drills down by ticker but doesn't show the rolling per-stock prediction history compactly. |
| G-4 | No flip-detection / alert primitives | A ticker that flipped UP→DOWN today gets the same visual weight as one that's been UP for 30 days. |
| G-5 | Magnitude band is per-prediction but not surfaced in the per-stock table | Today tab shows it only for the market call by default. |
| G-6 | `predict_all_stocks` iterates serially; no awareness of which tickers were "stale" (no fresh news since last run) | Wastes time on tickers with no input; can't say "the model declined to call X today because it had no fresh news." |
| G-7 | No CSV export of the per-stock table | Operators screen-cap or copy-paste. |
| G-8 | Input validation on the ticker list is loose | The current code strip+uppercases and accepts anything. Per security finding F-13. |

### 1.3 Proposal: persisted watchlists + per-stock detail enhancements

**Schema additions** (new tables — no migration on existing rows):

```python
class Watchlist(Base):
    """A named collection of tickers an operator wants tracked."""
    __tablename__ = "watchlists"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

class WatchlistMember(Base):
    """A (watchlist, symbol) edge. Symbols can belong to multiple lists."""
    __tablename__ = "watchlist_members"
    __table_args__ = (
        UniqueConstraint("watchlist_id", "symbol", name="uq_watchlist_members"),
        Index("ix_watchlist_members_watchlist", "watchlist_id"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    watchlist_id: Mapped[int] = mapped_column(
        ForeignKey("watchlists.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
```

**Why two tables (not just a CSV column on Watchlist):** lets us index
by symbol for "which lists contain GOOG?", supports per-member notes,
and the WatchlistMember.symbol column gets a natural join target for
the affinity graph (Section 2).

**Repo functions** (new in `storage/repo.py`):

- `create_watchlist(session, *, name, description=None) -> Watchlist`
- `delete_watchlist(session, *, name) -> bool`
- `add_to_watchlist(session, *, name, symbol, notes=None) -> WatchlistMember`
- `remove_from_watchlist(session, *, name, symbol) -> bool`
- `list_watchlists(session) -> list[Watchlist]`
- `watchlist_symbols(session, *, name=None) -> list[str]` — returns
  all symbols (deduped, sorted) across all lists if `name is None`,
  else just the named list.

**Symbol validation** (`finn_predictor/ingestion/symbols.py`, new):

- `valid_ticker(s: str) -> bool` matches `^[A-Z0-9.^-]{1,16}$`.
- `parse_ticker_list(raw: str, max_len: int = 50) -> list[str]` —
  strips, uppercases, dedupes, drops invalid, enforces the size cap
  (per F-08). Returns `(valid, rejected)` so the UI can show what was
  dropped and why.

**UI changes**:

- Sidebar adds a *Watchlist* expander with: a dropdown of saved lists,
  Save / Rename / Delete buttons, and a textarea (current behaviour
  for the unnamed default list).
- Today tab gets a *Watchlist* selector at the top — defaults to "All
  watched" (union of every list) but can be filtered to one list.
- Per-stock table adds two columns: **Flipped** (boolean — is today's
  label different from yesterday's?) and **Streak** (number of
  consecutive days at the current label). These come from a new
  `storage/repo.streaks_for(symbols)` query that does the lookup in
  one SQL pass.
- Per-stock table adds a **Band** column (low / median / high) when
  magnitude calibration is fitted.
- A *Download CSV* button at the bottom of the per-stock table.

**`ingestion/jobs.py` changes**:

- `run_daily_ingest` accepts an optional `watchlist_name` arg. When
  set, only fetches news + predicts for that list's symbols (default
  remains "all watchlists union" for backward compat with the current
  CSV-textbox path).

**Test plan** (must all be new tests in `tests/`):

- `tests/test_watchlists.py` — CRUD on Watchlist + WatchlistMember;
  uniqueness; cascade delete; `watchlist_symbols` union semantics.
- `tests/test_symbol_validation.py` — `parse_ticker_list` accepts good
  tickers, rejects garbage, enforces cap, normalizes case.
- `tests/test_streaks.py` — `streaks_for` returns correct streak
  lengths + flipped booleans on a synthetic prediction history.
- `tests/test_ui_helpers.py` (extend) — assert the per-stock table
  builder produces a DataFrame with the new columns and that the CSV
  bytes round-trip.

---

## 2. Affinity + theme tracking

### 2.1 What exists today

- `RelatedEntity` table with `relationship ∈ {PEER, SUPPLIER, CUSTOMER, ETF_HOLDING}`.
- `predictor/focus.refresh_company_relationships()` pulls peers (via
  `company_peers`) and supply-chain rows (via `stock_supply_chain`)
  for a given ticker.
- `predictor/focus.refresh_sector_constituents()` populates
  `ETF_HOLDING` rows for sector ETFs.
- The Focus tab uses these caches to render side-by-side prediction
  cards for `subject + its related entities`.
- `sectors.py` falls back to the cached `ETF_HOLDING` rows when
  `/etf/holdings` is gated.

### 2.2 Gaps

The user's brief calls for tracking by affinity — "competition,
supplier, controlling interest entity" — and themes. Mapping that to
what exists:

| User concept | Closest existing primitive | Gap |
|---|---|---|
| Competition | `PEER` (Finnhub peer set) | Peer ≠ competitor exactly; needs explicit `COMPETITOR` relationship that the user can curate per-ticker |
| Supplier | `SUPPLIER` ✅ | Exists; just not surfaced as a "track this affinity" UX |
| Customer | `CUSTOMER` ✅ | Same |
| Controlling interest entity | (nothing) | 13-F data (`institutional_ownership` / `institutional_portfolio` / `institutional_profile`) is available in the Finnhub client but unused |
| Theme | (nothing) | `stock_investment_theme` returns theme constituents; we don't ingest or surface them |
| User wants to *follow* a stock or theme | The sidebar textbox of tickers | No first-class "tracked entity" — see Section 1's Watchlist proposal |

### 2.3 Proposal: extend RelatedEntity + add InvestmentTheme + ThemeMember

**Schema additions / changes**:

1. **Extend `RelatedEntity.relationship` allowed values** to also
   include `COMPETITOR`, `INSTITUTIONAL_HOLDER`, and `THEME_MEMBER`.
   Backward-compatible: the column is `String(32)`, and existing rows
   keep their values. We document the new allowed set in the model
   docstring and enforce it in the `upsert_related_entity` helper (no
   DB-level CHECK constraint, to keep SQLite portability).

2. **New table `InvestmentTheme`** — one row per theme Finnhub returns
   from `/stock/investment-theme`:

   ```python
   class InvestmentTheme(Base):
       __tablename__ = "investment_themes"
       id: Mapped[int] = mapped_column(Integer, primary_key=True)
       theme_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
       name: Mapped[str] = mapped_column(String(128), nullable=False)
       description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
       fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
   ```

   The theme code is Finnhub's identifier (e.g. `financialExchangesData`).
   We treat themes like sectors: each theme can have a synthesized
   Prediction (target_symbol = `THEME:<code>`).

3. **Theme membership lives in `RelatedEntity` with relationship =
   `THEME_MEMBER`**, `source_symbol = theme_code`, `related_symbol =
   <ticker>`. No new table for the edge — reuses the existing
   constraint shape.

**New ingestion**:

- `ingestion/affinity.py` (new module):
  - `refresh_institutional_holders(gateway, symbol)` — calls
    `institutional_ownership(symbol, ...)` and writes
    `RelatedEntity(source_symbol=<ticker>, related_symbol=<institution_name_or_cik>, relationship="INSTITUTIONAL_HOLDER", rank=ownership_pct)`.
  - `refresh_investment_themes(gateway)` — calls
    `stock_investment_theme()` for the configured theme list (curated
    in `storage/sector_membership.py`-style constants because the API
    needs a theme code per call) and writes `InvestmentTheme` rows
    + `RelatedEntity(relationship="THEME_MEMBER")` edges.

**Curating competitors** (G-4 + the user's "competition" ask):

- The Finnhub peer set is a reasonable seed but operators want to
  curate it. New CLI: `python -m finn_predictor.cli set-competitor
  SYMBOL COMPETITOR_SYMBOL` (and `unset-competitor`). Writes/removes
  a `RelatedEntity(relationship="COMPETITOR")` row.
- UI Focus tab gets a *Competitors* section under the existing Peers
  section. Operator can "promote a peer to competitor" (delete the
  PEER row, write a COMPETITOR row).

**New predictor**: themes get the same treatment as sectors.

- `predictor/themes.py` (new module) — mirrors `predictor/sectors.py`:
  - `predict_theme(session, *, theme_code, scorer, ...)` aggregates
    the recency-weighted sentiment of every `THEME_MEMBER` ticker for
    `theme_code` and emits a `Prediction(target_symbol="THEME:<code>")`.
  - `predict_all_themes(session, *, scorer, ...)` iterates the
    `InvestmentTheme` table.

**Affinity-weighted per-stock prediction (the headline feature)**:

This is the new behaviour. Today, `predict_stock(symbol)` aggregates
sentiment only from articles with `symbol == <ticker>`. With affinity,
the call optionally blends in sentiment from related entities at
configurable weights.

```python
# Defaults — tunable via env + the Learning loop.
AFFINITY_WEIGHTS_DEFAULT = {
    "SELF":                  1.00,
    "COMPETITOR":            -0.30,  # competitor up ⇒ headwind for us
    "PEER":                   0.10,  # peer up ⇒ mild industry tailwind
    "SUPPLIER":               0.15,  # supplier up ⇒ supply chain ok
    "CUSTOMER":               0.25,  # customer up ⇒ demand outlook ok
    "INSTITUTIONAL_HOLDER":   0.05,  # 13F holder's other moves matter a little
    "THEME_MEMBER":           0.10,  # peer in a tracked theme
}
```

These are sign-aware. A *positive* sentiment headline about a
competitor contributes negatively to the target's sentiment_index; a
positive headline about a supplier contributes positively.

**Implementation site**: `predictor/aggregate.py` gains an
`affinity_blend` mode where instead of "all articles for one symbol",
it loads:

```
(articles where symbol = TARGET)            with weight  +1.0  (SELF)
(articles where symbol = competitor of TARGET) with weight -0.30
(articles where symbol = supplier of TARGET)   with weight +0.15
... etc
```

and aggregates the union, multiplying each article's contribution by
its affinity weight.

**Schema impact on `Prediction`**: none — the call is still one row
per `(target_symbol, prediction_date, model_version)`. We extend
`model_version` to encode the blend mode: `vader+aff:default` vs the
existing `vader`, so we can A/B blended vs unblended on the same
ticker and the upsert constraint cleanly keeps both.

**Self-improvement**: the affinity weight table becomes a new
`LearnedWeight` dimension `AFFINITY_WEIGHT` (key = relationship). The
training loop's Bayesian opt extends to fit them from closed
outcomes, same shape as the existing source-weight fit.

**UI surface**:

- Sidebar adds: *Affinity-blended predictions* checkbox (off by
  default until calibration shows a hit-rate improvement).
- Today tab: per-stock row gets an info icon — clicking opens a
  popover that breaks the prediction down by affinity (X% from
  SELF, Y% from competitors, etc.). Reuses the existing
  `article_contributions` mechanism, just colored by relationship.
- Focus tab → Company mode: new section "**Affinity graph**" — a
  small graph view (or two-column table fallback) showing the target
  + its competitors + suppliers + customers + top institutional
  holders, each with its own latest prediction.
- New tab: **Themes** — mirror of the Sectors tab, listing every
  tracked `InvestmentTheme` with its synthesized prediction and
  constituents.

**Test plan**:

- `tests/test_affinity_ingestion.py` — mock gateway returns canned
  `company_peers`, `stock_supply_chain`, `institutional_ownership`,
  `stock_investment_theme` payloads. Assert correct rows in
  `RelatedEntity` + `InvestmentTheme`.
- `tests/test_predictor_themes.py` — exercise `predict_theme` like
  the existing `test_predictor_sectors.py` does.
- `tests/test_affinity_blend.py` — given a synthetic article set
  across symbols + a `RelatedEntity` graph, assert the blended
  `sentiment_index` matches a hand-computed weighted sum.
- `tests/test_focus.py` (extend) — Focus tab pulls competitors and
  institutional holders correctly.
- `tests/test_ui_helpers.py` (extend) — affinity-popover data shape.
- `tests/test_learning.py` (extend) — Bayesian opt search space
  includes `AFFINITY_WEIGHT` dimension.

---

## 3. Suggested order

Sized so each is one merge-able PR with tests:

| PR | Scope | Deps |
|---|---|---|
| **PR-1** | Symbol validation helper (`parse_ticker_list`, `valid_ticker`) + plug into existing sidebar | (none) — also addresses security F-08, F-13 |
| **PR-2** | Watchlist + WatchlistMember model + repo + minimal sidebar UI | PR-1 |
| **PR-3** | Streaks + Flipped + Band columns + CSV export on per-stock table | PR-2 |
| **PR-4** | RelatedEntity allowed-set extension + COMPETITOR CLI + Focus tab Competitors section | (none, but cleanest after PR-2) |
| **PR-5** | Institutional holders ingest + `INSTITUTIONAL_HOLDER` rows + Focus tab section | PR-4 |
| **PR-6** | InvestmentTheme model + ingest + `predict_themes` + new Themes tab | PR-4 |
| **PR-7** | Affinity-blended prediction mode in `aggregate.py` + opt-in toggle + tests + docs | PR-3..PR-6 |
| **PR-8** | `AFFINITY_WEIGHT` learning dimension in `learning/train.py` + simulate | PR-7 + ≥10 closed outcomes |

PRs 1–6 are roughly independent (after their direct prereqs) and
collectively land the user-visible feature; PR-7 is where the new
prediction shape goes live; PR-8 makes it self-tuning.

---

## 4. Open questions for sign-off

Locked in: scope (per-stock + affinity + themes), schema shape
(new Watchlist/WatchlistMember/InvestmentTheme + extended RelatedEntity),
test posture (one test file per PR), security posture (validation
helpers from PR-1 close F-08/F-13).

What needs your call:

- **Default affinity weights** above are a guess. We should ship them
  off-by-default and let calibration find them.
- **How many themes to seed** the constants with — the Finnhub
  `stock_investment_theme` endpoint requires one call per theme code,
  and there's no "list every theme" endpoint. I propose we seed 10–15
  common ones (e.g. `financialExchangesData`, `cyberSecurity`,
  `cleanEnergy`, `electricVehicles`, `aiSemis`) and add a CLI for
  adding more by code.
- **Competitor curation** — should the system auto-derive an initial
  `COMPETITOR` set from `PEER` (e.g. promote peers that share a
  Finnhub industry code), or stay strictly user-curated?
- **Watchlist persistence layer** — schema lives in the same SQLite
  DB as predictions. Acceptable, or do you want watchlists in a
  separate user-DB so a `RESET_DB=1` doesn't wipe them?

# NQ Order Flow Backtester

A tick-level backtesting and orderflow analysis framework for NQ futures using Databento MBP-1 data from CME Globex. Signals are generated in Python and sent via HTTP to a NinjaTrader 8 add-on for execution.

## 🚀 Live Trading Ready

**Analysis:** NQ.c.0 order flow (better liquidity, more reliable data)
**Execution:** MNQ (Micro NQ - 1/10th risk, same price levels)

- ✅ **Live signal generator** → Databento live API → Python signal detection → NT8 HTTP
- ✅ **NT8 add-on** → Receives signals, displays alerts, optional auto-execution
- ✅ **Complete setup guide** → See [LIVE_TRADING_SETUP.md](LIVE_TRADING_SETUP.md)
- ✅ **NQ vs MNQ reference** → See [NQ_vs_MNQ_REFERENCE.md](NQ_vs_MNQ_REFERENCE.md)

**Quick start:**
1. Start NT8 add-on (compile `nt8/NQOrderFlowSignalReceiver.cs`)
2. Run `run_live_signals.bat` or `python src/nqbt/live/signal_generator.py`
3. Trade during 9:30-11:00 AM ET

---

## Project Structure

```
nqorderflowbacktester/
├── pyproject.toml
├── .env                          # DATABENTO_API_KEY (not committed)
├── .env.example
├── run_live_signals.bat          # Quick launcher for live signal generator
├── docs/                         # 📄 Documentation & guides
│   ├── LIVE_TRADING_SETUP.md     # Complete live trading setup
│   ├── LIVE_STRATEGY_LOCKED.md   # Final locked strategy parameters
│   ├── NQ_vs_MNQ_REFERENCE.md    # NQ vs MNQ contract comparison
│   ├── OPTIMIZATION_GUIDE.md     # Optimization best practices
│   └── ...                       # Other strategy & setup guides
├── results/                      # 📊 Backtest results & analysis
│   ├── csv/                      # Optimization results, trade logs
│   └── html/                     # Equity curves, PnL calendars
├── logs/                         # 📝 Script execution logs
├── data/                         # Cached .dbn files (auto-created, gitignored)
├── output/                       # Script output files (auto-created)
├── nt8/
│   └── NQOrderFlowSignalReceiver.cs  # NinjaTrader 8 signal receiver add-on
├── scripts/
│   ├── cache_creation_scripts/   # Data cache builders
│   │   ├── build_daily_profile_cache.py     # Daily EOD profiles
│   │   ├── build_signal_cache.py            # RTH signals
│   │   └── build_signal_cache_full_session.py  # Full session signals
│   ├── fetch_sample.py           # Verify data access
│   ├── preview_ticks.py          # Preview normalized tick table
│   ├── validate_window.py        # View raw ticks for a specific time window
│   ├── large_volume_check.py     # Find large prints on a session day
│   ├── preview_profile_and_bars.py  # Full analysis output → output/profile_and_bars.txt
│   ├── hmm_study.py              # HMM overnight→RTH regime study → output/hmm_study.txt
│   ├── regime_inspect.py         # Inspect HMM predictions for specific dates → output/regime_inspect.txt
│   └── study_runner.py           # Streamlit UI — select and run research studies
└── src/nqbt/
    ├── data/
    │   ├── schema.py             # MBP-1 constants, Action/Side enums
    │   ├── loader.py             # Fetch, cache, and load Databento data
    │   └── normalizer.py         # Decode and clean raw ticks into trade-only DataFrame
    ├── analysis/
    │   ├── volume_profile.py     # Developing volume profile with POC, VAH, VAL
    │   ├── range_bars.py         # 40-range volumetric bars with internal profile
    │   ├── range_bars_streaming.py  # Streaming bar builder for live data
    │   ├── vwap.py               # Anchored VWAP with 3 standard deviation bands
    │   ├── absorption.py         # Absorption signal detector (6 methods, bar-based)
    │   ├── features.py           # 10-feature computation per 40-range bar for HMM
    │   └── hmm_regime.py         # GaussianHMM regime classifier (trending/ranging/chop)
    ├── studies/
    │   ├── base.py               # BaseStudy, StudyResult, Param — study interface
    │   └── registry.py           # All 24 research studies registered across 7 blocks
    ├── live/
    │   └── signal_generator.py   # Live signal generator (Databento → NT8)
    └── backtest/
        └── engine.py             # BacktestEngine, Strategy protocol, Position, Quote, Trade
```

---

## File Reference

### `src/nqbt/data/schema.py`
Constants for working with Databento MBP-1 data:
- `DATASET`, `SCHEMA`, `NQ_SYMBOL`, `STYPE` — Databento connection parameters
- `PRICE_SCALE = 1e9` — raw int64 prices, divide by this to get dollars
- `Action` — event types: `ADD`, `CANCEL`, `MODIFY`, `TRADE`, `FILL`, `CLEAR`
- `Side` — `BID`, `ASK`, `NONE`
- `MBP1_COLUMNS` — all columns in a decoded MBP-1 DataFrame

### `src/nqbt/data/loader.py`
All data access:
- `fetch(start, end)` — downloads MBP-1 data, caches as `.dbn` locally. Subsequent calls load from disk — no re-download.
- `load_df(store)` — converts `DBNStore` to pandas DataFrame, normalizes raw int64 prices to float dollars.
- `fetch_and_load(start, end)` — fetch + load in one call.

> **Data constraint:** Databento plan covers 1 year of history. Use dates from 2025-03-13 onward.

### `src/nqbt/data/normalizer.py`
Two normalizer functions:

**`normalize(df)`** — trade-only tick stream for orderflow analysis:
- Filters to `action == 'T'` only — drops adds, cancels, modifies
- Assigns aggressor: `side == 'A'` → `'buy'` (resting ask lifted), `side == 'B'` → `'sell'` (resting bid hit), `side == 'N'` dropped
- Assigns `session_date` — CME trading date (sessions open 6pm ET, so events before 6pm ET belong to that calendar date, events at/after 6pm ET belong to the next date)
- Output columns: `ts_event` (index), `price`, `size`, `aggressor`, `session_date`

**`normalize_enriched(df)`** — extended version for HMM feature computation:
- Same filtering and aggressor/session logic as `normalize()`
- Additionally keeps: `mid_price` = (bid_px_00 + ask_px_00) / 2, `bid_sz`, `ask_sz`
- Required by `features.py` for OBI, trend_ratio, realized_vol, and all midprice-based calculations
- Output columns: `ts_event` (index), `price`, `size`, `aggressor`, `session_date`, `mid_price`, `bid_sz`, `ask_sz`

### `src/nqbt/analysis/volume_profile.py`
Developing volume profile:
- `VolumeProfile.build(ticks)` — builds from any tick slice (pass a growing slice for developing profile)
- Level size: 10 ticks = 2.5 points per level
- Per level: `buy_vol`, `sell_vol`, `total_vol`, `delta`
- Computes `POC` (highest volume level), `VAH`, `VAL` (70% value area)
- `to_df()` returns profile as DataFrame with POC and value area flags
- `summary()` returns a one-line string of key levels

**Session convention:** profile starts at 6pm ET prior day and develops forward. For live strategy use, slice ticks to `[6pm ET prior day, current_time]` before calling `build()`.

### `src/nqbt/analysis/range_bars.py`
40-range volumetric bars:
- Bar closes when `high - low >= 40 ticks (10 points)`
- Each bar: `open_time`, OHLC, `buy_vol`, `sell_vol`, `delta`, `total_vol`, `poc()`
- Internal volume profile per bar at 5-tick levels (1.25 points)
- `build_range_bars(ticks)` → list of `RangeBar`
- `bars_to_df(bars)` → flat summary DataFrame
- `bar.levels_df()` → internal profile for a single bar

> **Note:** `open_time` is the timestamp of the first tick (bar open). NinjaTrader stamps range bars at the close — when cross-referencing with NT8, match against `close_time`, not `open_time`.

> **On gaps:** Range bars can have gaps between close and next open — this is real market data (no trade printed in that range). This is intentional and more accurate than NT8's cosmetic no-gap convention.

### `src/nqbt/analysis/vwap.py`
Anchored VWAP with standard deviation bands:
- `compute_vwap(ticks)` — computes cumulative VWAP tick-by-tick from session open
- `vwap_at(ticks, as_of)` — returns VWAP snapshot at a specific timestamp
- Bands at multipliers 1, 2, 3 (above and below) — 6 bands total
- Uses volume-weighted variance: `variance = Σ(size × price²) / Σ(size) - vwap²`
- Reset interval: 6pm ET prior day (pass the correct session slice)

### `src/nqbt/analysis/absorption.py`
Absorption signal detector — the core signal of the system. Operates on 40-range bars, not raw ticks.

**What absorption means here:**
Absorption is when one side aggressively attacks a price level (large directional volume) but price closes against them. The opposing side absorbed that aggression without giving ground.
- `buy_absorbed` — buy-dominated bar that closed bearish. Buyers were aggressive, sellers held. Bearish implication.
- `sell_absorbed` — sell-dominated bar that closed bullish. Sellers were aggressive, buyers held. Bullish implication.
- `unconfirmed` — combined signal fired but aggression aligned with bar direction. Strong trending bar, not absorption. Do not trade.

**Why bar-based instead of tick-based:**
Tick-level scanning detects aggression at individual price levels but has no natural unit for "how much is significant." Bar-based detection uses the 40-range bar as the unit of analysis — each bar already captures a complete price swing, making imbalance within it directly meaningful.

**`score_bars(bars)` → DataFrame**, one row per bar. Columns include per-method signal booleans, raw scores, and `signal_combined` + `absorption_side`.

**`combine_mode`:** `'all'` requires every enabled method to fire (highest confidence). `'any'` fires on any single method (more signals, noisier).

---

**Six detection methods — all toggleable:**

**Method 1 — Rolling Percentile Normalization** (`use_percentile=True`)
Signals when `abs(imbalance)` exceeds the Nth percentile of the last `lookback` bars. Adapts to volatility regime changes over time.

| Parameter | Default | Notes |
|---|---|---|
| `lookback` | `500` | 500–2000 bar range. Too short = overfits. Too long = adapts slowly. Optimize empirically. |
| `percentile` | `80.0` | 80 is starting point. 90 is more aggressive. Research will show win rate / frequency tradeoff. |

**Method 2 — Volume-Normalized Imbalance Ratio** (`use_ratio=True`)
`ratio = (buy_vol - sell_vol) / total_vol`. No lookback needed — naturally adjusts to participation level. 0.40 = 70/30 split.

| Parameter | Default | Notes |
|---|---|---|
| `min_ratio` | `0.40` | Lower = more signals. Higher = cleaner. |
| `min_volume_floor` | `50` | Prevents thin bars producing extreme spurious ratios. Start at 25th percentile of bar volumes. Needs empirical optimization. |

**Method 3 — Session Activity Scaling** (`use_activity=False`)
`threshold = activity_k × running_avg_bar_volume`. Scales the imbalance threshold to the day's character — thin sessions get a lower bar, heavy news-driven sessions require more. Responds in near real-time.

| Parameter | Default | Notes |
|---|---|---|
| `activity_k` | `2.0` | Multiplier on running average volume. |
| `activity_warmup_bars` | `20` | Bars before method activates. Early-session instability makes it unreliable until enough volume has accumulated. |

**Method 4 — Z-Score of Imbalance** (`use_zscore=False`)
`Z = (imbalance - rolling_mean) / rolling_std`. Directly measures how statistically exceptional the current bar is vs. recent history — not just how large.

| Parameter | Default | Notes |
|---|---|---|
| `zscore_lookback` | `100` | Rolling window for mean/std. |
| `zscore_threshold` | `2.0` | 1.5 = more signals, 2.0 = more selective. Combine with percentile check — imbalance distributions are fat-tailed and non-normal so Z-scores alone can understate tail risk. |

**Method 5 — ATR-Normalized Imbalance** (`use_atr=False`)
`strength = abs(imbalance) / ATR(n)`. Normalizes for volatility — when ATR is high, larger imbalances are expected. Directly captures "effort vs. result." Note: for 40-range bars, H-L is always 10 points, so ATR variation comes from gaps between bars (True Range includes prior close distance).

| Parameter | Default | Notes |
|---|---|---|
| `atr_period` | `14` | Bars in ATR calculation. |
| `atr_k` | `1.0` | Signal when strength >= this. |

**Method 6 — EWMA Adaptive Threshold** (`use_ewma=False`)
`signal when abs(imbalance) > ewma_k × EWMA(abs_imbalance, span)`. Exponentially weighted baseline — recent bars receive more weight, threshold drifts toward current conditions automatically. Faster adaptation than a simple rolling window with no fixed window boundary artifacts.

| Parameter | Default | Notes |
|---|---|---|
| `ewma_span` | `50` | Controls decay speed. Lower = faster adaptation, higher = smoother. |
| `ewma_k` | `2.0` | Higher = fewer but more extreme signals. |

---

**Planned:** A Streamlit app for interactive backtesting — select which methods to enable, dial in parameters, run across a date range, and view signal outcomes visually. This is the primary research tool for optimizing the absorption detector.

### `src/nqbt/analysis/features.py`
Ten-feature computation module for the HMM regime classifier. Features are computed per 40-range bar — each bar produces one feature vector. The HMM never sees raw ticks directly; raw ticks within each bar are the input used to derive the bar-level features.

**How the data flows:**
- Raw ticks within a bar → midprice series, log returns, OBI series, buy/sell volumes
- These are aggregated into one feature vector per bar
- The sequence of bar vectors is what gets fed to the HMM

**Feature computation — what uses 40-range bars vs raw ticks:**

| Feature | Tier | Input | What it measures |
|---|---|---|---|
| `delta_pct` | 1 | buy/sell vol per bar | `(buy_vol - sell_vol) / total_vol` — normalized conviction of aggression |
| `trend_ratio` | 1 | midprice of ticks within bar | net midprice move / total path length — directional vs noisy movement |
| `obi_avg` | 1 | bid_sz/ask_sz of ticks within bar | mean `(bid_sz - ask_sz) / (bid_sz + ask_sz)` — passive queue pressure |
| `realized_vol` | 1 | log returns of ticks within bar | `sqrt(Σ rt²)` — separates trending from choppy high-activity |
| `trade_arrival_rate` | 1 | tick count and bar duration | trades per second — urgency and conviction |
| `delta_acceleration` | 2 | `delta_pct[i] - delta_pct[i-1]` across bars | rate of change of delta — early warning before price confirms |
| `obi_trend` | 2 | OBI series of ticks within bar | linear regression slope of OBI — which direction passive pressure is building |
| `vwap_deviation` | 2 | bar close vs running VWAP | `(close - vwap) / vwap_std` — anchors regime to value |
| `autocorr_r1` | extra | log returns within bar | lag-1 autocorrelation — persistence of directional movement |
| `normalized_d_move` | extra | midprice and realized_vol within bar | net move / (realized_vol × open_price) — Sharpe-like effort-result ratio |

**Midprice:** `Mt = (bid_px_00 + ask_px_00) / 2` is used for all price-based calculations. Log returns `rt = ln(Mt / Mt-1)` feed into `realized_vol`, `autocorr_r1`, `trend_ratio`, and `normalized_d_move`.

**Normalization:** `StandardScaler.fit_transform()` is applied per session independently — each session's feature matrix is normalized on its own rows before being assembled into the HMM input. This removes session-level scale differences (a slow overnight vs a fast RTH session both present normalized features to the model).

`compute_session_features(enriched_ticks, bars, vwap_df)` → `np.ndarray` of shape `(n_valid_bars, 10)`

### `src/nqbt/analysis/hmm_regime.py`
Hidden Markov Model regime classifier. Two separate `GaussianHMM` models are trained — one on overnight sessions, one on RTH sessions.

**Model configuration:**
- `n_components=3` — three regimes: trending, ranging, chop
- `covariance_type='full'` — learns a full N×N covariance matrix per state, not just diagonal. This means the model learns not just the mean of each feature per regime but the complete joint distribution, including how features co-move. In trending, `delta_pct` and `trend_ratio` are expected to show high positive covariance. In chop, `realized_vol` is high while `trend_ratio` is low, producing negative covariance between them. This joint structure makes the classifier meaningfully more powerful than any single-feature threshold rule.
- `n_iter=200`

**State alignment** (post-training):
- `trending` — state with highest `abs(mean delta_pct)`
- `ranging` — state with lowest `mean realized_vol` among remaining
- `chop` — remaining state

**`RegimeHMM`** class:
- `fit(sessions)` — train on list of per-session feature matrices
- `align_states()` — map integer states to regime labels using emission means
- `predict_session(X)` — Viterbi decode, return mode regime for the session
- `predict_proba_session(X)` — fraction of bars spent in each regime

**`run_overnight_rth_study()`** — full research study:
- Trains both HMMs on training sessions
- For each test day: predicts overnight regime and RTH regime
- Builds contingency table: `P(RTH regime | overnight regime)`, row-normalized
- Runs chi-squared test of independence (p < 0.05 = statistically significant relationship)
- Reports per-state match rates and emission means for both models

### `scripts/hmm_study.py`
End-to-end HMM regime study script.
- **Training period:** 2025-03-17 to 2025-11-30
- **Test period:** 2025-12-01 to 2026-03-15 (never seen during training)
- Processes data day by day for memory efficiency — each day loads a 2-calendar-day window, normalizes to enriched ticks, builds 40-range bars, computes features, then discards raw data
- Caches feature arrays to `output/feature_cache/{date}_{session}.npy` — subsequent runs skip all data loading
- Saves trained models to `output/overnight_hmm.pkl` and `output/rth_hmm.pkl`
- Results written to `output/hmm_study.txt`

Run with:
```bash
python scripts/hmm_study.py
```
First run will be slow due to data fetching. All subsequent runs load from cache.

### `scripts/regime_inspect.py`
Inspects HMM regime predictions for specific trading dates.

- Requires trained models in `output/` — run `hmm_study.py` first
- For each date shows: predicted overnight and RTH regime, session-average feature values (z-scored), and all three state emission means side-by-side with the predicted state marked
- Loads feature arrays from cache if available, otherwise fetches raw data
- Results written to `output/regime_inspect.txt`

Run with:
```bash
python scripts/regime_inspect.py
```

### `src/nqbt/studies/base.py`
Base interface for all research studies:
- `Param` — descriptor for one configurable parameter (type, default, min/max, choices)
- `StudyResult` — return type: `summary` string + optional `tables` list of `(label, DataFrame)`
- `BaseStudy` — base class with `id`, `name`, `block`, `description`, `variables`. Default `run()` returns a not-yet-implemented stub — override in each study class to activate.

### `src/nqbt/studies/registry.py`
All 24 research studies registered across 7 blocks. Each study is a `BaseStudy` subclass with its parameter schema defined. To implement a study, add a `run()` method to the class. The Streamlit UI auto-discovers it with no other changes needed.

**Block 1 — Value Area Behavior** (4 studies)
VAH/VAL reaction rates, break-and-retest delta thresholds, post-break continuation vs failure, POC reach probability from VAH/VAL reversals.

**Block 2 — VWAP and Standard Deviation Bands** (3 studies)
Band reaction rates solo vs confluence with VAH/VAL/POC, extreme extension reversal rates by band level, VWAP band + VAH/VAL confluence zone behavior.

**Block 3 — Absorption and Aggression** (3 studies)
Absorption + immediate buying aggression signal quality, absorption score optimization sweep, delta threshold optimization sweep.

**Block 4 — Historical Delta Nodes** (3 studies)
Sell-delta node reversal rates, buy-delta node reversal rates, optimal decay halflife for node relevance.

**Block 5 — Regime Classification** (5 studies)
Overnight directional score as RTH setup predictor, overnight deceleration as failed auction predictor, bar-by-bar regime stabilization timing, extended observation window HMM comparison, post-11:00 favorable entry study.

**Block 6 — Session Structure** (2 studies)
Reaction location mapping given directional bias, intraday session state sequence classification.

**Block 7 — Prop Firm Optimization** (4 studies)
Monte Carlo risk optimization per funded account mode, challenge pass rate optimization, funded payout probability optimization, martingale viability study.

All Block 7 studies evaluate every strategy in both the prop firm context and the equivalent live account context. The comparison metric throughout is **EV per dollar of personal capital at risk**, not raw profit. Personal capital at risk is total fees paid across all accounts in the prop firm context, and account size in the live account context. A strategy is not concluded "not worth trading" until this comparison has been run — the prop firm structure can produce 5–10× better capital efficiency on the same underlying strategy.

| Context | Capital at Risk | Example Monthly Return |
|---|---|---|
| Live account | Account size (e.g. $50,000) | $500 / $50,000 = **1.0%** |
| Prop firm (20 accounts × $300 fee) | $6,000 total fees | $500 × 20 / $6,000 = **166%** |

Each Block 7 study accepts three shared comparison parameters: `live_account_size`, `prop_fee_per_account`, and `n_prop_accounts`.

### `scripts/study_runner.py`
Streamlit UI for running research studies. Exposes all 24 registered studies through a point-and-click interface.

Install UI dependencies:
```bash
pip install streamlit plotly
```

Launch with:
```bash
streamlit run scripts/study_runner.py
```

**Features:**
- Block → Study dropdown navigation in the sidebar
- Date range selector
- Auto-generated parameter controls (sliders, dropdowns, checkboxes) for every study's variables
- Results displayed inline as text + tables
- Results auto-saved to `output/studies/<id>_<start>_<end>.txt`
- Download button for the result file

### `src/nqbt/backtest/engine.py`
Core backtesting engine (to be wired up to analysis signals):
- `Quote`, `Trade`, `Fill` — market data structures
- `Position` — tracks size, average price, realized/unrealized PnL
- `Strategy` — Protocol interface: implement `on_quote()` and `on_trade()`, return orders as dicts
- `BacktestEngine` — replays MBP-1 events tick-by-tick, executes orders with slippage, outputs per-tick PnL snapshots

---

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```

2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip setuptools
   python -m pip install -e ".[dev]"
   ```

   For the Streamlit study runner, also install UI extras:
   ```bash
   pip install streamlit plotly
   ```

3. Copy `.env.example` to `.env` and add your Databento API key:
   ```
   DATABENTO_API_KEY=your_key_here
   ```

4. Verify data access:
   ```bash
   python scripts/fetch_sample.py
   ```

---

## Scripts

| Script | Purpose |
|---|---|
| `fetch_sample.py` | Verify API key and data access |
| `preview_ticks.py` | Print normalized tick table for a date range |
| `validate_window.py` | Print raw ticks for a specific ET time window |
| `large_volume_check.py` | Find all large prints (≥50 contracts) on a session day |
| `preview_profile_and_bars.py` | Full output: VWAP, volume profile, 40-range bars, absorption events → `output/profile_and_bars.txt` |
| `hmm_study.py` | HMM overnight→RTH regime predictability study → `output/hmm_study.txt` |
| `regime_inspect.py` | Inspect HMM predictions for specific dates → `output/regime_inspect.txt` |
| `study_runner.py` | Streamlit UI — select and run any of the 24 research studies |

---

## Key Data Facts

| Item | Value |
|---|---|
| Symbol | `NQ.c.0` (continuous front-month) |
| Dataset | `GLBX.MDP3` |
| Schema | `mbp-1` |
| Price encoding | raw int64 ÷ `1e9` = dollars |
| Tick size | 0.25 points |
| Tick value | $5.00 |
| Session open | 6pm ET prior calendar day |
| Timestamps | nanoseconds since Unix epoch (UTC), displayed in ET |

---

## What Needs to Be Built Next

### Research
- **Implement study logic** — the study runner UI and all 24 study stubs are registered. Implement `run()` in each study class in `registry.py` one at a time to activate them. Studies to prioritize first: 3.2 (absorption size optimization), 3.3 (delta threshold optimization), 5.1 (overnight score as RTH predictor), 1.1 (VAH/VAL reaction rate).
- **Absorption outcome tagging** — for each absorption signal, record the price movement over the next N bars to evaluate signal quality. This feeds directly into studies 3.1, 3.2, and 3.3.
- **Investigate pre-market window** — the 07:30–09:30 ET window may be a stronger overnight regime predictor than the full 18:00–09:30 overnight. Study 5.4 tests this directly.

### Signal Layer
- **Delta divergence** — price makes a new high/low but delta does not confirm.
- **Volume at price (VAP)** — track which price levels have the most total volume over the session, not just the value area.

### Strategy Layer
- **Entry logic** — combine absorption signal + VWAP context + profile position (above/below VAH, VAL, POC) + HMM regime filter to gate trade entries.
- **Trade window** — strategy operates 9:30am–11:00am ET. Profile develops from 6pm ET prior day up to the current bar. Levels recalculate every ~2 minutes.
- **Signal output format** — define the HTTP payload structure for sending signals to NT8.

### Live System
- **Databento live feed** — subscribe to real-time MBP-1 stream, apply same normalizer pipeline.
- **NT8 HTTP bridge** — NT8 exposes an HTTP listener on localhost; Python posts signal and trade management updates to it.
- **State management** — maintain rolling session state (developing profile, VWAP, absorption history, HMM regime) across the live session.

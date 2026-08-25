# NQ Order Flow Backtester

A tick-level backtesting and orderflow analysis framework for NQ futures using Databento MBP-1 data from CME Globex. Signals are generated in Python and sent via HTTP to a NinjaTrader 8 add-on for execution.

> **Scope note:** This README documents the original `src/nqbt` analysis package and the
> single-strategy live path. The **current** live system is the 4-way combined trader in
> `live/combined/` — see **[FILE_ORGANIZATION.md](FILE_ORGANIZATION.md)** for the map of what
> is actually in use today, including the strategy research directories under `scripts/`.

## 🚀 Live Trading

**Analysis:** NQ.c.0 order flow (better liquidity, more reliable data)
**Execution:** MNQ (Micro NQ - 1/10th risk, same price levels)

- **NT8 add-on** → Receives signals, displays alerts, optional auto-execution
- **Setup guide** → See [docs/LIVE_TRADING_SETUP.md](docs/LIVE_TRADING_SETUP.md)
- **NQ vs MNQ reference** → See [docs/NQ_vs_MNQ_REFERENCE.md](docs/NQ_vs_MNQ_REFERENCE.md)

**Quick start (current system):**
1. Start NT8 add-on (compile `nt8/NQMultiStratReceiver.cs`)
2. Run `python live/combined/run_phase1.py`

The legacy path (`run_live_signals.bat` → `src/nqbt/live/signal_generator.py`, 9:30–11:00 ET
single strategy) is **broken** — it imports `Level` from `range_bars`, which was renamed
`BarLevel`. It is kept for reference only.

---

## Project Structure

```
nqorderflowbacktester/
├── pyproject.toml
├── .env                          # DATABENTO_API_KEY (not committed)
├── .env.example
├── FILE_ORGANIZATION.md          # 🗺️ Map of the whole repo — start here
├── docs/                         # 📄 Documentation & guides
│   ├── LIVE_AND_BACKTEST_GUIDE.md    # Primary reference
│   ├── LIVE_TRADING_SETUP.md     # Complete live trading setup
│   ├── LIVE_STRATEGY_LOCKED.md   # Final locked strategy parameters
│   └── ...                       # Other strategy & setup guides
├── live/
│   ├── combined/                 # ⭐ CURRENT live system (4-way combined)
│   │   ├── run_phase1.py         # Main entry point
│   │   ├── config.py             # All strategy + execution parameters
│   │   ├── data_feed.py          # Databento live feed
│   │   ├── bar_builder.py        # MultiBarBuilder / Bar
│   │   ├── coordinator.py        # Cross-strategy risk arbitration
│   │   ├── {rv,b2,od,fabio_orb}_engine.py   # The four strategy engines
│   │   └── {nt8,mt5}_executor.py # Broker routing
│   ├── farm/                     # Multi-account farm (funded + eval brains)
│   └── *.py                      # Legacy single-strategy trader (superseded)
├── nt8/
│   ├── NQMultiStratReceiver.cs   # ⭐ Current multi-strategy receiver
│   └── NQOrderFlowSignalReceiver.cs  # Legacy single-strategy receiver
├── scripts/                      # Research threads, one dir per strategy/topic
│   ├── cache_creation_scripts/   # Data cache builders
│   │   ├── fetch_and_build_single_day.py    # One-command single session
│   │   ├── build_daily_profile_cache.py     # Daily EOD profiles
│   │   └── build_signal_cache.py            # RTH signals
│   ├── rough vol orderflow/      # RV strategy
│   ├── fabio_orb/                # Fabio ORB strategy
│   ├── overnight drift strategy/ # OD strategy
│   ├── overnight range strat/    # B2 strategy
│   ├── montecarlo/               # Prop-firm Monte Carlo library
│   ├── thetadata/                # QQQ/NDX greeks + daily gamma pipeline
│   └── ...                       # See FILE_ORGANIZATION.md for all 21 dirs
├── results/                      # 📊 Legacy top-level results (VWAP era)
├── logs/                         # 📝 Script execution logs
├── data/                         # Transient .dbn cache (gitignored)
└── src/nqbt/                     # Core package — only what live code still imports
    ├── data/
    │   ├── schema.py             # MBP-1 constants, Action/Side enums
    │   ├── loader.py             # Fetch, cache, and load Databento data
    │   └── normalizer.py         # Decode and clean raw ticks into trade-only DataFrame
    ├── analysis/
    │   ├── volume_profile.py     # Developing volume profile with POC, VAH, VAL
    │   ├── range_bars.py         # 40-range volumetric bars with internal profile
    │   ├── range_bars_streaming.py  # Streaming bar builder (legacy, broken)
    │   └── vwap.py               # Anchored VWAP with 3 standard deviation bands
    └── live/
        └── signal_generator.py   # Legacy live signal generator (broken)
```

> The HMM/regime research cluster (`features.py`, `hmm_regime.py`, `absorption.py`,
> `directional_hmm.py`, and 9 more), the 24-study registry (`studies/`), and the original
> `backtest/engine.py` were removed on 2026-07-25 — nothing imported them from any entry point.
> They remain in git history if needed.

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

> **Data constraint:** Databento covers roughly late 2025 onward. Earlier history (2020-12 →
> 2025-11) comes from the MarketTick archive on `D:/` instead — see the cache builders named
> `*_from_markettick.py` in `scripts/cache_creation_scripts/`.

### `src/nqbt/data/normalizer.py`
Two normalizer functions:

**`normalize(df)`** — trade-only tick stream for orderflow analysis:
- Filters to `action == 'T'` only — drops adds, cancels, modifies
- Assigns aggressor: `side == 'A'` → `'buy'` (resting ask lifted), `side == 'B'` → `'sell'` (resting bid hit), `side == 'N'` dropped
- Assigns `session_date` — CME trading date (sessions open 6pm ET, so events before 6pm ET belong to that calendar date, events at/after 6pm ET belong to the next date)
- Output columns: `ts_event` (index), `price`, `size`, `aggressor`, `session_date`

**`normalize_enriched(df)`** — extended version that also carries quote state:
- Same filtering and aggressor/session logic as `normalize()`
- Additionally keeps: `mid_price` = (bid_px_00 + ask_px_00) / 2, `bid_sz`, `ask_sz`
- Was built for the now-deleted `features.py` (OBI, trend_ratio, realized_vol). Still useful for
  any midprice- or book-imbalance-based calculation.
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

> The File Reference sections for `absorption.py`, `features.py`, `hmm_regime.py`,
> `scripts/hmm_study.py`, `scripts/regime_inspect.py`, `studies/base.py`,
> `studies/registry.py`, `scripts/study_runner.py`, and `backtest/engine.py` were removed
> along with those files on 2026-07-25. See git history if you need them.

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

3. Copy `.env.example` to `.env` and add your Databento API key:
   ```
   DATABENTO_API_KEY=your_key_here
   ```

---

## Scripts

The old loose diagnostic scripts in `scripts/` root (`fetch_sample.py`, `preview_ticks.py`,
`hmm_study.py`, `study_runner.py`, and friends) no longer exist. Research now lives in
per-topic directories — see **[FILE_ORGANIZATION.md](FILE_ORGANIZATION.md)** for the full list.

| Entry point | Purpose |
|---|---|
| `live/combined/run_phase1.py` | Run the current 4-way combined live system |
| `live/combined/replay_today.py` | Replay today's ticks through the same engines |
| `live/farm/start_farm_app.bat` | Launch the multi-account farm app |
| `scripts/cache_creation_scripts/fetch_and_build_single_day.py` | Download DBN + build all caches for one session |
| `scripts/thetadata/daily_pipeline/run_daily.py` | Daily QQQ/NDX gamma refresh (needs ThetaData Terminal on :25503) |

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

## Historical Roadmap (superseded)

This section described the original absorption + HMM research programme. That direction was
abandoned — the modules backing it were deleted on 2026-07-25, and the live system took a
different path (RV / B2 / OD / Fabio ORB in `live/combined/`). The live-system items below
were all built and shipped.

Two ideas from the old signal-layer list were never explored and may still be worth a look:

- **Delta divergence** — price makes a new high/low but delta does not confirm.
- **Volume at price (VAP)** — track which price levels have the most total volume over the
  session, not just the value area.

# File Organization Guide

Quick reference for where to find things. Last updated 2026-04-14.

## 📄 Documentation → `docs/`

**Primary reference:**
- `LIVE_AND_BACKTEST_GUIDE.md` — **Start here.** How live works, how to run live, how to run backtests, cache pipeline, troubleshooting.

**Setup / deeper dives:**
- `LIVE_TRADING_SETUP.md` — NT8 add-on compilation + connection
- `LIVE_STARTUP_GUIDE.md` — Original startup walk-through
- `LIVE_IMPLEMENTATION_SUMMARY.md` — Architecture overview
- `LIVE_STRATEGY_LOCKED.md` — Locked strategy parameters
- `WARM_START_GUIDE.md` — Warm start internals
- `BREAKEVEN_IMPLEMENTATION.md` — BE logic (historical reference)
- `smart_order_routing_plan.md` — Smart routing (LIMIT/STOP 9:30-12:00 ET)
- `account_rotation_plan.md` — 2-account rotation concept
- `OPTIMIZATION_GUIDE.md` — Optimization best practices
- `NQ_vs_MNQ_REFERENCE.md` — Contract comparison
- `VOLUME_PROFILE_STRUCTURE.md` — Volume profile data structure
- `15MIN_BARS_COMPARISON.md` — Timeframe comparison notes
- `EQUITY_CURVE_TEMPLATE.md` — HTML template reference

## 🐍 Live Trading Code → `live/`

- `live_trader.py` — Main loop. Databento feed → engine → NT8 HTTP.
- `signal_engine.py` — Stateful VWAP/ADX/ATR/bias/trend/signal logic.
- `bar_builder.py` — Range-bar + 5-min time-bar builders.
- `warm_start.py` — Fetches session history, replays into engine.
- `signal_client.py` — HTTP POST client → NT8 add-on (:8080).
- `config_gui.py` — Tkinter startup GUI (mode, accounts, rotation).
- `account_manager.py` — Per-account equity/HWM/DD/consistency + rotation.
- `test_execution.py` — Manual NT8 round-trip test.

## 🔷 NT8 Add-On → `nt8/`

- `NQOrderFlowSignalReceiver.cs` — HTTP listener, routes orders per-account.
- `NQOrderFlowATRStrategy.cs` — (older) native strategy version.

## 🐍 Core Python Package → `src/nqbt/`

- `data/loader.py`, `schema.py`, `normalizer.py` — Databento loading / normalization
- `analysis/` — Range bars, VWAP, volume profile, features, ADX
- `backtest/engine.py` — Backtest engine (Strategy protocol, Position, Trade)
- `live/` — (older) live signal generator; superseded by top-level `live/`

## 📜 Scripts → `scripts/`

### VWAP Reaction Strategy Backtests → `scripts/vwap_reaction_strat_backtest/`
Primary backtest location. Key scripts:
- `generate_combined_html.py` — Equity curve + PnL calendar across full date range.
- `zone_sweep_is_oos.py` — IS/OOS sweep of band-zone width for trending entries.
- `trending_vwap_atr_grid.py` — ATR SL/TP grid search for trending mode.
- `montecarlo_multi_account.py` — Multi-account Monte Carlo (Lucid/Apex pass rates).
- `montecarlo_combined.py` / `montecarlo_apex_full.py` — Other MC variants.
- `filtered_insample_oos.py` / `filtered_adx_insample_oos.py` — IS/OOS filter sweeps.
- `slippage_test.py` — Slippage resistance.
- `overfit_detection.py` — Full overfit test suite (parameter stability / walk-forward / MC DD / bootstrap / direction permutation).
- `adx_regime_analysis.py` / `regime_analysis.py` — ADX regime diagnostics.

### Cache Builders → `scripts/cache_creation_scripts/`
- `fetch_and_build_single_day.py` — **Single command** to download DBN + build all 4 caches (signal / vwap / timebars_5min / vwap_reaction) for one session.
- `build_signal_cache.py` / `build_signal_cache_full_session.py` — Range-bar signal caches.
- `build_daily_profile_cache.py` — Daily EOD volume profiles.
- `build_vwap_reaction_cache.py` — VWAP reaction signals (under `vwap_creation/`).

### Databento Downloads → `scripts/databentoDownloadScripts/`
Batch DBN downloaders (historical ranges).

### Other backtest dirs
- `scripts/poc_reaction_strat_backtest/` — POC reaction strategy (older).
- `scripts/rough_vol_optimization/` — Rough-vol regime work (older).
- `scripts/experimental/` — Experimental / prototyping scripts.

## 📊 Results → `results/`

- `results/equity/` — Equity curve HTMLs per config
- `results/calendar/` — PnL calendar HTMLs
- `results/montecarlo/` — MC outputs (CSV + HTML)
- `results/csv/` — Backtest trade logs, optimization grids
- `results/html/` — Misc visualizations

## 📝 Logs → `logs/`

Script execution logs (`backtest_*.log`, `build_*.log`, `atr_backtest_*.log`, etc.).

## 💾 External Data → `D:/trading_pythonbacktest_data/`

Not in repo (gitignored). All tick / bar / signal caches live here:
- `dbn/` — Raw DBN files (one per calendar day)
- `signal_cache/` — Range-bar signal caches (one .pkl per session date)
- `vwap_cache/` — 1-min VWAP caches
- `timebars_5min/` — 5-min OHLC bar caches
- `vwap_reaction_cache/` — VWAP reaction signal caches
- `daily_profile_cache/` — Daily EOD volume profiles

## 🔧 Root-Level Utilities

- `run_live_signals.bat` — Windows quick launcher for live trader.
- `.env` / `.env.example` — `DATABENTO_API_KEY` (not committed).
- `pyproject.toml` — Python package + deps.

## Quick Access

```bash
# Start live trader (GUI config, warm start)
python live/live_trader.py

# Build caches for a new session
python scripts/cache_creation_scripts/fetch_and_build_single_day.py 2026-04-13

# Rebuild equity curve + PnL calendar
python scripts/vwap_reaction_strat_backtest/generate_combined_html.py

# Run full overfit test suite
python scripts/vwap_reaction_strat_backtest/overfit_detection.py

# Main live/backtest reference
notepad docs/LIVE_AND_BACKTEST_GUIDE.md
```

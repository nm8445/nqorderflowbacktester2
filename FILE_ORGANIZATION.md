# File Organization Guide

Quick reference for where to find things. Last updated 2026-07-25.

## 🐍 Live Trading → `live/`

### `live/combined/` — **current live system** (4-way combined)

The active production path. Four strategy engines (RV, B2, OD, Fabio ORB) share one
Databento feed and one bar builder, coordinated into NT8/MT5 execution.

- `run_phase1.py` — **main entry point.** Feed → bar builder → engines → coordinator → executor.
- `config.py` — all strategy + execution parameters.
- `data_feed.py` — Databento live feed, `on_tick` fan-out.
- `bar_builder.py` — `MultiBarBuilder`, per-timeframe `Bar` construction.
- `warm_start.py` — replays cached session history into the builders.
- `coordinator.py` — cross-strategy position/risk arbitration.
- `nt8_executor.py` / `mt5_executor.py` — order routing to each broker.
- `tick_position_monitor.py` — tick-level position tracking.
- Engines: `rv_engine.py`, `b2_engine.py`, `b2_signal_engine.py`, `od_engine.py`, `fabio_orb_engine.py`
- RV support: `rv_features.py`, `rv_orderflow.py`
- Support: `active_contract.py`, `gamma_refresh.py`, `news_blackout.py`, `session_backfill.py`,
  `settle_recorder.py`, `rolling_cache.py`, `bar_persistor.py`
- Replay/verification: `replay_today.py`, `replay_rv.py`, `replay_b2.py`, `replay_od.py`,
  `replay_fabio.py`, `paper_tracker.py`
- `state/` — runtime JSON (active contract, coordinator, martingale, settles) + `replay/` tick parquets

### `live/farm/` — multi-account farm (funded + eval brains)

- `app.py` — farm web app (`start_farm_app.bat` to launch)
- `funded_state_machine.py` — funded-account lifecycle
- `eval_passer.py` / `sim_eval_passer.py` — eval-taking logic + simulator
- `accounts_client.py`, `farm_broadcast.py`, `farm_monitor.py`, `validate_funded_brain.py`
- `TEST_ADDON_SPEC.md` — NT8 test addon (:8082) contract

### `live/` root — **legacy single-strategy trader** (superseded)

Kept for reference only; `live/combined/` replaced it and shares no code with it.
`live_trader.py`, `signal_engine.py`, `bar_builder.py`, `warm_start.py`, `signal_client.py`,
`config_gui.py`, `account_manager.py`, `test_execution.py`.

⚠️ `run_live_signals.bat` → `src/nqbt/live/signal_generator.py` is also legacy and currently
**broken** (imports `Level` from `range_bars`, which was renamed `BarLevel`).

### `live/` planning + methodology docs

`GO_LIVE_PROGRESS.md`, `GAMMA_LEVELS_METHODOLOGY.md`, `scaling_plan.md`,
`GAMBLE_MILK_PLAYBOOK.md`, `reoptimization_guide.md`, `live_slow_path_audit.md`,
`mt5_setup_notes.md`, `mt5_external_watchdog_spec.md`, `od_green_sweep_top_configs.md`,
`morning_ema_cross_strategy.md`

## 🔷 NT8 Add-Ons → `nt8/`

- `NQMultiStratReceiver.cs` — **current** multi-strategy HTTP receiver
- `NQMultiStratReceiverTest.cs` — test addon (:8082) for the farm
- `NQOrderFlowSignalReceiver.cs` / `NQOrderFlowATRStrategy.cs` — legacy single-strategy versions

## 🐍 Core Python Package → `src/nqbt/`

Deliberately small — only what live code and cache builders still import:

- `analysis/range_bars.py` — range-bar construction (`BarLevel`, `RangeBar`, `build_range_bars`)
- `analysis/volume_profile.py` — volume profile / POC / value area
- `analysis/vwap.py` — VWAP + bands
- `data/normalizer.py` — raw MBP-1 DataFrame → clean trade/quote frames
- `data/schema.py` — `DATASET`, `SCHEMA`, `PRICE_SCALE`, `Action`, `Side`
- `data/loader.py` — Databento fetch/cache/load (currently unreferenced)
- `analysis/range_bars_streaming.py` + `live/signal_generator.py` — legacy, broken (see above)

The HMM/regime research cluster, the study registry, and the original `backtest/engine.py`
were removed on 2026-07-25 (unreachable from every entry point; recoverable from git history).

## 📜 Research & Strategy Scripts → `scripts/`

Each directory is a self-contained research thread. Scripts are standalone entry points
(`python scripts/<dir>/<script>.py`), not an imported library.

**Live strategies (backtest + optimization sources):**
- `rough vol orderflow/` — Rough Vol; core config, 3-way/4-way combined, PnL calendars
- `fabio_orb/` — Fabio ORB; sweeps, giveback variants, walk-forward
- `overnight drift strategy/` — OD; timeframe sweeps (incl. the 1hr config)
- `overnight range strat/` — B2 / overnight range break + gamma
- `rv_1min_pass_opt/` — RV retuned on 1-min for prop pass rate

**Prop-firm / capital modeling:**
- `montecarlo/` — the big MC library (per-firm sims, gambler's ruin, payout timing)
- `propfirm_milking/` — fixed-ATR-bracket milking configs
- `futurespropmc/` — futures eval/funded MC, DD-aware sizing
- `farm_income/` — farm throughput + income economics, 150k account MC
- `cfd prop firms/` — FundingPips-style CFD accounts

**Strategy research (concluded or shelved):**
- `orderflowmarketcontext/` — noise bands, options levels, POC/value-area regimes
- `value_area_revert_study/`, `fair_price_theory/`, `morning_ema_cross/`,
  `morning_delta_cross/`, `experimental_questions/`

**Infrastructure:**
- `cache_creation_scripts/` — all cache builders (Databento + MarketTick sources).
  `fetch_and_build_single_day.py` is the one-command path for a single session.
- `databentoDownloadScripts/` — batch DBN range downloads, DBN→parquet
- `thetadata/` — QQQ/NDX EOD greeks + `daily_pipeline/` (the 8am gamma refresh task)
- `overfit_framework/` — reusable overfit / walk-forward test suite
- `results/` — shared cross-strategy sweep outputs

## 📊 Results

- `results/` — legacy top-level results (VWAP-era configs, MC outputs, noise-band reports)
- `scripts/<strategy>/results/` — **current** per-strategy outputs; this is where new work lands

## 📄 Documentation → `docs/`

Start with `LIVE_AND_BACKTEST_GUIDE.md`. Also: `LIVE_TRADING_SETUP.md`,
`LIVE_STARTUP_GUIDE.md`, `LIVE_IMPLEMENTATION_SUMMARY.md`, `LIVE_STRATEGY_LOCKED.md`,
`WARM_START_GUIDE.md`, `BREAKEVEN_IMPLEMENTATION.md`, `OVERFIT_TEST_FRAMEWORK.md`,
`MONTECARLO_REPORT.md`, `OPTIMIZATION_GUIDE.md`, `NQ_vs_MNQ_REFERENCE.md`,
`VOLUME_PROFILE_STRUCTURE.md`, `15MIN_BARS_COMPARISON.md`, `EQUITY_CURVE_TEMPLATE.md`,
`smart_order_routing_plan.md`, `account_rotation_plan.md`

Note: `docs/*.md` is gitignored except `docs/README.md`.

## 💾 External Data → `D:/trading_pythonbacktest_data/`

Not in repo. All tick / bar / signal caches:
`dbn/`, `signal_cache/`, `vwap_cache/`, `timebars_5min/`, `vwap_reaction_cache/`,
`daily_profile_cache/`, `volumetric_5min_1tpl.parquet`, `markettick_1min_bars.parquet`

`data/` in-repo holds only `friend_trade_log_full.txt`; large `.dbn` files are transient.

## 🔧 Root

- `pyproject.toml` — package + deps
- `.env` / `.env.example` — `DATABENTO_API_KEY` (not committed)
- `run_live_signals.bat` — legacy launcher (broken, see above)

Regenerable research caches (`scripts/**/.cache/`, `scripts/rv_1min_pass_opt/cache/`,
`live/combined/state/replay/*.parquet`, `phase1_all_configs.csv`) are gitignored — they stay
on disk but out of git.

## Quick Access

```bash
# Start the current live system (4-way combined)
python live/combined/run_phase1.py

# Replay today's ticks through the same engines
python live/combined/replay_today.py

# Launch the multi-account farm app
live/farm/start_farm_app.bat

# Build caches for a new session
python scripts/cache_creation_scripts/fetch_and_build_single_day.py 2026-07-24

# Daily gamma refresh (needs ThetaData Terminal on :25503)
python scripts/thetadata/daily_pipeline/run_daily.py

# Main live/backtest reference
notepad docs/LIVE_AND_BACKTEST_GUIDE.md
```

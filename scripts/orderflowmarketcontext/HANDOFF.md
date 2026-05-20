# HANDOFF — Order Flow Market Context Work

Written 2026-04-24 for continuity across Claude Code sessions. Read this first
when resuming. Auto-memory in `~/.claude/projects/.../memory/` also applies
(schema reference, user prefs, etc) — that loads automatically.

## What we've built (current cache state)

All on `D:/trading_pythonbacktest_data/`:

### Raw primitive caches (no parameters — don't rebuild unless source changes)

| File | Size | Coverage | Notes |
|---|---|---|---|
| `volumetric_5min_1tpl.parquet` | 92 MB | 2020-12-01 → 2026-04-02 (1,393 sessions) | 5-min bars × per-0.25pt level (buy/sell vol). Session = 6pm-5pm ET. |
| `volumetric_1min_1tpl.parquet` | 219 MB | same | 1-min bars × per-0.25pt level. Enables any-N-min aggregation. |
| `cache/profiles/*.pkl` | ~220 MB combined | 2020-12-01 → 2026-04-08 | Developing volume profile per day, 1-min snapshots, cumulative from 6pm ET. `ticks_per_level=10` (2.5pt), `value_area_pct=0.68`. |
| `signal_cache_5yr/*.pkl` | — | 2020-12-01 → 2025-04-21 | 40-range volumetric bars. Rebuilt after MT parser fix. |
| `vwap_reaction_cache_5yr/*.pkl` | — | — | Derived from signal_cache_5yr. Also rebuilt. |

### Derived indices (no parameters — rebuildable quickly)

| File | Size | Purpose |
|---|---|---|
| `wick_delta_index_5min.parquet` | 20 MB | 1 row per 5-min bar, summaries: OHLC, `bar_total_delta`, `bot_wick_delta`, `top_wick_delta`, `bot_wick_low/top`, `top_wick_low/high`, `is_bullish`/`is_bearish`. |
| `wick_delta_buckets_5min.parquet` | 20 MB | 1 row per `(bar, wick_side, 5-tick bucket)`. Per-bucket buy/sell/delta. |
| `wick_delta_index_1min.parquet` | 82 MB | same as 5min but at 1-min. |
| `wick_delta_buckets_1min.parquet` | 56 MB | 1-min per-bucket detail. |
| `bar_context_1min.parquet` | 34 MB | Per-minute context: `dev_poc/vah/val` (session-developing, 6pm start), `noise_upper_14/lower_14/upper_90/lower_90`, `hod_eth_locked/lod_eth_locked`, `prev_rth_poc_d1/d2/d3`, `prev_week_iso_poc/vah/val`, `trailing_5d_poc/vah/val`, `is_rth/is_eth`. |
| `_tmp_db_trades_consolidated.parquet` | 752 MB | TEMP scratch file used during volumetric builds (deduped DB trade tape). Safe to delete if space is tight. |

### Parametric artifacts (regenerated with CLI args per sweep)

| File | Scope |
|---|---|
| `wick_zones_5min.parquet` | Zone lifecycle: `zone_id, zone_type (support/resistance), formed_at, zone_low, zone_high, run_bars, last_touch_time, invalidation_time, invalidation_reason`. Current: `min_wick_delta=80, min_run=2, stale_days=14`. |
| `wick_zone_touches_5min.parquet` | One row per zone touch event. |
| `signals_5min.parquet` | 8 named signal types (4 long, 4 short mirrors). Current: `min_wick_delta=80, strong_delta=200`. |
| `signals_1min.parquet` | same at 1-min TF. |

## Builder scripts (all in `scripts/cache_creation_scripts/`)

- `build_volumetric_5min_1tpl.py` / `build_volumetric_1min_1tpl.py` — raw caches. ~95 min per build, parallel 8 workers.
- `build_wick_delta_index.py --minutes 5` / `--minutes 1` — ~30 s each.
- `build_bar_context.py` — ~23 min (dominated by dev-profile iteration). Patched post-build to fix noise-band column lookup (keys are `date` objects, not strings).
- `build_wick_zones.py --min-wick-delta N --min-run N --stale-days N --suffix _tag` — ~17 s.
- `build_wick_signals.py --min-wick-delta N --strong-delta N --timeframe 5|1 --suffix _tag` — ~30 s.

## Data source status — important context

- **MarketTick parser was broken originally.** The old signal_cache_5yr builder treated `type=2` (L2 DOM depth events) as trades, producing 49× too many range bars. Fixed 2026-04-22 in `build_signal_cache_from_markettick.py`. Full schema in `~/.claude/projects/.../memory/reference_markettick_schema.md`: trades are `type=1 sub=2`, aggressor inferred via Lee-Ready from BBO (`type=1 sub=0/1`).
- **All MT-derived caches rebuilt after the fix.** Validation showed MT volume matches Databento trade volume within ~0.2-2% on overlap days (2025-04-15 sample: MT 520,354 vs DB 520,345).
- **8 Databento parquet files in `D:/trading_pythonbacktest_data/parquet/` are corrupt** (3 with 46-byte truncated download, 1 with bad footer, 1 stray 2024 file). Adjacent overlapping files cover the sessions, so no data loss. Consolidator skips them with warnings.

## The task in progress — "consolidation setup" backtest

### Signal definition (hybrid of `long_2bar_bothwicks` + strong-delta)

- **Bar N-1**: bearish close, `bot_wick_delta ≤ -min_wick_delta` (default 80)
- **Bar N**: bullish close, `bot_wick_delta ≤ -min_wick_delta`, `bar_total_delta ≥ strong_delta` (default 200)
- **Zone filter (strict)**: signal bar's low lies inside `[zone_low, zone_high]` of an active support zone (formed < signal_time, invalidation_time null or > signal_time)
- **Session filter**: RTH only (09:30-16:59 ET)

### Entry / SL / TP

- **Entry** = signal bar close + 0.125 pt adverse slippage
- **SL** = `signal_bar.bot_wick_low − 1 × ATR(14, 5-min)` − 0.125 pt slippage on fill
- **TP** = `entry + (entry − SL) + 1 × ATR` − 0.125 pt slippage on fill
- **ATR**: 14-period SMA of true range on 5-min bars
- **Exit**: SL or TP first hit intraday, else force-close at last RTH bar (16:59)
- **Costs**: $4.50 fixed RT commission+fees, slippage already baked into fills (~$5 RT)

### Backtest script + results

- Script: `scripts/orderflowmarketcontext/meanrevertingscripts/backtest_pullback_into_support.py`
- Run output: `scripts/orderflowmarketcontext/meanrevertingscripts/results_pullback_support/`
- Latest raw result: **340 trades, WR 40.9%, PF 1.10, +$56/trade, +$19k total** (5-year sample)
- Stacked filter (9-11 AM + in_va + inside both noise bands): **111 trades, 47.7% WR, PF 1.58, +$319/trade**
- Trade count is low — user flagged this is a context-gate strategy, not a standalone entry

### Breakdown by context (from the 340-trade raw sample)

- **Entry hour**: 10:00 AM is the clear winner (+$337/trade). Midday (12:00-14:00) loses money. 9:00 and 11:00 moderately positive.
- **Dev profile**: `in_va` (between VAH/VAL) wins (+$143/trade). Outside VA loses.
- **Noise bands**: `inside_both` (14d + 90d) wins (+$100/trade). `outside_both` is big loss (−$309/trade).
- **Prev-week ISO profile**: `in_week_va` wins (+$262/trade). Other buckets flat/negative.

### Open work — USER'S NEXT STEP

User wants to:
1. **Optimize the setup on IS data (2021 - 2024)**
2. **Test on OOS (2025 - April 2026)**

The existing script `backtest_pullback_into_support.py` runs the full sample.
To add IS/OOS:
- Filter trades by `signal_time.dt.year` for IS (2021-2024) and OOS (2025-2026)
- Re-run parameter sweeps on IS: `min_wick_delta ∈ {50, 80, 120}`, `strong_delta ∈ {150, 200, 300}`, ATR multipliers for SL/TP
- For each param combo, note IS metrics. Pick best-IS by a defensible rule (PF, expectancy, or combined).
- Report OOS metrics for the IS-picked params.
- **Do not tune on OOS.**

All the caches support this without rebuilding — param sweeps just rerun
`build_wick_signals.py` and `build_wick_zones.py` with different args and
`--suffix` to get multiple versions side-by-side, then rerun the backtest
script filtering to IS dates.

## Project folder the user is working in

`scripts/orderflowmarketcontext/`
```
orderflowmarketcontext/
├── HANDOFF.md                              ← THIS FILE
├── poc regime results/
│   └── README.md                           ← prior POC-trend regime analysis
├── meanrevertingscripts/
│   ├── backtest_pullback_into_support.py   ← current backtest
│   └── results_pullback_support/
│       ├── trades.csv
│       ├── by_entry_hour.csv
│       ├── by_dev_loc.csv
│       ├── by_noise_loc.csv
│       └── by_week_loc.csv
├── trendingscripts/                        (empty — for future trend strategies)
└── combination scripts/                    (empty — for multi-signal combinations)
```

Note: the folder names have spaces (`poc regime results`, `combination scripts`).
User confirmed this is intentional. Paths need quoting in shells / scripts.

## Design decisions on record (don't relitigate)

- **Wick definitions**: bullish candle bottom wick = `[low, open]`, top = `[close, high]`. Bearish bottom = `[low, close]`, top = `[open, high]`. All candles have both wicks.
- **Zone run rule**: extend only if new candle's wick overlaps running zone range. Run ends on first non-qualifying candle. Min run length 2. Zone must include at least one bullish close (support) or bearish close (resistance).
- **Zone invalidation**: ETH body-close through zone boundary = dead. RTH close-through = also dead but represents potential break-setup entry. Staleness: 14 trading days since last touch (clock resets on touch).
- **Developing profile**: cumulative from 6pm ET session start (ETH included). The user specifically said this is what they want for context, not RTH-only developing.
- **Signal pattern naming** (8 types — DON'T collide with "L2 = Level 2 market data"):
  - `long_1bar_abs`, `long_2bar_N1wick`, `long_2bar_Nwick`, `long_2bar_bothwicks`
  - `short_1bar_abs`, `short_2bar_N1wick`, `short_2bar_Nwick`, `short_2bar_bothwicks`
- **Aggressor inference**: MT uses Lee-Ready from BBO. DB uses `side='B' → buy`, `side='A' → sell`.
- **Only best-bid/best-ask trades matter for absorption.** No DOM depth events contaminate delta calcs.

## Useful quick-reference commands

```powershell
# Inspect a cache:
python -c "import pandas as pd; print(pd.read_parquet('D:/trading_pythonbacktest_data/wick_delta_index_5min.parquet').head())"

# Regenerate signals with different thresholds:
python scripts\cache_creation_scripts\build_wick_signals.py --min-wick-delta 60 --strong-delta 150 --timeframe 5 --suffix _looser

# Regenerate zones with different thresholds:
python scripts\cache_creation_scripts\build_wick_zones.py --min-wick-delta 50 --stale-days 21 --suffix _loose

# Rerun consolidation backtest:
python "scripts/orderflowmarketcontext/meanrevertingscripts/backtest_pullback_into_support.py"

# Watch a long-running build:
Get-Content C:\trading\nqorderflowbacktester\logs\build_SOMETHING.log -Wait -Tail 20
```

## Feedback preferences the user has consistently given

- Delete one-off diagnostic scripts after running (noted in auto-memory).
- Prefer query-time filters over rebuilt caches. "Everything parametric should
  be regenerable from the primitives."
- Folder names with spaces are OK — don't reformat without asking.
- When confused, ask specific questions with concrete examples, not abstract
  taxonomies.

## Known gotchas

- `bar_context_1min.parquet` noise-band columns are only populated during
  09:30-16:45 ET (sigma cache's session window) and for dates within the
  sigma cache range (sigma14: 2020-12-17 → 2025-11-29 / sigma90: 2021-03-24 → 2025-11-29).
  Outside that, all four noise columns are NaN.
- MT 1-min parquet (`markettick_1min_bars.parquet`) ends 2025-12-01. For analysis
  past that date, use `volumetric_1min_1tpl.parquet` or the Databento consolidated.
- The `long_2bar_bothwicks` signals cache does NOT include the `strong_delta`
  filter — that's applied at query time (filter `signal_bar_total_delta ≥ N`).
- Zone-lifecycle tracking uses nanosecond int comparisons to avoid
  timezone-naive/aware datetime64 collision (previous bug, since fixed).

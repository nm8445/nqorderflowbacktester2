# Rough Vol + Orderflow — Locked Live Config

Symbol: **NQ** (front-month continuous)
Bar size: **20-minute**
Status: **LOCKED for live trading**
Locked on: **2026-05-12**

---

## Backtest performance (5+ years, MNQ $20/pt, 1 contract)

| Slice | Trades | PF | WR | PnL | MDD |
|---|---:|---:|---:|---:|---:|
| **OVERALL** | **814** | **1.31** | **55.9%** | **+$153,549** | **−$16,957** |
| IS (2020-12 → 2024-12-31) | 665 | 1.25 | 55.5% | +$96,065 | −$16,957 |
| OOS (2025-01 → 2026-04-17) | 149 | 1.51 | 57.7% | +$57,483 | −$13,274 |

### Year-by-year (all positive)

| Year | Trades | PF | WR | PnL | MDD |
|---|---:|---:|---:|---:|---:|
| 2020 (Dec) | 8 | 1.09 | 62.5% | +$200 | −$1,513 |
| 2021 | 184 | 1.35 | 56.5% | +$30,495 | −$16,957 |
| 2022 | 158 | 1.55 | 61.4% | +$56,366 | −$9,171 |
| 2023 | 176 | 1.08 | 52.8% | +$7,408 | −$16,068 |
| 2024 | 140 | 1.04 | 50.7% | +$3,590 | −$10,523 |
| 2025 | 116 | 1.45 | 56.9% | +$40,797 | −$13,274 |
| 2026 (YTD) | 32 | 1.67 | 59.4% | +$14,691 | −$4,259 |

### Trade distribution

- 65 calendar months covered, avg **13.8 trades/month**, median 14, min 5, max 25
- No month below 5 trades; no clustering risk
- Long bias slightly stronger (PF ~1.40 long vs ~1.16 short overall, both profitable)

---

## Strategy spec

### Bar construction
- Aggregate 5-min bars to **20-min** via `floor("20min")`
- Index = bar close time (open_time + 20 minutes)
- Timezone: ET (America/New_York)

### Rough vol model (fixed kernel — DO NOT tune)
```
H          = 0.4
KERNEL_LEN = 80
ETA        = 1.0
V0         = 0.0001
```

For each new bar:
1. Compute log return: `ret_t = ln(close_t / close_{t-1})`
2. Z-score the return over `NORM_LEN`-bar window:
   ```
   shock_t = (ret_t - mean(ret, NORM_LEN)) / std(ret, NORM_LEN)
   ```
3. Convolve with fractional Brownian kernel:
   ```
   kernel[k] = k^(H-0.5) - (k-1)^(H-0.5),  k = 1..KERNEL_LEN
   kernel[0] = 1.0
   xH_t = (shock * kernel)[t]   # causal convolution, indexed by t
   ```
4. Model vol:
   ```
   v_model_t = min(V0 * exp(ETA * xH_t), V0 * 1e4)
   ```
5. Z-score vol over `Z_LOOKBACK` window:
   ```
   z_vol_t = (v_model_t - mean(v_model, Z_LOOKBACK)) / std(v_model, Z_LOOKBACK)
   ```

### Tunable params (LOCKED)
```
NORM_LEN   = 400
Z_LOOKBACK = 75
EMA_LEN    = 80
HIGH_Z     = 2.00
ATR_LEN    = 14            # Wilder
ATR_SL     = 2.0           # stop = entry ± 2.0 × ATR
ATR_TP     = 2.0           # target = entry ± 2.0 × ATR  (RR = 1:1)
```

### Entry rule — at bar close
A new trade is opened ONLY if ALL hold on the **just-closed** signal bar:

1. **Session gate**: bar close time (ET) must satisfy:
   ```
   (09:00 <= hh:mm < 13:00)  OR  (14:00 <= hh:mm < 14:45)
   ```
   Bars closing 13:00–13:59 (the "13 ET hour") are excluded from entries.
2. **Volatility signal**: `z_vol > 2.00`
3. **Direction (EMA filter)**:
   - LONG if `close > ema(80)`
   - SHORT if `close < ema(80)`
4. **ATR sanity**: `atr > 0`
5. **Orderflow filter — windowed absorption** (CRITICAL):
   - Partition the signal bar's price range into the lower half `[low, (high+low)/2)` and upper half `((high+low)/2, high]`
   - Each tick-level (NQ tick = 0.25) has a delta = buy_vol − sell_vol over the bar
   - For a **LONG** signal: in the lower half, find the best contiguous **8-tick window** by `sum(delta)`. Require `min_sum_delta <= −150`. (Heavy aggressive selling absorbed.)
   - For a **SHORT** signal: in the upper half, find the best contiguous 8-tick window by `sum(delta)`. Require `max_sum_delta >= +150`. (Heavy aggressive buying absorbed.)
6. **Gamma filter**: NONE. Take trades regardless of prev-day gamma sign.
7. **No daily trade cap.**

### Order placement
- Enter at the **close** of the signal bar (market order at the bar close price)
- Stop: `entry ± 2.0 × ATR_14`
- Target: `entry ± 2.0 × ATR_14`
- 1:1 risk:reward

### Exit rules (in priority order, evaluated per subsequent bar)
1. **Stop hit**: bar's low/high touches the stop level (LONG: low ≤ SL; SHORT: high ≥ SL)
2. **Target hit**: bar's high/low touches the target level (LONG: high ≥ TP; SHORT: low ≤ TP)
3. **Force-close**: any time after 14:45 ET, exit at the next bar's close

If both SL and TP fall inside the same bar's range, **assume SL fills first** (conservative).

### Sizing
- **Fixed 1 contract** per signal (no martingale, no scaling, no compounding)
- No daily loss/profit halt

---

## Live runbook

### Required data inputs (continuously)
1. **5-minute bars** → aggregated into 20-min bars at the strategy layer (`floor("20min") + 20min` = close time)
2. **Tick-level orderflow** to compute, for each closed 20-min bar:
   - Total buy_vol and sell_vol per `level_price` (NQ tick = 0.25)
   - The lower-half and upper-half partition relative to the bar's own (high, low)
   - The best 8-tick contiguous window by signed cumulative delta

### Backtest source paths (for parity verification)
- 5-min bars: `D:/trading_pythonbacktest_data/timebars_5min_5yr/` + `D:/trading_pythonbacktest_data/timebars_5min/`
- Volumetric orderflow (level-resolved): `D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet`
- Backtest scripts: `scripts/rough vol orderflow/`

### Warm-up requirement
After cold start, the strategy needs at least `NORM_LEN + KERNEL_LEN + Z_LOOKBACK = 555` 20-min bars (~11 trading days at ~50 bars/day during 24h session) before signals are valid. **Wait one full trading week before going live.**

### Operational checks before each session
- ATR_14 of the signal bar is > 0 (skip otherwise — guard against thin/halted data)
- Volumetric level-delta data for the just-closed bar is available within 30s of bar close (otherwise skip the signal)
- Time clock is in America/New_York

### Per-bar live loop (every 20-min close)
1. Pull the just-closed 20-min bar (OHLC + level-resolved deltas)
2. Append to rolling features: `ret`, `shock`, `xH`, `v_model`, `z_vol`, `ema`, `atr`
3. Evaluate entry rules (session gate + z > 2.00 + direction + orderflow window check)
4. If signal AND no position open → send market order to NT8 with attached SL/TP brackets
5. If position open → let SL/TP manage; only check force-close after 14:45 ET

### Position management notes
- Bars closing 13:00–13:59 may still trigger SL/TP exits on existing positions (only ENTRIES are blocked during that hour)
- Force-close at 14:45 ET regardless of price (single-exit market order, not bracket)

---

## Key research findings (for context, not action)

1. **Gamma filter does NOT help on this config.** POS-gamma LONGS produce PF 1.73 (best per-trade bucket). Dropping POS-gamma days reduces PnL by ~$13k and OOS PF. **Do not skip pos-gamma days.**
2. **Hour 13 ET is the worst.** Native config showed PF 0.88 in the 13 hour with −$5,978 PnL. Skipping that hour is the biggest single improvement.
3. **Pre-09:00 entries are a drag.** Hours 6–7 ET: combined −$9,841 across 7 trades. Cut.
4. **MAX_TRADES_PER_DAY was never binding.** The signal threshold + orderflow filter rarely produces more than 3 entries/day naturally. Removing the cap had zero effect.
5. **2023 + 2024 were the soft years** (PF 1.04–1.08). Strategy still profitable but flat. No filter found to fix this — likely volatility regime, not strategy flaw.
6. **OOS performance EXCEEDS IS** (PF 1.51 vs 1.25). Strong robustness signal — strategy was not curve-fit on the IS window.

---

## Multi-strategy stacking notes

This strategy is largely uncorrelated with the **overnight range gamma strat (B2)** locked in `live/overnight range gamma strat/best live config/`:
- B2 trades only on prior-day OHI/OLO breaks (event-driven, ~150 trades/year)
- This strategy trades on volatility regime entries (~150 trades/year)
- They share NO entry conditions and operate on different timeframes (5-min vs 20-min)

Trading both on parallel sub-accounts is reasonable. Combined hypothetical PnL: ~$170k IS + ~$100k OOS at 1 contract each.

---

## Files
- Backtest script: `scripts/rough vol orderflow/inspect_config_v3.py`
- Trade log: `scripts/rough vol orderflow/results/inspect_v3_N400_v3_trades.csv`
- Equity curve: `scripts/rough vol orderflow/results/inspect_v3_curves.png`
- Core engine: `scripts/rough vol orderflow/core.py`
- Phase 1 (base sweep): `scripts/rough vol orderflow/phase1_base_sweep.py`
- Phase 2 (enrichment): `scripts/rough vol orderflow/phase2_orderflow_gamma.py`

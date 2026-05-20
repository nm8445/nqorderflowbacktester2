# Overnight Range Gamma Strat — Locked Live Config

**Locked**: 2026-05-12
**Backtest range**: 2020-12 → 2026-05-06 (5+ years)
**Source script**: `scripts/overnight range strat/scripts/lock_v2_k08_lock045_mart_fc_filtered.py`
**Full results**: `scripts/overnight range strat/tradelogs/robust_configs/locked_v2_k08_lock045_mart_fc_filtered.txt`

---

## Strategy summary

A range-break / momentum-continuation NQ futures strategy that fires after the overnight range (OHI/OLO) is broken with confirmation. Uses MenthorQ-style gamma levels (QQQ + NDX, mapped to NQ via daily ratio/basis) as entry "near-level" anchors, a ratchet stop on 20-min bars, and a fixed-distance TP. Sizing uses a small post-force-close martingale that doubles only after a force-close loss.

---

## Entry (5-min bars)

```
variant          B2  (pinbar wick-anchored, single candle)
pinbar X         0.75    (wick / body ≥ X)
window N         15      ticks for orderflow absorption scan
delta D          70      |buy − sell| in best N-tick window
strict           True    require SHORT close < OLO and LONG close > OHI
band_K           0.25    proximity band = clip(0.25 × ATR_5min, 5, 20) NQ pts

confirmation     conf_N = 5,  conf_D = 75,  mode = HALF
                 (half-bar windowed delta must be ≥ conf_D in trade direction)

Bias logic (sticky):
  - 3 consecutive 5-min closes outside today's OHI/OLO → bias set
  - LONG_BREAK never auto-resets
  - SHORT_BREAK → can flip to LONG after 5 consecutive inside closes
                  (max 1 flip per day)
```

## Filters (applied at entry time)

```
session hours    entry must occur between 09:00 and 14:59 ET (drop 15:xx, 16:xx)
gamma regime     drop SHORT entries when prior-day EOD qqq_gamma_sign == +1
                 (POS-gamma days suppress mean-reversion shorts)
dedupe           chained Mode 1 by actual exit times — one trade in flight at a time
```

## Exit (20-min bar management)

```
TP (fixed):      entry ± 2.0 × ATR_at_entry  (ATR uses 20-min bars, EWMA period 14)

SL (ratchet + MFE guard):
  yellow_val      = close − sign × 2.5 × ATR_14_20min
                    (ratchet — never moves against position; uses prior 20-min bar's ATR)
  MFE-guard arms  when peak_MFE ≥ 0.8 × TP_distance during the trade
  mfe_stop        = entry ± 0.45 × peak_MFE
  effective_stop  = MORE FAVORABLE of (yellow_val, mfe_stop)
  trigger         close beyond effective_stop AND adverse-direction bar

Force close:     16:00 ET (RTH end) — exit at the 20-min bar's close
```

## Sizing (martingale FC-only)

```
default size              1 contract
FC loss (force-close)     next trade = 2 contracts
after 2-contract trade    reset to 1 (regardless of outcome)
SL_TRAIL loss             no size change (stays at 1)
TP wins                   stay at 1 until next FC loss
maximum size ever         2 contracts
```

## Instrument

```
primary       MNQ (Micro NQ)  — $2/pt, $0.50 tick, 0.25 pt tick size
              chosen for prop-firm DD safety
NQ basis      $20/pt for sizing math if scaled up
```

---

## Backtest performance (LEAK-AUDITED — see Notes section)

> Note: the *original* txt file in `tradelogs/robust_configs/locked_v2_k08_lock045_mart_fc_filtered.txt`
> uses a 20-min ATR that included the entry-spanning bar (small ~7% EWMA-weighted
> lookahead). Audited leak-fixed numbers below.

### With martingale FC-only (scaled)

| Period | n | WR | PF | Sharpe | Total NQ pts | Total $ MNQ | Max DD MNQ |
|---|---|---|---|---|---|---|---|
| IS (2020-12 → 2024-12) | 545 | 60.9% | 1.31 | 1.93 | +5,286 | +$10,572 | $-1,827 |
| OOS (2025-01 → 2025-11) | 111 | 58.6% | **1.42** | **2.24** | +1,787 | +$3,574 | $-1,612 |
| Combined | 656 | 60.5% | 1.33 | — | +7,072 | **+$14,145** | $-1,827 |

OOS PF (1.42) > IS PF (1.31) — robustness signature held after leak fix.

### No-mart baseline (1-contract reference)

| Period | n | WR | PF | Total NQ pts | $ MNQ |
|---|---|---|---|---|---|
| IS | 545 | 60.9% | 1.30 | +4,482 | +$8,964 |
| OOS | 111 | 58.6% | 1.25 | +958 | +$1,916 |
| Combined | 656 | 60.5% | 1.29 | +5,440 | +$10,881 |

---

## Data dependencies (must be current for live)

| Resource | Path / source | Refresh cadence |
|---|---|---|
| NQ 1-min bars | `D:/trading_pythonbacktest_data/markettick_1min_bars.parquet` | nightly from Databento DBN |
| NQ 5-min level-resolved volume | `D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet` | nightly |
| Overnight range (OHI/OLO) per day | `scripts/overnight range strat/scripts/parquets/range_break_full_sequence_per_day.parquet` | recomputed nightly |
| MenthorQ NQ-mapped levels (QQQ + NDX) | `D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet` | nightly via `build_menthorq_levels_bulk.py` after QQQ/NDX ThetaData EOD download |
| QQQ ThetaData (greeks + OI) | `D:/trading_pythonbacktest_data/QQQ_thetadata/{YYYY-MM-DD}/` | nightly via `download_qqq_eod.py` (requires local ThetaData terminal running on `127.0.0.1:25503`) |
| NDX ThetaData | `D:/trading_pythonbacktest_data/NDX_thetadata/{YYYY-MM-DD}/` | nightly via `download_ndx_eod.py` |

---

## Live runbook

### Pre-market (before 09:30 ET)
1. Confirm yesterday's MenthorQ levels are loaded in chart (auto if the parquet pipeline ran overnight)
2. Confirm overnight range (OHI/OLO) is computed and visible
3. Check QQQ gamma sign for today (prior-day EOD `qqq_gamma_sign`) — if `+1` (POS), suppress SHORT entries for the day

### Intraday signal recognition
A trade fires when ALL of these are true on a 5-min bar (the "signal bar"):
1. Bias is set in the trade direction (3 consecutive 5-min closes above OHI for LONG, below OLO for SHORT) OR a SHORT_FLIP_LONG has occurred
2. Bar is a pinbar in the trade direction: wick / body ≥ 0.75, where the relevant wick is below body for LONG / above body for SHORT
3. Bar overlaps a MenthorQ level zone (within `clip(0.25 × ATR_5min, 5, 20)` NQ pts of any of: qqq/ndx CR, PS, HVL, HVL_0DTE, g1–g10)
4. Approach filter: prior bar's close was on the far side of the level (close > level for LONG, close < level for SHORT)
5. Strict-close filter: signal bar's close > OHI (LONG) or close < OLO (SHORT)
6. Orderflow absorption: best 15-tick window in the wick has |delta| ≥ 70 (sellers absorbed for LONG, buyers absorbed for SHORT)

**Confirmation bar** (next 5-min bar after the signal):
- Must close in the trade direction (bullish for LONG, bearish for SHORT)
- HALF-window delta over 5 ticks ≥ +75 (LONG) or ≤ −75 (SHORT)

**Entry**: open of the 5-min bar AFTER the confirmation bar
- Equivalent: market order at confirmation bar close → fill at next bar open

### Filter gates before entry
- **Hour gate**: entry timestamp's hour must be in {9, 10, 11, 12, 13, 14}. Skip otherwise.
- **Gamma gate**: if entry is SHORT and prior-day `qqq_gamma_sign == +1`, skip.

### Position management
- **TP**: entry + 2.0 × ATR_at_entry (LONG) or entry − 2.0 × ATR_at_entry (SHORT). ATR computed from 20-min bars using EWMA period 14, using the 20-min bar that CLOSED before the 5-min entry bar (not the one containing entry).
- **Yellow ratchet stop**: at each 20-min bar close, compute `raw_yellow = close − sign × 2.5 × ATR_14_20min`. The active yellow level only ratchets in the favorable direction (max for LONG, min for SHORT) — never moves against position.
- **MFE guard**: once `peak_MFE ≥ 0.8 × |TP − entry|`, also compute `mfe_stop = entry + sign × 0.45 × peak_MFE`. Effective stop = more favorable of (yellow_val, mfe_stop).
- **Stop trigger**: a 20-min bar's CLOSE must be beyond effective stop AND the bar must be in the adverse direction (close < open for LONG stop; close > open for SHORT stop). Exit at that close price.
- **Force close**: at 16:00 ET, exit at that 20-min bar's close regardless of TP/SL state.

### Sizing
- Track size state across trades:
  - Start of day or after a 2-contract trade: size = 1
  - After a FORCE_CLOSE loss while at size 1: size = 2 for the next entry
  - After a SL_TRAIL or TP outcome at size 1: size stays at 1
- **Maximum size ever**: 2 contracts. Never escalate beyond.

### After exit
- Apply chained Mode 1: do not enter a new trade until the previous one has fully exited (TP, SL, or FC).
- Next eligible signal after the prior exit time is in play.

---

## Notes & known caveats

1. **ATR-leakage audit**: the original simulation used the 20-min bar CONTAINING the 5-min entry, which has ~75% of its data after entry time. This contributed ~7% EWMA weight of lookahead to the ATR used for SL/TP sizing. **Leak-fixed numbers** (above) use the most recently CLOSED 20-min bar's ATR. The OOS-stronger-than-IS PF signature survived the fix.

2. **POC/CR/PS/HVL definitions** in MenthorQ:
   - `cr` = strike with max calls-only gamma, ≥ spot, 0-45 DTE
   - `ps` = strike with max puts-only gamma, < spot, 0-45 DTE
   - `hvl` = nearest cumulative-GEX sign-flip strike within ±5% of spot, 0-45 DTE
   - `hvl_0dte` = same flip, but using only the 1-2 DTE chain (= 0-1 DTE from NTD perspective)
   - `g1`-`g10` = top-10 strikes by |Net GEX|/max + |Net DEX|/max in 0-1 DTE chain, excluding CR/PS/HVL/HVL_0DTE
   - `g1_alldte`-`g10_alldte` = same ranking on the 0-45 DTE chain (added 2026-05; not used by this strategy but available for comparison studies)

3. **Why hour 9-14 only**: hour-by-hour analysis showed hours 15-16 entries had PF < 1 in IS and the slight contribution wasn't worth the late-day variance. Hour 9 has highest PF (~1.94), hour 10 is the volume driver, hours 11-14 fill in the rest.

4. **Why drop POS-gamma SHORTs**: prior-day `qqq_gamma_sign = +1` indicates dealers net-long gamma, which dampens intraday volatility. Mean-reversion shorts in POS regime have historically underperformed (the gap closes slower or not at all). LONGS are unaffected by gamma sign in this strategy.

5. **Per-year P&L** (combined IS+OOS, mart-scaled, leak-fixed approximation):
   - 2020 (Dec only): +$1,540 NQ / +$154 MNQ
   - 2021: +$17,128 / +$1,713
   - 2022: +$42,294 / +$4,229
   - 2023: +$14,006 / +$1,401
   - 2024: +$24,384 / +$2,438
   - 2025 (through Nov): +$13,128 / +$1,313
   - 2026 (Jan-May 6): +$31,366 / +$3,137

6. **Worst dry months** are typically rangebound or late-break months (e.g., May 2021 took only 1 trade — bulk of B2 candidates fired at 15:xx after the hour filter cutoff).

---

## Files of interest

- `scripts/overnight range strat/scripts/lock_v2_k08_lock045_mart_fc_filtered.py` — script that produces the locked-config writeup
- `scripts/overnight range strat/scripts/sweep_ratchet_sl_fixed_tp.py` — entry filter helper (`filter_pre_dedupe`)
- `scripts/overnight range strat/scripts/test_pure_ratchet_exits.py` — 20-min bar builder + force-close time
- `scripts/overnight range strat/scripts/range_break_entry_signal_study.py` — bias-path computation, MenthorQ build helpers
- `results/html/locked_filtered_pnl_calendar.html` — interactive PnL calendar + equity curve with risk slider

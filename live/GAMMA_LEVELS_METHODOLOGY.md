# Gamma Levels Methodology — Reference

**Purpose:** This document captures EXACTLY how the MenthorQ-style gamma levels
that the B2 (Overnight Range Gamma) strategy depends on are computed. The
locked B2 backtest, signal generator, and live engine ALL read the same parquet
produced by this pipeline. If the methodology changes, B2 trade selection
changes — re-validate everything end-to-end.

**Last revised:** 2026-05-17

---

## 1. Output file

```
D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet
```

One row per trading day. 50 columns. Levels are mapped from QQQ/NDX strike
space into NQ price space using day-by-day ratio (QQQ) or basis (NDX). Columns
ending in `_nq` are the NQ-space level prices the strategy consumes.

---

## 2. The script that builds it

```
scripts/thetadata/daily_pipeline/build_menthorq_levels_bulk.py
```

Helpers it imports:
```
scripts/thetadata/daily_pipeline/menthorq_style_levels.py
  - qqq_strikes(date, (dte_lo, dte_hi))     -> (strikes_df, spot, atm_iv)
  - ndx_strikes(date, (dte_lo, dte_hi))     -> (strikes_df, spot, atm_iv)
  - expected_move(spot, iv)                 -> 1D EM = spot * iv * sqrt(1/252)
  - hvl_from_gex(strikes, spot)             -> cumulative-GEX sign-flip strike
  - menthorq_levels(strikes, spot, em, legacy_mode=True)
        -> (cr_tuple, ps_tuple, gex_list_of_10)
```

Run it directly:
```powershell
python scripts/thetadata/daily_pipeline/build_menthorq_levels_bulk.py
```

Daily incremental updater (only adds new days):
```
live/combined/gamma_refresh.py
```

---

## 3. NQ price snapshot

The mapping from QQQ/NDX strikes to NQ space requires a same-day NQ price.

- **Time:** 16:05 ET (close of the 1-min bar that opens at 16:04 and closes
  at 16:05). Equivalent to the close of the 5-min bar opening at 16:00.
- **Source:** `D:/trading_pythonbacktest_data/markettick_1min_bars.parquet`
- **Constants in `build_menthorq_levels_bulk.py`:**
  ```python
  SETTLE_HOUR = 16
  SETTLE_MIN  = 5
  ```

This time was chosen because that's what the ORIGINAL signal generation
parquet (`entry_signal_trades.parquet`) used. Do NOT change it — backtests
and live engine will diverge.

---

## 4. Expiry (DTE) rules per leg

### QQQ

| Use | DTE filter | Why |
|---|---|---|
| CR (call resistance) | 0-45 (full chain) | Methodology default |
| PS (put support) | 0-45 (full chain) | Methodology default |
| IV / EM | 0-45 (full chain) | ATM IV used for 1D expected move |
| HVL (gamma flip) | 0-45 (full chain) | `qqq_hvl_nq` column |
| HVL 0DTE (next-trading-day) | shift to NTD's session | Lookahead-free; `qqq_hvl_0dte_nq` column |
| GEX 1-10 | 0-1 DTE | `qqq_g1_nq` ... `qqq_g10_nq` columns |
| GEX 1-10 (all DTE, reference only) | 0-45 (full chain) | `qqq_g1_alldte_nq` ... `qqq_g10_alldte_nq` |

### NDX

| Use | DTE filter | Why |
|---|---|---|
| CR | 0-45 (full chain) | See note below |
| PS | 0-45 (full chain) | See note below |
| IV / EM | 0-45 (full chain) | ATM IV |
| HVL | 0-45 (full chain) | Used internally only (NDX HVLs not exported — pre-2023 expirations were inconsistent) |
| GEX 1-10 | **0-45 (full chain)** | **NOT 0-1 DTE.** User-confirmed reason: "for ndx it was all expiry because when i was backtesting i couldnt make ndx levels be 0-1 because it took a while for that expiry to even exist." So the backtest period (2020-12 → 2024-12) was run with NDX full chain GEX. Live must match. |

---

## 5. Legacy-mode level definitions

`menthorq_levels(..., legacy_mode=True)` matches the OLD methodology that
generated the original `entry_signal_trades.parquet`. This is the only mode
B2 backtest/live should ever use.

### Call Resistance (CR)
- Restrict to strikes WHERE `strike >= spot`.
- Pick the strike with the MAX `net_gex` (NOT calls-only gamma — uses NET GEX).
- Note: this is different from "modern" MenthorQ methodology which uses
  calls-only gamma. Backtest uses net_gex.

### Put Support (PS)
- Restrict to strikes WHERE `strike < spot`.
- Pick the strike with the MIN (most negative) `net_gex`.
- Same note: uses NET GEX, not puts-only.

### GEX 1..10
- Take all strikes EXCEPT the CR and PS strikes (those are reported
  separately — `CR and PS are SEPARATE from the GEX 1 to 10 levels`).
- Sort by `|net_gex|` descending.
- Take top 10.
- NO Expected Move window restriction.
- NO exclusion of HVL.

### HVL (Highest Volume Level / Gamma Flip)
- Strike where cumulative `net_gex` (sorted by strike) flips sign.
- Restricted to within ±5% of spot in `hvl_from_gex()`.
- Extended-search variant (no distance constraint) also computed:
  `qqq_hvl_extended_nq` and `qqq_hvl_dist_pct`.

### `net_gex` definition
- `net_gex = call_gamma * call_OI - put_gamma * put_OI` (per strike, summed
  across expirations in the DTE window)
- The signs in this convention: positive net_gex above spot → resistance.
  Negative below spot → support.

---

## 6. NQ-space mapping

For each level strike, convert into NQ price units:

### QQQ → NQ
```
qqq_ratio = NQ_settle_16:05ET / QQQ_spot_4:15PM_settle
qqq_level_in_nq_units = qqq_strike * qqq_ratio
```
Stored as `qqq_ratio` column. Recomputed per day.

### NDX → NQ
```
ndx_basis = NQ_settle_16:05ET - NDX_spot_4:15PM_settle
ndx_level_in_nq_units = ndx_strike + ndx_basis
```
Stored as `ndx_basis` column. Recomputed per day. This is the futures
cost-of-carry premium (NQ futures price > NDX spot).

---

## 7. Lookahead-free design

- Trade day D's signals use **D-1's EOD greeks** (which are available before
  D's RTH open).
- The locked B2 backtest (`lock_v2_k08_lock045_mart_fc_filtered.py`) and
  signal study both implement this via `attach_gamma_to_candidates(...)`
  and `levels_for_date(mq, prior_d)`.
- Daily refresh script (`live/combined/gamma_refresh.py`) is run at or after
  16:05 ET so D's row is available for D+1's session.

---

## 8. Output columns (50 total)

Per-day single row keyed on `date`:

### Day-level scalars (10 cols)
```
date, qqq_spot, ndx_spot, nq_settle,
qqq_ratio, ndx_basis,
qqq_iv, ndx_iv, qqq_em, ndx_em
```

### HVL columns (5 cols — QQQ only; NDX HVLs not exported)
```
qqq_hvl_nq           # full-chain HVL mapped to NQ
qqq_hvl_0dte_nq      # next-trading-day-DTE HVL mapped to NQ
qqq_hvl_extended_nq  # no-distance-constraint HVL
qqq_hvl_dist_pct     # distance pct of extended HVL from spot
qqq_gamma_sign       # +1 if cum_GEX(spot) > 0, -1 if < 0, 0 if zero
qqq_hvl_dte_filter   # string "(shift,shift+1)" — for audit
```

### MenthorQ levels mapped to NQ (24 cols)
```
qqq_cr_nq, qqq_ps_nq, qqq_g1_nq ... qqq_g10_nq   # 12 from QQQ (0-1 DTE GEX)
ndx_cr_nq, ndx_ps_nq, ndx_g1_nq ... ndx_g10_nq   # 12 from NDX (full chain)
```

### Additional "all DTE" GEX from QQQ (10 cols, reference only)
```
qqq_g1_alldte_nq ... qqq_g10_alldte_nq
```

The signal study reads ALL columns ending in `_nq` that start with `qqq_` or
`ndx_` as candidate levels. See `range_break_entry_signal_study.py:117`.

---

## 9. What CHANGED from the original (and how we recovered)

### Original issue (2026-05-16/17)
After cleaning up scripts, the rebuilt `menthorq_levels_nq.parquet` did NOT
match the level values in the original `entry_signal_trades.parquet`. Root
causes identified:

1. **Snapshot time:** was 16:00 in one build, 16:05 in another. Fixed by
   pinning to 16:05 (close of 1-min bar ending 16:05).
2. **NDX DTE for GEX:** was using 0-1 DTE in one build. Original used full
   chain because 0-1 DTE NDX options didn't exist reliably in pre-2023.
3. **Methodology:** Modern MenthorQ uses calls-only/puts-only gamma + EM
   window restriction + GEX/DEX combined ranking. Original code used
   `net_gex` for CR/PS and `|net_gex|` ranking with no EM window. Added
   `legacy_mode=True` flag to reproduce.
4. **CR/PS in GEX list:** Modern MenthorQ excludes them from the GEX 1-10
   ranking. Legacy methodology ALSO excludes them ("cr and ps are separate
   from the gex 1 to 10 levels"). Both reproduce this.

### Even after fixes, exact numerical match was NOT achieved
The legacy_mode rebuild matches the methodology PARAMETERS but the actual
greek values now (after ThetaData re-downloads or pipeline updates) produce
slightly different `net_gex` per strike than the values that were live when
the original signal parquet was generated. So `tp_1.00_idx` and `sl_1.00_idx`
columns in the signal parquet are NOT bit-exact reproducible.

### Resolution
- Accept the legacy_mode rebuild as the new baseline.
- Regenerated `entry_signal_trades.parquet` from this baseline.
- Re-ran locked B2 backtest. New PnL is slightly lower than the previous
  numbers reported on the combined-strategy chart (~$47K lower for B2 over
  the full IS+OOS period).
- Live engine uses the SAME parquet, so live and backtest are now consistent.

---

## 10. Live operation

1. **Daily refresh** (~16:10 ET, after settle):
   ```powershell
   python live/combined/gamma_refresh.py
   ```
   Appends the current day's row to `menthorq_levels_nq.parquet`.

2. **Live engine reads** the prior-day row at startup for the current
   session's gamma_sign and level set (lookahead-free).

3. **Backtest validation:** locked B2 reads the same parquet via
   `attach_gamma_to_candidates` → `prior_mq(date)` → prior-day row.

---

## 11. If you ever need to rebuild from scratch

```powershell
# Wipes the parquet and rebuilds from all available ThetaData EOD days
python scripts/thetadata/daily_pipeline/build_menthorq_levels_bulk.py

# Regenerate the signal pipeline (depends on gamma parquet)
python "scripts/overnight range strat/scripts/range_break_entry_signal_study.py"

# Add the confirmation-absorption columns (needed by locked filter)
python "scripts/overnight range strat/scripts/augment_with_confirmation_absorption.py" is

# Re-run the locked B2 backtest
python "scripts/overnight range strat/scripts/lock_v2_k08_lock045_mart_fc_filtered.py"

# Validate live engine still matches new backtest
python live/combined/replay_b2.py
```

If you change ANY of the inputs (DTE filter, snapshot time, methodology
flag, NDX/QQQ DTE assumptions), you MUST do steps 2-5 in order, or live
and backtest will diverge.

---

## 12. DO NOT change without re-validating end-to-end

| Setting | Locked value | File |
|---|---|---|
| NQ snapshot time | 16:05 ET | `build_menthorq_levels_bulk.py:46` |
| QQQ DTE for CR/PS/IV/EM | (0, 45) | `build_menthorq_levels_bulk.py:133` |
| QQQ DTE for GEX 1-10 | dynamic (next-trading-day shift) | computed from `next_td` lookup |
| NDX DTE for everything | (0, 45) | `build_menthorq_levels_bulk.py:156-160` |
| `legacy_mode` | `True` | `build_menthorq_levels_bulk.py:148, 150, 160` |
| CR/PS definition | max/min `net_gex` per side | `menthorq_style_levels.py` legacy branch |
| GEX 1-10 ranking | `|net_gex|` desc, exclude CR/PS | `menthorq_style_levels.py` legacy branch |

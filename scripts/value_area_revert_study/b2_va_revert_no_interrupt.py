"""B2 + VA-revert combined — NO-INTERRUPT variant.

Difference from b2_va_revert_realtime.py:
  - When bias-set occurs DURING a live VA-revert trade, DO NOT force-close.
  - Let VA-revert play out to its natural TP/SL/16:00 close.
  - State still transitions to B2_PHASE on bias-set, so subsequent entries
    are B2-only (no new VA-revert signals after bias).
  - B2 entries that occur WHILE VA-revert is still live are suppressed
    (one position at a time).

Rules summary:
  Pre-bias: VA-revert eligible (chained, stop-after-TP)
  On bias-set: state := B2_PHASE  (VA-revert keeps running if live)
  Post-bias + no current trade: take next B2 entry by time
  VA-revert TP -> DAY_DONE
  Force-close at 16:00 ET applies as normal exit for whichever side is live.
"""
from __future__ import annotations

import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
PROJECT_SCRIPTS = Path(__file__).parent.parent / "overnight range strat" / "scripts"
sys.path.insert(0, str(PROJECT_SCRIPTS))
from sweep_ratchet_sl_fixed_tp import filter_pre_dedupe
from test_pure_ratchet_exits import build_20min_bars
from range_break_entry_signal_study import load_range_per_day, load_5min_features
from b2_va_revert_combine import (
    VA_N, VA_SIG_D, VA_CONF_D, VA_SL_MULT, VA_HOURS, OVERSHOOT_LIMIT_PCT,
    FORCE_CLOSE, simulate_b2_exit, attach_gamma_b2, filter_b2, apply_mart_fc,
    qualifying_days_va,
)
from b2_va_revert_realtime import compute_first_bias_set_idx

ET = "America/New_York"
PARQUET_DIR_B2 = PROJECT_SCRIPTS / "parquets"
SIGS_VA   = Path(__file__).parent / "va_revert_signals.parquet"
OUT_TXT   = Path(__file__).parent.parent / "overnight range strat" / "tradelogs" / "robust_configs" / "b2_va_revert_no_interrupt.txt"


def simulate_day_no_interrupt(day_bars, ohi, olo, va_signals_today, b2_trades_today,
                                qualifies_va):
    """State machine WITHOUT force-close on bias-set."""
    closes = day_bars["close"].values
    bias_set_idx = compute_first_bias_set_idx(closes, ohi, olo)

    va_by_idx = {}
    if qualifies_va:
        for _, sig in va_signals_today.iterrows():
            va_by_idx.setdefault(int(sig["entry_idx"]), sig)

    b2_by_idx = {}
    for _, b2t in b2_trades_today.iterrows():
        ent_t = pd.Timestamp(b2t["entry_ts"])
        matches = np.where(day_bars["bar_open_time"] == ent_t)[0]
        if len(matches):
            b2_by_idx.setdefault(int(matches[0]), b2t)

    state = "VA_PHASE"
    current = None
    day_done = False
    trades = []

    for idx in range(len(day_bars)):
        bar = day_bars.iloc[idx]
        bar_time = pd.Timestamp(bar["bar_open_time"])

        # 1. Phase transition on bias-set -- NO force-close in this variant
        if state == "VA_PHASE" and bias_set_idx is not None and idx == bias_set_idx:
            state = "B2_PHASE"
            # do NOT touch current trade; let it run to natural exit

        # 2. Exit checks for active trade (VA-revert or B2)
        if current is not None:
            if current["type"] == "VA":
                sign = 1 if current["direction"] == "LONG" else -1
                sl_price = current["sl_price"]; tp_price = current["tp_price"]
                hi, lo, cl, op = float(bar["high"]), float(bar["low"]), float(bar["close"]), float(bar["open"])
                if bar_time.time() >= FORCE_CLOSE:
                    close_px = float(day_bars.iloc[idx - 1]["close"]) if idx > 0 else op
                    pnl = sign * (close_px - current["entry_price"])
                    trades.append({**current, "exit_idx": idx, "exit_time": bar_time,
                                   "outcome": "held->close", "pnl": pnl, "exit_price": close_px})
                    current = None
                    continue
                if sign > 0:
                    tp_hit = hi >= tp_price; sl_hit = lo <= sl_price
                else:
                    tp_hit = lo <= tp_price; sl_hit = hi >= sl_price
                if tp_hit and sl_hit:
                    pnl = sign * (sl_price - current["entry_price"])
                    trades.append({**current, "exit_idx": idx, "exit_time": bar_time,
                                   "outcome": "SL", "pnl": pnl, "exit_price": sl_price})
                    current = None
                elif tp_hit:
                    pnl = sign * (tp_price - current["entry_price"])
                    trades.append({**current, "exit_idx": idx, "exit_time": bar_time,
                                   "outcome": "TP", "pnl": pnl, "exit_price": tp_price})
                    current = None
                    day_done = True
                elif sl_hit:
                    pnl = sign * (sl_price - current["entry_price"])
                    trades.append({**current, "exit_idx": idx, "exit_time": bar_time,
                                   "outcome": "SL", "pnl": pnl, "exit_price": sl_price})
                    current = None
            elif current["type"] == "B2":
                if bar_time >= current["exit_time"]:
                    trades.append({**current, "exit_idx": idx,
                                   "outcome": current["b2_outcome"],
                                   "pnl": current["b2_pnl"],
                                   "exit_price": np.nan})
                    current = None

        # 3. New entry?
        if current is None and not day_done:
            if state == "VA_PHASE":
                if idx in va_by_idx:
                    sig = va_by_idx[idx]
                    entry_price = float(sig["entry_price"]); atr = float(sig["atr_at_entry"])
                    sl_pts = VA_SL_MULT * atr; tp_price = float(sig["target"])
                    sign = 1 if sig["direction"] == "LONG" else -1
                    sl_price = entry_price - sign * sl_pts
                    current = {"type": "VA", "direction": sig["direction"],
                               "entry_idx": idx, "entry_time": bar_time,
                               "entry_price": entry_price,
                               "tp_price": tp_price, "sl_price": sl_price,
                               "atr": atr, "sl_pts": sl_pts}
            elif state == "B2_PHASE":
                if idx in b2_by_idx:
                    b2t = b2_by_idx[idx]
                    current = {"type": "B2", "direction": b2t["direction"],
                               "entry_idx": idx, "entry_time": bar_time,
                               "entry_price": np.nan,
                               "exit_time": pd.Timestamp(b2t["exit_ts"]),
                               "b2_outcome": b2t["reason"],
                               "b2_pnl": float(b2t["pnl"]),
                               "b2_pnl_scaled": float(b2t["pnl_scaled"])}

    # End-of-day cleanup
    if current is not None:
        last = day_bars.iloc[-1]
        if current["type"] == "VA":
            sign = 1 if current["direction"] == "LONG" else -1
            pnl = sign * (float(last["close"]) - current["entry_price"])
            trades.append({**current, "exit_idx": len(day_bars)-1,
                           "exit_time": pd.Timestamp(last["bar_open_time"]),
                           "outcome": "held->close", "pnl": pnl,
                           "exit_price": float(last["close"])})
        else:
            trades.append({**current, "exit_idx": len(day_bars)-1,
                           "outcome": current["b2_outcome"],
                           "pnl": current["b2_pnl"],
                           "exit_price": np.nan})
    return trades


def main():
    print("loading data ...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR_B2 / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR_B2 / "entry_signal_trades_oos.parquet")
    is_cands  = filter_b2(attach_gamma_b2(filter_pre_dedupe(is_df)))
    oos_cands = filter_b2(attach_gamma_b2(filter_pre_dedupe(oos_df)))

    def run_b2(cands):
        rows = []
        last_exit = pd.Timestamp(0, tz=ET)
        for _, t in cands.sort_values("entry_time_et").iterrows():
            ent_t = pd.Timestamp(t["entry_time_et"])
            if ent_t <= last_exit: continue
            ex = simulate_b2_exit(t["direction"], ent_t, float(t["entry_price"]), bars20)
            if ex is None: continue
            pnl, reason, exit_ts = ex
            rows.append({"entry_ts": ent_t, "exit_ts": exit_ts,
                         "direction": t["direction"], "reason": reason, "pnl": pnl})
            last_exit = pd.Timestamp(exit_ts)
        return pd.DataFrame(rows)
    is_t  = run_b2(is_cands); oos_t = run_b2(oos_cands)
    b2_all = pd.concat([is_t, oos_t], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)
    sizes = apply_mart_fc(b2_all); b2_all["size"] = sizes
    b2_all["pnl_scaled"] = b2_all["pnl"] * sizes
    b2_all["date"] = b2_all["entry_ts"].dt.date

    va_sigs = pd.read_parquet(SIGS_VA)
    va_sigs["date"] = pd.to_datetime(va_sigs["date"]).dt.date
    va_sigs["entry_time"] = pd.to_datetime(va_sigs["entry_time"])
    va_sigs["entry_hour"] = va_sigs["entry_time"].dt.hour
    sig_col = f"sig_abs_w{VA_N}"; conf_col = f"conf_abs_w{VA_N}"
    va_sub = va_sigs[(va_sigs["distance_pct"] < OVERSHOOT_LIMIT_PCT) &
                      (va_sigs[sig_col].fillna(0)  >= VA_SIG_D) &
                      (va_sigs[conf_col].fillna(0) >= VA_CONF_D) &
                      (va_sigs["entry_hour"].isin(VA_HOURS))].copy()

    bars_all, _ = load_5min_features((dt.date(2020,12,1), dt.date(2026,5,7)))
    bars_by_day = {d: g.sort_values("bar_open_time").reset_index(drop=True)
                   for d, g in bars_all.groupby("session_date")}
    rng = load_range_per_day()
    qual = qualifying_days_va()
    qual_dates = set(qual.index)

    print("running NO-INTERRUPT state machine per day ...")
    all_trades = []
    for d, day_bars in bars_by_day.items():
        if d not in rng.index: continue
        ohi = float(rng.loc[d, "overnight_high"]); olo = float(rng.loc[d, "overnight_low"])
        va_signals_today = va_sub[va_sub["date"] == d]
        b2_trades_today  = b2_all[b2_all["date"] == d]
        qualifies = d in qual_dates
        trades = simulate_day_no_interrupt(day_bars, ohi, olo, va_signals_today,
                                              b2_trades_today, qualifies)
        for t in trades:
            t["date"] = d
            all_trades.append(t)
    rt_df = pd.DataFrame(all_trades)
    print(f"Total trades (no-interrupt): {len(rt_df)}")

    # Apply mart to B2 trades
    b2_mask = rt_df["type"] == "B2"
    rt_b2 = rt_df[b2_mask].copy().reset_index(drop=True)
    sizes_b2 = apply_mart_fc(rt_b2.assign(reason=rt_b2["outcome"]))
    rt_b2["size"] = sizes_b2; rt_b2["pnl_scaled"] = rt_b2["pnl"] * sizes_b2
    rt_df["pnl_final"] = rt_df["pnl"].copy()
    rt_df.loc[b2_mask, "pnl_final"] = rt_b2["pnl_scaled"].values
    rt_df["period"] = np.where(pd.to_datetime(rt_df["date"]).dt.date < dt.date(2025,1,1),
                                 "IS", "OOS")
    rt_df["year"] = pd.to_datetime(rt_df["date"]).dt.year

    # Compare with prior force-interrupt variant by reloading those trades from txt
    L = []
    L.append("=" * 200)
    L.append("REAL-TIME B2 + VA-REVERT  -- NO INTERRUPT VARIANT")
    L.append("=" * 200)
    L.append("Rule change: bias-set does NOT force-close an open VA-revert. VA runs to TP/SL/16:00.")
    L.append("")

    by_type = rt_df.groupby("type").agg(
        n=("pnl_final","count"),
        pnl=("pnl_final","sum"),
        wr=("pnl_final", lambda s: (s>0).mean()*100),
    ).round(2)
    by_type["dollars_nq"] = (by_type["pnl"]*20).round(0)
    L.append("Trade type counts:")
    L.append(by_type.to_string())
    L.append("")

    L.append("Trade outcomes by type:")
    by_outcome = rt_df.groupby(["type","outcome"]).agg(
        n=("pnl_final","count"),
        pnl=("pnl_final","sum"),
    ).round(2)
    L.append(by_outcome.to_string())

    L.append("")
    L.append("Totals:")
    for p in ("IS","OOS","ALL"):
        sub = rt_df if p == "ALL" else rt_df[rt_df["period"]==p]
        n = len(sub); tot = sub["pnl_final"].sum()
        b2_part = sub[sub["type"]=="B2"]["pnl_final"].sum()
        va_part = sub[sub["type"]=="VA"]["pnl_final"].sum()
        L.append(f"  {p:<4} n={n:>4}  total={tot:>+8.1f} pts  (${tot*20:>+10,.0f} NQ)   "
                 f"B2:{b2_part:>+8.1f}  VA:{va_part:>+8.1f}")

    L.append("")
    L.append("Per-year:")
    yr = rt_df.groupby("year").agg(
        n_total=("pnl_final","count"),
        n_va=("type", lambda s: (s=="VA").sum()),
        n_b2=("type", lambda s: (s=="B2").sum()),
        pnl_total=("pnl_final","sum"),
    ).round(1)
    yr["pnl_total_nq"] = (yr["pnl_total"]*20).round(0)
    L.append(yr.to_string())

    L.append("")
    L.append("=" * 200)
    L.append("VS PRIOR VARIANTS")
    L.append("=" * 200)
    b2_std_total = b2_all["pnl_scaled"].sum()
    rt_total = rt_df["pnl_final"].sum()
    L.append(f"  Standalone B2:               +{b2_std_total:8.1f} pts  (${b2_std_total*20:>+10,.0f} NQ)")
    L.append(f"  Force-interrupt combined:     (see b2_va_revert_realtime.txt for prior run)")
    L.append(f"  NO-INTERRUPT combined:       +{rt_total:8.1f} pts  (${rt_total*20:>+10,.0f} NQ)")
    L.append(f"  DELTA vs B2-only:            {rt_total-b2_std_total:>+8.1f} pts  "
             f"(${(rt_total-b2_std_total)*20:>+10,.0f} NQ)")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

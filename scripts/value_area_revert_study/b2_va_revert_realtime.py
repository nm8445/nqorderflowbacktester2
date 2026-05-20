"""Combined B2 + VA-revert with REAL-TIME state machine (no look-ahead).

Rules per session day:
  state := VA_PHASE  (initial — bias not yet set on the day)
  if VA-revert signal fires and no current trade -> enter VA-revert
  if VA-revert TP hits -> DAY_DONE (no more entries)
  if VA-revert SL hits -> back to VA_PHASE, can take next VA-revert signal
  if bias-set event (3 consec 5-min closes outside OHI/OLO):
     - if VA-revert is live -> FORCE-CLOSE at this bar's close
     - state := B2_PHASE
  in B2_PHASE: B2 trades chain normally (one at a time, all locked filtered rules)
  No overlapping trades.

B2 = locked filtered V2 K=0.8 lock=0.45 + mart-fc-only + hours 9-14 + drop POS+SHORT
VA-revert = N=20, sig_D=0, conf_D=150, sl=1.0, hours={9,11,12}, cap=75%
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
from test_pure_ratchet_exits import build_20min_bars, FORCE_CLOSE_TIME
from range_break_entry_signal_study import load_range_per_day, load_5min_features
from b2_va_revert_combine import (
    YMULT, TPMULT, MFE_K, MFE_LOCK, ALLOWED_HOURS_B2, DROP_POS_SHORT,
    VA_N, VA_SIG_D, VA_CONF_D, VA_SL_MULT, VA_HOURS, OVERSHOOT_LIMIT_PCT,
    MIN_VA_WIDTH, FORCE_CLOSE,
    simulate_b2_exit, attach_gamma_b2, filter_b2, apply_mart_fc,
    load_premarket_open, qualifying_days_va,
)

ET = "America/New_York"
ENTRY_N = 3      # 3 consec 5-min closes outside range = bias set
PARQUET_DIR_B2 = PROJECT_SCRIPTS / "parquets"
SIGS_VA   = Path(__file__).parent / "va_revert_signals.parquet"
OUT_TXT   = Path(__file__).parent.parent / "overnight range strat" / "tradelogs" / "robust_configs" / "b2_va_revert_realtime.txt"


def compute_first_bias_set_idx(closes: np.ndarray, ohi: float, olo: float, n: int = ENTRY_N):
    """Return first bar index where N consecutive closes outside OHI/OLO range
    fire the bias-set event. Returns None if never set."""
    above = 0; below = 0
    for i, c in enumerate(closes):
        if c > ohi:
            above += 1; below = 0
            if above >= n: return i
        elif c < olo:
            below += 1; above = 0
            if below >= n: return i
        else:
            above = 0; below = 0
    return None


def simulate_day(day_bars, ohi, olo, va_signals_today, b2_trades_today,
                  qualifies_va):
    """Run the real-time state machine for one session day.
    Returns list of trade dicts taken on this day.
    """
    closes = day_bars["close"].values
    bias_set_idx = compute_first_bias_set_idx(closes, ohi, olo)
    bias_set_time = day_bars.iloc[bias_set_idx]["bar_open_time"] if bias_set_idx is not None else None

    # Build entry-idx -> trade lookups
    va_by_idx = {}
    if qualifies_va:
        for _, sig in va_signals_today.iterrows():
            va_by_idx.setdefault(int(sig["entry_idx"]), sig)

    # B2 trades have entry_ts; map to bar index using tz-aware comparison
    b2_by_idx = {}
    for _, b2t in b2_trades_today.iterrows():
        ent_t = pd.Timestamp(b2t["entry_ts"])
        matches = np.where(day_bars["bar_open_time"] == ent_t)[0]
        if len(matches):
            b2_by_idx.setdefault(int(matches[0]), b2t)

    state = "VA_PHASE"  # initial
    current = None       # active trade dict
    day_done = False
    trades = []

    for idx in range(len(day_bars)):
        bar = day_bars.iloc[idx]
        bar_time = pd.Timestamp(bar["bar_open_time"])

        # 1. Phase transition: bias just set on THIS bar?
        if state == "VA_PHASE" and bias_set_idx is not None and idx == bias_set_idx:
            state = "B2_PHASE"
            if current is not None and current["type"] == "VA":
                # Force-close at this bar's close
                close_px = float(bar["close"])
                sign = 1 if current["direction"] == "LONG" else -1
                pnl = sign * (close_px - current["entry_price"])
                trades.append({**current, "exit_idx": idx,
                               "exit_time": bar_time, "outcome": "FORCE_BIAS_SET",
                               "pnl": pnl, "exit_price": close_px})
                current = None
            continue   # don't try to enter on this bar

        # 2. Exit checks for active trade
        if current is not None:
            if current["type"] == "VA":
                # VA-revert: simple TP/SL/force-close at 16:00
                sign = 1 if current["direction"] == "LONG" else -1
                sl_price = current["sl_price"]
                tp_price = current["tp_price"]
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
                    day_done = True  # VA TP -> day done
                elif sl_hit:
                    pnl = sign * (sl_price - current["entry_price"])
                    trades.append({**current, "exit_idx": idx, "exit_time": bar_time,
                                   "outcome": "SL", "pnl": pnl, "exit_price": sl_price})
                    current = None

            elif current["type"] == "B2":
                # B2 already has pre-computed exit time + outcome -- check if exited this bar or earlier
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
                    entry_price = float(sig["entry_price"])
                    atr = float(sig["atr_at_entry"])
                    sl_pts = VA_SL_MULT * atr
                    tp_price = float(sig["target"])
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
                               "b2_pnl": float(b2t["pnl"]),  # 1-contract; we'll apply mart sizing later
                               "b2_pnl_scaled": float(b2t["pnl_scaled"])}

    # End of day: if still in a trade, close at last bar
    if current is not None:
        last = day_bars.iloc[-1]
        if current["type"] == "VA":
            sign = 1 if current["direction"] == "LONG" else -1
            pnl = sign * (float(last["close"]) - current["entry_price"])
            trades.append({**current, "exit_idx": len(day_bars) - 1,
                           "exit_time": pd.Timestamp(last["bar_open_time"]),
                           "outcome": "held->close", "pnl": pnl,
                           "exit_price": float(last["close"])})
        elif current["type"] == "B2":
            trades.append({**current, "exit_idx": len(day_bars) - 1,
                           "outcome": current["b2_outcome"],
                           "pnl": current["b2_pnl"],
                           "exit_price": np.nan})
    return trades, bias_set_time


def main():
    # ------ Load all data ------
    print("loading 20-min bars + B2 candidates (locked filtered) ...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR_B2 / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR_B2 / "entry_signal_trades_oos.parquet")
    is_cands  = filter_b2(attach_gamma_b2(filter_pre_dedupe(is_df)))
    oos_cands = filter_b2(attach_gamma_b2(filter_pre_dedupe(oos_df)))

    print("simulating B2 chained (standalone outcomes for reference) ...")
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

    print(f"  Standalone B2 (mart-scaled): {len(b2_all)} trades, "
          f"total ${b2_all['pnl_scaled'].sum()*20:+,.0f} NQ")

    print("\nloading VA-revert signals + filtering to chosen config ...")
    va_sigs = pd.read_parquet(SIGS_VA)
    va_sigs["date"] = pd.to_datetime(va_sigs["date"]).dt.date
    va_sigs["entry_time"] = pd.to_datetime(va_sigs["entry_time"])
    va_sigs["entry_hour"] = va_sigs["entry_time"].dt.hour
    sig_col = f"sig_abs_w{VA_N}"; conf_col = f"conf_abs_w{VA_N}"
    va_sub = va_sigs[(va_sigs["distance_pct"] < OVERSHOOT_LIMIT_PCT) &
                      (va_sigs[sig_col].fillna(0)  >= VA_SIG_D) &
                      (va_sigs[conf_col].fillna(0) >= VA_CONF_D) &
                      (va_sigs["entry_hour"].isin(VA_HOURS))].copy()
    print(f"  VA-revert signals (filtered): {len(va_sub)}")

    print("loading 5-min bars + overnight ranges ...")
    bars_all, _ = load_5min_features((dt.date(2020,12,1), dt.date(2026,5,7)))
    bars_by_day = {d: g.sort_values("bar_open_time").reset_index(drop=True)
                   for d, g in bars_all.groupby("session_date")}
    rng = load_range_per_day()

    print("computing VA-revert qualifying days ...")
    qual = qualifying_days_va()
    qual_dates = set(qual.index)

    # ------ Run real-time simulator day-by-day ------
    print("\nrunning real-time state machine per day ...")
    all_trades = []
    bias_set_summary = []
    for d, day_bars in bars_by_day.items():
        if d not in rng.index: continue
        ohi = float(rng.loc[d, "overnight_high"])
        olo = float(rng.loc[d, "overnight_low"])
        va_signals_today = va_sub[va_sub["date"] == d]
        b2_trades_today  = b2_all[b2_all["date"] == d]
        qualifies = d in qual_dates
        trades, bias_set_time = simulate_day(day_bars, ohi, olo, va_signals_today,
                                              b2_trades_today, qualifies)
        for t in trades:
            t["date"] = d
            all_trades.append(t)
        bias_set_summary.append({"date": d, "bias_set_time": bias_set_time,
                                  "qualifies_va": qualifies,
                                  "n_va_signals": len(va_signals_today),
                                  "n_b2_entries": len(b2_trades_today)})
    rt_df = pd.DataFrame(all_trades)
    bs_df = pd.DataFrame(bias_set_summary)

    print(f"\nTotal trades taken under real-time rule: {len(rt_df)}")

    # ------ Apply mart-fc-only to B2-type trades in the real-time set ------
    rt_df = rt_df.sort_values(["date", "entry_time"]).reset_index(drop=True)
    # Mart only applies to B2 trades (separately from VA-revert which is 1-contract)
    b2_mask = rt_df["type"] == "B2"
    rt_b2 = rt_df[b2_mask].copy().reset_index(drop=True)
    sizes_b2 = apply_mart_fc(rt_b2.assign(reason=rt_b2["outcome"]))
    rt_b2["size"] = sizes_b2
    rt_b2["pnl_scaled"] = rt_b2["pnl"] * sizes_b2
    # Map back into rt_df
    rt_df["pnl_final"] = rt_df["pnl"].copy()
    rt_df.loc[b2_mask, "pnl_final"] = rt_b2["pnl_scaled"].values

    rt_df["period"] = np.where(pd.to_datetime(rt_df["date"]).dt.date < dt.date(2025,1,1),
                                 "IS", "OOS")
    rt_df["year"] = pd.to_datetime(rt_df["date"]).dt.year

    # ------ Stats ------
    L = []
    L.append("=" * 200)
    L.append("REAL-TIME COMBINED B2 + VA-REVERT (no lookahead)")
    L.append("=" * 200)
    L.append("Rules:")
    L.append("  - Pre-bias-set: VA-revert eligible (chained Mode 1, stop-after-TP)")
    L.append("  - On bias-set (3 consec 5-min closes outside range):")
    L.append("      * Force-close any open VA-revert at that bar's close")
    L.append("      * Switch to B2 phase")
    L.append("  - Post-bias-set: B2 eligible (locked filtered V2 + mart-fc-only)")
    L.append("  - One position at a time; no overlap")
    L.append("")
    L.append(f"B2 config: locked filtered V2 K=0.8 lock=0.45 + mart-fc-only + hours 9-14 + drop POS+SHORT")
    L.append(f"VA-revert config: N={VA_N}, sig_D={VA_SIG_D}, conf_D={VA_CONF_D}, "
             f"sl={VA_SL_MULT}, hours={sorted(VA_HOURS)}, cap={OVERSHOOT_LIMIT_PCT}%")
    L.append("")

    by_type = rt_df.groupby("type").agg(
        n=("pnl_final","count"),
        pnl=("pnl_final","sum"),
        wr=("pnl_final", lambda s: (s>0).mean()*100),
    ).round(2)
    by_type["dollars_nq"] = (by_type["pnl"]*20).round(0)
    L.append("Trade type counts under real-time rule:")
    L.append(by_type.to_string())

    L.append("")
    L.append("Trade outcomes under real-time rule (by type):")
    by_outcome = rt_df.groupby(["type","outcome"]).agg(
        n=("pnl_final","count"),
        pnl=("pnl_final","sum"),
    ).round(2)
    L.append(by_outcome.to_string())

    L.append("")
    L.append("Totals (period-split):")
    for p in ("IS","OOS","ALL"):
        sub = rt_df if p == "ALL" else rt_df[rt_df["period"]==p]
        n = len(sub); tot = sub["pnl_final"].sum()
        b2_part = sub[sub["type"]=="B2"]["pnl_final"].sum()
        va_part = sub[sub["type"]=="VA"]["pnl_final"].sum()
        L.append(f"  {p:<4} n={n:>4}  total={tot:>+8.1f} pts  (${tot*20:>+10,.0f} NQ)   "
                 f"B2-part: {b2_part:>+8.1f}   VA-part: {va_part:>+8.1f}")

    L.append("")
    L.append("Per-year breakdown ($ NQ):")
    yr = rt_df.groupby("year").agg(
        n_total=("pnl_final","count"),
        n_va=("type", lambda s: (s=="VA").sum()),
        n_b2=("type", lambda s: (s=="B2").sum()),
        pnl_total=("pnl_final","sum"),
        pnl_b2=("pnl_final", lambda s: s[rt_df.loc[s.index, "type"]=="B2"].sum()),
    ).round(1)
    yr["pnl_total_nq"] = (yr["pnl_total"]*20).round(0)
    yr["pnl_b2_nq"] = (yr["pnl_b2"]*20).round(0)
    L.append(yr.to_string())

    # ------ Compare vs B2-standalone and VA-standalone ------
    L.append("")
    L.append("=" * 200)
    L.append("COMPARISON TO STANDALONE STRATEGIES")
    L.append("=" * 200)
    b2_std_total = b2_all["pnl_scaled"].sum()
    L.append(f"  Standalone B2 (mart-scaled, no VA): {b2_std_total:>+8.1f} pts  (${b2_std_total*20:>+10,.0f} NQ)  "
             f"n={len(b2_all)}")
    rt_total = rt_df["pnl_final"].sum()
    L.append(f"  Combined real-time (B2 + VA):       {rt_total:>+8.1f} pts  (${rt_total*20:>+10,.0f} NQ)  "
             f"n={len(rt_df)}")
    delta = rt_total - b2_std_total
    L.append(f"  DELTA (combined - B2-only):         {delta:>+8.1f} pts  (${delta*20:>+10,.0f} NQ)")
    L.append("")
    L.append("Notes:")
    L.append("  - Combined will USE FEWER B2 trades than standalone (some get suppressed because")
    L.append("    a VA-revert was live when B2 bias set, OR because day_done after VA TP).")
    L.append("  - Combined ADDS VA-revert trades on days where VA fired before bias was set.")
    L.append("")
    n_b2_taken = (rt_df["type"]=="B2").sum()
    n_b2_standalone = len(b2_all)
    L.append(f"  B2 trades taken in combined: {n_b2_taken}  vs standalone: {n_b2_standalone}  "
             f"(diff: {n_b2_taken - n_b2_standalone})")

    # Force-bias-set closes for VA-revert
    n_force = ((rt_df["type"]=="VA") & (rt_df["outcome"]=="FORCE_BIAS_SET")).sum()
    n_va_tp = ((rt_df["type"]=="VA") & (rt_df["outcome"]=="TP")).sum()
    n_va_sl = ((rt_df["type"]=="VA") & (rt_df["outcome"]=="SL")).sum()
    n_va_hc = ((rt_df["type"]=="VA") & (rt_df["outcome"]=="held->close")).sum()
    L.append(f"")
    L.append(f"  VA-revert outcomes:  TP={n_va_tp}   SL={n_va_sl}   held->close={n_va_hc}   "
             f"FORCE_BIAS_SET={n_force}")
    if n_force > 0:
        force_pnl = rt_df[(rt_df["type"]=="VA") & (rt_df["outcome"]=="FORCE_BIAS_SET")]["pnl_final"].sum()
        L.append(f"  Force-bias-set VA closes pnl: {force_pnl:>+.1f} pts (${force_pnl*20:>+,.0f} NQ)")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

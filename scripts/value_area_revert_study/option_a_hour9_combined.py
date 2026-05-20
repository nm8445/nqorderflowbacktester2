"""Option A: find the best VA-revert config restricted to hour-9 entries,
then plug it into the combined real-time (no-interrupt) state machine.

Two-stage:
  1) Sweep (N, sig_D, conf_D, sl) on hour-9-only VA-revert signals.
     Pick top config by min(IS_PF, OOS_PF), require n>=50 trades.
  2) Run combined real-time backtest (no-interrupt variant) using that config.
"""
from __future__ import annotations

import datetime as dt
import sys
import warnings
from itertools import product
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
    OVERSHOOT_LIMIT_PCT, FORCE_CLOSE,
    simulate_b2_exit, attach_gamma_b2, filter_b2, apply_mart_fc,
    qualifying_days_va,
)
from b2_va_revert_realtime import compute_first_bias_set_idx
from b2_va_revert_no_interrupt import simulate_day_no_interrupt

ET = "America/New_York"
PARQUET_DIR_B2 = PROJECT_SCRIPTS / "parquets"
SIGS_VA   = Path(__file__).parent / "va_revert_signals.parquet"
OUT_TXT   = Path(__file__).parent.parent / "overnight range strat" / "tradelogs" / "robust_configs" / "option_a_hour9_combined.txt"

N_GRID       = [5, 10, 15, 20]
SIG_D_GRID   = [0, 50, 100, 150]
CONF_D_GRID  = [0, 50, 100, 150]
SL_MULT_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
MIN_TRADES   = 50
HOUR9 = {9}


# ============================== VA-revert sweep helpers ==============================

def simulate_va_exit(direction, entry_price, entry_idx, bars_day, sl_pts, tp_price):
    sign = 1 if direction == "LONG" else -1
    sl_price = entry_price - sign * sl_pts
    n = len(bars_day)
    for k in range(entry_idx + 1, n):
        bar = bars_day.iloc[k]
        bt = pd.Timestamp(bar["bar_open_time"])
        if bt.time() >= FORCE_CLOSE:
            close_px = float(bars_day.iloc[k - 1]["close"]) if k > 0 else float(bar["open"])
            return ("held->close", sign * (close_px - entry_price), bt)
        hi, lo = float(bar["high"]), float(bar["low"])
        if sign > 0:
            tp_hit = hi >= tp_price; sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price; sl_hit = hi >= sl_price
        if tp_hit and sl_hit: return ("SL", sign*(sl_price-entry_price), bt)
        if tp_hit: return ("TP", sign*(tp_price-entry_price), bt)
        if sl_hit: return ("SL", sign*(sl_price-entry_price), bt)
    last = bars_day.iloc[-1]
    return ("held->close", sign*(float(last["close"])-entry_price),
            pd.Timestamp(last["bar_open_time"]))


def run_va_chained(sigs, bars_by_day, sl_mult):
    rows = []
    last_exit = pd.Timestamp(0, tz=ET)
    for _, t in sigs.sort_values("entry_time").iterrows():
        ent_t = pd.Timestamp(t["entry_time"])
        if ent_t <= last_exit: continue
        d = pd.to_datetime(t["date"]).date()
        bars_day = bars_by_day.get(d)
        if bars_day is None: continue
        entry_idx = int(t["entry_idx"])
        if entry_idx >= len(bars_day): continue
        entry_price = float(t["entry_price"]); atr = float(t["atr_at_entry"])
        sl_pts = sl_mult * atr; tp_price = float(t["target"])
        outcome, pnl, exit_t = simulate_va_exit(
            t["direction"], entry_price, entry_idx, bars_day, sl_pts, tp_price)
        rows.append({"date": d, "pnl": pnl, "outcome": outcome})
        last_exit = pd.Timestamp(exit_t)
    return pd.DataFrame(rows)


def stats(df):
    n = len(df)
    if n == 0: return {"n":0,"wr":0,"pf":0,"total":0,"mdd":0}
    pnl = df["pnl"].values
    wins = pnl[pnl>0]; losses = pnl[pnl<0]
    wr = (pnl>0).mean()*100
    pf = wins.sum()/abs(losses.sum()) if losses.sum() != 0 else float("inf")
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); mdd = (eq-peak).min()
    return {"n":n,"wr":wr,"pf":pf,"total":pnl.sum(),"mdd":mdd}


def main():
    # ----------- STEP 1: hour-9-only sweep -----------
    print("loading VA-revert signals + bars ...")
    va = pd.read_parquet(SIGS_VA)
    va["date"] = pd.to_datetime(va["date"]).dt.date
    va["entry_time"] = pd.to_datetime(va["entry_time"])
    va["entry_hour"] = va["entry_time"].dt.hour
    va_hour9 = va[(va["distance_pct"] < OVERSHOOT_LIMIT_PCT) &
                  (va["entry_hour"].isin(HOUR9))].copy()
    print(f"  total hour-9 candidates (before sig/conf filter): {len(va_hour9)}")

    bars_all, _ = load_5min_features((dt.date(2020,12,1), dt.date(2026,5,7)))
    bars_by_day = {d: g.sort_values("bar_open_time").reset_index(drop=True)
                   for d, g in bars_all.groupby("session_date")}

    is_cut = dt.date(2025, 1, 1)
    print(f"\nsweeping {len(N_GRID)*len(SIG_D_GRID)*len(CONF_D_GRID)*len(SL_MULT_GRID)} combos on hour-9 ...")
    rows = []
    for N, sig_D, conf_D in product(N_GRID, SIG_D_GRID, CONF_D_GRID):
        sig_c = f"sig_abs_w{N}"; conf_c = f"conf_abs_w{N}"
        sub = va_hour9[(va_hour9[sig_c].fillna(0) >= sig_D) &
                       (va_hour9[conf_c].fillna(0) >= conf_D)]
        if len(sub) < MIN_TRADES: continue
        for sl in SL_MULT_GRID:
            is_sigs  = sub[sub["date"] < is_cut]
            oos_sigs = sub[sub["date"] >= is_cut]
            it = run_va_chained(is_sigs,  bars_by_day, sl)
            ot = run_va_chained(oos_sigs, bars_by_day, sl)
            at = pd.concat([it, ot], ignore_index=True)
            s_is, s_oos, s_all = stats(it), stats(ot), stats(at)
            rows.append({"N":N,"sig_D":sig_D,"conf_D":conf_D,"sl":sl,
                         "is_n":s_is["n"],"is_wr":s_is["wr"],"is_pf":s_is["pf"],"is_total":s_is["total"],
                         "oos_n":s_oos["n"],"oos_wr":s_oos["wr"],"oos_pf":s_oos["pf"],"oos_total":s_oos["total"],
                         "all_n":s_all["n"],"all_pf":s_all["pf"],"all_total":s_all["total"]})
    df = pd.DataFrame(rows)
    df["min_pf"] = df[["is_pf","oos_pf"]].min(axis=1)
    print(f"  swept combos: {len(df)}")

    L = []
    L.append("=" * 200)
    L.append("OPTION A: HOUR-9-ONLY VA-REVERT SWEEP -> TOP CONFIG -> COMBINED REAL-TIME (NO-INTERRUPT)")
    L.append("=" * 200)
    qual = df[(df["is_n"] >= 30) & (df["is_total"] > 0) & (df["oos_total"] > 0)].sort_values("min_pf", ascending=False)
    L.append(f"\nHour-9 sweep: {len(qual)} configs with IS_n>=30 AND both periods profitable")
    if not qual.empty:
        L.append("\nTOP 10 BY min(IS_PF, OOS_PF):")
        L.append(qual.head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        L.append("\nTOP 5 BY combined total pts:")
        L.append(qual.sort_values("all_total", ascending=False).head(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if qual.empty:
        # fallback: relax requirement
        relax = df[(df["is_total"] > 0)].sort_values("min_pf", ascending=False)
        L.append(f"\nNo robust configs; relaxed (IS profitable only): {len(relax)} configs")
        if not relax.empty:
            L.append(relax.head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        top = relax.iloc[0] if not relax.empty else None
    else:
        top = qual.iloc[0]

    if top is None:
        print("No viable hour-9 config found.")
        OUT_TXT.write_text("\n".join(L), encoding="utf-8")
        return

    chosen = {"N": int(top["N"]), "sig_D": int(top["sig_D"]),
              "conf_D": int(top["conf_D"]), "sl": float(top["sl"])}
    L.append("\n" + "=" * 200)
    L.append(f"CHOSEN HOUR-9 CONFIG: N={chosen['N']}, sig_D={chosen['sig_D']}, "
             f"conf_D={chosen['conf_D']}, sl={chosen['sl']}")
    L.append("=" * 200)

    # ----------- STEP 2: combined real-time backtest with chosen config -----------
    print("\nloading B2 (locked filtered V2 + mart-fc-only) ...")
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
    sizes_b2 = apply_mart_fc(b2_all); b2_all["size"] = sizes_b2
    b2_all["pnl_scaled"] = b2_all["pnl"] * sizes_b2
    b2_all["date"] = b2_all["entry_ts"].dt.date

    # Filter VA signals to chosen hour-9 config
    sig_c = f"sig_abs_w{chosen['N']}"; conf_c = f"conf_abs_w{chosen['N']}"
    va_sub = va_hour9[(va_hour9[sig_c].fillna(0) >= chosen['sig_D']) &
                       (va_hour9[conf_c].fillna(0) >= chosen['conf_D'])].copy()
    print(f"  Hour-9 VA-revert signals with chosen filters: {len(va_sub)}")

    # ----- monkey-patch sl_mult into simulate_day_no_interrupt via env var? -----
    # Easier: copy the function inline with the chosen SL multiple.
    import b2_va_revert_combine as bvr
    bvr.VA_SL_MULT = chosen['sl']   # mutate the module-level constant referenced by simulate_day_no_interrupt

    rng = load_range_per_day()
    qual_va = qualifying_days_va()
    qual_dates = set(qual_va.index)

    print("running NO-INTERRUPT combined per day ...")
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
    print(f"  total trades: {len(rt_df)}")

    # Apply mart to B2 trades in combined
    b2_mask = rt_df["type"] == "B2"
    rt_b2 = rt_df[b2_mask].copy().reset_index(drop=True)
    sizes_b2c = apply_mart_fc(rt_b2.assign(reason=rt_b2["outcome"]))
    rt_b2["size"] = sizes_b2c
    rt_b2["pnl_scaled"] = rt_b2["pnl"] * sizes_b2c
    rt_df["pnl_final"] = rt_df["pnl"].copy()
    rt_df.loc[b2_mask, "pnl_final"] = rt_b2["pnl_scaled"].values
    rt_df["period"] = np.where(pd.to_datetime(rt_df["date"]).dt.date < is_cut, "IS", "OOS")
    rt_df["year"] = pd.to_datetime(rt_df["date"]).dt.year

    L.append("")
    L.append("Trade counts under combined (no-interrupt) with hour-9 VA-revert:")
    L.append(rt_df.groupby("type").agg(n=("pnl_final","count"), pnl=("pnl_final","sum"),
                                        wr=("pnl_final", lambda s: (s>0).mean()*100)).round(2).to_string())
    L.append("")
    L.append("Outcomes by type:")
    L.append(rt_df.groupby(["type","outcome"]).agg(n=("pnl_final","count"), pnl=("pnl_final","sum")).round(2).to_string())
    L.append("")
    L.append("Period totals:")
    for p in ("IS","OOS","ALL"):
        sub = rt_df if p == "ALL" else rt_df[rt_df["period"]==p]
        n = len(sub); tot = sub["pnl_final"].sum()
        b2_part = sub[sub["type"]=="B2"]["pnl_final"].sum()
        va_part = sub[sub["type"]=="VA"]["pnl_final"].sum()
        L.append(f"  {p:<4} n={n:>4}  total={tot:>+8.1f} pts  (${tot*20:>+10,.0f})   B2:{b2_part:>+8.1f}  VA:{va_part:>+8.1f}")

    L.append("")
    L.append("Per-year:")
    yr = rt_df.groupby("year").agg(
        n_total=("pnl_final","count"),
        n_va=("type", lambda s: (s=="VA").sum()),
        n_b2=("type", lambda s: (s=="B2").sum()),
        pnl_total=("pnl_final","sum"),
    ).round(1)
    yr["pnl_nq"] = (yr["pnl_total"]*20).round(0)
    L.append(yr.to_string())

    b2_std_total = b2_all["pnl_scaled"].sum()
    rt_total = rt_df["pnl_final"].sum()
    L.append("")
    L.append("=" * 200)
    L.append("HEADLINE")
    L.append("=" * 200)
    L.append(f"  Standalone B2:                  +{b2_std_total:8.1f} pts  (${b2_std_total*20:>+10,.0f} NQ)")
    L.append(f"  Combined no-interrupt (hour-9): +{rt_total:8.1f} pts  (${rt_total*20:>+10,.0f} NQ)")
    L.append(f"  DELTA vs B2-only:                {rt_total-b2_std_total:>+8.1f} pts  "
             f"(${(rt_total-b2_std_total)*20:>+10,.0f} NQ)")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""Find best VA-revert config with:
  - n_total >= 200 trades
  - Trades spread across many months (not clumped)
  - Hour filter: keep hours where IS-only PF >= HOUR_PF_MIN (slightly losing hours OK
    if user wants to keep trade count up)
  - Both IS and OOS profitable

Sweep over (N, sig_D, conf_D, sl_mult) at cap=75%. For each, identify good hours
from IS-only data, re-evaluate on filtered set.
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
from range_break_entry_signal_study import load_5min_features

CACHE_SIGS = Path(__file__).parent / "va_revert_signals.parquet"
OUT_TXT    = Path(__file__).parent.parent / "overnight range strat" / "tradelogs" / "robust_configs" / "va_revert_hourfilter_sweep.txt"
ET = "America/New_York"
FORCE_CLOSE = dt.time(16, 0)

CAP          = 75.0
N_GRID       = [5, 10, 15, 20]
SIG_D_GRID   = [0, 50, 100, 150]
CONF_D_GRID  = [0, 50, 100, 150]
SL_MULT_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
MIN_TOTAL_N  = 200
MIN_OOS_N    = 20
HOUR_PF_MIN  = 0.95     # keep hours where IS PF >= 0.95 (slightly losing OK)
MIN_COVERAGE = 0.50     # >=50% of months in window must have at least 1 trade
MAX_GAP_MO   = 6        # no 6+ month dry stretches


def simulate_exit(direction, entry_price, entry_idx, bars_day, sl_pts, tp_price):
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
        if tp_hit and sl_hit:
            return ("SL", sign * (sl_price - entry_price), bt)
        if tp_hit:
            return ("TP", sign * (tp_price - entry_price), bt)
        if sl_hit:
            return ("SL", sign * (sl_price - entry_price), bt)
    last = bars_day.iloc[-1]
    return ("held->close", sign * (float(last["close"]) - entry_price),
            pd.Timestamp(last["bar_open_time"]))


def run_chained(sigs, bars_by_day, sl_mult):
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
        entry_price = float(t["entry_price"])
        atr = float(t["atr_at_entry"])
        sl_pts = sl_mult * atr
        outcome, pnl, exit_t = simulate_exit(
            t["direction"], entry_price, entry_idx, bars_day, sl_pts, float(t["target"]))
        rows.append({"entry_time": ent_t, "date": d, "direction": t["direction"],
                     "outcome": outcome, "pnl": pnl,
                     "hour": ent_t.hour, "month": ent_t.strftime("%Y-%m")})
        last_exit = pd.Timestamp(exit_t)
    return pd.DataFrame(rows)


def pf(arr):
    arr = np.asarray(arr)
    if len(arr) == 0: return 0.0
    wins = arr[arr > 0].sum()
    losses = abs(arr[arr < 0].sum())
    if losses == 0: return float("inf") if wins > 0 else 0.0
    return wins / losses


def spread_stats(trades):
    if len(trades) == 0:
        return {"n_months": 0, "coverage": 0.0, "max_gap": 999, "median_per_month": 0}
    trades = trades.copy()
    trades["month_dt"] = pd.to_datetime(trades["month"] + "-01")
    months_present = trades["month_dt"].drop_duplicates().sort_values()
    full_range = pd.date_range(start=trades["month_dt"].min(), end=trades["month_dt"].max(), freq="MS")
    coverage = len(months_present) / max(len(full_range), 1)
    # Largest gap (in months) between consecutive trade-months
    if len(months_present) < 2:
        max_gap = 0
    else:
        diffs = months_present.diff().dt.days.iloc[1:] / 30
        max_gap = int(diffs.max())
    counts_per_month = trades.groupby("month_dt").size()
    return {"n_months": len(months_present),
            "coverage": coverage * 100,
            "max_gap": max_gap,
            "median_per_month": counts_per_month.median()}


def main():
    print("loading signals + bars ...")
    sigs = pd.read_parquet(CACHE_SIGS)
    sigs["date"] = pd.to_datetime(sigs["date"]).dt.date
    sigs["entry_time"] = pd.to_datetime(sigs["entry_time"])
    sigs = sigs[sigs["distance_pct"] < CAP].copy()
    print(f"  signals at cap={CAP}%: {len(sigs)}")
    bars_all, _ = load_5min_features((dt.date(2020,12,1), dt.date(2026,5,7)))
    bars_by_day = {d: g.sort_values("bar_open_time").reset_index(drop=True)
                   for d, g in bars_all.groupby("session_date")}

    print(f"sweeping configs (filter hours where IS-only PF >= {HOUR_PF_MIN}) ...")
    rows = []
    total = len(N_GRID) * len(SIG_D_GRID) * len(CONF_D_GRID) * len(SL_MULT_GRID)
    done = 0
    for N, sig_D, conf_D, sl in product(N_GRID, SIG_D_GRID, CONF_D_GRID, SL_MULT_GRID):
        sig_c  = f"sig_abs_w{N}"
        conf_c = f"conf_abs_w{N}"
        sub = sigs[(sigs[sig_c].fillna(0)  >= sig_D) &
                   (sigs[conf_c].fillna(0) >= conf_D)]
        if len(sub) < MIN_TOTAL_N:
            done += 1; continue
        trades = run_chained(sub, bars_by_day, sl)
        if len(trades) < MIN_TOTAL_N:
            done += 1; continue

        trades["period"] = np.where(trades["date"] < dt.date(2025,1,1), "IS", "OOS")
        is_only = trades[trades["period"] == "IS"]
        # Per-hour IS PF
        hour_pfs = is_only.groupby("hour")["pnl"].apply(lambda s: pf(s.values))
        # Keep hours with IS PF >= HOUR_PF_MIN
        keep_hours = set(hour_pfs[hour_pfs >= HOUR_PF_MIN].index.tolist())
        if not keep_hours:
            done += 1; continue
        filt = trades[trades["hour"].isin(keep_hours)]
        if len(filt) < MIN_TOTAL_N:
            done += 1; continue

        is_f  = filt[filt["period"] == "IS"]
        oos_f = filt[filt["period"] == "OOS"]
        if len(oos_f) < MIN_OOS_N:
            done += 1; continue
        is_pnl  = is_f["pnl"].values; oos_pnl = oos_f["pnl"].values
        if is_pnl.sum() <= 0 or oos_pnl.sum() <= 0:
            done += 1; continue

        spread = spread_stats(filt)
        if spread["coverage"] < MIN_COVERAGE * 100 or spread["max_gap"] > MAX_GAP_MO:
            done += 1; continue

        all_pnl = filt["pnl"].values
        rows.append({
            "N": N, "sig_D": sig_D, "conf_D": conf_D, "sl_mult": sl,
            "keep_hours": ",".join(str(h) for h in sorted(keep_hours)),
            "n_total": len(filt), "n_is": len(is_f), "n_oos": len(oos_f),
            "wr_all": float((all_pnl > 0).mean() * 100),
            "wr_is":  float((is_pnl > 0).mean() * 100),
            "wr_oos": float((oos_pnl > 0).mean() * 100),
            "pf_is":  pf(is_pnl), "pf_oos": pf(oos_pnl), "pf_all": pf(all_pnl),
            "total_is": is_pnl.sum(), "total_oos": oos_pnl.sum(), "total_all": all_pnl.sum(),
            "months_with_trades": spread["n_months"],
            "coverage_pct": spread["coverage"],
            "max_gap_months": spread["max_gap"],
            "median_trades_per_month": spread["median_per_month"],
        })
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{total}  qualifying so far: {len(rows)}")

    df = pd.DataFrame(rows)
    df["min_pf"] = df[["pf_is","pf_oos"]].min(axis=1)
    print(f"qualifying configs: {len(df)}")

    cols = ["N","sig_D","conf_D","sl_mult","keep_hours",
            "n_total","n_is","n_oos","wr_all","pf_is","pf_oos","min_pf",
            "total_is","total_oos","total_all",
            "months_with_trades","coverage_pct","max_gap_months","median_trades_per_month"]

    L = []
    L.append("=" * 220)
    L.append(f"VA-REVERT HOUR-FILTER SWEEP  (cap={CAP}%, IS-only hour PF >= {HOUR_PF_MIN}, "
             f"n>={MIN_TOTAL_N}, OOS_n>={MIN_OOS_N}, coverage>={int(MIN_COVERAGE*100)}%, gap<={MAX_GAP_MO}mo)")
    L.append("=" * 220)
    L.append(f"Found {len(df)} qualifying configs")
    if not df.empty:
        L.append("")
        L.append("TOP 20 BY min(IS_PF, OOS_PF):")
        L.append(df.sort_values("min_pf", ascending=False).head(20)[cols].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"))
        L.append("")
        L.append("TOP 20 BY combined total $:")
        L.append(df.sort_values("total_all", ascending=False).head(20)[cols].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"))
        L.append("")
        L.append("TOP 10 BY trade count (sample size):")
        L.append(df.sort_values("n_total", ascending=False).head(10)[cols].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"))

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

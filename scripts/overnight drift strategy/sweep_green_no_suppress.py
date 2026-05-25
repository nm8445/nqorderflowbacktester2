"""OD sweep — green widening WITHOUT yellow_suppress (= original live behavior).

Anchor: your ORIGINAL LIVE OD config (pre-upgrade):
  yellow_atr_mult=1.30, green_base=82.5, green_decay=1.5, green_atr_mult=1.0,
  yellow_suppress=0, martingale ON

Tests: keep yellow_suppress=0 + yellow_atr_mult=1.30 (live defaults), sweep
just green dimensions (green_base, green_decay, green_atr_mult).

Goal: find a config that captures some of the wider-green benefit WITHOUT the
yellow_suppress=30 tail risk (the cause of the 5/14 disaster).

IS/OOS 60/40 chronological split on full 5.5 yr dataset (now including 5/7-22).
Pass criteria: must beat ORIGINAL live config on both IS and OOS.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import (
    StrategyParams, build_full_20min_series, run_backtest, trades_to_df,
)

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"
OUT_DIR = Path(__file__).parent / "results"

# ANCHOR = original LIVE config (pre-upgrade)
LIVE_ORIGINAL = dict(
    yellow_atr_len=14, yellow_atr_mult=1.30,
    yellow_drift=0.0, yellow_mode="pure_ratchet",
    green_atr_len=14, green_atr_mult=1.00,
    green_base=82.5, green_decay=1.5,
    red_intercept=0.0, red_drift=0.45,
    use_be=False, use_martingale=True, base_qty=1, loss_qty=2,
    tp_intrabar_fill=False,
    yellow_suppress_bars=0,
)

# Sweep green dimensions only — keep yellow at original (1.30) and suppress=0
GREEN_BASE_GRID  = [82.5, 100, 120, 140, 160, 180, 200, 225, 250]
GREEN_DECAY_GRID = [1.0, 1.5, 2.0, 2.5]
GREEN_ATR_GRID   = [1.0, 1.25, 1.5, 1.75]

N_WORKERS = 6
_BARS = None


def _init_worker(bars):
    global _BARS
    _BARS = bars


def _run_cell(args):
    gb, gd, gatr = args
    cfg = dict(LIVE_ORIGINAL)
    cfg["green_base"] = gb
    cfg["green_decay"] = gd
    cfg["green_atr_mult"] = gatr
    trades = run_backtest(_BARS, StrategyParams(**cfg))
    if not trades:
        return {"green_base": gb, "green_decay": gd, "green_atr_mult": gatr, "n_all": 0}
    df = trades_to_df(trades)
    df["exit_date"] = pd.to_datetime(df["exit_time"]).dt.tz_convert("America/New_York").dt.normalize()
    df = df.sort_values("exit_date").reset_index(drop=True)
    df["pnl_$"] = (df["exit_price"] - df["entry_price"]) * df["qty"] * 20.0

    dates = sorted(df["exit_date"].dt.normalize().unique())
    cutoff_idx = int(len(dates) * 0.6)
    cutoff = dates[cutoff_idx] if cutoff_idx < len(dates) else dates[-1]
    is_mask  = df["exit_date"] <  cutoff
    oos_mask = df["exit_date"] >= cutoff

    def stats(sub):
        n = len(sub)
        if n == 0: return dict(n=0, net=0, pf=0, mdd=0, worst=0)
        pnls = sub["pnl_$"].values
        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else 99.0
        cum = pnls.cumsum()
        mdd = float((cum - np.maximum.accumulate(cum)).min())
        return dict(n=n, net=round(float(pnls.sum()), 0), pf=round(pf, 3),
                    mdd=round(mdd, 0), worst=round(float(pnls.min()), 0))
    s_is = stats(df[is_mask])
    s_oos = stats(df[oos_mask])
    s_all = stats(df)
    return {
        "green_base": gb, "green_decay": gd, "green_atr_mult": gatr,
        "n_all": s_all["n"], "all_net": s_all["net"], "all_PF": s_all["pf"],
        "all_mdd": s_all["mdd"], "all_worst": s_all["worst"],
        "is_net": s_is["net"], "is_PF": s_is["pf"], "is_mdd": s_is["mdd"],
        "oos_net": s_oos["net"], "oos_PF": s_oos["pf"], "oos_mdd": s_oos["mdd"],
        "oos_worst": s_oos["worst"],
    }


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Building 20-min bars...")
    t0 = time.time()
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(f"  {len(bars):,} bars ({time.time()-t0:.1f}s)")

    configs = [(gb, gd, gatr) for gb in GREEN_BASE_GRID for gd in GREEN_DECAY_GRID for gatr in GREEN_ATR_GRID]
    print(f"\n[{time.strftime('%H:%M:%S')}] Running {len(configs)} cells on {N_WORKERS} workers...")
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(bars,)) as ex:
        results = list(ex.map(_run_cell, configs))
    print(f"[{time.strftime('%H:%M:%S')}] Done in {time.time()-t1:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "od_green_no_suppress_sweep.csv", index=False)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

    anchor = df[(df["green_base"] == 82.5) & (df["green_decay"] == 1.5) & (df["green_atr_mult"] == 1.0)].iloc[0]
    print(f"\n=== ANCHOR (original live: green_base=82.5, decay=1.5, gatr=1.0, suppress=0) ===")
    print(f"  ALL: ${anchor['all_net']:,.0f}  PF {anchor['all_PF']}  MDD ${anchor['all_mdd']:,.0f}  worst ${anchor['all_worst']:,.0f}")
    print(f"  IS:  ${anchor['is_net']:,.0f}  PF {anchor['is_PF']}")
    print(f"  OOS: ${anchor['oos_net']:,.0f}  PF {anchor['oos_PF']}  worst ${anchor['oos_worst']:,.0f}")

    a_is = anchor["is_net"]; a_oos = anchor["oos_net"]; a_mdd = anchor["all_mdd"]
    swept = df[~((df["green_base"] == 82.5) & (df["green_decay"] == 1.5) & (df["green_atr_mult"] == 1.0))].copy()
    swept["IS_delta"] = (swept["is_net"] - a_is).round(0)
    swept["OOS_delta"] = (swept["oos_net"] - a_oos).round(0)
    swept["beat_both"] = (swept["IS_delta"] > 0) & (swept["OOS_delta"] > 0)
    swept["net_mdd_ratio"] = swept["all_net"] / swept["all_mdd"].abs()

    print(f"\n=== TOP 15 by COMBINED IS+OOS DELTA (must beat both) ===")
    cols = ["green_base", "green_decay", "green_atr_mult", "n_all",
            "all_net", "all_mdd", "all_worst", "is_net", "oos_net",
            "IS_delta", "OOS_delta", "all_PF"]
    winners = swept[swept["beat_both"]].assign(combined=lambda x: x["IS_delta"] + x["OOS_delta"]).sort_values("combined", ascending=False).head(15)
    if len(winners) == 0:
        print("  NONE beat original live config on both IS and OOS.")
    else:
        print(winners[cols].to_string(index=False))

    print(f"\n=== TOP 10 by NET/|MDD| RATIO (risk-adjusted, must beat anchor on both) ===")
    ra = swept[swept["beat_both"]].sort_values("net_mdd_ratio", ascending=False).head(10)
    if len(ra) == 0:
        print("  NONE")
    else:
        print(ra[cols + ["net_mdd_ratio"]].to_string(index=False))

    print(f"\n=== PARETO WINNERS (better net AND better worst-trade vs anchor) ===")
    pareto = swept[swept["beat_both"] & (swept["all_worst"] > anchor["all_worst"])].sort_values("all_net", ascending=False).head(10)
    if len(pareto) == 0:
        print("  NONE — wider green increases worst-trade size")
    else:
        print(pareto[cols].to_string(index=False))


if __name__ == "__main__":
    main()

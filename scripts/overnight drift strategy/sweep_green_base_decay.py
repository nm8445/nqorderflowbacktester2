"""OD strategy — sweep green_base and green_decay, IS/OOS validation.

Question: does increasing the green TP distance help OD extract more from
trending nights? Must hold up in IS AND OOS to be worth deploying.

Baseline (locked N=25 production config):
  green_base=82.5, green_decay=1.5, green_atr_mult=1.00,
  yellow_atr_mult=1.30, yellow_suppress_bars=25, use_martingale=True

Sweep:
  green_base in [60, 80, 100, 120, 150, 200, 250, 300]
  green_decay in [0.5, 1.0, 1.5, 2.0]
All other params held at locked values.

IS/OOS: chronological 60/40 split on trade exit dates.
Pass criteria: both IS and OOS net $ must beat baseline's respective phase.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import (  # noqa: E402
    StrategyParams,
    build_full_20min_series,
    run_backtest,
    trades_to_df,
)

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# Locked baseline values
BASE_GREEN = 82.5
BASE_DECAY = 1.5

# Sweep grid — fine grain around 150 and 200; extended high end to test > 300
GREEN_BASE_GRID = [130, 140, 145, 150, 155, 160, 170, 180, 190,
                   200, 215, 225, 250, 275, 300, 350, 400, 500]
GREEN_DECAY_GRID = [1.0, 1.25, 1.5, 1.75, 2.0]
N_WORKERS = 6

# Locked non-sweep params
LOCKED_PARAMS = dict(
    yellow_atr_len=14, yellow_atr_mult=1.30, yellow_drift=0.0,
    yellow_mode="pure_ratchet",
    green_atr_len=14, green_atr_mult=1.00,
    red_intercept=0.0, red_drift=0.45,
    use_be=False, use_martingale=True, base_qty=1, loss_qty=2,
    tp_intrabar_fill=False,
    yellow_suppress_bars=25,
)

_BARS = None


def _init_worker(bars):
    global _BARS
    _BARS = bars


def _run_cell(args):
    green_base, green_decay = args
    params = StrategyParams(green_base=green_base, green_decay=green_decay, **LOCKED_PARAMS)
    trades = run_backtest(_BARS, params)
    if not trades:
        return {"green_base": green_base, "green_decay": green_decay,
                "n_all": 0, "is_n": 0, "oos_n": 0}
    df = trades_to_df(trades)
    df["exit_date"] = pd.to_datetime(df["exit_time"]).dt.tz_convert("America/New_York").dt.normalize()
    df = df.sort_values("exit_date").reset_index(drop=True)
    # PnL in dollars per locked sizing (base 1, loss 2 contracts; NQ $20/pt)
    df["pnl_$"] = (df["exit_price"] - df["entry_price"]) * df["qty"] * 20.0

    # Chronological 60/40 split on unique session dates
    dates = df["exit_date"].dt.normalize().unique()
    dates = sorted(dates)
    cutoff_idx = int(len(dates) * 0.6)
    cutoff = dates[cutoff_idx] if cutoff_idx < len(dates) else dates[-1]
    is_mask  = df["exit_date"] <  cutoff
    oos_mask = df["exit_date"] >= cutoff

    def stats(sub):
        n = len(sub)
        if n == 0:
            return dict(n=0, wr=0, net=0, pf=0, mdd=0, avg=0)
        pnls = sub["pnl_$"].values
        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else 99.0
        cum = pnls.cumsum()
        mdd = float((cum - np.maximum.accumulate(cum)).min())
        return dict(
            n=n,
            wr=round(len(wins) / n * 100, 1),
            net=round(float(pnls.sum()), 0),
            pf=round(pf, 3),
            mdd=round(mdd, 0),
            avg=round(float(pnls.mean()), 1),
        )

    s_is  = stats(df[is_mask])
    s_oos = stats(df[oos_mask])
    s_all = stats(df)
    return {
        "green_base": green_base, "green_decay": green_decay,
        "n_all": s_all["n"], "all_net": s_all["net"], "all_PF": s_all["pf"], "all_mdd": s_all["mdd"],
        "is_n": s_is["n"], "is_wr": s_is["wr"], "is_net": s_is["net"], "is_PF": s_is["pf"], "is_mdd": s_is["mdd"],
        "oos_n": s_oos["n"], "oos_wr": s_oos["wr"], "oos_net": s_oos["net"], "oos_PF": s_oos["pf"], "oos_mdd": s_oos["mdd"],
    }


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Building 20-min bar series...")
    t0 = time.time()
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(f"[{time.strftime('%H:%M:%S')}]   {len(bars):,} bars ({time.time()-t0:.1f}s)")

    # All (green_base, green_decay) combos + baseline
    configs = [(gb, gd) for gb in GREEN_BASE_GRID for gd in GREEN_DECAY_GRID]
    configs_with_baseline = [(BASE_GREEN, BASE_DECAY)] + configs
    print(f"\n[{time.strftime('%H:%M:%S')}] Running {len(configs_with_baseline)} configs on {N_WORKERS} workers...")
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(bars,)) as ex:
        results = list(ex.map(_run_cell, configs_with_baseline))
    print(f"[{time.strftime('%H:%M:%S')}] Done in {time.time()-t1:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "od_green_sweep.csv", index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

    baseline = df[(df["green_base"] == BASE_GREEN) & (df["green_decay"] == BASE_DECAY)].iloc[0]
    bl_is = baseline["is_net"]
    bl_oos = baseline["oos_net"]
    bl_all = baseline["all_net"]
    bl_is_pf = baseline["is_PF"]
    bl_oos_pf = baseline["oos_PF"]

    print(f"\n=== BASELINE (locked: green_base={BASE_GREEN}, green_decay={BASE_DECAY}) ===")
    print(f"  ALL: ${bl_all:,.0f}  PF {baseline['all_PF']}  MDD ${baseline['all_mdd']:,.0f}")
    print(f"  IS:  ${bl_is:,.0f}  PF {bl_is_pf}   n={int(baseline['is_n'])}")
    print(f"  OOS: ${bl_oos:,.0f}  PF {bl_oos_pf}   n={int(baseline['oos_n'])}")

    # Filter sweep results (exclude baseline)
    swept = df[~((df["green_base"] == BASE_GREEN) & (df["green_decay"] == BASE_DECAY))].copy()
    swept["beat_IS"]  = swept["is_net"]  > bl_is
    swept["beat_OOS"] = swept["oos_net"] > bl_oos
    swept["beat_both"] = swept["beat_IS"] & swept["beat_OOS"]
    swept["IS_delta"]  = (swept["is_net"]  - bl_is).round(0)
    swept["OOS_delta"] = (swept["oos_net"] - bl_oos).round(0)

    print(f"\n=== SWEEP RESULTS (sorted by IS+OOS combined delta) ===")
    swept["combined_delta"] = swept["IS_delta"] + swept["OOS_delta"]
    cols_show = ["green_base", "green_decay", "n_all", "is_net", "is_PF", "oos_net", "oos_PF",
                 "all_net", "all_mdd", "IS_delta", "OOS_delta", "beat_both"]
    sorted_swept = swept.sort_values("combined_delta", ascending=False)
    print(sorted_swept[cols_show].to_string(index=False))

    print(f"\n=== CONFIGS BEATING BASELINE IN BOTH IS AND OOS ===")
    winners = sorted_swept[sorted_swept["beat_both"]]
    if len(winners) == 0:
        print("  NONE. Baseline holds up — no green_base or green_decay change beats it in both phases.")
    else:
        print(winners[cols_show].to_string(index=False))

    # --- Special focus: 150/1.5 + neighbors (low-MDD candidate) ---
    print(f"\n=== FOCUS: 150/1.5 NEIGHBORHOOD (low-MDD candidate) ===")
    nbr150 = df[(df["green_base"].between(140, 170)) & (df["green_decay"].between(1.25, 1.75))]
    print(nbr150[["green_base", "green_decay", "n_all", "is_net", "is_PF",
                  "oos_net", "oos_PF", "all_net", "all_mdd"]]
          .sort_values(["green_base", "green_decay"]).to_string(index=False))

    # --- Special focus: 200/x.x peak analysis ---
    print(f"\n=== FOCUS: 200 PEAK + EXTREME HIGH END (180/190/200/215/225/...500) ===")
    high_end = df[df["green_base"] >= 180].sort_values(["green_base", "green_decay"])
    print(high_end[["green_base", "green_decay", "n_all", "is_net", "is_PF",
                    "oos_net", "oos_PF", "all_net", "all_mdd"]].to_string(index=False))

    # --- Best Net / MDD ratio (risk-adjusted) ---
    print(f"\n=== TOP 10 by NET/|MDD| RATIO (risk-adjusted) ===")
    df["net_mdd_ratio"] = df["all_net"] / df["all_mdd"].abs()
    best_ratio = df[df["all_net"] > 0].sort_values("net_mdd_ratio", ascending=False).head(10)
    print(best_ratio[["green_base", "green_decay", "all_net", "all_mdd", "net_mdd_ratio",
                       "is_net", "oos_net"]].to_string(index=False))

    print(f"\nSaved: {OUT_DIR / 'od_green_sweep.csv'}")


if __name__ == "__main__":
    main()

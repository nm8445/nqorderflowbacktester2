"""OD strategy — 3D sweep on top of validated 160/2.0 base.

Anchor: green_base=160, green_decay=2.0 (validated by full overfit suite,
4/5 strict + 1 borderline same as locked baseline).

Sweep:
  yellow_suppress_bars: [25, 28, 30, 32, 35]
  yellow_atr_mult:      [1.2, 1.3, 1.4, 1.5]
  green_atr_mult:       [0.75, 1.0, 1.25, 1.5]

5 * 4 * 4 = 80 cells. IS/OOS 60/40 chronological split.
Pass criteria: beat 160/2.0 anchor on BOTH IS and OOS net $.

Single-dimension Test 1 hinted at improvements (yellow_suppress=30 → +11%
PnL, yellow_atr=1.4 → +2%, green_atr=1.5 → +4%). This sweep checks if any
combination holds up under proper IS/OOS scrutiny.
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

# Anchor (validated 160/2.0)
ANCHOR_PARAMS = dict(
    yellow_atr_len=14, yellow_atr_mult=1.30, yellow_drift=0.0,
    yellow_mode="pure_ratchet",
    green_atr_len=14, green_atr_mult=1.00,
    green_base=160.0, green_decay=2.0,
    red_intercept=0.0, red_drift=0.45,
    use_be=False, use_martingale=True, base_qty=1, loss_qty=2,
    tp_intrabar_fill=False,
    yellow_suppress_bars=25,
)

# Sweep dimensions
YELLOW_SUPPRESS_GRID = [25, 28, 30, 32, 35]
YELLOW_ATR_GRID      = [1.2, 1.3, 1.4, 1.5]
GREEN_ATR_GRID       = [0.75, 1.0, 1.25, 1.5]

N_WORKERS = 6
_BARS = None


def _init_worker(bars):
    global _BARS
    _BARS = bars


def _run_cell(args):
    ys, yatr, gatr = args
    cfg = dict(ANCHOR_PARAMS)
    cfg["yellow_suppress_bars"] = ys
    cfg["yellow_atr_mult"] = yatr
    cfg["green_atr_mult"] = gatr
    params = StrategyParams(**cfg)
    trades = run_backtest(_BARS, params)
    if not trades:
        return {"yellow_suppress": ys, "yellow_atr_mult": yatr, "green_atr_mult": gatr, "n_all": 0}
    df = trades_to_df(trades)
    df["exit_date"] = pd.to_datetime(df["exit_time"]).dt.tz_convert("America/New_York").dt.normalize()
    df = df.sort_values("exit_date").reset_index(drop=True)
    df["pnl_$"] = (df["exit_price"] - df["entry_price"]) * df["qty"] * 20.0

    # Chronological 60/40 split on session dates
    dates = sorted(df["exit_date"].dt.normalize().unique())
    cutoff_idx = int(len(dates) * 0.6)
    cutoff = dates[cutoff_idx] if cutoff_idx < len(dates) else dates[-1]
    is_mask  = df["exit_date"] <  cutoff
    oos_mask = df["exit_date"] >= cutoff

    def stats(sub):
        n = len(sub)
        if n == 0: return dict(n=0, wr=0, net=0, pf=0, mdd=0)
        pnls = sub["pnl_$"].values
        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else 99.0
        cum = pnls.cumsum()
        mdd = float((cum - np.maximum.accumulate(cum)).min())
        return dict(n=n, wr=round(len(wins)/n*100, 1),
                    net=round(float(pnls.sum()), 0), pf=round(pf, 3),
                    mdd=round(mdd, 0))
    s_is  = stats(df[is_mask])
    s_oos = stats(df[oos_mask])
    s_all = stats(df)
    return {
        "yellow_suppress": ys, "yellow_atr_mult": yatr, "green_atr_mult": gatr,
        "n_all": s_all["n"], "all_net": s_all["net"], "all_PF": s_all["pf"], "all_mdd": s_all["mdd"],
        "is_net": s_is["net"], "is_PF": s_is["pf"], "is_mdd": s_is["mdd"],
        "oos_net": s_oos["net"], "oos_PF": s_oos["pf"], "oos_mdd": s_oos["mdd"],
    }


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Building 20-min bars...")
    t0 = time.time()
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(f"[{time.strftime('%H:%M:%S')}]   {len(bars):,} bars ({time.time()-t0:.1f}s)")

    configs = [(ys, yatr, gatr)
               for ys in YELLOW_SUPPRESS_GRID
               for yatr in YELLOW_ATR_GRID
               for gatr in GREEN_ATR_GRID]
    # Anchor (160/2.0 with default yellow=25, yellow_atr=1.3, green_atr=1.0) is in the grid as (25, 1.3, 1.0)
    print(f"\n[{time.strftime('%H:%M:%S')}] Running {len(configs)} cells on {N_WORKERS} workers...")
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(bars,)) as ex:
        results = list(ex.map(_run_cell, configs))
    print(f"[{time.strftime('%H:%M:%S')}] Done in {time.time()-t1:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "od_yellow_atr_3d_sweep.csv", index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

    # Find anchor row
    anchor_row = df[(df["yellow_suppress"] == 25) & (df["yellow_atr_mult"] == 1.3) & (df["green_atr_mult"] == 1.0)].iloc[0]
    a_is = anchor_row["is_net"]; a_oos = anchor_row["oos_net"]; a_all = anchor_row["all_net"]
    a_mdd = anchor_row["all_mdd"]
    print(f"\n=== ANCHOR (validated 160/2.0, yellow_suppress=25, yellow_atr=1.3, green_atr=1.0) ===")
    print(f"  ALL: ${a_all:,.0f}  IS: ${a_is:,.0f}  OOS: ${a_oos:,.0f}  MDD: ${a_mdd:,.0f}")

    swept = df[~((df["yellow_suppress"] == 25) & (df["yellow_atr_mult"] == 1.3) & (df["green_atr_mult"] == 1.0))].copy()
    swept["IS_delta"]  = (swept["is_net"]  - a_is).round(0)
    swept["OOS_delta"] = (swept["oos_net"] - a_oos).round(0)
    swept["beat_IS"]   = swept["is_net"]  > a_is
    swept["beat_OOS"]  = swept["oos_net"] > a_oos
    swept["beat_both"] = swept["beat_IS"] & swept["beat_OOS"]
    swept["combined_delta"] = swept["IS_delta"] + swept["OOS_delta"]
    swept["net_mdd_ratio"]  = swept["all_net"] / swept["all_mdd"].abs()

    print(f"\n=== TOP 20 by COMBINED IS+OOS DELTA (must beat both) ===")
    winners = swept[swept["beat_both"]].sort_values("combined_delta", ascending=False).head(20)
    cols = ["yellow_suppress", "yellow_atr_mult", "green_atr_mult",
            "all_net", "all_mdd", "is_net", "oos_net", "IS_delta", "OOS_delta", "all_PF"]
    if len(winners) == 0:
        print("  NONE. Anchor 160/2.0 at default yellow/atr is already optimal in this region.")
    else:
        print(winners[cols].to_string(index=False))

    print(f"\n=== TOP 10 by NET/|MDD| RATIO (risk-adjusted, must beat both IS+OOS) ===")
    ratio_winners = swept[swept["beat_both"] & (swept["all_net"] > 0)].sort_values("net_mdd_ratio", ascending=False).head(10)
    if len(ratio_winners) == 0:
        print("  NONE.")
    else:
        print(ratio_winners[cols + ["net_mdd_ratio"]].to_string(index=False))

    print(f"\n=== PARETO WINNERS (better than anchor on BOTH net AND mdd) ===")
    pareto = swept[swept["beat_both"] & (swept["all_mdd"] > a_mdd)].sort_values("all_net", ascending=False).head(10)
    if len(pareto) == 0:
        print("  NONE. Any net improvement comes with worse MDD.")
    else:
        print(pareto[cols].to_string(index=False))

    print(f"\n=== ALL CELLS BEATING ANCHOR ON BOTH IS+OOS (count: {swept['beat_both'].sum()} / {len(swept)}) ===")
    print(f"\nSaved: {OUT_DIR / 'od_yellow_atr_3d_sweep.csv'}")


if __name__ == "__main__":
    main()

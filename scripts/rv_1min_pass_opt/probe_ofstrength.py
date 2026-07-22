"""Probe: does a STRONGER orderflow imbalance amplify the momentum edge?

Uses the magnitude cache (best window delta per bar) to sweep threshold d for free. Reports
mom_cont win rate IS/OOS + volume, all-session and RTH-afternoon-only (the best session bucket).
If a stronger d pushes win rate past ~56% at usable volume, there's a thread; else conclude.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from live.combined.rv_features import RVConfig
from features import compute_features
from data import load_bars, build_orderflow_magnitudes
from backtest import run, BTParams

SPLIT = pd.Timestamp("2024-04-05")
D_GRID = [20, 40, 60, 80, 120, 160]
HZ = 1.0


def main():
    bars = load_bars()
    feat = compute_features(bars["close"].to_numpy(), bars["high"].to_numpy(),
                            bars["low"].to_numpy(), RVConfig(NORM_LEN=200, Z_LOOKBACK=100))
    mpath = build_orderflow_magnitudes(8)
    mags = pd.read_parquet(mpath)
    ml = mags["best_long_mag"].reindex(bars.index, fill_value=0.0).to_numpy()
    ms = mags["best_short_mag"].reindex(bars.index, fill_value=0.0).to_numpy()
    nd = bars["et"].dt.tz_localize(None).dt.normalize().nunique()

    print(f"mom_cont, HIGH_Z={HZ}; sweeping orderflow d (all-session | RTHpm 12-16 ET only)\n")
    print(f"  {'d':>4} {'ISwin':>6} {'OOSwin':>7} {'tr/d':>5} | {'pm_ISwin':>8} {'pm_OOSwin':>9} {'pm_tr/d':>7}")
    for d in D_GRID:
        lo = ml <= -d
        so = ms >= d
        tr = run(bars, feat, BTParams(high_z=HZ, k=8.0, atr_max=150.0,
                 use_orderflow=True, direction_mode="mom_cont"), lo, so)
        ent = pd.to_datetime(tr["entry_ts"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
        hr = (ent.dt.hour + ent.dt.minute / 60.0).to_numpy()
        is_m = (ent < SPLIT).to_numpy()
        win = (tr["pnl_$"] > 0).to_numpy()
        pm = (hr >= 12) & (hr < 16)
        def wr(mask):
            return win[mask].mean() * 100 if mask.any() else float("nan")
        print(f"  {d:>4} {wr(is_m):>5.1f}% {wr(~is_m):>6.1f}% {len(tr)/nd:>5.2f} | "
              f"{wr(pm & is_m):>7.1f}% {wr(pm & ~is_m):>8.1f}% {pm.sum()/nd:>7.2f}")


if __name__ == "__main__":
    main()

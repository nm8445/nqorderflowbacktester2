"""Probe: is the momentum+orderflow edge concentrated in a session window?

Buckets mom_cont+orderflow(d=40) trades by ET entry hour and by named session, IS vs OOS win rate +
volume. If a session shows materially higher win rate (>~55%) with usable volume, that's the thread.
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
from data import load_bars
from backtest import run, BTParams

SPLIT = pd.Timestamp("2024-04-05")
SESSIONS = [("Asia 19-00", 19, 24), ("Eur 00-04", 0, 4), ("Eur 04-08", 4, 8),
            ("preRTH 08-0930", 8, 9.5), ("RTHam 0930-12", 9.5, 12),
            ("RTHpm 12-16", 12, 16)]


def main():
    bars = load_bars()
    feat = compute_features(bars["close"].to_numpy(), bars["high"].to_numpy(),
                            bars["low"].to_numpy(), RVConfig(NORM_LEN=200, Z_LOOKBACK=100))
    fl = pd.read_parquet(HERE / "cache" / "orderflow_flags_n8_d40.parquet")
    lo = fl["long_ok"].reindex(bars.index, fill_value=False).to_numpy()
    so = fl["short_ok"].reindex(bars.index, fill_value=False).to_numpy()
    nd = bars["et"].dt.tz_localize(None).dt.normalize().nunique()

    for hz in (1.0, 2.0):
        tr = run(bars, feat, BTParams(high_z=hz, k=8.0, atr_max=150.0,
                 use_orderflow=True, direction_mode="mom_cont"), lo, so)
        ent = pd.to_datetime(tr["entry_ts"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
        hr = ent.dt.hour + ent.dt.minute / 60.0
        is_m = (ent < SPLIT).to_numpy()
        win = (tr["pnl_$"] > 0).to_numpy()
        print(f"\n=== mom_cont + orderflow(d=40), HIGH_Z={hz} ===")
        print(f"  {'session':>16} {'ISwin':>6} {'OOSwin':>7} {'tr/day':>7} {'n':>6}")
        for name, h0, h1 in SESSIONS:
            sel = ((hr >= h0) & (hr < h1)).to_numpy()
            if sel.sum() == 0:
                continue
            isw = win[sel & is_m].mean() * 100 if (sel & is_m).any() else float("nan")
            osw = win[sel & ~is_m].mean() * 100 if (sel & ~is_m).any() else float("nan")
            print(f"  {name:>16} {isw:>5.1f}% {osw:>6.1f}% {sel.sum()/nd:>7.2f} {sel.sum():>6}")


if __name__ == "__main__":
    main()

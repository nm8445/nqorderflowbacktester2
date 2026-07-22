"""Edge-existence probe: does ANY direction bet have edge after a vol spike, at high throughput?

Sweeps direction_mode x HIGH_Z x orderflow, reporting IS gross win rate AND trades/day side by side
(pass-rate optimization wants BOTH edge and volume). If nothing clears ~53-54% win, 1-min RV at 1:1
is structurally a coinflip. Cost-independent (gross win rate) is the edge read; net EV @ k=8 shown too.
"""
from __future__ import annotations

import sys
import time
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

CFG = RVConfig(NORM_LEN=200, Z_LOOKBACK=100)   # mid-grid; stage1 showed lookback ~irrelevant
MODES = ["ema_cont", "ema_revert", "mom_cont", "mom_fade"]
HZ_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
K = 8.0
COST = 2.0
SPLIT = pd.Timestamp("2024-04-05")


def main():
    t0 = time.time()
    bars = load_bars()
    n_days = bars["et"].dt.tz_localize(None).dt.normalize().nunique()
    feat = compute_features(bars["close"].to_numpy(), bars["high"].to_numpy(),
                            bars["low"].to_numpy(), CFG)
    flags = pd.read_parquet(HERE / "cache" / "orderflow_flags_n8_d40.parquet")
    lo = flags["long_ok"].reindex(bars.index, fill_value=False).to_numpy()
    so = flags["short_ok"].reindex(bars.index, fill_value=False).to_numpy()
    ent_et = lambda tr: pd.to_datetime(tr["entry_ts"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    print(f"loaded ({time.time()-t0:.0f}s); cfg={CFG.NORM_LEN}/{CFG.Z_LOOKBACK}; {n_days} trading days\n")

    for of in (False, True):
        tag = f"orderflow {'ON d=40' if of else 'OFF'}"
        print(f"=== {tag} ===")
        print(f"  {'mode':>11} " + " ".join(f"{hz:>5.1f}z" for hz in HZ_GRID))
        # win-rate block
        for mode in MODES:
            wins, vols = [], []
            for hz in HZ_GRID:
                tr = run(bars, feat, BTParams(high_z=hz, k=K, atr_max=150.0,
                         use_orderflow=of, direction_mode=mode), lo, so)
                if tr.empty:
                    wins.append(np.nan); vols.append(0.0); continue
                im = (ent_et(tr) < SPLIT).to_numpy()
                wins.append((tr["pnl_$"].to_numpy()[im] > 0).mean() if im.any() else np.nan)
                vols.append(len(tr) / n_days)
            print(f"  win% {mode:>6} " + " ".join(f"{w*100:>5.1f}" if np.isfinite(w) else "  -  " for w in wins))
            print(f"  tr/d {mode:>6} " + " ".join(f"{v:>5.1f}" for v in vols))
        print()
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

"""
Constrained pure_ratchet grid sweep with per-fold robustness scoring.

Locked:
  yellow_mode      = pure_ratchet
  yellow_atr_len   = 14
  yellow_drift     = inert in pure_ratchet (set to 0.0)
  green_atr_len    = 14
  red_intercept    = 0.0  (Pine default)
  red_drift        = 0.45 (Pine default)
  use_be           = False
  use_martingale   = False, qty=1

Sweep:
  yellow_atr_mult  in [1.00, 1.15, 1.30, 1.45, 1.60]                          (5)
  green_atr_mult   in [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50]              (7)
  green_base       in [75, 77.5, ..., 120]                                    (19)
  green_decay      in [1.00, 1.10, ..., 2.00]                                 (11)

Total: 5 * 7 * 19 * 11 = 7,315 configs.

For each config, compute PF and gross over 5 windows:
  Pre-fold: 2020-12 .. 2022-11
  Fold 1:   2022-12 .. 2023-11
  Fold 2:   2023-12 .. 2024-11
  Fold 3:   2024-12 .. 2025-11
  Fold 4:   2025-12 .. 2026-05

Then filter by per-fold robustness (min PF threshold) and rank.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import (  # noqa: E402
    StrategyParams,
    build_full_20min_series,
    run_backtest,
    trades_to_df,
)

PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLES = "D:/trading_pythonbacktest_data/timebars_5min"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
TZ = "America/New_York"

FOLDS = [
    ("Pre",  "2020-12-01", "2022-11-30"),
    ("F1",   "2022-12-01", "2023-11-30"),
    ("F2",   "2023-12-01", "2024-11-30"),
    ("F3",   "2024-12-01", "2025-11-30"),
    ("F4",   "2025-12-01", "2026-05-31"),
]

_BARS: pd.DataFrame | None = None


def _init(bars: pd.DataFrame) -> None:
    global _BARS
    _BARS = bars


def _to_params(s: dict) -> StrategyParams:
    return StrategyParams(
        yellow_atr_len=14,
        yellow_atr_mult=s["y_mult"],
        yellow_drift=0.0,
        yellow_mode="pure_ratchet",
        green_atr_len=14,
        green_atr_mult=s["g_mult"],
        green_base=s["g_base"],
        green_decay=s["g_decay"],
        red_intercept=0.0,
        red_drift=0.45,
        use_be=False,
        use_martingale=False,
        base_qty=1,
        loss_qty=1,
    )


def _trial(args: tuple[int, dict]) -> tuple[int, dict, np.ndarray, np.ndarray]:
    tid, s = args
    trades = run_backtest(_BARS, _to_params(s))
    if not trades:
        return tid, s, np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    df = trades_to_df(trades)
    et = pd.to_datetime(df["entry_time"]).astype("int64").to_numpy()
    pnl = df["pnl_dollars"].to_numpy(dtype=np.float64)
    return tid, s, et, pnl


def _fold_stats(pnl: np.ndarray) -> dict:
    if len(pnl) == 0:
        return {"trades": 0, "pf": np.nan, "gross": 0.0, "win": np.nan}
    wins = (pnl > 0).sum()
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    pf = gw / gl if gl > 0 else float("inf")
    return {"trades": int(len(pnl)), "pf": float(pf), "gross": float(pnl.sum()), "win": float(wins / len(pnl) * 100)}


def main(n_trials_cap: int | None = None) -> None:
    print("Loading bars...", flush=True)
    bars = build_full_20min_series(PARQUET, PICKLES)
    print(f"  bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}\n", flush=True)

    Y_MULTS = [1.00, 1.15, 1.30, 1.45, 1.60]
    G_MULTS = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50]
    G_BASES = [round(75 + 2.5 * i, 1) for i in range(19)]  # 75..120
    G_DECAYS = [round(1.0 + 0.10 * i, 2) for i in range(11)]  # 1.0..2.0

    configs = []
    for ym in Y_MULTS:
        for gm in G_MULTS:
            for gb in G_BASES:
                for gd in G_DECAYS:
                    configs.append(
                        {"y_mult": ym, "g_mult": gm, "g_base": gb, "g_decay": gd}
                    )
    if n_trials_cap:
        configs = configs[:n_trials_cap]
    print(f"Configs to run: {len(configs)}")

    fold_ns = []
    for _, lo, hi in FOLDS:
        fold_ns.append((
            pd.Timestamp(lo, tz=TZ).value,
            pd.Timestamp(hi + " 23:59:59", tz=TZ).value,
        ))

    workers = 4
    print(f"Workers: {workers}\n")

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(bars,)) as ex:
        futs = [ex.submit(_trial, (i, c)) for i, c in enumerate(configs)]
        done = 0
        for f in as_completed(futs):
            tid, s, et, pnl = f.result()
            r = {"trial": tid, **s}
            # full-period
            full = _fold_stats(pnl)
            r["all_trades"] = full["trades"]
            r["all_pf"] = full["pf"]
            r["all_gross"] = full["gross"]
            r["all_win"] = full["win"]
            # per-fold
            for (label, _, _), (lo_ns, hi_ns) in zip(FOLDS, fold_ns):
                m = (et >= lo_ns) & (et <= hi_ns)
                fs = _fold_stats(pnl[m])
                r[f"{label}_pf"] = fs["pf"]
                r[f"{label}_gross"] = fs["gross"]
                r[f"{label}_trades"] = fs["trades"]
            # min PF across folds (robustness)
            fold_pfs = [r[f"{lbl}_pf"] for lbl, _, _ in FOLDS]
            finite_pfs = [p for p in fold_pfs if np.isfinite(p)]
            r["min_fold_pf"] = min(finite_pfs) if finite_pfs else np.nan
            r["mean_fold_pf"] = float(np.mean(finite_pfs)) if finite_pfs else np.nan
            r["num_folds_pf_ge_1"] = sum(1 for p in fold_pfs if p >= 1.0)
            rows.append(r)
            done += 1
            if done % 500 == 0 or done == len(configs):
                rate = done / (time.time() - t0)
                eta = (len(configs) - done) / rate if rate else 0
                print(f"  {done}/{len(configs)}  ({rate:.1f}/s, eta {eta:.0f}s)", flush=True)
    print(f"\nSwept {len(rows)} configs in {time.time() - t0:.1f}s\n")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pure_ratchet_sweep.csv", index=False)
    print(f"Saved -> {OUT / 'pure_ratchet_sweep.csv'}")

    print(f"\nDistribution of min_fold_pf:")
    print(f"  >= 1.30: {(df['min_fold_pf']>=1.30).sum()}")
    print(f"  >= 1.20: {(df['min_fold_pf']>=1.20).sum()}")
    print(f"  >= 1.10: {(df['min_fold_pf']>=1.10).sum()}")
    print(f"  >= 1.05: {(df['min_fold_pf']>=1.05).sum()}")
    print(f"  >= 1.00: {(df['min_fold_pf']>=1.00).sum()}")
    print(f"  total:   {len(df)}")

    print(f"\nDistribution of num_folds_pf_ge_1 (out of 5):")
    print(df["num_folds_pf_ge_1"].value_counts().sort_index().to_string())

    # Robust filter: min_fold_pf >= 1.05 (every fold turns a real profit)
    print(f"\n=== TOP 20 by min_fold_pf (robust: every fold PF >= 1.05) ===")
    rob = df[df["min_fold_pf"] >= 1.05].copy()
    rob = rob.sort_values("min_fold_pf", ascending=False).head(20).reset_index(drop=True)
    show_cols = [
        "y_mult", "g_mult", "g_base", "g_decay",
        "all_pf", "all_gross", "min_fold_pf", "mean_fold_pf",
        "Pre_pf", "F1_pf", "F2_pf", "F3_pf", "F4_pf",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(rob[show_cols].round(2).to_string())

    # Also top by all_pf (full period)
    print(f"\n=== TOP 15 by all-period PF (no robustness filter) ===")
    top = df.sort_values("all_pf", ascending=False).head(15).reset_index(drop=True)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(top[show_cols].round(2).to_string())

    # Best by gross when min_fold_pf >= 1.0
    print(f"\n=== TOP 15 by gross_$ (min_fold_pf >= 1.0) ===")
    g = df[df["min_fold_pf"] >= 1.0].copy()
    g = g.sort_values("all_gross", ascending=False).head(15).reset_index(drop=True)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(g[show_cols].round(2).to_string())


if __name__ == "__main__":
    main()

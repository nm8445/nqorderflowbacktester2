"""
Walk-forward optimizer with yellow_mode locked to `pure_ratchet`.

Design:
- Sample N random configs once.
- Run each backtest on the FULL bar series (parallelized).
- For each fold, slice trades into train / test windows by entry_time.
- Within each fold: rank configs by TRAIN profit factor (with min-trade
  floor), pick the top one, record TEST metrics for that config.
- Stitch all test-window trades from selected configs end-to-end => "honest"
  out-of-sample equity curve.

Folds: rolling 24-month train / 12-month test / 12-month step.
  Fold 1: train 2020-12..2022-11 | test 2022-12..2023-11
  Fold 2: train 2021-12..2023-11 | test 2023-12..2024-11
  Fold 3: train 2022-12..2024-11 | test 2024-12..2025-11
  Fold 4: train 2023-12..2025-11 | test 2025-12..2026-05 (short tail)

Comparison baselines reported alongside walk-forward stitched OOS:
- Default Pine config (yellow_mode=pure_ratchet, others as v4 defaults).
- Pure-ratchet config that maximises PF on the FULL 5yr (pretend-perfect
  in-sample fit).

Usage:
    python "scripts/overnight drift strategy/walk_forward.py" 1500
"""

from __future__ import annotations

import os
import random
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

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

TZ = "America/New_York"

# Walk-forward folds (each as (train_start, train_end, test_start, test_end])
FOLDS = [
    ("2020-12-01", "2022-11-30", "2022-12-01", "2023-11-30"),
    ("2021-12-01", "2023-11-30", "2023-12-01", "2024-11-30"),
    ("2022-12-01", "2024-11-30", "2024-12-01", "2025-11-30"),
    ("2023-12-01", "2025-11-30", "2025-12-01", "2026-05-31"),
]
MIN_TRAIN_TRADES = 150  # require this many train trades to be eligible


_BARS: pd.DataFrame | None = None


def _init(bars: pd.DataFrame) -> None:
    global _BARS
    _BARS = bars


def sample_params(rng: random.Random) -> dict:
    """Pure-ratchet locked. yellow_drift is unused by pure_ratchet so we drop it."""
    return {
        "atr_len": rng.choice([7, 10, 12, 14, 16, 20, 25, 30]),
        "yellow_atr_mult": round(rng.uniform(0.75, 3.0), 2),
        "green_atr_mult": round(rng.uniform(0.5, 4.0), 2),
        "green_base": round(rng.uniform(20.0, 200.0), 1),
        "green_decay": round(rng.uniform(0.0, 3.0), 2),
        "red_intercept": round(rng.uniform(-50.0, 50.0), 1),
        "red_drift": round(rng.uniform(0.0, 1.5), 2),
    }


def _to_strategy_params(s: dict) -> StrategyParams:
    return StrategyParams(
        yellow_atr_len=s["atr_len"],
        yellow_atr_mult=s["yellow_atr_mult"],
        yellow_drift=0.0,  # unused by pure_ratchet
        yellow_mode="pure_ratchet",
        green_atr_len=s["atr_len"],
        green_atr_mult=s["green_atr_mult"],
        green_base=s["green_base"],
        green_decay=s["green_decay"],
        red_intercept=s["red_intercept"],
        red_drift=s["red_drift"],
        use_be=False,
        use_martingale=False,
        base_qty=1,
        loss_qty=1,
    )


def _trial(args: tuple[int, dict]) -> tuple[int, dict, np.ndarray, np.ndarray]:
    """Run one backtest. Return (trial_id, params_dict, entry_unix_ns, pnl_dollars)."""
    trial_id, sample = args
    params = _to_strategy_params(sample)
    trades_list = run_backtest(_BARS, params)
    if not trades_list:
        return trial_id, sample, np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    df = trades_to_df(trades_list)
    et = pd.to_datetime(df["entry_time"]).astype("int64").to_numpy()
    pnl = df["pnl_dollars"].to_numpy(dtype=np.float64)
    return trial_id, sample, et, pnl


def _stats(pnl: np.ndarray) -> dict:
    if len(pnl) == 0:
        return {"trades": 0, "win%": np.nan, "avg_$": np.nan, "gross_$": 0.0, "PF": np.nan}
    wins = (pnl > 0).sum()
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    return {
        "trades": int(len(pnl)),
        "win%": float(wins / len(pnl) * 100),
        "avg_$": float(pnl.mean()),
        "gross_$": float(pnl.sum()),
        "PF": float(gw / gl) if gl > 0 else float("inf"),
    }


def main(n_trials: int) -> None:
    print("Loading bars...", flush=True)
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(f"  bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}", flush=True)

    rng = random.Random(7)
    samples = [sample_params(rng) for _ in range(n_trials)]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Running {n_trials} pure_ratchet trials on {workers} workers...", flush=True)

    t0 = time.time()
    results: dict[int, tuple[dict, np.ndarray, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(bars,)) as ex:
        futs = [ex.submit(_trial, (i, s)) for i, s in enumerate(samples)]
        done = 0
        for f in as_completed(futs):
            tid, s, et, pnl = f.result()
            results[tid] = (s, et, pnl)
            done += 1
            if done % 200 == 0 or done == n_trials:
                rate = done / (time.time() - t0)
                eta = (n_trials - done) / rate if rate > 0 else 0
                print(f"  {done}/{n_trials}  ({rate:.1f}/s, eta {eta:.0f}s)", flush=True)
    print(f"\nBacktests done in {time.time() - t0:.1f}s\n", flush=True)

    # ---------- per-fold selection ----------
    fold_rows = []
    stitched_pnl: list[np.ndarray] = []
    stitched_et: list[np.ndarray] = []
    selected_per_fold = []

    for fi, (tr0, tr1, te0, te1) in enumerate(FOLDS):
        tr0_ns = pd.Timestamp(tr0, tz=TZ).value
        tr1_ns = pd.Timestamp(tr1 + " 23:59:59", tz=TZ).value
        te0_ns = pd.Timestamp(te0, tz=TZ).value
        te1_ns = pd.Timestamp(te1 + " 23:59:59", tz=TZ).value

        eligible = []
        for tid, (s, et, pnl) in results.items():
            tr_mask = (et >= tr0_ns) & (et <= tr1_ns)
            tr_pnl = pnl[tr_mask]
            tr_stats = _stats(tr_pnl)
            if tr_stats["trades"] < MIN_TRAIN_TRADES or not np.isfinite(tr_stats["PF"]):
                continue
            te_mask = (et >= te0_ns) & (et <= te1_ns)
            te_pnl = pnl[te_mask]
            te_et = et[te_mask]
            te_stats = _stats(te_pnl)
            eligible.append((tid, s, tr_stats, te_stats, te_et, te_pnl))

        if not eligible:
            print(f"Fold {fi+1}: no eligible configs (min train trades {MIN_TRAIN_TRADES})")
            continue

        # Pick by train PF
        eligible.sort(key=lambda r: r[2]["PF"], reverse=True)
        best = eligible[0]
        tid, s, tr_stats, te_stats, te_et, te_pnl = best
        stitched_et.append(te_et)
        stitched_pnl.append(te_pnl)
        selected_per_fold.append({
            "fold": fi + 1,
            "train": f"{tr0} -> {tr1}",
            "test": f"{te0} -> {te1}",
            "trial_id": tid,
            **{f"p_{k}": v for k, v in s.items()},
            "tr_trades": tr_stats["trades"],
            "tr_win%": tr_stats["win%"],
            "tr_PF": tr_stats["PF"],
            "tr_gross_$": tr_stats["gross_$"],
            "te_trades": te_stats["trades"],
            "te_win%": te_stats["win%"],
            "te_PF": te_stats["PF"],
            "te_gross_$": te_stats["gross_$"],
        })
        fold_rows.append(selected_per_fold[-1])

    sel_df = pd.DataFrame(selected_per_fold)
    print("=== Walk-forward selections (top by train PF, applied to test) ===")
    show_cols = [
        "fold", "train", "test",
        "p_atr_len", "p_yellow_atr_mult",
        "p_green_atr_mult", "p_green_base", "p_green_decay",
        "p_red_intercept", "p_red_drift",
        "tr_trades", "tr_win%", "tr_PF", "tr_gross_$",
        "te_trades", "te_win%", "te_PF", "te_gross_$",
    ]
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(sel_df[show_cols].round(2).to_string(index=False))

    # Stitched
    if stitched_pnl:
        all_pnl = np.concatenate(stitched_pnl)
        s = _stats(all_pnl)
        print("\n=== STITCHED out-of-sample (concatenating all 4 test windows) ===")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k:9s}: {v:,.2f}")
            else:
                print(f"  {k:9s}: {v}")

    # ---------- baseline 1: Pine v4 defaults with pure_ratchet ----------
    print("\n=== Baseline 1: Pine v4 defaults (yellow_mode=pure_ratchet) ===")
    p_def = StrategyParams(yellow_mode="pure_ratchet", use_be=False, use_martingale=False,
                           base_qty=1, loss_qty=1)
    df_def = trades_to_df(run_backtest(bars, p_def))
    df_def["entry_time"] = pd.to_datetime(df_def["entry_time"]).dt.tz_convert(TZ)
    et_def = df_def["entry_time"].astype("int64").to_numpy()
    pnl_def = df_def["pnl_dollars"].to_numpy(dtype=np.float64)
    print(f"  Full sample:  {_stats(pnl_def)}")
    for fi, (_, _, te0, te1) in enumerate(FOLDS):
        m = (et_def >= pd.Timestamp(te0, tz=TZ).value) & (
            et_def <= pd.Timestamp(te1 + " 23:59:59", tz=TZ).value
        )
        s_te = _stats(pnl_def[m])
        print(f"  Fold {fi+1} test: {s_te}")

    # ---------- baseline 2: best full-sample pure_ratchet config ----------
    print("\n=== Baseline 2: best PF on FULL sample (overfit reference) ===")
    full_scores = []
    for tid, (s, et, pnl) in results.items():
        st = _stats(pnl)
        if st["trades"] < 800:
            continue
        if not np.isfinite(st["PF"]):
            continue
        full_scores.append((st["PF"], tid, s, et, pnl, st))
    full_scores.sort(reverse=True)
    if full_scores:
        pf, tid, s, et, pnl, st = full_scores[0]
        print(f"  trial {tid}: {s}")
        print(f"  full sample: {st}")
        for fi, (_, _, te0, te1) in enumerate(FOLDS):
            m = (et >= pd.Timestamp(te0, tz=TZ).value) & (
                et <= pd.Timestamp(te1 + " 23:59:59", tz=TZ).value
            )
            print(f"  Fold {fi+1} test: {_stats(pnl[m])}")

    sel_df.to_csv(OUT_DIR / "walk_forward_selections.csv", index=False)
    print(f"\nSaved -> {OUT_DIR / 'walk_forward_selections.csv'}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    main(n)

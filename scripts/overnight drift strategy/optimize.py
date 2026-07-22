"""
Random-search optimizer for the Overnight Drift yellow/green/red bands.

Design choices:
- IS  = 2020-12-01 -> 2023-12-31  (training)
- OOS = 2024-01-01 -> end of data (test)
- BE OFF, martingale OFF (qty=1) so we isolate band geometry, not sizing.
- atr_len locked single value for both yellow and green (avoids co-linear
  twin-window noise). Sampled jointly with the rest.
- Score: profit factor on IS with a >= MIN_TRADES floor to suppress trivial
  configs. Trials below the floor are dropped from the leaderboard.
- Output: full trial table (CSV) plus a top-20 IS/OOS comparison printed.

Parallelism: ProcessPoolExecutor with an initializer that loads bars once per
worker.

Usage:
    python "scripts/overnight drift strategy/optimize.py" 5000
"""

from __future__ import annotations

import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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

IS_END = pd.Timestamp("2024-01-01", tz="America/New_York")
MIN_IS_TRADES = 200  # require at least this many IS trades to be on the leaderboard

# Per-worker bars cache (set in _init)
_BARS: pd.DataFrame | None = None


def _init(bars: pd.DataFrame) -> None:
    global _BARS
    _BARS = bars


def sample_params(rng: random.Random) -> dict:
    """One random draw from the search space."""
    # ~25% of draws leave giveback OFF (yellow_giveback=0) so pure-geometry configs
    # stay in the leaderboard and the giveback-on configs have an apples-to-apples
    # baseline. The rest sweep the giveback knobs jointly with the band geometry.
    giveback = 0.0 if rng.random() < 0.25 else round(rng.uniform(0.1, 0.9), 2)
    return {
        "atr_len": rng.choice([7, 10, 12, 14, 16, 20, 25, 30]),
        "yellow_atr_mult": round(rng.uniform(0.75, 3.0), 2),
        "yellow_drift": round(rng.uniform(0.0, 3.0), 2),
        "yellow_mode": rng.choice(["drift_floor", "pure_ratchet", "raw_atr"]),
        "green_atr_mult": round(rng.uniform(0.5, 4.0), 2),
        "green_base": round(rng.uniform(20.0, 200.0), 1),
        "green_decay": round(rng.uniform(0.0, 3.0), 2),
        # red is NOT swept: it only enters via green (green_val = entry + (red_intercept
        # + green_base) + (red_drift - green_decay)*bars + green_atr*ATR), so red_intercept
        # is co-linear with green_base and red_drift with green_decay. Pinned to live values
        # in _to_strategy_params; green_base/green_decay carry the full target geometry.
        # Bearish giveback knobs (inert when yellow_giveback == 0)
        "yellow_giveback": giveback,
        "scale_by_body": rng.random() < 0.5,
        "max_giveback_atr": round(rng.uniform(0.25, 1.5), 2),
        "giveback_min_gap_atr": round(rng.uniform(0.0, 0.6), 2),
    }


def _to_strategy_params(s: dict) -> StrategyParams:
    return StrategyParams(
        yellow_atr_len=s["atr_len"],
        yellow_atr_mult=s["yellow_atr_mult"],
        yellow_drift=s["yellow_drift"],
        yellow_mode=s["yellow_mode"],
        green_atr_len=s["atr_len"],
        green_atr_mult=s["green_atr_mult"],
        green_base=s["green_base"],
        green_decay=s["green_decay"],
        red_intercept=s.get("red_intercept", 0.0),   # pinned to live (co-linear with green_base)
        red_drift=s.get("red_drift", 0.45),           # pinned to live (co-linear with green_decay)
        yellow_giveback=s.get("yellow_giveback", 0.0),
        scale_by_body=s.get("scale_by_body", True),
        max_giveback_atr=s.get("max_giveback_atr", 0.75),
        giveback_min_gap_atr=s.get("giveback_min_gap_atr", 0.0),
        use_be=False,
        use_martingale=False,
        base_qty=1,
        loss_qty=1,
    )


def _split_metrics(trades: pd.DataFrame) -> dict:
    """Trades pre-tagged with `period` in {'IS','OOS'}. Returns flat dict."""
    out = {}
    for period in ("IS", "OOS"):
        sub = trades[trades["period"] == period]
        if len(sub) == 0:
            out.update(
                {
                    f"{period}_trades": 0,
                    f"{period}_win%": np.nan,
                    f"{period}_avg_$": np.nan,
                    f"{period}_gross_$": 0.0,
                    f"{period}_PF": np.nan,
                }
            )
            continue
        wins = (sub["pnl_dollars"] > 0).sum()
        gw = sub.loc[sub["pnl_dollars"] > 0, "pnl_dollars"].sum()
        gl = abs(sub.loc[sub["pnl_dollars"] < 0, "pnl_dollars"].sum())
        out[f"{period}_trades"] = int(len(sub))
        out[f"{period}_win%"] = float(wins / len(sub) * 100)
        out[f"{period}_avg_$"] = float(sub["pnl_dollars"].mean())
        out[f"{period}_gross_$"] = float(sub["pnl_dollars"].sum())
        out[f"{period}_PF"] = float(gw / gl) if gl > 0 else float("inf")
    return out


def _trial(sample: dict) -> dict:
    params = _to_strategy_params(sample)
    trades_list = run_backtest(_BARS, params)
    df = trades_to_df(trades_list)
    if df.empty:
        return {**sample, **_split_metrics(df.assign(period="IS"))}
    df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_convert("America/New_York")
    df["period"] = np.where(df["entry_time"] < IS_END, "IS", "OOS")
    return {**sample, **_split_metrics(df)}


def main(n_trials: int) -> None:
    print(f"Loading bars...", flush=True)
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(f"  bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}", flush=True)

    # ---- live baseline (giveback OFF, current locked geometry) for comparison ----
    # Same marti-OFF / BE-OFF geometry-only basis as the sweep, so PFs are comparable.
    live = {
        "atr_len": 14, "yellow_atr_mult": 1.30, "yellow_drift": 0.0,
        "yellow_mode": "pure_ratchet", "green_atr_mult": 1.00, "green_base": 82.5,
        "green_decay": 1.50, "red_intercept": 0.0, "red_drift": 0.45,
        "yellow_giveback": 0.0, "scale_by_body": True, "max_giveback_atr": 0.75,
        "giveback_min_gap_atr": 0.0,
    }
    global _BARS
    _BARS = bars
    live_m = _trial(live)
    print(
        f"\nLIVE baseline (giveback OFF): "
        f"IS PF {live_m['IS_PF']:.3f} ({live_m['IS_trades']} td) | "
        f"OOS PF {live_m['OOS_PF']:.3f} ({live_m['OOS_trades']} td) | "
        f"OOS gross ${live_m['OOS_gross_$']:,.0f}",
        flush=True,
    )

    rng = random.Random(42)
    samples = [sample_params(rng) for _ in range(n_trials)]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Running {n_trials} trials on {workers} workers...", flush=True)

    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(bars,)) as ex:
        futs = [ex.submit(_trial, s) for s in samples]
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 100 == 0 or done == n_trials:
                rate = done / (time.time() - t0)
                eta = (n_trials - done) / rate if rate > 0 else 0
                print(f"  {done}/{n_trials}  ({rate:.1f}/s, eta {eta:.0f}s)", flush=True)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s\n", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "optimize_trials.csv", index=False)
    print(f"Wrote full trial table -> {OUT_DIR / 'optimize_trials.csv'}\n")

    eligible = df[df["IS_trades"] >= MIN_IS_TRADES].copy()
    print(f"Eligible trials (IS_trades >= {MIN_IS_TRADES}): {len(eligible)}/{len(df)}\n")

    # ---------------- top 20 by IS PF ----------------
    top = eligible.sort_values("IS_PF", ascending=False).head(20).reset_index(drop=True)
    cols = [
        "atr_len",
        "yellow_atr_mult",
        "yellow_drift",
        "yellow_mode",
        "green_atr_mult",
        "green_base",
        "green_decay",
        "yellow_giveback",
        "scale_by_body",
        "max_giveback_atr",
        "giveback_min_gap_atr",
        "IS_trades",
        "IS_win%",
        "IS_avg_$",
        "IS_PF",
        "IS_gross_$",
        "OOS_trades",
        "OOS_win%",
        "OOS_avg_$",
        "OOS_PF",
        "OOS_gross_$",
    ]
    print("=== Top 20 by IS profit factor ===")
    with pd.option_context("display.width", 260, "display.max_columns", None):
        print(top[cols].round(2).to_string())

    # ---------------- top 20 by OOS PF (sanity check on overfit) ----------------
    top_oos = eligible.sort_values("OOS_PF", ascending=False).head(20).reset_index(drop=True)
    print("\n=== Top 20 by OOS profit factor ===")
    with pd.option_context("display.width", 260, "display.max_columns", None):
        print(top_oos[cols].round(2).to_string())

    # ---------------- correlation IS PF vs OOS PF ----------------
    valid = eligible.dropna(subset=["IS_PF", "OOS_PF"])
    if len(valid) > 5:
        # cap PF at 5 to avoid inf-skewed correlation
        is_pf = valid["IS_PF"].clip(upper=5)
        oos_pf = valid["OOS_PF"].clip(upper=5)
        rho = float(np.corrcoef(is_pf, oos_pf)[0, 1])
        print(f"\nIS->OOS PF correlation (clipped at 5): {rho:.3f}")

    # ---------------- does giveback beat live? ----------------
    live_oos = live_m["OOS_PF"]
    live_is = live_m["IS_PF"]
    gb_on = eligible[eligible["yellow_giveback"] > 0]
    gb_off = eligible[eligible["yellow_giveback"] == 0]
    print("\n=== Giveback vs live baseline ===")
    print(f"live: IS PF {live_is:.3f} | OOS PF {live_oos:.3f}")
    for label, sub in (("giveback ON", gb_on), ("giveback OFF", gb_off)):
        if sub.empty:
            print(f"  {label:14s}: no eligible trials")
            continue
        beat_oos = sub[sub["OOS_PF"] > live_oos]
        beat_both = sub[(sub["OOS_PF"] > live_oos) & (sub["IS_PF"] > live_is)]
        print(
            f"  {label:14s}: {len(sub)} eligible | "
            f"{len(beat_oos)} beat live OOS PF | {len(beat_both)} beat live on BOTH IS & OOS"
        )

    # Best robust giveback config (rank by the weaker of IS/OOS PF — hardest to overfit)
    if not gb_on.empty:
        gb = gb_on.copy()
        gb["robust_PF"] = gb[["IS_PF", "OOS_PF"]].min(axis=1)
        best = gb.sort_values("robust_PF", ascending=False).head(10).reset_index(drop=True)
        print("\n=== Top 10 giveback-ON configs by min(IS_PF, OOS_PF) ===")
        with pd.option_context("display.width", 260, "display.max_columns", None):
            print(best[cols].round(2).to_string())

    print(f"\nFull table at: {OUT_DIR / 'optimize_trials.csv'}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(n)

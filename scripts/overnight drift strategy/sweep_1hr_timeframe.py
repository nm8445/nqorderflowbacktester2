"""
Overnight Drift on the 1-HOUR timeframe — FB-style breathing yellow + force-close-hour sweep.

Motivation
----------
Live OD runs on 20-min bars, pure_ratchet yellow (up only), force-close 08:00 ET.
This script asks two things:
  1. Does OD run BETTER on 60-min bars with the FB-style "breathing" yellow (giveback:
     ratchets UP toward close-k*ATR, but on a bar following a bearish candle RETREATS a
     body-scaled fraction of the gap — so yellow moves both up and down)?
  2. Does holding later than 08:00 (force-close at 09/10/11/.../16 ET) help or hurt?

Method (mirrors optimize.py so PFs are comparable)
--------------------------------------------------
- Geometry-only basis: martingale OFF, BE OFF, qty=1 — isolates band geometry, not sizing.
- IS  = 2020-12-01 -> 2023-12-31 ;  OOS = 2024-01-01 -> end.
- The bar to beat is the ACTUAL LIVE 20-MIN config (run on 20-min bars, geometry-only).
- Rank by robust min(IS_PF, OOS_PF) — the weaker half is hardest to overfit.

Three stages
------------
  A. Live 20-min baseline + naive live-geometry-on-1hr baseline (force=8, giveback off).
  B. FOCUSED force-close sweep: freeze live-equivalent 1hr geometry, vary force_hour only
     (giveback off AND on) — clean read on "hold longer".
  C. RANDOM SEARCH over all band params + force_hour + giveback knobs, giveback ON for ~85%
     of draws (the FB breathing yellow the user asked for), 15% off as baseline.

Usage:
    python "scripts/overnight drift strategy/sweep_1hr_timeframe.py" 5000
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
    load_1min_parquet,
    load_5min_pickles,
    run_backtest,
    trades_to_df,
)

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

IS_END = pd.Timestamp("2024-01-01", tz="America/New_York")
MIN_IS_TRADES = 200
FORCE_HOURS = [8, 9, 10, 11, 12, 13, 14, 15, 16]

# Per-worker bars cache
_BARS: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Bar building (generic frequency; mirrors overnight_drift_strategy helpers)
# ---------------------------------------------------------------------------


def _resample(bars: pd.DataFrame, freq: str, tz: str = "America/New_York") -> pd.DataFrame:
    if bars.index.tz is None:
        bars = bars.tz_localize("UTC")
    et = bars.tz_convert(tz)
    agg = et.resample(freq, origin="start_day", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return agg.dropna()


def build_series(freq: str) -> pd.DataFrame:
    df1m = load_1min_parquet(PARQUET_PATH)
    cutoff = df1m.index.max()
    df5m_extra = load_5min_pickles(PICKLE_FOLDER, after_date=cutoff.tz_convert(None).normalize())
    a = _resample(df1m, freq)
    if not df5m_extra.empty:
        b = _resample(df5m_extra, freq)
        out = pd.concat([a, b])
        out = out[~out.index.duplicated(keep="last")].sort_index()
    else:
        out = a
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _split_metrics(trades: pd.DataFrame) -> dict:
    out = {}
    for period in ("IS", "OOS", "FULL"):
        if period == "FULL":
            sub = trades
        else:
            sub = trades[trades["period"] == period]
        if len(sub) == 0:
            out.update({f"{period}_trades": 0, f"{period}_win%": np.nan,
                        f"{period}_gross_$": 0.0, f"{period}_PF": np.nan, f"{period}_maxDD_$": 0.0})
            continue
        wins = (sub["pnl_dollars"] > 0).sum()
        gw = sub.loc[sub["pnl_dollars"] > 0, "pnl_dollars"].sum()
        gl = abs(sub.loc[sub["pnl_dollars"] < 0, "pnl_dollars"].sum())
        eq = sub.sort_values("entry_time")["pnl_dollars"].cumsum()
        out[f"{period}_trades"] = int(len(sub))
        out[f"{period}_win%"] = float(wins / len(sub) * 100)
        out[f"{period}_gross_$"] = float(sub["pnl_dollars"].sum())
        out[f"{period}_PF"] = float(gw / gl) if gl > 0 else float("inf")
        out[f"{period}_maxDD_$"] = float((eq - eq.cummax()).min())
    return out


def _run(sample: dict) -> dict:
    """Run one config (geometry-only) and return metrics tagged IS/OOS/FULL."""
    p = StrategyParams(
        forced_hour=sample.get("force_hour", 8),
        yellow_atr_len=sample["atr_len"],
        yellow_atr_mult=sample["yellow_atr_mult"],
        yellow_drift=sample.get("yellow_drift", 0.0),
        yellow_mode=sample["yellow_mode"],
        green_atr_len=sample["atr_len"],
        green_atr_mult=sample["green_atr_mult"],
        green_base=sample["green_base"],
        green_decay=sample["green_decay"],
        red_intercept=0.0,
        red_drift=0.45,
        yellow_giveback=sample.get("yellow_giveback", 0.0),
        scale_by_body=sample.get("scale_by_body", True),
        max_giveback_atr=sample.get("max_giveback_atr", 0.75),
        giveback_min_gap_atr=sample.get("giveback_min_gap_atr", 0.0),
        use_be=False,
        use_martingale=False,
        base_qty=1,
        loss_qty=1,
    )
    df = trades_to_df(run_backtest(_BARS, p))
    if df.empty:
        return {**sample, **_split_metrics(df.assign(period="IS", entry_time=pd.NaT))}
    df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_convert("America/New_York")
    df["period"] = np.where(df["entry_time"] < IS_END, "IS", "OOS")
    return {**sample, **_split_metrics(df)}


def _init(bars: pd.DataFrame) -> None:
    global _BARS
    _BARS = bars


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def sample_params(rng: random.Random) -> dict:
    # FB-style breathing yellow is the focus: giveback ON for ~85% of draws.
    giveback = 0.0 if rng.random() < 0.15 else round(rng.uniform(0.1, 0.9), 2)
    mode = rng.choice(["pure_ratchet", "drift_floor"])
    return {
        "force_hour": rng.choice(FORCE_HOURS),
        "atr_len": rng.choice([7, 10, 12, 14, 16, 20, 25, 30]),
        "yellow_atr_mult": round(rng.uniform(0.75, 3.0), 2),
        "yellow_drift": round(rng.uniform(0.0, 3.0), 2) if mode == "drift_floor" else 0.0,
        "yellow_mode": mode,
        "green_atr_mult": round(rng.uniform(0.5, 4.0), 2),
        "green_base": round(rng.uniform(20.0, 200.0), 1),
        "green_decay": round(rng.uniform(0.0, 3.0), 2),
        "yellow_giveback": giveback,
        "scale_by_body": rng.random() < 0.5,
        "max_giveback_atr": round(rng.uniform(0.25, 1.5), 2),
        "giveback_min_gap_atr": round(rng.uniform(0.0, 0.6), 2),
    }


LIVE_GEOM = {
    "atr_len": 14, "yellow_atr_mult": 1.30, "yellow_drift": 0.0, "yellow_mode": "pure_ratchet",
    "green_atr_mult": 1.00, "green_base": 82.5, "green_decay": 1.50,
    "yellow_giveback": 0.0, "scale_by_body": True, "max_giveback_atr": 0.75, "giveback_min_gap_atr": 0.0,
}


def _fmt(m: dict, label: str) -> str:
    return (f"{label:42s} IS {m['IS_PF']:.3f} ({m['IS_trades']:>3d}) | "
            f"OOS {m['OOS_PF']:.3f} ({m['OOS_trades']:>3d}) | "
            f"FULL {m['FULL_PF']:.3f} net ${m['FULL_gross_$']:>9,.0f} DD ${m['FULL_maxDD_$']:>9,.0f}")


def main(n_trials: int) -> None:
    global _BARS

    # ---- Stage A: baselines ----
    print("Building 20-min bars (live baseline)...", flush=True)
    bars20 = build_series("20min")
    _BARS = bars20
    live20 = _run({**LIVE_GEOM, "force_hour": 8})
    print(_fmt(live20, "LIVE 20-min (force=8, giveback off)"), flush=True)
    live_is, live_oos, live_robust = live20["IS_PF"], live20["OOS_PF"], min(live20["IS_PF"], live20["OOS_PF"])

    print("\nBuilding 60-min bars...", flush=True)
    bars60 = build_series("60min")
    print(f"  60-min bars: {len(bars60):,}  range: {bars60.index.min()} -> {bars60.index.max()}", flush=True)
    _BARS = bars60
    naive1h = _run({**LIVE_GEOM, "force_hour": 8})
    print(_fmt(naive1h, "1hr naive (live geom, force=8, gb off)"), flush=True)

    # ---- Stage B: focused force-close-hour sweep at live-equivalent 1hr geometry ----
    print("\n=== Stage B: force-close-hour sweep (1hr, live geometry) ===", flush=True)
    print("  giveback OFF (pure_ratchet up-only yellow):", flush=True)
    rowsB = []
    for fh in FORCE_HOURS:
        m = _run({**LIVE_GEOM, "force_hour": fh})
        rowsB.append({"giveback": "off", "force_hour": fh, **{k: m[k] for k in
                     ("IS_PF", "OOS_PF", "FULL_PF", "FULL_gross_$", "FULL_maxDD_$", "FULL_win%", "FULL_trades")}})
        print("   " + _fmt(m, f"force={fh:>2d}:00"), flush=True)
    # a representative FB breathing-yellow geometry for the force sweep
    gb_geom = {**LIVE_GEOM, "yellow_giveback": 0.5, "scale_by_body": True,
               "max_giveback_atr": 0.75, "giveback_min_gap_atr": 0.3}
    print("  giveback ON (FB breathing yellow, gb=0.5):", flush=True)
    for fh in FORCE_HOURS:
        m = _run({**gb_geom, "force_hour": fh})
        rowsB.append({"giveback": "on", "force_hour": fh, **{k: m[k] for k in
                     ("IS_PF", "OOS_PF", "FULL_PF", "FULL_gross_$", "FULL_maxDD_$", "FULL_win%", "FULL_trades")}})
        print("   " + _fmt(m, f"force={fh:>2d}:00"), flush=True)
    pd.DataFrame(rowsB).to_csv(OUT_DIR / "sweep_1hr_forceclose.csv", index=False)

    # ---- Stage C: random search over everything ----
    rng = random.Random(42)
    samples = [sample_params(rng) for _ in range(n_trials)]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"\n=== Stage C: random search, {n_trials} trials on {workers} workers ===", flush=True)
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(bars60,)) as ex:
        futs = [ex.submit(_run, s) for s in samples]
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 250 == 0 or done == n_trials:
                rate = done / (time.time() - t0)
                print(f"  {done}/{n_trials}  ({rate:.1f}/s, eta {(n_trials-done)/rate:.0f}s)", flush=True)
    print(f"Done in {time.time()-t0:.0f}s\n", flush=True)

    df = pd.DataFrame(results)
    df["robust_PF"] = df[["IS_PF", "OOS_PF"]].min(axis=1)
    df.to_csv(OUT_DIR / "sweep_1hr_trials.csv", index=False)

    elig = df[df["IS_trades"] >= MIN_IS_TRADES].copy()
    print(f"Live 20-min bar-to-beat: IS {live_is:.3f} | OOS {live_oos:.3f} | robust {live_robust:.3f}")
    print(f"Eligible 1hr trials (IS_trades >= {MIN_IS_TRADES}): {len(elig)}/{len(df)}")
    beat = elig[(elig["IS_PF"] > live_is) & (elig["OOS_PF"] > live_oos)]
    beat_robust = elig[elig["robust_PF"] > live_robust]
    print(f"1hr configs beating live on BOTH IS & OOS: {len(beat)}")
    print(f"1hr configs beating live robust min(IS,OOS): {len(beat_robust)}")

    cols = ["force_hour", "atr_len", "yellow_atr_mult", "yellow_mode", "yellow_drift",
            "green_atr_mult", "green_base", "green_decay",
            "yellow_giveback", "scale_by_body", "max_giveback_atr", "giveback_min_gap_atr",
            "IS_PF", "OOS_PF", "FULL_PF", "robust_PF", "FULL_gross_$", "FULL_maxDD_$", "FULL_trades"]
    top = elig.sort_values("robust_PF", ascending=False).head(20)
    print("\n=== Top 20 1hr configs by robust min(IS,OOS) PF ===")
    with pd.option_context("display.width", 260, "display.max_columns", None):
        print(top[cols].round(3).to_string(index=False))

    # force_hour marginal effect across the whole eligible search
    print("\n=== Eligible-trial PF by force_hour (median robust_PF, count) ===")
    g = elig.groupby("force_hour")["robust_PF"].agg(["median", "max", "count"]).round(3)
    print(g.to_string())

    print(f"\nFull table -> {OUT_DIR / 'sweep_1hr_trials.csv'}")
    print(f"Force-close sweep -> {OUT_DIR / 'sweep_1hr_forceclose.csv'}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    main(n)

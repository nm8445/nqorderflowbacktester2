"""POC regime filter study — extends the original 2/3-day monotonic POC trends
to include 4 reversal patterns (V-trough and inverted-V peak).

For each trading day D:
  - RTH POC via snapshot subtraction:
    POC(D) = argmax( profile_at_16:59 - profile_at_9:30 ) on volume-per-level
  - open_930 = NQ 1-min close at 9:30 ET
  - close_1659 = NQ 1-min close at 16:59 ET
  - day_ret_pts = close_1659 - open_930 (NQ points, signed)

Original scenarios from existing README:
  - 2-day monotonic up:   POC(D-1) > POC(D-2)
  - 2-day monotonic down: POC(D-1) < POC(D-2)
  - 3-day monotonic up:   POC(D-1) > POC(D-2) > POC(D-3)
  - 3-day monotonic down: POC(D-1) < POC(D-2) < POC(D-3)

NEW scenarios in this run (3-day reversal patterns):
  A. Inverted-V peak: POC(D-1) < POC(D-2) > POC(D-3)
     A1. open_930(D) > POC(D-1)
     A2. open_930(D) < POC(D-1)
  B. V trough:        POC(D-1) > POC(D-2) < POC(D-3)
     B1. open_930(D) < POC(D-1)
     B2. open_930(D) > POC(D-1)

Data sources:
  - Profiles: D:/trading_pythonbacktest_data/cache/profiles/{date}_refresh_minutes=1.pkl
  - 1-min NQ bars: D:/trading_pythonbacktest_data/markettick_1min_bars.parquet

Output:
  - per-day parquet at .../poc regime results/scripts/poc_per_day.parquet
  - printed summary tables (replicates the original README scenarios + new ones)
"""

from __future__ import annotations

import datetime as dt
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROFILES_DIR = Path("D:/trading_pythonbacktest_data/cache/profiles")
NQ_1MIN      = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT_DIR      = Path(__file__).parent
PER_DAY_OUT  = OUT_DIR / "poc_per_day.parquet"


def list_dates() -> list[dt.date]:
    """Dates we have in both profiles cache AND 1-min NQ parquet."""
    dates = []
    for p in sorted(PROFILES_DIR.glob("*_refresh_minutes=1.pkl")):
        try:
            d = dt.date.fromisoformat(p.name.split("_")[0])
            dates.append(d)
        except: pass
    return dates


def find_nearest_snap(snaps: list[dict], target_utc: pd.Timestamp) -> dict | None:
    """Snapshot whose refresh_time is closest to target_utc, within 5 min."""
    best = None; best_diff = pd.Timedelta(minutes=10)
    for s in snaps:
        d = abs(s["refresh_time"] - target_utc)
        if d < best_diff:
            best_diff = d; best = s
    return best


def rth_poc(date: dt.date) -> float | None:
    """Compute RTH POC for `date` via snapshot subtraction (9:30 ET vs 16:59 ET)."""
    p = PROFILES_DIR / f"{date.isoformat()}_refresh_minutes=1.pkl"
    if not p.exists(): return None
    with open(p, "rb") as f:
        snaps = pickle.load(f)

    open_et = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                           hour=9, minute=30, tz="America/New_York")
    close_et = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                            hour=16, minute=59, tz="America/New_York")
    open_utc = open_et.tz_convert("UTC")
    close_utc = close_et.tz_convert("UTC")
    snap_open = find_nearest_snap(snaps, open_utc)
    snap_close = find_nearest_snap(snaps, close_utc)
    if snap_open is None or snap_close is None: return None

    levels_open = snap_open["profile"].levels
    levels_close = snap_close["profile"].levels
    all_prices = set(levels_open) | set(levels_close)
    diff = {}
    for K in all_prices:
        ov = (levels_open[K].buy_vol + levels_open[K].sell_vol) if K in levels_open else 0
        cv = (levels_close[K].buy_vol + levels_close[K].sell_vol) if K in levels_close else 0
        diff[K] = cv - ov
    if not diff or max(diff.values()) <= 0: return None
    return float(max(diff, key=diff.get))


def load_nq() -> pd.Series:
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    return df["close"].sort_index()


def nq_at(nq: pd.Series, date: dt.date, h: int, m: int) -> float:
    target = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                          hour=h, minute=m, tz="America/New_York")
    win = nq.loc[target - pd.Timedelta(minutes=2): target + pd.Timedelta(minutes=1)]
    return float(win.iloc[-1]) if not win.empty else np.nan


def build_per_day_table() -> pd.DataFrame:
    print("loading NQ 1-min bars...")
    nq = load_nq()
    print(f"  range: {nq.index.min()} -> {nq.index.max()}")

    dates = list_dates()
    print(f"profiles available: {len(dates)} dates")

    rows = []
    t0 = time.time()
    for i, d in enumerate(dates):
        # Skip weekends explicitly (cache may include them)
        if d.weekday() >= 5:
            continue
        try:
            poc = rth_poc(d)
        except Exception as e:
            poc = None
        if poc is None: continue

        op  = nq_at(nq, d, 9, 30)
        cl  = nq_at(nq, d, 16, 59)
        if not (np.isfinite(op) and np.isfinite(cl)): continue

        rows.append({"date": d, "poc": poc,
                     "open_930": op, "close_1659": cl,
                     "day_ret_pts": cl - op})
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(dates)}  elapsed={time.time()-t0:.0f}s  rows={len(rows)}")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["poc_d1"] = df["poc"].shift(1)
    df["poc_d2"] = df["poc"].shift(2)
    df["poc_d3"] = df["poc"].shift(3)
    return df


# ------------------------------ Reporting ------------------------------

def stats(label: str, sub: pd.DataFrame, base_n: int):
    n = len(sub)
    if n == 0:
        print(f"  {label:<60}  n=0"); return
    p = (sub["day_ret_pts"] > 0).mean()
    m = sub["day_ret_pts"].mean()
    md = sub["day_ret_pts"].median()
    pct = n / base_n if base_n else 0.0
    print(f"  {label:<60}  n={n:>4} ({pct:>5.1%})  P(>0)={p:.2%}  "
          f"mean={m:+6.2f} pts  median={md:+6.2f} pts")


def main():
    df = build_per_day_table()
    df.to_parquet(PER_DAY_OUT, compression="zstd", index=False)
    print(f"\nwrote {PER_DAY_OUT}  ({len(df)} rows)")

    valid = df.dropna(subset=["poc_d1","poc_d2","poc_d3"]).copy()
    n = len(valid)
    print(f"\nvalid sample (with 3 prior POCs): {n}")
    print()

    print("=== Baseline ===")
    stats("BASE  (all valid days)", valid, n)
    print(f"  P(positive day) baseline: {(valid['day_ret_pts'] > 0).mean():.2%}")
    print(f"  Mean day return baseline: {valid['day_ret_pts'].mean():+.2f} pts")
    print()

    print("=== Original scenarios (sanity replication) ===")
    rose2 = valid[valid["poc_d1"] > valid["poc_d2"]]
    fell2 = valid[valid["poc_d1"] < valid["poc_d2"]]
    print("\n--- 2-day POC trend ---")
    stats("rose2  (POC(D-1) > POC(D-2))", rose2, n)
    stats("  + open > POC(D-1)", rose2[rose2["open_930"] > rose2["poc_d1"]], n)
    stats("  + open <= POC(D-1)", rose2[rose2["open_930"] <= rose2["poc_d1"]], n)
    stats("fell2  (POC(D-1) < POC(D-2))", fell2, n)
    stats("  + open < POC(D-1)", fell2[fell2["open_930"] < fell2["poc_d1"]], n)
    stats("  + open >= POC(D-1)", fell2[fell2["open_930"] >= fell2["poc_d1"]], n)

    rose3 = valid[(valid["poc_d1"] > valid["poc_d2"]) & (valid["poc_d2"] > valid["poc_d3"])]
    fell3 = valid[(valid["poc_d1"] < valid["poc_d2"]) & (valid["poc_d2"] < valid["poc_d3"])]
    print("\n--- 3-day POC trend (monotonic) ---")
    stats("rose3  (D-1 > D-2 > D-3)", rose3, n)
    stats("  + open > POC(D-1)", rose3[rose3["open_930"] > rose3["poc_d1"]], n)
    stats("  + open <= POC(D-1)", rose3[rose3["open_930"] <= rose3["poc_d1"]], n)
    stats("fell3  (D-1 < D-2 < D-3)", fell3, n)
    stats("  + open < POC(D-1)", fell3[fell3["open_930"] < fell3["poc_d1"]], n)
    stats("  + open >= POC(D-1)", fell3[fell3["open_930"] >= fell3["poc_d1"]], n)

    print("\n=== NEW: 3-day reversal patterns ===")

    print("\n--- A. Inverted-V peak: POC(D-1) < POC(D-2) > POC(D-3) ---")
    invV = valid[(valid["poc_d1"] < valid["poc_d2"]) & (valid["poc_d2"] > valid["poc_d3"])]
    stats("invV  (D-1 < D-2 > D-3)", invV, n)
    stats("  A1. + open > POC(D-1)", invV[invV["open_930"] > invV["poc_d1"]], n)
    stats("  A2. + open < POC(D-1)", invV[invV["open_930"] < invV["poc_d1"]], n)

    print("\n--- B. V trough: POC(D-1) > POC(D-2) < POC(D-3) ---")
    vT = valid[(valid["poc_d1"] > valid["poc_d2"]) & (valid["poc_d2"] < valid["poc_d3"])]
    stats("vTrough  (D-1 > D-2 < D-3)", vT, n)
    stats("  B1. + open < POC(D-1)", vT[vT["open_930"] < vT["poc_d1"]], n)
    stats("  B2. + open > POC(D-1)", vT[vT["open_930"] > vT["poc_d1"]], n)


if __name__ == "__main__":
    sys.exit(main())

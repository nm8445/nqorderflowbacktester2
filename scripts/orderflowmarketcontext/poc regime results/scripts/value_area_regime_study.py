"""Value-area regime study (extends POC regime study).

Computes RTH POC, VAH, VAL per day from the developing profile cache
(snapshot subtraction). Then tests these cohorts:

BEARISH SETUP — VAH(D-1) < VAL(D-2)  (D-1 VA strictly below D-2 VA)
  Sub-cohort A: D opens in [VAL(D-1), POC(D-1)]   — lower half of D-1 VA
  Sub-cohort B: D opens in [POC(D-1), VAH(D-1)]   — upper half of D-1 VA

BULLISH SETUP — VAL(D-1) > VAH(D-2)  (D-1 VA strictly above D-2 VA)
  Sub-cohort A: D opens above VAH(D-1)
  Sub-cohort B: D opens in [POC(D-1), VAH(D-1)]   — upper half of D-1 VA
  Sub-cohort C: D opens in [VAL(D-1), POC(D-1)]   — lower half of D-1 VA

Day return = NQ close at 17:00 ET − NQ open at 9:30 ET.
Reports n, P(close > open), P(close < open), mean ret, t-stat, p-value.
"""

from __future__ import annotations

import datetime as dt
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

PROFILES_DIR = Path("D:/trading_pythonbacktest_data/cache/profiles")
NQ_1MIN      = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT_DIR      = Path(__file__).parent
PER_DAY_OUT  = OUT_DIR / "value_area_per_day.parquet"

VA_PCT = 0.70


def list_dates() -> list[dt.date]:
    out = []
    for p in sorted(PROFILES_DIR.glob("*_refresh_minutes=1.pkl")):
        try:
            d = dt.date.fromisoformat(p.name.split("_")[0])
            out.append(d)
        except: pass
    return out


def find_nearest_snap(snaps, target_utc):
    best, bd = None, pd.Timedelta(minutes=10)
    for s in snaps:
        d = abs(s["refresh_time"] - target_utc)
        if d < bd: bd, best = d, s
    return best


def rth_profile(date: dt.date) -> dict[float, int] | None:
    """Returns {price_level: total_volume} for the RTH session via subtraction."""
    p = PROFILES_DIR / f"{date.isoformat()}_refresh_minutes=1.pkl"
    if not p.exists(): return None
    with open(p, "rb") as f: snaps = pickle.load(f)
    open_t = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                          hour=9, minute=30, tz="America/New_York").tz_convert("UTC")
    close_t = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                           hour=16, minute=59, tz="America/New_York").tz_convert("UTC")
    s_open = find_nearest_snap(snaps, open_t)
    s_close = find_nearest_snap(snaps, close_t)
    if s_open is None or s_close is None: return None
    o_levels = s_open["profile"].levels
    c_levels = s_close["profile"].levels
    diff = {}
    for K in set(o_levels) | set(c_levels):
        ov = (o_levels[K].buy_vol + o_levels[K].sell_vol) if K in o_levels else 0
        cv = (c_levels[K].buy_vol + c_levels[K].sell_vol) if K in c_levels else 0
        v = cv - ov
        if v > 0:
            diff[K] = v
    return diff if diff else None


def compute_poc_va(profile: dict[float, int], pct: float = VA_PCT
                   ) -> tuple[float, float, float] | None:
    """Returns (POC, VAH, VAL). Standard "walk outward from POC" algorithm."""
    if not profile: return None
    sorted_levels = sorted(profile.keys())
    poc_price = max(profile, key=profile.get)
    total_vol = sum(profile.values())
    target = pct * total_vol

    # idx-based walk
    idx_of = {p: i for i, p in enumerate(sorted_levels)}
    poc_idx = idx_of[poc_price]
    cum = profile[poc_price]
    lo_idx = poc_idx; hi_idx = poc_idx
    while cum < target:
        next_above = sorted_levels[hi_idx + 1] if hi_idx + 1 < len(sorted_levels) else None
        next_below = sorted_levels[lo_idx - 1] if lo_idx - 1 >= 0 else None
        if next_above is None and next_below is None: break
        v_above = profile.get(next_above, 0) if next_above is not None else -1
        v_below = profile.get(next_below, 0) if next_below is not None else -1
        if v_above >= v_below:
            hi_idx += 1; cum += profile[sorted_levels[hi_idx]]
        else:
            lo_idx -= 1; cum += profile[sorted_levels[lo_idx]]
    return poc_price, sorted_levels[hi_idx], sorted_levels[lo_idx]


def load_nq_et() -> pd.Series:
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    return df["close"].sort_index()


def nq_at(nq, date, h, m):
    target = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                          hour=h, minute=m, tz="America/New_York")
    win = nq.loc[target - pd.Timedelta(minutes=2):target + pd.Timedelta(minutes=1)]
    return float(win.iloc[-1]) if not win.empty else np.nan


def build():
    print("loading NQ bars...")
    nq = load_nq_et()
    print(f"  range: {nq.index.min()} -> {nq.index.max()}")

    dates = list_dates()
    print(f"profile files: {len(dates)}")
    rows = []
    t0 = time.time()
    for i, d in enumerate(dates):
        if d.weekday() >= 5: continue
        prof = rth_profile(d)
        if prof is None: continue
        pv = compute_poc_va(prof)
        if pv is None: continue
        poc, vah, val = pv
        op = nq_at(nq, d, 9, 30); cl = nq_at(nq, d, 17, 0)
        if not (np.isfinite(op) and np.isfinite(cl)): continue
        rows.append({"date": d, "poc": poc, "vah": vah, "val": val,
                     "open_930": op, "close_1700": cl,
                     "day_ret_pts": cl - op})
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(dates)}  elapsed={time.time()-t0:.0f}s  rows={len(rows)}")
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    for c in ["poc", "vah", "val"]:
        df[f"{c}_d1"] = df[c].shift(1)
        df[f"{c}_d2"] = df[c].shift(2)
    df.to_parquet(PER_DAY_OUT, compression="zstd", index=False)
    print(f"\nwrote {PER_DAY_OUT}  ({len(df)} rows)")
    return df


def stats(label: str, sub: pd.DataFrame, base_n: int):
    n = len(sub)
    if n < 2:
        print(f"  {label:<70}  n={n:>4}"); return
    rets = sub["day_ret_pts"].dropna().values
    n = len(rets)
    p_up = float((rets > 0).mean())
    p_dn = float((rets < 0).mean())
    m = float(rets.mean()); md = float(np.median(rets))
    t, p = ss.ttest_1samp(rets, 0)
    sig = "  ***" if p<0.001 else ("   **" if p<0.01 else ("    *" if p<0.05 else "     "))
    pct_of_base = n / base_n if base_n else 0.0
    print(f"  {label:<70}  n={n:>4} ({pct_of_base:>5.1%})  "
          f"P(up)={p_up:.1%}  P(down)={p_dn:.1%}  mean={m:+6.2f}  med={md:+6.2f}  "
          f"t={t:+5.2f}  p={p:.4f}{sig}")


def main():
    if PER_DAY_OUT.exists():
        print(f"loading cached per-day parquet from {PER_DAY_OUT}...")
        df = pd.read_parquet(PER_DAY_OUT)
    else:
        df = build()

    valid = df.dropna(subset=["poc_d1","vah_d1","val_d1",
                              "poc_d2","vah_d2","val_d2"]).copy()
    n_total = len(valid)
    print(f"\nvalid sample (with 2 prior VAs): {n_total}")
    print(f"date range: {valid['date'].min()} -> {valid['date'].max()}")
    print()

    print("Significance markers: * p<0.05, ** p<0.01, *** p<0.001")
    print(f"{'cohort':<70}  {'n':>4} ({'%base':>5})  {'P(up)':<9}  {'P(down)':<11}  {'mean':>7}  {'median':>7}  {'t':>5}  {'p':>5}")
    print()

    print("=" * 130)
    print("BASELINE")
    print("=" * 130)
    stats("ALL valid days", valid, n_total)

    # ------------------------------ BEARISH SETUP ------------------------------
    bear = valid[valid["vah_d1"] < valid["val_d2"]]  # D-1 VA strictly below D-2 VA
    print()
    print("=" * 130)
    print(f"BEARISH SETUP — VAH(D-1) < VAL(D-2)   (n={len(bear)}, {len(bear)/n_total:.1%} of valid days)")
    print("=" * 130)
    stats("ALL bearish-setup days", bear, n_total)

    # Sub-cohorts based on D open relative to D-1 VA
    op = bear["open_930"]
    sub_lower = bear[(op >= bear["val_d1"]) & (op <= bear["poc_d1"])]
    sub_upper = bear[(op >= bear["poc_d1"]) & (op <= bear["vah_d1"])]
    sub_below_va = bear[op < bear["val_d1"]]   # bonus
    sub_above_va = bear[op > bear["vah_d1"]]   # bonus
    stats("  + open in [VAL(D-1), POC(D-1)] (lower half of D-1 VA)", sub_lower, n_total)
    stats("  + open in [POC(D-1), VAH(D-1)] (upper half of D-1 VA)", sub_upper, n_total)
    stats("  + open below VAL(D-1)  (bonus — gap-down through D-1 VA)", sub_below_va, n_total)
    stats("  + open above VAH(D-1)  (bonus — gap-up through D-1 VA)", sub_above_va, n_total)

    # ------------------------------ BULLISH SETUP ------------------------------
    bull = valid[valid["val_d1"] > valid["vah_d2"]]  # D-1 VA strictly above D-2 VA
    print()
    print("=" * 130)
    print(f"BULLISH SETUP — VAL(D-1) > VAH(D-2)   (n={len(bull)}, {len(bull)/n_total:.1%} of valid days)")
    print("=" * 130)
    stats("ALL bullish-setup days", bull, n_total)

    op2 = bull["open_930"]
    sub_above2 = bull[op2 > bull["vah_d1"]]
    sub_upper2 = bull[(op2 >= bull["poc_d1"]) & (op2 <= bull["vah_d1"])]
    sub_lower2 = bull[(op2 >= bull["val_d1"]) & (op2 <= bull["poc_d1"])]
    sub_below2 = bull[op2 < bull["val_d1"]]   # bonus
    stats("  + open above VAH(D-1)", sub_above2, n_total)
    stats("  + open in [POC(D-1), VAH(D-1)] (upper half of D-1 VA)", sub_upper2, n_total)
    stats("  + open in [VAL(D-1), POC(D-1)] (lower half of D-1 VA)", sub_lower2, n_total)
    stats("  + open below VAL(D-1)  (bonus — gap-down through D-1 VA)", sub_below2, n_total)


if __name__ == "__main__":
    sys.exit(main())

"""Value-area setup × overnight-range break study.

For each trading day D:
  - Setup classification (same as value_area_regime_study.py):
      * BEARISH setup: VAH(D-1) < VAL(D-2)  (D-1 VA strictly below D-2 VA)
      * BULLISH setup: VAL(D-1) > VAH(D-2)  (D-1 VA strictly above D-2 VA)
  - Overnight range = (high, low) of NQ 1-min bars from 18:00 ET on D-1 to
    09:30 ET on D. Range "locks" at 9:30 (static for rest of session).
  - Trigger detection: walk RTH 5-min closes from 9:30 to 17:00 ET.
      * long_break  = 3 consecutive 5-min closes ABOVE high_range
      * short_break = 3 consecutive 5-min closes BELOW low_range
      * Counters reset on any close back inside [low, high].
      * First-trigger-only: classify by whichever fires first.
  - Outcome: P(close > open) — close at 17:00 ET vs open at 9:30 ET.

Cohorts reported:
  * Baseline (all valid days)
  * Bearish setup baseline + long_break / short_break / no_trigger sub-cohorts
  * Bullish setup baseline + long_break / short_break / no_trigger sub-cohorts
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

NQ_1MIN  = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
VA_PARQ  = Path(__file__).parent / "value_area_per_day.parquet"
OUT_DIR  = Path(__file__).parent
PER_DAY_OUT = OUT_DIR / "overnight_range_break_per_day.parquet"


def load_nq_1m() -> pd.DataFrame:
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    return df.sort_index()


def overnight_range(nq: pd.DataFrame, date: dt.date) -> tuple[float, float] | None:
    """High/low between 18:00 ET on D-1 and 09:30 ET on D using 1-min bars."""
    prev = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                        tz="America/New_York") - pd.Timedelta(days=1)
    start = prev.replace(hour=18, minute=0, second=0)
    end = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                       hour=9, minute=30, tz="America/New_York")
    seg = nq.loc[start:end]
    if seg.empty: return None
    return float(seg["high"].max()), float(seg["low"].min())


def detect_trigger(rth5: pd.Series, high: float, low: float
                   ) -> tuple[str, pd.Timestamp | None]:
    """Walk 5-min closes from 9:30 to 17:00. First trigger wins.
    Returns (trigger_label, time_of_trigger_or_None)."""
    above = 0; below = 0
    for t, c in rth5.items():
        if c > high:
            above += 1; below = 0
            if above >= 3: return "long_break", t
        elif c < low:
            below += 1; above = 0
            if below >= 3: return "short_break", t
        else:
            above = 0; below = 0
    return "no_trigger", None


def main():
    print("loading NQ 1-min bars...")
    nq = load_nq_1m()
    print(f"  range: {nq.index.min()} -> {nq.index.max()}")

    # Build 5-min RTH closes (resample 1-min)
    rth_mask = (nq.index.time >= dt.time(9, 30)) & (nq.index.time <= dt.time(17, 0))
    rth_1m = nq[rth_mask]
    rth_5m = rth_1m["close"].resample("5min", origin="epoch").last().dropna()
    rth_5m = rth_5m[(rth_5m.index.time >= dt.time(9, 30)) &
                    (rth_5m.index.time <= dt.time(17, 0))]
    rth_5m_by_date = {d: g for d, g in rth_5m.groupby(rth_5m.index.date)}
    print(f"  rth 5-min days: {len(rth_5m_by_date)}")

    va = pd.read_parquet(VA_PARQ)
    va["date"] = pd.to_datetime(va["date"]).dt.date
    print(f"  VA parquet rows: {len(va)}")

    rows = []
    t0 = time.time()
    for i, r in enumerate(va.itertuples(index=False), 1):
        d = r.date
        rng = overnight_range(nq, d)
        if rng is None: continue
        high, low = rng
        bars = rth_5m_by_date.get(d)
        if bars is None or len(bars) < 5: continue
        trigger, t_trig = detect_trigger(bars, high, low)
        rows.append({
            "date": d,
            "open_930": r.open_930,
            "close_1700": r.close_1700,
            "day_ret_pts": r.close_1700 - r.open_930,
            "overnight_high": high,
            "overnight_low": low,
            "trigger": trigger,
            "trigger_time": t_trig.strftime("%H:%M") if t_trig else None,
            "vah_d1": r.vah_d1, "val_d1": r.val_d1, "poc_d1": r.poc_d1,
            "vah_d2": r.vah_d2, "val_d2": r.val_d2, "poc_d2": r.poc_d2,
        })
        if i % 200 == 0:
            print(f"  {i}/{len(va)}  elapsed={time.time()-t0:.0f}s  rows={len(rows)}")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["vah_d1","val_d1","vah_d2","val_d2"])
    df.to_parquet(PER_DAY_OUT, compression="zstd", index=False)
    print(f"\nwrote {PER_DAY_OUT}  ({len(df)} rows)\n")

    # Setup classification
    df["bearish_setup"] = df["vah_d1"] < df["val_d2"]
    df["bullish_setup"] = df["val_d1"] > df["vah_d2"]

    # Sanity check: trigger distribution
    print(f"trigger distribution overall:")
    print(df["trigger"].value_counts().to_string())
    print()

    # ------------------------------ Reporting ------------------------------

    def fmt(label, sub: pd.DataFrame, base_n: int):
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
        pct = n / base_n if base_n else 0.0
        print(f"  {label:<70}  n={n:>4} ({pct:>5.1%})  P(up)={p_up:.1%}  P(down)={p_dn:.1%}  "
              f"mean={m:+6.2f}  med={md:+6.2f}  t={t:+5.2f}  p={p:.4f}{sig}")

    base_n = len(df)
    print(f"valid sample: {base_n} days  (date range {df['date'].min()} -> {df['date'].max()})")
    print()
    print("Sig markers: * p<0.05, ** p<0.01, *** p<0.001")
    print(f"{'cohort':<70}  {'n':>4} ({'%base':>5})  {'P(up)':<9}  {'P(dn)':<9}  {'mean':>7}  {'med':>7}  {'t':>5}  {'p':>5}")
    print()

    print("=" * 130)
    print("BASELINE")
    print("=" * 130)
    fmt("ALL valid days", df, base_n)
    fmt("ALL + long_break  (3 consec 5-min closes above overnight range)",
        df[df["trigger"]=="long_break"], base_n)
    fmt("ALL + short_break (3 consec 5-min closes below overnight range)",
        df[df["trigger"]=="short_break"], base_n)
    fmt("ALL + no_trigger  (range never broken)",
        df[df["trigger"]=="no_trigger"], base_n)

    bear = df[df["bearish_setup"]]
    print()
    print("=" * 130)
    print(f"BEARISH SETUP — VAH(D-1) < VAL(D-2)")
    print("=" * 130)
    fmt("ALL bearish-setup days", bear, base_n)
    fmt("  + long_break  (range broken UP first)",
        bear[bear["trigger"]=="long_break"], base_n)
    fmt("  + short_break (range broken DOWN first)",
        bear[bear["trigger"]=="short_break"], base_n)
    fmt("  + no_trigger  (range never broken — chop inside)",
        bear[bear["trigger"]=="no_trigger"], base_n)

    bull = df[df["bullish_setup"]]
    print()
    print("=" * 130)
    print(f"BULLISH SETUP — VAL(D-1) > VAH(D-2)")
    print("=" * 130)
    fmt("ALL bullish-setup days", bull, base_n)
    fmt("  + long_break  (range broken UP first)",
        bull[bull["trigger"]=="long_break"], base_n)
    fmt("  + short_break (range broken DOWN first)",
        bull[bull["trigger"]=="short_break"], base_n)
    fmt("  + no_trigger  (range never broken — chop inside)",
        bull[bull["trigger"]=="no_trigger"], base_n)


if __name__ == "__main__":
    sys.exit(main())

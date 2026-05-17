"""Overnight-range break + reversal study (no VA filter).

For each trading day D:
  1. Compute overnight range (1-min high/low from 18:00 D-1 to 09:30 D).
  2. Detect first range break with N=3 consecutive 5-min closes outside range:
       - long_break  = 3 consec closes ABOVE high_range
       - short_break = 3 consec closes BELOW low_range
       - first-trigger-only
  3. After trigger, detect reversal: N consec 5-min closes back inside the
     broken level. Test reversal thresholds rev_N ∈ {1, 2, 3}.
  4. After reversal threshold met, classify rest of the day:
       - reasserts: at least one subsequent 5-min close goes BACK in trigger
                    direction (e.g., for short_break: another close < low)
       - holds:     no subsequent close in trigger direction
  5. Also separately track: 17:00 close on trigger side of broken level

Outputs:
  - Trigger time distribution by 30-min buckets (long_break, short_break)
  - Reversal sub-cohort stats per direction × rev_N
  - 17:00 close-side rate per cohort
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

NQ_1MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT_DIR = Path(__file__).parent
PER_DAY_OUT = OUT_DIR / "range_break_reversal_per_day.parquet"

ENTRY_N = 3   # consecutive closes outside range to trigger break
REVERSAL_NS = [1, 2, 3]


def load_nq_1m() -> pd.DataFrame:
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    return df.sort_index()


def overnight_range(nq: pd.DataFrame, date: dt.date):
    prev = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                        tz="America/New_York") - pd.Timedelta(days=1)
    start = prev.replace(hour=18, minute=0, second=0)
    end = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                       hour=9, minute=30, tz="America/New_York")
    seg = nq.loc[start:end]
    if seg.empty: return None
    return float(seg["high"].max()), float(seg["low"].min())


def analyze_day(rth5: pd.Series, high: float, low: float):
    """Returns dict with keys for trigger detection, reversal detection, and
    end-of-day classification."""
    closes = list(rth5.items())
    out = {
        "trigger": "no_trigger", "trigger_time": None,
        "trigger_idx": None,
    }
    # Entry trigger: ENTRY_N consec closes outside range, first wins
    above = 0; below = 0
    for i, (t, c) in enumerate(closes):
        if c > high:
            above += 1; below = 0
            if above >= ENTRY_N:
                out["trigger"] = "long_break"; out["trigger_time"] = t; out["trigger_idx"] = i
                break
        elif c < low:
            below += 1; above = 0
            if below >= ENTRY_N:
                out["trigger"] = "short_break"; out["trigger_time"] = t; out["trigger_idx"] = i
                break
        else:
            above = 0; below = 0
    # If no trigger, return early
    if out["trigger"] == "no_trigger":
        return out

    # After trigger, detect reversal at each rev_N threshold
    direction = out["trigger"]
    start_idx = out["trigger_idx"] + 1
    after_trigger = closes[start_idx:]
    # Reversal counter: closes back inside the broken side
    # For short_break (broke below low): reversal = close > low
    # For long_break (broke above high): reversal = close < high
    consecutive_back = 0
    rev_times = {n: None for n in REVERSAL_NS}
    rev_idx_map = {n: None for n in REVERSAL_NS}
    for j, (t, c) in enumerate(after_trigger):
        if direction == "short_break":
            back_inside = c > low
        else:
            back_inside = c < high
        if back_inside:
            consecutive_back += 1
            for n in REVERSAL_NS:
                if rev_times[n] is None and consecutive_back >= n:
                    rev_times[n] = t
                    rev_idx_map[n] = start_idx + j
        else:
            consecutive_back = 0
    # For each reversal threshold, classify what happens after
    for n in REVERSAL_NS:
        if rev_times[n] is None:
            out[f"rev{n}_status"] = "no_reversal"
            out[f"rev{n}_time"] = None
        else:
            after_rev = closes[rev_idx_map[n] + 1:]
            reasserts = False
            for tt, cc in after_rev:
                if direction == "short_break" and cc < low:
                    reasserts = True; break
                if direction == "long_break" and cc > high:
                    reasserts = True; break
            out[f"rev{n}_status"] = "reasserts" if reasserts else "holds"
            out[f"rev{n}_time"] = rev_times[n]
    return out


def main():
    print("loading NQ 1-min bars...")
    nq = load_nq_1m()
    print(f"  range: {nq.index.min()} -> {nq.index.max()}")

    rth_mask = (nq.index.time >= dt.time(9, 30)) & (nq.index.time <= dt.time(17, 0))
    rth_1m = nq[rth_mask]
    rth_5m = rth_1m["close"].resample("5min", origin="epoch").last().dropna()
    rth_5m = rth_5m[(rth_5m.index.time >= dt.time(9, 30)) &
                    (rth_5m.index.time <= dt.time(17, 0))]
    rth_5m_by_date = {d: g for d, g in rth_5m.groupby(rth_5m.index.date)}
    print(f"  5-min RTH days: {len(rth_5m_by_date)}")

    rows = []
    t0 = time.time()
    for i, (d, bars) in enumerate(rth_5m_by_date.items(), 1):
        if d.weekday() >= 5: continue
        if len(bars) < 5: continue
        rng = overnight_range(nq, d)
        if rng is None: continue
        high, low = rng
        out = analyze_day(bars, high, low)
        out["date"] = d
        out["overnight_high"] = high; out["overnight_low"] = low
        out["open_930"] = float(bars.iloc[0])
        out["close_1700"] = float(bars.iloc[-1])
        out["day_ret_pts"] = out["close_1700"] - out["open_930"]
        out["close_above_high"] = out["close_1700"] > high
        out["close_below_low"]  = out["close_1700"] < low
        rows.append(out)
        if i % 200 == 0:
            print(f"  {i}/{len(rth_5m_by_date)}  elapsed={time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    df.to_parquet(PER_DAY_OUT, compression="zstd", index=False)
    print(f"\nwrote {PER_DAY_OUT}  ({len(df)} rows)")

    base_n = len(df)
    print(f"\nvalid sample: {base_n}  range: {df['date'].min()} -> {df['date'].max()}")
    print(f"trigger distribution: {df['trigger'].value_counts().to_dict()}")

    # ---------------- Trigger time distribution ----------------
    print()
    print("=" * 100)
    print("TRIGGER TIME DISTRIBUTION  (entry = 3 consec 5-min closes outside range)")
    print("=" * 100)

    df["t_min"] = df["trigger_time"].apply(
        lambda t: t.hour * 60 + t.minute if t is not None else None)
    starts = list(range(9*60+30, 17*60, 30))

    def bucket_label(s, e):
        sh, sm = divmod(s, 60); eh, em = divmod(e, 60)
        return f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"

    for direction in ["long_break", "short_break"]:
        print(f"\n--- {direction} (n={(df['trigger']==direction).sum()}) ---")
        d_sub = df[df["trigger"] == direction]
        for s in starts:
            e = s + 30
            cnt = ((d_sub["t_min"] >= s) & (d_sub["t_min"] < e)).sum()
            pct = cnt / len(d_sub) if len(d_sub) else 0
            bar = "#" * int(pct * 50)
            print(f"  {bucket_label(s, e):<14}  n={cnt:>4} ({pct:>5.1%})  {bar}")

    # ---------------- Reversal sub-cohort stats ----------------
    def fmt(label, sub: pd.DataFrame, broken_level_col: str, direction: str):
        n = len(sub)
        if n < 2:
            print(f"  {label:<60}  n={n}"); return
        rets = sub["day_ret_pts"].dropna().values
        n = len(rets)
        p_up = float((rets > 0).mean())
        p_dn = float((rets < 0).mean())
        m = float(rets.mean())
        # Trigger-side close: close < low for short, close > high for long
        if direction == "short_break":
            trig_side_rate = float(sub["close_below_low"].mean())
            trig_label = "P(close<low)"
        else:
            trig_side_rate = float(sub["close_above_high"].mean())
            trig_label = "P(close>high)"
        t, p = ss.ttest_1samp(rets, 0)
        sig = "  ***" if p<0.001 else ("   **" if p<0.01 else ("    *" if p<0.05 else "     "))
        print(f"  {label:<60}  n={n:>4}  P(up)={p_up:.1%}  {trig_label}={trig_side_rate:.1%}  "
              f"mean={m:+6.2f}  t={t:+5.2f}  p={p:.4f}{sig}")

    print()
    print("=" * 130)
    print("REVERSAL ANALYSIS  (after trigger, count consec closes back inside; check if break re-asserts)")
    print("=" * 130)
    for direction in ["short_break", "long_break"]:
        d_sub = df[df["trigger"] == direction]
        n_total = len(d_sub)
        broken_label = "low_range" if direction == "short_break" else "high_range"
        print(f"\n--- {direction.upper()} cohort  (n={n_total}, range break = {broken_label}) ---")
        fmt(f"  ALL {direction} days  (no reversal filter)", d_sub, broken_label, direction)
        for n in REVERSAL_NS:
            print(f"\n  Reversal threshold = {n} consec close{'s' if n > 1 else ''} back inside:")
            fmt(f"    no_reversal  (never made {n} consec close{'s' if n>1 else ''} inside)",
                d_sub[d_sub[f"rev{n}_status"] == "no_reversal"], broken_label, direction)
            fmt(f"    reasserts    (after rev{n}, broken level retested)",
                d_sub[d_sub[f"rev{n}_status"] == "reasserts"], broken_label, direction)
            fmt(f"    holds        (after rev{n}, no retest — false break confirmed)",
                d_sub[d_sub[f"rev{n}_status"] == "holds"], broken_label, direction)


if __name__ == "__main__":
    sys.exit(main())

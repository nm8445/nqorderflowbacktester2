"""Overnight-range break + reversal study v2 (forward-looking only).

Fixes the lookahead bias from v1:
  - Drops the "no_reversal" cohort (which used retrospective filtering)
  - Reports PnL FROM the reversal point (decision moment) to 17:00 close,
    not from open to close
  - Threshold labeling: >=N consecutive 5-min closes back inside the broken
    level

For each day:
  1. Compute overnight range from 1-min bars (18:00 D-1 to 09:30 D)
  2. Detect first 3-consec range break (first-trigger-only):
       long_break  = 3 consec closes ABOVE high_range
       short_break = 3 consec closes BELOW low_range
  3. After trigger, for each reversal threshold N ∈ {1, 2, 3}:
       Find first moment when >=N consec closes back inside the broken level.
       If found (T_rev_N): walk forward to 17:00 and check whether the broken
         level is breached again ("reasserts") or not ("holds")
       Record price at T_rev_N for forward PnL calc

Outputs:
  - Section A: trigger time distribution
  - Section B: ALL trigger days (no reversal filter) — bias correctness
  - Section C: reasserts vs holds at each >=N threshold, with PnL measured
    from T_rev_N to 17:00 close
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
PER_DAY_OUT = OUT_DIR / "range_break_reversal_v2_per_day.parquet"

ENTRY_N = 3
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
    closes = list(rth5.items())
    out = {"trigger": "no_trigger", "trigger_time": None, "trigger_idx": None,
           "trigger_price": None, "open_price": float(rth5.iloc[0]),
           "close_1700": float(rth5.iloc[-1])}

    # Entry trigger
    above = 0; below = 0
    for i, (t, c) in enumerate(closes):
        if c > high:
            above += 1; below = 0
            if above >= ENTRY_N:
                out.update(trigger="long_break", trigger_time=t,
                           trigger_idx=i, trigger_price=float(c)); break
        elif c < low:
            below += 1; above = 0
            if below >= ENTRY_N:
                out.update(trigger="short_break", trigger_time=t,
                           trigger_idx=i, trigger_price=float(c)); break
        else:
            above = 0; below = 0
    if out["trigger"] == "no_trigger":
        return out

    direction = out["trigger"]
    start_idx = out["trigger_idx"] + 1
    after = closes[start_idx:]

    # Find first T_rev_N for each N
    consecutive_back = 0
    rev_info = {n: {"t": None, "idx": None, "price": None} for n in REVERSAL_NS}
    for j, (t, c) in enumerate(after):
        back_inside = (c > low) if direction == "short_break" else (c < high)
        if back_inside:
            consecutive_back += 1
            for n in REVERSAL_NS:
                if rev_info[n]["t"] is None and consecutive_back >= n:
                    rev_info[n] = {"t": t, "idx": start_idx + j, "price": float(c)}
        else:
            consecutive_back = 0

    # For each N, classify forward-looking outcome from T_rev_N
    for n in REVERSAL_NS:
        if rev_info[n]["t"] is None:
            out[f"rev{n}_observed"] = False
            out[f"rev{n}_status"] = "never_reversed"  # observable only at 17:00
            out[f"rev{n}_price"] = None
            out[f"rev{n}_pnl_from_rev"] = None
            out[f"rev{n}_close_on_bias_side"] = None
            continue
        out[f"rev{n}_observed"] = True
        rev_idx = rev_info[n]["idx"]
        rev_price = rev_info[n]["price"]
        out[f"rev{n}_price"] = rev_price
        out[f"rev{n}_time"] = rev_info[n]["t"]
        # Walk forward from rev_idx+1; check for reassert
        after_rev = closes[rev_idx + 1:]
        reasserts = False
        for tt, cc in after_rev:
            if direction == "short_break" and cc < low: reasserts = True; break
            if direction == "long_break"  and cc > high: reasserts = True; break
        out[f"rev{n}_status"] = "reasserts" if reasserts else "holds"
        # PnL FROM rev_price to 17:00 close, signed in direction of original bias
        if direction == "long_break":
            out[f"rev{n}_pnl_from_rev"] = out["close_1700"] - rev_price
            out[f"rev{n}_close_on_bias_side"] = out["close_1700"] > high
        else:
            out[f"rev{n}_pnl_from_rev"] = rev_price - out["close_1700"]
            out[f"rev{n}_close_on_bias_side"] = out["close_1700"] < low
    return out


def main():
    print("loading NQ 1-min bars...")
    nq = load_nq_1m()
    print(f"  range: {nq.index.min()} -> {nq.index.max()}")

    rth_mask = (nq.index.time >= dt.time(9, 30)) & (nq.index.time <= dt.time(17, 0))
    rth_5m = nq[rth_mask]["close"].resample("5min", origin="epoch").last().dropna()
    rth_5m = rth_5m[(rth_5m.index.time >= dt.time(9, 30)) &
                    (rth_5m.index.time <= dt.time(17, 0))]
    rth_by_date = {d: g for d, g in rth_5m.groupby(rth_5m.index.date)}
    print(f"  RTH days: {len(rth_by_date)}")

    rows = []
    t0 = time.time()
    for i, (d, bars) in enumerate(rth_by_date.items(), 1):
        if d.weekday() >= 5: continue
        if len(bars) < 5: continue
        rng = overnight_range(nq, d)
        if rng is None: continue
        high, low = rng
        out = analyze_day(bars, high, low)
        out["date"] = d
        out["overnight_high"] = high; out["overnight_low"] = low
        out["close_above_high"] = out["close_1700"] > high
        out["close_below_low"]  = out["close_1700"] < low
        out["open_to_close"] = out["close_1700"] - out["open_price"]
        rows.append(out)
        if i % 200 == 0:
            print(f"  {i}/{len(rth_by_date)}  elapsed={time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    df.to_parquet(PER_DAY_OUT, compression="zstd", index=False)
    print(f"\nwrote {PER_DAY_OUT}  ({len(df)} rows)")

    base_n = len(df)
    print(f"\nvalid sample: {base_n}  range: {df['date'].min()} -> {df['date'].max()}")
    print(f"trigger distribution: {df['trigger'].value_counts().to_dict()}")

    # ---------------- Section A: Trigger time distribution ----------------
    print()
    print("=" * 100)
    print("SECTION A — TRIGGER TIME DISTRIBUTION  (entry = 3 consec 5-min closes outside range)")
    print("=" * 100)
    df["t_min"] = df["trigger_time"].apply(
        lambda t: t.hour * 60 + t.minute if t is not None else None)
    starts = list(range(9*60+30, 17*60, 30))

    def bucket_label(s, e):
        sh, sm = divmod(s, 60); eh, em = divmod(e, 60)
        return f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"

    for direction in ["long_break", "short_break"]:
        d_sub = df[df["trigger"] == direction]
        n = len(d_sub)
        print(f"\n--- {direction} (n={n}) ---")
        for s in starts:
            e = s + 30
            cnt = ((d_sub["t_min"] >= s) & (d_sub["t_min"] < e)).sum()
            pct = cnt / n if n else 0
            bar = "#" * int(pct * 50)
            print(f"  {bucket_label(s, e):<14}  n={cnt:>4} ({pct:>5.1%})  {bar}")

    # ---------------- Section B: ALL trigger days — bias correctness ----------------
    print()
    print("=" * 110)
    print("SECTION B — BIAS CORRECTNESS  (no reversal filter, PnL = open_930 to close_1700)")
    print("=" * 110)
    print("Sig markers: * p<0.05, ** p<0.01, *** p<0.001\n")

    def report_bias(label, sub: pd.DataFrame, direction: str):
        n = len(sub)
        if n < 2:
            print(f"  {label:<60}  n={n}"); return
        rets = sub["open_to_close"].dropna().values
        n = len(rets)
        if direction == "long_break":
            p_correct = float((rets > 0).mean()); m = float(rets.mean())
            t, p = ss.ttest_1samp(rets, 0)
        else:
            p_correct = float((rets < 0).mean()); m = float(rets.mean())
            t, p = ss.ttest_1samp(rets, 0)
        sig = "  ***" if p<0.001 else ("   **" if p<0.01 else ("    *" if p<0.05 else "     "))
        print(f"  {label:<60}  n={n:>4}  P(bias correct)={p_correct:.1%}  "
              f"mean(open->close)={m:+6.2f} pts  t={t:+5.2f}  p={p:.4f}{sig}")

    report_bias("ALL long_break days  (bias=LONG)",
                df[df["trigger"]=="long_break"], "long_break")
    report_bias("ALL short_break days  (bias=SHORT)",
                df[df["trigger"]=="short_break"], "short_break")

    # ---------------- Section C: Reversal observed → reasserts / holds ----------------
    print()
    print("=" * 130)
    print("SECTION C — REVERSAL ANALYSIS  (forward-looking from T_rev to 17:00)")
    print("=" * 130)
    print("PnL is measured FROM the reversal-observed price (decision moment) to 17:00 close,")
    print("signed in original bias direction. P(reassert) = price closes beyond broken level by 17:00.\n")

    def report_rev(label, sub: pd.DataFrame, direction: str, n_col: int):
        n = len(sub)
        if n < 2:
            print(f"  {label:<65}  n={n}"); return
        pnl = sub[f"rev{n_col}_pnl_from_rev"].dropna().values
        n = len(pnl)
        p_profit = float((pnl > 0).mean())
        m = float(pnl.mean())
        # P(close on bias side) — equivalent to "reasserts at end of day"
        p_bias = float(sub[f"rev{n_col}_close_on_bias_side"].mean())
        # Among these, what fraction were classified "reasserts" before 17:00?
        p_reass = float((sub[f"rev{n_col}_status"]=="reasserts").mean())
        t, p = ss.ttest_1samp(pnl, 0)
        sig = "  ***" if p<0.001 else ("   **" if p<0.01 else ("    *" if p<0.05 else "     "))
        print(f"  {label:<65}  n={n:>4}  P(bias side at close)={p_bias:.1%}  "
              f"P(reasserts intraday)={p_reass:.1%}  mean_pnl_from_rev={m:+6.2f}  "
              f"t={t:+5.2f}  p={p:.4f}{sig}")

    for direction, label_dir in [("short_break", "SHORT_BREAK"),
                                  ("long_break", "LONG_BREAK")]:
        d_sub = df[df["trigger"] == direction]
        n_total = len(d_sub)
        print(f"\n--- {label_dir} cohort  (n={n_total} trigger days) ---")
        for n in REVERSAL_NS:
            obs = d_sub[d_sub[f"rev{n}_observed"] == True]
            n_obs = len(obs)
            pct = n_obs / n_total if n_total else 0
            print(f"\n  Reversal threshold >={n}:  n_observed = {n_obs} ({pct:.1%} of trigger days)")
            report_rev(f"    >={n} reversal — REASSERTS  (price re-broke level by 17:00)",
                       obs[obs[f"rev{n}_status"]=="reasserts"], direction, n)
            report_rev(f"    >={n} reversal — HOLDS      (no re-break by 17:00)",
                       obs[obs[f"rev{n}_status"]=="holds"], direction, n)


if __name__ == "__main__":
    sys.exit(main())

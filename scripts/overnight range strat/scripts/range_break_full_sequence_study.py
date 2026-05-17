"""Overnight-range break + full-sequence reversal study.

Extends overnight_range_break_reversal_v2 by tracking the FULL post-trigger
sequence and distinguishing three post-reclaim outcomes (not just reassert/holds):

  REASSERT  = 3 consec 5-min closes back outside on original break side
              (original bias wins — pullback inside range was just a pullback)
  OPPOSITE  = 3 consec 5-min closes outside on the OPPOSITE side
              (full double-break reversal — original break was a trap)
  CHOP      = neither happens by 17:00 (no follow-through either way)

First-occurrence wins: whichever of REASSERT / OPPOSITE fires first determines
the day's classification. Days that fire neither = CHOP.

Entry trigger and the post-reclaim re-trigger both use 3 consecutive 5-min
closes (symmetric definition).

For each day:
  1. Compute overnight range from 1-min bars (18:00 ET D-1 to 09:30 ET D)
  2. Walk RTH 5-min closes 9:30->17:00 for FIRST 3-consec break (long/short)
  3. After trigger, locate first moment of >=N consec closes back inside the
     broken level for each N in {1, 2, 3}
  4. From each T_rev_N, walk forward to 17:00 and classify outcome (3-way)
  5. Record forward PnL from T_rev_N to 17:00 close, signed two ways

Outputs:
  Section A — entry trigger time distribution
  Section B — full trigger cohort: P(close beyond broken level at 17:00)
              [the stat the v2 file did NOT report directly]
  Section C — for each N, frequency of REASSERT / OPPOSITE / CHOP and forward
              PnL signed in original-bias and in reversal directions
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
PARQUET_DIR = OUT_DIR / "parquets"
PARQUET_DIR.mkdir(exist_ok=True)
PER_DAY_OUT = PARQUET_DIR / "range_break_full_sequence_per_day.parquet"

ENTRY_N = 3
RETRIGGER_N = 3  # same as entry (symmetric)
REVERSAL_NS = [1, 2, 3, 4, 5, 6]


def load_nq_1m() -> pd.DataFrame:
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    return df.sort_index()


def overnight_range(nq: pd.DataFrame, date: dt.date):
    prev = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                        tz="America/New_York") - pd.Timedelta(days=1)
    start = prev.replace(hour=18, minute=0, second=0)
    end = pd.Timestamp(year=date.year, month=date.month, day=date.day,
                       hour=9, minute=30, tz="America/New_York")
    seg = nq.loc[start:end]
    if seg.empty:
        return None
    return float(seg["high"].max()), float(seg["low"].min())


def classify_post_reclaim(closes_after, high, low, direction):
    """Walk forward from a reclaim moment. Whichever 3-consec event fires
    first wins. Returns (outcome, time, idx_offset, price)."""
    above = 0
    below = 0
    for j, (t, c) in enumerate(closes_after):
        if c > high:
            above += 1
            below = 0
            if above >= RETRIGGER_N:
                outcome = "reassert" if direction == "long_break" else "opposite"
                return outcome, t, j, float(c)
        elif c < low:
            below += 1
            above = 0
            if below >= RETRIGGER_N:
                outcome = "reassert" if direction == "short_break" else "opposite"
                return outcome, t, j, float(c)
        else:
            above = 0
            below = 0
    return "chop", None, None, None


def analyze_day(rth5: pd.Series, high: float, low: float):
    closes = list(rth5.items())
    out = {
        "trigger": "no_trigger", "trigger_time": None, "trigger_idx": None,
        "trigger_price": None, "open_price": float(rth5.iloc[0]),
        "close_1700": float(rth5.iloc[-1]),
    }

    # Entry trigger (first wins)
    above = 0
    below = 0
    for i, (t, c) in enumerate(closes):
        if c > high:
            above += 1
            below = 0
            if above >= ENTRY_N:
                out.update(trigger="long_break", trigger_time=t,
                           trigger_idx=i, trigger_price=float(c))
                break
        elif c < low:
            below += 1
            above = 0
            if below >= ENTRY_N:
                out.update(trigger="short_break", trigger_time=t,
                           trigger_idx=i, trigger_price=float(c))
                break
        else:
            above = 0
            below = 0

    if out["trigger"] == "no_trigger":
        return out

    direction = out["trigger"]
    start_idx = out["trigger_idx"] + 1
    after = closes[start_idx:]

    # Find first T_rev_N for each N (>=N consec closes back inside)
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

    # Classify outcome from each T_rev_N
    for n in REVERSAL_NS:
        if rev_info[n]["t"] is None:
            out[f"rev{n}_observed"] = False
            out[f"rev{n}_outcome"] = "no_reclaim"
            out[f"rev{n}_price"] = None
            out[f"rev{n}_pnl_bias"] = None
            out[f"rev{n}_pnl_reversal"] = None
            continue
        out[f"rev{n}_observed"] = True
        rev_idx = rev_info[n]["idx"]
        rev_price = rev_info[n]["price"]
        out[f"rev{n}_price"] = rev_price
        out[f"rev{n}_time"] = rev_info[n]["t"]
        outcome, _, _, _ = classify_post_reclaim(
            closes[rev_idx + 1:], high, low, direction)
        out[f"rev{n}_outcome"] = outcome
        # PnL from T_rev_N to 17:00 close
        delta = out["close_1700"] - rev_price
        if direction == "long_break":
            out[f"rev{n}_pnl_bias"] = delta            # bias=long, +ve = up
            out[f"rev{n}_pnl_reversal"] = -delta       # reversal=short
        else:
            out[f"rev{n}_pnl_bias"] = -delta           # bias=short
            out[f"rev{n}_pnl_reversal"] = delta        # reversal=long
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
        if d.weekday() >= 5:
            continue
        if len(bars) < 5:
            continue
        rng = overnight_range(nq, d)
        if rng is None:
            continue
        high, low = rng
        out = analyze_day(bars, high, low)
        out["date"] = d
        out["overnight_high"] = high
        out["overnight_low"] = low
        out["close_above_high"] = out["close_1700"] > high
        out["close_below_low"] = out["close_1700"] < low
        out["close_inside_range"] = (
            (out["close_1700"] >= low) & (out["close_1700"] <= high))
        rows.append(out)
        if i % 200 == 0:
            print(f"  {i}/{len(rth_by_date)}  elapsed={time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    df.to_parquet(PER_DAY_OUT, compression="zstd", index=False)
    print(f"\nwrote {PER_DAY_OUT}  ({len(df)} rows)")

    base_n = len(df)
    print(f"\nvalid sample: {base_n}  range: {df['date'].min()} -> {df['date'].max()}")
    print(f"trigger distribution: {df['trigger'].value_counts().to_dict()}")

    # ---------------- Section A: entry trigger time distribution ----------------
    print()
    print("=" * 100)
    print("SECTION A - ENTRY TRIGGER TIME DISTRIBUTION  (3 consec 5-min closes outside range)")
    print("=" * 100)
    df["t_min"] = df["trigger_time"].apply(
        lambda t: t.hour * 60 + t.minute if t is not None else None)
    starts = list(range(9 * 60 + 30, 17 * 60, 30))

    def bucket_label(s, e):
        sh, sm = divmod(s, 60)
        eh, em = divmod(e, 60)
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

    # ---------------- Section B: full trigger cohort - close location ----------------
    print()
    print("=" * 110)
    print("SECTION B - FULL TRIGGER COHORT  (no reversal filter, where did 17:00 close land?)")
    print("=" * 110)

    def report_close_location(label, sub, direction):
        n = len(sub)
        if n == 0:
            print(f"  {label:<55}  n={n}")
            return
        if direction == "long_break":
            p_beyond = float(sub["close_above_high"].mean())
            beyond_label = "above HOD"
            p_opposite = float(sub["close_below_low"].mean())
            opp_label = "below LOD"
        else:
            p_beyond = float(sub["close_below_low"].mean())
            beyond_label = "below LOD"
            p_opposite = float(sub["close_above_high"].mean())
            opp_label = "above HOD"
        p_inside = float(sub["close_inside_range"].mean())
        print(f"  {label:<55}  n={n:>4}  P(close {beyond_label})={p_beyond:.1%}  "
              f"P(close inside range)={p_inside:.1%}  P(close {opp_label})={p_opposite:.1%}")

    report_close_location("ALL long_break  (bias=LONG)",
                          df[df["trigger"] == "long_break"], "long_break")
    report_close_location("ALL short_break (bias=SHORT)",
                          df[df["trigger"] == "short_break"], "short_break")
    no_trig = df[df["trigger"] == "no_trigger"]
    n_nt = len(no_trig)
    if n_nt:
        p_above = float(no_trig["close_above_high"].mean())
        p_below = float(no_trig["close_below_low"].mean())
        p_inside = float(no_trig["close_inside_range"].mean())
        print(f"  {'NO trigger (range never broken)':<55}  n={n_nt:>4}  "
              f"P(close above HOD)={p_above:.1%}  P(close inside)={p_inside:.1%}  "
              f"P(close below LOD)={p_below:.1%}")

    # ---------------- Section C: 3-way reclaim outcome by threshold ----------------
    print()
    print("=" * 130)
    print("SECTION C - POST-RECLAIM OUTCOME  (3-way: REASSERT / OPPOSITE / CHOP, threshold = 3 consec 5-min)")
    print("=" * 130)
    print("Conditional on reclaim>=N: P(reassert) = original bias wins, P(opposite) = full double-break reversal,")
    print("P(chop) = neither fires by 17:00.\n")

    def report_three_way(direction, label_dir):
        d_sub = df[df["trigger"] == direction]
        n_total = len(d_sub)
        print(f"\n--- {label_dir} cohort  (n={n_total} entry-trigger days) ---")
        for n in REVERSAL_NS:
            obs = d_sub[d_sub[f"rev{n}_observed"] == True]
            n_obs = len(obs)
            pct = n_obs / n_total if n_total else 0
            print(f"\n  Reclaim threshold >={n}:  n_observed = {n_obs} "
                  f"({pct:.1%} of trigger days)")
            if n_obs == 0:
                continue
            counts = obs[f"rev{n}_outcome"].value_counts()
            p_reassert = counts.get("reassert", 0) / n_obs
            p_opposite = counts.get("opposite", 0) / n_obs
            p_chop = counts.get("chop", 0) / n_obs

            # PnL summaries by outcome bucket
            def pnl_stats(bucket):
                grp = obs[obs[f"rev{n}_outcome"] == bucket]
                if len(grp) == 0:
                    return "n=0"
                pb = grp[f"rev{n}_pnl_bias"].dropna().values
                pr = grp[f"rev{n}_pnl_reversal"].dropna().values
                return (f"n={len(grp):>3}  "
                        f"mean_bias={pb.mean():+6.2f}  "
                        f"mean_rev={pr.mean():+6.2f}")

            print(f"    P(REASSERT)={p_reassert:5.1%}  {pnl_stats('reassert')}")
            print(f"    P(OPPOSITE)={p_opposite:5.1%}  {pnl_stats('opposite')}")
            print(f"    P(CHOP)    ={p_chop:5.1%}  {pnl_stats('chop')}")

    report_three_way("short_break", "SHORT_BREAK")
    report_three_way("long_break", "LONG_BREAK")

    # ---------------- Section D: marginal (rev>=N AND not >=N+1) ----------------
    print()
    print("=" * 130)
    print("SECTION D - MARGINAL  (exactly N closes back inside, did NOT reach N+1)")
    print("=" * 130)
    print("Strips out the 'eventually reaches a higher reclaim threshold' contamination,")
    print("so each row tells you what happened conditional on stopping AT that N.\n")

    def report_marginal(direction, label_dir):
        d_sub = df[df["trigger"] == direction]
        print(f"\n--- {label_dir} cohort ---")
        for n in REVERSAL_NS:
            if n < max(REVERSAL_NS):
                mask = ((d_sub[f"rev{n}_observed"] == True) &
                        (d_sub[f"rev{n+1}_observed"] == False))
                label = f"  Exactly {n} (reached >={n} but not >={n+1})"
            else:
                mask = (d_sub[f"rev{n}_observed"] == True)
                label = f"  >={n}"
            sub = d_sub[mask]
            n_obs = len(sub)
            if n_obs == 0:
                print(f"  {label:<45}  n=0")
                continue
            counts = sub[f"rev{n}_outcome"].value_counts()
            p_reassert = counts.get("reassert", 0) / n_obs
            p_opposite = counts.get("opposite", 0) / n_obs
            p_chop = counts.get("chop", 0) / n_obs
            print(f"  {label:<45}  n={n_obs:>4}  "
                  f"P(reassert)={p_reassert:5.1%}  "
                  f"P(opposite)={p_opposite:5.1%}  "
                  f"P(chop)={p_chop:5.1%}")

    report_marginal("short_break", "SHORT_BREAK")
    report_marginal("long_break", "LONG_BREAK")


if __name__ == "__main__":
    sys.exit(main())

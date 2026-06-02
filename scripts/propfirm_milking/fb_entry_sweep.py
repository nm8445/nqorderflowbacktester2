"""Phase 2 (FB) — re-sweep Fabio ORB entry params (N_confirm, delta) with the fixed
ATR bracket + IS/OOS, to test whether relaxing N (the prior sweep hinted N=1 -> ~57% WR
at ORB-RR1.0) still lifts win rate under our RR~1.33 ATR bracket.

Regenerates FB entries from 5-min delta bars (volumetric parquet, buy_vol-sell_vol summed
over levels -> per-bar delta; NEVER counts L2 as trades), feeds them through the same
fixed-bracket first-touch + IS/OOS evaluator.

Run:  python -m scripts.propfirm_milking.fb_entry_sweep
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import numpy as np
import pandas as pd

from scripts.propfirm_milking.common import (
    ET, RESULTS_DIR, eval_bracket, stats_block, is_oos_cutoff,
)
from scripts.propfirm_milking.entries import packs_from_fills

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
CACHE_5M = RESULTS_DIR / "_fb_5min_delta_bars.parquet"

ORB_START, ORB_END, TRADE_END, SKIP = 830, 900, 1400, 930
N_GRID = [1, 2, 3, 4]
DELTA_GRID = [200, 300]
# Bracket sub-grid (RR ~1.3-1.5, tag-friendly) to find best per (N,delta)
ATR_LENS = [14, 20, 28]
SL_GRID = [1.5, 2.0, 2.25, 2.5]
TP_GRID = [2.0, 2.5, 3.0, 3.25, 3.5]
TAGGED_MIN = 0.70


def build_5min_delta_bars() -> pd.DataFrame:
    if CACHE_5M.exists():
        return pd.read_parquet(CACHE_5M)
    print("  aggregating volumetric parquet -> per-bar 5-min delta bars (one-time)...")
    df = pd.read_parquet(VOL_PARQUET, columns=[
        "bar_close_time", "open", "high", "low", "close", "buy_vol", "sell_vol"])
    g = df.groupby("bar_close_time").agg(
        open=("open", "first"), high=("high", "first"),
        low=("low", "first"), close=("close", "first"),
        buy=("buy_vol", "sum"), sell=("sell_vol", "sum"),
    ).dropna(subset=["close"])
    g["delta"] = g["buy"] - g["sell"]
    g = g[["open", "high", "low", "close", "delta"]].sort_index()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    g.to_parquet(CACHE_5M)
    return g


def generate_fb_entries(bars5: pd.DataFrame, n_confirm: int, delta_thresh: float):
    """Return (fills, signs) for FB with the given N_confirm and delta threshold."""
    idx = bars5.index
    hhmm = (idx.hour * 100 + idx.minute).values
    dates = idx.date
    close = bars5["close"].values
    high = bars5["high"].values
    low = bars5["low"].values
    delta = bars5["delta"].values

    fills, signs = [], []
    # group contiguous rows by date
    df = pd.DataFrame({"hhmm": hhmm, "date": dates,
                       "close": close, "high": high, "low": low, "delta": delta},
                      index=idx)
    for d, day in df.groupby("date", sort=True):
        orb = day[(day["hhmm"] > ORB_START) & (day["hhmm"] <= ORB_END)]
        if orb.empty:
            continue
        orb_high = float(orb["high"].max())
        orb_low = float(orb["low"].min())
        post = day[(day["hhmm"] > ORB_END) & (day["hhmm"] <= TRADE_END)]
        if post.empty:
            continue
        c = post["close"].values
        de = post["delta"].values
        hh = post["hhmm"].values
        ts = post.index
        for i in range(len(post)):
            if hh[i] == SKIP:
                continue
            if i + 1 < n_confirm:
                continue
            if not np.all(c[i - n_confirm + 1:i + 1] > orb_high):
                continue
            if de[i] < delta_thresh:
                continue
            if orb_low >= c[i]:
                continue
            fills.append(ts[i])
            signs.append(1)
            break   # one trade/day
    return fills, signs


def best_bracket(packs, cutoff):
    """Small bracket sub-sweep; return best qualifying row by min(IS_wr,OOS_wr)."""
    best = None
    for a in ATR_LENS:
        for sl in SL_GRID:
            for tp in TP_GRID:
                rr = tp / sl
                if rr < 1.25 or rr > 1.55:
                    continue
                tr = eval_bracket(packs, a, sl, tp)
                if len(tr) < 100:
                    continue
                is_tr = tr[tr["date"] < cutoff]
                oos_tr = tr[tr["date"] >= cutoff]
                i = stats_block(is_tr)
                o = stats_block(oos_tr)
                al = stats_block(tr)
                if i["tagged_frac"] < TAGGED_MIN or o["tagged_frac"] < TAGGED_MIN:
                    continue
                if i["net"] <= 0 or o["net"] <= 0:
                    continue
                min_wr = min(i["wr"], o["wr"])
                row = dict(atr_len=a, sl=sl, tp=tp, rr=round(rr, 3), n=al["n"],
                           wr=al["wr"], is_wr=i["wr"], oos_wr=o["wr"], min_wr=min_wr,
                           net=al["net"], tagged=al["tagged_frac"], mean_mnq=al["mean_mnq"])
                if best is None or min_wr > best["min_wr"]:
                    best = row
    return best


def main():
    pd.set_option("display.width", 200)
    t0 = time.time()
    bars5 = build_5min_delta_bars()
    print(f"  5-min delta bars: {len(bars5):,}  ({time.time()-t0:.1f}s)\n")

    rows = []
    for delta in DELTA_GRID:
        for n in N_GRID:
            fills, signs = generate_fb_entries(bars5, n, delta)
            if len(fills) < 100:
                print(f"  N={n} delta={delta}: only {len(fills)} entries, skip")
                continue
            packs = packs_from_fills("FB", fills, signs, verbose=False)
            cutoff = is_oos_cutoff([p.date for p in packs])
            b = best_bracket(packs, cutoff)
            if b is None:
                print(f"  N={n} delta={delta}: {len(fills)} entries, NO qualifying bracket")
                continue
            b.update(dict(N=n, delta=delta, n_entries=len(fills)))
            rows.append(b)
            print(f"  N={n} delta={delta}: entries={len(fills)} -> best "
                  f"atr{b['atr_len']}/sl{b['sl']}/tp{b['tp']} (RR{b['rr']}) "
                  f"WR {b['wr']}% (IS {b['is_wr']}/OOS {b['oos_wr']}) tagged {b['tagged']} net ${b['net']:,.0f}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / "FB_entry_sweep.csv", index=False)
        print("\n=== FB ENTRY-PARAM SWEEP (best bracket per N,delta) ===")
        cols = ["N", "delta", "n_entries", "atr_len", "sl", "tp", "rr",
                "wr", "is_wr", "oos_wr", "min_wr", "tagged", "net", "mean_mnq"]
        print(df[cols].sort_values("min_wr", ascending=False).to_string(index=False))
        print(f"\nLocked FB baseline: N=4, ~628 entries, WR ~48.4% (IS 47.9/OOS 49.2)")
        print(f"Saved -> {RESULTS_DIR / 'FB_entry_sweep.csv'}")


if __name__ == "__main__":
    main()

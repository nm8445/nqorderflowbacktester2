"""Experimental: does FB's edge depend on how many consecutive BULLISH 20-min candles precede entry?

Hypothesis (user): entering after ~4 consecutive bullish 20-min candles = catching the top of an
already-completed move -> lower expectancy. FB is LONG-only (4 closes above ORB high), so it always
enters into momentum; this asks whether *too much* prior momentum is worse.

Method (no lookahead): a 20-min candle is bullish if close > open. For each FB entry, walk back from
the last COMPLETED 20-min bar at/before entry and count the run of consecutive bullish candles on the
SAME ET session date (don't count across the overnight gap). Bucket trades by run length (0..5+) and
report n / win% / avg$ / total$ for ALL, IS (oldest 60%), OOS (newest 40%).

Run: python scripts/experimental_questions/fb_consec_bullish_20m.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRADES = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
PARQ = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
ET = "America/New_York"


def load_bars_20m():
    b = pd.read_parquet(PARQ, columns=["open", "close"])
    if b.index.tz is None:
        b.index = b.index.tz_localize("UTC")
    b.index = b.index.tz_convert(ET)
    b = b.sort_index()
    o = b["open"].resample("20min", label="right", closed="right").first().dropna()
    c = b["close"].resample("20min", label="right", closed="right").last().dropna()
    df = pd.DataFrame({"open": o, "close": c}).dropna()
    return (df.index.values.astype("int64"),
            (df["close"].values > df["open"].values),          # bullish flag
            df.index.tz_convert(ET).date)                       # ET date per bar (numpy array of date)


def consec_bullish_before(entry_ns, btimes, bullish, bdates, entry_date):
    """Run of consecutive bullish 20-min candles ending at the last completed bar at/before entry,
    restricted to the same session date."""
    idx = np.searchsorted(btimes, entry_ns, "right") - 1     # last bar with close_time <= entry
    run = 0
    j = idx
    while j >= 0 and bullish[j] and bdates[j] == entry_date:
        run += 1
        j -= 1
    return run


def main():
    btimes, bullish, bdates = load_bars_20m()
    bdates = np.array(bdates)

    t = pd.read_csv(TRADES)
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True).dt.tz_convert(ET)
    fb = t[t["strat"] == "FB"].sort_values("entry_ts").reset_index(drop=True)
    ent_ns = fb["entry_ts"].values.astype("int64")
    ent_dates = fb["entry_ts"].dt.date.values
    pnl = fb["pnl_$"].values.astype(float)

    runs = np.array([consec_bullish_before(ent_ns[i], btimes, bullish, bdates, ent_dates[i])
                     for i in range(len(fb))])

    # IS/OOS chronological split by entry date
    dates = np.sort(fb["entry_ts"].dt.date.unique())
    cutoff = dates[int(0.6 * len(dates))]
    oos = fb["entry_ts"].dt.date.values >= cutoff

    def bucket_id(r):
        return min(r, 5)                                       # 0,1,2,3,4,5+(=5)

    bid = np.array([bucket_id(r) for r in runs])

    def report(mask, label):
        print(f"\n=== {label}  (n={mask.sum()}, total ${pnl[mask].sum():,.0f}, "
              f"win {(pnl[mask] > 0).mean()*100:.1f}%) ===")
        print(f"  {'consec bull':>11} {'n':>5} {'win%':>6} {'avg $':>9} {'total $':>10} {'$/trade vs base':>16}")
        base = pnl[mask].mean()
        for k in range(6):
            sel = mask & (bid == k)
            if sel.sum() == 0:
                continue
            p = pnl[sel]
            lbl = "5+" if k == 5 else str(k)
            print(f"  {lbl:>11} {sel.sum():>5} {(p>0).mean()*100:>5.1f}% {p.mean():>9,.0f} "
                  f"{p.sum():>10,.0f} {p.mean()-base:>+16,.0f}")

    print(f"FB consecutive-bullish-20m experiment | {len(fb)} trades | "
          f"bullish = close>open, same-session run")
    report(np.ones(len(fb), bool), "ALL")
    report(~oos, "IN-SAMPLE (oldest 60%)")
    report(oos, "OUT-OF-SAMPLE (newest 40%)")

    # Simple monotonic check: avg $ by run, all-sample
    print("\nAvg $/trade by exact run (all):")
    for k in range(0, 6):
        sel = runs == k if k < 5 else runs >= 5
        if sel.sum():
            print(f"  run {('5+' if k==5 else k)}: {sel.sum():>4} tr  ${pnl[sel].mean():>7,.0f}/tr")


if __name__ == "__main__":
    main()

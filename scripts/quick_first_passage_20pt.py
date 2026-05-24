"""First-passage probability: from a given entry timestamp, does NQ hit
entry+20 pts BEFORE entry-20 pts (long TP), or vice versa (long SL)?

Two scenarios:
  1. OD strategy entry: every weekday at 19:00 ET (Mon evening through Fri evening)
  2. Sunday hedge entry: every Sunday at 19:20 ET

For each, also restrict to recent regime (last 12 months: 2025-05 onward).

Output:
  Long TP %  = price moved +20 pts before -20 pts
  Long SL %  = price moved -20 pts before +20 pts
  Tie/incon %= both reached in same 1-min bar (order ambiguous)

Symmetric brackets => short TP% = long SL%, short SL% = long TP%.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

BARS = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
ET = "America/New_York"
BRACKET = 20.0   # ±20 pts
MAX_HORIZON_BARS = 60 * 14  # 14 hours of 1-min bars (OD holds ~13 hrs)
RECENT_START = pd.Timestamp("2025-05-01", tz=ET)


def first_passage(entry_price: float, fwd_bars: pd.DataFrame) -> str:
    """Return 'up', 'down', 'tie', or 'no_hit'."""
    up_level = entry_price + BRACKET
    dn_level = entry_price - BRACKET
    for ts, row in fwd_bars.iterrows():
        hit_up = row["high"] >= up_level
        hit_dn = row["low"]  <= dn_level
        if hit_up and hit_dn:
            return "tie"
        if hit_up:
            return "up"
        if hit_dn:
            return "down"
    return "no_hit"


def analyze(entries: pd.DatetimeIndex, bars_et: pd.DataFrame, label: str) -> dict:
    """For each entry timestamp, compute first-passage. Returns counts and pcts."""
    results = []
    for entry_ts in entries:
        # Get entry price = OPEN of the bar that STARTS at entry_ts
        # (assume MOC fills at the next minute's open)
        if entry_ts not in bars_et.index:
            continue
        entry_price = bars_et.loc[entry_ts, "open"]
        # Forward bars (skip the entry bar itself — use bars AFTER entry)
        i_entry = bars_et.index.get_loc(entry_ts)
        fwd = bars_et.iloc[i_entry + 1 : i_entry + 1 + MAX_HORIZON_BARS]
        outcome = first_passage(entry_price, fwd)
        results.append((entry_ts, outcome))

    df = pd.DataFrame(results, columns=["entry_ts", "outcome"])
    n = len(df)
    if n == 0:
        print(f"{label}: NO ENTRIES FOUND")
        return {}
    counts = df["outcome"].value_counts()
    up = counts.get("up", 0)
    dn = counts.get("down", 0)
    tie = counts.get("tie", 0)
    no = counts.get("no_hit", 0)
    # Resolved = up + down (exclude ties and no-hits)
    resolved = up + dn
    print(f"\n{label}: n={n}")
    print(f"  Long TP first (price +20 before -20): {up:4d}  ({up/n*100:5.2f}%)")
    print(f"  Long SL first (price -20 before +20): {dn:4d}  ({dn/n*100:5.2f}%)")
    print(f"  Same-bar tie (ambiguous order):       {tie:4d}  ({tie/n*100:5.2f}%)")
    print(f"  Neither hit in {MAX_HORIZON_BARS} min:           {no:4d}  ({no/n*100:5.2f}%)")
    if resolved > 0:
        print(f"  RESOLVED ratio: Long TP = {up/resolved*100:.1f}%  |  Long SL = {dn/resolved*100:.1f}%")
    return {"label": label, "n": n, "up": up, "dn": dn, "tie": tie, "no": no}


def main():
    print(f"Loading 1-min bars from {BARS}...")
    bars = pd.read_parquet(BARS)
    bars.index = bars.index.tz_convert(ET)
    print(f"  {len(bars):,} bars, {bars.index.min()} -> {bars.index.max()}")

    # Build OD entries: every Mon-Fri at 19:00 ET
    all_dates = pd.date_range(bars.index.min().normalize(), bars.index.max().normalize(), freq="D", tz=ET)
    od_entries = [d.replace(hour=19, minute=0) for d in all_dates if d.weekday() < 5]  # Mon-Fri
    od_entries = pd.DatetimeIndex(od_entries)

    # Sunday hedge entries: every Sunday at 19:20 ET
    sun_entries = [d.replace(hour=19, minute=20) for d in all_dates if d.weekday() == 6]
    sun_entries = pd.DatetimeIndex(sun_entries)

    print(f"\nOD potential entries (weekday 19:00 ET): {len(od_entries)}")
    print(f"Sunday hedge potential entries (Sun 19:20 ET): {len(sun_entries)}")

    # === All-time ===
    print(f"\n{'='*70}")
    print(f"ALL HISTORY ({bars.index.min().date()} -> {bars.index.max().date()})")
    print(f"{'='*70}")
    r1 = analyze(od_entries, bars, "OD entry (weekday 19:00 ET)")
    r2 = analyze(sun_entries, bars, "Sunday hedge entry (Sun 19:20 ET)")

    # === Recent regime: last 12 months ===
    print(f"\n{'='*70}")
    print(f"RECENT REGIME ({RECENT_START.date()} -> {bars.index.max().date()})")
    print(f"{'='*70}")
    od_recent = od_entries[od_entries >= RECENT_START]
    sun_recent = sun_entries[sun_entries >= RECENT_START]
    r3 = analyze(od_recent, bars, "OD entry (recent)")
    r4 = analyze(sun_recent, bars, "Sunday hedge entry (recent)")

    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("  Long TP%   = % of entries where price moved +20 before -20")
    print("  Short TP%  = Long SL% (mirror of long)")
    print("  Short SL%  = Long TP%")
    print("  If Long TP > 50% -> bullish drift from that entry time")
    print("  If Long TP < 50% -> bearish drift")


if __name__ == "__main__":
    main()

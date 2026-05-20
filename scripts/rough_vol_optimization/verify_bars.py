"""
Dump raw 15-min bar data for April 16-17 2026 to verify prices.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
ET = "America/New_York"

# Build bars — same logic as backtest
frames = []
for f in sorted(TIMEBARS_DIR.glob("timebars_5min_202*.pkl")):
    with open(f, "rb") as fh:
        bars = pickle.load(fh)
    if not bars: continue
    rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
             "low": b["low"], "close": b["close"]} for b in bars]
    df5 = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df5.index += pd.Timedelta(minutes=5)
    df5["group"] = df5.index.floor("15min")
    agg = df5.groupby("group").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"))
    agg.index += pd.Timedelta(minutes=15)
    frames.append(agg)
df = pd.concat(frames).sort_index()
df = df[~df.index.duplicated(keep="first")]

df.index = pd.DatetimeIndex([
    t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
    else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
    for t in df.index
])

# Show April 16-17 session bars
print("=" * 90)
print("BAR TIMESTAMP CONVENTION: index = bar CLOSE time")
print("  Bar at 10:00 AM = candle from 09:45 to 10:00")
print("  Bar at 10:15 AM = candle from 10:00 to 10:15")
print("=" * 90)
print()

for date_str in ["2026-04-16", "2026-04-17"]:
    mask = (df.index.date == pd.Timestamp(date_str).date()) & \
           (df.index.strftime("%H:%M") >= "09:30") & \
           (df.index.strftime("%H:%M") <= "16:00")
    day = df[mask]
    print(f"--- {date_str} ---")
    print(f"{'Bar Close Time':<25} {'Open':>12} {'High':>12} {'Low':>12} {'Close':>12}")
    print("-" * 75)
    for idx, row in day.iterrows():
        bar_open = idx - pd.Timedelta(minutes=15)
        print(f"{bar_open.strftime('%I:%M')}-{idx.strftime('%I:%M %p'):<18} "
              f"${row['open']:>10,.2f} ${row['high']:>10,.2f} ${row['low']:>10,.2f} ${row['close']:>10,.2f}")
    print()

# Also show the raw 5-min bars that feed into the 10:00 AM bar on Apr 16
print("=" * 90)
print("RAW 5-MIN BARS that build the 10:00 AM 15-min bar on Apr 16")
print("=" * 90)
for f in sorted(TIMEBARS_DIR.glob("timebars_5min_2026-04-16*.pkl")):
    with open(f, "rb") as fh:
        bars = pickle.load(fh)
    if not bars: continue
    for b in bars:
        ot = b["open_time"]
        if hasattr(ot, "tz_convert") and ot.tzinfo:
            ot_et = ot.tz_convert(ET)
        else:
            ot_et = pd.Timestamp(ot).tz_localize("UTC").tz_convert(ET)
        ct = ot_et + pd.Timedelta(minutes=5)
        if "09:30" <= ct.strftime("%H:%M") <= "10:15":
            print(f"  5min open={ot_et.strftime('%H:%M')} close={ct.strftime('%H:%M')}  "
                  f"O=${b['open']:,.2f} H=${b['high']:,.2f} L=${b['low']:,.2f} C=${b['close']:,.2f}")

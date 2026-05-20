"""
Debug: show exactly which 5-min bars feed into each 15-min bar.
Check if our aggregation is off by one 5-min bar vs TradingView.
"""
import pickle
import pandas as pd
from pathlib import Path

TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
ET = "America/New_York"

# Load just April 16
all_5min = []
for f in sorted(TIMEBARS_DIR.glob("timebars_5min_202*.pkl")):
    with open(f, "rb") as fh:
        bars = pickle.load(fh)
    if not bars: continue
    for b in bars:
        ot = b["open_time"]
        if hasattr(ot, "tz_convert") and ot.tzinfo:
            ot_et = ot.tz_convert(ET)
        else:
            ot_et = pd.Timestamp(ot).tz_localize("UTC").tz_convert(ET)
        if ot_et.date() == pd.Timestamp("2026-04-16").date():
            ct_et = ot_et + pd.Timedelta(minutes=5)
            all_5min.append({
                "open_time": ot_et, "close_time": ct_et,
                "open": b["open"], "high": b["high"],
                "low": b["low"], "close": b["close"]
            })

all_5min.sort(key=lambda x: x["open_time"])

# Show 5-min bars around 9:30-10:30
print("=" * 100)
print("RAW 5-MIN BARS — April 16, 2026 (9:30-10:30 AM ET)")
print("=" * 100)
for b in all_5min:
    if "09:25" <= b["open_time"].strftime("%H:%M") <= "10:25":
        print(f"  open={b['open_time'].strftime('%H:%M')} close={b['close_time'].strftime('%H:%M')}  "
              f"O=${b['open']:,.2f} H=${b['high']:,.2f} L=${b['low']:,.2f} C=${b['close']:,.2f}")

print()
print("=" * 100)
print("CURRENT METHOD: floor(close_time, 15min)")
print("  This groups by close_time floored to 15min")
print("=" * 100)

# Current method
df5 = pd.DataFrame(all_5min).set_index("close_time").sort_index()
df5_session = df5[(df5.index.strftime("%H:%M") >= "09:30") & (df5.index.strftime("%H:%M") <= "10:30")]
df5_session["group"] = df5_session.index.floor("15min")
for grp, sub in df5_session.groupby("group"):
    close_15 = grp + pd.Timedelta(minutes=15)
    print(f"\n  15-min bar labeled {close_15.strftime('%H:%M')} (group={grp.strftime('%H:%M')})")
    print(f"  Contains 5-min bars:")
    for idx, row in sub.iterrows():
        print(f"    {row['open_time'].strftime('%H:%M')}-{idx.strftime('%H:%M')}  C=${row['close']:,.2f}")
    agg_close = sub["close"].iloc[-1]
    print(f"  => 15-min close = ${agg_close:,.2f}")

print()
print("=" * 100)
print("CORRECT METHOD: floor(open_time, 15min)")
print("  TradingView groups by bar OPEN time")
print("=" * 100)

# Correct method — group by open_time floored
df5b = pd.DataFrame(all_5min).sort_values("open_time")
df5b_session = df5b[(df5b["open_time"].dt.strftime("%H:%M") >= "09:30") &
                     (df5b["open_time"].dt.strftime("%H:%M") <= "10:25")]
df5b_session = df5b_session.copy()
df5b_session["group"] = df5b_session["open_time"].dt.floor("15min")
for grp, sub in df5b_session.groupby("group"):
    print(f"\n  15-min bar labeled {grp.strftime('%H:%M')} (TV convention = open time)")
    print(f"  Contains 5-min bars:")
    for _, row in sub.iterrows():
        print(f"    {row['open_time'].strftime('%H:%M')}-{row['close_time'].strftime('%H:%M')}  C=${row['close']:,.2f}")
    agg_close = sub["close"].iloc[-1]
    print(f"  => 15-min close = ${agg_close:,.2f}")

"""
Bar count diagnostic -- verify 40-range bar construction for a single date.

Prints:
  - Raw tick count for the overnight session window
  - Number of range bars built and expected bar size
  - Price min / max / range for the session
  - Distribution of bar range_points (should cluster around 10.0 for 40-tick bars)
  - Average bar volume and duration
  - First and last 5 bar prices so you can sanity-check against a chart

Run with:
    python scripts/diagnose_bars.py
"""

import gc
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize_enriched
from nqbt.analysis.range_bars import build_range_bars, TICK_SIZE

# -- Config --------------------------------------------------------------------
TRADING_DATE = date(2026, 3, 3)      # change to any date you want to inspect
SESSION      = "overnight"           # "overnight" or "rth"
ET           = "America/New_York"
RANGE_TICKS  = 40
# ------------------------------------------------------------------------------

prev_day = TRADING_DATE - timedelta(days=1)

if SESSION == "overnight":
    start = pd.Timestamp(f"{prev_day} 18:00:00", tz=ET)
    end   = pd.Timestamp(f"{TRADING_DATE} 09:30:00", tz=ET)
else:
    start = pd.Timestamp(f"{TRADING_DATE} 09:30:00", tz=ET)
    end   = pd.Timestamp(f"{TRADING_DATE} 11:00:00", tz=ET)

print(f"Session   : {SESSION.upper()} for {TRADING_DATE}")
print(f"Window    : {start.strftime('%Y-%m-%d %H:%M %Z')} -> {end.strftime('%Y-%m-%d %H:%M %Z')}")
print(f"Range bar : {RANGE_TICKS} ticks = {RANGE_TICKS * TICK_SIZE:.2f} points")
print()

print("Loading data...")
raw      = fetch_and_load(str(prev_day), str(TRADING_DATE + timedelta(days=1)))
enriched = normalize_enriched(raw)
del raw
gc.collect()

ticks = enriched[(enriched.index >= start) & (enriched.index <= end)].copy()

print("-- Tick counts ----------------------------------------------------------")
print(f"  Total ticks in window  : {len(ticks):,}")
print(f"  Buy ticks              : {(ticks['aggressor'] == 'buy').sum():,}")
print(f"  Sell ticks             : {(ticks['aggressor'] == 'sell').sum():,}")
print()

if len(ticks) == 0:
    print("No ticks in this window. Check date or data coverage.")
    exit()

print("-- Price distribution ---------------------------------------------------")
print(f"  Min price    : {ticks['price'].min():.2f}")
print(f"  Max price    : {ticks['price'].max():.2f}")
print(f"  Total range  : {ticks['price'].max() - ticks['price'].min():.2f} points")
net_range = ticks['price'].max() - ticks['price'].min()
bar_size  = RANGE_TICKS * TICK_SIZE
print(f"  Expected bars (net range / bar size) ~= {net_range / bar_size:.0f}"
      f"  (floor -- actual count higher due to back-and-forth)")
print()

# Check for price outliers that could cause spurious bar closures
p1   = ticks['price'].quantile(0.01)
p99  = ticks['price'].quantile(0.99)
outliers = ticks[(ticks['price'] < p1) | (ticks['price'] > p99)]
print("-- Outlier check (outside 1st-99th percentile) --------------------------")
print(f"  1st pct  : {p1:.2f}   99th pct  : {p99:.2f}")
print(f"  Outlier ticks : {len(outliers):,}  ({100 * len(outliers) / len(ticks):.2f}%)")
if 0 < len(outliers) <= 20:
    print(outliers[['price', 'size', 'aggressor']].to_string())
print()

# Build bars
print("Building range bars...")
bars   = build_range_bars(ticks, range_ticks=RANGE_TICKS, ticks_per_level=5)
closed = [b for b in bars if b.closed]

print("-- Bar counts -----------------------------------------------------------")
print(f"  Total bars (incl. unclosed) : {len(bars):,}")
print(f"  Closed bars                 : {len(closed):,}")
session_seconds = (end - start).total_seconds()
if len(closed) > 0:
    print(f"  Avg time per bar            : {session_seconds / len(closed):.1f} seconds")
print()

if len(closed) == 0:
    print("No closed bars in this window.")
    exit()

ranges    = np.array([b.range_points for b in closed])
volumes   = np.array([b.total_vol    for b in closed])
durations = np.array([
    (b.close_time - b.open_time).total_seconds()
    for b in closed
    if b.close_time is not None and b.open_time is not None
])

print(f"-- Range distribution (should be ~{bar_size:.1f} for all bars) ----------------")
print(f"  min    : {ranges.min():.4f}")
print(f"  mean   : {ranges.mean():.4f}")
print(f"  max    : {ranges.max():.4f}")
print(f"  std    : {ranges.std():.4f}")
print(f"  Bars with range < {bar_size * 0.5:.1f} (suspiciously small) : "
      f"{(ranges < bar_size * 0.5).sum():,}")
print()

print("-- Volume distribution --------------------------------------------------")
print(f"  min    : {volumes.min():,}")
print(f"  mean   : {volumes.mean():.1f}")
print(f"  median : {np.median(volumes):.1f}")
print(f"  max    : {volumes.max():,}")
print(f"  Bars with total_vol < 5 (nearly empty) : {(volumes < 5).sum():,}")
print()

if len(durations) > 0:
    print("-- Duration distribution (seconds per bar) ------------------------------")
    print(f"  min    : {durations.min():.1f}s")
    print(f"  mean   : {durations.mean():.1f}s")
    print(f"  median : {np.median(durations):.1f}s")
    print(f"  max    : {durations.max():.1f}s")
    print()

print("-- First 5 closed bars --------------------------------------------------")
for b in closed[:5]:
    open_et  = b.open_time.tz_convert(ET).strftime("%H:%M:%S")
    close_et = b.close_time.tz_convert(ET).strftime("%H:%M:%S")
    print(f"  {open_et}->{close_et}  O:{b.open:.2f} H:{b.high:.2f} L:{b.low:.2f} "
          f"C:{b.close:.2f}  range:{b.range_points:.2f}  vol:{b.total_vol}")

print()
print("-- Last 5 closed bars ---------------------------------------------------")
for b in closed[-5:]:
    open_et  = b.open_time.tz_convert(ET).strftime("%H:%M:%S")
    close_et = b.close_time.tz_convert(ET).strftime("%H:%M:%S")
    print(f"  {open_et}->{close_et}  O:{b.open:.2f} H:{b.high:.2f} L:{b.low:.2f} "
          f"C:{b.close:.2f}  range:{b.range_points:.2f}  vol:{b.total_vol}")

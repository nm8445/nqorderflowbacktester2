"""
Build pivot point cache from 5-minute bars.

For each trading day, detects pivot highs/lows on both 1-min and 5-min timeframes
using the detect_pivots function. Saves results as pickle files with left/right
length encoded in the filename for easy parameter sweeps.

Cache location: D:/trading_pythonbacktest_data/pivot_cache/
Filename format: {date}_L{left}_R{right}.pkl
"""

import gc
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pivot_detection import PivotLevel, detect_pivots

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
TIMEBARS_5MIN_DIR = DATA_DIR / "timebars_5min"
PIVOT_CACHE_DIR = DATA_DIR / "pivot_cache"

# Pivot detection settings
LEFT_LEN = 10
RIGHT_LEN = 10


def load_5min_bars_for_date(target_date: str) -> pd.DataFrame | None:
    """
    Load 5-min bars for the trading session on target_date.

    Each 5-min bar file already contains the full session
    (6pm prev day to ~5pm target day), so we only need the target date file.
    """
    target = datetime.strptime(target_date, "%Y-%m-%d").date()

    bar_file = TIMEBARS_5MIN_DIR / f"timebars_5min_{target.strftime('%Y_%m_%d')}.pkl"
    if not bar_file.exists():
        return None

    with open(bar_file, 'rb') as fp:
        bars = pickle.load(fp)

    if not bars:
        return None

    all_bars = []
    for bar in bars:
        all_bars.append({
            'timestamp': bar['open_time'],
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'buy_vol': bar.get('buy_vol', 0),
            'sell_vol': bar.get('sell_vol', 0),
        })

    df = pd.DataFrame(all_bars).set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]

    if df.empty:
        return None

    return df


def resample_to_1min(bars_5min: pd.DataFrame) -> pd.DataFrame:
    """
    We don't have raw 1-min bars, but we do have 5-min bars.
    Since we can't create true 1-min from 5-min, we use 5-min as-is
    and label this timeframe accordingly.

    If 1-min tick-built bars become available later, swap this out.
    """
    # For now, return the 5-min bars — pivot detection still works,
    # it just means "1-min pivots" are actually 5-min until we have 1-min data.
    # TODO: build true 1-min bars from DBN ticks
    return bars_5min


def process_single_date(target_date: str, left_len: int, right_len: int) -> dict:
    """Process a single date and save pivot cache."""
    try:
        print(f"Processing {target_date}...", end=" ", flush=True)

        PIVOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = PIVOT_CACHE_DIR / f"{target_date}_L{left_len}_R{right_len}.pkl"

        if cache_file.exists():
            print("already cached")
            return {"date": target_date, "status": "cached"}

        # Load 5-min bars for this session
        bars_5min = load_5min_bars_for_date(target_date)
        if bars_5min is None or len(bars_5min) < (left_len + right_len + 1):
            print("insufficient data")
            return {"date": target_date, "status": "insufficient_data"}

        # Detect pivots on 5-min bars
        pivots_5min = detect_pivots(bars_5min, left_len=left_len, right_len=right_len)

        # Convert to serializable format
        pivots_5min_data = [
            {
                "price": p.price,
                "bar_time": p.bar_time,
                "kind": p.kind,
            }
            for p in pivots_5min
        ]

        # Separate highs and lows for convenience
        pivot_highs_5min = [p for p in pivots_5min_data if p["kind"] == "high"]
        pivot_lows_5min = [p for p in pivots_5min_data if p["kind"] == "low"]

        cache_data = {
            "date": target_date,
            "left_len": left_len,
            "right_len": right_len,
            "pivots_5min": pivots_5min_data,
            "pivot_highs_5min": pivot_highs_5min,
            "pivot_lows_5min": pivot_lows_5min,
            "num_bars_5min": len(bars_5min),
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"OK | 5min: {len(pivot_highs_5min)}H {len(pivot_lows_5min)}L")

        return {"date": target_date, "status": "success"}

    except Exception as e:
        print(f"ERROR - {e}")
        return {"date": target_date, "status": "error", "error": str(e)}


def build_pivot_cache(
    start_date: str,
    end_date: str,
    left_len: int = LEFT_LEN,
    right_len: int = RIGHT_LEN,
):
    """Build pivot cache for date range."""

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(f"Building pivot cache for {len(dates)} dates")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Settings: left={left_len}, right={right_len}")
    print(f"Cache directory: {PIVOT_CACHE_DIR}")
    print()

    PIVOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for d in dates:
        results.append(process_single_date(d, left_len, right_len))
        gc.collect()

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    cached = sum(1 for r in results if r["status"] == "cached")
    errors = sum(1 for r in results if r["status"] == "error")
    insufficient = sum(1 for r in results if r["status"] == "insufficient_data")

    print()
    print("=" * 80)
    print("PIVOT CACHE BUILD COMPLETE")
    print("=" * 80)
    print(f"Total dates:       {len(dates)}")
    print(f"Already cached:    {cached}")
    print(f"Newly cached:      {success}")
    print(f"Insufficient data: {insufficient}")
    print(f"Errors:            {errors}")
    print(f"Settings:          L{left_len}_R{right_len}")
    print(f"Cache location:    {PIVOT_CACHE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    START_DATE = "2025-03-13"
    END_DATE = "2026-04-08"

    build_pivot_cache(
        start_date=START_DATE,
        end_date=END_DATE,
        left_len=LEFT_LEN,
        right_len=RIGHT_LEN,
    )

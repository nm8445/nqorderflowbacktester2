"""
Build VWAP reaction signal cache for full 5-year dataset.

Uses:
  - signal_cache_5yr (MarketTick Dec 2020 - Mar 2025) + signal_cache (Databento Apr 2025+)
  - vwap_cache_5yr (MarketTick) + vwap_cache (Databento)
  - timebars_5min_5yr (MarketTick) + timebars_5min (Databento)

Output: vwap_reaction_cache_5yr/

Usage:
    python -u scripts/cache_creation_scripts/vwap_creation/build_vwap_reaction_cache_5yr.py
"""
import gc
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
# Add the vwap_creation dir for the detection logic
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_vwap_reaction_cache import (
    detect_vwap_reactions, load_5min_bars_for_date,
    VWAP_ZONE_POINTS, ATR_PERIOD,
)

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")

# MarketTick 5yr caches (Dec 2020 - Mar 2025)
MT_SIGNAL_DIR = DATA_DIR / "signal_cache_5yr"
MT_VWAP_DIR = DATA_DIR / "vwap_cache_5yr"
MT_TIMEBARS_DIR = DATA_DIR / "timebars_5min_5yr"

# Databento caches (Mar 2025 - Apr 2026)
DB_SIGNAL_DIR = DATA_DIR / "signal_cache"
DB_VWAP_DIR = DATA_DIR / "vwap_cache"
DB_TIMEBARS_DIR = DATA_DIR / "timebars_5min"

# Output
OUTPUT_DIR = DATA_DIR / "vwap_reaction_cache_5yr"

# Cutoff: use Databento from this date onward (it's higher quality for overlap)
DB_PRIORITY_FROM = "2025-04-01"


def find_file(date_str, signal=False, vwap=False, timebars=False):
    """Find the best available file for a date, preferring Databento for recent dates."""
    use_db = date_str >= DB_PRIORITY_FROM

    if signal:
        fname = f"{date_str}.pkl"
        if use_db and (DB_SIGNAL_DIR / fname).exists():
            return DB_SIGNAL_DIR / fname
        if (MT_SIGNAL_DIR / fname).exists():
            return MT_SIGNAL_DIR / fname
        if (DB_SIGNAL_DIR / fname).exists():
            return DB_SIGNAL_DIR / fname
        return None

    if vwap:
        fname = f"{date_str}.pkl"
        if use_db and (DB_VWAP_DIR / fname).exists():
            return DB_VWAP_DIR / fname
        if (MT_VWAP_DIR / fname).exists():
            return MT_VWAP_DIR / fname
        if (DB_VWAP_DIR / fname).exists():
            return DB_VWAP_DIR / fname
        return None

    if timebars:
        fname = f"timebars_5min_{date_str.replace('-', '_')}.pkl"
        if use_db and (DB_TIMEBARS_DIR / fname).exists():
            return DB_TIMEBARS_DIR / fname
        if (MT_TIMEBARS_DIR / fname).exists():
            return MT_TIMEBARS_DIR / fname
        if (DB_TIMEBARS_DIR / fname).exists():
            return DB_TIMEBARS_DIR / fname
        return None


def load_5min_bars_from_file(tb_file):
    """Load 5-min bars from a specific file path and compute ATR."""
    with open(tb_file, 'rb') as f:
        bars = pickle.load(f)
    if not bars:
        return None

    rows = [{
        'timestamp': b['open_time'],
        'high': b['high'],
        'low': b['low'],
        'close': b['close'],
    } for b in bars]

    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    df.index = df.index + pd.Timedelta(minutes=5)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['atr'] = df['tr'].rolling(ATR_PERIOD, min_periods=1).mean()
    return df


def process_date(date_str):
    """Process a single date for VWAP reaction signals."""
    cache_file = OUTPUT_DIR / f"{date_str}.pkl"
    if cache_file.exists():
        return "cached"

    # Find all required files
    sig_file = find_file(date_str, signal=True)
    vwap_file = find_file(date_str, vwap=True)
    tb_file = find_file(date_str, timebars=True)

    if sig_file is None:
        return "no_signal"
    if vwap_file is None:
        return "no_vwap"

    # Load data
    with open(sig_file, 'rb') as f:
        signal_data = pickle.load(f)
    bars = signal_data["bars"]
    if not bars:
        return "empty_bars"

    with open(vwap_file, 'rb') as f:
        vwap_df = pickle.load(f)
    if vwap_df is None or (hasattr(vwap_df, 'empty') and vwap_df.empty):
        return "empty_vwap"

    # Load 5-min bars
    bars_5min = None
    if tb_file is not None:
        bars_5min = load_5min_bars_from_file(tb_file)

    # Detect signals
    signals = detect_vwap_reactions(bars, vwap_df, bars_5min=bars_5min)

    shorts = sum(1 for s in signals if s["direction"] == "short")
    longs = sum(1 for s in signals if s["direction"] == "long")

    cache_data = {
        "date": date_str,
        "signals": signals,
        "num_signals": len(signals),
        "num_shorts": shorts,
        "num_longs": longs,
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    return f"{shorts}S {longs}L"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all available dates from both signal caches
    all_dates = set()
    for f in MT_SIGNAL_DIR.glob("*.pkl"):
        all_dates.add(f.stem)
    for f in DB_SIGNAL_DIR.glob("*.pkl"):
        all_dates.add(f.stem)

    dates = sorted(all_dates)
    existing = len(list(OUTPUT_DIR.glob("*.pkl")))

    print(f"Total dates with signal cache: {len(dates)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Existing VWAP reaction cache: {existing}")
    print(f"DB priority from: {DB_PRIORITY_FROM}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    success = 0
    cached = 0
    no_vwap = 0
    errors = 0

    for i, d in enumerate(dates):
        result = process_date(d)
        if result == "cached":
            cached += 1
        elif result in ("no_signal", "no_vwap", "empty_bars", "empty_vwap"):
            no_vwap += 1
        elif "S" in result:
            success += 1
        else:
            errors += 1

        if (i + 1) % 50 == 0 or i < 5:
            print(f"  [{i+1}/{len(dates)}] {d}  {result}", flush=True)

        if (i + 1) % 200 == 0:
            gc.collect()

    total = len(list(OUTPUT_DIR.glob("*.pkl")))
    print()
    print("=" * 70)
    print("VWAP REACTION CACHE (5YR) BUILD COMPLETE")
    print("=" * 70)
    print(f"Newly cached:    {success}")
    print(f"Already cached:  {cached}")
    print(f"No VWAP/signal:  {no_vwap}")
    print(f"Errors:          {errors}")
    print(f"Total files:     {total}")
    print("=" * 70)


if __name__ == "__main__":
    main()

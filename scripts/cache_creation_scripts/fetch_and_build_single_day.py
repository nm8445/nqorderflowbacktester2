"""
Fetch DBN data from Databento for a single session and build all caches needed
for the combined VWAP backtest.

Pipeline:
  1. Download DBN files (need prev_day, target_day, next_day for session boundaries)
  2. Build signal_cache (40-range bars + absorption signals)
  3. Build vwap_cache (1-min session VWAP)
  4. Build timebars_5min (5-min OHLC bars)
  5. Build vwap_reaction_cache (VWAP reaction signals)

Usage:
  python fetch_and_build_single_day.py 2026-04-13
"""

import gc
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from nqbt.analysis.range_bars import build_range_bars
from nqbt.analysis.vwap import compute_vwap
from nqbt.data.normalizer import normalize

load_dotenv()

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
DBN_DIR = DATA_DIR / "dbn"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_5MIN_DIR = DATA_DIR / "timebars_5min"
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"

DATASET = "GLBX.MDP3"
SYMBOLS = ["NQ.c.0"]
SCHEMA = "mbp-1"
STYPE = "continuous"


def download_dbn(date_str: str):
    """Download single-day DBN file if not already present."""
    output_file = DBN_DIR / f"{date_str}.dbn"
    if output_file.exists():
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"  [OK] {date_str}.dbn already exists ({size_mb:.1f} MB)")
        return True

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        print("  ERROR: DATABENTO_API_KEY not found in .env")
        return False

    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    next_day = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"  [DL] {date_str}.dbn downloading...", end=" ", flush=True)
    try:
        client = db.Historical(api_key)
        client.timeseries.get_range(
            dataset=DATASET,
            symbols=SYMBOLS,
            schema=SCHEMA,
            start=date_str,
            end=next_day,
            stype_in=STYPE,
            path=str(output_file),
        )
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"OK ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def load_data_fast(start_date, end_date):
    """Load tick data from dbn files covering date range."""
    all_dfs = []
    current = start_date
    while current <= end_date:
        single_file = DBN_DIR / f"{current}.dbn"
        if single_file.exists():
            store = db.DBNStore.from_file(str(single_file))
            df = store.to_df()
            if not df.empty:
                all_dfs.append(df)
            del store
        current += timedelta(days=1)

    if all_dfs:
        combined = pd.concat(all_dfs)
        del all_dfs
        combined = combined[~combined.index.duplicated(keep='first')].sort_index()
        return normalize(combined)
    return None


def build_signal_cache(target_date):
    """Build signal cache (range bars) for target date."""
    from build_signal_cache import (
        build_range_bars, detect_enhanced_signals,
        build_refreshing_profiles, generate_entry_signals,
    )

    cache_file = SIGNAL_CACHE_DIR / f"{target_date}.pkl"
    if cache_file.exists():
        print(f"  [OK] signal_cache already exists")
        return True

    target = datetime.strptime(str(target_date), "%Y-%m-%d").date()
    prev_date = target - timedelta(days=1)
    next_date = target + timedelta(days=1)

    ticks = load_data_fast(prev_date, next_date)
    if ticks is None:
        print(f"  [FAIL] No tick data for signal_cache")
        return False

    session_ticks = ticks[ticks["session_date"] == target].copy()
    if session_ticks.empty:
        print(f"  [FAIL] No session ticks for {target_date}")
        return False

    bars = build_range_bars(session_ticks, range_ticks=40, ticks_per_level=5)
    if len(bars) < 10:
        print(f"  [FAIL] Only {len(bars)} bars (need >= 10)")
        return False

    all_signals = detect_enhanced_signals(bars, min_delta=30)
    profiles = build_refreshing_profiles(session_ticks, target, refresh_minutes=1)
    entry_signals = generate_entry_signals(all_signals, profiles, proximity_ticks=100,
                                           start_time_et="09:30", end_time_et="11:00")

    cache_data = {
        "date": target,
        "bars": bars,
        "all_signals": all_signals,
        "profiles": profiles,
        "entry_signals": entry_signals,
        "num_bars": len(bars),
        "num_all_signals": len(all_signals),
        "num_entry_signals": len(entry_signals),
    }

    SIGNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"  [OK] signal_cache: {len(bars)} bars, {len(all_signals)} signals")
    return True


def build_vwap_cache(target_date):
    """Build 1-min VWAP cache for target date."""
    cache_file = VWAP_CACHE_DIR / f"{target_date}.pkl"
    if cache_file.exists():
        print(f"  [OK] vwap_cache already exists")
        return True

    target = datetime.strptime(str(target_date), "%Y-%m-%d").date()
    prev_date = target - timedelta(days=1)
    next_date = target + timedelta(days=1)

    ticks = load_data_fast(prev_date, next_date)
    if ticks is None:
        print(f"  [FAIL] No tick data for vwap_cache")
        return False

    session_ticks = ticks[ticks["session_date"] == target].copy()
    if session_ticks.empty:
        print(f"  [FAIL] No session ticks")
        return False

    vwap_df = compute_vwap(session_ticks)
    if vwap_df.empty:
        print(f"  [FAIL] Empty VWAP")
        return False

    vwap_1min = vwap_df.resample("1min").last().ffill()

    VWAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(vwap_1min, f)

    first_vwap = vwap_1min["vwap"].iloc[0]
    last_vwap = vwap_1min["vwap"].iloc[-1]
    print(f"  [OK] vwap_cache: {len(vwap_1min)} minutes | VWAP {first_vwap:.0f} -> {last_vwap:.0f}")
    return True


def build_timebars_5min(target_date):
    """Build 5-min OHLC bars from tick data."""
    target = datetime.strptime(str(target_date), "%Y-%m-%d").date()
    date_fmt = target.strftime("%Y_%m_%d")
    output_file = TIMEBARS_5MIN_DIR / f"timebars_5min_{date_fmt}.pkl"

    if output_file.exists():
        print(f"  [OK] timebars_5min already exists")
        return True

    prev_date = target - timedelta(days=1)
    next_date = target + timedelta(days=1)

    ticks = load_data_fast(prev_date, next_date)
    if ticks is None:
        print(f"  [FAIL] No tick data for timebars_5min")
        return False

    session_ticks = ticks[ticks["session_date"] == target].copy()
    if session_ticks.empty:
        print(f"  [FAIL] No session ticks")
        return False

    # Data is already trade-filtered by normalize()
    trades = session_ticks.copy()
    if trades.empty:
        print(f"  [FAIL] No trades in session")
        return False

    # Convert index to ET for resampling
    if trades.index.tz is None:
        trades.index = trades.index.tz_localize("UTC")
    trades_et = trades.copy()
    trades_et.index = trades_et.index.tz_convert(ET)

    # Resample to 5-min OHLC
    ohlc = trades_et["price"].resample("5min").ohlc()
    vol = trades_et["size"].resample("5min").sum()
    ohlc["volume"] = vol
    ohlc = ohlc.dropna()

    if ohlc.empty:
        print(f"  [FAIL] No 5min bars")
        return False

    bars_list = []
    for ts, row in ohlc.iterrows():
        bars_list.append({
            "open_time": ts,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": int(row["volume"]),
        })

    TIMEBARS_5MIN_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(bars_list, f)

    print(f"  [OK] timebars_5min: {len(bars_list)} bars")
    return True


def build_vwap_reaction_cache(target_date):
    """Build VWAP reaction signal cache."""
    sys.path.insert(0, str(Path(__file__).parent / "vwap_creation"))
    from vwap_creation.build_vwap_reaction_cache import (
        detect_vwap_reactions, load_5min_bars_for_date,
    )

    cache_file = VWAP_REACTION_CACHE_DIR / f"{target_date}.pkl"
    if cache_file.exists():
        print(f"  [OK] vwap_reaction_cache already exists")
        return True

    # Load signal cache (range bars)
    signal_file = SIGNAL_CACHE_DIR / f"{target_date}.pkl"
    if not signal_file.exists():
        print(f"  [FAIL] No signal cache")
        return False
    with open(signal_file, 'rb') as f:
        signal_data = pickle.load(f)
    bars = signal_data["bars"]

    # Load VWAP cache
    vwap_file = VWAP_CACHE_DIR / f"{target_date}.pkl"
    if not vwap_file.exists():
        print(f"  [FAIL] No VWAP cache")
        return False
    with open(vwap_file, 'rb') as f:
        vwap_df = pickle.load(f)

    # Load 5-min bars
    bars_5min = load_5min_bars_for_date(target_date)

    # Detect signals
    signals = detect_vwap_reactions(bars, vwap_df, bars_5min=bars_5min)

    shorts = sum(1 for s in signals if s["direction"] == "short")
    longs = sum(1 for s in signals if s["direction"] == "long")

    cache_data = {
        "date": target_date,
        "signals": signals,
        "num_signals": len(signals),
        "num_shorts": shorts,
        "num_longs": longs,
    }

    VWAP_REACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"  [OK] vwap_reaction_cache: {shorts}S {longs}L ({len(signals)} total)")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_and_build_single_day.py YYYY-MM-DD")
        sys.exit(1)

    target_date = sys.argv[1]
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    prev_date = target - timedelta(days=1)
    next_date = target + timedelta(days=1)

    print("=" * 80)
    print(f"FETCH & BUILD CACHES FOR SESSION: {target_date}")
    print(f"Need DBN data: {prev_date} to {next_date}")
    print("=" * 80)
    print()

    # Step 1: Download DBN files
    print("Step 1: Download DBN data")
    DBN_DIR.mkdir(parents=True, exist_ok=True)
    dates_needed = [prev_date, target, next_date]
    for d in dates_needed:
        d_str = d.strftime("%Y-%m-%d")
        download_dbn(d_str)
    print()

    # Step 2: Build signal cache (range bars)
    print("Step 2: Build signal cache (range bars)")
    if not build_signal_cache(target_date):
        print("  WARNING: signal cache failed, downstream caches will fail")
    gc.collect()
    print()

    # Step 3: Build VWAP cache
    print("Step 3: Build VWAP cache (1-min)")
    build_vwap_cache(target_date)
    gc.collect()
    print()

    # Step 4: Build 5-min timebars
    print("Step 4: Build 5-min timebars")
    build_timebars_5min(target_date)
    gc.collect()
    print()

    # Step 5: Build VWAP reaction cache
    print("Step 5: Build VWAP reaction cache")
    build_vwap_reaction_cache(target_date)
    gc.collect()
    print()

    print("=" * 80)
    print("ALL CACHES BUILT")
    print("=" * 80)


if __name__ == "__main__":
    main()

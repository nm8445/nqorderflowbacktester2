"""
Build daily volume profile cache with EOD metrics.

Creates one profile per day from 6pm (prev day) to 4:59:59pm (current day).
Saves:
- VAH, VAL, POC (end of day values)
- Total delta in zones:
  - 10 points above and below POC
  - Above and below VAH
  - Above and below VAL

Cache location: D:/trading_pythonbacktest_data/daily_profile_cache/
"""

from datetime import datetime, timedelta
import gc
import pandas as pd
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from nqbt.analysis.volume_profile import VolumeProfile
from nqbt.data.normalizer import normalize
import databento as db

TICK_SIZE = 0.25
ET = "America/New_York"
CACHE_DIR = Path("D:/trading_pythonbacktest_data")
DAILY_PROFILE_CACHE_DIR = CACHE_DIR / "daily_profile_cache"


def calculate_delta_zones(profile: VolumeProfile, zone_size_points: float = 10.0):
    """
    Calculate total delta in zones around key levels.

    Args:
        profile: VolumeProfile object
        zone_size_points: Size of zone in points (default 10.0)

    Returns:
        dict with delta zone metrics
    """
    if not profile or not profile.levels:
        return {
            "poc_zone_above_delta": 0,
            "poc_zone_below_delta": 0,
            "vah_zone_above_delta": 0,
            "vah_zone_below_delta": 0,
            "val_zone_above_delta": 0,
            "val_zone_below_delta": 0,
        }

    poc = profile.poc
    vah = profile.vah
    val = profile.val

    # POC zones: 10 points above and below
    poc_upper = poc + zone_size_points
    poc_lower = poc - zone_size_points

    poc_zone_above_delta = 0
    poc_zone_below_delta = 0

    for price, level in profile.levels.items():
        if poc < price <= poc_upper:
            poc_zone_above_delta += level.delta
        elif poc_lower <= price < poc:
            poc_zone_below_delta += level.delta

    # VAH zones: above and below
    vah_zone_above_delta = 0
    vah_zone_below_delta = 0

    for price, level in profile.levels.items():
        if price > vah:
            vah_zone_above_delta += level.delta
        elif price < vah:
            vah_zone_below_delta += level.delta

    # VAL zones: above and below
    val_zone_above_delta = 0
    val_zone_below_delta = 0

    for price, level in profile.levels.items():
        if price > val:
            val_zone_above_delta += level.delta
        elif price < val:
            val_zone_below_delta += level.delta

    return {
        "poc_zone_above_delta": poc_zone_above_delta,
        "poc_zone_below_delta": poc_zone_below_delta,
        "vah_zone_above_delta": vah_zone_above_delta,
        "vah_zone_below_delta": vah_zone_below_delta,
        "val_zone_above_delta": val_zone_above_delta,
        "val_zone_below_delta": val_zone_below_delta,
    }


def build_daily_profile(session_ticks, current_date):
    """
    Build a single volume profile for the entire session.
    Session: 6pm (prev day) to 4:59:59pm (current day)

    Args:
        session_ticks: DataFrame with tick data
        current_date: Date string (YYYY-MM-DD)

    Returns:
        dict with profile data
    """
    if session_ticks.empty or len(session_ticks) < 100:
        return None

    # Build single profile for entire session
    profile = VolumeProfile.build(session_ticks, ticks_per_level=10)

    if not profile:
        return None

    # Calculate delta zones
    delta_zones = calculate_delta_zones(profile, zone_size_points=10.0)

    # Extract EOD metrics
    prev_day = datetime.strptime(current_date, "%Y-%m-%d").date() - timedelta(days=1)
    current_day = datetime.strptime(current_date, "%Y-%m-%d").date()

    profile_data = {
        "date": current_date,
        "session_start": pd.Timestamp(f"{prev_day} 18:00:00", tz=ET),
        "session_end": pd.Timestamp(f"{current_day} 16:59:59", tz=ET),
        "vah": profile.vah,
        "val": profile.val,
        "poc": profile.poc,
        "total_volume": sum(lv.total_vol for lv in profile.levels.values()),
        "total_delta": sum(lv.delta for lv in profile.levels.values()),
        "num_ticks": len(session_ticks),
        **delta_zones  # Add all delta zone metrics
    }

    return profile_data


def find_dbn_files_for_date(current_date: str, dbn_dir: Path) -> list:
    """
    Find ALL DBN files whose date range covers the target date.

    Handles naming patterns:
      - Simple: 2026-03-30.dbn
      - Range:  NQ_c_0_mbp-1_2025-03-18_2025-03-20.dbn
      - Full-day range: NQ_mbp1_2025-03-18_to_2025-03-20.dbn

    A file's range covers the target if start <= target <= end.
    We also include files where start == target-1 (previous day),
    since the session starts at 6pm the previous calendar day.
    """
    import re
    target = datetime.strptime(current_date, "%Y-%m-%d").date()
    prev_day = target - timedelta(days=1)
    matching = []

    for dbn_file in dbn_dir.glob("*.dbn"):
        filename = dbn_file.stem  # without .dbn

        # Pattern 1: Simple date (2026-03-30)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", filename):
            file_date = datetime.strptime(filename, "%Y-%m-%d").date()
            if file_date == target or file_date == prev_day:
                matching.append(dbn_file)
            continue

        # Pattern 2: Extract all dates from filename
        dates_found = re.findall(r"\d{4}-\d{2}-\d{2}", filename)
        if len(dates_found) >= 2:
            file_start = datetime.strptime(dates_found[0], "%Y-%m-%d").date()
            file_end = datetime.strptime(dates_found[-1], "%Y-%m-%d").date()
            # Include if target or prev_day falls within the file's range
            if file_start <= target <= file_end or file_start <= prev_day <= file_end:
                matching.append(dbn_file)

    return sorted(matching)


def process_single_date(current_date: str) -> dict:
    """Process a single date and save to cache."""
    try:
        print(f"Processing {current_date}...", end=" ", flush=True)

        # Check if already cached
        DAILY_PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = DAILY_PROFILE_CACHE_DIR / f"{current_date}.pkl"

        if cache_file.exists():
            print("already cached")
            return {"date": current_date, "status": "cached", "profile": None}

        # Find all DBN files that may contain data for this date
        dbn_dir = CACHE_DIR / "dbn"
        dbn_files = find_dbn_files_for_date(current_date, dbn_dir)

        if not dbn_files:
            print("no DBN files found")
            return {"date": current_date, "status": "no_dbn_file", "profile": None}

        # Pre-compute session time window for early filtering
        prev_day = datetime.strptime(current_date, "%Y-%m-%d").date() - timedelta(days=1)
        current_day = datetime.strptime(current_date, "%Y-%m-%d").date()
        session_start_utc = pd.Timestamp(f"{prev_day} 18:00:00", tz=ET).tz_convert("UTC")
        session_end_utc = pd.Timestamp(f"{current_day} 16:59:59", tz=ET).tz_convert("UTC")

        # Load DBN files one at a time, time-filter immediately to limit memory
        all_dfs = []
        for dbn_file in dbn_files:
            try:
                data = db.DBNStore.from_file(str(dbn_file))
                df = data.to_df()
                del data  # free raw DBN data immediately
                if not df.empty:
                    # Time-filter early to discard out-of-session rows
                    df = df[(df.index >= session_start_utc) & (df.index <= session_end_utc)]
                    if not df.empty:
                        all_dfs.append(df)
                    else:
                        del df
                else:
                    del df
            except Exception:
                continue

        if not all_dfs:
            print("no data in DBN files")
            return {"date": current_date, "status": "no_data", "profile": None}

        # Concatenate, deduplicate by index, sort
        df = pd.concat(all_dfs)
        del all_dfs
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Normalize (filters to trades with valid aggressor: buy/sell)
        session_ticks = normalize(df)
        del df
        gc.collect()

        # Build profile
        profile_data = build_daily_profile(session_ticks, current_date)
        del session_ticks
        gc.collect()

        if not profile_data:
            print("insufficient data")
            return {"date": current_date, "status": "insufficient_data", "profile": None}

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(profile_data, f)

        print(f"OK | VAH:{profile_data['vah']:.0f} POC:{profile_data['poc']:.0f} VAL:{profile_data['val']:.0f} delta:{profile_data['total_delta']:,}")

        return {
            "date": current_date,
            "status": "success",
            "profile": profile_data
        }

    except Exception as e:
        print(f"ERROR - {e}")
        return {"date": current_date, "status": "error", "error": str(e), "profile": None}


def build_all_daily_profiles(start_date: str, end_date: str, max_workers: int = 4):
    """
    Build daily profile cache for date range.

    Loads from existing DBN files in D:/trading_pythonbacktest_data/dbn/

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_workers: Max parallel workers (default 4, keep low to limit RAM)
    """

    # Generate date range
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(f"Building daily profile cache for {len(dates)} dates")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Cache directory: {DAILY_PROFILE_CACHE_DIR}")
    print(f"Workers: {max_workers}")
    print()

    # Create cache directory
    DAILY_PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Process in batches to limit concurrent RAM usage
    # Each batch runs max_workers dates in parallel, then we wait for completion
    results = []
    for i in range(0, len(dates), max_workers):
        batch = dates[i:i + max_workers]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(process_single_date, batch))
        results.extend(batch_results)
        gc.collect()

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    cached = sum(1 for r in results if r["status"] == "cached")
    errors = sum(1 for r in results if r["status"] == "error")
    no_data = sum(1 for r in results if r["status"] == "no_data")
    no_dbn = sum(1 for r in results if r["status"] == "no_dbn_file")

    print()
    print("="*80)
    print("DAILY PROFILE CACHE BUILD COMPLETE")
    print("="*80)
    print(f"Total dates:       {len(dates)}")
    print(f"Already cached:    {cached}")
    print(f"Newly cached:      {success}")
    print(f"No DBN file:       {no_dbn}")
    print(f"No data:           {no_data}")
    print(f"Errors:            {errors}")
    print()
    print(f"Cache location: {DAILY_PROFILE_CACHE_DIR}")
    print("="*80)


if __name__ == "__main__":
    # Build cache for all available data
    # Adjust dates as needed
    START_DATE = "2026-03-04"
    END_DATE = "2026-04-08"

    build_all_daily_profiles(
        start_date=START_DATE,
        end_date=END_DATE,
        max_workers=4,
    )

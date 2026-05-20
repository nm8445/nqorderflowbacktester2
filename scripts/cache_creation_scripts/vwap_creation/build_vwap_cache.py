"""
Build per-day session VWAP cache at 1-minute resolution.

Session VWAP anchors at 6pm ET previous day and runs through 4:59pm ET.
Loads raw ticks via load_data_fast(), filters to session_date, computes
cumulative VWAP tick-by-tick, then resamples to 1-min for fast lookups.

Output: D:/trading_pythonbacktest_data/vwap_cache/{YYYY-MM-DD}.pkl
  Each file contains a DataFrame indexed by UTC timestamp (1-min freq)
  with columns: vwap, std, std1_upper, std1_lower, std2_upper, std2_lower,
  std3_upper, std3_lower.
"""

import gc
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from nqbt.analysis.vwap import compute_vwap
from nqbt.data.normalizer import normalize
import databento as db

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
VWAP_CACHE_DIR = DATA_DIR / "vwap_cache"


def load_data_fast(start_date, end_date):
    """Load data from parquet or dbn (same as build_signal_cache.py)."""
    parquet_file = DATA_DIR / "parquet" / f"NQ_c_0_mbp-1_{start_date}_{end_date}.parquet"
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        return normalize(df)

    dbn_file = DATA_DIR / "dbn" / f"NQ_c_0_mbp-1_{start_date}_{end_date}.dbn"
    if dbn_file.exists():
        store = db.DBNStore.from_file(str(dbn_file))
        df = store.to_df()
        return normalize(df)

    # Fallback: individual single-day dbn files
    all_dfs = []
    current = start_date
    while current <= end_date:
        single_file = DATA_DIR / "dbn" / f"{current}.dbn"
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


def process_single_date(target_date: str) -> dict:
    """Build VWAP cache for a single session date."""
    try:
        print(f"Processing {target_date}...", end=" ", flush=True)

        VWAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = VWAP_CACHE_DIR / f"{target_date}.pkl"

        if cache_file.exists():
            print("already cached")
            return {"date": target_date, "status": "cached"}

        target = datetime.strptime(target_date, "%Y-%m-%d").date()
        prev_date = target - timedelta(days=1)
        next_date = target + timedelta(days=1)

        ticks = load_data_fast(prev_date, next_date)
        if ticks is None:
            print("no data")
            return {"date": target_date, "status": "no_data"}

        # Filter to this session date
        session_ticks = ticks[ticks["session_date"] == target].copy()
        if session_ticks.empty:
            print("no session ticks")
            return {"date": target_date, "status": "no_data"}

        # Compute tick-by-tick VWAP
        vwap_df = compute_vwap(session_ticks)

        if vwap_df.empty:
            print("empty VWAP")
            return {"date": target_date, "status": "empty_vwap"}

        # Resample to 1-min resolution (forward-fill to cover gaps)
        vwap_1min = vwap_df.resample("1min").last().ffill()

        # Save
        with open(cache_file, 'wb') as f:
            pickle.dump(vwap_1min, f)

        first_vwap = vwap_1min["vwap"].iloc[0]
        last_vwap = vwap_1min["vwap"].iloc[-1]
        n_minutes = len(vwap_1min)
        print(f"OK | {n_minutes} minutes | VWAP {first_vwap:.0f} -> {last_vwap:.0f}")

        del ticks, session_ticks, vwap_df, vwap_1min
        return {"date": target_date, "status": "success"}

    except Exception as e:
        print(f"ERROR - {e}")
        return {"date": target_date, "status": "error", "error": str(e)}


def build_vwap_cache(start_date: str, end_date: str):
    """Build VWAP cache for date range."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(f"Building VWAP cache for {len(dates)} dates")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Cache directory: {VWAP_CACHE_DIR}")
    print()

    VWAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for d in dates:
        results.append(process_single_date(d))
        gc.collect()

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    cached = sum(1 for r in results if r["status"] == "cached")
    errors = sum(1 for r in results if r["status"] == "error")
    no_data = sum(1 for r in results if r["status"] == "no_data")

    print()
    print("=" * 80)
    print("VWAP CACHE BUILD COMPLETE")
    print("=" * 80)
    print(f"Total dates:       {len(dates)}")
    print(f"Already cached:    {cached}")
    print(f"Newly cached:      {success}")
    print(f"No data:           {no_data}")
    print(f"Errors:            {errors}")
    print(f"Cache location:    {VWAP_CACHE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    START_DATE = "2025-03-13"
    END_DATE = "2026-04-08"

    build_vwap_cache(
        start_date=START_DATE,
        end_date=END_DATE,
    )

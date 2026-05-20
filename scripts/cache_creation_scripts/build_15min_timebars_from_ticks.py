"""
Build 15-minute time bars directly from tick data (not from resampled 5-min bars).

This will create proper OHLC bars with accurate volume, which affects:
- Volatility calculations (different closes = different returns)
- Trade count (more accurate bar structure)
- ATR calculations (based on true high/low range)
"""

import pickle
from pathlib import Path
import pandas as pd
import databento as db
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

# Paths
DATA_DIR = Path("D:/trading_pythonbacktest_data")
DBN_DIR = DATA_DIR / "dbn"
OUTPUT_DIR = DATA_DIR / "timebars_15min_from_ticks"

ET = "America/New_York"
TICK_SIZE = 0.25


def build_15min_bars_from_dbn(dbn_file: Path) -> dict:
    """Build 15-min bars from a single DBN file."""
    try:
        # Extract date from filename
        filename = dbn_file.name

        # Handle both naming patterns
        if filename.startswith("NQ_c_0"):
            # Format: NQ_c_0_mbp-1_2025-03-13_2025-03-15.dbn
            parts = filename.replace(".dbn", "").split("_")
            date_str = parts[3]  # Start date
        else:
            # Format: 2026-03-30.dbn
            date_str = filename.replace(".dbn", "")

        print(f"Processing {date_str}...", end=" ", flush=True)

        # Load DBN file
        data = db.DBNStore.from_file(str(dbn_file))
        df = data.to_df()

        if df.empty:
            print("no data")
            return {"date": date_str, "status": "no_data", "bars": []}

        # Filter to trades only (action == 'T')
        trades = df[df["action"] == "T"].copy()

        if trades.empty:
            print("no trades")
            return {"date": date_str, "status": "no_trades", "bars": []}

        # Convert to ET
        if trades.index.tz is None:
            trades.index = trades.index.tz_localize('UTC')
        trades.index = trades.index.tz_convert(ET)

        # Normalize prices (divide by 1e9 if needed)
        if 'price' in trades.columns and trades['price'].dtype == 'int64':
            trades['price'] = trades['price'] / 1e9

        # Create 15-min OHLC bars
        bars_15min = trades['price'].resample('15min').ohlc()
        volume_15min = trades['size'].resample('15min').sum()

        # Combine OHLC and volume
        bars_15min['volume'] = volume_15min
        bars_15min = bars_15min.dropna()

        if bars_15min.empty:
            print("no bars after resample")
            return {"date": date_str, "status": "no_bars", "bars": []}

        # Convert to list of dicts
        bars_list = []
        for ts, row in bars_15min.iterrows():
            bars_list.append({
                'open_time': ts,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume'])
            })

        print(f"OK ({len(bars_list)} bars)")
        return {"date": date_str, "status": "success", "bars": bars_list}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"date": "unknown", "status": "error", "bars": []}


def build_all_15min_bars(parallel: bool = True):
    """Build 15-min bars from all DBN files."""

    # Get all DBN files
    dbn_files = sorted(DBN_DIR.glob("*.dbn"))

    if not dbn_files:
        raise FileNotFoundError(f"No DBN files in {DBN_DIR}")

    print("="*80)
    print("BUILDING 15-MIN TIME BARS FROM TICK DATA")
    print("="*80)
    print(f"DBN files found: {len(dbn_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process files
    if parallel:
        num_workers = min(8, max(1, cpu_count() - 2))  # Max 8 workers
        print(f"Using {num_workers} workers")
        print()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(build_15min_bars_from_dbn, dbn_files))
    else:
        results = [build_15min_bars_from_dbn(f) for f in dbn_files]

    # Save results by date
    print()
    print("Saving bars by date...")

    saved_count = 0
    for result in results:
        if result["status"] == "success" and result["bars"]:
            date_str = result["date"]
            bars_list = result["bars"]

            # Group bars by date (some DBN files span multiple days)
            bars_by_date = {}
            for bar in bars_list:
                bar_date = bar['open_time'].date()
                if bar_date not in bars_by_date:
                    bars_by_date[bar_date] = []
                bars_by_date[bar_date].append(bar)

            # Save each date (merge with existing if file exists)
            for bar_date, date_bars in bars_by_date.items():
                date_str_formatted = str(bar_date).replace('-', '_')
                output_file = OUTPUT_DIR / f"timebars_15min_{date_str_formatted}.pkl"

                # Load existing bars if file exists
                if output_file.exists():
                    with open(output_file, 'rb') as f:
                        existing_bars = pickle.load(f)

                    # Merge: combine existing and new bars, remove duplicates by timestamp
                    all_bars = existing_bars + date_bars
                    # Deduplicate by open_time
                    seen_times = set()
                    merged_bars = []
                    for bar in all_bars:
                        bar_time = bar['open_time']
                        if bar_time not in seen_times:
                            seen_times.add(bar_time)
                            merged_bars.append(bar)
                    # Sort by time
                    merged_bars.sort(key=lambda b: b['open_time'])
                    date_bars = merged_bars

                with open(output_file, 'wb') as f:
                    pickle.dump(date_bars, f)

                saved_count += 1

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    no_data = sum(1 for r in results if r["status"] == "no_data")
    no_trades = sum(1 for r in results if r["status"] == "no_trades")
    errors = sum(1 for r in results if r["status"] == "error")

    print()
    print("="*80)
    print("15-MIN TIMEBAR BUILD COMPLETE")
    print("="*80)
    print(f"Total DBN files:   {len(dbn_files)}")
    print(f"Successful:        {success}")
    print(f"Date files saved:  {saved_count}")
    print(f"No data:           {no_data}")
    print(f"No trades:         {no_trades}")
    print(f"Errors:            {errors}")
    print()
    print(f"Output directory:  {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    build_all_15min_bars(parallel=True)

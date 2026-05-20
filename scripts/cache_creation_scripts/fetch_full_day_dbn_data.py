"""
Fetch full 24-hour DBN data from Databento for NQ futures.

This will download MBP-1 data for complete trading days (not just overnight).
"""

import databento as db
from pathlib import Path
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path("D:/trading_pythonbacktest_data")
DBN_DIR = DATA_DIR / "dbn_full_day"
SYMBOL = "NQ.c.0"  # Continuous front month
DATASET = "GLBX.MDP3"
SCHEMA = "mbp-1"

# Date range to fetch (adjust as needed)
START_DATE = "2024-10-31"
END_DATE = "2026-04-07"

# Create output directory
DBN_DIR.mkdir(parents=True, exist_ok=True)


def fetch_dbn_data():
    """Fetch DBN data from Databento."""

    # Get API key
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not found in environment")

    # Create client
    client = db.Historical(api_key)

    print("="*80)
    print("FETCHING FULL 24-HOUR DBN DATA FROM DATABENTO")
    print("="*80)
    print(f"Symbol: {SYMBOL}")
    print(f"Dataset: {DATASET}")
    print(f"Schema: {SCHEMA}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output directory: {DBN_DIR}")
    print()

    # Parse dates
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")

    # Download data in chunks (daily or weekly to avoid timeouts)
    current_date = start_dt
    chunk_days = 7  # Download 1 week at a time

    while current_date <= end_dt:
        chunk_end = min(current_date + timedelta(days=chunk_days - 1), end_dt)

        start_str = current_date.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        output_file = DBN_DIR / f"NQ_mbp1_{start_str}_to_{end_str}.dbn"

        if output_file.exists():
            print(f"SKIP {start_str} to {end_str} (already exists)")
        else:
            print(f"Fetching {start_str} to {end_str}...", end=" ", flush=True)

            try:
                # Fetch data for full days (no time restriction)
                client.timeseries.get_range(
                    dataset=DATASET,
                    symbols=[SYMBOL],
                    schema=SCHEMA,
                    start=start_str,
                    end=end_str,
                    path=str(output_file),
                    stype_in="continuous"
                )

                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"OK ({file_size_mb:.1f} MB)")

            except Exception as e:
                print(f"ERROR: {e}")

        # Move to next chunk
        current_date = chunk_end + timedelta(days=1)

    print()
    print("="*80)
    print("FETCH COMPLETE")
    print("="*80)

    # List all downloaded files
    dbn_files = sorted(DBN_DIR.glob("*.dbn"))
    total_size = sum(f.stat().st_size for f in dbn_files)
    total_size_gb = total_size / (1024 * 1024 * 1024)

    print(f"Total DBN files: {len(dbn_files)}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print(f"Output directory: {DBN_DIR}")
    print()
    print("Next step: Run build_15min_timebars_from_ticks.py with new DBN directory")
    print("="*80)


if __name__ == "__main__":
    fetch_dbn_data()

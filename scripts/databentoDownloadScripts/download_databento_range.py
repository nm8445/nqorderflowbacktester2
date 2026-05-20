"""
Download DBN files from Databento for a date range.
"""

from datetime import datetime, timedelta
from pathlib import Path
import os
from dotenv import load_dotenv
import databento as db

# Load environment
load_dotenv()
api_key = os.getenv("DATABENTO_API_KEY")
if not api_key:
    raise ValueError("DATABENTO_API_KEY not found in .env")

# Configuration
DBN_DIR = Path("D:/trading_pythonbacktest_data/dbn")
DATASET = "GLBX.MDP3"
SYMBOLS = ["NQ.c.0"]
SCHEMA = "mbp-1"
STYPE = "continuous"

# Date range
START_DATE = "2026-05-07"
END_DATE = "2026-05-11"


def download_dbn_files(start_date: str, end_date: str):
    """Download DBN files for each date in range."""

    # Create directory if needed
    DBN_DIR.mkdir(parents=True, exist_ok=True)

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Generate date list
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    print("="*80)
    print("DATABENTO DBN DOWNLOAD")
    print("="*80)
    print(f"Dataset:  {DATASET}")
    print(f"Schema:   {SCHEMA}")
    print(f"Symbols:  {SYMBOLS}")
    print(f"Range:    {start_date} to {end_date}")
    print(f"Dates:    {len(dates)}")
    print(f"Output:   {DBN_DIR}")
    print("="*80)
    print()

    # Create client
    client = db.Historical(api_key)

    # Download each date
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        output_file = DBN_DIR / f"{date_str}.dbn"

        # Skip if already exists
        if output_file.exists():
            print(f"[OK] {date_str}: already exists ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")
            continue

        try:
            print(f"[DL] {date_str}: downloading...", end=" ", flush=True)

            # Download for this date
            # We need to include next day to get full session (6pm current day to next day close)
            next_day = (date + timedelta(days=1)).strftime("%Y-%m-%d")

            # Download and save directly to DBN file
            client.timeseries.get_range(
                dataset=DATASET,
                symbols=SYMBOLS,
                schema=SCHEMA,
                start=date_str,
                end=next_day,
                stype_in=STYPE,
                path=str(output_file)  # Save directly to file
            )

            file_size = output_file.stat().st_size / 1024 / 1024
            print(f"OK ({file_size:.1f} MB)")

        except Exception as e:
            print(f"ERROR: {e}")

    print()
    print("="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)

    # Summary
    dbn_files = list(DBN_DIR.glob("*.dbn"))
    total_size = sum(f.stat().st_size for f in dbn_files) / 1024 / 1024

    print(f"Total DBN files: {len(dbn_files)}")
    print(f"Total size:      {total_size:.1f} MB")
    print(f"Location:        {DBN_DIR}")
    print("="*80)


if __name__ == "__main__":
    download_dbn_files(START_DATE, END_DATE)

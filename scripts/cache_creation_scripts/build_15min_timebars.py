"""
Build 15-minute time bars from existing 5-minute bars.

Resamples 5-min bars to 15-min for swing/intraday strategy.
"""

import pickle
from pathlib import Path
import pandas as pd

# Paths
DATA_DIR = Path("D:/trading_pythonbacktest_data")
INPUT_DIR = DATA_DIR / "timebars_5min"
OUTPUT_DIR = DATA_DIR / "timebars_15min"

ET = "America/New_York"


def build_15min_bars():
    """Build 15-minute bars from 5-minute bars."""

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all 5-min bar files
    timebar_files = sorted(INPUT_DIR.glob("*.pkl"))

    if not timebar_files:
        raise FileNotFoundError(f"No 5-min bars in {INPUT_DIR}")

    print(f"Loading {len(timebar_files)} 5-min bar cache files...")
    all_bars = []

    for f in timebar_files:
        with open(f, 'rb') as fp:
            time_bars = pickle.load(fp)

        for bar in time_bars:
            all_bars.append({
                'timestamp': bar['open_time'],
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar.get('volume', 0)
            })

    df = pd.DataFrame(all_bars)
    df = df.set_index('timestamp').sort_index()

    # Convert to ET
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(ET)

    print(f"  Loaded {len(df):,} 5-min bars from {df.index[0]} to {df.index[-1]}")
    print()

    # Resample to 15-min bars
    print("Resampling to 15-minute bars...")

    df_15min = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"  Created {len(df_15min):,} 15-min bars")
    print()

    # Group by date and save
    print("Saving 15-min bars by date...")

    dates = df_15min.index.date
    unique_dates = sorted(set(dates))

    saved_count = 0
    for date in unique_dates:
        date_bars = df_15min[df_15min.index.date == date]

        if date_bars.empty:
            continue

        # Convert to list of dicts for pickle
        bars_list = []
        for ts, row in date_bars.iterrows():
            bars_list.append({
                'open_time': ts,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume'])
            })

        # Save to file
        date_str = str(date).replace('-', '_')
        output_file = OUTPUT_DIR / f"timebars_15min_{date_str}.pkl"

        with open(output_file, 'wb') as f:
            pickle.dump(bars_list, f)

        saved_count += 1
        if saved_count % 50 == 0:
            print(f"  Saved {saved_count}/{len(unique_dates)} dates...")

    print()
    print("="*80)
    print("15-MINUTE TIMEBAR BUILD COMPLETE")
    print("="*80)
    print(f"Total 15-min bars: {len(df_15min):,}")
    print(f"Date files saved:  {saved_count}")
    print(f"Output directory:  {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("BUILDING 15-MINUTE TIME BARS FROM 5-MINUTE BARS")
    print("="*80)
    print()

    build_15min_bars()

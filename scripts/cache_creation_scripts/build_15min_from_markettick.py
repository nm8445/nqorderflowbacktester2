"""
Build 15-min OHLC bar cache from MarketTick L2 zip files.
Streams from zip -> filters Type=2 (trades) -> builds 15-min bars per CSV -> saves parquet.
Memory-efficient: processes one CSV at a time, never holds raw trades.

Source: D:/trading_pythonbacktest_data/NQ L2 data/*.zip
Output: D:/trading_pythonbacktest_data/markettick_15min_bars.parquet
"""
import zipfile
import io
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time
import sys


ZIP_DIR = Path("D:/trading_pythonbacktest_data/NQ L2 data")
OUTPUT = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")


def process_csv_from_zip(zf, csv_name):
    """Stream one CSV from open zipfile, return 15-min bars as dict of {minute_key: OHLCV}."""
    bars = {}  # key = floor_minute_str -> {open, high, low, close, volume, count, first_ts}

    with zf.open(csv_name) as f:
        for raw_line in f:
            line = raw_line.decode('utf-8', errors='replace').strip()
            parts = line.split(';')
            if len(parts) < 5:
                continue
            if parts[2] != '2':  # Only trades (fast string check)
                continue
            try:
                price = float(parts[3])
                volume = int(parts[4])
            except ValueError:
                continue

            # Floor timestamp to 15-min: parse YYYYMMDDHHMMSS, floor MM to 15-min
            ts = parts[0]
            if len(ts) < 14:
                continue
            minute = int(ts[10:12])
            floored_min = (minute // 15) * 15
            bar_key = ts[:10] + f"{floored_min:02d}"  # YYYYMMDDHHMM

            if bar_key not in bars:
                bars[bar_key] = {
                    'open': price, 'high': price, 'low': price,
                    'close': price, 'volume': volume, 'count': 1
                }
            else:
                b = bars[bar_key]
                if price > b['high']:
                    b['high'] = price
                if price < b['low']:
                    b['low'] = price
                b['close'] = price
                b['volume'] += volume
                b['count'] += 1

    return bars


def bars_dict_to_df(bars):
    """Convert {bar_key: OHLCV} dict to DataFrame with proper timestamps."""
    if not bars:
        return pd.DataFrame()

    rows = []
    for key, b in bars.items():
        # key = YYYYMMDDHHMM (floored open time)
        ts = pd.Timestamp(
            year=int(key[0:4]), month=int(key[4:6]), day=int(key[6:8]),
            hour=int(key[8:10]), minute=int(key[10:12]), tz="UTC"
        )
        # Close time = open + 15min
        close_ts = ts + pd.Timedelta(minutes=15)
        rows.append({
            'timestamp': close_ts,
            'open': b['open'], 'high': b['high'], 'low': b['low'],
            'close': b['close'], 'volume': b['volume'], 'tick_count': b['count']
        })

    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    return df


def main():
    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files", flush=True)

    start = time.time()
    all_bars = {}  # global bar_key -> OHLCV dict, merge as we go
    total_trades = 0
    total_csvs = 0

    for i, zf_path in enumerate(zip_files):
        t0 = time.time()
        with zipfile.ZipFile(zf_path) as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]

            for csv_name in sorted(csv_files):
                bars = process_csv_from_zip(zf, csv_name)

                # Merge into global bars
                for key, b in bars.items():
                    if key not in all_bars:
                        all_bars[key] = b
                    else:
                        existing = all_bars[key]
                        if b['high'] > existing['high']:
                            existing['high'] = b['high']
                        if b['low'] < existing['low']:
                            existing['low'] = b['low']
                        existing['close'] = b['close']
                        existing['volume'] += b['volume']
                        existing['count'] += b['count']

                trade_count = sum(b['count'] for b in bars.values())
                total_trades += trade_count
                total_csvs += 1

        dt = time.time() - t0
        print(f"[{i+1}/{len(zip_files)}] {zf_path.name}  {len(csv_files)} csvs  {dt:.0f}s  bars_so_far={len(all_bars)}", flush=True)

    elapsed = time.time() - start
    print(f"\nProcessed {total_trades:,} trades from {total_csvs} CSVs in {elapsed:.0f}s", flush=True)

    # Convert to DataFrame
    print("Building DataFrame...", flush=True)
    df = bars_dict_to_df(all_bars)

    print(f"Total bars: {len(df):,}", flush=True)
    print(f"Date range: {df.index[0]} to {df.index[-1]}", flush=True)

    df.to_parquet(OUTPUT)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Saved to {OUTPUT} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()

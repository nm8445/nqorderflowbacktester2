"""
Build 1-min OHLCV bar cache from MarketTick L2 zip files.
Streams from zip -> filters Type=2 (trades) -> builds 1-min bars per CSV -> saves parquet.

Uses multiprocessing to process zip files in parallel for ~4-6x speedup.

Source: D:/trading_pythonbacktest_data/NQ L2 data/*.zip
Output: D:/trading_pythonbacktest_data/markettick_1min_bars.parquet
"""
import zipfile
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time


ZIP_DIR = Path("D:/trading_pythonbacktest_data/NQ L2 data")
OUTPUT = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")


def process_single_zip(zf_path_str: str) -> dict:
    """Process one zip file, return dict of {minute_key: OHLCV}."""
    zf_path = Path(zf_path_str)
    bars = {}

    with zipfile.ZipFile(zf_path) as zf:
        csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
        for csv_name in sorted(csv_files):
            with zf.open(csv_name) as f:
                for raw_line in f:
                    line = raw_line.decode('utf-8', errors='replace')
                    # Quick filter: find second semicolon, check if char is '2'
                    idx1 = line.find(';')
                    if idx1 < 0:
                        continue
                    idx2 = line.find(';', idx1 + 1)
                    if idx2 < 0:
                        continue
                    if line[idx1+1:idx2] != '1':
                        # Type field is parts[1], but we need parts[2] for trade type
                        pass

                    parts = line.split(';')
                    if len(parts) < 5:
                        continue
                    if parts[2] != '2':
                        continue
                    try:
                        price = float(parts[3])
                        volume = int(parts[4])
                    except ValueError:
                        continue

                    ts = parts[0]
                    if len(ts) < 12:
                        continue
                    bar_key = ts[:12]  # YYYYMMDDHHMM

                    if bar_key not in bars:
                        bars[bar_key] = [price, price, price, price, volume, 1]
                        # [open, high, low, close, volume, count]
                    else:
                        b = bars[bar_key]
                        if price > b[1]:
                            b[1] = price
                        if price < b[2]:
                            b[2] = price
                        b[3] = price
                        b[4] += volume
                        b[5] += 1

    return bars


def merge_bars(all_bars: dict, new_bars: dict):
    """Merge new_bars into all_bars in place."""
    for key, b in new_bars.items():
        if key not in all_bars:
            all_bars[key] = b
        else:
            existing = all_bars[key]
            if b[1] > existing[1]:
                existing[1] = b[1]  # high
            if b[2] < existing[2]:
                existing[2] = b[2]  # low
            existing[3] = b[3]  # close (last wins by sort order)
            existing[4] += b[4]  # volume
            existing[5] += b[5]  # count


def bars_dict_to_df(bars: dict) -> pd.DataFrame:
    """Convert {bar_key: [O,H,L,C,V,count]} dict to DataFrame."""
    if not bars:
        return pd.DataFrame()

    rows = []
    for key, b in bars.items():
        ts = pd.Timestamp(
            year=int(key[0:4]), month=int(key[4:6]), day=int(key[6:8]),
            hour=int(key[8:10]), minute=int(key[10:12]), tz="UTC"
        )
        close_ts = ts + pd.Timedelta(minutes=1)
        rows.append({
            'timestamp': close_ts,
            'open': b[0], 'high': b[1], 'low': b[2],
            'close': b[3], 'volume': b[4], 'tick_count': b[5]
        })

    return pd.DataFrame(rows).set_index('timestamp').sort_index()


def main():
    zip_files = sorted([f for f in ZIP_DIR.glob("*.zip") if f.name[0].isdigit()])
    print(f"Found {len(zip_files)} data zip files", flush=True)

    n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} workers", flush=True)

    start = time.time()

    # Process in batches to show progress and manage memory
    batch_size = n_workers * 2
    all_bars = {}
    processed = 0

    for batch_start in range(0, len(zip_files), batch_size):
        batch = zip_files[batch_start:batch_start + batch_size]
        batch_paths = [str(f) for f in batch]

        with Pool(n_workers) as pool:
            results = pool.map(process_single_zip, batch_paths)

        for bars in results:
            merge_bars(all_bars, bars)

        processed += len(batch)
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(zip_files) - processed) / rate if rate > 0 else 0
        print(f"[{processed}/{len(zip_files)}] bars={len(all_bars):,}  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    elapsed = time.time() - start
    total_trades = sum(b[5] for b in all_bars.values())
    print(f"\nProcessed {total_trades:,} trades in {elapsed:.0f}s", flush=True)

    print("Building DataFrame...", flush=True)
    df = bars_dict_to_df(all_bars)

    print(f"Total bars: {len(df):,}", flush=True)
    print(f"Date range: {df.index[0]} to {df.index[-1]}", flush=True)

    df.to_parquet(OUTPUT)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Saved to {OUTPUT} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()

"""
Build 15-minute OHLC bars from tick data (DBN files).

Uses to_ndarray() for speed, filters trades via numpy, builds OHLC directly.
Output: D:/trading_pythonbacktest_data/15min_bars.parquet
"""

import gc
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

DATA_DIR = Path("D:/trading_pythonbacktest_data")
DBN_DIR = DATA_DIR / "dbn"
OUTPUT = DATA_DIR / "15min_bars.parquet"
ET = "America/New_York"
PRICE_SCALE = 1e9


def process_dbn(dbn_file: Path) -> pd.DataFrame | None:
    """Extract 15-min OHLC from a single DBN file using ndarray."""
    try:
        store = db.DBNStore.from_file(str(dbn_file))
        arr = store.to_ndarray()
        del store

        if len(arr) == 0:
            return None

        # Filter trades only (action == b'T')
        mask = arr["action"] == b"T"
        ts = arr["ts_event"][mask]
        prices = arr["price"][mask].astype(np.float64) / PRICE_SCALE
        del arr, mask
        gc.collect()

        if len(prices) == 0:
            return None

        # Build DatetimeIndex from nanosecond timestamps
        idx = pd.DatetimeIndex(ts, tz="UTC").tz_convert(ET)
        del ts

        # Create Series and resample to 15-min OHLC
        s = pd.Series(prices, index=idx)
        del prices, idx
        gc.collect()

        ohlc = s.resample("15min").ohlc().dropna()
        del s
        gc.collect()

        return ohlc

    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        return None


def main():
    dbn_files = sorted(DBN_DIR.glob("*.dbn"))
    print(f"Found {len(dbn_files)} DBN files")
    print(f"Output: {OUTPUT}\n")

    all_frames = []
    for i, f in enumerate(dbn_files, 1):
        print(f"[{i}/{len(dbn_files)}] {f.name}...", end=" ", flush=True)
        ohlc = process_dbn(f)
        if ohlc is not None and not ohlc.empty:
            all_frames.append(ohlc)
            print(f"OK ({len(ohlc)} bars)", flush=True)
        else:
            print("skip", flush=True)
        gc.collect()

    print(f"\nCombining {len(all_frames)} frames...")
    combined = pd.concat(all_frames)
    del all_frames
    gc.collect()

    # Remove duplicate timestamps (overlapping DBN files)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()

    combined.to_parquet(OUTPUT)
    print(f"\nSaved {len(combined):,} bars to {OUTPUT}")
    print(f"Date range: {combined.index[0]} to {combined.index[-1]}")


if __name__ == "__main__":
    main()

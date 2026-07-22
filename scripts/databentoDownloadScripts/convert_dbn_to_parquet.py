"""Convert single-day DBN files to range-named parquets matching existing convention.

Naming convention: NQ_c_0_mbp-1_{start}_{end}.parquet  where end = start + 2 days
                   (matches existing files which span 2-day windows, except weekends)

Each single-day DBN gets converted to one parquet so the consolidator picks them up.
Output dir: D:/trading_pythonbacktest_data/parquet/
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import databento as db
import pandas as pd

DBN_DIR = Path("D:/trading_pythonbacktest_data/dbn")
PARQUET_DIR = Path("D:/trading_pythonbacktest_data/parquet")
PARQUET_DIR.mkdir(exist_ok=True)


def date_already_covered(date: dt.date, parquet_dir: Path) -> bool:
    """Check whether a parquet exists whose date range covers `date`."""
    for p in parquet_dir.glob("NQ_c_0_mbp-1_*.parquet"):
        parts = p.stem.split("_")
        try:
            start_d = dt.date.fromisoformat(parts[-2])
            end_d   = dt.date.fromisoformat(parts[-1])
            if start_d <= date <= end_d:
                return True
        except Exception:
            continue
    return False


def convert(dbn_path: Path) -> Path | None:
    """Convert one single-day DBN to a parquet file."""
    date_str = dbn_path.stem  # e.g., "2026-04-04"
    try:
        date = dt.date.fromisoformat(date_str)
    except ValueError:
        print(f"  SKIP {dbn_path.name} — non-date filename")
        return None

    # Output file: span date -> date+2 days (matching existing pattern)
    end_date = date + dt.timedelta(days=2)
    out_name = f"NQ_c_0_mbp-1_{date_str}_{end_date.isoformat()}.parquet"
    out_path = PARQUET_DIR / out_name
    if out_path.exists():
        print(f"  EXISTS {out_name}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return out_path

    if dbn_path.stat().st_size < 500:
        print(f"  SKIP {dbn_path.name} ({dbn_path.stat().st_size} bytes — likely empty/weekend)")
        return None

    print(f"  CONVERT {dbn_path.name}  -> {out_name}", end=" ", flush=True)
    t0 = time.time()
    store = db.DBNStore.from_file(str(dbn_path))
    df = store.to_df()
    df.to_parquet(out_path, compression="snappy", index=True)
    elapsed = time.time() - t0
    print(f"OK  ({len(df):,} rows, {out_path.stat().st_size/1024/1024:.1f} MB, {elapsed:.0f}s)")
    return out_path


def main():
    start = dt.date(2026, 5, 24)
    end   = dt.date(2026, 6, 12)
    if len(sys.argv) >= 3:
        start = dt.date.fromisoformat(sys.argv[1])
        end   = dt.date.fromisoformat(sys.argv[2])

    print(f"Converting DBN -> parquet for {start} to {end}")
    print(f"DBN dir:     {DBN_DIR}")
    print(f"Parquet dir: {PARQUET_DIR}")
    print()

    converted = 0
    skipped = 0
    for dbn_path in sorted(DBN_DIR.glob("*.dbn")):
        try:
            d = dt.date.fromisoformat(dbn_path.stem)
        except ValueError:
            continue
        if d < start or d > end:
            continue
        result = convert(dbn_path)
        if result is not None:
            converted += 1
        else:
            skipped += 1

    print()
    print(f"DONE  converted={converted}  skipped={skipped}")


if __name__ == "__main__":
    sys.exit(main())

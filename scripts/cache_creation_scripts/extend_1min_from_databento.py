"""Extend markettick_1min_bars.parquet with 1-min bars from Databento DB trades.

Reads the consolidated DB trades parquet (built by build_volumetric_5min_1tpl.py)
and computes 1-min OHLCV bars for any dates beyond the existing parquet's coverage.
Appends to markettick_1min_bars.parquet.

Schema match:
  index: timestamp (UTC, datetime, BAR CLOSE TIME like the markettick build)
  cols : open, high, low, close, volume, tick_count
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

DB_CONS  = Path("D:/trading_pythonbacktest_data/_tmp_db_trades_consolidated.parquet")
EXISTING = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")


def main():
    print(f"loading existing markettick_1min_bars.parquet...")
    existing = pd.read_parquet(EXISTING)
    if existing.index.tz is None:
        existing.index = existing.index.tz_localize("UTC")
    last_existing = existing.index.max()
    print(f"  existing range: {existing.index.min()} -> {last_existing}, {len(existing):,} bars")

    print(f"loading consolidated DB trades...")
    db_trades = pd.read_parquet(DB_CONS)
    db_trades["ts"] = pd.to_datetime(db_trades["ts"], utc=True)
    print(f"  DB trades range: {db_trades['ts'].min()} -> {db_trades['ts'].max()}, "
          f"{len(db_trades):,} rows")

    # Take only DB trades AFTER existing coverage
    cutoff = last_existing
    new = db_trades[db_trades["ts"] > cutoff].copy()
    print(f"  trades after cutoff ({cutoff}): {len(new):,}")
    if new.empty:
        print("nothing to append")
        return

    print(f"resampling to 1-min OHLCV...")
    new = new.set_index("ts").sort_index()
    ohlc = new["price"].resample("1min", label="right", closed="right").ohlc()
    vol  = new["size"].resample("1min", label="right", closed="right").sum()
    cnt  = new["price"].resample("1min", label="right", closed="right").count()
    bars = ohlc.copy()
    bars["volume"] = vol.astype("int64")
    bars["tick_count"] = cnt.astype("int64")
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars.index.name = "timestamp"
    print(f"  new bars: {len(bars):,}, range {bars.index.min()} -> {bars.index.max()}")

    print(f"concatenating + writing back to parquet...")
    combined = pd.concat([existing, bars])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    print(f"  combined: {len(combined):,} bars, range {combined.index.min()} -> {combined.index.max()}")

    combined.to_parquet(EXISTING, compression="zstd")
    print(f"  wrote {EXISTING}, size={EXISTING.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())

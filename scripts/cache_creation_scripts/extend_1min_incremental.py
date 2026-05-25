"""Incremental 1-min bar extender — only reads NEW raw DB parquets.

Skips the slow full-consolidation step in build_volumetric_5min_1tpl.py.
Goes directly from raw date-ranged DB parquets to 1-min bars, only processing
files that contain dates AFTER the existing 1-min bars parquet's max date.

This makes the update O(new_files) instead of O(all_files).

Pattern:
  1. Read existing markettick_1min_bars.parquet, find max timestamp
  2. List all NQ_c_0_mbp-1_*.parquet in raw dir
  3. Filter to files whose END date is AFTER existing max
  4. Read each, filter to trade rows, compute 1-min OHLCV
  5. Concat new bars, dedupe, append to existing parquet
"""
from __future__ import annotations
import datetime as dt
from pathlib import Path
import pandas as pd

RAW_DIR  = Path("D:/trading_pythonbacktest_data/parquet")
EXISTING = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")


def main():
    print(f"[1/4] Loading existing markettick_1min_bars.parquet...")
    existing = pd.read_parquet(EXISTING)
    if existing.index.tz is None:
        existing.index = existing.index.tz_localize("UTC")
    last_existing = existing.index.max()
    print(f"  existing range: {existing.index.min()} -> {last_existing}, {len(existing):,} bars")
    cutoff_date = last_existing.normalize()

    print(f"\n[2/4] Finding raw DB parquets covering dates after {cutoff_date.date()}...")
    candidates = []
    for p in sorted(RAW_DIR.glob("NQ_c_0_mbp-1_*.parquet")):
        parts = p.stem.split("_")
        try:
            start_d = dt.date.fromisoformat(parts[-2])
            end_d   = dt.date.fromisoformat(parts[-1])
        except Exception:
            continue
        # Include if file's end date >= cutoff (overlap or fully past cutoff)
        if end_d >= cutoff_date.date():
            candidates.append((start_d, end_d, p))
    candidates.sort()
    print(f"  Found {len(candidates)} candidate file(s):")
    for s, e, p in candidates:
        print(f"    {s} -> {e}   {p.name}  ({p.stat().st_size/1024/1024:.0f} MB)")

    if not candidates:
        print("Nothing to do — existing parquet is already up-to-date.")
        return

    print(f"\n[3/4] Reading + filtering trade rows from candidate files...")
    all_trades = []
    for _, _, p in candidates:
        df = pd.read_parquet(p, columns=["ts_event", "action", "side", "price", "size", "sequence"])
        mask = (df["action"] == "T") & (df["side"].isin(["A", "B"]))
        tr = df[mask][["ts_event", "side", "price", "size", "sequence"]].copy()
        all_trades.append(tr)
        print(f"  {p.name}: +{len(tr):,} trades")
    all_tr = pd.concat(all_trades, ignore_index=True)
    print(f"  pre-dedupe: {len(all_tr):,} trades")
    all_tr = all_tr.drop_duplicates(subset=["ts_event", "sequence", "price", "size", "side"])
    print(f"  post-dedupe: {len(all_tr):,} trades")

    all_tr["ts"] = pd.to_datetime(all_tr["ts_event"], utc=True)
    # Drop trades older than the existing cutoff (we only want NEW data)
    new = all_tr[all_tr["ts"] > last_existing].copy()
    print(f"  trades after cutoff ({last_existing}): {len(new):,}")
    if new.empty:
        print("No new trades to append.")
        return

    print(f"\n[4/4] Resampling to 1-min OHLCV and appending...")
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

    combined = pd.concat([existing, bars])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    print(f"  combined: {len(combined):,} bars, range {combined.index.min()} -> {combined.index.max()}")
    combined.to_parquet(EXISTING, compression="zstd")
    print(f"  wrote {EXISTING}, size={EXISTING.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()

"""Incremental daily gamma refresh job — append only new days to menthorq_levels_nq.parquet.

This replaces the bulk builder for daily operations. Instead of reprocessing all
1,330+ days (~6 min), this:
  1. Loads existing parquet → finds latest date
  2. Determines target end date (today, or --end-date)
  3. For each new day requiring computation, checks data availability:
     - QQQ greeks dir present
     - NDX prices/OI dir present
     - NQ settle available (from daily_settles.json OR markettick_1min_bars.parquet)
  4. Computes new rows via menthorq_style_levels.process_one()
  5. Appends to parquet
  6. Updates state/last_gamma_refresh.json

Scheduled daily ~17:00 ET via Windows Task Scheduler.

Usage:
    python live/combined/gamma_refresh.py            # incremental: only new days
    python live/combined/gamma_refresh.py --end-date 2026-05-15
    python live/combined/gamma_refresh.py --force-rebuild  # full rebuild
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "thetadata" / "daily_pipeline"))

from live.combined.config import SETTLES_JSON, STATE_DIR

# Reuse the bulk builder's helpers and the per-day computation
import menthorq_style_levels as mq
import build_menthorq_levels_bulk as bulk

PARQUET = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
LAST_REFRESH_JSON = STATE_DIR / "last_gamma_refresh.json"


def load_nq_settles_from_archive() -> dict[dt.date, float]:
    """NQ settle from markettick_1min_bars.parquet at 16:00 ET close.
    Returns {date: close_price}."""
    return bulk.load_nq_settles()


def load_nq_settles_from_live_state() -> dict[dt.date, float]:
    """Load settles captured by the live system's settle_recorder."""
    if not SETTLES_JSON.exists():
        return {}
    raw = json.loads(SETTLES_JSON.read_text())
    return {dt.date.fromisoformat(k): float(v) for k, v in raw.items()}


def load_nq_settles_from_rolling_cache() -> dict[dt.date, float]:
    """Read the 16:05 ET close from each session pickle in the rolling cache.
    This fills the gap when markettick_1min_bars.parquet is stale.
    """
    import pickle
    from live.combined.config import LIVE_WARMSTART_CACHE_DIR, ET_TZ
    if not LIVE_WARMSTART_CACHE_DIR.exists():
        return {}
    settles = {}
    for pkl in LIVE_WARMSTART_CACHE_DIR.glob("nq_5min_*.pkl"):
        try:
            session_date = dt.date.fromisoformat(pkl.stem.replace("nq_5min_", ""))
            with open(pkl, "rb") as f:
                bars = pickle.load(f)
            # Find the bar whose open_time (ET) == 16:00 on session_date
            # Pickle stores open_time as naive UTC; convert to ET.
            for b in bars:
                ot = pd.Timestamp(b["open_time"])
                if ot.tz is None:
                    ot_et = ot.tz_localize("UTC").tz_convert(ET_TZ)
                else:
                    ot_et = ot.tz_convert(ET_TZ)
                if ot_et.date() == session_date and ot_et.hour == 16 and ot_et.minute == 0:
                    settles[session_date] = float(b["close"])
                    break
        except Exception as e:
            print(f"  [gamma_refresh] cache settle read failed for {pkl.name}: {e}")
    return settles


def get_combined_settles() -> dict[dt.date, float]:
    """Merge all sources (archive, live state, rolling cache).
    Priority (later overrides earlier on conflict):
      1. archive (markettick_1min_bars.parquet — most reliable for historical)
      2. rolling cache pickles (covers recent days when archive is stale)
      3. live state json (captured by settle_recorder while engine is running)
    """
    settles = load_nq_settles_from_archive()
    settles.update(load_nq_settles_from_rolling_cache())
    settles.update(load_nq_settles_from_live_state())
    return settles


def latest_date_in_parquet() -> dt.date | None:
    if not PARQUET.exists():
        return None
    df = pd.read_parquet(PARQUET, columns=["date"])
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).dt.date.max()


def find_new_days(latest_in_parquet: dt.date | None,
                  end_date: dt.date,
                  settles: dict[dt.date, float]) -> list[dt.date]:
    """Return list of trading days needing computation: dates after latest_in_parquet,
    up to end_date, where ALL required data is available (settle + QQQ + NDX dirs)."""
    qqq_dirs = bulk.date_dirs(bulk.QQQ_ROOT)
    ndx_dirs = bulk.date_dirs(bulk.NDX_ROOT)
    settle_dates = set(settles.keys())

    start = (latest_in_parquet + dt.timedelta(days=1)) if latest_in_parquet else dt.date(2020, 12, 1)
    candidates = []
    d = start
    while d <= end_date:
        if d.weekday() < 5 and d in qqq_dirs and d in ndx_dirs and d in settle_dates:
            candidates.append(d)
        d += dt.timedelta(days=1)
    return candidates


def process_new_days(new_dates: list[dt.date], settles: dict[dt.date, float]) -> list[dict]:
    """Compute new rows using bulk.process_one. Returns list of dicts."""
    if not new_dates:
        return []

    # Build "next trading day" lookup for QQQ HVL 0DTE filter
    qqq_avail = bulk.date_dirs(bulk.QQQ_ROOT)
    ndx_avail = bulk.date_dirs(bulk.NDX_ROOT)
    all_avail_sorted = sorted(qqq_avail & ndx_avail & set(settles.keys()))
    next_td = {}
    for i, d in enumerate(all_avail_sorted[:-1]):
        next_td[d] = all_avail_sorted[i + 1]
    if all_avail_sorted:
        # For the LATEST available date, walk forward to the next weekday
        # (Mon-Fri). On Friday this jumps 3 days to Monday — important because
        # QQQ options expiring Sat/Sun don't exist, so a DTE=(1,2) filter for a
        # Friday would return an empty chain and crash.
        last = all_avail_sorted[-1]
        next_d = last + dt.timedelta(days=1)
        while next_d.weekday() >= 5:   # 5=Sat, 6=Sun
            next_d += dt.timedelta(days=1)
        next_td[last] = next_d

    rows = []
    for d in new_dates:
        ntd = next_td.get(d, d + dt.timedelta(days=1))
        shift = (ntd - d).days
        qqq_hvl_filter = (shift, shift + 1)
        try:
            t0 = time.time()
            row = bulk.process_one(d, settles[d], qqq_hvl_filter)
            rows.append(row)
            print(f"  [{d}] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  [{d}] ERROR: {str(e)[:80]}")
    return rows


def append_to_parquet(new_rows: list[dict]) -> None:
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.date
    if PARQUET.exists():
        existing = pd.read_parquet(PARQUET)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        # Drop any existing rows for the new dates (in case of re-run)
        existing = existing[~existing["date"].isin(new_df["date"])]
        merged = pd.concat([existing, new_df], ignore_index=True).sort_values("date")
    else:
        merged = new_df.sort_values("date")
    merged.to_parquet(PARQUET, compression="zstd", index=False)
    print(f"  wrote {len(merged)} rows total ({len(new_rows)} new) to {PARQUET}")


def update_refresh_state(end_date: dt.date, new_count: int) -> None:
    LAST_REFRESH_JSON.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "last_run_utc": dt.datetime.utcnow().isoformat(),
        "target_end_date": end_date.isoformat(),
        "rows_added": new_count,
    }
    LAST_REFRESH_JSON.write_text(json.dumps(state, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="Delete parquet and rebuild from scratch (slow)")
    args = ap.parse_args()

    end_date = (dt.date.fromisoformat(args.end_date) if args.end_date else dt.date.today())
    print(f"[gamma_refresh] target end date: {end_date}")

    if args.force_rebuild and PARQUET.exists():
        print("[gamma_refresh] force-rebuild: deleting existing parquet")
        PARQUET.unlink()

    latest = latest_date_in_parquet()
    print(f"[gamma_refresh] latest in parquet: {latest}")

    settles = get_combined_settles()
    print(f"[gamma_refresh] {len(settles)} NQ settles available "
          f"({min(settles)} -> {max(settles)} if any)")

    new_dates = find_new_days(latest, end_date, settles)
    print(f"[gamma_refresh] {len(new_dates)} new days to process")
    if new_dates:
        print(f"  range: {new_dates[0]} -> {new_dates[-1]}")

    rows = process_new_days(new_dates, settles)
    append_to_parquet(rows)
    update_refresh_state(end_date, len(rows))
    print(f"[gamma_refresh] done. Added {len(rows)} rows.")


if __name__ == "__main__":
    main()

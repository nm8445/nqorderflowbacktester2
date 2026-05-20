"""
Build developing volume profile cache from MarketTick data.

Matches the format used by the Databento-built cache at cache/profiles/:
one pickle per session date, holding a list of {refresh_time, VolumeProfile}
entries — one per minute of the session (6pm ET prev day to 4:59:59 PM ET).

Each VolumeProfile is CUMULATIVE from session start through refresh_time, so
any time-window profile (RTH, swing-to-swing, etc.) can be derived by
subtracting two snapshots' level dicts.

Data source: MarketTick L2 zip files (D:/trading_pythonbacktest_data/NQ L2 data/*.zip).
Trade stream: type=1 sub=2 only (NOT type=2 which is L2 DOM depth).
Aggressor inference: Lee-Ready (trade vs. prevailing BBO + tick rule fallback).
See memory/reference_markettick_schema.md.

Output: D:/trading_pythonbacktest_data/cache/profiles/{YYYY-MM-DD}_refresh_minutes=1.pkl

Default stop date: 2025-03-13 (exclusive) — Databento coverage begins there.

Usage:
    python -u scripts/cache_creation_scripts/build_developing_profile_cache_from_markettick.py
    python -u scripts/cache_creation_scripts/build_developing_profile_cache_from_markettick.py --test 2025-04-15
"""
import sys
import zipfile
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_signal_cache_from_markettick import (
    extract_trades_with_side, parse_utc_ts, get_session_date
)
from nqbt.analysis.volume_profile import VolumeProfile, ProfileLevel
from nqbt.analysis.range_bars import TICK_SIZE

ET = "America/New_York"
ZIP_DIR = Path("D:/trading_pythonbacktest_data/NQ L2 data")
CACHE_DIR = Path("D:/trading_pythonbacktest_data/cache/profiles")

WORKERS = 8
TICKS_PER_LEVEL = 10        # 2.5 pts per level (matches existing Databento cache)
VALUE_AREA_PCT = 0.68       # matches existing Databento cache
DATABENTO_START = "2025-03-13"  # exclusive cutoff


def build_developing_snapshots(trades, session_start_utc, session_end_utc):
    """Given session trades (sorted list of (ts_str, price, size, aggressor)),
    emit one VolumeProfile snapshot per minute boundary from session_start+1min
    through session_end. Each snapshot is cumulative from session_start."""
    level_size = round(TICKS_PER_LEVEL * TICK_SIZE, 10)
    levels = defaultdict(lambda: [0, 0])  # price -> [buy_vol, sell_vol]

    # Minute boundaries
    boundaries = []
    t = session_start_utc + pd.Timedelta(minutes=1)
    while t <= session_end_utc:
        boundaries.append(t)
        t += pd.Timedelta(minutes=1)

    snapshots = []
    trade_idx = 0
    n_trades = len(trades)

    for boundary in boundaries:
        # Absorb trades strictly before this boundary
        while trade_idx < n_trades:
            ts_str, price, size, aggressor = trades[trade_idx]
            ts = parse_utc_ts(ts_str)
            if ts >= boundary:
                break
            price_ticks = round(price / TICK_SIZE)
            lvl_idx = price_ticks // TICKS_PER_LEVEL
            lvl_price = round(lvl_idx * TICKS_PER_LEVEL * TICK_SIZE, 4)
            if aggressor == "buy":
                levels[lvl_price][0] += size
            else:
                levels[lvl_price][1] += size
            trade_idx += 1

        if not levels:
            snapshots.append({
                "refresh_time": boundary,
                "profile": VolumeProfile(
                    levels={}, poc=0.0, vah=0.0, val=0.0, total_volume=0,
                    ticks_per_level=TICKS_PER_LEVEL, value_area_pct=VALUE_AREA_PCT),
            })
            continue

        profile_levels = {
            p: ProfileLevel(price=p, buy_vol=bs[0], sell_vol=bs[1])
            for p, bs in levels.items()
        }
        total = sum(bs[0] + bs[1] for bs in levels.values())
        sorted_prices = sorted(levels.keys())
        poc = max(levels, key=lambda p: levels[p][0] + levels[p][1])
        poc_idx = sorted_prices.index(poc)

        va_target = total * VALUE_AREA_PCT
        va_vol = levels[poc][0] + levels[poc][1]
        lo, hi = poc_idx, poc_idx
        while va_vol < va_target:
            can_up = hi + 1 < len(sorted_prices)
            can_dn = lo - 1 >= 0
            if not can_up and not can_dn:
                break
            up_v = (levels[sorted_prices[hi + 1]][0] + levels[sorted_prices[hi + 1]][1]) if can_up else -1
            dn_v = (levels[sorted_prices[lo - 1]][0] + levels[sorted_prices[lo - 1]][1]) if can_dn else -1
            if up_v >= dn_v:
                hi += 1; va_vol += up_v
            else:
                lo -= 1; va_vol += dn_v

        vah = sorted_prices[hi] + level_size
        val = sorted_prices[lo]

        snapshots.append({
            "refresh_time": boundary,
            "profile": VolumeProfile(
                levels=profile_levels, poc=poc, vah=vah, val=val,
                total_volume=total, ticks_per_level=TICKS_PER_LEVEL,
                value_area_pct=VALUE_AREA_PCT),
        })

    return snapshots


def process_session(args):
    session_date_iso, csv_entries = args
    cache_file = CACHE_DIR / f"{session_date_iso}_refresh_minutes=1.pkl"
    if cache_file.exists():
        return (session_date_iso, "cached", 0)

    session_date_obj = datetime.strptime(session_date_iso, "%Y-%m-%d").date()

    # Session window (ET): 18:00 prev day -> 16:59:59 current day
    prev_et = pd.Timestamp(session_date_obj - timedelta(days=1), tz=ET).replace(hour=18, minute=0, second=0)
    end_et  = pd.Timestamp(session_date_obj, tz=ET).replace(hour=16, minute=59, second=59)
    session_start_utc = prev_et.tz_convert("UTC")
    session_end_utc = end_et.tz_convert("UTC")

    # Stream MarketTick CSVs, carry BBO state across the day boundary
    state = {"best_bid": None, "best_ask": None,
             "last_trade_price": None, "last_aggressor": "buy"}
    all_trades = []
    for zip_path_str, csv_name in csv_entries:
        try:
            with zipfile.ZipFile(zip_path_str) as zf:
                rows = extract_trades_with_side(zf, csv_name, state=state)
            all_trades.extend(rows)
        except Exception as e:
            return (session_date_iso, f"error_reading_{csv_name}:{e}", 0)

    if not all_trades:
        return (session_date_iso, "no_trades", 0)

    all_trades.sort(key=lambda t: t[0])

    # Filter to trades belonging to this session
    session_trades = []
    for t in all_trades:
        try:
            ts = parse_utc_ts(t[0])
        except Exception:
            continue
        if get_session_date(ts) == session_date_obj:
            session_trades.append(t)

    if len(session_trades) < 100:
        return (session_date_iso, f"few_trades_{len(session_trades)}", 0)

    snapshots = build_developing_snapshots(session_trades, session_start_utc, session_end_utc)
    if not snapshots:
        return (session_date_iso, "no_snapshots", 0)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(snapshots, f)

    return (session_date_iso, "ok", len(snapshots))


def index_csvs():
    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    all_csv_entries = []
    for zf_path in zip_files:
        with zipfile.ZipFile(zf_path) as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    basename = name.split("/")[-1].replace(".csv", "")
                    if len(basename) == 8 and basename.isdigit():
                        all_csv_entries.append((str(zf_path), name, basename))
    all_csv_entries.sort(key=lambda x: x[2])
    csv_by_date = defaultdict(list)
    for zf_path_str, csv_name, date_str in all_csv_entries:
        csv_by_date[date_str].append((zf_path_str, csv_name))
    return csv_by_date


def build_session_task(session_iso, csv_by_date):
    d = datetime.strptime(session_iso, "%Y-%m-%d").date()
    prev_cal = (d - timedelta(days=1)).strftime("%Y%m%d")
    curr_cal = d.strftime("%Y%m%d")
    entries = []
    if prev_cal in csv_by_date:
        entries.extend(csv_by_date[prev_cal])
    if curr_cal in csv_by_date:
        entries.extend(csv_by_date[curr_cal])
    return (session_iso, entries) if entries else None


def run_test(date_iso):
    """Build a single session and write it to a temp path for manual inspection."""
    csv_by_date = index_csvs()
    task = build_session_task(date_iso, csv_by_date)
    if task is None:
        print(f"No MarketTick CSVs found for {date_iso}")
        return

    # Write to a temp file instead of CACHE_DIR so we don't stomp on Databento data
    import tempfile
    tmpdir = Path(tempfile.gettempdir()) / "mt_dev_profile_test"
    tmpdir.mkdir(exist_ok=True)
    tmp_cache_file = tmpdir / f"{date_iso}_refresh_minutes=1.pkl"
    if tmp_cache_file.exists():
        tmp_cache_file.unlink()

    # Temporarily override CACHE_DIR for this call
    global CACHE_DIR
    orig = CACHE_DIR
    CACHE_DIR = tmpdir
    try:
        t0 = time.time()
        result = process_session(task)
        print(f"TEST result: {result}  elapsed={time.time()-t0:.1f}s")
        print(f"Output: {tmp_cache_file}")
    finally:
        CACHE_DIR = orig


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output: {CACHE_DIR}")
    print(f"Workers: {WORKERS}")
    print(f"Stop date (exclusive): {DATABENTO_START}")

    csv_by_date = index_csvs()
    calendar_dates = sorted(csv_by_date.keys())
    print(f"Calendar dates: {calendar_dates[0]} to {calendar_dates[-1]} ({len(calendar_dates)} days)")

    first_cal = datetime.strptime(calendar_dates[0], "%Y%m%d").date()
    stop_date = datetime.strptime(DATABENTO_START, "%Y-%m-%d").date()

    tasks = []
    cur = first_cal
    while cur < stop_date:
        if cur.weekday() < 5:
            sess_iso = cur.strftime("%Y-%m-%d")
            cache_file = CACHE_DIR / f"{sess_iso}_refresh_minutes=1.pkl"
            if not cache_file.exists():
                task = build_session_task(sess_iso, csv_by_date)
                if task is not None:
                    tasks.append(task)
        cur += timedelta(days=1)

    print(f"Sessions to build: {len(tasks)}")
    existing = len(list(CACHE_DIR.glob("*.pkl")))
    print(f"Already in cache dir: {existing}", flush=True)

    if not tasks:
        print("Nothing to do.")
        return

    start_time = time.time()
    done = ok = errors = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_session, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            sess_iso = futures[fut]
            done += 1
            try:
                sd, status, nsnaps = fut.result()
                if status == "ok":
                    ok += 1
                if done <= 5 or done % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta_min = ((len(tasks) - done) / rate / 60) if rate > 0 else 0.0
                    print(f"  [{done}/{len(tasks)}] {sd} {status}  snaps={nsnaps}  "
                          f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  ETA={eta_min:.1f}min",
                          flush=True)
            except Exception as e:
                errors += 1
                print(f"  ERROR {sess_iso}: {e}", flush=True)

    elapsed = time.time() - start_time
    cached_files = len(list(CACHE_DIR.glob("*.pkl")))

    print()
    print("=" * 70)
    print("DEVELOPING PROFILE CACHE (MARKETTICK) BUILD COMPLETE")
    print("=" * 70)
    print(f"Sessions processed:    {done}")
    print(f"  newly written:       {ok}")
    print(f"  errors:              {errors}")
    print(f"Total cached files:    {cached_files}")
    print(f"Time elapsed:          {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"Cache location:        {CACHE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--test":
        run_test(sys.argv[2])
    else:
        main()

"""
Fast VWAP cache builder — vectorized version.
Skips already-cached dates, picks up where the slow version left off.

Key speedups vs original:
1. Bulk string slicing for timestamp parsing (no per-tick pd.Timestamp)
2. Numpy arrays for all cumulative math
3. Only create DatetimeIndex at resample step

Source: D:/trading_pythonbacktest_data/NQ L2 data/*.zip
Output: D:/trading_pythonbacktest_data/vwap_cache_5yr/{YYYY-MM-DD}.pkl
"""
import zipfile
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import time
import io

ZIP_DIR = Path("D:/trading_pythonbacktest_data/NQ L2 data")
CACHE_DIR = Path("D:/trading_pythonbacktest_data/vwap_cache_5yr")
ET = "America/New_York"


def extract_trades_fast(zf, csv_name):
    """
    Extract trades from a CSV inside a zip. Returns numpy arrays (ts_int, price, size).
    ts_int is YYYYMMDDHHMMSS as int64 for fast session assignment.
    """
    data = zf.read(csv_name)
    lines = data.split(b'\n')

    ts_list = []
    price_list = []
    size_list = []

    for line in lines:
        if not line:
            continue
        # Split on semicolons
        parts = line.split(b';')
        if len(parts) < 5:
            continue
        if parts[1] != b'2':  # Only trades
            continue
        try:
            size = int(parts[4])
        except (ValueError, IndexError):
            continue
        if size <= 0:
            continue
        ts_raw = parts[0]
        if len(ts_raw) < 14:
            continue
        try:
            price = float(parts[3])
        except (ValueError, IndexError):
            continue
        # Store timestamp as integer YYYYMMDDHHMMSS (first 14 chars)
        ts_list.append(ts_raw[:14])
        price_list.append(price)
        size_list.append(size)

    if not ts_list:
        return None, None, None

    prices = np.array(price_list, dtype=np.float64)
    sizes = np.array(size_list, dtype=np.float64)

    return ts_list, prices, sizes


def ts_bytes_to_components(ts_list):
    """Convert list of b'YYYYMMDDHHMMSS' to arrays of (date_int, hour)."""
    # Vectorized: decode all at once
    n = len(ts_list)
    date_ints = np.empty(n, dtype=np.int32)
    hours = np.empty(n, dtype=np.int8)

    for i, ts in enumerate(ts_list):
        date_ints[i] = int(ts[:8])
        hours[i] = int(ts[8:10])

    return date_ints, hours


def get_session_dates_vectorized(date_ints, hours):
    """
    Vectorized session date assignment.
    UTC hour >= 22 (6pm ET in winter) or >= 21 (6pm ET in summer) means next session.
    Simplified: if UTC hour >= 22, session = next calendar day. Otherwise session = same day.
    (This is approximate but covers most cases correctly for CME futures.)
    """
    # More accurate: convert to ET conceptually
    # UTC 22:00 = ET 17:00 (winter) or ET 18:00 (summer)
    # Session starts at 18:00 ET = 22:00/23:00 UTC depending on DST
    # Simpler: if hour >= 22 UTC, assign to next day's session
    # if hour < 22 UTC, assign to same day's session
    # Edge case: during DST, session start is 22:00 UTC. During EST, it's 23:00 UTC.
    # For safety, use hour >= 22 (catches both)
    next_day_mask = hours >= 22
    session_dates = date_ints.copy()
    # For next-day entries, add 1 day
    if next_day_mask.any():
        for i in np.where(next_day_mask)[0]:
            d = datetime.strptime(str(date_ints[i]), "%Y%m%d") + timedelta(days=1)
            session_dates[i] = int(d.strftime("%Y%m%d"))
    return session_dates


def compute_vwap_1min(ts_list, prices, sizes):
    """
    Compute VWAP + std bands, resample to 1-min. Vectorized.
    ts_list: list of b'YYYYMMDDHHMMSS...' byte strings
    Returns DataFrame with 1-min UTC index.
    """
    n = len(ts_list)
    cum_vol = np.cumsum(sizes)
    cum_pv = np.cumsum(prices * sizes)
    cum_pv2 = np.cumsum(prices ** 2 * sizes)

    vwap = cum_pv / cum_vol
    variance = np.maximum((cum_pv2 / cum_vol) - vwap ** 2, 0.0)
    std = np.sqrt(variance)

    # Build minute keys from timestamps (YYYYMMDDHHMM as string)
    # Take last trade per minute
    min_keys = [ts[:12] for ts in ts_list]

    # Find last index per minute
    last_per_min = {}
    for i, mk in enumerate(min_keys):
        last_per_min[mk] = i

    sorted_mins = sorted(last_per_min.keys())
    indices = [last_per_min[mk] for mk in sorted_mins]

    # Extract values at those indices
    vwap_1m = vwap[indices]
    std_1m = std[indices]

    # Parse minute timestamps to DatetimeIndex
    timestamps = pd.to_datetime(
        [m.decode() if isinstance(m, bytes) else m for m in sorted_mins],
        format="%Y%m%d%H%M",
        utc=True
    )

    df = pd.DataFrame({
        "vwap": vwap_1m,
        "std": std_1m,
        "std1_upper": vwap_1m + std_1m,
        "std1_lower": vwap_1m - std_1m,
        "std2_upper": vwap_1m + 2 * std_1m,
        "std2_lower": vwap_1m - 2 * std_1m,
        "std3_upper": vwap_1m + 3 * std_1m,
        "std3_lower": vwap_1m - 3 * std_1m,
    }, index=timestamps)
    df.index.name = "ts_event"

    # Forward-fill to regular 1-min grid
    full_idx = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="UTC")
    df = df.reindex(full_idx).ffill()

    return df


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files")
    print(f"Output: {CACHE_DIR}")

    # Existing cache
    existing = set(f.stem for f in CACHE_DIR.glob("*.pkl"))
    print(f"Already cached: {len(existing)} sessions")

    # Collect all CSV entries
    all_csv_entries = []
    for zf_path in zip_files:
        with zipfile.ZipFile(zf_path) as zf:
            for name in zf.namelist():
                if name.endswith('.csv'):
                    basename = name.split('/')[-1].replace('.csv', '')
                    if len(basename) == 8 and basename.isdigit():
                        all_csv_entries.append((zf_path, name, basename))

    all_csv_entries.sort(key=lambda x: x[2])
    print(f"Found {len(all_csv_entries)} CSV files")

    # Group by calendar date
    csv_by_date = defaultdict(list)
    for zf_path, csv_name, date_str in all_csv_entries:
        csv_by_date[date_str].append((zf_path, csv_name))

    calendar_dates = sorted(csv_by_date.keys())
    print(f"Calendar dates: {calendar_dates[0]} to {calendar_dates[-1]}")

    start = time.time()
    total_trades = 0
    total_sessions = 0
    skipped = 0

    # Accumulate trades by session date
    session_trades = defaultdict(lambda: ([], [], []))  # session_date_int -> (ts_list, prices, sizes)

    for idx, cal_date_str in enumerate(calendar_dates):
        entries = csv_by_date[cal_date_str]

        # Load trades from all CSVs for this calendar date
        day_ts = []
        day_prices = []
        day_sizes = []

        for zf_path, csv_name in entries:
            with zipfile.ZipFile(zf_path) as zf:
                ts, pr, sz = extract_trades_fast(zf, csv_name)
            if ts is None:
                continue
            day_ts.extend(ts)
            day_prices.append(pr)
            day_sizes.append(sz)

        if not day_ts:
            continue

        prices_arr = np.concatenate(day_prices)
        sizes_arr = np.concatenate(day_sizes)
        total_trades += len(day_ts)

        # Assign to sessions
        date_ints, hours = ts_bytes_to_components(day_ts)
        session_dates = get_session_dates_vectorized(date_ints, hours)

        # Group by session date
        unique_sessions = np.unique(session_dates)
        for sd_int in unique_sessions:
            mask = session_dates == sd_int
            sd_str = str(sd_int)
            sd_iso = f"{sd_str[:4]}-{sd_str[4:6]}-{sd_str[6:8]}"

            # Skip if already cached
            if sd_iso in existing:
                continue

            ts_sub = [day_ts[i] for i in np.where(mask)[0]]
            pr_sub = prices_arr[mask]
            sz_sub = sizes_arr[mask]

            old_ts, old_pr, old_sz = session_trades[sd_int]
            old_ts.extend(ts_sub)
            session_trades[sd_int] = (old_ts,
                                       np.concatenate([old_pr, pr_sub]) if len(old_pr) else pr_sub,
                                       np.concatenate([old_sz, sz_sub]) if len(old_sz) else sz_sub)

        # Finalize sessions that are complete (session date <= calendar date)
        cal_date_int = int(cal_date_str)
        sessions_to_finalize = [sd for sd in list(session_trades.keys()) if sd <= cal_date_int]

        for sd_int in sorted(sessions_to_finalize):
            sd_str = str(sd_int)
            sd_iso = f"{sd_str[:4]}-{sd_str[4:6]}-{sd_str[6:8]}"

            if sd_iso in existing:
                del session_trades[sd_int]
                skipped += 1
                continue

            ts_all, pr_all, sz_all = session_trades.pop(sd_int)

            if len(ts_all) < 10:
                continue

            # Sort by timestamp
            sort_idx = sorted(range(len(ts_all)), key=lambda i: ts_all[i])
            ts_sorted = [ts_all[i] for i in sort_idx]
            pr_sorted = pr_all[sort_idx]
            sz_sorted = sz_all[sort_idx]

            vwap_df = compute_vwap_1min(ts_sorted, pr_sorted, sz_sorted)
            if vwap_df.empty:
                continue

            cache_file = CACHE_DIR / f"{sd_iso}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(vwap_df, f)
            total_sessions += 1
            existing.add(sd_iso)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start
            pending = len(session_trades)
            print(f"  [{idx+1}/{len(calendar_dates)}] {cal_date_str}  "
                  f"{elapsed:.0f}s  new_sessions={total_sessions}  skipped={skipped}  "
                  f"pending={pending}  trades={total_trades:,}", flush=True)

    # Finalize remaining
    for sd_int in sorted(session_trades.keys()):
        sd_str = str(sd_int)
        sd_iso = f"{sd_str[:4]}-{sd_str[4:6]}-{sd_str[6:8]}"
        if sd_iso in existing:
            continue
        ts_all, pr_all, sz_all = session_trades[sd_int]
        if len(ts_all) < 10:
            continue
        sort_idx = sorted(range(len(ts_all)), key=lambda i: ts_all[i])
        ts_sorted = [ts_all[i] for i in sort_idx]
        pr_sorted = pr_all[sort_idx]
        sz_sorted = sz_all[sort_idx]
        vwap_df = compute_vwap_1min(ts_sorted, pr_sorted, sz_sorted)
        if vwap_df.empty:
            continue
        cache_file = CACHE_DIR / f"{sd_iso}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(vwap_df, f)
        total_sessions += 1

    elapsed = time.time() - start
    total_cached = len(list(CACHE_DIR.glob("*.pkl")))

    print()
    print("=" * 70)
    print("VWAP CACHE BUILD COMPLETE")
    print("=" * 70)
    print(f"Total trades processed:  {total_trades:,}")
    print(f"Calendar days:           {len(calendar_dates)}")
    print(f"Sessions newly cached:   {total_sessions}")
    print(f"Sessions skipped (exist):{skipped}")
    print(f"Total cached files:      {total_cached}")
    print(f"Time elapsed:            {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Cache location:          {CACHE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

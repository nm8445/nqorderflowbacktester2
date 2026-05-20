"""
Build per-day session VWAP cache from MarketTick L2 zip files.

Session VWAP anchors at 6pm ET previous day, runs through ~5pm ET.
Streams trades (Type=2) from zips, computes cumulative VWAP + std bands
tick-by-tick, resamples to 1-min for fast lookups.

Source: D:/trading_pythonbacktest_data/NQ L2 data/*.zip
Output: D:/trading_pythonbacktest_data/vwap_cache_5yr/{YYYY-MM-DD}.pkl

Each pkl = DataFrame indexed by UTC timestamp (1-min), columns:
  vwap, std, std1_upper, std1_lower, std2_upper, std2_lower, std3_upper, std3_lower
"""
import zipfile
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import time
import gc

ZIP_DIR = Path("D:/trading_pythonbacktest_data/NQ L2 data")
CACHE_DIR = Path("D:/trading_pythonbacktest_data/vwap_cache_5yr")
ET = "America/New_York"


def parse_ts(ts_str):
    """Parse MarketTick timestamp YYYYMMDDHHMMSSNNNNNN -> pd.Timestamp (UTC)."""
    return pd.Timestamp(
        year=int(ts_str[0:4]), month=int(ts_str[4:6]), day=int(ts_str[6:8]),
        hour=int(ts_str[8:10]), minute=int(ts_str[10:12]), second=int(ts_str[12:14]),
        tz="UTC"
    )


def extract_trades_from_csv(zf, csv_name):
    """Stream one CSV, return list of (utc_timestamp, price, size) for trades only."""
    trades = []
    with zf.open(csv_name) as f:
        for raw_line in f:
            parts = raw_line.decode('utf-8', errors='replace').strip().split(';')
            if len(parts) < 5:
                continue
            if parts[1] != '2':  # Only trades
                continue
            try:
                price = float(parts[3])
                size = int(parts[4])
            except ValueError:
                continue
            if size <= 0:
                continue
            ts_str = parts[0]
            if len(ts_str) < 14:
                continue
            trades.append((ts_str, price, size))
    return trades


def compute_vwap_from_trades(trades_list):
    """
    Compute cumulative VWAP + std bands from trade list, resample to 1-min.

    trades_list: list of (ts_str, price, size) sorted by time
    Returns: DataFrame with 1-min index, columns: vwap, std, std1_upper/lower, etc.
    """
    if not trades_list:
        return pd.DataFrame()

    prices = np.array([t[1] for t in trades_list], dtype=np.float64)
    sizes = np.array([t[2] for t in trades_list], dtype=np.float64)

    cum_vol = np.cumsum(sizes)
    cum_pv = np.cumsum(prices * sizes)
    cum_pv2 = np.cumsum(prices ** 2 * sizes)

    vwap = cum_pv / cum_vol
    variance = np.maximum((cum_pv2 / cum_vol) - vwap ** 2, 0.0)
    std = np.sqrt(variance)

    # Parse timestamps
    timestamps = pd.DatetimeIndex([parse_ts(t[0]) for t in trades_list])

    df = pd.DataFrame({
        "vwap": vwap, "std": std,
        "std1_upper": vwap + std, "std1_lower": vwap - std,
        "std2_upper": vwap + 2 * std, "std2_lower": vwap - 2 * std,
        "std3_upper": vwap + 3 * std, "std3_lower": vwap - 3 * std,
    }, index=timestamps)
    df.index.name = "ts_event"

    # Resample to 1-min, forward fill
    df_1min = df.resample("1min").last().ffill()
    return df_1min


def get_session_date(utc_ts):
    """
    Determine CME session date for a UTC timestamp.
    Session runs 6pm ET previous day -> ~5pm ET current day.
    A trade at 10pm UTC on Jan 5 = 5pm ET Jan 5 = session date Jan 6
    A trade at 2pm UTC on Jan 5 = 9am ET Jan 5 = session date Jan 5
    """
    et_ts = utc_ts.tz_convert(ET) if utc_ts.tzinfo else pd.Timestamp(utc_ts, tz="UTC").tz_convert(ET)
    if et_ts.hour >= 18:
        return (et_ts + timedelta(days=1)).date()
    else:
        return et_ts.date()


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files")
    print(f"Output: {CACHE_DIR}")

    # First pass: collect all trades grouped by session date
    # We process zip by zip, CSV by CSV, and bucket trades by session date
    # Then compute VWAP per session date

    start = time.time()
    total_trades = 0
    total_csvs = 0
    total_sessions = 0
    skipped = 0

    # Process each CSV (each CSV = 1 calendar day of ticks)
    # Collect all CSVs across all zips first
    all_csv_entries = []  # (zip_path, csv_name, date_str)
    for zf_path in zip_files:
        with zipfile.ZipFile(zf_path) as zf:
            for name in zf.namelist():
                if name.endswith('.csv'):
                    # Extract date from path like 202012/20201221.csv
                    basename = name.split('/')[-1].replace('.csv', '')
                    if len(basename) == 8 and basename.isdigit():
                        all_csv_entries.append((zf_path, name, basename))

    all_csv_entries.sort(key=lambda x: x[2])
    print(f"Found {len(all_csv_entries)} CSV files across all zips")

    # Group CSVs by calendar date
    csv_by_date = defaultdict(list)
    for zf_path, csv_name, date_str in all_csv_entries:
        csv_by_date[date_str].append((zf_path, csv_name))

    calendar_dates = sorted(csv_by_date.keys())
    print(f"Calendar dates: {calendar_dates[0]} to {calendar_dates[-1]}")

    # We need to process pairs of consecutive calendar days to build each session.
    # Session for trading date D starts at 6pm ET on (D-1) and ends ~5pm ET on D.
    # So we need trades from calendar day (D-1) after 6pm ET + calendar day D before 6pm ET.

    # First, load all trades per calendar day with timestamps
    # To be memory efficient, we process in windows of 2 calendar days

    # Get sorted unique session dates from the data range
    first_cal = datetime.strptime(calendar_dates[0], "%Y%m%d").date()
    last_cal = datetime.strptime(calendar_dates[-1], "%Y%m%d").date()
    print(f"Date range: {first_cal} to {last_cal}")

    # Process day by day: for each calendar date, load its trades,
    # assign them to session dates, accumulate
    session_trades = defaultdict(list)  # session_date -> [(ts_str, price, size), ...]
    prev_date_trades = []  # trades from previous calendar day that belong to next session

    for idx, cal_date_str in enumerate(calendar_dates):
        entries = csv_by_date[cal_date_str]

        # Load trades from all CSVs for this calendar date
        day_trades = []
        for zf_path, csv_name in entries:
            with zipfile.ZipFile(zf_path) as zf:
                day_trades.extend(extract_trades_from_csv(zf, csv_name))
        day_trades.sort(key=lambda x: x[0])

        trade_count = len(day_trades)
        total_trades += trade_count
        total_csvs += 1

        # Split trades by session date
        for t in day_trades:
            ts_str = t[0]
            utc_ts = parse_ts(ts_str)
            sess_date = get_session_date(utc_ts)
            session_trades[sess_date].append(t)

        # Check if we can finalize any session dates
        # A session is complete once we've processed the calendar day matching that session date
        cal_date = datetime.strptime(cal_date_str, "%Y%m%d").date()

        # Finalize sessions whose date <= cal_date (their evening portion is from cal_date-1, already loaded)
        sessions_to_finalize = [sd for sd in session_trades if sd <= cal_date]

        for sess_date in sorted(sessions_to_finalize):
            cache_file = CACHE_DIR / f"{sess_date}.pkl"
            if cache_file.exists():
                del session_trades[sess_date]
                skipped += 1
                continue

            trades = session_trades.pop(sess_date)
            trades.sort(key=lambda x: x[0])

            if len(trades) < 10:
                continue

            vwap_df = compute_vwap_from_trades(trades)
            if vwap_df.empty:
                continue

            with open(cache_file, 'wb') as f:
                pickle.dump(vwap_df, f)
            total_sessions += 1

            del trades

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start
            pending = len(session_trades)
            print(f"  [{idx+1}/{len(calendar_dates)}] {cal_date_str}  "
                  f"{elapsed:.0f}s  sessions_saved={total_sessions}  skipped={skipped}  "
                  f"pending={pending}  trades_so_far={total_trades:,}", flush=True)
            gc.collect()

    # Finalize remaining sessions
    for sess_date in sorted(session_trades.keys()):
        cache_file = CACHE_DIR / f"{sess_date}.pkl"
        if cache_file.exists():
            skipped += 1
            continue
        trades = session_trades[sess_date]
        trades.sort(key=lambda x: x[0])
        if len(trades) < 10:
            continue
        vwap_df = compute_vwap_from_trades(trades)
        if vwap_df.empty:
            continue
        with open(cache_file, 'wb') as f:
            pickle.dump(vwap_df, f)
        total_sessions += 1

    elapsed = time.time() - start
    cached_files = len(list(CACHE_DIR.glob("*.pkl")))

    print()
    print("=" * 70)
    print("VWAP CACHE BUILD COMPLETE")
    print("=" * 70)
    print(f"Total trades processed:  {total_trades:,}")
    print(f"Calendar days:           {len(calendar_dates)}")
    print(f"Sessions newly cached:   {total_sessions}")
    print(f"Sessions skipped (exist):{skipped}")
    print(f"Total cached files:      {cached_files}")
    print(f"Time elapsed:            {elapsed:.0f}s")
    print(f"Cache location:          {CACHE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

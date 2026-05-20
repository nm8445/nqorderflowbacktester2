"""
Build 5-min time bars from MarketTick L2 zip files.

Each bar has OHLCV + buy_vol/sell_vol/tick_count.
Session: 6pm ET prev day → 5pm ET. One pkl per session date.

MarketTick side: subtype=1 → buy, subtype=0 → sell.

Source: D:/trading_pythonbacktest_data/NQ L2 data/*.zip
Output: D:/trading_pythonbacktest_data/timebars_5min_5yr/timebars_5min_{YYYY}_{MM}_{DD}.pkl

Usage:
    python -u scripts/cache_creation_scripts/build_5min_from_markettick.py
"""
import zipfile
import pickle
import gc
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

ZIP_DIR = Path("D:/trading_pythonbacktest_data/NQ L2 data")
CACHE_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min_5yr")
ET = "America/New_York"


def parse_utc_ts(ts_str):
    return pd.Timestamp(
        year=int(ts_str[0:4]), month=int(ts_str[4:6]), day=int(ts_str[6:8]),
        hour=int(ts_str[8:10]), minute=int(ts_str[10:12]), second=int(ts_str[12:14]),
        tz="UTC",
    )


def get_session_date(utc_ts):
    et_ts = utc_ts.tz_convert(ET)
    if et_ts.hour >= 18:
        return (et_ts + timedelta(days=1)).date()
    return et_ts.date()


def extract_trades_with_side(zf, csv_name):
    trades = []
    with zf.open(csv_name) as f:
        for raw_line in f:
            parts = raw_line.decode('utf-8', errors='replace').strip().split(';')
            if len(parts) < 5 or parts[1] != '2':
                continue
            subtype = parts[2]
            if subtype == '1':
                is_buy = True
            elif subtype == '0':
                is_buy = False
            else:
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
            trades.append((ts_str, price, size, is_buy))
    return trades


def build_5min_bars(trades_list):
    """
    Build 5-minute OHLCV bars from trade list.
    Returns list of dicts matching the Databento timebars_5min format.
    """
    if not trades_list:
        return []

    # Build a DataFrame for easy resampling
    timestamps = [parse_utc_ts(t[0]) for t in trades_list]
    prices = [t[1] for t in trades_list]
    sizes = [t[2] for t in trades_list]
    is_buys = [t[3] for t in trades_list]

    df = pd.DataFrame({
        "price": prices,
        "size": sizes,
        "is_buy": is_buys,
    }, index=pd.DatetimeIndex(timestamps))
    df = df.sort_index()

    # Resample to 5-min bars
    ohlc = df["price"].resample("5min").ohlc()
    vol = df["size"].resample("5min").sum()
    buy_vol = df.loc[df["is_buy"], "size"].resample("5min").sum()
    sell_vol = df.loc[~df["is_buy"], "size"].resample("5min").sum()
    tick_count = df["price"].resample("5min").count()

    # Combine and drop bars with no trades
    combined = pd.DataFrame({
        "open": ohlc["open"],
        "high": ohlc["high"],
        "low": ohlc["low"],
        "close": ohlc["close"],
        "volume": vol,
        "buy_vol": buy_vol.reindex(ohlc.index, fill_value=0).astype(int),
        "sell_vol": sell_vol.reindex(ohlc.index, fill_value=0).astype(int),
        "tick_count": tick_count,
    }).dropna(subset=["open"])

    # Convert to list of dicts (same format as Databento timebars_5min)
    bars = []
    for ts, row in combined.iterrows():
        bars.append({
            "open_time": ts,
            "close_time": ts + pd.Timedelta(minutes=5),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
            "buy_vol": int(row["buy_vol"]),
            "sell_vol": int(row["sell_vol"]),
            "tick_count": int(row["tick_count"]),
        })
    return bars


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files")
    print(f"Output: {CACHE_DIR}")

    all_csv_entries = []
    for zf_path in zip_files:
        with zipfile.ZipFile(zf_path) as zf:
            for name in zf.namelist():
                if name.endswith('.csv'):
                    basename = name.split('/')[-1].replace('.csv', '')
                    if len(basename) == 8 and basename.isdigit():
                        all_csv_entries.append((zf_path, name, basename))
    all_csv_entries.sort(key=lambda x: x[2])

    csv_by_date = defaultdict(list)
    for zf_path, csv_name, date_str in all_csv_entries:
        csv_by_date[date_str].append((zf_path, csv_name))

    calendar_dates = sorted(csv_by_date.keys())
    print(f"Calendar dates: {calendar_dates[0]} to {calendar_dates[-1]} ({len(calendar_dates)} days)")

    start_time = time.time()
    session_trades = defaultdict(list)
    total_sessions = 0
    skipped = 0

    for idx, cal_date_str in enumerate(calendar_dates):
        entries = csv_by_date[cal_date_str]

        day_trades = []
        for zf_path, csv_name in entries:
            with zipfile.ZipFile(zf_path) as zf:
                day_trades.extend(extract_trades_with_side(zf, csv_name))
        day_trades.sort(key=lambda x: x[0])

        for t in day_trades:
            utc_ts = parse_utc_ts(t[0])
            sess_date = get_session_date(utc_ts)
            session_trades[sess_date].append(t)

        cal_date = datetime.strptime(cal_date_str, "%Y%m%d").date()
        sessions_to_finalize = [sd for sd in session_trades if sd <= cal_date]

        for sess_date in sorted(sessions_to_finalize):
            fname = f"timebars_5min_{sess_date.strftime('%Y_%m_%d')}.pkl"
            cache_file = CACHE_DIR / fname
            if cache_file.exists():
                del session_trades[sess_date]
                skipped += 1
                continue

            trades = session_trades.pop(sess_date)
            trades.sort(key=lambda x: x[0])

            if len(trades) < 100:
                continue

            bars = build_5min_bars(trades)
            del trades

            if len(bars) < 10:
                continue

            with open(cache_file, 'wb') as f:
                pickle.dump(bars, f)
            total_sessions += 1

            del bars

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            pending = len(session_trades)
            print(f"  [{idx+1}/{len(calendar_dates)}] {cal_date_str}  "
                  f"{elapsed:.0f}s  saved={total_sessions}  skipped={skipped}  "
                  f"pending={pending}", flush=True)
            gc.collect()

    # Finalize remaining
    for sess_date in sorted(session_trades.keys()):
        fname = f"timebars_5min_{sess_date.strftime('%Y_%m_%d')}.pkl"
        cache_file = CACHE_DIR / fname
        if cache_file.exists():
            skipped += 1
            continue
        trades = session_trades[sess_date]
        trades.sort(key=lambda x: x[0])
        if len(trades) < 100:
            continue
        bars = build_5min_bars(trades)
        if len(bars) < 10:
            continue
        with open(cache_file, 'wb') as f:
            pickle.dump(bars, f)
        total_sessions += 1

    elapsed = time.time() - start_time
    cached_files = len(list(CACHE_DIR.glob("*.pkl")))

    print()
    print("=" * 70)
    print("5-MIN TIMEBARS (MARKETTICK) BUILD COMPLETE")
    print("=" * 70)
    print(f"Sessions newly cached:   {total_sessions}")
    print(f"Sessions skipped (exist):{skipped}")
    print(f"Total cached files:      {cached_files}")
    print(f"Time elapsed:            {elapsed:.0f}s")
    print(f"Cache location:          {CACHE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

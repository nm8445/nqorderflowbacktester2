"""
Build signal_cache_5yr + timebars_5min_5yr from MarketTick L2 zips.

Each worker reads its own session's zip files and builds range bars + 5-min bars.
Uses numba JIT for fast range bar boundary detection.
Uses imap_unordered to avoid straggler delays.

MarketTick side: parts[2]='1' → buy aggressor, parts[2]='0' → sell aggressor.

Usage:
    python -u scripts/cache_creation_scripts/build_all_caches_from_markettick.py
"""
import zipfile
import pickle
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nqbt.analysis.range_bars import RangeBar, BarLevel

ZIP_DIR = Path("D:/trading_pythonbacktest_data/NQ L2 data")
DATA_DIR = Path("D:/trading_pythonbacktest_data")
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache_5yr"
TIMEBAR_CACHE_DIR = DATA_DIR / "timebars_5min_5yr"

ET = "America/New_York"
TICK_SIZE = 0.25
RANGE_POINTS = 10.0  # 40 ticks * 0.25
LEVEL_SIZE = 1.25    # 5 ticks * 0.25
TICKS_PER_LEVEL = 5
N_WORKERS = 8


@njit(cache=True)
def find_bar_boundaries(prices, range_points):
    n = len(prices)
    ends = np.empty(n, dtype=np.int64)
    count = 0
    hi = prices[0]
    lo = prices[0]
    for i in range(n):
        p = prices[i]
        if p > hi: hi = p
        if p < lo: lo = p
        if hi - lo >= range_points - 1e-9:
            ends[count] = i
            count += 1
            if i + 1 < n:
                hi = prices[i + 1]
                lo = prices[i + 1]
    return ends[:count]

_dummy = find_bar_boundaries(np.array([1.0, 2.0, 15.0]), 10.0)


def parse_utc_ts(ts_str):
    return pd.Timestamp(
        year=int(ts_str[0:4]), month=int(ts_str[4:6]), day=int(ts_str[6:8]),
        hour=int(ts_str[8:10]), minute=int(ts_str[10:12]), second=int(ts_str[12:14]),
        tz="UTC",
    )

def get_session_date(ts_str):
    utc_ts = parse_utc_ts(ts_str)
    et_ts = utc_ts.tz_convert(ET)
    if et_ts.hour >= 18:
        return (et_ts + timedelta(days=1)).date()
    return et_ts.date()


def process_session(args):
    """Worker: read trades from zips, build range bars + 5-min bars, save."""
    sess_date, csv_entries = args
    sig_file = SIGNAL_CACHE_DIR / f"{sess_date}.pkl"
    tb_file = TIMEBAR_CACHE_DIR / f"timebars_5min_{sess_date.strftime('%Y_%m_%d')}.pkl"

    if sig_file.exists() and tb_file.exists():
        return {"date": str(sess_date), "status": "skip"}

    # Read trades from zip files
    all_trades = []
    for zf_path_str, csv_name in csv_entries:
        with zipfile.ZipFile(zf_path_str) as zf:
            with zf.open(csv_name) as f:
                for raw_line in f:
                    parts = raw_line.decode('utf-8', errors='replace').strip().split(';')
                    if len(parts) < 5 or parts[1] != '2':
                        continue
                    sub = parts[2]
                    if sub == '1': is_buy = True
                    elif sub == '0': is_buy = False
                    else: continue
                    try:
                        price = float(parts[3])
                        size = int(parts[4])
                    except ValueError: continue
                    if size <= 0: continue
                    ts_str = parts[0]
                    if len(ts_str) < 14: continue
                    all_trades.append((ts_str, price, size, is_buy))

    if not all_trades:
        return {"date": str(sess_date), "status": "no_trades"}

    all_trades.sort(key=lambda x: x[0])

    # Filter to this session only
    session_trades = [t for t in all_trades if get_session_date(t[0]) == sess_date]
    del all_trades

    if len(session_trades) < 100:
        return {"date": str(sess_date), "status": "too_few"}

    # Extract arrays
    ts_strs = [t[0] for t in session_trades]
    prices = np.array([t[1] for t in session_trades], dtype=np.float64)
    sizes = np.array([t[2] for t in session_trades], dtype=np.int64)
    is_buys = np.array([t[3] for t in session_trades], dtype=np.bool_)
    n = len(prices)
    del session_trades

    timestamps = [parse_utc_ts(s) for s in ts_strs]
    result = {"date": str(sess_date), "trades": n}

    # ── Range bars ──
    if not sig_file.exists():
        bar_ends = find_bar_boundaries(prices, RANGE_POINTS)
        bars = []
        bar_start = 0

        for bi in range(len(bar_ends)):
            be = bar_ends[bi]
            sl = slice(bar_start, be + 1)
            ps = prices[sl]; ss = sizes[sl]; bs = is_buys[sl]

            price_ticks = np.round(ps / TICK_SIZE).astype(np.int64)
            level_indices = price_ticks // TICKS_PER_LEVEL
            level_prices = np.round(level_indices * LEVEL_SIZE, 4)

            levels_obj = {}
            for lp in np.unique(level_prices):
                mask = level_prices == lp
                levels_obj[float(lp)] = BarLevel(price=float(lp),
                    buy_vol=int(ss[mask & bs].sum()), sell_vol=int(ss[mask & ~bs].sum()))

            bars.append(RangeBar(
                open_time=timestamps[bar_start], close_time=timestamps[be],
                open=float(ps[0]), high=float(ps.max()), low=float(ps.min()), close=float(ps[-1]),
                buy_vol=int(ss[bs].sum()), sell_vol=int(ss[~bs].sum()), levels=levels_obj))
            bar_start = be + 1

        if bar_start < n:
            sl = slice(bar_start, n)
            ps = prices[sl]; ss = sizes[sl]; bs = is_buys[sl]
            price_ticks = np.round(ps / TICK_SIZE).astype(np.int64)
            level_indices = price_ticks // TICKS_PER_LEVEL
            level_prices = np.round(level_indices * LEVEL_SIZE, 4)
            levels_obj = {}
            for lp in np.unique(level_prices):
                mask = level_prices == lp
                levels_obj[float(lp)] = BarLevel(price=float(lp),
                    buy_vol=int(ss[mask & bs].sum()), sell_vol=int(ss[mask & ~bs].sum()))
            bars.append(RangeBar(
                open_time=timestamps[bar_start], close_time=None,
                open=float(ps[0]), high=float(ps.max()), low=float(ps.min()), close=float(ps[-1]),
                buy_vol=int(ss[bs].sum()), sell_vol=int(ss[~bs].sum()), levels=levels_obj))

        if len(bars) >= 10:
            signals = []
            for i in range(len(bars) - 1):
                sb = bars[i]; cb = bars[i + 1]
                if not sb.closed or not cb.closed: continue
                sb_bear = sb.close < sb.open; sb_bull = sb.close > sb.open
                cb_bear = cb.close < cb.open; cb_bull = cb.close > cb.open
                if not sb_bear and not sb_bull: continue
                absorption = []
                for p, lv in sb.levels.items():
                    d = lv.delta
                    if sb_bear and d >= 30: absorption.append((p, d))
                    elif sb_bull and d <= -30: absorption.append((p, d))
                if not absorption: continue
                if sb_bear and cb_bear:
                    signals.append({"bar_index": i, "signal_bar_index": i, "confirm_bar_index": i+1,
                        "signal_bar": sb, "confirm_bar": cb, "signal_type": "buyer_absorbed",
                        "direction": "bearish", "confirm_time": cb.close_time, "absorption_levels": absorption})
                if sb_bull and cb_bull:
                    signals.append({"bar_index": i, "signal_bar_index": i, "confirm_bar_index": i+1,
                        "signal_bar": sb, "confirm_bar": cb, "signal_type": "seller_absorbed",
                        "direction": "bullish", "confirm_time": cb.close_time, "absorption_levels": absorption})

            with open(sig_file, 'wb') as f:
                pickle.dump({"date": sess_date, "bars": bars, "all_signals": signals,
                    "profiles": {}, "entry_signals": [], "num_bars": len(bars),
                    "num_all_signals": len(signals), "num_entry_signals": 0}, f)
            result["bars"] = len(bars)
            result["signals"] = len(signals)

    # ── 5-min bars ──
    if not tb_file.exists():
        ts_ns = np.array([t.value for t in timestamps], dtype=np.int64)
        five_min_ns = 5 * 60 * 1_000_000_000
        bucket_ns = (ts_ns // five_min_ns) * five_min_ns
        unique_buckets = np.unique(bucket_ns)
        bars_5min = []
        for bk in unique_buckets:
            mask = bucket_ns == bk
            ps = prices[mask]; ss = sizes[mask]; bs = is_buys[mask]
            bucket_ts = pd.Timestamp(int(bk), unit='ns', tz='UTC')
            bars_5min.append({"open_time": bucket_ts,
                "close_time": bucket_ts + pd.Timedelta(minutes=5),
                "open": float(ps[0]), "high": float(ps.max()),
                "low": float(ps.min()), "close": float(ps[-1]),
                "volume": int(ss.sum()), "buy_vol": int(ss[bs].sum()),
                "sell_vol": int(ss[~bs].sum()), "tick_count": int(mask.sum())})
        if len(bars_5min) >= 10:
            with open(tb_file, 'wb') as f:
                pickle.dump(bars_5min, f)
            result["5min_bars"] = len(bars_5min)

    result["status"] = "ok"
    return result


def main():
    SIGNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TIMEBAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Index all CSVs
    print(f"Indexing {ZIP_DIR}...", flush=True)
    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zips")

    all_csv_entries = []
    for zf_path in zip_files:
        with zipfile.ZipFile(zf_path) as zf:
            for name in zf.namelist():
                if name.endswith('.csv'):
                    basename = name.split('/')[-1].replace('.csv', '')
                    if len(basename) == 8 and basename.isdigit():
                        all_csv_entries.append((str(zf_path), name, basename))
    all_csv_entries.sort(key=lambda x: x[2])

    csv_by_caldate = defaultdict(list)
    for zf_path, csv_name, date_str in all_csv_entries:
        csv_by_caldate[date_str].append((zf_path, csv_name))
    calendar_dates = sorted(csv_by_caldate.keys())
    print(f"Calendar dates: {calendar_dates[0]} to {calendar_dates[-1]} ({len(calendar_dates)})")

    # Build session → CSV mapping
    session_csvs = defaultdict(list)
    for cal_date_str in calendar_dates:
        cal_date = datetime.strptime(cal_date_str, "%Y%m%d").date()
        for entry in csv_by_caldate[cal_date_str]:
            session_csvs[cal_date].append(entry)
            session_csvs[cal_date + timedelta(days=1)].append(entry)

    # Filter to sessions needing processing
    existing_sig = {f.stem for f in SIGNAL_CACHE_DIR.glob("*.pkl")}
    existing_tb = set()
    for f in TIMEBAR_CACHE_DIR.glob("*.pkl"):
        parts = f.stem.replace("timebars_5min_", "").split("_")
        if len(parts) == 3:
            existing_tb.add(f"{parts[0]}-{parts[1]}-{parts[2]}")

    tasks = []
    for sd in sorted(session_csvs.keys()):
        if str(sd) in existing_sig and str(sd) in existing_tb:
            continue
        tasks.append((sd, session_csvs[sd]))

    print(f"Existing: {len(existing_sig)} signal, {len(existing_tb)} 5min")
    print(f"Sessions to process: {len(tasks)}")
    print(f"Workers: {N_WORKERS}")

    if not tasks:
        print("All done!")
        return

    # Process with imap_unordered for best throughput
    print(f"\nProcessing...\n", flush=True)
    start = time.time()
    done = 0
    ok = 0

    with Pool(N_WORKERS) as pool:
        for result in pool.imap_unordered(process_session, tasks, chunksize=1):
            done += 1
            if result["status"] == "ok":
                ok += 1
            if done % 10 == 0 or done <= 5:
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(tasks)}] {result['date']}  "
                      f"{result.get('trades', 0):,} ticks  {result.get('bars', '-')} bars  "
                      f"{elapsed:.0f}s  eta={eta:.0f}s  ({rate:.2f}/s)", flush=True)

    elapsed = time.time() - start
    sig_count = len(list(SIGNAL_CACHE_DIR.glob("*.pkl")))
    tb_count = len(list(TIMEBAR_CACHE_DIR.glob("*.pkl")))

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"Sessions processed: {ok}")
    print(f"Signal cache files: {sig_count}")
    print(f"5-min bar files:    {tb_count}")
    print(f"Time:               {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()

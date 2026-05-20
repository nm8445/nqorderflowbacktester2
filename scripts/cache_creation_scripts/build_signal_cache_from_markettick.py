"""
Build signal_cache (40-range bars) from MarketTick L2 zip files.

Reads the MarketTick trade tape, infers aggressor side from the prevailing BBO,
groups by CME session date (6pm ET prev day → 5pm ET), builds 40-tick
(10-point) range bars with per-level buy/sell delta, detects absorption
signals, and caches per-day pkl files.

MarketTick format (see memory/reference_markettick_schema.md):
  type=1 records have 5 cols: ts;type;subtype;price;size
    sub=0 → BBO bid update
    sub=1 → BBO ask update
    sub=2 → TRADE print (no aggressor flag; inferred from BBO + tick rule)
    sub=3 → cumulative session volume counter (price=0, size=cum_vol)
    sub=4/5/7/8/9 → rare session stats (skip)
  type=2 records have 7 cols: ts;type;subtype;price;size;level;flag
    sub=0 → L2 bid-side depth event
    sub=1 → L2 ask-side depth event
    (NOT trades — ignore here)

Aggressor inference (Lee-Ready):
  - trade price >= best_ask → buy aggressor
  - trade price <= best_bid → sell aggressor
  - inside spread → tick rule vs prior trade (up=buy, down=sell, zero=carry)

Source: D:/trading_pythonbacktest_data/NQ L2 data/*.zip
Output: D:/trading_pythonbacktest_data/signal_cache_5yr/{YYYY-MM-DD}.pkl

Usage:
    python -u scripts/cache_creation_scripts/build_signal_cache_from_markettick.py
"""
import zipfile
import pickle
import gc
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

WORKERS = 8

# Add project root so we can import range_bars
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nqbt.analysis.range_bars import build_range_bars, RangeBar

ZIP_DIR = Path("D:/trading_pythonbacktest_data/NQ L2 data")
CACHE_DIR = Path("D:/trading_pythonbacktest_data/signal_cache_5yr")
ET = "America/New_York"


def parse_utc_ts(ts_str):
    """Parse MarketTick timestamp YYYYMMDDHHMMSS[ffffff] → pd.Timestamp (UTC) with sub-second precision."""
    y = int(ts_str[0:4]); mo = int(ts_str[4:6]); d = int(ts_str[6:8])
    h = int(ts_str[8:10]); mi = int(ts_str[10:12]); s = int(ts_str[12:14])
    frac = ts_str[14:]
    if frac:
        # Pad or truncate to microseconds (6 digits)
        frac = (frac + "000000")[:6]
        try:
            micro = int(frac)
        except ValueError:
            micro = 0
    else:
        micro = 0
    return pd.Timestamp(
        year=y, month=mo, day=d, hour=h, minute=mi, second=s,
        microsecond=micro, tz="UTC",
    )


def get_session_date(utc_ts):
    """CME session date: 6pm ET prev day → ~5pm ET. Trades after 6pm ET belong to next day."""
    et_ts = utc_ts.tz_convert(ET)
    if et_ts.hour >= 18:
        return (et_ts + timedelta(days=1)).date()
    return et_ts.date()


def extract_trades_with_side(zf, csv_name, state=None):
    """
    Stream a MarketTick CSV, track rolling BBO from type=1 sub=0/1 updates,
    and emit trades (type=1 sub=2) with aggressor inferred from the BBO.

    state is an optional dict carrying BBO / last-trade info across CSVs so the
    first trade of each new CSV can still get a correct aggressor. If None, a
    fresh state is created internally.

    Returns list of (ts_str, price, size, aggressor) where aggressor='buy'|'sell'.
    """
    if state is None:
        state = {"best_bid": None, "best_ask": None,
                 "last_trade_price": None, "last_aggressor": "buy"}

    trades = []
    with zf.open(csv_name) as f:
        for raw_line in f:
            parts = raw_line.decode("utf-8", errors="replace").strip().split(";")
            if len(parts) < 4:
                continue
            # Only type=1 records carry trade/BBO info; type=2 is L2 depth (skip).
            if parts[1] != "1":
                continue
            subtype = parts[2]
            try:
                price = float(parts[3])
            except ValueError:
                continue

            if subtype == "0":
                # BBO bid update
                state["best_bid"] = price
                continue
            if subtype == "1":
                # BBO ask update
                state["best_ask"] = price
                continue
            if subtype != "2":
                # sub=3 cumulative counter, sub=4/5/7/8/9 stats — skip
                continue

            # Trade print
            if len(parts) < 5:
                continue
            try:
                size = int(parts[4])
            except ValueError:
                continue
            if size <= 0:
                continue
            ts_str = parts[0]
            if len(ts_str) < 14:
                continue

            # Lee-Ready aggressor inference
            best_bid = state["best_bid"]
            best_ask = state["best_ask"]
            if best_ask is not None and price >= best_ask:
                aggressor = "buy"
            elif best_bid is not None and price <= best_bid:
                aggressor = "sell"
            else:
                # Inside spread (or no BBO yet): tick rule vs prior trade
                last_px = state["last_trade_price"]
                if last_px is None:
                    aggressor = state["last_aggressor"]
                elif price > last_px:
                    aggressor = "buy"
                elif price < last_px:
                    aggressor = "sell"
                else:
                    aggressor = state["last_aggressor"]

            trades.append((ts_str, price, size, aggressor))
            state["last_trade_price"] = price
            state["last_aggressor"] = aggressor

    return trades


def trades_to_dataframe(trades_list):
    """Convert list of (ts_str, price, size, aggressor) to normalized tick DataFrame."""
    if not trades_list:
        return pd.DataFrame()

    timestamps = [parse_utc_ts(t[0]) for t in trades_list]
    prices = [t[1] for t in trades_list]
    sizes = [t[2] for t in trades_list]
    aggressors = [t[3] for t in trades_list]

    df = pd.DataFrame({
        "price": prices,
        "size": sizes,
        "aggressor": pd.Categorical(aggressors, categories=["buy", "sell"]),
    }, index=pd.DatetimeIndex(timestamps, name="ts_event"))
    return df.sort_index()


def analyze_bar_characteristics(bar, min_delta=30):
    """Check absorption levels in a bar (same logic as build_signal_cache.py)."""
    closed_bearish = bar.close < bar.open
    closed_bullish = bar.close > bar.open
    if not closed_bearish and not closed_bullish:
        return {"absorption_levels": []}
    absorption_levels = []
    for price, lv in bar.levels.items():
        delta = lv.delta
        if closed_bearish and delta >= min_delta:
            absorption_levels.append((price, delta))
        elif closed_bullish and delta <= -min_delta:
            absorption_levels.append((price, delta))
    return {"absorption_levels": absorption_levels}


def detect_enhanced_signals(bars, min_delta=30):
    """Detect absorption signals (same as build_signal_cache.py)."""
    signals = []
    for i in range(len(bars) - 1):
        signal_bar = bars[i]
        confirm_bar = bars[i + 1]
        if not signal_bar.closed or not confirm_bar.closed:
            continue
        signal_chars = analyze_bar_characteristics(signal_bar, min_delta)
        signal_bearish = signal_bar.close < signal_bar.open
        signal_bullish = signal_bar.close > signal_bar.open
        confirm_bearish = confirm_bar.close < confirm_bar.open
        confirm_bullish = confirm_bar.close > confirm_bar.open
        has_abs = len(signal_chars["absorption_levels"]) > 0
        if signal_bearish and confirm_bearish and has_abs:
            signals.append({
                "bar_index": i, "signal_bar_index": i, "confirm_bar_index": i + 1,
                "signal_bar": signal_bar, "confirm_bar": confirm_bar,
                "signal_type": "buyer_absorbed", "direction": "bearish",
                "confirm_time": confirm_bar.close_time,
                "absorption_levels": signal_chars["absorption_levels"],
            })
        if signal_bullish and confirm_bullish and has_abs:
            signals.append({
                "bar_index": i, "signal_bar_index": i, "confirm_bar_index": i + 1,
                "signal_bar": signal_bar, "confirm_bar": confirm_bar,
                "signal_type": "seller_absorbed", "direction": "bullish",
                "confirm_time": confirm_bar.close_time,
                "absorption_levels": signal_chars["absorption_levels"],
            })
    return signals


def process_session(args):
    """Worker: build one session date's signal cache from the relevant MarketTick CSVs.
    args = (session_date_iso, [(zip_path_str, csv_name), ...]) — CSVs chronologically ordered."""
    session_date_iso, csv_entries = args
    cache_file = CACHE_DIR / f"{session_date_iso}.pkl"
    if cache_file.exists():
        return (session_date_iso, "cached", 0, 0)

    session_date_obj = datetime.strptime(session_date_iso, "%Y-%m-%d").date()

    # Stream both CSVs, carrying BBO state so quote warmup survives the day-boundary.
    state = {"best_bid": None, "best_ask": None,
             "last_trade_price": None, "last_aggressor": "buy"}
    all_trades = []
    for zip_path_str, csv_name in csv_entries:
        try:
            with zipfile.ZipFile(zip_path_str) as zf:
                rows = extract_trades_with_side(zf, csv_name, state=state)
            all_trades.extend(rows)
        except Exception as e:
            return (session_date_iso, f"error_reading_{csv_name}:{e}", 0, 0)

    if not all_trades:
        return (session_date_iso, "no_trades", 0, 0)

    all_trades.sort(key=lambda t: t[0])

    # Filter to trades belonging to this session (CME 6pm ET boundary)
    filtered = []
    for t in all_trades:
        try:
            utc_ts = parse_utc_ts(t[0])
        except Exception:
            continue
        if get_session_date(utc_ts) == session_date_obj:
            filtered.append(t)

    if len(filtered) < 100:
        return (session_date_iso, f"few_trades_{len(filtered)}", 0, 0)

    ticks_df = trades_to_dataframe(filtered)
    del filtered, all_trades

    bars = build_range_bars(ticks_df, range_ticks=40, ticks_per_level=5)
    del ticks_df

    if len(bars) < 10:
        return (session_date_iso, f"few_bars_{len(bars)}", len(bars), 0)

    all_signals = detect_enhanced_signals(bars, min_delta=30)

    cache_data = {
        "date": session_date_obj,
        "bars": bars,
        "all_signals": all_signals,
        "profiles": {},
        "entry_signals": [],
        "num_bars": len(bars),
        "num_all_signals": len(all_signals),
        "num_entry_signals": 0,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)

    return (session_date_iso, "ok", len(bars), len(all_signals))


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files")
    print(f"Output: {CACHE_DIR}")
    print(f"Workers: {WORKERS}")

    # Index all CSVs across zips (store zip path as str for pickling across workers)
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

    calendar_dates = sorted(csv_by_date.keys())
    print(f"Calendar dates: {calendar_dates[0]} to {calendar_dates[-1]} ({len(calendar_dates)} days)")

    # Build list of session-date tasks (Mon-Fri only — CME is dark Sat, Sun morning).
    first_cal = datetime.strptime(calendar_dates[0], "%Y%m%d").date()
    last_cal = datetime.strptime(calendar_dates[-1], "%Y%m%d").date()

    tasks = []
    cur = first_cal
    while cur <= last_cal + timedelta(days=1):
        if cur.weekday() < 5:  # Mon-Fri
            sess_iso = cur.strftime("%Y-%m-%d")
            prev_cal = (cur - timedelta(days=1)).strftime("%Y%m%d")
            curr_cal = cur.strftime("%Y%m%d")
            csv_entries = []
            if prev_cal in csv_by_date:
                csv_entries.extend(csv_by_date[prev_cal])
            if curr_cal in csv_by_date:
                csv_entries.extend(csv_by_date[curr_cal])
            if csv_entries:
                cache_file = CACHE_DIR / f"{sess_iso}.pkl"
                if not cache_file.exists():
                    tasks.append((sess_iso, csv_entries))
        cur += timedelta(days=1)

    print(f"Sessions to build: {len(tasks)}")
    existing = len(list(CACHE_DIR.glob("*.pkl")))
    print(f"Already cached: {existing}")
    print(flush=True)

    if not tasks:
        print("Nothing to do.")
        return

    start_time = time.time()
    done = 0
    ok = 0
    skipped = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_session, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            sess_iso = futures[fut]
            done += 1
            try:
                sd, status, nbars, nsigs = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "cached":
                    skipped += 1
                if done <= 5 or done % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta_min = ((len(tasks) - done) / rate / 60) if rate > 0 else 0.0
                    print(f"  [{done}/{len(tasks)}] {sd} {status}  "
                          f"bars={nbars} sigs={nsigs}  "
                          f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  ETA={eta_min:.1f}min",
                          flush=True)
            except Exception as e:
                errors += 1
                print(f"  ERROR {sess_iso}: {e}", flush=True)

    elapsed = time.time() - start_time
    cached_files = len(list(CACHE_DIR.glob("*.pkl")))

    print()
    print("=" * 70)
    print("SIGNAL CACHE (MARKETTICK) BUILD COMPLETE")
    print("=" * 70)
    print(f"Sessions processed:    {done}")
    print(f"  newly written:       {ok}")
    print(f"  already cached:      {skipped}")
    print(f"  errors:              {errors}")
    print(f"Total cached files:    {cached_files}")
    print(f"Time elapsed:          {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"Cache location:        {CACHE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Build CORRECT 5-min NQ bars from MarketTick CSV trade prints.

Reads only type=1 sub=2 (trade) and type=1 sub=0/1 (BBO bid/ask) records.
Classifies each trade with Lee-Ready (vs prevailing BBO) + tick-rule fallback.

Output schema (one pkl per session date in CACHE_DIR):
  open_time, close_time   (UTC, 5-min boundary)
  open, high, low, close  (from trade prices only)
  volume                  (total contracts traded)
  buy_vol, sell_vol       (contracts, aggressor-classified)
  uptick_count            (count of buy-aggressor TRADES, MultiCharts UpTicks)
  downtick_count          (count of sell-aggressor TRADES, MultiCharts DownTicks)
  trade_count             (total trades)

Session = 18:00 ET prev-day -> 17:00 ET (CME futures Sunday-Friday).

Run:
    python -u scripts/fabio_orb/build_trade_5min_bars.py
or with limit / parallelism:
    python -u scripts/fabio_orb/build_trade_5min_bars.py --workers 8 --limit 5
"""
from __future__ import annotations

import argparse
import pickle
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

ZIP_DIR = Path("D:/trading_pythonbacktest_data/NQ L2 data")
CACHE_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min_trades_5yr")
ET = "America/New_York"
BAR_SECONDS = 300  # 5 min


# ============================== file index ==============================

CSV_DATE_RE = re.compile(r"(\d{8})\.csv$")


def index_csvs() -> list[tuple[str, str, str]]:
    """
    Walk every zip in ZIP_DIR and list (zip_path, internal_csv_name, yyyymmdd).
    Sorted ascending by date.
    """
    items: list[tuple[str, str, str]] = []
    for zp in sorted(ZIP_DIR.glob("*.zip")):
        try:
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    m = CSV_DATE_RE.search(n)
                    if not m:
                        continue
                    items.append((str(zp), n, m.group(1)))
        except zipfile.BadZipFile:
            continue
    items.sort(key=lambda t: t[2])
    return items


# ============================== per-day worker ==============================

def parse_ts_utc(ts_str: bytes) -> datetime:
    """ts_str like 20240603093000123456 — parse to UTC datetime."""
    return datetime(
        int(ts_str[0:4]), int(ts_str[4:6]), int(ts_str[6:8]),
        int(ts_str[8:10]), int(ts_str[10:12]), int(ts_str[12:14]),
        tzinfo=timezone.utc,
    )


def bar_open_utc(ts_secs_epoch: int) -> int:
    """Floor a UTC epoch second to 5-min boundary."""
    return ts_secs_epoch - (ts_secs_epoch % BAR_SECONDS)


def session_date_et(ts_utc: datetime) -> str:
    """Map a UTC ts to session date string YYYYMMDD using 18:00 ET roll."""
    # convert utc -> et naively
    et = ts_utc.astimezone(tz=None)  # not reliable; use pytz-like via pd
    # safer: use pandas
    et = pd.Timestamp(ts_utc).tz_convert(ET)
    if et.hour >= 18:
        et = et + pd.Timedelta(days=1)
    return et.strftime("%Y%m%d")


def process_csv(args: tuple[str, str, str]) -> tuple[str, int, dict | None]:
    """
    Process one CSV file. Returns (yyyymmdd, n_trades, bars_dict) where
    bars_dict maps session_date_str -> list of bar dicts.
    """
    zip_path, csv_name, yyyymmdd = args
    bars_by_session: dict[str, dict[int, dict]] = defaultdict(dict)
    # per-bar accumulators keyed by bar_open_epoch
    # bar dict fields:
    #   open, high, low, close (price, scaled)
    #   volume, buy_vol, sell_vol
    #   uptick_count, downtick_count, trade_count
    n_trades = 0
    best_bid = None
    best_ask = None
    prev_price = None
    prev_cls = 1  # default to up
    try:
        with zipfile.ZipFile(zip_path) as z:
            with z.open(csv_name) as f:
                for raw in f:
                    # quick reject: line must start with ts then ;1; for trade or BBO
                    if len(raw) < 20:
                        continue
                    # find first two semicolons fast
                    s1 = raw.find(b";")
                    if s1 < 14:
                        continue
                    s2 = raw.find(b";", s1 + 1)
                    if s2 == -1:
                        continue
                    # type field
                    if raw[s1 + 1:s2] != b"1":
                        continue
                    s3 = raw.find(b";", s2 + 1)
                    if s3 == -1:
                        continue
                    sub = raw[s2 + 1:s3]
                    if sub == b"0":
                        # bid update
                        s4 = raw.find(b";", s3 + 1)
                        if s4 == -1:
                            continue
                        try:
                            best_bid = float(raw[s3 + 1:s4])
                        except ValueError:
                            continue
                    elif sub == b"1":
                        # ask update
                        s4 = raw.find(b";", s3 + 1)
                        if s4 == -1:
                            continue
                        try:
                            best_ask = float(raw[s3 + 1:s4])
                        except ValueError:
                            continue
                    elif sub == b"2":
                        # trade
                        s4 = raw.find(b";", s3 + 1)
                        if s4 == -1:
                            continue
                        s5 = raw.find(b";", s4 + 1)
                        if s5 == -1:
                            s5 = len(raw)
                        try:
                            price = float(raw[s3 + 1:s4])
                            size = int(raw[s4 + 1:s5])
                        except ValueError:
                            continue
                        if size <= 0:
                            continue
                        # classify aggressor
                        if best_ask is not None and price >= best_ask:
                            cls = 1  # buy aggressor (uptick)
                        elif best_bid is not None and price <= best_bid:
                            cls = 0  # sell aggressor (downtick)
                        elif prev_price is not None:
                            if price > prev_price:
                                cls = 1
                            elif price < prev_price:
                                cls = 0
                            else:
                                cls = prev_cls
                        else:
                            cls = 1
                        prev_cls = cls
                        prev_price = price

                        # timestamp parse — first 14 chars are YYYYMMDDHHMMSS
                        ts = raw[:14]
                        ts_utc = datetime(
                            int(ts[0:4]), int(ts[4:6]), int(ts[6:8]),
                            int(ts[8:10]), int(ts[10:12]), int(ts[12:14]),
                            tzinfo=timezone.utc,
                        )
                        epoch = int(ts_utc.timestamp())
                        bar_open = epoch - (epoch % BAR_SECONDS)

                        # session date (ET 18:00 roll)
                        et_hour_offset = -5 if ts_utc.month in (12, 1, 2, 11) else -4  # rough
                        # use pandas for safety once per session-detect — but per-trade we need speed.
                        # Use a cached session-date keyed by date(et).
                        # Simplification: use UTC date with -5h offset (close enough; pkls in original cache
                        # may use a slightly different roll, but session boundaries are loose here).
                        # Better: do exact ET conversion lazily.
                        et = pd.Timestamp(ts_utc).tz_convert(ET)
                        if et.hour >= 18:
                            sdate = (et + pd.Timedelta(days=1)).strftime("%Y%m%d")
                        else:
                            sdate = et.strftime("%Y%m%d")

                        session_bars = bars_by_session[sdate]
                        b = session_bars.get(bar_open)
                        if b is None:
                            b = {
                                "open_time": bar_open,
                                "open": price, "high": price, "low": price, "close": price,
                                "volume": 0,
                                "buy_vol": 0, "sell_vol": 0,
                                "uptick_count": 0, "downtick_count": 0,
                                "trade_count": 0,
                            }
                            session_bars[bar_open] = b
                        if price > b["high"]:
                            b["high"] = price
                        if price < b["low"]:
                            b["low"] = price
                        b["close"] = price
                        b["volume"] += size
                        b["trade_count"] += 1
                        if cls == 1:
                            b["buy_vol"] += size
                            b["uptick_count"] += 1
                        else:
                            b["sell_vol"] += size
                            b["downtick_count"] += 1
                        n_trades += 1
    except Exception as e:
        return (yyyymmdd, 0, None)

    # finalize: for each session date, produce sorted list of bars
    out: dict[str, list[dict]] = {}
    for sdate, sbars in bars_by_session.items():
        rows = []
        for bar_open, b in sorted(sbars.items()):
            close_t = bar_open + BAR_SECONDS
            rows.append({
                "open_time": pd.Timestamp(bar_open, unit="s", tz="UTC"),
                "close_time": pd.Timestamp(close_t, unit="s", tz="UTC"),
                "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"],
                "volume": b["volume"],
                "buy_vol": b["buy_vol"], "sell_vol": b["sell_vol"],
                "uptick_count": b["uptick_count"], "downtick_count": b["downtick_count"],
                "trade_count": b["trade_count"],
            })
        out[sdate] = rows
    return (yyyymmdd, n_trades, out)


# ============================== driver ==============================

def merge_into_session_files(per_csv_outputs: list[tuple[str, int, dict | None]]) -> None:
    """
    Combine outputs across CSVs (some sessions span midnight UTC and may appear
    in two CSV days) into per-session pkls.
    """
    merged: dict[str, dict[pd.Timestamp, dict]] = defaultdict(dict)
    for _, _, out in per_csv_outputs:
        if out is None:
            continue
        for sdate, rows in out.items():
            day_bars = merged[sdate]
            for r in rows:
                k = r["open_time"]
                if k in day_bars:
                    # merge (rare; same bar appearing in two CSV days)
                    a = day_bars[k]
                    a["high"] = max(a["high"], r["high"])
                    a["low"] = min(a["low"], r["low"])
                    a["close"] = r["close"]   # later wins; not exact, but rare
                    a["volume"] += r["volume"]
                    a["buy_vol"] += r["buy_vol"]
                    a["sell_vol"] += r["sell_vol"]
                    a["uptick_count"] += r["uptick_count"]
                    a["downtick_count"] += r["downtick_count"]
                    a["trade_count"] += r["trade_count"]
                else:
                    day_bars[k] = r

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sdate, day_bars in merged.items():
        rows = [day_bars[k] for k in sorted(day_bars.keys())]
        # filename matches existing format style
        yyyy = sdate[:4]; mm = sdate[4:6]; dd = sdate[6:8]
        out_file = CACHE_DIR / f"timebars_5min_{yyyy}_{mm}_{dd}.pkl"
        with open(out_file, "wb") as fp:
            pickle.dump(rows, fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only first N CSV files (debug)")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    print("Indexing CSVs...", flush=True)
    items = index_csvs()
    print(f"  {len(items)} CSV files across all zips", flush=True)

    if not args.no_skip_existing:
        existing_dates = {
            f.name.replace("timebars_5min_", "").replace(".pkl", "").replace("_", "")
            for f in CACHE_DIR.glob("timebars_5min_*.pkl")
        }
        # session date may differ from csv date; we use CSV date as proxy.
        # To be safe, skip ONLY if BOTH this csv-day and the next-csv-day's pkls exist.
        # Simpler: just always reprocess. Fast-skip is risky here. Default = full rebuild.
        # Provide --no-skip-existing as no-op for back-compat.
        pass

    if args.limit:
        items = items[:args.limit]
        print(f"Limited to first {args.limit} CSVs for debug", flush=True)

    print(f"Processing with {args.workers} workers...", flush=True)
    import time
    t0 = time.time()
    results: list[tuple[str, int, dict | None]] = []
    if args.workers == 1:
        for i, it in enumerate(items, 1):
            r = process_csv(it)
            results.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(items)}  last={it[2]} trades={r[1]:,}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_csv, items, chunksize=4), 1):
                results.append(r)
                if i % 20 == 0:
                    print(f"  {i}/{len(items)}  trades_in_last={r[1]:,}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"All CSVs processed in {time.time()-t0:.0f}s", flush=True)

    print("Merging session files...", flush=True)
    merge_into_session_files(results)
    n = len(list(CACHE_DIR.glob("timebars_5min_*.pkl")))
    print(f"Wrote {n} session pkl files to {CACHE_DIR}", flush=True)


if __name__ == "__main__":
    main()

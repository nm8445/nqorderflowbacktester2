"""Download NDX + NDXP option EOD prices and open_interest from ThetaData v3.

Standard plan blocks index-options greeks endpoints, so we pull pricing data
(option/history/eod) and open_interest. Greeks are derived locally later via
put-call parity for spot, IV inversion, and Black-Scholes (separate script).

Range: 2020-12-01 -> 2026-05-01.
Output:
    D:/trading_pythonbacktest_data/NDX_thetadata/{YYYY-MM-DD}/eod.parquet
    D:/trading_pythonbacktest_data/NDX_thetadata/{YYYY-MM-DD}/oi.parquet
Each parquet has a `root` column distinguishing NDX vs NDXP.
Resumable: skips dates where both files already exist.
"""

from __future__ import annotations

import datetime as dt
import io
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

BASE = "http://127.0.0.1:25503/v3"
OUT_ROOT = Path("D:/trading_pythonbacktest_data/NDX_thetadata")
START = dt.date(2020, 12, 1)
END   = dt.date(2026, 5, 20)  # auto-updated by run_daily.py
TIMEOUT = 90.0
MAX_RETRIES = 3
ROOTS = ("NDX", "NDXP")


def fetch(path: str, params: dict) -> pd.DataFrame:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = httpx.get(f"{BASE}{path}",
                          params={**params, "format": "ndjson"},
                          timeout=TIMEOUT)
            if r.status_code == 472:
                return pd.DataFrame()
            r.raise_for_status()
            if not r.text.strip():
                return pd.DataFrame()
            return pd.read_json(io.StringIO(r.text), lines=True)
        except (httpx.HTTPError, ValueError) as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {MAX_RETRIES}: {last_err}")


def trading_days(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def download_day(date: dt.date) -> tuple[str, int, int, float]:
    day_dir = OUT_ROOT / date.strftime("%Y-%m-%d")
    eod_path = day_dir / "eod.parquet"
    oi_path = day_dir / "oi.parquet"
    nodata_path = day_dir / ".no_data"
    if (eod_path.exists() and oi_path.exists()) or nodata_path.exists():
        return ("skip", -1, -1, 0.0)

    t0 = time.time()
    date_s = date.strftime("%Y%m%d")
    eod_parts = []
    oi_parts = []
    for root in ROOTS:
        eod = fetch("/option/history/eod",
                    {"symbol": root, "expiration": "*",
                     "start_date": date_s, "end_date": date_s})
        if not eod.empty:
            eod["root"] = root
            eod_parts.append(eod)
        oi = fetch("/option/history/open_interest",
                   {"symbol": root, "expiration": "*",
                    "start_date": date_s, "end_date": date_s})
        if not oi.empty:
            oi["root"] = root
            oi_parts.append(oi)

    if not eod_parts and not oi_parts:
        day_dir.mkdir(parents=True, exist_ok=True)
        nodata_path.touch()
        return ("nodata", 0, 0, time.time() - t0)

    day_dir.mkdir(parents=True, exist_ok=True)
    if eod_parts:
        eod_df = pd.concat(eod_parts, ignore_index=True)
        eod_df.to_parquet(eod_path, compression="zstd", index=False)
    if oi_parts:
        oi_df = pd.concat(oi_parts, ignore_index=True)
        oi_df.to_parquet(oi_path, compression="zstd", index=False)

    eod_n = sum(len(p) for p in eod_parts)
    oi_n = sum(len(p) for p in oi_parts)
    return ("ok", eod_n, oi_n, time.time() - t0)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    days = list(trading_days(START, END))
    print(f"target: {len(days)} weekdays from {START} to {END}")
    print(f"output: {OUT_ROOT}")
    print(f"roots:  {ROOTS}")
    print("-" * 70, flush=True)

    overall_t0 = time.time()
    counters = {"ok": 0, "skip": 0, "nodata": 0, "err": 0}
    bytes_written = 0

    for i, d in enumerate(days, 1):
        try:
            status, n_eod, n_oi, secs = download_day(d)
        except Exception as e:
            counters["err"] += 1
            print(f"[{i:>4}/{len(days)}] {d} ERROR: {type(e).__name__}: {e}", flush=True)
            continue
        counters[status] += 1

        if status == "ok":
            day_dir = OUT_ROOT / d.strftime("%Y-%m-%d")
            for f in day_dir.glob("*.parquet"):
                bytes_written += f.stat().st_size

        if i % 25 == 0 or status == "err" or i == len(days):
            elapsed = time.time() - overall_t0
            done = counters["ok"] + counters["skip"] + counters["nodata"]
            rate = done / elapsed if elapsed > 0 else 0
            eta_min = (len(days) - i) / rate / 60 if rate > 0 else 0
            print(f"[{i:>4}/{len(days)}] {d}  "
                  f"ok={counters['ok']}  skip={counters['skip']}  "
                  f"nodata={counters['nodata']}  err={counters['err']}  "
                  f"size={bytes_written / 1e6:>6.0f}MB  "
                  f"rate={rate:.1f}/s  eta={eta_min:.0f}min",
                  flush=True)

    elapsed = time.time() - overall_t0
    print("-" * 70)
    print(f"done in {elapsed/60:.1f} min")
    print(f"counters: {counters}")
    print(f"total written: {bytes_written / 1e6:.0f} MB")


if __name__ == "__main__":
    sys.exit(main())

"""Download QQQ EOD greeks + open_interest from ThetaData v3 (Standard tier).

Range: 2020-12-01 → today's last completed trading day.
Output: D:/trading_pythonbacktest_data/QQQ_thetadata/{YYYY-MM-DD}/{greeks_eod,open_interest}.parquet
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
OUT_ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
START = dt.date(2020, 12, 1)
END   = dt.date(2026, 5, 20)  # auto-updated by run_daily.py
TIMEOUT = 90.0
MAX_RETRIES = 3


def fetch(path: str, params: dict) -> pd.DataFrame:
    """GET ndjson with retries. Returns empty DataFrame on 472 (no data — holiday/weekend)."""
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
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_err}")


def trading_days(start: dt.date, end: dt.date):
    """Mon-Fri only. ThetaData returns empty on holidays — that's fine."""
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def download_day(date: dt.date) -> tuple[str, int, int, float]:
    """Returns (status, greeks_rows, oi_rows, elapsed_sec)."""
    day_dir = OUT_ROOT / date.strftime("%Y-%m-%d")
    g_path = day_dir / "greeks_eod.parquet"
    o_path = day_dir / "open_interest.parquet"
    if g_path.exists() and o_path.exists():
        return ("skip", -1, -1, 0.0)

    t0 = time.time()
    date_s = date.strftime("%Y%m%d")
    common = {"symbol": "QQQ", "expiration": "*",
              "start_date": date_s, "end_date": date_s}

    greeks = fetch("/option/history/greeks/eod", common)
    oi = fetch("/option/history/open_interest", common)

    if greeks.empty and oi.empty:
        # Holiday/non-trading day — write a marker file so we don't retry
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / ".no_data").touch()
        return ("nodata", 0, 0, time.time() - t0)

    day_dir.mkdir(parents=True, exist_ok=True)
    if not greeks.empty:
        greeks.to_parquet(g_path, compression="zstd", index=False)
    if not oi.empty:
        oi.to_parquet(o_path, compression="zstd", index=False)
    return ("ok", len(greeks), len(oi), time.time() - t0)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    days = list(trading_days(START, END))
    print(f"target: {len(days)} weekdays from {START} to {END}")
    print(f"output: {OUT_ROOT}")
    print("-" * 70, flush=True)

    overall_t0 = time.time()
    counters = {"ok": 0, "skip": 0, "nodata": 0, "err": 0}
    bytes_written = 0

    for i, d in enumerate(days, 1):
        try:
            status, gr, oi, secs = download_day(d)
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

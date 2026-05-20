"""Rolling 15-day warm-start cache.

Maintains a fixed-size cache of recent NQ 5-min bars on disk so live engines
can warm-start fast without downloading from Databento every time.

Design:
  - One pickle per CME trading session date: `nq_5min_YYYY-MM-DD.pkl`
    Contains all 5-min bars for the session ending on that date (18:00 ET
    of D-1 through 17:00 ET of D — full ETH session).
  - Bar dicts match the existing pickle schema (read by warm_start.py):
       {open_time (UTC), open, high, low, close, buy_vol, sell_vol, tick_count}
  - On startup or daily refresh, fetch any missing days from Databento
    Historical API. Databento has a ~30-min delay on historical data, so
    we never fetch the CURRENT trading session — only days that ended
    >24h ago. Today's bars are built from the LIVE stream during the day
    and persisted at EOD (separate path — not implemented here).
  - Prune any pickle older than KEEP_TRADING_DAYS old.

Usage:
  cache = RollingWarmstartCache()
  status = cache.ensure_recent_cached(today=date.today())
  # status: {'fetched': [...], 'cached': [...], 'pruned': [...], 'missing': [...]}
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv

load_dotenv()   # ensure DATABENTO_API_KEY is loaded when this module is imported

from live.combined.config import (
    LIVE_WARMSTART_CACHE_DIR, ET_TZ, WARM_START_TRADING_DAYS,
    DATABENTO_DATASET, DATABENTO_SYMBOL, DATABENTO_SCHEMA, DATABENTO_STYPE,
)

KEEP_TRADING_DAYS = WARM_START_TRADING_DAYS   # how many days to retain on disk
CACHE_FILE_PATTERN = "nq_5min_{date}.pkl"


def _list_cached_dates(cache_dir: Path = LIVE_WARMSTART_CACHE_DIR) -> list[dt.date]:
    if not cache_dir.exists():
        return []
    dates = []
    for p in cache_dir.glob("nq_5min_*.pkl"):
        try:
            stem = p.stem.replace("nq_5min_", "")
            dates.append(dt.date.fromisoformat(stem))
        except ValueError:
            continue
    return sorted(dates)


def _trading_weekdays_back(today: dt.date, n: int) -> list[dt.date]:
    """Return the last `n` weekdays (Mon-Fri) that ended STRICTLY BEFORE today.
    Today's session is excluded because Databento has a 30-min delay and we
    can't fetch the current session reliably."""
    out = []
    d = today - dt.timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:   # Mon=0 .. Fri=4
            out.append(d)
        d -= dt.timedelta(days=1)
    return sorted(out)


def _file_for(date: dt.date, cache_dir: Path = LIVE_WARMSTART_CACHE_DIR) -> Path:
    return cache_dir / CACHE_FILE_PATTERN.format(date=date.isoformat())


def _aggregate_trades_to_5min_bars(trades: list[dict]) -> list[dict]:
    """Aggregate a list of trade dicts (each with ts_et, price, size, side)
    into 5-min bars keyed by floor(ts_et, '5min')."""
    if not trades:
        return []
    df = pd.DataFrame(trades)
    df["ts_et"] = pd.to_datetime(df["ts_et"])
    if df["ts_et"].dt.tz is None:
        df["ts_et"] = df["ts_et"].dt.tz_localize("UTC").dt.tz_convert(ET_TZ)
    else:
        df["ts_et"] = df["ts_et"].dt.tz_convert(ET_TZ)
    df["bar_open_et"] = df["ts_et"].dt.floor("5min")
    grouped = df.groupby("bar_open_et")
    bars = []
    for bar_open, g in grouped:
        buys  = g[g["side"] == "B"]["size"].sum()
        sells = g[g["side"] == "A"]["size"].sum()
        bars.append({
            "open_time": bar_open.tz_convert("UTC").tz_localize(None),   # naive UTC (matches existing pickles)
            "open":  float(g["price"].iloc[0]),
            "high":  float(g["price"].max()),
            "low":   float(g["price"].min()),
            "close": float(g["price"].iloc[-1]),
            "buy_vol":  int(buys),
            "sell_vol": int(sells),
            "tick_count": len(g),
        })
    return bars


def fetch_day_from_databento(date: dt.date,
                              end_override: dt.datetime | None = None) -> list[dict]:
    """Fetch one ETH session's trades from Databento Historical and aggregate
    into 5-min bars. ETH session = 18:00 ET (D-1) → 17:00 ET (D).

    end_override: if given, use this as the end timestamp instead of 17:00 ET.
        Used for fetching TODAY's partial session — caller passes (now - 30 min)
        to stay safely within Databento Historical's delay window.
    """
    import databento as db
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not in env")
    client = db.Historical(api_key)

    # Session window: 18:00 ET prior day → 17:00 ET this day (or override)
    prior_day = date - dt.timedelta(days=1)
    start_et = pd.Timestamp(f"{prior_day} 18:00", tz=ET_TZ)
    if end_override is not None:
        end_et = pd.Timestamp(end_override).tz_convert(ET_TZ) if pd.Timestamp(end_override).tzinfo else pd.Timestamp(end_override, tz=ET_TZ)
    else:
        end_et = pd.Timestamp(f"{date} 17:00", tz=ET_TZ)
    start_utc = start_et.tz_convert("UTC")
    end_utc   = end_et.tz_convert("UTC")

    print(f"  [cache] fetching {date} from Databento "
          f"({start_et.strftime('%Y-%m-%d %H:%M ET')} -> {end_et.strftime('%Y-%m-%d %H:%M ET')})...")

    df = client.timeseries.get_range(
        dataset=DATABENTO_DATASET,
        schema=DATABENTO_SCHEMA,   # mbp-1
        symbols=[DATABENTO_SYMBOL],
        stype_in=DATABENTO_STYPE,
        start=start_utc,
        end=end_utc,
    ).to_df()

    if df.empty:
        return []

    # Databento v3 returns ts_recv as the DataFrame INDEX, not a column.
    # Bring it into a column for consistent downstream handling.
    if "ts_recv" not in df.columns:
        df = df.reset_index()
    # If index was unnamed, the reset gives "index"; rename if needed
    if "ts_recv" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "ts_recv"})

    # MBP-1 includes book quotes too; keep only trades (action='T')
    df = df[df["action"] == "T"].copy()
    if df.empty:
        return []

    df["ts_et"] = pd.to_datetime(df["ts_recv"], utc=True)
    trades = [
        {
            "ts_et": ts,
            "price": float(p),
            "size":  int(sz),
            "side":  str(sd),
        }
        for ts, p, sz, sd in zip(df["ts_et"], df["price"], df["size"], df["side"])
    ]
    bars = _aggregate_trades_to_5min_bars(trades)
    return bars


@dataclass
class RollingWarmstartCache:
    cache_dir: Path = LIVE_WARMSTART_CACHE_DIR
    keep_days: int = KEEP_TRADING_DAYS

    def cached_dates(self) -> list[dt.date]:
        return _list_cached_dates(self.cache_dir)

    def needed_dates(self, today: dt.date) -> list[dt.date]:
        return _trading_weekdays_back(today, self.keep_days)

    def missing_dates(self, today: dt.date) -> list[dt.date]:
        cached = set(self.cached_dates())
        return [d for d in self.needed_dates(today) if d not in cached]

    def save_day(self, date: dt.date, bars: list[dict]) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = _file_for(date, self.cache_dir)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(bars, f)
        os.replace(tmp, path)   # atomic
        return path

    def fetch_and_save(self, date: dt.date) -> int:
        bars = fetch_day_from_databento(date)
        if not bars:
            print(f"  [cache] {date}: no trades returned (holiday or weekend?), skipping")
            return 0
        self.save_day(date, bars)
        print(f"  [cache] saved {date}: {len(bars)} bars")
        return len(bars)

    def prune_old(self, today: dt.date) -> list[dt.date]:
        """Delete cached pickles older than the keep window.

        NEVER deletes the current OR next-upcoming session date — those are
        either being built live right now (today's RTH session) or about to
        start (next CME session after 18:00 ET). Backfill thread relies on
        these pickles existing.
        """
        keep = set(self.needed_dates(today))
        # Always preserve today (current RTH session) AND tomorrow (next CME
        # session that opens at 18:00 ET today — already in progress if we're
        # past 18:00 ET).
        keep.add(today)
        keep.add(today + dt.timedelta(days=1))
        pruned = []
        for d in self.cached_dates():
            if d not in keep:
                _file_for(d, self.cache_dir).unlink(missing_ok=True)
                pruned.append(d)
        return pruned

    def ensure_recent_cached(self, today: dt.date | None = None) -> dict:
        """Main entry — call at startup AND nightly. Returns status dict."""
        today = today or dt.date.today()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        missing = self.missing_dates(today)
        fetched = []
        failed = []
        for d in missing:
            try:
                n_bars = self.fetch_and_save(d)
                if n_bars > 0:
                    fetched.append(d)
            except Exception as e:
                print(f"  [cache] FAILED to fetch {d}: {e}")
                failed.append(d)
                time.sleep(2)   # brief backoff before next day

        pruned = self.prune_old(today)
        final = self.cached_dates()

        status = {
            "today": today,
            "needed": self.needed_dates(today),
            "fetched": fetched,
            "failed": failed,
            "pruned": pruned,
            "cached_after": final,
            "cache_dir": str(self.cache_dir),
        }
        return status


def print_status(status: dict) -> None:
    print(f"\n=== Rolling Warmstart Cache Status ===")
    print(f"  Today:   {status['today']}")
    print(f"  Cache dir: {status['cache_dir']}")
    print(f"  Needed (last {WARM_START_TRADING_DAYS} weekdays): {len(status['needed'])} days")
    print(f"  Fetched this run: {len(status['fetched'])}  {status['fetched']}")
    if status['failed']:
        print(f"  FAILED: {status['failed']}")
    print(f"  Pruned:  {len(status['pruned'])}  {status['pruned']}")
    print(f"  Cached now: {len(status['cached_after'])} days "
          f"({status['cached_after'][0]} -> {status['cached_after'][-1]})"
          if status['cached_after'] else "  Cache empty!")


if __name__ == "__main__":
    cache = RollingWarmstartCache()
    status = cache.ensure_recent_cached()
    print_status(status)

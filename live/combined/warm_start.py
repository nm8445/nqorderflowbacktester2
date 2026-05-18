"""Warm-start: load historical bars from local pickle archive and feed them
to the bar builders so indicators (ATR, EMA, kernel features) are seeded
before going live."""
from __future__ import annotations

import datetime as dt
import pickle
from pathlib import Path
from typing import Iterator
import pandas as pd

from live.combined.config import (
    TIMEBARS_5MIN_DIRS, LIVE_WARMSTART_CACHE_DIR,
    ET_TZ, WARM_START_TRADING_DAYS,
)
from live.combined.bar_builder import MultiBarBuilder, Bar


def _list_5min_pickle_files() -> list[Path]:
    """Return all 5-min pickle files sorted by date, PREFERRING the live
    rolling cache over the research archive. Later folders override earlier
    ones for the same date.

    Lookup order (first = highest priority):
      1. LIVE_WARMSTART_CACHE_DIR / nq_5min_YYYY-MM-DD.pkl  (rolling 15-day cache)
      2. TIMEBARS_5MIN_DIRS[*] / timebars_5min_YYYY_MM_DD.pkl  (research archive)
    """
    by_date: dict[dt.date, Path] = {}

    # Research archive first (lower priority — gets overwritten by live cache below)
    for d in TIMEBARS_5MIN_DIRS:
        if not d.exists(): continue
        for f in sorted(d.glob("timebars_5min_*.pkl")):
            try:
                parts = f.stem.replace("timebars_5min_", "").split("_")
                file_date = dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
                by_date[file_date] = f
            except (ValueError, IndexError):
                continue

    # Live cache last so it OVERRIDES research archive for overlapping dates
    if LIVE_WARMSTART_CACHE_DIR.exists():
        for f in sorted(LIVE_WARMSTART_CACHE_DIR.glob("nq_5min_*.pkl")):
            try:
                file_date = dt.date.fromisoformat(f.stem.replace("nq_5min_", ""))
                by_date[file_date] = f
            except ValueError:
                continue

    return [by_date[k] for k in sorted(by_date.keys())]


def _file_date(p: Path) -> dt.date:
    """Extract date from filename. Handles both:
      - timebars_5min_YYYY_MM_DD.pkl (research archive)
      - nq_5min_YYYY-MM-DD.pkl (live cache)
    """
    if p.stem.startswith("nq_5min_"):
        return dt.date.fromisoformat(p.stem.replace("nq_5min_", ""))
    parts = p.stem.replace("timebars_5min_", "").split("_")
    return dt.date(int(parts[0]), int(parts[1]), int(parts[2]))


def load_warm_start_bars(n_trading_days: int = WARM_START_TRADING_DAYS) -> list[dict]:
    """Load the last N trading days' worth of 5-min bars from the pickle archive.

    Returns list of bar dicts: {open_time, open, high, low, close, buy_vol, sell_vol}
    """
    all_files = _list_5min_pickle_files()
    if not all_files:
        return []
    # Take the last N files (sorted by date already)
    selected = all_files[-n_trading_days:]
    print(f"[warm_start] loading {len(selected)} pickle days: "
          f"{_file_date(selected[0])} -> {_file_date(selected[-1])}")
    rows = []
    for p in selected:
        with open(p, "rb") as fh:
            day_bars = pickle.load(fh)
        rows.extend(day_bars)
    return rows


def feed_warm_start_to_builders(multi: MultiBarBuilder,
                                 n_trading_days: int = WARM_START_TRADING_DAYS) -> int:
    """Feed historical 5-min bars to the bar builders as if they were live ticks
    (one synthetic tick per bar = bar close + total volume).

    This is a fast warm-start — full tick-level replay would be much slower and
    isn't necessary for indicator seeding.

    Returns number of bars fed.
    """
    bars = load_warm_start_bars(n_trading_days)
    if not bars:
        print("[warm_start] no historical bars found — skipping seed")
        return 0
    n_fed = 0
    for b in bars:
        # Convert to ET-tz-aware timestamp
        ot = pd.Timestamp(b["open_time"])
        if ot.tz is None:
            ot = ot.tz_localize("UTC")
        ot_et = ot.tz_convert(ET_TZ)
        close_time = ot_et + pd.Timedelta(seconds=300)

        # Inject synthetic ticks at:
        #   1. bar open (price=open, volume=0 marker)
        #   2. bar high
        #   3. bar low
        #   4. bar close with full volume (split by buy/sell)
        # This gives the builder a complete OHLC + delta picture.
        # We use close_time - 1ms for the close tick so the bar finalizes at close.
        buy_vol = int(b.get("buy_vol", 0))
        sell_vol = int(b.get("sell_vol", 0))

        # All ticks fall WITHIN the bar window so the builder aggregates correctly.
        # We use small offsets from bar open:
        t0 = ot_et + pd.Timedelta(milliseconds=1)
        t1 = ot_et + pd.Timedelta(milliseconds=100)
        t2 = ot_et + pd.Timedelta(milliseconds=200)
        t3 = ot_et + pd.Timedelta(milliseconds=300)
        multi.on_tick(t0, float(b["open"]), 0, "N")
        if b["high"] != b["open"]:
            multi.on_tick(t1, float(b["high"]), 0, "N")
        if b["low"] != b["high"]:
            multi.on_tick(t2, float(b["low"]), 0, "N")
        # Final tick at close with the volume split
        if buy_vol > 0:
            multi.on_tick(t3, float(b["close"]), buy_vol, "B")
        if sell_vol > 0:
            multi.on_tick(t3, float(b["close"]), sell_vol, "A")
        if buy_vol == 0 and sell_vol == 0:
            multi.on_tick(t3, float(b["close"]), 0, "N")
        n_fed += 1

    # Force-close any in-progress bars so we have a clean state for live ticks
    for builder in multi.builders.values():
        builder.force_close_current()

    print(f"[warm_start] fed {n_fed} historical 5-min bars to builders")
    return n_fed

"""Live bar persistor — appends each closed 5-min bar to today's pickle.

Subscribes to the 5-min bar builder. As each bar closes, the bar dict is
appended to `nq_5min_YYYY-MM-DD.pkl` (atomic write).

CME session convention: a bar's session date = the date at which the
session ENDS (17:00 ET). So a bar at 18:05 ET Mon belongs to Tue's session
(because Mon 18:00 → Tue 17:00 is "Tue's session").

When the next session day starts (~18:00 ET), bars automatically route to
the new pickle filename. No rename needed.

Usage:
    persistor = LiveBarPersistor(cache_dir=LIVE_WARMSTART_CACHE_DIR)
    b5.subscribe(persistor.on_bar)
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from live.combined.config import LIVE_WARMSTART_CACHE_DIR, ET_TZ
from live.combined.bar_builder import Bar


def _session_date_for_bar(bar_open_et: pd.Timestamp) -> dt.date:
    """A bar's CME session date = next 17:00 ET cutoff.
    Bars with open_time in [18:00 D-1, 17:00 D) belong to session date D.
    """
    if bar_open_et.tzinfo is None:
        bar_open_et = bar_open_et.tz_localize(ET_TZ)
    else:
        bar_open_et = bar_open_et.tz_convert(ET_TZ)
    # If bar open is between 18:00 and 23:59:59, session date = next day
    if bar_open_et.hour >= 18:
        return (bar_open_et + pd.Timedelta(days=1)).date()
    return bar_open_et.date()


def _pickle_path(session_date: dt.date, cache_dir: Path) -> Path:
    return cache_dir / f"nq_5min_{session_date.isoformat()}.pkl"


def _bar_to_dict(bar: Bar) -> dict:
    """Convert Bar dataclass → pickle schema dict (naive UTC timestamps,
    matching the existing pickle format)."""
    ot = pd.Timestamp(bar.open_time)
    if ot.tz is None:
        ot_utc = ot.tz_localize("UTC")
    else:
        ot_utc = ot.tz_convert("UTC")
    return {
        "open_time": ot_utc.tz_localize(None),   # naive UTC (matches existing pickles)
        "open":  float(bar.open),
        "high":  float(bar.high),
        "low":   float(bar.low),
        "close": float(bar.close),
        "buy_vol":  int(bar.buy_vol),
        "sell_vol": int(bar.sell_vol),
        "tick_count": int(bar.tick_count),
    }


@dataclass
class LiveBarPersistor:
    """Subscribes to a 5-min BarBuilder; appends each bar to today's session pickle."""
    cache_dir: Path = LIVE_WARMSTART_CACHE_DIR
    n_bars_written: int = field(default=0, init=False)
    last_session_date: Optional[dt.date] = field(default=None, init=False)
    # In-memory list of today's bars (avoids re-reading pickle on every write)
    _today_bars: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_existing_for_session(self, session_date: dt.date) -> list[dict]:
        path = _pickle_path(session_date, self.cache_dir)
        if not path.exists():
            return []
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[bar_persistor] failed to read {path}: {e} (starting fresh)")
            return []

    def _atomic_write(self, session_date: dt.date, bars: list[dict]) -> None:
        path = _pickle_path(session_date, self.cache_dir)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(bars, f)
        os.replace(tmp, path)

    def on_bar(self, bar: Bar) -> None:
        """Called by BarBuilder when a 5-min bar closes."""
        bar_open_et = pd.Timestamp(bar.open_time)
        if bar_open_et.tzinfo is None:
            bar_open_et = bar_open_et.tz_localize("UTC").tz_convert(ET_TZ)
        else:
            bar_open_et = bar_open_et.tz_convert(ET_TZ)
        session_date = _session_date_for_bar(bar_open_et)

        # Session rolled? Reload buffer from disk (in case another process wrote)
        if session_date != self.last_session_date:
            self._today_bars = self._load_existing_for_session(session_date)
            self.last_session_date = session_date

        new_dict = _bar_to_dict(bar)
        new_ot = new_dict["open_time"]
        # Dedup: drop any existing bar with same open_time
        self._today_bars = [b for b in self._today_bars
                             if b["open_time"] != new_ot]
        self._today_bars.append(new_dict)
        self._today_bars.sort(key=lambda b: b["open_time"])

        try:
            self._atomic_write(session_date, self._today_bars)
            self.n_bars_written += 1
        except Exception as e:
            print(f"[bar_persistor] write FAILED for {session_date}: {e}")

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
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from live.combined.config import LIVE_WARMSTART_CACHE_DIR, ET_TZ
from live.combined.bar_builder import Bar

# Async write queue — pickle write (~10-30ms) moved off the bar-close thread
# so the engine consumer thread (post slow-client fix) isn't stalled at each
# 5-min boundary. Each entry: (session_date, list_of_bar_dicts_snapshot).
WRITE_QUEUE_MAXSIZE = 1000


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
    """Subscribes to a 5-min BarBuilder; appends each bar to today's session pickle.

    Async write architecture: on_bar() does an in-memory update + queue.put_nowait()
    (microseconds). A dedicated writer thread drains the queue and does the
    pickle.dump + os.replace. This keeps the engine consumer thread responsive
    even at session-boundary bar closes when the pickle is largest."""
    cache_dir: Path = LIVE_WARMSTART_CACHE_DIR
    n_bars_written: int = field(default=0, init=False)
    n_writes_queued: int = field(default=0, init=False)
    n_write_errors: int = field(default=0, init=False)
    n_queue_full_drops: int = field(default=0, init=False)
    last_session_date: Optional[dt.date] = field(default=None, init=False)
    # In-memory list of today's bars (avoids re-reading pickle on every write)
    _today_bars: list[dict] = field(default_factory=list, init=False)
    _write_queue: queue.Queue = field(default=None, init=False)
    _writer_thread: Optional[threading.Thread] = field(default=None, init=False)
    _stop_event: threading.Event = field(default=None, init=False)

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._write_queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="bar-persistor")
        self._writer_thread.start()

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

    def _writer_loop(self) -> None:
        """Drain write queue and write each snapshot to disk. Runs forever."""
        while not self._stop_event.is_set():
            try:
                session_date, bars_snapshot = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._atomic_write(session_date, bars_snapshot)
                self.n_bars_written += 1
            except Exception as e:
                self.n_write_errors += 1
                print(f"[bar_persistor] write FAILED for {session_date}: {e}")
        # Drain remaining writes on shutdown so we don't lose recent bars
        try:
            while True:
                session_date, bars_snapshot = self._write_queue.get_nowait()
                try:
                    self._atomic_write(session_date, bars_snapshot)
                    self.n_bars_written += 1
                except Exception as e:
                    self.n_write_errors += 1
        except queue.Empty:
            pass

    def on_bar(self, bar: Bar) -> None:
        """Called by BarBuilder when a 5-min bar closes.

        Hot path: only does in-memory list update + non-blocking queue.put.
        Disk IO happens on the writer thread."""
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

        # Snapshot the list (so writer thread sees a consistent view even if
        # the next bar fires before the previous write finishes) and enqueue.
        try:
            self._write_queue.put_nowait((session_date, list(self._today_bars)))
            self.n_writes_queued += 1
        except queue.Full:
            self.n_queue_full_drops += 1
            print(f"[bar_persistor] write queue FULL — dropped snapshot for {session_date}. "
                  f"Disk likely too slow OR writer thread crashed.")

    def stop(self) -> None:
        """Signal writer thread to exit; drains pending writes first."""
        self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10)

"""Multi-timeframe bar builders — ET-anchored, source-agnostic.

Each bar builder accepts ticks via on_tick(ts, price, size, side) and emits
completed bars via subscriber callbacks. Multiple timeframes can run in parallel
from one tick stream.

Bar boundaries:
  - 5-min: aligned to ET clock (09:00, 09:05, 09:10, ...)
  - 20-min: anchored to midnight ET (00:00, 00:20, 00:40, ..., 19:00, ...)

Side encoding (matches Databento MBP-1):
  'B' = bid (buy was aggressor)
  'A' = ask (sell was aggressor)
  'N' = unknown (counted in tick_count only)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional
import pandas as pd

from live.combined.config import ET_TZ, BAR_5MIN_SECS, BAR_20MIN_SECS


@dataclass
class Bar:
    open_time: pd.Timestamp   # ET-tz-aware
    close_time: pd.Timestamp  # ET-tz-aware (open_time + interval)
    open: float
    high: float
    low: float
    close: float
    buy_vol: int = 0
    sell_vol: int = 0
    tick_count: int = 0
    timeframe_secs: int = 0   # 300 for 5-min, 1200 for 20-min
    # Per-price-level volume breakdown — populated by BarBuilder as ticks come in.
    # Used by RV's windowed_absorption_check and B2's signal pipeline.
    # Schema: {price_float: [buy_vol, sell_vol]}
    level_volumes: dict = field(default_factory=dict)

    @property
    def total_vol(self) -> int:
        return self.buy_vol + self.sell_vol

    @property
    def delta(self) -> int:
        return self.buy_vol - self.sell_vol

    def level_volumes_tuples(self) -> dict:
        """Return level_volumes as {price: (buy, sell)} (tuples instead of lists)
        — matches the schema RV's windowed_absorption_check expects."""
        return {p: (lv[0], lv[1]) for p, lv in self.level_volumes.items()}


def floor_to_boundary(ts_et: pd.Timestamp, interval_secs: int) -> pd.Timestamp:
    """Floor ts to the nearest bar boundary, anchored to midnight ET.

    For 5-min: floors to 00:00, 00:05, 00:10, ...
    For 20-min: floors to 00:00, 00:20, 00:40, ...
    """
    midnight = ts_et.normalize()  # 00:00 ET on the same date
    secs_since_midnight = (ts_et - midnight).total_seconds()
    floored_secs = (int(secs_since_midnight) // interval_secs) * interval_secs
    return midnight + pd.Timedelta(seconds=floored_secs)


class BarBuilder:
    """Single-timeframe bar builder. Stateful — process ticks one at a time."""

    def __init__(self, timeframe_secs: int, name: str = ""):
        self.timeframe_secs = timeframe_secs
        self.name = name or f"{timeframe_secs}s"
        self.current: Optional[Bar] = None
        self.subscribers: list[Callable[[Bar], None]] = []
        self.n_bars_emitted = 0

    def subscribe(self, callback: Callable[[Bar], None]) -> None:
        """Register a callback to receive each completed bar."""
        self.subscribers.append(callback)

    def on_tick(self, ts_et: pd.Timestamp, price: float, size: int, side: str) -> Optional[Bar]:
        """Process one tick. Returns the completed bar if this tick closed one, else None."""
        if ts_et.tzinfo is None:
            raise ValueError("ts_et must be tz-aware (ET)")

        bar_open = floor_to_boundary(ts_et, self.timeframe_secs)
        bar_close = bar_open + pd.Timedelta(seconds=self.timeframe_secs)

        # Initialize first bar
        if self.current is None:
            self.current = Bar(
                open_time=bar_open, close_time=bar_close,
                open=price, high=price, low=price, close=price,
                timeframe_secs=self.timeframe_secs,
            )
            self._apply_tick(price, size, side)
            return None

        # If this tick falls into a NEW bar window, close the old, start new
        completed = None
        if bar_open != self.current.open_time:
            # Emit completed bar
            completed = self.current
            self.n_bars_emitted += 1
            for cb in self.subscribers:
                try:
                    cb(completed)
                except Exception as e:
                    import traceback
                    print(f"[bar_builder {self.name}] subscriber error: {e}")
                    traceback.print_exc()
            # Start new bar with this tick
            self.current = Bar(
                open_time=bar_open, close_time=bar_close,
                open=price, high=price, low=price, close=price,
                timeframe_secs=self.timeframe_secs,
            )
            self._apply_tick(price, size, side)
            return completed

        # Same bar, update
        self._apply_tick(price, size, side)
        return None

    def _apply_tick(self, price: float, size: int, side: str) -> None:
        b = self.current
        if price > b.high: b.high = price
        if price < b.low: b.low = price
        b.close = price
        if side == "B":
            b.buy_vol += size
            lv = b.level_volumes.setdefault(price, [0, 0])
            lv[0] += size
        elif side == "A":
            b.sell_vol += size
            lv = b.level_volumes.setdefault(price, [0, 0])
            lv[1] += size
        b.tick_count += 1

    def force_close_current(self) -> Optional[Bar]:
        """Emit the current in-progress bar (use at shutdown or for boundary-crossing time events)."""
        if self.current is None:
            return None
        completed = self.current
        self.n_bars_emitted += 1
        for cb in self.subscribers:
            try:
                cb(completed)
            except Exception as e:
                print(f"[bar_builder {self.name}] subscriber error: {e}")
        self.current = None
        return completed


class MultiBarBuilder:
    """Dispatches ticks to multiple BarBuilder instances (one per timeframe).

    Typical use:
        mb = MultiBarBuilder()
        b5 = mb.add_timeframe(300, name="5min")
        b20 = mb.add_timeframe(1200, name="20min")
        b5.subscribe(my_rv_handler)
        b20.subscribe(my_od_handler)

        # Then feed ticks:
        for tick in feed:
            mb.on_tick(tick.ts, tick.price, tick.size, tick.side)
    """

    def __init__(self):
        self.builders: dict[int, BarBuilder] = {}

    def add_timeframe(self, secs: int, name: str = "") -> BarBuilder:
        if secs in self.builders:
            return self.builders[secs]
        b = BarBuilder(secs, name=name)
        self.builders[secs] = b
        return b

    def on_tick(self, ts_et: pd.Timestamp, price: float, size: int, side: str) -> None:
        for b in self.builders.values():
            b.on_tick(ts_et, price, size, side)

    def get(self, secs: int) -> Optional[BarBuilder]:
        return self.builders.get(secs)

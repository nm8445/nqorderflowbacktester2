"""
Streaming range bar builder for live data.
Builds range bars incrementally as ticks arrive.
"""

from dataclasses import dataclass
from typing import Optional
from .range_bars import RangeBar, Level


class StreamingRangeBarBuilder:
    """Build range bars incrementally from streaming ticks."""

    def __init__(self, range_ticks: int, ticks_per_level: int, tick_size: float = 0.25):
        self.range_ticks = range_ticks
        self.ticks_per_level = ticks_per_level
        self.tick_size = tick_size

        # Current bar state
        self.current_bar = None
        self.completed_bars = []

    def add_tick(self, tick_data: dict) -> Optional[RangeBar]:
        """
        Add a tick and return completed bar if range is met.

        Args:
            tick_data: dict with keys: ts_event, price, bid_px_00, ask_px_00,
                      bid_sz_00, ask_sz_00, bid_ct_00, ask_ct_00

        Returns:
            Completed RangeBar if range met, else None
        """
        price = tick_data['price']
        timestamp = tick_data['ts_event']

        # Start new bar if needed
        if self.current_bar is None:
            self.current_bar = RangeBar(
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0,
                open_time=timestamp,
                close_time=timestamp,
                closed=False,
                levels={},
                first_level_price=None
            )

        # Update current bar
        self.current_bar.close = price
        self.current_bar.close_time = timestamp
        self.current_bar.high = max(self.current_bar.high, price)
        self.current_bar.low = min(self.current_bar.low, price)

        # Update levels
        if self.current_bar.first_level_price is None:
            self.current_bar.first_level_price = price

        level_price = self._get_level_price(price)

        if level_price not in self.current_bar.levels:
            self.current_bar.levels[level_price] = Level(
                price=level_price,
                bid_volume=0,
                ask_volume=0,
                delta=0,
                trades=0
            )

        level = self.current_bar.levels[level_price]

        # Update level data
        bid_vol = tick_data.get('bid_sz_00', 0)
        ask_vol = tick_data.get('ask_sz_00', 0)

        level.bid_volume += bid_vol
        level.ask_volume += ask_vol
        level.delta = level.bid_volume - level.ask_volume
        level.trades += 1

        self.current_bar.volume += (bid_vol + ask_vol)

        # Check if bar range is complete
        range_points = (self.current_bar.high - self.current_bar.low)
        range_in_ticks = int(range_points / self.tick_size)

        if range_in_ticks >= self.range_ticks:
            # Close current bar
            self.current_bar.closed = True
            completed = self.current_bar
            self.completed_bars.append(completed)

            # Start new bar
            self.current_bar = None

            return completed

        return None

    def _get_level_price(self, price: float) -> float:
        """Get the level price for a given tick price."""
        level_size = self.ticks_per_level * self.tick_size
        return round((price // level_size) * level_size, 2)

    def get_current_bar(self) -> Optional[RangeBar]:
        """Get the current incomplete bar."""
        return self.current_bar

    def get_completed_bars(self) -> list[RangeBar]:
        """Get all completed bars."""
        return self.completed_bars

    def get_recent_bars(self, n: int = 50) -> list[RangeBar]:
        """Get N most recent completed bars."""
        return self.completed_bars[-n:]

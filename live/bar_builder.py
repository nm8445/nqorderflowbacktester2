"""
Live bar builders - processes ticks one at a time.

- LiveBarBuilder: Range bars (40-tick range)
- LiveTimeBarBuilder: Time bars (5-minute intervals for ATR)
"""
from typing import Optional
from dataclasses import dataclass
import pandas as pd

from nqbt.analysis.range_bars import RangeBar, BarLevel, TICK_SIZE


@dataclass
class TimeBar:
    """5-minute time bar for ATR calculation."""
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    buy_vol: int = 0
    sell_vol: int = 0
    tick_count: int = 0
    closed: bool = False


class LiveBarBuilder:
    """
    Stateful tick-by-tick range bar builder.

    Emits a completed RangeBar when high - low >= range_points.
    """

    def __init__(self, range_ticks: int = 40, ticks_per_level: int = 5):
        self.range_points = range_ticks * TICK_SIZE      # 10.0 points for 40 ticks
        self.level_size = ticks_per_level * TICK_SIZE    # 1.25 points for 5 ticks
        self.current: Optional[RangeBar] = None

    def on_tick(
        self,
        ts: pd.Timestamp,
        price: float,
        size: int,
        aggressor: str,  # "buy" or "sell"
    ) -> Optional[RangeBar]:
        """
        Process one trade tick.

        Returns
        -------
        RangeBar if this tick closed a bar, else None.
        """
        # Start new bar if none exists
        if self.current is None:
            self.current = RangeBar(
                open_time=ts,
                close_time=None,
                open=price,
                high=price,
                low=price,
                close=price,
            )

        bar = self.current

        # Update OHLC
        bar.high = max(bar.high, price)
        bar.low = min(bar.low, price)
        bar.close = price

        # Update volume
        if aggressor == "buy":
            bar.buy_vol += size
        else:
            bar.sell_vol += size

        # Update internal volume profile level
        level_idx = round(price / TICK_SIZE) // round(self.level_size / TICK_SIZE)
        level_price = round(level_idx * self.level_size, 4)

        if level_price not in bar.levels:
            bar.levels[level_price] = BarLevel(price=level_price)

        if aggressor == "buy":
            bar.levels[level_price].buy_vol += size
        else:
            bar.levels[level_price].sell_vol += size

        # Check if bar should close
        if bar.high - bar.low >= self.range_points - 1e-9:
            bar.close_time = ts
            # bar.closed is auto-computed from close_time (read-only property)
            self.current = None
            return bar

        return None


class LiveTimeBarBuilder:
    """
    Stateful tick-by-tick time bar builder.

    Emits a completed TimeBar when time interval (e.g., 5 minutes) elapses.
    """

    def __init__(self, interval_minutes: int = 5):
        self.interval = pd.Timedelta(minutes=interval_minutes)
        self.current: Optional[TimeBar] = None

    def on_tick(
        self,
        ts: pd.Timestamp,
        price: float,
        size: int,
        aggressor: str,  # "buy" or "sell"
    ) -> Optional[TimeBar]:
        """
        Process one trade tick for time bars.

        Returns
        -------
        TimeBar if this tick closed a bar, else None.
        """
        # Start new bar if none exists
        if self.current is None:
            # Floor timestamp to interval boundary (e.g., 10:32:47 → 10:30:00)
            bar_start = ts.floor(self.interval)
            bar_end = bar_start + self.interval

            self.current = TimeBar(
                open_time=bar_start,
                close_time=bar_end,
                open=price,
                high=price,
                low=price,
                close=price,
            )

        bar = self.current

        # Check if this tick belongs to next bar
        if ts >= bar.close_time:
            # Close current bar
            bar.closed = True
            completed_bar = bar

            # Start new bar
            bar_start = ts.floor(self.interval)
            bar_end = bar_start + self.interval

            self.current = TimeBar(
                open_time=bar_start,
                close_time=bar_end,
                open=price,
                high=price,
                low=price,
                close=price,
            )

            # Update new bar with this tick
            if aggressor == "buy":
                self.current.buy_vol += size
            else:
                self.current.sell_vol += size
            self.current.tick_count += 1

            return completed_bar

        # Update current bar
        bar.high = max(bar.high, price)
        bar.low = min(bar.low, price)
        bar.close = price

        if aggressor == "buy":
            bar.buy_vol += size
        else:
            bar.sell_vol += size

        bar.tick_count += 1

        return None

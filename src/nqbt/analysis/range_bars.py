"""
40-range volumetric bars for NQ orderflow analysis.

A range bar closes when its high - low reaches the specified number of ticks.
Each bar contains:
  - OHLC prices and timestamps
  - Total buy/sell volume and delta
  - Internal volume profile at 5 ticks per level (1.25 points)

Usage
-----
    from nqbt.analysis.range_bars import build_range_bars

    bars = build_range_bars(ticks, range_ticks=40, ticks_per_level=5)
    # bars is a list of RangeBar objects
    # call bars_to_df(bars) for a flat DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

TICK_SIZE = 0.25


@dataclass
class BarLevel:
    """Volume at a single price level inside a range bar."""
    price: float
    buy_vol: int = 0
    sell_vol: int = 0

    @property
    def total_vol(self) -> int:
        return self.buy_vol + self.sell_vol

    @property
    def delta(self) -> int:
        return self.buy_vol - self.sell_vol


@dataclass
class RangeBar:
    """
    A single completed (or in-progress) range bar.

    Attributes
    ----------
    open_time   : timestamp of the first tick in this bar
    close_time  : timestamp of the closing tick (None if bar still open)
    open        : price of first tick
    high        : highest price seen
    low         : lowest price seen
    close       : price of closing tick
    buy_vol     : total aggressive buy volume
    sell_vol    : total aggressive sell volume
    levels      : internal volume profile keyed by level price
    """
    open_time:  pd.Timestamp
    close_time: Optional[pd.Timestamp]
    open:       float
    high:       float
    low:        float
    close:      float
    buy_vol:    int = 0
    sell_vol:   int = 0
    levels:     dict = field(default_factory=dict)

    @property
    def total_vol(self) -> int:
        return self.buy_vol + self.sell_vol

    @property
    def delta(self) -> int:
        return self.buy_vol - self.sell_vol

    @property
    def range_points(self) -> float:
        return self.high - self.low

    @property
    def closed(self) -> bool:
        return self.close_time is not None

    def poc(self) -> Optional[float]:
        """Price level with highest volume in this bar."""
        if not self.levels:
            return None
        return max(self.levels, key=lambda p: self.levels[p].total_vol)

    def levels_df(self) -> pd.DataFrame:
        """Internal profile as a DataFrame sorted by price descending."""
        rows = [
            {
                "price":     lv.price,
                "buy_vol":   lv.buy_vol,
                "sell_vol":  lv.sell_vol,
                "total_vol": lv.total_vol,
                "delta":     lv.delta,
            }
            for lv in sorted(self.levels.values(), key=lambda l: l.price, reverse=True)
        ]
        return pd.DataFrame(rows).set_index("price")


def build_range_bars(
    ticks: pd.DataFrame,
    range_ticks: int = 40,
    ticks_per_level: int = 5,
) -> list[RangeBar]:
    """
    Build range bars from a normalized tick DataFrame.

    Parameters
    ----------
    ticks : pd.DataFrame
        Output of normalizer.normalize().
    range_ticks : int
        Bar closes when high - low >= range_ticks * TICK_SIZE.
        Default 40 ticks = 10 points for NQ.
    ticks_per_level : int
        Internal profile level size in ticks. Default 5 = 1.25 points.

    Returns
    -------
    list[RangeBar]
        Completed bars. The last bar may be open (close_time is None).
    """
    range_points  = range_ticks * TICK_SIZE   # 10.0 points
    level_size    = ticks_per_level * TICK_SIZE  # 1.25 points

    bars: list[RangeBar] = []
    current: Optional[RangeBar] = None

    for ts, row in ticks.iterrows():
        price     = row["price"]
        size      = int(row["size"])
        aggressor = row["aggressor"]

        # Start a new bar if none is open
        if current is None:
            current = RangeBar(
                open_time=ts, close_time=None,
                open=price, high=price, low=price, close=price,
            )

        # Update OHLC
        current.high  = max(current.high, price)
        current.low   = min(current.low,  price)
        current.close = price

        # Update bar-level volume
        if aggressor == "buy":
            current.buy_vol += size
        else:
            current.sell_vol += size

        # Update internal profile level
        price_ticks = round(price / TICK_SIZE)
        level_idx   = price_ticks // ticks_per_level
        level_price = round(level_idx * ticks_per_level * TICK_SIZE, 4)

        if level_price not in current.levels:
            current.levels[level_price] = BarLevel(price=level_price)

        if aggressor == "buy":
            current.levels[level_price].buy_vol += size
        else:
            current.levels[level_price].sell_vol += size

        # Close bar if range is reached
        if current.high - current.low >= range_points - 1e-9:
            current.close_time = ts
            bars.append(current)
            current = None

    # Append the last open bar if it exists
    if current is not None:
        bars.append(current)

    return bars


def bars_to_df(bars: list[RangeBar]) -> pd.DataFrame:
    """
    Flatten a list of RangeBar objects into a summary DataFrame.
    One row per bar. Does not include internal level detail.
    """
    rows = [
        {
            "open_time":    b.open_time,
            "close_time":   b.close_time,
            "open":         b.open,
            "high":         b.high,
            "low":          b.low,
            "close":        b.close,
            "buy_vol":      b.buy_vol,
            "sell_vol":     b.sell_vol,
            "total_vol":    b.total_vol,
            "delta":        b.delta,
            "range_points": b.range_points,
            "poc":          b.poc(),
            "closed":       b.closed,
        }
        for b in bars
    ]
    return pd.DataFrame(rows).set_index("open_time")

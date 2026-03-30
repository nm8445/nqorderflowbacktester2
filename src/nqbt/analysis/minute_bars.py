"""
1-minute bar aggregation from MBP-1 tick data.

Used exclusively for regime classification (HMM training and prediction).
The 40-range bar pipeline in range_bars.py is untouched and continues to
serve execution-level orderflow features.

MinuteBar uses mid_price for OHLC, consistent with all regime classification
features which compute price movements from mid_price.
Empty minutes (no ticks) are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class MinuteBar:
    """A single 1-minute aggregated bar."""
    open_time:  pd.Timestamp
    close_time: pd.Timestamp
    open:       float
    high:       float
    low:        float
    close:      float
    buy_vol:    int = 0
    sell_vol:   int = 0

    @property
    def total_vol(self) -> int:
        return self.buy_vol + self.sell_vol

    @property
    def delta(self) -> int:
        return self.buy_vol - self.sell_vol


def build_minute_bars(ticks: pd.DataFrame) -> list[MinuteBar]:
    """
    Aggregate enriched tick data into 1-minute bars.

    Parameters
    ----------
    ticks : pd.DataFrame
        Output of normalizer.normalize_enriched(), sliced to the session window.
        Required columns: mid_price, aggressor, size.

    Returns
    -------
    list[MinuteBar]
        One bar per non-empty calendar minute, ordered chronologically.
        Empty minutes (no ticks) are omitted.
    """
    if ticks.empty:
        return []

    minute_floor = ticks.index.floor("min")
    bars: list[MinuteBar] = []

    for minute_ts, group in ticks.groupby(minute_floor):
        if group.empty:
            continue

        mids     = group["mid_price"].to_numpy(dtype=float)
        buy_vol  = int(group.loc[group["aggressor"] == "buy",  "size"].sum())
        sell_vol = int(group.loc[group["aggressor"] == "sell", "size"].sum())

        bars.append(MinuteBar(
            open_time  = group.index[0],
            close_time = group.index[-1],
            open       = float(mids[0]),
            high       = float(mids.max()),
            low        = float(mids.min()),
            close      = float(mids[-1]),
            buy_vol    = buy_vol,
            sell_vol   = sell_vol,
        ))

    return bars

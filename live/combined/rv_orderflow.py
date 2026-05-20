"""Orderflow window absorption check for RV strategy.

Per locked config:
  - Lower half of bar = [low, (high+low)/2)
  - Upper half of bar = ((high+low)/2, high]
  - Each tick-level (NQ tick = 0.25) has delta = buy_vol - sell_vol over the bar
  - LONG: best contiguous 8-tick window in LOWER half, min sum_delta <= -150
  - SHORT: best contiguous 8-tick window in UPPER half, max sum_delta >= +150

The level dict maps {level_price (float) -> (buy_vol, sell_vol)}.
"""
from __future__ import annotations

import numpy as np

NQ_TICK = 0.25
WINDOW_N_TICKS = 8       # 8-tick contiguous window
WINDOW_D = 150           # |sum_delta| threshold


def windowed_absorption_check(high: float, low: float,
                               level_volumes: dict[float, tuple[int, int]],
                               window_n_ticks: int = WINDOW_N_TICKS,
                               window_d: float = WINDOW_D) -> tuple[bool, bool]:
    """Returns (long_ok, short_ok).

    level_volumes: dict mapping level_price -> (buy_vol, sell_vol) for this bar.
    """
    if not level_volumes or high <= low:
        return False, False

    mid = (high + low) / 2.0

    # Partition levels into lower and upper halves
    prices = sorted(level_volumes.keys())
    lower_prices = [p for p in prices if p < mid]
    upper_prices = [p for p in prices if p > mid]
    # Note: strictly < and > per backtest (mid itself excluded)

    long_ok = False
    short_ok = False

    # LONG check: lower half, find best (most NEGATIVE) contiguous 8-tick window
    if len(lower_prices) >= window_n_ticks:
        lo_min, lo_max = lower_prices[0], lower_prices[-1]
        # Build dense array from lo_min to lo_max in tick steps
        n_slots = int(round((lo_max - lo_min) / NQ_TICK)) + 1
        arr = np.zeros(n_slots, dtype=np.float64)
        for p in lower_prices:
            buy, sell = level_volumes[p]
            slot = int(round((p - lo_min) / NQ_TICK))
            if 0 <= slot < n_slots:
                arr[slot] = buy - sell
        if n_slots >= window_n_ticks:
            csum = np.cumsum(arr)
            csum_pad = np.concatenate(([0.0], csum))
            windows = csum_pad[window_n_ticks:] - csum_pad[:-window_n_ticks]
            if windows.min() <= -window_d:
                long_ok = True

    # SHORT check: upper half, find best (most POSITIVE) contiguous 8-tick window
    if len(upper_prices) >= window_n_ticks:
        up_min, up_max = upper_prices[0], upper_prices[-1]
        n_slots = int(round((up_max - up_min) / NQ_TICK)) + 1
        arr = np.zeros(n_slots, dtype=np.float64)
        for p in upper_prices:
            buy, sell = level_volumes[p]
            slot = int(round((p - up_min) / NQ_TICK))
            if 0 <= slot < n_slots:
                arr[slot] = buy - sell
        if n_slots >= window_n_ticks:
            csum = np.cumsum(arr)
            csum_pad = np.concatenate(([0.0], csum))
            windows = csum_pad[window_n_ticks:] - csum_pad[:-window_n_ticks]
            if windows.max() >= window_d:
                short_ok = True

    return long_ok, short_ok

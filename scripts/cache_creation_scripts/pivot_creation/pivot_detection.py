"""
Pivot point detection for Fair Price Theory strategy.

Pivot highs and pivot lows using configurable left/right bar counts.
A pivot is confirmed only after right_len bars have closed beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class PivotLevel:
    """A single pivot high or low."""
    price: float
    bar_time: datetime
    kind: str              # "high" or "low"
    broken: bool = False   # True once price closes through it


def detect_pivots(
    bars: pd.DataFrame,
    left_len: int = 10,
    right_len: int = 10,
) -> list[PivotLevel]:
    """
    Detect all pivot highs and lows in a bar series.

    A pivot high at bar i exists when:
      high[i] > high[i-j] for j in 1..left_len
      high[i] > high[i+j] for j in 1..right_len

    A pivot low at bar i exists when:
      low[i] < low[i-j] for j in 1..left_len
      low[i] < low[i+j] for j in 1..right_len

    Pivot is confirmed at bar i + right_len (needs right_len future bars).

    Parameters
    ----------
    bars : pd.DataFrame
        Must have 'high' and 'low' columns. Index is timestamps.
    left_len : int
    right_len : int

    Returns
    -------
    list[PivotLevel]
        All detected pivots, in chronological order.
    """
    highs = bars["high"].values
    lows  = bars["low"].values
    times = bars.index
    n     = len(bars)

    pivots = []

    for i in range(left_len, n - right_len):
        # Check pivot high
        is_ph = True
        for j in range(1, left_len + 1):
            if highs[i] <= highs[i - j]:
                is_ph = False
                break
        if is_ph:
            for j in range(1, right_len + 1):
                if highs[i] <= highs[i + j]:
                    is_ph = False
                    break
        if is_ph:
            pivots.append(PivotLevel(
                price=highs[i],
                bar_time=times[i],
                kind="high",
            ))

        # Check pivot low
        is_pl = True
        for j in range(1, left_len + 1):
            if lows[i] >= lows[i - j]:
                is_pl = False
                break
        if is_pl:
            for j in range(1, right_len + 1):
                if lows[i] >= lows[i + j]:
                    is_pl = False
                    break
        if is_pl:
            pivots.append(PivotLevel(
                price=lows[i],
                bar_time=times[i],
                kind="low",
            ))

    return pivots


def get_active_pivots(
    pivots: list[PivotLevel],
    current_close: float,
) -> list[PivotLevel]:
    """
    Return pivots that have NOT been broken.

    A pivot high is broken when price closes above it.
    A pivot low is broken when price closes below it.

    This function also marks broken pivots in-place.
    """
    for p in pivots:
        if p.broken:
            continue
        if p.kind == "high" and current_close > p.price:
            p.broken = True
        elif p.kind == "low" and current_close < p.price:
            p.broken = True

    return [p for p in pivots if not p.broken]


def get_pre_session_pivots(
    pivots: list[PivotLevel],
    session_open_time: datetime,
) -> list[PivotLevel]:
    """
    Filter to pivots that formed before session open (9:30 AM).

    Only these pre-session pivots are used for mean reversion entries.
    """
    return [p for p in pivots if p.bar_time < session_open_time]

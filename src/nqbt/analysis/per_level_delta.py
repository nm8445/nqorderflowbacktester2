"""
Per-level delta extractor using 5-tick (1.25 pt) price zones.

Groups tick data into 5-tick zones and computes net signed delta per zone.
An adaptive percentile threshold (computed per-session) flags which zones
have statistically significant absorption activity — mirrors NinjaTrader
BigTradesVolumetric zone logic.

Zone price = floor(price / 1.25) * 1.25 (lower bound of zone).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ZONE_SIZE = 1.25  # 5 ticks * 0.25 pt/tick

_EMPTY_COLUMNS = [
    "zone_price", "buy_vol", "sell_vol", "delta", "abs_delta",
    "pct_rank", "significant",
]


def compute_per_level_delta(
    ticks: pd.DataFrame,
    zone_ticks: int = 5,
    percentile: float = 70.0,
) -> pd.DataFrame:
    """
    Group ticks by 5-tick price zones and compute net signed delta per zone.

    Parameters
    ----------
    ticks : pd.DataFrame
        Normalized tick DataFrame with columns: price (float64), size (int64),
        aggressor (category: 'buy'/'sell'). Index: ts_event (UTC).
    zone_ticks : int
        Zone width in ticks. Default 5 = 1.25 points.
    percentile : float
        Percentile threshold for flagging significant zones. Default 70.

    Returns
    -------
    pd.DataFrame
        Columns: zone_price, buy_vol, sell_vol, delta, abs_delta,
                 pct_rank, significant. Sorted by zone_price ascending.
    """
    if ticks.empty:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    zone_size = zone_ticks * 0.25
    zone_price = (ticks["price"] // zone_size) * zone_size

    buy_mask  = ticks["aggressor"] == "buy"
    sell_mask = ticks["aggressor"] == "sell"

    buy_vol_s  = ticks["size"].where(buy_mask,  0)
    sell_vol_s = ticks["size"].where(sell_mask, 0)

    agg = pd.DataFrame({
        "zone_price": zone_price,
        "buy_vol":    buy_vol_s,
        "sell_vol":   sell_vol_s,
    }).groupby("zone_price", sort=True).sum().reset_index()

    agg["delta"]     = agg["buy_vol"] - agg["sell_vol"]
    agg["abs_delta"] = agg["delta"].abs()

    # Percentile rank: fraction of zones with strictly smaller abs_delta, * 100
    abs_vals = agg["abs_delta"].to_numpy(dtype=float)
    n = len(abs_vals)
    if n > 1:
        ranks = np.array([
            float(np.mean(abs_vals < v)) * 100.0
            for v in abs_vals
        ])
    else:
        ranks = np.zeros(n)

    agg["pct_rank"]    = ranks
    agg["significant"] = agg["pct_rank"] >= percentile

    return agg[_EMPTY_COLUMNS].reset_index(drop=True)


def session_delta_zones(
    ticks: pd.DataFrame,
    zone_ticks: int = 5,
    percentile: float = 70.0,
) -> pd.DataFrame:
    """
    Return only the significant delta zones for a session.

    Parameters
    ----------
    ticks : pd.DataFrame
        Normalized tick DataFrame (same spec as compute_per_level_delta).
    zone_ticks : int
        Zone width in ticks. Default 5 = 1.25 points.
    percentile : float
        Percentile threshold. Default 70.

    Returns
    -------
    pd.DataFrame
        Subset of compute_per_level_delta() where significant == True.
        Empty DataFrame with correct columns if no ticks supplied.
    """
    if ticks.empty:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    result = compute_per_level_delta(ticks, zone_ticks=zone_ticks, percentile=percentile)
    return result[result["significant"]].reset_index(drop=True)

"""
Anchored VWAP with standard deviation bands for NQ orderflow analysis.

Reset interval: each CME session (6pm ET previous day → market close).
Computed tick-by-tick on a cumulative basis — at any point in the session
the VWAP reflects all volume traded from session open up to that tick.

Standard deviation bands use volume-weighted variance:
    variance = Σ(size × price²) / Σ(size)  -  vwap²
    std      = sqrt(variance)
    band_N   = vwap ± N × std   (N = 1, 2, 3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vwap(
    ticks: pd.DataFrame,
    std_multipliers: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """
    Compute cumulative anchored VWAP and standard deviation bands.

    Parameters
    ----------
    ticks : pd.DataFrame
        Output of normalizer.normalize() — already filtered to a single
        session window (6pm ET prior day through desired end time).
    std_multipliers : tuple
        Multipliers for the standard deviation bands. Default (1, 2, 3).

    Returns
    -------
    pd.DataFrame
        Same index as ticks, columns:
            vwap
            std
            std1_upper, std1_lower
            std2_upper, std2_lower
            std3_upper, std3_lower
    """
    price  = ticks["price"].to_numpy()
    size   = ticks["size"].to_numpy(dtype=float)

    cum_vol       = np.cumsum(size)
    cum_pv        = np.cumsum(price * size)
    cum_pv2       = np.cumsum(price ** 2 * size)

    vwap     = cum_pv / cum_vol
    variance = (cum_pv2 / cum_vol) - vwap ** 2
    # Clamp tiny negatives from floating point
    variance = np.maximum(variance, 0.0)
    std      = np.sqrt(variance)

    result = pd.DataFrame({"vwap": vwap, "std": std}, index=ticks.index)

    for m in std_multipliers:
        result[f"std{m}_upper"] = vwap + m * std
        result[f"std{m}_lower"] = vwap - m * std

    return result


def vwap_at(
    ticks: pd.DataFrame,
    as_of: pd.Timestamp,
    std_multipliers: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """
    Return VWAP and bands at a specific point in time.

    Parameters
    ----------
    ticks   : session ticks from 6pm ET prior day
    as_of   : compute VWAP using all ticks up to and including this timestamp
    std_multipliers : band multipliers

    Returns
    -------
    dict with keys: vwap, std, std1_upper, std1_lower, std2_upper, ...
    """
    window = ticks[ticks.index <= as_of]
    if window.empty:
        return {}
    return compute_vwap(window, std_multipliers).iloc[-1].to_dict()

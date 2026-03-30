"""
8-feature computation for the 2-state directional overnight HMM.

Direction is irrelevant to overnight classification — a sustained upward
overnight and a sustained downward overnight are both DIRECTIONAL.  All
signed features are therefore converted to their absolute values before
normalization so the model learns magnitude, not sign.

Features (priority order):
  abs_delta_pct          — abs((buy_vol - sell_vol) / total_vol)
                           High = committed aggression regardless of direction.
  trend_ratio            — abs(net_move) / total_path
                           High = price moved efficiently (low back-and-forth).
  abs_normalized_d_move  — abs(net_move / (realized_vol * open_price))
                           Large move relative to volatility, direction-agnostic.
  realized_vol           — sqrt(sum(log_returns^2))
                           Active session regardless of direction.
  trade_arrival_rate     — trades per second
                           Urgency and participation level.
  overnight_deceleration — slope_second_half_of_session - slope_first_half
                           Negative = trend lost momentum into the open
                           (failed-auction warning).  Computed cumulatively
                           per bar using all bars seen so far in the session.
  abs_obi_avg            — abs(mean (bid_sz - ask_sz) / (bid_sz + ask_sz))
                           Book leaning consistently one way, either direction.
  autocorr_r1            — lag-1 autocorrelation of log returns
                           Positive = returns persist (directional).
                           Negative = returns reverse (ranging/chop).

The original signed features delta_pct, normalized_d_move, obi_avg are
replaced by their absolute-value counterparts.  trend_ratio, realized_vol,
trade_arrival_rate, and autocorr_r1 carry over from features.py unchanged
(they are already unsigned or sign-agnostic).  overnight_deceleration is new.
vwap_deviation, delta_acceleration, and obi_trend are dropped entirely.

All features are z-score normalized per session independently via
StandardScaler.fit_transform() before assembly into the HMM input matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

OVERNIGHT_FEATURE_NAMES = [
    "abs_delta_pct",
    "trend_ratio",
    "abs_normalized_d_move",
    "realized_vol",
    "trade_arrival_rate",
    "overnight_deceleration",
    "abs_obi_avg",
    "autocorr_r1",
]

MIN_TICKS_PER_BAR = 3


def _deceleration(closes: np.ndarray) -> float:
    """
    Compute slope_second_half - slope_first_half using all bar closes seen so far.

    Returns 0.0 if fewer than 4 bars are available.
    A negative value means the trend lost momentum (second half weaker than first).
    A positive value means the trend was sustained or accelerating.
    """
    n = len(closes)
    if n < 4:
        return 0.0
    mid = n // 2
    first  = closes[:mid]
    second = closes[mid:]
    x1 = np.arange(len(first),  dtype=float)
    x2 = np.arange(len(second), dtype=float)
    s1 = float(np.polyfit(x1, first,  1)[0]) if len(first)  >= 2 else 0.0
    s2 = float(np.polyfit(x2, second, 1)[0]) if len(second) >= 2 else 0.0
    return s2 - s1


def _bar_features(
    ticks:       pd.DataFrame,
    closes_so_far: np.ndarray,   # close prices of all bars up to and including this one
) -> np.ndarray:
    n = len(ticks)
    if n == 0:
        return np.zeros(len(OVERNIGHT_FEATURE_NAMES))

    buy_vol   = float(ticks.loc[ticks["aggressor"] == "buy",  "size"].sum())
    sell_vol  = float(ticks.loc[ticks["aggressor"] == "sell", "size"].sum())
    total_vol = buy_vol + sell_vol
    delta_pct = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
    abs_delta_pct = abs(delta_pct)

    mids = ticks["mid_price"].to_numpy(dtype=float)

    if n >= 2 and np.all(mids > 0):
        diffs       = np.diff(mids)
        net_move    = mids[-1] - mids[0]
        total_path  = float(np.sum(np.abs(diffs)))
        trend_ratio = abs(net_move) / total_path if total_path > 1e-9 else 0.0
        log_returns = np.log(mids[1:] / mids[:-1])
        realized_vol = float(np.sqrt(np.sum(log_returns ** 2)))
    else:
        trend_ratio  = 0.0
        log_returns  = np.array([])
        realized_vol = 0.0
        net_move     = 0.0

    if realized_vol > 1e-9 and n >= 2 and mids[0] > 0:
        abs_normalized_d_move = abs(net_move) / (realized_vol * mids[0])
    else:
        abs_normalized_d_move = 0.0

    if n >= 2:
        duration           = (ticks.index[-1] - ticks.index[0]).total_seconds()
        trade_arrival_rate = float(n / duration) if duration > 0 else float(n)
    else:
        trade_arrival_rate = 0.0

    bid_sz   = ticks["bid_sz"].to_numpy(dtype=float)
    ask_sz   = ticks["ask_sz"].to_numpy(dtype=float)
    total_sz = bid_sz + ask_sz
    obi_vals = np.where(total_sz > 0, (bid_sz - ask_sz) / total_sz, 0.0)
    abs_obi_avg = abs(float(np.mean(obi_vals)))

    if len(log_returns) >= 3 and np.std(log_returns) > 1e-12:
        ac = float(np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1])
        autocorr_r1 = 0.0 if np.isnan(ac) else ac
    else:
        autocorr_r1 = 0.0

    overnight_deceleration = _deceleration(closes_so_far)

    return np.array([
        abs_delta_pct,
        trend_ratio,
        abs_normalized_d_move,
        realized_vol,
        trade_arrival_rate,
        overnight_deceleration,
        abs_obi_avg,
        autocorr_r1,
    ])


def compute_overnight_features(
    enriched_ticks: pd.DataFrame,
    bars:           list,
    min_ticks:      int  = MIN_TICKS_PER_BAR,
    normalize:      bool = True,
) -> np.ndarray:
    """
    Compute the 8-feature directional matrix for one overnight session.

    Parameters
    ----------
    enriched_ticks : pd.DataFrame
        Output of normalizer.normalize_enriched(), sliced to the overnight window.
    bars : list[RangeBar]
        40-range bars built from the same overnight window.
    min_ticks : int
        Bars with fewer ticks are skipped.
    normalize : bool
        If True, apply StandardScaler per-session before returning.

    Returns
    -------
    np.ndarray
        Shape (n_valid_bars, 8). Empty array if fewer than 2 valid bars.
    """
    rows: list[np.ndarray] = []
    closes_so_far: list[float] = []

    for bar in bars:
        close_ts = bar.close_time if bar.close_time is not None else bar.open_time
        bar_ticks = enriched_ticks[
            (enriched_ticks.index >= bar.open_time) &
            (enriched_ticks.index <= close_ts)
        ]
        if len(bar_ticks) < min_ticks:
            continue

        closes_so_far.append(bar.close)
        feat = _bar_features(bar_ticks, np.array(closes_so_far))
        rows.append(feat)

    if len(rows) < 2:
        return np.empty((0, len(OVERNIGHT_FEATURE_NAMES)))

    X = np.array(rows)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize and X.shape[0] >= 2:
        X = StandardScaler().fit_transform(X)

    return X

"""
Session-level feature vector for RTH regime classification.

Computes one feature vector of shape (8,) per RTH session (09:30-11:00 ET)
from 1-minute bars built internally from enriched ticks.

Architecture note
-----------------
The overnight HMM uses per-bar feature matrices (n_bars, 8) stacked across all
training sessions with a lengths array passed to Baum-Welch.  This works because
overnight bar counts are high (184-2039 bars) giving per-session StandardScaler
sufficient variance.

The RTH session is 90 minutes.  On 1-minute bars this produces exactly 90 bars —
not enough within-session variance for per-session normalization on slow features.
The fix is one feature vector per session so the HMM sees a matrix of shape
(n_sessions, 8) with no lengths array at all.  Cross-session StandardScaler is
fitted on training session vectors only.

Features
--------
  abs_delta_pct      — abs((buy_vol - sell_vol) / total_vol)
  trend_ratio        — abs(session_close - session_open) / (high - low)
  realized_vol       — std of 1-minute log returns
  autocorr_r1        — lag-1 autocorrelation of 1-minute log returns
  trend_consistency  — fraction of bars closing in the session direction
  trade_arrival_rate — total trades / 90.0  (trades per minute)

Removed features
----------------
  max_excursion_ratio — identical to trend_ratio (same equation, same inputs);
                        collinear features destabilize diagonal covariance estimation.
  vwap_dev_mean       — near-zero cross-session variance (training std 0.053 on
                        mean 0.86); no discriminative power between states.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nqbt.analysis.minute_bars import build_minute_bars

RTH_SESSION_FEATURE_NAMES = [
    "abs_delta_pct",
    "trend_ratio",
    "realized_vol",
    "autocorr_r1",
    "trend_consistency",
    "trade_arrival_rate",
]

_N_FEATURES = len(RTH_SESSION_FEATURE_NAMES)   # 6


def compute_rth_session_features(ticks: pd.DataFrame) -> np.ndarray:
    """
    Compute one session-level feature vector for one RTH session.

    Parameters
    ----------
    ticks : pd.DataFrame
        Enriched ticks sliced to 09:30-11:00 ET for one trading date.
        Must contain columns: price, size, aggressor, mid_price,
        bid_sz, ask_sz (standard normalize_enriched() output).

    Returns
    -------
    np.ndarray, shape (6,)
        One value per feature.  Returns zeros if ticks are insufficient.
    """
    if ticks.empty:
        return np.zeros(_N_FEATURES)

    bars = build_minute_bars(ticks)
    if len(bars) < 2:
        return np.zeros(_N_FEATURES)

    # ── Volume deltas (from ticks) ─────────────────────────────────────────────
    buy_vol   = float(ticks.loc[ticks["aggressor"] == "buy",  "size"].sum())
    sell_vol  = float(ticks.loc[ticks["aggressor"] == "sell", "size"].sum())
    total_vol = buy_vol + sell_vol
    abs_delta_pct = abs(buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0

    # ── Session OHLC from bars ─────────────────────────────────────────────────
    session_open  = float(bars[0].open)
    session_close = float(bars[-1].close)
    session_high  = float(max(b.high for b in bars))
    session_low   = float(min(b.low  for b in bars))
    price_range   = session_high - session_low
    net_move      = session_close - session_open

    # ── trend_ratio ───────────────────────────────────────────────────────────
    if price_range < 0.01:
        trend_ratio = 0.0
    else:
        trend_ratio = abs(net_move) / price_range

    # ── 1-minute log returns (bar closes) ─────────────────────────────────────
    closes      = np.array([b.close for b in bars], dtype=float)
    log_returns = np.log(closes[1:] / closes[:-1])

    realized_vol = float(np.std(log_returns)) if len(log_returns) >= 2 else 0.0

    if len(log_returns) >= 3 and np.std(log_returns) > 1e-12:
        ac = float(np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1])
        autocorr_r1 = 0.0 if np.isnan(ac) else ac
    else:
        autocorr_r1 = 0.0

    # ── trend_consistency ─────────────────────────────────────────────────────
    if abs(net_move) < 1.0:
        trend_consistency = 0.5
    else:
        direction         = 1 if net_move > 0 else -1
        directional_count = sum(
            1 for b in bars if (b.close - b.open) * direction > 0
        )
        trend_consistency = directional_count / len(bars)

    # ── trade_arrival_rate ────────────────────────────────────────────────────
    trade_arrival_rate = len(ticks) / 90.0   # trades per minute over 90-min session

    result = np.array([
        abs_delta_pct,
        trend_ratio,
        realized_vol,
        autocorr_r1,
        trend_consistency,
        trade_arrival_rate,
    ])
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

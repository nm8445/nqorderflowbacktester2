"""
Feature computation for HMM regime classification.

Thirteen features per bar (10 per-bar + 3 session-level excursion constants).
The midprice Mt = (bid_px_00 + ask_px_00) / 2 is used for all price-based
calculations. Log returns rt = ln(Mt / Mt-1) feed into realized_vol,
autocorr_r1, trend_ratio, and normalized_d_move.

Per-bar features (z-scored per session via StandardScaler):
  Tier 1 — orthogonal, highest signal:
    delta_pct          — (buy_vol - sell_vol) / total_vol
    trend_ratio        — net midprice move / total path length
    obi_avg            — mean (bid_sz - ask_sz) / (bid_sz + ask_sz) across ticks
    realized_vol       — sqrt(sum(log_returns^2))
    trade_arrival_rate — trades per second

  Tier 2 — meaningfully orthogonal to Tier 1:
    delta_acceleration — delta_pct[i] - delta_pct[i-1] across bars
    obi_trend          — linear regression slope of OBI time series within bar
    vwap_deviation     — (close_price - vwap) / vwap_std at bar close

  Additional:
    autocorr_r1        — lag-1 autocorrelation of log returns
    normalized_d_move  — net_move / (realized_vol * open_price), Sharpe-like ratio

Session-level excursion features (appended after z-scoring, same value every row):
    max_excursion_ratio — abs(session_close - session_open) / (session_high - session_low)
                          1.0 = price moved cleanly in one direction; near 0 = whipsaw
    trend_consistency   — fraction of bars that close in the session's direction
    drawdown_ratio      — max_adverse_excursion / abs(net_move)
                          Low = clean trend; high = significant counter-trend pullbacks

All per-bar features are z-score normalized per session independently using
StandardScaler.fit_transform() before being assembled into the HMM input matrix.
Session-level features are appended after normalization to avoid std=0 collapse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "delta_pct",
    "trend_ratio",
    "obi_avg",
    "realized_vol",
    "trade_arrival_rate",
    "delta_acceleration",
    "obi_trend",
    "vwap_deviation",
    "autocorr_r1",
    "normalized_d_move",
    # session-level (constant per session, appended after z-scoring)
    "max_excursion_ratio",
    "trend_consistency",
    "drawdown_ratio",
]

_N_BAR_FEATURES = 10          # per-bar features only (before session columns are appended)
N_BAR_FEATURES  = _N_BAR_FEATURES  # public alias — use this for overnight feature counts

MIN_TICKS_PER_BAR = 3


def _session_features(valid_bars: list) -> np.ndarray:
    """
    Compute 3 session-level excursion features from completed RangeBar objects.

    Returns np.ndarray shape (3,): [max_excursion_ratio, trend_consistency, drawdown_ratio]
    """
    if len(valid_bars) < 2:
        return np.zeros(3)

    session_open  = valid_bars[0].open
    session_close = valid_bars[-1].close
    session_high  = max(b.high for b in valid_bars)
    session_low   = min(b.low  for b in valid_bars)
    net_move      = session_close - session_open
    price_range   = session_high - session_low

    max_excursion_ratio = abs(net_move) / price_range if price_range > 1e-9 else 0.0

    bullish = net_move >= 0
    directional_count = sum(
        1 for b in valid_bars
        if (bullish and b.close > b.open) or (not bullish and b.close < b.open)
    )
    trend_consistency = directional_count / len(valid_bars)

    abs_net = abs(net_move)
    if abs_net < 1.0:  # guard: < 1 point net move → ratio is meaningless
        drawdown_ratio = 0.0
    else:
        max_mae = 0.0
        if bullish:
            running_min = float("inf")
            for b in reversed(valid_bars):
                running_min = min(running_min, b.low)
                mae = b.high - running_min
                if mae > max_mae:
                    max_mae = mae
        else:
            running_max = float("-inf")
            for b in reversed(valid_bars):
                running_max = max(running_max, b.high)
                mae = running_max - b.low
                if mae > max_mae:
                    max_mae = mae
        drawdown_ratio = max_mae / abs_net

    return np.array([max_excursion_ratio, trend_consistency, drawdown_ratio])


def _bar_features(
    ticks: pd.DataFrame,
    prev_delta_pct: float,
    vwap_val: float,
    vwap_std: float,
) -> np.ndarray:
    n = len(ticks)
    if n == 0:
        return np.zeros(_N_BAR_FEATURES)

    buy_vol   = float(ticks.loc[ticks["aggressor"] == "buy",  "size"].sum())
    sell_vol  = float(ticks.loc[ticks["aggressor"] == "sell", "size"].sum())
    total_vol = buy_vol + sell_vol
    delta_pct = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0

    mids = ticks["mid_price"].to_numpy(dtype=float)

    if n >= 2 and np.all(mids > 0):
        diffs       = np.diff(mids)
        net_move    = mids[-1] - mids[0]
        total_path  = float(np.sum(np.abs(diffs)))
        trend_ratio = net_move / total_path if total_path > 1e-9 else 0.0
        log_returns = np.log(mids[1:] / mids[:-1])
        realized_vol = float(np.sqrt(np.sum(log_returns ** 2)))
    else:
        trend_ratio  = 0.0
        log_returns  = np.array([])
        realized_vol = 0.0

    bid_sz   = ticks["bid_sz"].to_numpy(dtype=float)
    ask_sz   = ticks["ask_sz"].to_numpy(dtype=float)
    total_sz = bid_sz + ask_sz
    obi_vals = np.where(total_sz > 0, (bid_sz - ask_sz) / total_sz, 0.0)
    obi_avg  = float(np.mean(obi_vals))

    if n >= 2:
        duration           = (ticks.index[-1] - ticks.index[0]).total_seconds()
        trade_arrival_rate = float(n / duration) if duration > 0 else float(n)
    else:
        trade_arrival_rate = 0.0

    delta_acceleration = delta_pct - prev_delta_pct

    if n >= 3:
        obi_trend = float(np.polyfit(np.arange(n, dtype=float), obi_vals, 1)[0])
    else:
        obi_trend = 0.0

    vwap_deviation = (mids[-1] - vwap_val) / vwap_std if vwap_std > 1e-6 else 0.0

    if len(log_returns) >= 3 and np.std(log_returns) > 1e-12:
        ac = float(np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1])
        autocorr_r1 = 0.0 if np.isnan(ac) else ac
    else:
        autocorr_r1 = 0.0

    if realized_vol > 1e-9 and n >= 2 and mids[0] > 0:
        normalized_d_move = (mids[-1] - mids[0]) / (realized_vol * mids[0])
    else:
        normalized_d_move = 0.0

    return np.array([
        delta_pct, trend_ratio, obi_avg, realized_vol, trade_arrival_rate,
        delta_acceleration, obi_trend, vwap_deviation, autocorr_r1, normalized_d_move,
    ])


def compute_session_features(
    enriched_ticks: pd.DataFrame,
    bars: list,
    vwap_df: pd.DataFrame,
    min_ticks: int = MIN_TICKS_PER_BAR,
    normalize: bool = True,
    include_session_features: bool = True,
) -> np.ndarray:
    """
    Compute z-score normalized feature matrix for one session.

    Parameters
    ----------
    enriched_ticks : pd.DataFrame
        Output of normalizer.normalize_enriched(), sliced to the session window.
    bars : list[RangeBar]
        40-range bars built from the same session window.
    vwap_df : pd.DataFrame
        Output of vwap.compute_vwap() anchored at session open.
    min_ticks : int
        Minimum ticks per bar to include. Bars below this are skipped.
    normalize : bool
        If True, apply StandardScaler per-session before returning.

    Returns
    -------
    np.ndarray
        Shape (n_valid_bars, 13) when include_session_features=True,
        shape (n_valid_bars, 10) when False. Empty array if fewer than 2 valid bars.
    """
    rows:       list[np.ndarray] = []
    valid_bars: list             = []
    prev_delta_pct = 0.0

    for bar in bars:
        close_ts = bar.close_time if bar.close_time is not None else bar.open_time
        bar_ticks = enriched_ticks[
            (enriched_ticks.index >= bar.open_time) &
            (enriched_ticks.index <= close_ts)
        ]
        if len(bar_ticks) < min_ticks:
            continue

        vwap_slice = vwap_df[vwap_df.index <= close_ts]
        if vwap_slice.empty:
            vwap_val, vwap_std = 0.0, 1.0
        else:
            row      = vwap_slice.iloc[-1]
            vwap_val = float(row["vwap"])
            vwap_std = max(float(row["std"]), 1e-6)

        feat = _bar_features(bar_ticks, prev_delta_pct, vwap_val, vwap_std)
        rows.append(feat)
        valid_bars.append(bar)
        prev_delta_pct = feat[0]

    n_out = len(FEATURE_NAMES) if include_session_features else _N_BAR_FEATURES
    if len(rows) < 2:
        return np.empty((0, n_out))

    X = np.array(rows)                                         # (n, 10)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize and X.shape[0] >= 2:
        X = StandardScaler().fit_transform(X)

    if include_session_features:
        # Append session-level features as constant columns (same value every row)
        sess = _session_features(valid_bars)                   # (3,)
        sess_cols = np.tile(sess, (X.shape[0], 1))             # (n, 3)
        X = np.hstack([X, sess_cols])                          # (n, 13)

    return X

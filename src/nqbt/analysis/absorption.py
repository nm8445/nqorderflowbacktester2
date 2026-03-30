"""
Absorption signal detector based on 40-range bar imbalances.

Six detection methods, all optional, designed to be used in combination.
The combined signal requires whichever subset of methods you enable to all agree.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHOD 1 — Rolling Percentile Normalization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Computes abs(buy_vol - sell_vol) for each bar. Signals when the current
bar's imbalance exceeds the Nth percentile of the last `lookback` bars.
Adapts to changing volatility regimes over time.

Optimizable:
  lookback   [500]  — too short = overfits recent volatility, too long = adapts too slowly
  percentile [80.0] — 80 is starting point, 90 is more aggressive

Watch out for: early-session bars with limited history will underperform.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHOD 2 — Volume-Normalized Imbalance Ratio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ratio = (buy_vol - sell_vol) / total_vol
Expresses imbalance as a fraction of total bar volume. No lookback needed.
Naturally adjusts to participation level — a 0.40 ratio means 70/30 split.

Optimizable:
  min_ratio        [0.40] — lower = more signals, noisier
  min_volume_floor [50]   — minimum bar volume before evaluating ratio.
                            prevents thin bars producing extreme spurious ratios.
                            start at the 25th percentile of bar volumes in your dataset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHOD 3 — Session VWAP / Activity Scaling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    threshold = activity_k × running_avg_bar_volume
Uses the session's running average bar volume as a live measure of activity.
Scales the imbalance threshold to the day's character — a thin RTH open
session has a lower threshold than a heavy news-driven session.

Optimizable:
  activity_k             [2.0]  — multiplier on running avg volume
  activity_warmup_bars   [20]   — bars needed before method activates.
                                  early-session instability before enough
                                  volume has accumulated makes this unreliable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHOD 4 — Z-Score of Imbalance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Z = (current_imbalance - rolling_mean) / rolling_std
Standardizes imbalance relative to recent history. Directly measures how
exceptional the current bar is, not just how large.

Optimizable:
  zscore_lookback   [100]  — rolling window for mean and std
  zscore_threshold  [2.0]  — signal when abs(Z) >= this. 1.5 = more signals,
                             2.0 = more selective

Watch out for: imbalance distributions are fat-tailed and non-normal.
Z-scores can understate tail risk. Best combined with the percentile check.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHOD 5 — ATR-Normalized Imbalance (Effort-Adjusted)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    absorption_strength = abs(raw_imbalance) / ATR(n)
Normalizes imbalance by market volatility. When volatility is high, larger
absolute imbalances are expected naturally — dividing by ATR removes that noise.
Directly captures the "effort vs. result" concept: large imbalance relative
to normal volatility = meaningful absorption.

Note on range bars: since every 40-range bar has H-L = 10 points by definition,
ATR variation comes from gaps between bars (True Range includes prev close to
current high/low). This still provides useful volatility normalization.

Optimizable:
  atr_period  [14]   — bars in ATR calculation
  atr_k       [1.0]  — signal when abs_imbalance / ATR >= this

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHOD 6 — EWMA Adaptive Threshold
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses an exponentially weighted moving average of recent imbalances as a
dynamic baseline. Recent bars receive more weight so the threshold drifts
toward current market conditions automatically. Faster adaptation than a
simple rolling window, no fixed window boundary artifacts.

    signal when abs_imbalance > ewma_k × EWMA(abs_imbalance, span)

Optimizable:
  ewma_span  [50]   — controls decay speed. lower = faster adaptation,
                      higher = smoother, slower
  ewma_k     [2.0]  — multiplier. higher = fewer but more extreme signals

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSORPTION CLASSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Applied after the combined signal fires:
  buy_absorbed  — buy-dominated imbalance, bar closed bearish.
                  Buyers were aggressive, sellers held ground. Bearish.
  sell_absorbed — sell-dominated imbalance, bar closed bullish.
                  Sellers were aggressive, buyers held ground. Bullish.
  unconfirmed   — imbalance aligns with bar direction. Aggression succeeded.
                  NOT absorption — do not trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from nqbt.analysis.range_bars import RangeBar


@dataclass
class BarSignal:
    """Full signal scores for a single range bar across all active methods."""
    bar_index:          int
    open_time:          pd.Timestamp
    price_open:         float
    price_close:        float
    bar_direction:      str
    raw_imbalance:      int
    abs_imbalance:      int
    imbalance_ratio:    float
    total_vol:          int
    # Method scores
    percentile_rank:    Optional[float]
    zscore:             Optional[float]
    atr_strength:       Optional[float]
    ewma_baseline:      Optional[float]
    activity_threshold: Optional[float]
    # Method signals
    signal_percentile:  bool
    signal_ratio:       bool
    signal_activity:    bool
    signal_zscore:      bool
    signal_atr:         bool
    signal_ewma:        bool
    # Combined
    signal_combined:    bool
    absorption_side:    Optional[str]


def score_bars(
    bars: list[RangeBar],
    # Method 1 — rolling percentile
    use_percentile: bool = True,
    lookback: int = 500,
    percentile: float = 80.0,
    # Method 2 — volume-normalized ratio
    use_ratio: bool = True,
    min_ratio: float = 0.40,
    min_volume_floor: int = 50,
    # Method 3 — session activity scaling
    use_activity: bool = False,
    activity_k: float = 2.0,
    activity_warmup_bars: int = 20,
    # Method 4 — z-score
    use_zscore: bool = False,
    zscore_lookback: int = 100,
    zscore_threshold: float = 2.0,
    # Method 5 — ATR-normalized
    use_atr: bool = False,
    atr_period: int = 14,
    atr_k: float = 1.0,
    # Method 6 — EWMA adaptive threshold
    use_ewma: bool = False,
    ewma_span: int = 50,
    ewma_k: float = 2.0,
    # Combined logic — 'all' requires every enabled method, 'any' requires one
    combine_mode: str = "all",
) -> pd.DataFrame:
    """
    Score range bars for absorption signals across up to 6 detection methods.

    Parameters
    ----------
    bars : list[RangeBar]
    use_* : bool
        Toggle each method on/off. Methods 1 and 2 are on by default.
    combine_mode : str
        'all' — every enabled method must fire (highest confidence, fewer signals)
        'any' — any enabled method firing is sufficient (more signals, noisier)

    Returns
    -------
    pd.DataFrame
        One row per bar. Columns include per-method signals, scores, and
        the final signal_combined + absorption_side.
    """
    if not bars:
        return pd.DataFrame()

    raw_imb   = np.array([b.buy_vol - b.sell_vol for b in bars], dtype=float)
    abs_imb   = np.abs(raw_imb)
    total_vol = np.array([b.total_vol for b in bars], dtype=float)
    highs     = np.array([b.high for b in bars], dtype=float)
    lows      = np.array([b.low for b in bars], dtype=float)
    closes    = np.array([b.close for b in bars], dtype=float)
    ratios    = np.where(total_vol > 0, raw_imb / total_vol, 0.0)

    # Pre-compute True Range for ATR
    true_ranges = np.zeros(len(bars))
    for i in range(len(bars)):
        hl = highs[i] - lows[i]
        if i == 0:
            true_ranges[i] = hl
        else:
            true_ranges[i] = max(hl, abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))

    # Pre-compute EWMA of abs_imbalance (pandas for simplicity)
    ewma_series = pd.Series(abs_imb).ewm(span=ewma_span, adjust=False).mean().to_numpy()

    signals: list[BarSignal] = []

    for i, bar in enumerate(bars):
        # ── Method 1: rolling percentile ──────────────────────────────
        if use_percentile and i >= 2:
            start    = max(0, i - lookback)
            pct_rank = float(np.mean(abs_imb[start:i] < abs_imb[i]) * 100)
            sig_pct  = pct_rank >= percentile
        else:
            pct_rank = None
            sig_pct  = False

        # ── Method 2: volume-normalized ratio ─────────────────────────
        if use_ratio:
            sig_rat = bar.total_vol >= min_volume_floor and abs(ratios[i]) >= min_ratio
        else:
            sig_rat = False

        # ── Method 3: session activity scaling ────────────────────────
        if use_activity and i >= activity_warmup_bars:
            running_avg_vol   = float(np.mean(total_vol[max(0, i - lookback):i]))
            act_threshold     = activity_k * running_avg_vol
            sig_act           = abs_imb[i] >= act_threshold
        else:
            act_threshold = None
            sig_act       = False

        # ── Method 4: z-score ─────────────────────────────────────────
        if use_zscore and i >= 3:
            start      = max(0, i - zscore_lookback)
            window     = raw_imb[start:i]
            roll_mean  = np.mean(window)
            roll_std   = np.std(window)
            if roll_std > 0:
                z       = (raw_imb[i] - roll_mean) / roll_std
                sig_z   = abs(z) >= zscore_threshold
            else:
                z, sig_z = 0.0, False
        else:
            z, sig_z = None, False

        # ── Method 5: ATR-normalized imbalance ────────────────────────
        if use_atr and i >= atr_period:
            atr         = float(np.mean(true_ranges[i - atr_period:i]))
            atr_strength = abs_imb[i] / atr if atr > 0 else 0.0
            sig_atr      = atr_strength >= atr_k
        else:
            atr_strength = None
            sig_atr      = False

        # ── Method 6: EWMA adaptive threshold ─────────────────────────
        if use_ewma and i >= 2:
            baseline  = ewma_series[i - 1]   # prior bar's EWMA, not current
            sig_ewma  = abs_imb[i] >= ewma_k * baseline if baseline > 0 else False
        else:
            baseline  = None
            sig_ewma  = False

        # ── Combined signal ───────────────────────────────────────────
        active_signals = []
        if use_percentile and i >= 2: active_signals.append(sig_pct)
        if use_ratio:                  active_signals.append(sig_rat)
        if use_activity and i >= activity_warmup_bars: active_signals.append(sig_act)
        if use_zscore and i >= 3:      active_signals.append(sig_z)
        if use_atr and i >= atr_period: active_signals.append(sig_atr)
        if use_ewma and i >= 2:        active_signals.append(sig_ewma)

        if not active_signals:
            sig_combined = False
        elif combine_mode == "all":
            sig_combined = all(active_signals)
        else:
            sig_combined = any(active_signals)

        # ── Absorption classification ─────────────────────────────────
        bar_dir  = "bull" if bar.close >= bar.open else "bear"
        dominant = "buy" if raw_imb[i] >= 0 else "sell"

        if sig_combined:
            if dominant == "buy" and bar_dir == "bear":
                absorption_side = "buy_absorbed"
            elif dominant == "sell" and bar_dir == "bull":
                absorption_side = "sell_absorbed"
            else:
                absorption_side = "unconfirmed"
        else:
            absorption_side = None

        signals.append(BarSignal(
            bar_index          = i,
            open_time          = bar.open_time,
            price_open         = bar.open,
            price_close        = bar.close,
            bar_direction      = bar_dir,
            raw_imbalance      = int(raw_imb[i]),
            abs_imbalance      = int(abs_imb[i]),
            imbalance_ratio    = round(float(ratios[i]), 4),
            total_vol          = bar.total_vol,
            percentile_rank    = round(pct_rank, 1) if pct_rank is not None else None,
            zscore             = round(float(z), 3) if z is not None else None,
            atr_strength       = round(float(atr_strength), 3) if atr_strength is not None else None,
            ewma_baseline      = round(float(baseline), 1) if baseline is not None else None,
            activity_threshold = round(float(act_threshold), 1) if act_threshold is not None else None,
            signal_percentile  = sig_pct,
            signal_ratio       = sig_rat,
            signal_activity    = sig_act,
            signal_zscore      = sig_z,
            signal_atr         = sig_atr,
            signal_ewma        = sig_ewma,
            signal_combined    = sig_combined,
            absorption_side    = absorption_side,
        ))

    rows = [
        {
            "open_time":          s.open_time,
            "price_open":         s.price_open,
            "price_close":        s.price_close,
            "bar_direction":      s.bar_direction,
            "raw_imbalance":      s.raw_imbalance,
            "imbalance_ratio":    s.imbalance_ratio,
            "total_vol":          s.total_vol,
            "percentile_rank":    s.percentile_rank,
            "zscore":             s.zscore,
            "atr_strength":       s.atr_strength,
            "ewma_baseline":      s.ewma_baseline,
            "activity_threshold": s.activity_threshold,
            "signal_percentile":  s.signal_percentile,
            "signal_ratio":       s.signal_ratio,
            "signal_activity":    s.signal_activity,
            "signal_zscore":      s.signal_zscore,
            "signal_atr":         s.signal_atr,
            "signal_ewma":        s.signal_ewma,
            "signal_combined":    s.signal_combined,
            "absorption_side":    s.absorption_side,
        }
        for s in signals
    ]

    return pd.DataFrame(rows).set_index("open_time")

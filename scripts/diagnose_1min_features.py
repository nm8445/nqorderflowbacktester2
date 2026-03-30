"""
Diagnostic: validate 1-minute bar feature values before HMM retraining.

Checks raw (un-normalized) feature values for specific sessions to confirm
1-minute bars produce sensible signals before committing to a full retrain.

Sessions checked:
  Overnight - Feb 18 2026 (18:00 Feb 17 -> 09:30 Feb 18)
    Expected: high abs_delta_pct, high trend_ratio, negative autocorr_r1
    (This session was classified directional at 77% confidence on range bars)

  RTH - Dec 8 2025 and Feb 18 2026 (09:30 -> 11:00)
    Dec 8  expected: trend_consistency > 0.65, drawdown_ratio < 0.35
    Feb 18 expected: trend_consistency > 0.60, drawdown_ratio < 0.40

Run from project root:
    python scripts/diagnose_1min_features.py
"""

import gc
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize_enriched
from nqbt.analysis.minute_bars import build_minute_bars
from nqbt.analysis.vwap import compute_vwap
from nqbt.analysis.features import compute_session_features, FEATURE_NAMES
from nqbt.analysis.overnight_features import (
    compute_overnight_features,
    OVERNIGHT_FEATURE_NAMES,
)

ET = "America/New_York"

OVERNIGHT_TARGET = date(2026, 2, 18)   # overnight = 18:00 Feb 17 to 09:30 Feb 18
RTH_TARGETS      = [date(2025, 12, 8), date(2026, 2, 18)]


def load_ticks(trading_date: date):
    prev_day = trading_date - timedelta(days=1)
    raw      = fetch_and_load(str(prev_day), str(trading_date + timedelta(days=1)))
    enriched = normalize_enriched(raw)
    del raw
    gc.collect()
    return enriched


def diagnose_overnight(trading_date: date) -> None:
    print(f"\n{'='*65}")
    print(f"  OVERNIGHT DIAGNOSTIC: {trading_date}")
    print(f"  Session: 18:00 {trading_date - timedelta(days=1)} -> 09:30 {trading_date}")
    print(f"{'='*65}")

    enriched = load_ticks(trading_date)
    prev_day = trading_date - timedelta(days=1)
    start    = pd.Timestamp(f"{prev_day} 18:00:00", tz=ET)
    end      = pd.Timestamp(f"{trading_date} 09:30:00", tz=ET)
    ticks    = enriched[(enriched.index >= start) & (enriched.index < end)]
    del enriched
    gc.collect()

    print(f"  Tick count: {len(ticks):,}")
    bars = build_minute_bars(ticks)
    print(f"  1-minute bars: {len(bars)}")

    if len(bars) < 5:
        print("  INSUFFICIENT DATA")
        return

    feat_raw = compute_overnight_features(ticks, bars, min_ticks=1, normalize=False)
    if feat_raw.shape[0] == 0:
        print("  INSUFFICIENT VALID BARS")
        return

    print(f"  Valid bars (min_ticks>=1): {feat_raw.shape[0]}")
    print()
    print(f"  {'Feature':<28}  {'Session Mean':>14}  {'Session Std':>12}")
    print(f"  {'-'*58}")
    for fi, fname in enumerate(OVERNIGHT_FEATURE_NAMES):
        col  = feat_raw[:, fi]
        mean = float(np.nanmean(col))
        std  = float(np.nanstd(col))
        print(f"  {fname:<28}  {mean:>+14.6f}  {std:>12.6f}")

    print()
    print("  Key signals (session means):")
    idx = {n: i for i, n in enumerate(OVERNIGHT_FEATURE_NAMES)}
    abs_d = float(np.nanmean(feat_raw[:, idx["abs_delta_pct"]]))
    tr    = float(np.nanmean(feat_raw[:, idx["trend_ratio"]]))
    ac    = float(np.nanmean(feat_raw[:, idx["autocorr_r1"]]))
    dec   = float(feat_raw[-1, idx["overnight_deceleration"]])   # last bar = full-session decel
    print(f"    abs_delta_pct (session mean)       : {abs_d:+.6f}  (expect > 0.10)")
    print(f"    trend_ratio   (session mean)       : {tr:+.6f}  (expect > 0.40)")
    print(f"    autocorr_r1   (session mean)       : {ac:+.6f}  (expect negative for directional)")
    print(f"    overnight_decel (last bar value)   : {dec:+.6f}  (negative = deceleration into open)")


def diagnose_rth(trading_date: date) -> None:
    print(f"\n{'='*65}")
    print(f"  RTH DIAGNOSTIC: {trading_date}")
    print(f"  Session: 09:30 -> 11:00 {trading_date}")
    print(f"{'='*65}")

    enriched = load_ticks(trading_date)
    start    = pd.Timestamp(f"{trading_date} 09:30:00", tz=ET)
    end      = pd.Timestamp(f"{trading_date} 11:00:00", tz=ET)
    ticks    = enriched[(enriched.index >= start) & (enriched.index <= end)]
    del enriched
    gc.collect()

    print(f"  Tick count: {len(ticks):,}")
    bars = build_minute_bars(ticks)
    print(f"  1-minute bars: {len(bars)}")

    if len(bars) < 5:
        print("  INSUFFICIENT DATA")
        return

    vwap_df  = compute_vwap(ticks)
    feat_raw = compute_session_features(
        ticks, bars, vwap_df,
        min_ticks=1, normalize=False,
        include_session_features=True,
    )
    if feat_raw.shape[0] == 0:
        print("  INSUFFICIENT VALID BARS")
        return

    print(f"  Valid bars (min_ticks>=1): {feat_raw.shape[0]}")

    # Session-level excursion features are constant across all rows (appended after z-scoring)
    # With normalize=False they are raw values in all rows — read from row 0
    max_excur = float(feat_raw[0, 10])
    trend_con = float(feat_raw[0, 11])
    drawdown  = float(feat_raw[0, 12])

    # Session direction from bar OHLC (first bar open, last bar close)
    session_open  = bars[0].open
    session_close = bars[-1].close
    direction     = "bullish" if session_close > session_open else "bearish"
    net_move      = abs(session_close - session_open)

    print()
    print(f"  Session direction : {direction}")
    print(f"  Net move          : {net_move:.2f} points")
    print(f"  Session open      : {session_open:.2f}")
    print(f"  Session close     : {session_close:.2f}")
    print()
    print(f"  {'Feature':<28}  {'Raw Value':>12}  Threshold")
    print(f"  {'-'*55}")

    if trading_date == date(2025, 12, 8):
        thresholds = {"trend_consistency": "> 0.65", "drawdown_ratio": "< 0.35"}
    else:
        thresholds = {"trend_consistency": "> 0.60", "drawdown_ratio": "< 0.40"}

    print(f"  {'max_excursion_ratio':<28}  {max_excur:>12.6f}  (net_move / gross_range)")
    print(f"  {'trend_consistency':<28}  {trend_con:>12.6f}  {thresholds['trend_consistency']}")
    print(f"  {'drawdown_ratio':<28}  {drawdown:>12.6f}  {thresholds['drawdown_ratio']}")

    # Also show per-bar feature means for context
    print()
    print(f"  Per-bar feature means (raw, first 10):")
    print(f"  {'Feature':<28}  {'Mean':>12}")
    print(f"  {'-'*44}")
    for fi, fname in enumerate(FEATURE_NAMES[:10]):
        mean = float(np.nanmean(feat_raw[:, fi]))
        print(f"  {fname:<28}  {mean:>+12.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("1-MINUTE BAR FEATURE DIAGNOSTIC")
    print("Do NOT retrain until values are confirmed.")
    print()

    print("Loading overnight diagnostic...")
    diagnose_overnight(OVERNIGHT_TARGET)

    for rth_date in RTH_TARGETS:
        diagnose_rth(rth_date)

    print()
    print("="*65)
    print("DIAGNOSTIC COMPLETE. Review values above before retraining.")
    print("="*65)

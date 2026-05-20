"""
Build VWAP reaction signal cache.

Continuation strategy logic:
  LONG:  Price trending above VWAP (support). Pulls back into VWAP zone
         (VWAP +/- 3 pts). Seller absorption on signal bar + bullish confirm.
         5-min bias must be long. Buy for continuation.
  SHORT: Price trending below VWAP (resistance). Pushes up into VWAP zone.
         Buyer absorption on signal bar + bearish confirm.
         5-min bias must be short. Sell for continuation.

Bias: determined by most recent 5-min candle close vs VWAP at that time.
      Initial bias at 7pm ET = current price vs VWAP.
      Holds until flipped by a new 5-min close on the other side.

Entry window: 7pm ET to 4:59pm ET (skip first hour for VWAP to develop).

Data sources:
  - signal_cache: 40-range bars
  - vwap_cache: 1-min VWAP values
  - timebars_5min: 5-min bars for ATR + bias tracking

Cache location: D:/trading_pythonbacktest_data/vwap_reaction_cache/
"""

import gc
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_5MIN_DIR = DATA_DIR / "timebars_5min"
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"

VWAP_ZONE_POINTS = 3.0   # +/- 3 points around VWAP
ATR_SL_MULTIPLE = 1.0    # default SL buffer (parameterized in backtest)
ATR_TP_MULTIPLE = 2.0    # default TP buffer (parameterized in backtest)
ATR_PERIOD = 14           # standard ATR lookback


def load_5min_bars_for_date(target_date: str) -> pd.DataFrame | None:
    """Load 5-min bars for a date, return DataFrame with high/low/close, time index, and ATR."""
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    bar_file = TIMEBARS_5MIN_DIR / f"timebars_5min_{target.strftime('%Y_%m_%d')}.pkl"
    if not bar_file.exists():
        return None

    with open(bar_file, 'rb') as fp:
        bars = pickle.load(fp)

    if not bars:
        return None

    rows = [{
        'timestamp': b['open_time'],
        'high': b['high'],
        'low': b['low'],
        'close': b['close'],
    } for b in bars]

    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    # Index at close time (open + 5min) so lookups only see completed bars
    df.index = df.index + pd.Timedelta(minutes=5)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['atr'] = df['tr'].rolling(ATR_PERIOD, min_periods=1).mean()
    return df


def get_atr_at_time(bars_5min: pd.DataFrame, signal_time) -> float | None:
    """Get the most recent 5-min ATR value at or before signal_time."""
    if bars_5min is None or bars_5min.empty:
        return None
    prior = bars_5min[bars_5min.index <= signal_time]
    if prior.empty:
        return None
    return float(prior['atr'].iloc[-1])


def has_absorption(bar, min_delta: int = 30) -> dict:
    """
    Check if a bar has absorption at any price level.

    Bearish absorption (for shorts): bar closes bearish, has buy delta >= min_delta
    Bullish absorption (for longs): bar closes bullish, has sell delta <= -min_delta
    """
    closed_bearish = bar.close < bar.open
    closed_bullish = bar.close > bar.open

    bearish_levels = []
    bullish_levels = []

    for price, lv in bar.levels.items():
        delta = lv.delta
        if closed_bearish and delta >= min_delta:
            bearish_levels.append((price, delta))
        elif closed_bullish and delta <= -min_delta:
            bullish_levels.append((price, delta))

    return {
        "bearish": closed_bearish and len(bearish_levels) > 0,
        "bullish": closed_bullish and len(bullish_levels) > 0,
        "bearish_levels": bearish_levels,
        "bullish_levels": bullish_levels,
    }


def get_vwap_at_time(vwap_df: pd.DataFrame, bar_time) -> float | None:
    """Get VWAP value at or before bar_time from 1-min VWAP cache."""
    if vwap_df is None or vwap_df.empty:
        return None
    prior = vwap_df[vwap_df.index <= bar_time]
    if prior.empty:
        return None
    return float(prior['vwap'].iloc[-1])


def get_5min_bias(bars_5min: pd.DataFrame, vwap_df: pd.DataFrame, as_of_time) -> str | None:
    """
    Determine directional bias from the most recent 5-min candle close vs VWAP zone.

    Only flips when close is outside the zone (+/- 3 pts). Inside zone = hold previous.
    Returns 'long' if last 5-min close > VWAP + zone, 'short' if < VWAP - zone, None if no data.

    NOTE: This is a stateless per-bar check. The cache builder calls it once per signal,
    so "hold previous" is approximated by scanning backwards through recent bars.
    """
    if bars_5min is None or bars_5min.empty or vwap_df is None or vwap_df.empty:
        return None

    prior_bars = bars_5min[bars_5min.index <= as_of_time]
    if prior_bars.empty:
        return None

    # Walk backwards through recent 5-min bars to find the most recent one
    # that closed outside the VWAP zone — that determines current bias.
    for idx in range(len(prior_bars) - 1, max(len(prior_bars) - 50, -1), -1):
        bar_close = float(prior_bars['close'].iloc[idx])
        bar_time = prior_bars.index[idx]
        vwap_at_bar = get_vwap_at_time(vwap_df, bar_time)
        if vwap_at_bar is None:
            continue
        zone_high = vwap_at_bar + VWAP_ZONE_POINTS
        zone_low = vwap_at_bar - VWAP_ZONE_POINTS
        if bar_close > zone_high:
            return "long"
        elif bar_close < zone_low:
            return "short"
    return None


def detect_vwap_reactions(
    bars: list,
    vwap_df: pd.DataFrame,
    bars_5min: pd.DataFrame | None = None,
    zone_points: float = VWAP_ZONE_POINTS,
    min_delta: int = 30,
    atr_sl_multiple: float = ATR_SL_MULTIPLE,
    atr_tp_multiple: float = ATR_TP_MULTIPLE,
) -> list[dict]:
    """
    Scan range bars for VWAP reaction continuation setups with absorption confirmation.

    LONG setup:
      1. Signal bar touches VWAP zone (+/- zone_points), bullish absorption
      2. Confirm bar also touches the zone and closes bullish
      3. 5-min bias is long (price trending above VWAP)
      4. Price approached VWAP from above in prior 10 bars (pullback to support)

    SHORT setup:
      1. Signal bar touches VWAP zone, bearish absorption
      2. Confirm bar also touches the zone and closes bearish
      3. 5-min bias is short (price trending below VWAP)
      4. Price approached VWAP from below in prior 10 bars (push up to resistance)
    """
    signals = []

    # Entry window: 7pm ET to 4:59pm ET (skip first hour of session for VWAP development)
    entry_start = "19:00"
    entry_end = "16:59"

    for i in range(len(bars) - 1):
        signal_bar = bars[i]
        confirm_bar = bars[i + 1]

        if not signal_bar.closed or not confirm_bar.closed:
            continue

        bar_time = signal_bar.close_time

        # Time filter: 7pm ET to 4:59pm ET
        if hasattr(bar_time, 'tz_convert'):
            bar_time_et = bar_time.tz_convert(ET)
        else:
            bar_time_et = pd.Timestamp(bar_time, tz='UTC').tz_convert(ET)

        hour_min = bar_time_et.strftime("%H:%M")
        # Valid times: 19:00-23:59 or 00:00-16:59
        if "17:00" <= hour_min < "19:00":
            continue

        # Check absorption on signal bar
        absorption = has_absorption(signal_bar, min_delta)
        if not absorption["bearish"] and not absorption["bullish"]:
            continue

        # Get VWAP at signal bar's time
        vwap_value = get_vwap_at_time(vwap_df, bar_time)
        if vwap_value is None:
            continue

        vwap_zone_high = vwap_value + zone_points
        vwap_zone_low = vwap_value - zone_points

        # Signal bar must touch the VWAP zone
        if not (signal_bar.high >= vwap_zone_low and signal_bar.low <= vwap_zone_high):
            continue

        # Confirm bar must also touch the VWAP zone
        confirm_vwap = get_vwap_at_time(vwap_df, confirm_bar.close_time)
        if confirm_vwap is None:
            confirm_vwap = vwap_value
        confirm_zone_high = confirm_vwap + zone_points
        confirm_zone_low = confirm_vwap - zone_points

        if not (confirm_bar.high >= confirm_zone_low and confirm_bar.low <= confirm_zone_high):
            continue

        # Get 5-min bias
        bias = get_5min_bias(bars_5min, vwap_df, bar_time)
        if bias is None:
            continue

        # Approach direction: check prior 10 bars
        lookback = min(10, i)
        approached_from_above = any(
            bars[j].close > vwap_zone_high for j in range(i - lookback, i)
            if bars[j].closed
        )
        approached_from_below = any(
            bars[j].close < vwap_zone_low for j in range(i - lookback, i)
            if bars[j].closed
        )

        confirm_bearish = confirm_bar.close < confirm_bar.open
        confirm_bullish = confirm_bar.close > confirm_bar.open

        # Get 5-min ATR at signal time
        atr = get_atr_at_time(bars_5min, bar_time)
        atr_buffer = atr * atr_sl_multiple if atr else 0.0
        atr_tp_buffer = atr * atr_tp_multiple if atr else 0.0

        # LONG: bias is long (VWAP = support), approached from above (pullback),
        #       bullish absorption, confirm closes bullish
        if (bias == "long"
                and approached_from_above
                and absorption["bullish"]
                and confirm_bullish):
            stop_loss = signal_bar.low - atr_buffer
            target = confirm_bar.close + atr_tp_buffer
            signals.append({
                "bar_index": i,
                "direction": "long",
                "signal_time": bar_time,
                "confirm_time": confirm_bar.close_time,
                "entry_price": confirm_bar.close,
                "stop_loss": stop_loss,
                "target": target,
                "atr": atr,
                "atr_buffer": atr_buffer,
                "vwap": vwap_value,
                "vwap_zone_high": vwap_zone_high,
                "vwap_zone_low": vwap_zone_low,
                "bias": bias,
                "signal_bar_open": signal_bar.open,
                "signal_bar_high": signal_bar.high,
                "signal_bar_low": signal_bar.low,
                "signal_bar_close": signal_bar.close,
                "confirm_bar_close": confirm_bar.close,
                "absorption_levels": absorption["bullish_levels"],
            })

        # SHORT: bias is short (VWAP = resistance), approached from below (push up),
        #        bearish absorption, confirm closes bearish
        if (bias == "short"
                and approached_from_below
                and absorption["bearish"]
                and confirm_bearish):
            stop_loss = signal_bar.high + atr_buffer
            target = confirm_bar.close - atr_tp_buffer
            signals.append({
                "bar_index": i,
                "direction": "short",
                "signal_time": bar_time,
                "confirm_time": confirm_bar.close_time,
                "entry_price": confirm_bar.close,
                "stop_loss": stop_loss,
                "target": target,
                "atr": atr,
                "atr_buffer": atr_buffer,
                "vwap": vwap_value,
                "vwap_zone_high": vwap_zone_high,
                "vwap_zone_low": vwap_zone_low,
                "bias": bias,
                "signal_bar_open": signal_bar.open,
                "signal_bar_high": signal_bar.high,
                "signal_bar_low": signal_bar.low,
                "signal_bar_close": signal_bar.close,
                "confirm_bar_close": confirm_bar.close,
                "absorption_levels": absorption["bearish_levels"],
            })

    return signals


def process_single_date(target_date: str) -> dict:
    """Process a single date for VWAP reaction signals."""
    try:
        print(f"Processing {target_date}...", end=" ", flush=True)

        VWAP_REACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = VWAP_REACTION_CACHE_DIR / f"{target_date}.pkl"

        if cache_file.exists():
            print("already cached")
            return {"date": target_date, "status": "cached"}

        # Load signal cache (40-range bars)
        signal_file = SIGNAL_CACHE_DIR / f"{target_date}.pkl"
        if not signal_file.exists():
            print("no signal cache")
            return {"date": target_date, "status": "no_signal_cache"}

        with open(signal_file, 'rb') as f:
            signal_data = pickle.load(f)

        bars = signal_data["bars"]
        if not bars:
            print("empty bars")
            return {"date": target_date, "status": "empty_data"}

        # Load VWAP cache
        vwap_file = VWAP_CACHE_DIR / f"{target_date}.pkl"
        if not vwap_file.exists():
            print("no VWAP cache")
            return {"date": target_date, "status": "no_vwap_cache"}

        with open(vwap_file, 'rb') as f:
            vwap_df = pickle.load(f)

        if vwap_df is None or vwap_df.empty:
            print("empty VWAP")
            return {"date": target_date, "status": "empty_vwap"}

        # Load 5-min bars for ATR + bias
        bars_5min = load_5min_bars_for_date(target_date)

        # Detect VWAP reaction signals
        signals = detect_vwap_reactions(bars, vwap_df, bars_5min=bars_5min)

        # Save
        shorts = sum(1 for s in signals if s["direction"] == "short")
        longs = sum(1 for s in signals if s["direction"] == "long")

        cache_data = {
            "date": target_date,
            "signals": signals,
            "num_signals": len(signals),
            "num_shorts": shorts,
            "num_longs": longs,
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        vwap_val = vwap_df["vwap"].iloc[-1]
        print(f"OK | VWAP {vwap_val:.0f} | {shorts}S {longs}L")

        return {"date": target_date, "status": "success"}

    except Exception as e:
        print(f"ERROR - {e}")
        return {"date": target_date, "status": "error", "error": str(e)}


def build_vwap_reaction_cache(start_date: str, end_date: str):
    """Build VWAP reaction cache for date range."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(f"Building VWAP reaction cache for {len(dates)} dates")
    print(f"Date range: {start_date} to {end_date}")
    print(f"VWAP zone: +/-{VWAP_ZONE_POINTS} points")
    print(f"Cache directory: {VWAP_REACTION_CACHE_DIR}")
    print()

    VWAP_REACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for d in dates:
        results.append(process_single_date(d))
        gc.collect()

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    cached = sum(1 for r in results if r["status"] == "cached")
    errors = sum(1 for r in results if r["status"] == "error")
    no_signal = sum(1 for r in results if r["status"] == "no_signal_cache")
    no_vwap = sum(1 for r in results if r["status"] == "no_vwap_cache")

    print()
    print("=" * 80)
    print("VWAP REACTION CACHE BUILD COMPLETE")
    print("=" * 80)
    print(f"Total dates:       {len(dates)}")
    print(f"Already cached:    {cached}")
    print(f"Newly cached:      {success}")
    print(f"No signal cache:   {no_signal}")
    print(f"No VWAP cache:     {no_vwap}")
    print(f"Errors:            {errors}")
    print(f"Cache location:    {VWAP_REACTION_CACHE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    START_DATE = "2025-03-13"
    END_DATE = "2026-04-17"

    build_vwap_reaction_cache(
        start_date=START_DATE,
        end_date=END_DATE,
    )

"""
Build POC reaction signal cache.

Strategy logic:
  SHORT: Price is above today's developing VAH, enters prev day's POC zone
         (POC ± 10 points) from below → sell toward today's developing POC.
  LONG:  Price is below today's developing VAL, enters prev day's POC zone
         (POC ± 10 points) from above → buy toward today's developing POC.

Data sources:
  - daily_profile_cache: previous day's EOD POC
  - signal_cache: 40-range bars + developing profiles (VAH/VAL/POC per minute)

Cache location: D:/trading_pythonbacktest_data/poc_reaction_cache/
"""

import gc
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import numpy as np

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
DAILY_PROFILE_CACHE_DIR = DATA_DIR / "daily_profile_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
TIMEBARS_5MIN_DIR = DATA_DIR / "timebars_5min"
POC_REACTION_CACHE_DIR = DATA_DIR / "poc_reaction_cache"

POC_ZONE_POINTS = 10.0  # ±10 points around prev day POC
ATR_SL_MULTIPLE = 0.50  # stop loss buffer = 0.50 × 5-min ATR
ATR_TP_MULTIPLE = 1.00  # target = 1.00 × 5-min ATR from entry
ATR_PERIOD = 14         # standard ATR lookback
MAX_POC_DAYS = 10       # rolling window of recent POCs


def get_recent_pocs(target_date: str, max_days: int = MAX_POC_DAYS) -> list[dict]:
    """
    Get the last N trading days' EOD POCs from daily profile cache.

    Returns list of dicts with 'date' and 'poc', most recent first.
    Skips weekends/holidays (days with no cache file).
    """
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    pocs = []
    # Look back up to max_days * 2 calendar days to find enough trading days
    for i in range(1, max_days * 2 + 1):
        prev = target - timedelta(days=i)
        cache_file = DAILY_PROFILE_CACHE_DIR / f"{prev.strftime('%Y-%m-%d')}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                profile = pickle.load(f)
            pocs.append({
                "date": prev.strftime("%Y-%m-%d"),
                "poc": profile["poc"],
            })
            if len(pocs) >= max_days:
                break
    return pocs


def load_5min_bars_for_date(target_date: str) -> pd.DataFrame | None:
    """Load 5-min bars for a date, return DataFrame with high/low/close and time index."""
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
    # Calculate true range and rolling ATR
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
    # Find bars at or before signal time
    prior = bars_5min[bars_5min.index <= signal_time]
    if prior.empty:
        return None
    return float(prior['atr'].iloc[-1])


def get_developing_profile_at(profiles: dict, bar_time) -> object | None:
    """
    Get the most recent developing profile at or before bar_time.
    Profiles are keyed by timestamp, refreshed every minute.
    """
    best_profile = None
    best_time = None
    for prof_time, profile in profiles.items():
        if prof_time <= bar_time and profile is not None:
            if best_time is None or prof_time > best_time:
                best_profile = profile
                best_time = prof_time
    return best_profile


def has_absorption(bar, min_delta: int = 30) -> dict:
    """
    Check if a bar has absorption at any price level.

    Bearish absorption (for shorts): bar closes bearish, has buy delta >= min_delta
      (buyers absorbed into selling pressure)
    Bullish absorption (for longs): bar closes bullish, has sell delta <= -min_delta
      (sellers absorbed into buying pressure)

    Returns dict with 'bearish' and 'bullish' bools + absorption levels.
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


def detect_poc_reactions(
    bars: list,
    profiles: dict,
    recent_pocs: list[dict],
    bars_5min: pd.DataFrame | None = None,
    zone_points: float = POC_ZONE_POINTS,
    min_delta: int = 30,
    atr_sl_multiple: float = ATR_SL_MULTIPLE,
    atr_tp_multiple: float = ATR_TP_MULTIPLE,
) -> list[dict]:
    """
    Scan range bars for POC reaction setups with absorption confirmation.

    Checks against a rolling window of recent POCs (up to 10 trading days).

    SHORT setup:
      1. Signal bar touches a recent POC zone (±zone_points), bearish absorption
      2. Confirm bar also touches the zone and closes bearish
      3. Price is at/above today's developing VAH (±5 pts)
      4. Price approached from below in prior 10 bars

    LONG setup:
      1. Signal bar touches a recent POC zone (±zone_points), bullish absorption
      2. Confirm bar also touches the zone and closes bullish
      3. Price is at/below today's developing VAL (±5 pts)
      4. Price approached from above in prior 10 bars
    """
    signals = []

    for i in range(len(bars) - 1):
        signal_bar = bars[i]
        confirm_bar = bars[i + 1]

        if not signal_bar.closed or not confirm_bar.closed:
            continue

        bar_time = signal_bar.close_time

        # Check absorption on signal bar
        absorption = has_absorption(signal_bar, min_delta)
        if not absorption["bearish"] and not absorption["bullish"]:
            continue

        # Get developing profile at signal bar's time
        dev_profile = get_developing_profile_at(profiles, bar_time)
        if dev_profile is None:
            continue

        dev_vah = dev_profile.vah
        dev_val = dev_profile.val
        dev_poc = dev_profile.poc

        # Location filter: ±5 points from developing VAH/VAL
        bar_close = signal_bar.close
        at_or_above_vah = bar_close >= dev_vah - 5.0
        at_or_below_val = bar_close <= dev_val + 5.0

        # Get 5-min ATR at signal time for stop/target buffer
        atr = get_atr_at_time(bars_5min, bar_time)
        atr_buffer = atr * atr_sl_multiple if atr else 0.0
        atr_tp_buffer = atr * atr_tp_multiple if atr else 0.0

        # Check each recent POC
        for poc_entry in recent_pocs:
            poc_price = poc_entry["poc"]
            poc_date = poc_entry["date"]
            poc_zone_high = poc_price + zone_points
            poc_zone_low = poc_price - zone_points

            # Signal bar must touch the POC zone
            if not (signal_bar.high >= poc_zone_low and signal_bar.low <= poc_zone_high):
                continue

            # Confirm bar must also touch the POC zone
            if not (confirm_bar.high >= poc_zone_low and confirm_bar.low <= poc_zone_high):
                continue

            # Approach direction: check prior 10 bars
            lookback = min(10, i)
            approached_from_below = any(
                bars[j].close < poc_zone_low for j in range(i - lookback, i)
                if bars[j].closed
            )
            approached_from_above = any(
                bars[j].close > poc_zone_high for j in range(i - lookback, i)
                if bars[j].closed
            )

            confirm_bearish = confirm_bar.close < confirm_bar.open
            confirm_bullish = confirm_bar.close > confirm_bar.open

            # SHORT: at/above developing VAH, approached from below,
            #        bearish absorption, confirm closes bearish
            if (at_or_above_vah
                    and approached_from_below
                    and absorption["bearish"]
                    and confirm_bearish):
                stop_loss = signal_bar.high + atr_buffer
                target = dev_vah  # locked at entry
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
                    "poc_source_date": poc_date,
                    "prev_poc": poc_price,
                    "poc_zone_high": poc_zone_high,
                    "poc_zone_low": poc_zone_low,
                    "dev_vah": dev_vah,
                    "dev_val": dev_val,
                    "dev_poc": dev_poc,
                    "signal_bar_open": signal_bar.open,
                    "signal_bar_high": signal_bar.high,
                    "signal_bar_low": signal_bar.low,
                    "signal_bar_close": signal_bar.close,
                    "confirm_bar_close": confirm_bar.close,
                    "absorption_levels": absorption["bearish_levels"],
                })

            # LONG: at/below developing VAL, approached from above,
            #       bullish absorption, confirm closes bullish
            if (at_or_below_val
                    and approached_from_above
                    and absorption["bullish"]
                    and confirm_bullish):
                stop_loss = signal_bar.low - atr_buffer
                target = dev_val  # locked at entry
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
                    "poc_source_date": poc_date,
                    "prev_poc": poc_price,
                    "poc_zone_high": poc_zone_high,
                    "poc_zone_low": poc_zone_low,
                    "dev_vah": dev_vah,
                    "dev_val": dev_val,
                    "dev_poc": dev_poc,
                    "signal_bar_open": signal_bar.open,
                    "signal_bar_high": signal_bar.high,
                    "signal_bar_low": signal_bar.low,
                    "signal_bar_close": signal_bar.close,
                    "confirm_bar_close": confirm_bar.close,
                    "absorption_levels": absorption["bullish_levels"],
                })

    return signals


def process_single_date(target_date: str) -> dict:
    """Process a single date for POC reaction signals."""
    try:
        print(f"Processing {target_date}...", end=" ", flush=True)

        POC_REACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = POC_REACTION_CACHE_DIR / f"{target_date}.pkl"

        if cache_file.exists():
            print("already cached")
            return {"date": target_date, "status": "cached"}

        # Get rolling window of recent POCs (up to 10 trading days)
        recent_pocs = get_recent_pocs(target_date)
        if not recent_pocs:
            print("no recent POCs")
            return {"date": target_date, "status": "no_prev_poc"}

        # Load signal cache for this date (has bars + developing profiles)
        signal_file = SIGNAL_CACHE_DIR / f"{target_date}.pkl"
        if not signal_file.exists():
            print("no signal cache")
            return {"date": target_date, "status": "no_signal_cache"}

        with open(signal_file, 'rb') as f:
            signal_data = pickle.load(f)

        bars = signal_data["bars"]
        profiles = signal_data["profiles"]

        if not bars or not profiles:
            print("empty signal data")
            return {"date": target_date, "status": "empty_data"}

        # Load 5-min bars for ATR calculation
        bars_5min = load_5min_bars_for_date(target_date)

        # Detect POC reaction signals
        signals = detect_poc_reactions(bars, profiles, recent_pocs, bars_5min=bars_5min)

        # Save
        shorts = sum(1 for s in signals if s["direction"] == "short")
        longs = sum(1 for s in signals if s["direction"] == "long")

        cache_data = {
            "date": target_date,
            "recent_pocs": recent_pocs,
            "num_pocs": len(recent_pocs),
            "signals": signals,
            "num_signals": len(signals),
            "num_shorts": shorts,
            "num_longs": longs,
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        poc_prices = [f"{p['poc']:.0f}" for p in recent_pocs[:3]]
        print(f"OK | {len(recent_pocs)} POCs [{', '.join(poc_prices)}...] | {shorts}S {longs}L")

        return {"date": target_date, "status": "success"}

    except Exception as e:
        print(f"ERROR - {e}")
        return {"date": target_date, "status": "error", "error": str(e)}


def build_poc_reaction_cache(start_date: str, end_date: str):
    """Build POC reaction cache for date range."""

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(f"Building POC reaction cache for {len(dates)} dates")
    print(f"Date range: {start_date} to {end_date}")
    print(f"POC zone: ±{POC_ZONE_POINTS} points")
    print(f"Cache directory: {POC_REACTION_CACHE_DIR}")
    print()

    POC_REACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for d in dates:
        results.append(process_single_date(d))
        gc.collect()

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    cached = sum(1 for r in results if r["status"] == "cached")
    errors = sum(1 for r in results if r["status"] == "error")
    no_prev = sum(1 for r in results if r["status"] == "no_prev_poc")
    no_signal = sum(1 for r in results if r["status"] == "no_signal_cache")

    print()
    print("=" * 80)
    print("POC REACTION CACHE BUILD COMPLETE")
    print("=" * 80)
    print(f"Total dates:       {len(dates)}")
    print(f"Already cached:    {cached}")
    print(f"Newly cached:      {success}")
    print(f"No prev POC:       {no_prev}")
    print(f"No signal cache:   {no_signal}")
    print(f"Errors:            {errors}")
    print(f"Cache location:    {POC_REACTION_CACHE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    START_DATE = "2025-03-13"
    END_DATE = "2026-04-08"

    build_poc_reaction_cache(
        start_date=START_DATE,
        end_date=END_DATE,
    )

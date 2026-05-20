"""
Build signal cache for FULL SESSION: 7pm-4:55pm (prop firm hours)

Same logic as build_signal_cache.py but removes RTH-only filter.
Saves to: signal_cache_full_session/
"""

from datetime import datetime, timedelta
import pandas as pd
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

from nqbt.analysis.range_bars import build_range_bars, RangeBar
from nqbt.analysis.volume_profile import VolumeProfile
from nqbt.data.normalizer import normalize
import databento as db

TICK_SIZE = 0.25
ET = "America/New_York"
CACHE_DIR = Path("D:/trading_pythonbacktest_data")
SIGNAL_CACHE_DIR = CACHE_DIR / "signal_cache_full_session"  # New directory


def analyze_bar_characteristics(bar: RangeBar, min_delta: int = 30) -> dict:
    """Analyze absorption levels in a bar."""
    closed_bearish = bar.close < bar.open
    closed_bullish = bar.close > bar.open

    if not closed_bearish and not closed_bullish:
        return {"absorption_levels": []}

    absorption_levels = []
    for price, lv in bar.levels.items():
        delta = lv.delta
        if closed_bearish and delta >= min_delta:
            absorption_levels.append((price, delta))
        elif closed_bullish and delta <= -min_delta:
            absorption_levels.append((price, delta))

    return {"absorption_levels": absorption_levels}


def detect_enhanced_signals(bars: list[RangeBar], min_delta: int = 30) -> list[dict]:
    """Detect absorption signals with bar references."""
    signals = []

    for i in range(len(bars) - 1):
        signal_bar = bars[i]
        confirm_bar = bars[i + 1]

        if not signal_bar.closed or not confirm_bar.closed:
            continue

        signal_chars = analyze_bar_characteristics(signal_bar, min_delta)
        signal_bearish = signal_bar.close < signal_bar.open
        signal_bullish = signal_bar.close > signal_bar.open
        confirm_bearish = confirm_bar.close < confirm_bar.open
        confirm_bullish = confirm_bar.close > confirm_bar.open

        has_absorption = len(signal_chars["absorption_levels"]) > 0

        if signal_bearish and confirm_bearish and has_absorption:
            signals.append({
                "bar_index": i,
                "signal_bar_index": i,
                "confirm_bar_index": i + 1,
                "signal_bar": signal_bar,
                "confirm_bar": confirm_bar,
                "signal_type": "buyer_absorbed",
                "direction": "bearish",
                "confirm_time": confirm_bar.close_time,
                "absorption_levels": signal_chars["absorption_levels"]
            })

        if signal_bullish and confirm_bullish and has_absorption:
            signals.append({
                "bar_index": i,
                "signal_bar_index": i,
                "confirm_bar_index": i + 1,
                "signal_bar": signal_bar,
                "confirm_bar": confirm_bar,
                "signal_type": "seller_absorbed",
                "direction": "bullish",
                "confirm_time": confirm_bar.close_time,
                "absorption_levels": signal_chars["absorption_levels"]
            })

    return signals


def build_refreshing_profiles(session_ticks, current_date, refresh_minutes=1):
    """Build volume profiles that refresh every N minutes."""
    prev_day = current_date - timedelta(days=1)
    session_start_et = pd.Timestamp(f"{prev_day} 18:00:00", tz=ET)
    session_start_utc = session_start_et.tz_convert("UTC")

    profiles = {}
    session_ticks = session_ticks[session_ticks.index >= session_start_utc].copy()

    if session_ticks.empty:
        return profiles

    first_tick_time = session_ticks.index[0]
    last_tick_time = session_ticks.index[-1]

    refresh_delta = timedelta(minutes=refresh_minutes)
    current_window_start = first_tick_time

    while current_window_start <= last_tick_time:
        window_ticks = session_ticks[session_ticks.index < current_window_start].copy()

        if len(window_ticks) >= 100:
            profile = VolumeProfile.build(window_ticks, ticks_per_level=10)
            profiles[current_window_start] = profile

        current_window_start += refresh_delta

    return profiles


def generate_entry_signals(all_signals, profiles, proximity_ticks=40,
                          start_time_et="19:00", end_time_et="16:55"):
    """
    Generate entry signals near VAL/VAH with matching cluster bias.

    FULL SESSION: 7pm-4:55pm (no RTH filter)
    Wait 1hr after 6pm profile reset before trading.
    """
    entry_signals = []
    proximity_points = proximity_ticks * TICK_SIZE

    for signal in all_signals:
        # Filter to trading hours: 7pm-4:55pm (1hr wait after 6pm reset)
        confirm_time_et = signal["confirm_time"].tz_convert(ET)
        hour = confirm_time_et.hour
        minute = confirm_time_et.minute

        # Skip if before 7pm or after 4:55pm
        if hour < 19 and hour >= 17:  # 5pm-7pm, skip
            continue
        if hour == 16 and minute > 55:  # After 4:55pm, skip
            continue
        if hour == 17:  # 5pm hour, skip
            continue

        confirm_time = signal["confirm_time"]

        active_profile = None
        for profile_time, profile in sorted(profiles.items()):
            if profile_time >= confirm_time:
                active_profile = profile
                break

        if not active_profile:
            continue

        val_price = active_profile.val
        vah_price = active_profile.vah

        # Get profile range to clip cluster boundaries
        profile_low = min(active_profile.levels.keys()) if active_profile.levels else val_price
        profile_high = max(active_profile.levels.keys()) if active_profile.levels else vah_price

        near_val = False
        near_vah = False
        val_bias = None
        vah_bias = None

        if signal["signal_type"] == "seller_absorbed":
            signal_price_ref = signal["signal_bar"].high
        else:
            signal_price_ref = signal["signal_bar"].low

        if abs(signal_price_ref - val_price) <= proximity_points:
            near_val = True
            val_cluster_low = max(val_price - proximity_points, profile_low)
            val_cluster_high = val_price + proximity_points

            val_delta = sum(
                lv.delta for price, lv in active_profile.levels.items()
                if val_cluster_low <= price <= val_cluster_high
            )
            val_bias = "bullish" if val_delta >= 20 else ("bearish" if val_delta <= -20 else "neutral")

        if abs(signal_price_ref - vah_price) <= proximity_points:
            near_vah = True
            vah_cluster_low = vah_price - proximity_points
            vah_cluster_high = min(vah_price + proximity_points, profile_high)

            vah_delta = sum(
                lv.delta for price, lv in active_profile.levels.items()
                if vah_cluster_low <= price <= vah_cluster_high
            )
            vah_bias = "bullish" if vah_delta >= 20 else ("bearish" if vah_delta <= -20 else "neutral")

        entry_type = None
        if signal["signal_type"] == "seller_absorbed":
            if near_val and val_bias == "bullish":
                entry_type = "BUY"
            elif near_vah and vah_bias == "bullish":
                entry_type = "BUY"

        elif signal["signal_type"] == "buyer_absorbed":
            if near_val and val_bias == "bearish":
                entry_type = "SELL"
            elif near_vah and vah_bias == "bearish":
                entry_type = "SELL"

        if entry_type:
            entry_signals.append({
                **signal,
                "entry_type": entry_type
            })

    return entry_signals


def load_data_fast(start_date, end_date):
    """Load data from parquet or dbn."""
    data_dir = CACHE_DIR

    parquet_file = data_dir / "parquet" / f"NQ_c_0_mbp-1_{start_date}_{end_date}.parquet"
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        return normalize(df)

    dbn_file = data_dir / "dbn" / f"NQ_c_0_mbp-1_{start_date}_{end_date}.dbn"
    if dbn_file.exists():
        store = db.DBNStore.from_file(str(dbn_file))
        df = store.to_df()
        return normalize(df)

    return None


def process_and_cache_single_day(date_obj):
    """Process single day and cache bars, profiles, signals."""
    current_date = date_obj

    try:
        prev_date = current_date - timedelta(days=1)
        next_date = current_date + timedelta(days=1)

        ticks = load_data_fast(prev_date, next_date)
        if ticks is None:
            return {"date": current_date, "status": "no_data"}

        session_ticks = ticks[ticks["session_date"] == current_date].copy()
        if session_ticks.empty:
            return {"date": current_date, "status": "no_data"}

        # Build bars
        bars = build_range_bars(session_ticks, range_ticks=40, ticks_per_level=5)
        if len(bars) < 10:
            return {"date": current_date, "status": "insufficient_bars"}

        # Detect all signals
        all_signals = detect_enhanced_signals(bars, min_delta=30)

        # Build profiles
        profiles = build_refreshing_profiles(session_ticks, current_date, refresh_minutes=1)

        # Generate entry signals (7pm-4:55pm filter)
        entry_signals = generate_entry_signals(all_signals, profiles, proximity_ticks=100,
                                               start_time_et="19:00", end_time_et="16:55")

        # Cache everything
        cache_data = {
            "date": current_date,
            "bars": bars,
            "all_signals": all_signals,
            "profiles": profiles,
            "entry_signals": entry_signals,
            "num_bars": len(bars),
            "num_all_signals": len(all_signals),
            "num_entry_signals": len(entry_signals)
        }

        cache_file = SIGNAL_CACHE_DIR / f"{current_date}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        return {
            "date": current_date,
            "status": "success",
            "num_bars": len(bars),
            "num_entry_signals": len(entry_signals)
        }

    except Exception as e:
        return {"date": current_date, "status": "error", "error": str(e)}


if __name__ == '__main__':
    print("="*80)
    print("BUILDING FULL SESSION SIGNAL CACHE (7pm-4:55pm)")
    print("="*80)
    print()

    # Create cache directory
    SIGNAL_CACHE_DIR.mkdir(exist_ok=True, parents=True)

    # Date range
    start_date = datetime(2025, 3, 13).date()
    end_date = datetime(2026, 3, 30).date()

    all_dates = []
    current = start_date
    while current <= end_date:
        all_dates.append(current)
        current += timedelta(days=1)

    print(f"Processing {len(all_dates)} days from {start_date} to {end_date}")
    print(f"Cache directory: {SIGNAL_CACHE_DIR}")
    print()

    # Check existing cache
    existing_cache = list(SIGNAL_CACHE_DIR.glob("*.pkl"))
    if existing_cache:
        print(f"Found {len(existing_cache)} existing cached days")
        print("Keeping existing cache, will only process missing days")
        print()

    # Filter to days that need processing
    cached_dates = {datetime.strptime(f.stem, "%Y-%m-%d").date() for f in SIGNAL_CACHE_DIR.glob("*.pkl")}
    dates_to_process = [d for d in all_dates if d not in cached_dates]

    if not dates_to_process:
        print("All days already cached!")
        print()
    else:
        print(f"Processing {len(dates_to_process)} days...")
        print()

        num_cores = cpu_count()
        workers = max(1, min(8, int(num_cores * 0.5)))
        print(f"Using {workers} workers")
        print()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_and_cache_single_day, dates_to_process, chunksize=5))

        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]
        no_data = [r for r in results if r.get("status") == "no_data"]

        print()
        print("="*80)
        print("CACHE BUILD COMPLETE")
        print("="*80)
        print()
        print(f"Processed: {len(dates_to_process)} days")
        print(f"Successful: {len(successful)}")
        print(f"No data: {len(no_data)}")
        print(f"Failed: {len(failed)}")
        print()

        if successful:
            total_bars = sum(r["num_bars"] for r in successful)
            total_signals = sum(r["num_entry_signals"] for r in successful)
            print(f"Total bars cached: {total_bars:,}")
            print(f"Total entry signals cached: {total_signals:,}")
            print()

    # Summary
    all_cached = list(SIGNAL_CACHE_DIR.glob("*.pkl"))
    print(f"Cache directory contains {len(all_cached)} days")
    print(f"Total cache size: {sum(f.stat().st_size for f in all_cached) / 1024 / 1024:.1f} MB")
    print()
    print(f"Session: 7pm-4:55pm (1hr wait after 6pm profile reset)")
    print(f"Next: Update generate_breakeven_equity_curve.py to use this cache")
    print()

"""
Warm start: Load overnight session data before going live.

If starting at 9 AM, this fetches historical data from 6 PM (previous day)
to current time, builds bars, computes VWAP, and establishes bias,
then switches to live mode.

This gives you the SAME engine state as if you'd been running since 6 PM.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import databento as db

sys.path.insert(0, str(Path(__file__).parent.parent))

from nqbt.data.normalizer import normalize
from nqbt.analysis.range_bars import build_range_bars
from live.signal_engine import LiveSignalEngine
from live.bar_builder import TimeBar

log = logging.getLogger(__name__)

ET = "America/New_York"


def get_session_start_time() -> pd.Timestamp:
    """Get current session start time (6 PM ET previous day)."""
    now_et = pd.Timestamp.now(tz=ET)

    if now_et.hour < 18:
        session_date = now_et.date()
        prev_day = session_date - timedelta(days=1)
    else:
        prev_day = now_et.date()

    session_start = pd.Timestamp(f"{prev_day} 18:00:00", tz=ET)
    return session_start.tz_convert("UTC")


def fetch_session_history(api_key: str, session_start_utc: pd.Timestamp, current_time_utc: pd.Timestamp):
    """Fetch historical data from session start to current time."""
    safe_end_time = current_time_utc - timedelta(minutes=30)

    log.info(f"[warm_start] Fetching historical data from {session_start_utc} to {safe_end_time}")
    log.info(f"[warm_start] (Using 30-min delay to account for historical data lag)")

    client = db.Historical(key=api_key)

    start_str = session_start_utc.isoformat()
    end_str = safe_end_time.isoformat()

    try:
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=["NQ.c.0"],
            schema="mbp-1",
            stype_in="continuous",
            start=start_str,
            end=end_str,
        )

        df = data.to_df()

        if df.empty:
            log.warning("[warm_start] No historical data returned")
            return None

        df = df[(df.index >= session_start_utc) & (df.index <= current_time_utc)]
        ticks = normalize(df)

        log.info(f"[warm_start] Fetched {len(ticks):,} ticks from history")
        return ticks

    except Exception as e:
        log.error(f"[warm_start] Failed to fetch historical data: {e}")
        return None


def build_time_bars(ticks: pd.DataFrame, interval_minutes: int = 5) -> list[TimeBar]:
    """Build 5-minute time bars from tick data."""
    if ticks.empty:
        return []

    time_bars = []
    interval = pd.Timedelta(minutes=interval_minutes)

    ticks = ticks.sort_index()
    ticks['bar_start'] = ticks.index.floor(interval)

    for bar_start, group in ticks.groupby('bar_start'):
        bar_end = bar_start + interval

        time_bar = TimeBar(
            open_time=bar_start,
            close_time=bar_end,
            open=group.iloc[0]['price'],
            high=group['price'].max(),
            low=group['price'].min(),
            close=group.iloc[-1]['price'],
            buy_vol=group[group['aggressor'] == 'buy']['size'].sum(),
            sell_vol=group[group['aggressor'] == 'sell']['size'].sum(),
            tick_count=len(group),
            closed=True
        )

        time_bars.append(time_bar)

    return time_bars


def warm_start_engine(engine: LiveSignalEngine, api_key: str) -> bool:
    """
    Warm start the signal engine with overnight session data.

    Loads historical ticks, computes VWAP incrementally, builds time bars
    for ATR + bias, and builds range bars for entry signal detection.
    """
    now_utc = pd.Timestamp.now(tz="UTC")
    now_et = now_utc.tz_convert(ET)
    session_start_utc = get_session_start_time()
    session_start_et = session_start_utc.tz_convert(ET)

    hours_since_start = (now_utc - session_start_utc).total_seconds() / 3600

    log.info("=" * 80)
    log.info("WARM START - VWAP Reaction Strategy")
    log.info("=" * 80)
    log.info(f"Current time:     {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    log.info(f"Session start:    {session_start_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    log.info(f"Hours since open: {hours_since_start:.1f} hours")
    log.info("")

    if hours_since_start < 1:
        log.info("[warm_start] Session just started, skipping warm start")
        return True

    # Fetch historical data
    ticks = fetch_session_history(api_key, session_start_utc, now_utc)

    if ticks is None or ticks.empty:
        log.warning("[warm_start] No historical data available, starting cold")
        return False

    # Initialize session in engine
    engine._check_session_reset(session_start_utc)

    # Feed all ticks to engine for VWAP + std dev computation
    log.info("[warm_start] Computing session VWAP + std dev bands from historical ticks...")
    for _, tick in ticks.iterrows():
        p, s = tick['price'], tick['size']
        engine._cum_pv += p * s
        engine._cum_vol += s
        engine._cum_pv2 += p * p * s

    vwap = engine.vwap
    std1_upper = engine.std1_upper
    std1_lower = engine.std1_lower
    if vwap is not None:
        band_str = f" | STD1: {std1_lower:.2f}/{std1_upper:.2f}" if std1_upper is not None else ""
        log.info(f"[warm_start] VWAP computed: {vwap:.2f}{band_str} (from {len(ticks):,} ticks)")
    else:
        log.warning("[warm_start] VWAP computation failed")

    # Build TIME BARS for ATR + bias
    log.info("[warm_start] Building 5-minute time bars for ATR + bias...")
    time_bars = build_time_bars(ticks, interval_minutes=5)

    if len(time_bars) < engine.atr_period + 1:
        log.warning(f"[warm_start] Only {len(time_bars)} time bars, need {engine.atr_period + 1}+ for ATR")

    log.info(f"[warm_start] Built {len(time_bars)} time bars")

    # Feed time bars to engine for ATR + bias tracking
    for time_bar in time_bars:
        engine.on_time_bar_close(time_bar)

    # Build RANGE BARS for entry signal history
    log.info("[warm_start] Building range bars from historical ticks...")
    bars = build_range_bars(ticks, range_ticks=40, ticks_per_level=5)

    log.info(f"[warm_start] Built {len(bars)} range bars")

    # Feed range bars to engine (populates bar buffer for approach direction checks).
    # Skip the last bar if it's unclosed (close_time=None); live ticks will close it.
    for bar in bars:
        if bar.close_time is None:
            continue
        engine.on_bar_close(bar)

    # Clear any phantom position from historical replay - live must start flat.
    engine._position = None

    # Status report
    atr = engine._calculate_atr()
    adx = engine.adx
    warm_start_ok = True

    log.info("=" * 80)
    if vwap is not None:
        log.info(f"  VWAP:  {vwap:.2f}")
    else:
        log.info(f"  VWAP:  NOT READY")
        warm_start_ok = False

    if atr is not None:
        log.info(f"  ATR:   {atr:.2f} (from {len(time_bars)} time bars)")
    else:
        log.info(f"  ATR:   NOT READY ({len(time_bars)}/{engine.atr_period + 1} time bars)")
        warm_start_ok = False

    if adx is not None:
        base_ok = "YES" if 15.0 <= adx < 30.0 else "NO"
        trend_ok = "YES" if adx >= 30.0 else "NO"
        log.info(f"  ADX:   {adx:.1f} (base 15-30: {base_ok} | trend 30+: {trend_ok})")
    else:
        log.info(f"  ADX:   NOT READY ({engine._adx_tr_count} TR values, need ~{14*2} for full warmup)")

    log.info(f"  Bias:  {engine._current_bias or 'NONE'}")
    log.info(f"  Trend: {engine._trend_direction or 'NONE'} (need {14} consecutive 5m bars)")
    if engine.std1_upper is not None:
        log.info(f"  STD1:  {engine.std1_lower:.2f} / {engine.std1_upper:.2f}")
    log.info(f"  Bars:  {len(engine._bars)} range bars in buffer")

    if warm_start_ok:
        log.info("WARM START COMPLETE - Combined strategy ready (base + trending band)")
    else:
        log.info("WARM START INCOMPLETE - Some indicators not ready")
    log.info("=" * 80)
    log.info("")

    return warm_start_ok


if __name__ == "__main__":
    """Test warm start independently."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("DATABENTO_API_KEY not set in .env")
        sys.exit(1)

    engine = LiveSignalEngine(atr_period=14, stop_mult=0.50, target_mult=1.90,
                              trend_stop_mult=1.00, trend_target_mult=1.00)
    success = warm_start_engine(engine, api_key)

    if success:
        print("\nWarm start successful! Engine is ready for live trading.")
    else:
        print("\nWarm start incomplete. Check logs above.")

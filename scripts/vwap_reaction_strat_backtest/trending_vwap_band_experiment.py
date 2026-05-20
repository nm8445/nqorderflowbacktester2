"""
Trending VWAP Band Experiment

When price is trending (N consecutive 5-min closes on one side of VWAP),
look for absorption signals at std dev bands instead of VWAP itself.

Rules:
  - Trending UP: 5-min close above VWAP for N bars -> look for LONG at std1/2 upper band
    (band = dynamic support during uptrend pullback)
    Entry zone: band +/- 3 pts. Price must pull back from above into the zone.
    If 5-min closes below zone -> pause longs until closes back above zone.

  - Trending DOWN: 5-min close below VWAP for N bars -> look for SHORT at std1/2 lower band
    (band = dynamic resistance during downtrend push-up)
    Entry zone: band +/- 3 pts. Price must push up from below into the zone.
    If 5-min closes above zone -> pause shorts until closes back below zone.

  - Same absorption + confirmation logic as base VWAP strat
  - ATR-based SL/TP (grid sweep)
  - Tests std dev 1 and std dev 2 bands

Usage:
    python scripts/vwap_reaction_strat_backtest/trending_vwap_band_experiment.py
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import build_adx_lookup, get_adx_at_time

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_5MIN_DIR = DATA_DIR / "timebars_5min"

ZONE_POINTS = 3.0
# Asymmetric band zone
BAND_ZONE_ABOVE = 3.0  # outside (away from VWAP)
BAND_ZONE_BELOW = 4.0  # inside  (toward VWAP)
ATR_PERIOD = 14
MIN_DELTA = 30

TICK_SIZE = 0.25
TICK_VALUE = 5.0
POINT_VALUE = TICK_VALUE / TICK_SIZE  # $20 per point

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"

START_DATE = "2025-03-13"
END_DATE = "2026-04-08"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_5min_bars(target_date: str):
    bar_file = TIMEBARS_5MIN_DIR / f"timebars_5min_{target_date.replace('-','_')}.pkl"
    if not bar_file.exists():
        return None
    bars = load_pickle(bar_file)
    if not bars:
        return None
    rows = [{'timestamp': b['open_time'], 'high': b['high'], 'low': b['low'], 'close': b['close']} for b in bars]
    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['prev_close']),
                                     abs(df['low'] - df['prev_close'])))
    df['atr'] = df['tr'].rolling(ATR_PERIOD, min_periods=1).mean()
    return df


def has_absorption(bar, min_delta: int = MIN_DELTA):
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
    }


def precompute_day_state(bars_5min, vwap_df, lookback: int, std_band: int):
    """
    Precompute per-5min-bar: trending state, band values, ATR.
    Returns a DataFrame indexed by 5-min bar timestamp with columns:
      trend ('up', 'down', or None), band_upper, band_lower, atr, close, vwap
    """
    if bars_5min is None or vwap_df is None or bars_5min.empty or vwap_df.empty:
        return None

    # Vectorized VWAP lookup: for each 5-min bar, get the VWAP at that time
    # Use merge_asof for speed
    vwap_series = vwap_df[['vwap', f'std{std_band}_upper', f'std{std_band}_lower']].copy()
    vwap_series = vwap_series.sort_index()

    bars_sorted = bars_5min.sort_index()

    # Align timezones to UTC for merge_asof
    if bars_sorted.index.tz is not None:
        bars_sorted.index = bars_sorted.index.tz_convert('UTC')
    else:
        bars_sorted.index = bars_sorted.index.tz_localize('UTC')
    if vwap_series.index.tz is not None:
        vwap_series.index = vwap_series.index.tz_convert('UTC')
    else:
        vwap_series.index = vwap_series.index.tz_localize('UTC')

    merged = pd.merge_asof(
        bars_sorted[['close', 'atr']],
        vwap_series,
        left_index=True, right_index=True,
        direction='backward',
    )

    if merged.empty:
        return None

    # Compute above/below VWAP per bar
    merged['above_vwap'] = merged['close'] > merged['vwap']
    merged['below_vwap'] = merged['close'] < merged['vwap']

    # Rolling consecutive count
    # For "all N bars above": rolling min of above_vwap (as int) over lookback
    merged['above_int'] = merged['above_vwap'].astype(int)
    merged['below_int'] = merged['below_vwap'].astype(int)

    merged['all_above'] = merged['above_int'].rolling(lookback, min_periods=lookback).min() == 1
    merged['all_below'] = merged['below_int'].rolling(lookback, min_periods=lookback).min() == 1

    merged['trend'] = None
    merged.loc[merged['all_above'], 'trend'] = 'up'
    merged.loc[merged['all_below'], 'trend'] = 'down'

    merged = merged.rename(columns={
        f'std{std_band}_upper': 'band_upper',
        f'std{std_band}_lower': 'band_lower',
    })

    # Option B approach latches: per trend run, has any 5-min close reached the
    # active band? Once latched True for a run, stays True for the rest of that run.
    trend_run_id = (merged['trend'] != merged['trend'].shift()).cumsum()
    hit_upper = (merged['trend'] == 'up')   & (merged['close'] >= merged['band_upper'])
    hit_lower = (merged['trend'] == 'down') & (merged['close'] <= merged['band_lower'])
    merged['approach_from_above'] = hit_upper.groupby(trend_run_id).cummax().astype(bool)
    merged['approach_from_below'] = hit_lower.groupby(trend_run_id).cummax().astype(bool)
    merged.loc[merged['trend'].isna(), ['approach_from_above', 'approach_from_below']] = False

    return merged[['close', 'atr', 'vwap', 'band_upper', 'band_lower', 'trend',
                   'approach_from_above', 'approach_from_below']]


def detect_trending_signals(bars, day_state):
    """
    Detect absorption signals at std dev bands when market is trending.
    day_state is precomputed DataFrame from precompute_day_state().
    """
    if day_state is None or day_state.empty:
        return []

    signals = []

    # Build a sorted index for fast lookups
    state_index = day_state.index
    state_values = day_state.values  # [close, atr, vwap, band_upper, band_lower, trend,
                                     #  approach_from_above, approach_from_below]
    col_atr = 1
    col_vwap = 2
    col_upper = 3
    col_lower = 4
    col_trend = 5
    col_close = 0
    col_app_above = 6
    col_app_below = 7

    def lookup_state(ts):
        """Get the most recent state at or before ts."""
        idx = state_index.searchsorted(ts, side='right') - 1
        if idx < 0:
            return None
        return state_values[idx]

    for i in range(len(bars) - 1):
        signal_bar = bars[i]
        confirm_bar = bars[i + 1]

        if not signal_bar.closed or not confirm_bar.closed:
            continue

        bar_time = signal_bar.close_time

        # Time filter
        if hasattr(bar_time, 'tz_convert'):
            bar_time_et = bar_time.tz_convert(ET)
        else:
            bar_time_et = pd.Timestamp(bar_time, tz='UTC').tz_convert(ET)
        hour_min = bar_time_et.strftime("%H:%M")
        if "17:00" <= hour_min < "19:00":
            continue

        # Check absorption
        absorption = has_absorption(signal_bar)
        if not absorption["bearish"] and not absorption["bullish"]:
            continue

        # Look up precomputed state
        state = lookup_state(bar_time)
        if state is None:
            continue

        trend = state[col_trend]
        if trend is None:
            continue

        atr = state[col_atr]
        vwap = state[col_vwap]
        band_upper = state[col_upper]
        band_lower = state[col_lower]
        last_5min_close = state[col_close]

        if atr is None or np.isnan(atr) or atr <= 0:
            continue

        if trend == "up":
            # Uptrend: look for LONG at upper band (pullback support)
            band_value = band_upper
            # Upper band: 3 pts above (outside), 4 pts below (inside/toward VWAP)
            zone_high = band_value + BAND_ZONE_ABOVE
            zone_low = band_value - BAND_ZONE_BELOW

            # Signal bar must touch the band zone
            if not (signal_bar.high >= zone_low and signal_bar.low <= zone_high):
                continue

            # Confirm bar must also touch zone (use confirm time's band)
            c_state = lookup_state(confirm_bar.close_time)
            c_band = c_state[col_upper] if c_state is not None else band_value
            c_zone_high = c_band + BAND_ZONE_ABOVE
            c_zone_low = c_band - BAND_ZONE_BELOW
            if not (confirm_bar.high >= c_zone_low and confirm_bar.low <= c_zone_high):
                continue

            # Bullish absorption + bullish confirm
            if not absorption["bullish"]:
                continue
            if confirm_bar.close <= confirm_bar.open:
                continue

            # Band bias: last 5-min close must be >= zone_low (not broken below)
            if last_5min_close < zone_low:
                continue

            # Option B approach gate: 5-min bar must have closed at/above band in this trend run
            if not bool(state[col_app_above]):
                continue

            signals.append({
                "bar_index": i,
                "direction": "long",
                "signal_time": bar_time,
                "confirm_time": confirm_bar.close_time,
                "entry_price": confirm_bar.close,
                "atr": float(atr),
                "vwap": float(vwap),
                "band_value": float(band_value),
                "trend": trend,
                "signal_bar_high": signal_bar.high,
                "signal_bar_low": signal_bar.low,
            })

        elif trend == "down":
            # Downtrend: look for SHORT at lower band (push-up resistance)
            band_value = band_lower
            # Lower band: 3 pts below (outside), 4 pts above (inside/toward VWAP)
            zone_high = band_value + BAND_ZONE_BELOW
            zone_low = band_value - BAND_ZONE_ABOVE

            if not (signal_bar.high >= zone_low and signal_bar.low <= zone_high):
                continue

            c_state = lookup_state(confirm_bar.close_time)
            c_band = c_state[col_lower] if c_state is not None else band_value
            c_zone_high = c_band + BAND_ZONE_BELOW
            c_zone_low = c_band - BAND_ZONE_ABOVE
            if not (confirm_bar.high >= c_zone_low and confirm_bar.low <= c_zone_high):
                continue

            if not absorption["bearish"]:
                continue
            if confirm_bar.close >= confirm_bar.open:
                continue

            # Band bias: last 5-min close must be <= zone_high (not broken above)
            if last_5min_close > zone_high:
                continue

            # Option B approach gate: 5-min bar must have closed at/below band in this trend run
            if not bool(state[col_app_below]):
                continue

            signals.append({
                "bar_index": i,
                "direction": "short",
                "signal_time": bar_time,
                "confirm_time": confirm_bar.close_time,
                "entry_price": confirm_bar.close,
                "atr": float(atr),
                "vwap": float(vwap),
                "band_value": float(band_value),
                "trend": trend,
                "signal_bar_high": signal_bar.high,
                "signal_bar_low": signal_bar.low,
            })

    return signals


def simulate_trades(signals, bars, sl_mult, tp_mult):
    """Walk bars forward to simulate SL/TP exits."""
    trades = []
    last_exit_time = None

    for signal in signals:
        entry_price = signal["entry_price"]
        direction = signal["direction"]
        atr = signal["atr"]
        confirm_bar_idx = signal["bar_index"] + 1
        confirm_time = signal["confirm_time"]

        if hasattr(confirm_time, 'tz_convert'):
            confirm_et = confirm_time.tz_convert(ET)
        else:
            confirm_et = pd.Timestamp(confirm_time, tz='UTC').tz_convert(ET)
        if "16:00" <= confirm_et.strftime("%H:%M") < "19:10":
            continue

        if last_exit_time is not None and confirm_time <= last_exit_time:
            continue

        if direction == "long":
            sl_price = entry_price - atr * sl_mult
            tp_price = entry_price + atr * tp_mult
        else:
            sl_price = entry_price + atr * sl_mult
            tp_price = entry_price - atr * tp_mult

        exit_price = None
        exit_time = None
        exit_reason = None

        for j in range(confirm_bar_idx + 1, len(bars)):
            bar = bars[j]
            if not bar.closed:
                continue

            bar_ct = bar.close_time
            if hasattr(bar_ct, 'tz_convert'):
                bar_et = bar_ct.tz_convert(ET)
            else:
                bar_et = pd.Timestamp(bar_ct, tz='UTC').tz_convert(ET)
            if bar_et.strftime("%H:%M") >= FORCE_CLOSE:
                exit_price = bar.close
                exit_time = bar.close_time
                exit_reason = "session_close"
                break

            if direction == "long":
                if bar.low <= sl_price:
                    exit_price = sl_price
                    exit_time = bar.close_time
                    exit_reason = "stop_loss"
                    break
                if bar.high >= tp_price:
                    exit_price = tp_price
                    exit_time = bar.close_time
                    exit_reason = "target"
                    break
            else:
                if bar.high >= sl_price:
                    exit_price = sl_price
                    exit_time = bar.close_time
                    exit_reason = "stop_loss"
                    break
                if bar.low <= tp_price:
                    exit_price = tp_price
                    exit_time = bar.close_time
                    exit_reason = "target"
                    break

        if exit_price is None:
            last_bar = bars[-1]
            exit_price = last_bar.close
            exit_time = last_bar.close_time
            exit_reason = "eod"

        if direction == "long":
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price

        trades.append({
            "date": signal.get("date", ""),
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_points": pnl_points,
            "pnl_dollars": pnl_points * POINT_VALUE,
            "atr": atr,
            "vwap": signal["vwap"],
            "band_value": signal["band_value"],
            "trend": signal["trend"],
            "confirm_time": confirm_time,
        })
        last_exit_time = exit_time

    return trades


def run_experiment(std_band: int, trending_lookback: int, sl_mult: float, tp_mult: float,
                   adx_lookup=None, adx_min: float = 30.0,
                   start_override: str = None, end_override: str = None):
    """Run full experiment for one config. Only takes signals where ADX >= adx_min."""
    start = datetime.strptime(start_override or START_DATE, "%Y-%m-%d").date()
    end = datetime.strptime(end_override or END_DATE, "%Y-%m-%d").date()

    all_trades = []
    total_signals = 0
    adx_filtered = 0
    days_with_signals = 0

    for vwap_file in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        date_str = vwap_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue

        signal_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        tb_file = TIMEBARS_5MIN_DIR / f"timebars_5min_{date_str.replace('-','_')}.pkl"
        if not signal_file.exists() or not tb_file.exists():
            continue

        vwap_df = load_pickle(vwap_file)
        signal_data = load_pickle(signal_file)
        bars = signal_data["bars"]
        bars_5min = load_5min_bars(date_str)

        if not bars or bars_5min is None:
            continue

        # Precompute trending state for the day (fast)
        day_state = precompute_day_state(bars_5min, vwap_df, trending_lookback, std_band)

        signals = detect_trending_signals(bars, day_state)
        for s in signals:
            s["date"] = date_str

        total_signals += len(signals)

        # ADX filter: only keep signals where ADX >= adx_min
        if adx_lookup is not None:
            filtered = []
            for s in signals:
                ct = s["confirm_time"]
                if hasattr(ct, 'tz_convert'):
                    ct_et = ct.tz_convert("America/New_York")
                else:
                    ct_et = pd.Timestamp(ct, tz='UTC').tz_convert("America/New_York")
                adx_val = get_adx_at_time(ct_et, adx_lookup)
                if not np.isnan(adx_val) and adx_val >= adx_min:
                    s["adx"] = adx_val
                    filtered.append(s)
            adx_filtered += len(signals) - len(filtered)
            signals = filtered

        if signals:
            days_with_signals += 1

        day_trades = simulate_trades(signals, bars, sl_mult, tp_mult)
        all_trades.extend(day_trades)

    return all_trades, total_signals, days_with_signals, adx_filtered


def print_results(trades, label, total_signals, days_with_signals, adx_filtered=0):
    if not trades:
        print(f"\n  {label}")
        print(f"  -> No trades ({total_signals} raw signals, {adx_filtered} ADX-filtered, {days_with_signals} days)")
        return

    df = pd.DataFrame(trades)
    total = len(df)
    winners = df[df["pnl_points"] > 0]
    losers = df[df["pnl_points"] < 0]

    win_rate = len(winners) / total * 100
    avg_win = winners["pnl_points"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl_points"].mean() if len(losers) > 0 else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    gross_profit = winners["pnl_points"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_points"].sum()) if len(losers) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    cumulative = df["pnl_points"].cumsum()
    max_dd = (cumulative - cumulative.cummax()).min()
    total_pnl = df["pnl_points"].sum()

    shorts = df[df["direction"] == "short"]
    longs = df[df["direction"] == "long"]
    sl_exits = df[df["exit_reason"] == "stop_loss"]
    tp_exits = df[df["exit_reason"] == "target"]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Raw signals: {total_signals} ({adx_filtered} ADX-filtered) on {days_with_signals} days")
    print(f"  Trades:      {total} | W: {len(winners)} ({win_rate:.1f}%) | L: {len(losers)}")
    print(f"  Avg win:     {avg_win:+.2f} pts | Avg loss: {avg_loss:+.2f} pts")
    print(f"  Expectancy:  {expectancy:+.2f} pts/trade (${expectancy * POINT_VALUE:+,.0f})")
    print(f"  PF:          {profit_factor:.2f}")
    print(f"  Total P&L:   {total_pnl:+.2f} pts (${total_pnl * POINT_VALUE:+,.0f})")
    print(f"  Max DD:      {max_dd:.2f} pts (${max_dd * POINT_VALUE:+,.0f})")

    if len(shorts) > 0:
        s_wr = len(shorts[shorts['pnl_points'] > 0]) / len(shorts) * 100
        print(f"  Shorts:      {len(shorts)} | {shorts['pnl_points'].sum():+.2f} pts | WR: {s_wr:.1f}%")
    if len(longs) > 0:
        l_wr = len(longs[longs['pnl_points'] > 0]) / len(longs) * 100
        print(f"  Longs:       {len(longs)} | {longs['pnl_points'].sum():+.2f} pts | WR: {l_wr:.1f}%")

    print(f"  Exits:       Target={len(tp_exits)} SL={len(sl_exits)}")


def main():
    print("=" * 70)
    print("  TRENDING VWAP BAND EXPERIMENT — ADX >= 30 — IN/OUT OF SAMPLE")
    print("  Filling gaps where base VWAP strat (ADX 15-30) doesn't trade")
    print(f"  Full period: {START_DATE} to {END_DATE}")
    print(f"  In-sample:   {START_DATE} to 2025-09-25")
    print(f"  Out-of-sample: 2025-09-26 to {END_DATE}")
    print("=" * 70)
    sys.stdout.flush()

    # Pre-build ADX lookup once (shared across all configs)
    print("Building ADX lookup table...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done ({len(adx_lookup)} rows)")
    sys.stdout.flush()

    IS_END = "2025-09-25"
    OOS_START = "2025-09-26"

    sl_tp_configs = [
        (0.50, 1.50),
        (0.50, 2.10),
        (0.75, 2.00),
        (1.00, 2.00),
    ]

    lookbacks = [10, 14, 20]
    bands = [1, 2]

    for std_band in bands:
        for lookback in lookbacks:
            print(f"\n{'#'*70}")
            print(f"  STD DEV {std_band} | TRENDING LOOKBACK = {lookback} bars | ADX >= 30")
            print(f"{'#'*70}")
            sys.stdout.flush()

            for sl_mult, tp_mult in sl_tp_configs:
                # In-sample
                is_label = f"[IS] STD{std_band} | LB={lookback} | SL {sl_mult}x / TP {tp_mult}x"
                is_trades, is_signals, is_days, is_filt = run_experiment(
                    std_band=std_band,
                    trending_lookback=lookback,
                    sl_mult=sl_mult,
                    tp_mult=tp_mult,
                    adx_lookup=adx_lookup,
                    adx_min=30.0,
                    start_override=START_DATE,
                    end_override=IS_END,
                )
                print_results(is_trades, is_label, is_signals, is_days, is_filt)

                # Out-of-sample
                oos_label = f"[OOS] STD{std_band} | LB={lookback} | SL {sl_mult}x / TP {tp_mult}x"
                oos_trades, oos_signals, oos_days, oos_filt = run_experiment(
                    std_band=std_band,
                    trending_lookback=lookback,
                    sl_mult=sl_mult,
                    tp_mult=tp_mult,
                    adx_lookup=adx_lookup,
                    adx_min=30.0,
                    start_override=OOS_START,
                    end_override=END_DATE,
                )
                print_results(oos_trades, oos_label, oos_signals, oos_days, oos_filt)
                sys.stdout.flush()

    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

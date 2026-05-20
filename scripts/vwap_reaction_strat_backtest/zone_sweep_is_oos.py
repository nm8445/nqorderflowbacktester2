"""
Band zone sweep — IS/OOS test.

Sweeps the trending strategy's ZONE_POINTS parameter from 3 to 10 in steps of 1.
Base VWAP strat keeps ZONE_POINTS=3.
Runs combined strategy (base + trending, one-trade-at-a-time lock) per zone.

Splits data 50/50 IS vs OOS:
  IS:  2025-03-13 - 2025-09-25
  OOS: 2025-09-26 - 2026-04-13

Reports per-zone metrics for IS and OOS separately so you can see robustness.

Usage:
  python scripts/vwap_reaction_strat_backtest/zone_sweep_is_oos.py
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time, passes_adx_filter,
    LUNCH_START, LUNCH_END,
)
from trending_vwap_atr_grid import (
    load_5min_bars as load_5min_bars_trending,
    has_absorption,
)

VWAP_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_PRICE_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_DIR = DATA_DIR / "timebars_5min"

POINT_VALUE = 20.0
ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"

BASE_SL = 0.50
BASE_TP = 1.90
BASE_ZONE = 3.0  # base strat zone stays fixed

TREND_SL = 1.00
TREND_TP = 1.00
TREND_STD = 1
TREND_LB = 14
TREND_ADX_MIN = 30.0
MIN_DELTA = 30
ATR_PERIOD = 14

START_DATE = "2025-03-13"
END_DATE = "2026-04-13"
IS_END = "2025-09-25"
OOS_START = "2025-09-26"


def precompute_trend_state(bars_5min, vwap_df, lookback, std_band):
    """Build merged 5-min state with trend direction."""
    if bars_5min is None or vwap_df is None or bars_5min.empty or vwap_df.empty:
        return None
    vwap_cols = ['vwap', f'std{std_band}_upper', f'std{std_band}_lower']
    vwap_series = vwap_df[vwap_cols].copy().sort_index()
    bs = bars_5min[['close', 'atr']].sort_index()
    if bs.index.tz is not None:
        bs.index = bs.index.tz_convert('UTC')
    else:
        bs.index = bs.index.tz_localize('UTC')
    if vwap_series.index.tz is not None:
        vwap_series.index = vwap_series.index.tz_convert('UTC')
    else:
        vwap_series.index = vwap_series.index.tz_localize('UTC')
    merged = pd.merge_asof(bs, vwap_series, left_index=True, right_index=True, direction='backward')
    if merged.empty:
        return None
    merged['above_int'] = (merged['close'] > merged['vwap']).astype(int)
    merged['below_int'] = (merged['close'] < merged['vwap']).astype(int)
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
    # active band? A "trend run" = contiguous block of same trend direction.
    # Once latched True for a run, stays True for the rest of that run.
    trend_run_id = (merged['trend'] != merged['trend'].shift()).cumsum()
    hit_upper = (merged['trend'] == 'up')   & (merged['close'] >= merged['band_upper'])
    hit_lower = (merged['trend'] == 'down') & (merged['close'] <= merged['band_lower'])
    merged['approach_from_above'] = hit_upper.groupby(trend_run_id).cummax().astype(bool)
    merged['approach_from_below'] = hit_lower.groupby(trend_run_id).cummax().astype(bool)
    # Outside of a trend run, flags are False
    merged.loc[merged['trend'].isna(), ['approach_from_above', 'approach_from_below']] = False

    return merged[['close', 'atr', 'vwap', 'band_upper', 'band_lower', 'trend',
                   'approach_from_above', 'approach_from_below']]


def detect_trending_signals_zone(bars, day_state, zone_points):
    """Trending signals with configurable zone."""
    if day_state is None or day_state.empty:
        return []
    signals = []
    state_index = day_state.index
    state_values = day_state.values

    def lookup(ts):
        idx = state_index.searchsorted(ts, side='right') - 1
        return state_values[idx] if idx >= 0 else None

    for i in range(len(bars) - 1):
        sb = bars[i]
        cb = bars[i + 1]
        if not sb.closed or not cb.closed:
            continue
        bt = sb.close_time
        if hasattr(bt, 'tz_convert'):
            bt_et = bt.tz_convert(ET)
        else:
            bt_et = pd.Timestamp(bt, tz='UTC').tz_convert(ET)
        hm = bt_et.strftime("%H:%M")
        if "17:00" <= hm < "19:00":
            continue

        is_bull, is_bear = has_absorption(sb)
        if not is_bull and not is_bear:
            continue

        state = lookup(bt)
        if state is None:
            continue
        trend = state[5]
        if trend is None:
            continue
        atr = state[1]
        if atr is None or np.isnan(atr) or atr <= 0:
            continue
        vwap = state[2]
        last_5m_close = state[0]

        approach_above = bool(state[6])
        approach_below = bool(state[7])

        if trend == "up":
            band = state[3]
            zh, zl = band + zone_points, band - zone_points
            if not (sb.high >= zl and sb.low <= zh):
                continue
            cs = lookup(cb.close_time)
            cb_val = cs[3] if cs is not None else band
            if not (cb.high >= cb_val - zone_points and cb.low <= cb_val + zone_points):
                continue
            if not is_bull or cb.close <= cb.open:
                continue
            if last_5m_close < zl:  # band bias paused
                continue
            # Option B approach gate: 5-min bar must have closed at/above band in this trend run
            if not approach_above:
                continue
            signals.append({
                "source": "trend", "bar_index": i, "direction": "long",
                "confirm_time": cb.close_time, "entry_price": cb.close,
                "atr": float(atr),
            })

        elif trend == "down":
            band = state[4]
            zh, zl = band + zone_points, band - zone_points
            if not (sb.high >= zl and sb.low <= zh):
                continue
            cs = lookup(cb.close_time)
            cb_val = cs[4] if cs is not None else band
            if not (cb.high >= cb_val - zone_points and cb.low <= cb_val + zone_points):
                continue
            if not is_bear or cb.close >= cb.open:
                continue
            if last_5m_close > zh:
                continue
            # Option B approach gate: 5-min bar must have closed at/below band in this trend run
            if not approach_below:
                continue
            signals.append({
                "source": "trend", "bar_index": i, "direction": "short",
                "confirm_time": cb.close_time, "entry_price": cb.close,
                "atr": float(atr),
            })

    return signals


def prepare_base_signals(base_signals, adx_lookup):
    """Filter base signals through ADX 15-30, time windows. Returns list with SL/TP set."""
    out = []
    for s in base_signals:
        ct = s["confirm_time"]
        if hasattr(ct, "tz_convert"):
            ct_et = ct.tz_convert(ET)
        else:
            ct_et = pd.Timestamp(ct, tz="UTC").tz_convert(ET)
        hm = ct_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10":
            continue
        if LUNCH_START <= hm < LUNCH_END:
            continue
        atr = s["atr"]
        if atr is None or atr <= 0:
            continue
        adx = get_adx_at_time(ct_et, adx_lookup)
        if not passes_adx_filter(adx):
            continue
        ep = s["entry_price"]
        d = s["direction"]
        if d == "long":
            sl = ep - atr * BASE_SL
            tp = ep + atr * BASE_TP
            if tp <= ep or sl >= ep:
                continue
        else:
            sl = ep + atr * BASE_SL
            tp = ep - atr * BASE_TP
            if tp >= ep or sl <= ep:
                continue
        out.append({
            "source": "base", "bar_index": s["bar_index"],
            "confirm_time": ct, "entry_price": ep, "direction": d,
            "sl": sl, "tp": tp, "atr": atr,
        })
    return out


def prepare_trend_signals(trend_signals, adx_lookup):
    """Filter trend signals through ADX 30+, time windows. Add SL/TP."""
    out = []
    for s in trend_signals:
        ct = s["confirm_time"]
        if hasattr(ct, "tz_convert"):
            ct_et = ct.tz_convert(ET)
        else:
            ct_et = pd.Timestamp(ct, tz="UTC").tz_convert(ET)
        hm = ct_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10":
            continue
        if LUNCH_START <= hm < LUNCH_END:
            continue
        atr = s["atr"]
        if atr is None or atr <= 0:
            continue
        adx = get_adx_at_time(ct_et, adx_lookup)
        if np.isnan(adx) or adx < TREND_ADX_MIN:
            continue
        ep = s["entry_price"]
        d = s["direction"]
        if d == "long":
            sl = ep - atr * TREND_SL
            tp = ep + atr * TREND_TP
        else:
            sl = ep + atr * TREND_SL
            tp = ep - atr * TREND_TP
        out.append({
            "source": "trend", "bar_index": s["bar_index"],
            "confirm_time": ct, "entry_price": ep, "direction": d,
            "sl": sl, "tp": tp, "atr": atr,
        })
    return out


def simulate_day_combined(bars, base_sigs, trend_sigs):
    """Merge, sort, simulate with position lock."""
    all_sigs = base_sigs + trend_sigs
    all_sigs.sort(key=lambda x: x["confirm_time"])
    trades = []
    last_exit = None
    for sig in all_sigs:
        if last_exit is not None and sig["confirm_time"] <= last_exit:
            continue
        d = sig["direction"]
        ep = sig["entry_price"]
        sl = sig["sl"]
        tp = sig["tp"]
        cbi = sig["bar_index"] + 1
        exit_price = None
        exit_time = None
        exit_reason = None
        for j in range(cbi + 1, len(bars)):
            bar = bars[j]
            if not bar.closed:
                continue
            bct = bar.close_time
            if hasattr(bct, "tz_convert"):
                bet = bct.tz_convert(ET)
            else:
                bet = pd.Timestamp(bct, tz="UTC").tz_convert(ET)
            if bet.strftime("%H:%M") >= FORCE_CLOSE:
                exit_price = bar.close; exit_time = bct; exit_reason = "eod"; break
            if d == "long":
                if bar.low <= sl: exit_price = sl; exit_time = bct; exit_reason = "sl"; break
                if bar.high >= tp: exit_price = tp; exit_time = bct; exit_reason = "tp"; break
            else:
                if bar.high >= sl: exit_price = sl; exit_time = bct; exit_reason = "sl"; break
                if bar.low <= tp: exit_price = tp; exit_time = bct; exit_reason = "tp"; break
        if exit_price is None:
            exit_price = bars[-1].close
            exit_time = bars[-1].close_time
            exit_reason = "eod"
        pnl_pts = (exit_price - ep) if d == "long" else (ep - exit_price)
        trades.append({
            "source": sig["source"],
            "date": str(pd.Timestamp(sig["confirm_time"]).tz_convert(ET).date()),
            "pnl_pts": pnl_pts,
            "pnl_dollars": pnl_pts * POINT_VALUE,
            "direction": d,
        })
        last_exit = exit_time
    return trades


def compute_metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "pf": 0, "dd": 0, "avg_win": 0, "avg_loss": 0,
                "n_base": 0, "n_trend": 0, "trend_pnl": 0, "base_pnl": 0}
    df = pd.DataFrame(trades)
    n = len(df)
    pnl = df["pnl_dollars"].sum()
    wins = df[df["pnl_dollars"] > 0]
    losses = df[df["pnl_dollars"] < 0]
    wr = len(wins) / n * 100
    gp = wins["pnl_dollars"].sum() if len(wins) else 0
    gl = abs(losses["pnl_dollars"].sum()) if len(losses) else 1
    pf = gp / gl if gl > 0 else 999
    cum = df["pnl_dollars"].cumsum()
    dd = float((cum - cum.cummax()).min())
    avg_w = wins["pnl_dollars"].mean() if len(wins) else 0
    avg_l = losses["pnl_dollars"].mean() if len(losses) else 0
    base = df[df["source"] == "base"]
    trend = df[df["source"] == "trend"]
    return {
        "n": n, "pnl": pnl, "wr": wr, "pf": pf, "dd": dd,
        "avg_win": avg_w, "avg_loss": avg_l,
        "n_base": len(base), "n_trend": len(trend),
        "base_pnl": base["pnl_dollars"].sum() if len(base) else 0,
        "trend_pnl": trend["pnl_dollars"].sum() if len(trend) else 0,
    }


def main():
    print("=" * 110)
    print("  BAND ZONE SWEEP — IS/OOS (trending zone 3-10, base fixed at 3)")
    print(f"  IS: {START_DATE} - {IS_END}  |  OOS: {OOS_START} - {END_DATE}")
    print("=" * 110)
    print()

    print("Building ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done ({len(adx_lookup)} rows)")

    is_end_d = datetime.strptime(IS_END, "%Y-%m-%d").date()
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    # Preload per-day data once
    print("Loading per-day data...", flush=True)
    day_cache = []
    for cache_file in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        try:
            fd = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd < start_d or fd > end_d:
            continue
        signal_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_file.exists():
            continue
        with open(signal_file, "rb") as f:
            signal_data = pickle.load(f)
        bars = signal_data.get("bars") or []
        if not bars:
            continue
        with open(cache_file, "rb") as f:
            vwap_data = pickle.load(f)
        base_raw = vwap_data.get("signals") or []

        # Trending precompute
        vwap_price_file = VWAP_PRICE_CACHE_DIR / f"{date_str}.pkl"
        tb_file = TIMEBARS_DIR / f"timebars_5min_{date_str.replace('-','_')}.pkl"
        day_state = None
        bars_5min = None
        if vwap_price_file.exists() and tb_file.exists():
            with open(vwap_price_file, "rb") as f:
                vwap_df = pickle.load(f)
            bars_5min = load_5min_bars_trending(date_str)
            if bars_5min is not None:
                day_state = precompute_trend_state(bars_5min, vwap_df, TREND_LB, TREND_STD)
        day_cache.append({
            "date": date_str, "file_date": fd,
            "bars": bars, "base_raw": base_raw, "day_state": day_state,
        })
    print(f"  {len(day_cache)} days loaded")
    print()

    zones = list(range(3, 11))  # 3..10
    rows = []

    for zone in zones:
        is_trades = []
        oos_trades = []
        for dc in day_cache:
            bars = dc["bars"]
            # Base signals: filter once per day using prepare_base_signals (zone is fixed at BASE_ZONE via raw cache's zone; but raw cache was built at zone=3 which is what we want)
            base_sigs = prepare_base_signals(dc["base_raw"], adx_lookup)
            # Trending signals at this zone
            trend_raw = detect_trending_signals_zone(bars, dc["day_state"], zone) if dc["day_state"] is not None else []
            trend_sigs = prepare_trend_signals(trend_raw, adx_lookup)
            day_trades = simulate_day_combined(bars, base_sigs, trend_sigs)
            for t in day_trades:
                if dc["file_date"] <= is_end_d:
                    is_trades.append(t)
                else:
                    oos_trades.append(t)

        is_m = compute_metrics(is_trades)
        oos_m = compute_metrics(oos_trades)
        rows.append((zone, is_m, oos_m))

    # Report
    print(f"{'Zone':>5s} | {'IS Trades':>10s} {'IS PnL':>10s} {'IS WR':>7s} {'IS PF':>6s} {'IS DD':>9s} {'IS Trend':>9s} | "
          f"{'OOS Trds':>9s} {'OOS PnL':>10s} {'OOS WR':>7s} {'OOS PF':>6s} {'OOS DD':>9s} {'OOS Trend':>10s}")
    print("-" * 140)
    for zone, is_m, oos_m in rows:
        star = " *" if zone == 3 else "  "
        print(f"{zone:>4.1f}{star} | {is_m['n']:>10d} ${is_m['pnl']:>+9,.0f} {is_m['wr']:>6.1f}% {is_m['pf']:>6.2f} ${is_m['dd']:>+8,.0f} "
              f"{is_m['n_trend']:>3d}/${is_m['trend_pnl']:>+6,.0f} | "
              f"{oos_m['n']:>9d} ${oos_m['pnl']:>+9,.0f} {oos_m['wr']:>6.1f}% {oos_m['pf']:>6.2f} ${oos_m['dd']:>+8,.0f} "
              f"{oos_m['n_trend']:>3d}/${oos_m['trend_pnl']:>+6,.0f}")
    print()
    print("* = current production value. 'Trend' col = (# trend trades / trend-only pnl)")
    print()

    # Total column
    print("COMBINED IS+OOS SUMMARY:")
    print(f"{'Zone':>5s} | {'Trades':>7s} {'Total PnL':>11s} {'WR':>6s} {'PF':>6s} {'Max DD':>10s} {'Trend #':>8s} {'Trend PnL':>10s}")
    print("-" * 80)
    for zone, is_m, oos_m in rows:
        # Reconstitute combined by resumming (approx; use sum of pnl, trades)
        total_n = is_m['n'] + oos_m['n']
        total_pnl = is_m['pnl'] + oos_m['pnl']
        total_trend = is_m['n_trend'] + oos_m['n_trend']
        total_trend_pnl = is_m['trend_pnl'] + oos_m['trend_pnl']
        star = " *" if zone == 3 else "  "
        print(f"{zone:>4.1f}{star} | {total_n:>7d} ${total_pnl:>+10,.0f} {'--':>6s} {'--':>6s} {'--':>10s} {total_trend:>8d} ${total_trend_pnl:>+9,.0f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

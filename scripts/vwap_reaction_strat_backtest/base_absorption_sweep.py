"""
Base strategy absorption parameter sweep — IS/OOS 50/50 split.

Sweeps:
  1. Per-level delta threshold: 20 to 100 (step 10)
  2. Full-bar delta threshold: 30 to 150 (step 10)
  3. Location filter: absorption on correct half of candle

All use zone-based bias, look-ahead-fixed 5-min bars, SL=0.5 TP=1.5 (and 0.5/0.5).
No ADX filter.

Usage:
    python scripts/vwap_reaction_strat_backtest/base_absorption_sweep.py
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import ET, DATA_DIR, LUNCH_START, LUNCH_END

SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_DIR = DATA_DIR / "timebars_5min"

POINT_VALUE = 20.0
TICK_SIZE = 0.25
VWAP_ZONE_POINTS = 3.0
ATR_PERIOD = 14

ENTRY_SLIP_TICKS = 0
SL_SLIP_TICKS = 2
TP_SLIP_TICKS = 0
FORCE_CLOSE = "16:58"

START_DATE = "2025-03-13"
END_DATE = "2026-04-17"


def to_et(ct):
    return ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)


def load_5min_bars_fixed(date_str):
    bar_file = TIMEBARS_DIR / f"timebars_5min_{date_str.replace('-','_')}.pkl"
    if not bar_file.exists():
        return None
    with open(bar_file, "rb") as f:
        bars = pickle.load(f)
    if not bars:
        return None
    rows = [{'timestamp': b['open_time'], 'high': b['high'], 'low': b['low'], 'close': b['close']} for b in bars]
    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    df.index = df.index + pd.Timedelta(minutes=5)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['prev_close']),
                                     abs(df['low'] - df['prev_close'])))
    df['atr'] = df['tr'].rolling(ATR_PERIOD, min_periods=1).mean()
    return df


def get_vwap_at_time(vwap_df, bar_time):
    if vwap_df is None or vwap_df.empty:
        return None
    prior = vwap_df[vwap_df.index <= bar_time]
    if prior.empty:
        return None
    return float(prior['vwap'].iloc[-1])


def get_atr_at_time(bars_5min, signal_time):
    if bars_5min is None or bars_5min.empty:
        return None
    prior = bars_5min[bars_5min.index <= signal_time]
    if prior.empty:
        return None
    return float(prior['atr'].iloc[-1])


def get_5min_bias_zone(bars_5min, vwap_df, as_of_time):
    """Zone-based bias: only flip when 5m close is outside VWAP +/- 3."""
    if bars_5min is None or bars_5min.empty or vwap_df is None or vwap_df.empty:
        return None
    prior_bars = bars_5min[bars_5min.index <= as_of_time]
    if prior_bars.empty:
        return None
    for idx in range(len(prior_bars) - 1, max(len(prior_bars) - 50, -1), -1):
        bar_close = float(prior_bars['close'].iloc[idx])
        bar_time = prior_bars.index[idx]
        vwap_at_bar = get_vwap_at_time(vwap_df, bar_time)
        if vwap_at_bar is None:
            continue
        if bar_close > vwap_at_bar + VWAP_ZONE_POINTS:
            return "long"
        elif bar_close < vwap_at_bar - VWAP_ZONE_POINTS:
            return "short"
    return None


def detect_signals_parametric(bars, vwap_df, bars_5min,
                               level_delta_min=30,
                               bar_delta_min=None,
                               require_location=False):
    """
    Detect base VWAP reaction signals with configurable absorption.

    Modes (pick one via arguments):
      - level_delta_min > 0, bar_delta_min=None: per-level absorption (original)
      - bar_delta_min > 0, level_delta_min=None: full-bar delta
      - require_location=True: absorption must be on correct half of candle

    For SELLS (bearish candle):
      - Per-level: any level has delta >= level_delta_min (buyer absorption)
      - Bar delta: bar.delta >= bar_delta_min (positive delta on bearish bar)
      - Location: the absorbing level must be in upper half (price >= midpoint)

    For BUYS (bullish candle):
      - Per-level: any level has delta <= -level_delta_min (seller absorption)
      - Bar delta: bar.delta <= -bar_delta_min (negative delta on bullish bar)
      - Location: the absorbing level must be in lower half (price <= midpoint)
    """
    signals = []

    for i in range(len(bars) - 1):
        sb = bars[i]
        cb = bars[i + 1]
        if not sb.closed or not cb.closed:
            continue

        bt = sb.close_time
        bt_et = to_et(bt)
        hm = bt_et.strftime("%H:%M")
        if "17:00" <= hm < "19:00":
            continue
        if "16:00" <= hm < "19:10":
            continue

        closed_bearish = sb.close < sb.open
        closed_bullish = sb.close > sb.open
        if not closed_bearish and not closed_bullish:
            continue

        # Check absorption based on mode
        has_bear_abs = False  # bearish absorption (for shorts): buyers absorbed
        has_bull_abs = False  # bullish absorption (for longs): sellers absorbed

        midpoint = (sb.high + sb.low) / 2.0

        if bar_delta_min is not None:
            # Full-bar delta mode
            if closed_bearish and sb.delta >= bar_delta_min:
                has_bear_abs = True
            elif closed_bullish and sb.delta <= -bar_delta_min:
                has_bull_abs = True
        else:
            # Per-level delta mode
            for price, lv in sb.levels.items():
                delta = lv.delta
                if closed_bearish and delta >= level_delta_min:
                    if require_location:
                        if price >= midpoint:
                            has_bear_abs = True
                    else:
                        has_bear_abs = True
                elif closed_bullish and delta <= -level_delta_min:
                    if require_location:
                        if price <= midpoint:
                            has_bull_abs = True
                    else:
                        has_bull_abs = True

        if not has_bear_abs and not has_bull_abs:
            continue

        # VWAP zone check
        vwap_value = get_vwap_at_time(vwap_df, bt)
        if vwap_value is None:
            continue

        zh = vwap_value + VWAP_ZONE_POINTS
        zl = vwap_value - VWAP_ZONE_POINTS
        if not (sb.high >= zl and sb.low <= zh):
            continue

        # Confirm bar zone check
        confirm_vwap = get_vwap_at_time(vwap_df, cb.close_time)
        if confirm_vwap is None:
            confirm_vwap = vwap_value
        czh = confirm_vwap + VWAP_ZONE_POINTS
        czl = confirm_vwap - VWAP_ZONE_POINTS
        if not (cb.high >= czl and cb.low <= czh):
            continue

        # Zone-based bias
        bias = get_5min_bias_zone(bars_5min, vwap_df, bt)
        if bias is None:
            continue

        # Approach direction
        lookback = min(10, i)
        approached_above = any(bars[j].close > zh for j in range(i - lookback, i) if bars[j].closed)
        approached_below = any(bars[j].close < zl for j in range(i - lookback, i) if bars[j].closed)

        confirm_bearish = cb.close < cb.open
        confirm_bullish = cb.close > cb.open

        atr = get_atr_at_time(bars_5min, bt)
        if atr is None or atr <= 0:
            continue

        if bias == "long" and approached_above and has_bull_abs and confirm_bullish:
            signals.append({"bar_index": i, "direction": "long",
                            "confirm_time": cb.close_time, "entry_price": cb.close, "atr": atr})
        if bias == "short" and approached_below and has_bear_abs and confirm_bearish:
            signals.append({"bar_index": i, "direction": "short",
                            "confirm_time": cb.close_time, "entry_price": cb.close, "atr": atr})

    return signals


def simulate(bars, sigs, sl_mult, tp_mult):
    sigs = sorted(sigs, key=lambda x: x["confirm_time"])
    trades, last_exit = [], None
    for sig in sigs:
        ct_et = to_et(sig["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        if LUNCH_START <= hm < LUNCH_END:
            continue
        if last_exit is not None and sig["confirm_time"] <= last_exit:
            continue
        d, ep, atr = sig["direction"], sig["entry_price"], sig["atr"]
        if d == "long":
            sl = ep - atr * sl_mult
            tp = ep + atr * tp_mult
        else:
            sl = ep + atr * sl_mult
            tp = ep - atr * tp_mult

        cbi = sig["bar_index"] + 1
        ex_price = ex_time = None
        for j in range(cbi + 1, len(bars)):
            b = bars[j]
            if not b.closed:
                continue
            bet = to_et(b.close_time)
            if bet.strftime("%H:%M") >= FORCE_CLOSE:
                ex_price, ex_time = b.close, b.close_time
                break
            if d == "long":
                if b.low <= sl:
                    ex_price, ex_time = sl - SL_SLIP_TICKS * TICK_SIZE, b.close_time
                    break
                if b.high >= tp:
                    ex_price, ex_time = tp - TP_SLIP_TICKS * TICK_SIZE, b.close_time
                    break
            else:
                if b.high >= sl:
                    ex_price, ex_time = sl + SL_SLIP_TICKS * TICK_SIZE, b.close_time
                    break
                if b.low <= tp:
                    ex_price, ex_time = tp + TP_SLIP_TICKS * TICK_SIZE, b.close_time
                    break
        if ex_price is None:
            ex_price, ex_time = bars[-1].close, bars[-1].close_time
        pnl = ((ex_price - ep) if d == "long" else (ep - ex_price)) * POINT_VALUE
        trades.append({"date": str(to_et(sig["confirm_time"]).date()), "pnl": pnl})
        last_exit = ex_time
    return trades


def stats(trades):
    if not trades:
        return dict(n=0, pnl=0, wr=0, pf=0, dd=0, avg=0)
    df = pd.DataFrame(trades)
    n = len(df)
    pnl = df["pnl"].sum()
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    wr = len(wins) / n * 100
    gp = wins["pnl"].sum() if len(wins) else 0
    gl = abs(losses["pnl"].sum()) if len(losses) else 1
    pf = gp / gl if gl > 0 else float("inf")
    cum = df["pnl"].cumsum()
    dd = float((cum - cum.cummax()).min())
    avg = pnl / n
    return dict(n=n, pnl=pnl, wr=wr, pf=pf, dd=dd, avg=avg)


def main():
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    # Collect dates
    all_dates = []
    for sig_file in sorted(SIGNAL_CACHE_DIR.glob("*.pkl")):
        ds = sig_file.stem
        try:
            fd = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd < start_d or fd > end_d:
            continue
        vwap_file = VWAP_CACHE_DIR / f"{ds}.pkl"
        tb_file = TIMEBARS_DIR / f"timebars_5min_{ds.replace('-','_')}.pkl"
        if vwap_file.exists() and tb_file.exists():
            all_dates.append(fd)

    split = len(all_dates) // 2
    is_dates = set(all_dates[:split])
    print(f"Total days: {len(all_dates)} | IS: {split}d | OOS: {len(all_dates)-split}d")

    # Pre-load all data
    print("Loading data...")
    day_data = {}
    for fd in all_dates:
        ds = fd.strftime("%Y-%m-%d")
        with open(SIGNAL_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            bars = pickle.load(f).get("bars") or []
        if not bars:
            continue
        with open(VWAP_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            vwap_df = pickle.load(f)
        bars_5min = load_5min_bars_fixed(ds)
        if bars_5min is None:
            continue
        day_data[fd] = (bars, vwap_df, bars_5min)
    print(f"Loaded {len(day_data)} days\n")

    sl_tp_configs = [(0.5, 0.5), (0.5, 1.5)]

    def run_sweep(label, sl_m, tp_m, **detect_kwargs):
        is_tr, oos_tr = [], []
        for fd, (bars, vwap_df, bars_5min) in day_data.items():
            sigs = detect_signals_parametric(bars, vwap_df, bars_5min, **detect_kwargs)
            day_trades = simulate(bars, sigs, sl_m, tp_m)
            (is_tr if fd in is_dates else oos_tr).extend(day_trades)
        return stats(is_tr), stats(oos_tr)

    def print_row(label, sis, soos):
        print(f"  {label:>30} | {sis['n']:>5} ${sis['pnl']:>+9,.0f} {sis['pf']:>5.2f} ${sis['dd']:>+7,.0f} {sis['wr']:>5.1f} ${sis['avg']:>+6,.0f}"
              f"  |  {soos['n']:>5} ${soos['pnl']:>+9,.0f} {soos['pf']:>5.2f} ${soos['dd']:>+7,.0f} {soos['wr']:>5.1f} ${soos['avg']:>+6,.0f}")

    for sl_m, tp_m in sl_tp_configs:
        # =====================================================================
        # PART 1: Per-level delta threshold sweep
        # =====================================================================
        print("=" * 140)
        print(f"  PART 1: PER-LEVEL DELTA THRESHOLD SWEEP (SL={sl_m} TP={tp_m})")
        print("=" * 140)
        print(f"  {'Config':>30} | {'IS n':>5} {'IS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}"
              f"  |  {'OOS n':>5} {'OOS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}")
        print("  " + "-" * 135)

        for thresh in range(20, 110, 10):
            si, so = run_sweep(f"level_delta>={thresh}", sl_m, tp_m,
                               level_delta_min=thresh, bar_delta_min=None, require_location=False)
            print_row(f"level_delta>={thresh}", si, so)

        # =====================================================================
        # PART 2: Full-bar delta threshold sweep
        # =====================================================================
        print()
        print("=" * 140)
        print(f"  PART 2: FULL-BAR DELTA THRESHOLD SWEEP (SL={sl_m} TP={tp_m})")
        print(f"  Bearish bar + positive bar delta >= threshold = short signal")
        print(f"  Bullish bar + negative bar delta <= -threshold = long signal")
        print("=" * 140)
        print(f"  {'Config':>30} | {'IS n':>5} {'IS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}"
              f"  |  {'OOS n':>5} {'OOS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}")
        print("  " + "-" * 135)

        for thresh in range(10, 160, 10):
            si, so = run_sweep(f"bar_delta>={thresh}", sl_m, tp_m,
                               level_delta_min=None, bar_delta_min=thresh, require_location=False)
            print_row(f"bar_delta>={thresh}", si, so)

        # =====================================================================
        # PART 3: Per-level delta + location filter
        # =====================================================================
        print()
        print("=" * 140)
        print(f"  PART 3: PER-LEVEL DELTA + LOCATION FILTER (SL={sl_m} TP={tp_m})")
        print(f"  Bullish bar: sell absorption must be on LOWER half of candle")
        print(f"  Bearish bar: buy absorption must be on UPPER half of candle")
        print("=" * 140)
        print(f"  {'Config':>30} | {'IS n':>5} {'IS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}"
              f"  |  {'OOS n':>5} {'OOS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}")
        print("  " + "-" * 135)

        for thresh in range(20, 110, 10):
            si_no, so_no = run_sweep(f"lvl>={thresh} no-loc", sl_m, tp_m,
                                      level_delta_min=thresh, bar_delta_min=None, require_location=False)
            si_loc, so_loc = run_sweep(f"lvl>={thresh} +location", sl_m, tp_m,
                                        level_delta_min=thresh, bar_delta_min=None, require_location=True)
            print_row(f"lvl>={thresh} no-loc", si_no, so_no)
            print_row(f"lvl>={thresh} +location", si_loc, so_loc)
            print()

        # =====================================================================
        # PART 4: Combined bar_delta + level_delta
        # =====================================================================
        print()
        print("=" * 140)
        print(f"  PART 4: BAR DELTA + LEVEL DELTA COMBINED (SL={sl_m} TP={tp_m})")
        print(f"  Requires BOTH: bar-level delta AND full-bar delta thresholds met")
        print("=" * 140)
        print(f"  {'Config':>30} | {'IS n':>5} {'IS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}"
              f"  |  {'OOS n':>5} {'OOS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}")
        print("  " + "-" * 135)

        # For combined, need a custom detection
        for bar_thresh in [30, 50, 80]:
            for lvl_thresh in [20, 30, 40, 50]:
                is_tr, oos_tr = [], []
                for fd, (bars, vwap_df, bars_5min) in day_data.items():
                    sigs = detect_signals_combined(bars, vwap_df, bars_5min, lvl_thresh, bar_thresh)
                    day_trades = simulate(bars, sigs, sl_m, tp_m)
                    (is_tr if fd in is_dates else oos_tr).extend(day_trades)
                si, so = stats(is_tr), stats(oos_tr)
                print_row(f"bar>={bar_thresh}+lvl>={lvl_thresh}", si, so)


def detect_signals_combined(bars, vwap_df, bars_5min, level_delta_min, bar_delta_min):
    """Requires BOTH bar-level and full-bar delta thresholds."""
    signals = []
    for i in range(len(bars) - 1):
        sb = bars[i]
        cb = bars[i + 1]
        if not sb.closed or not cb.closed:
            continue

        bt = sb.close_time
        bt_et = to_et(bt)
        hm = bt_et.strftime("%H:%M")
        if "17:00" <= hm < "19:00" or "16:00" <= hm < "19:10":
            continue

        closed_bearish = sb.close < sb.open
        closed_bullish = sb.close > sb.open
        if not closed_bearish and not closed_bullish:
            continue

        has_bear_abs = False
        has_bull_abs = False

        # Check full-bar delta first
        if closed_bearish and sb.delta >= bar_delta_min:
            # Also check per-level
            for price, lv in sb.levels.items():
                if lv.delta >= level_delta_min:
                    has_bear_abs = True
                    break
        elif closed_bullish and sb.delta <= -bar_delta_min:
            for price, lv in sb.levels.items():
                if lv.delta <= -level_delta_min:
                    has_bull_abs = True
                    break

        if not has_bear_abs and not has_bull_abs:
            continue

        vwap_value = get_vwap_at_time(vwap_df, bt)
        if vwap_value is None:
            continue

        zh = vwap_value + VWAP_ZONE_POINTS
        zl = vwap_value - VWAP_ZONE_POINTS
        if not (sb.high >= zl and sb.low <= zh):
            continue

        confirm_vwap = get_vwap_at_time(vwap_df, cb.close_time)
        if confirm_vwap is None:
            confirm_vwap = vwap_value
        czh = confirm_vwap + VWAP_ZONE_POINTS
        czl = confirm_vwap - VWAP_ZONE_POINTS
        if not (cb.high >= czl and cb.low <= czh):
            continue

        bias = get_5min_bias_zone(bars_5min, vwap_df, bt)
        if bias is None:
            continue

        lookback = min(10, i)
        approached_above = any(bars[j].close > zh for j in range(i - lookback, i) if bars[j].closed)
        approached_below = any(bars[j].close < zl for j in range(i - lookback, i) if bars[j].closed)

        confirm_bearish = cb.close < cb.open
        confirm_bullish = cb.close > cb.open

        atr = get_atr_at_time(bars_5min, bt)
        if atr is None or atr <= 0:
            continue

        if bias == "long" and approached_above and has_bull_abs and confirm_bullish:
            signals.append({"bar_index": i, "direction": "long",
                            "confirm_time": cb.close_time, "entry_price": cb.close, "atr": atr})
        if bias == "short" and approached_below and has_bear_abs and confirm_bearish:
            signals.append({"bar_index": i, "direction": "short",
                            "confirm_time": cb.close_time, "entry_price": cb.close, "atr": atr})

    return signals


if __name__ == "__main__":
    main()

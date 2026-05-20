"""
Trending strategy per-level delta minimum optimization.
Sweeps level_delta from 20 to 100 in steps of 10.
Uses best known params: STD1, LB=14, ADX>=30, SL=0.75, TP=1.25.
Also tests SL=0.5/TP=1.5 and SL=1.0/TP=1.0 for comparison.
IS/OOS 50/50 chronological split.

Usage:
    python scripts/vwap_reaction_strat_backtest/trending_lvldelta_sweep.py
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
BAND_ZONE_ABOVE = 3.0
BAND_ZONE_BELOW = 4.0
ATR_PERIOD = 14
POINT_VALUE = 20.0

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"

START_DATE = "2025-03-13"
END_DATE = "2026-04-17"

SL_TP_CONFIGS = [
    (0.5, 1.5, "0.5/1.5"),
    (0.75, 1.25, "0.75/1.25"),
    (1.0, 1.0, "1.0/1.0"),
    (1.0, 1.5, "1.0/1.5"),
    (0.5, 1.0, "0.5/1.0"),
]


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
    df.index = df.index + pd.Timedelta(minutes=5)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['prev_close']),
                                     abs(df['low'] - df['prev_close'])))
    df['atr'] = df['tr'].rolling(ATR_PERIOD, min_periods=1).mean()
    return df


def has_absorption(bar, min_delta):
    closed_bearish = bar.close < bar.open
    closed_bullish = bar.close > bar.open
    has_bear = False
    has_bull = False
    for price, lv in bar.levels.items():
        delta = lv.delta
        if closed_bearish and delta >= min_delta:
            has_bear = True
        elif closed_bullish and delta <= -min_delta:
            has_bull = True
    return has_bull, has_bear


def precompute_day_state(bars_5min, vwap_df, lookback, std_band):
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

    trend_run_id = (merged['trend'] != merged['trend'].shift()).cumsum()
    hit_upper = (merged['trend'] == 'up')   & (merged['close'] >= merged['band_upper'])
    hit_lower = (merged['trend'] == 'down') & (merged['close'] <= merged['band_lower'])
    merged['approach_from_above'] = hit_upper.groupby(trend_run_id).cummax().astype(bool)
    merged['approach_from_below'] = hit_lower.groupby(trend_run_id).cummax().astype(bool)
    merged.loc[merged['trend'].isna(), ['approach_from_above', 'approach_from_below']] = False

    return merged[['close', 'atr', 'vwap', 'band_upper', 'band_lower', 'trend',
                   'approach_from_above', 'approach_from_below']]


def detect_signals(bars, day_state, min_delta):
    """Detect absorption signals at std dev bands during trends."""
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

        is_bull, is_bear = has_absorption(sb, min_delta)
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
            zh, zl = band + BAND_ZONE_ABOVE, band - BAND_ZONE_BELOW
            if not (sb.high >= zl and sb.low <= zh):
                continue
            cs = lookup(cb.close_time)
            cb_val = cs[3] if cs is not None else band
            if not (cb.high >= cb_val - BAND_ZONE_BELOW and cb.low <= cb_val + BAND_ZONE_ABOVE):
                continue
            if not is_bull or cb.close <= cb.open:
                continue
            if last_5m_close < zl:
                continue
            if not approach_above:
                continue
            signals.append({"bar_index": i, "direction": "long", "confirm_time": cb.close_time,
                            "entry_price": cb.close, "atr": float(atr), "date": ""})

        elif trend == "down":
            band = state[4]
            zh, zl = band + BAND_ZONE_BELOW, band - BAND_ZONE_ABOVE
            if not (sb.high >= zl and sb.low <= zh):
                continue
            cs = lookup(cb.close_time)
            cb_val = cs[4] if cs is not None else band
            if not (cb.high >= cb_val - BAND_ZONE_ABOVE and cb.low <= cb_val + BAND_ZONE_BELOW):
                continue
            if not is_bear or cb.close >= cb.open:
                continue
            if last_5m_close > zh:
                continue
            if not approach_below:
                continue
            signals.append({"bar_index": i, "direction": "short", "confirm_time": cb.close_time,
                            "entry_price": cb.close, "atr": float(atr), "date": ""})

    return signals


def simulate(signals, bars, sl_mult, tp_mult):
    trades = []
    last_exit = None
    for s in signals:
        ep = s["entry_price"]
        d = s["direction"]
        atr = s["atr"]
        ci = s["bar_index"] + 1
        ct = s["confirm_time"]

        if hasattr(ct, 'tz_convert'):
            ct_et = ct.tz_convert(ET)
        else:
            ct_et = pd.Timestamp(ct, tz='UTC').tz_convert(ET)
        if "16:00" <= ct_et.strftime("%H:%M") < "19:10":
            continue
        if last_exit is not None and ct <= last_exit:
            continue

        sl = ep - atr * sl_mult if d == "long" else ep + atr * sl_mult
        tp = ep + atr * tp_mult if d == "long" else ep - atr * tp_mult

        xp, xt = None, None
        for j in range(ci + 1, len(bars)):
            b = bars[j]
            if not b.closed:
                continue
            bct = b.close_time
            if hasattr(bct, 'tz_convert'):
                bet = bct.tz_convert(ET)
            else:
                bet = pd.Timestamp(bct, tz='UTC').tz_convert(ET)
            if bet.strftime("%H:%M") >= FORCE_CLOSE:
                xp, xt = b.close, bct
                break
            if d == "long":
                if b.low <= sl: xp, xt = sl, bct; break
                if b.high >= tp: xp, xt = tp, bct; break
            else:
                if b.high >= sl: xp, xt = sl, bct; break
                if b.low <= tp: xp, xt = tp, bct; break

        if xp is None:
            lb = bars[-1]
            xp, xt = lb.close, lb.close_time

        pnl = (xp - ep) if d == "long" else (ep - xp)
        trades.append({"pnl": pnl * POINT_VALUE, "direction": d})
        last_exit = xt

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

    print("Building ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done ({len(adx_lookup)} rows)")

    # Load day data
    print("Loading day data...", end=" ", flush=True)
    all_dates = []
    day_raw = {}
    for vwap_file in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        ds = vwap_file.stem
        try:
            fd = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd < start_d or fd > end_d:
            continue
        sf = SIGNAL_CACHE_DIR / f"{ds}.pkl"
        tf = TIMEBARS_5MIN_DIR / f"timebars_5min_{ds.replace('-','_')}.pkl"
        if not sf.exists() or not tf.exists():
            continue
        vwap_df = load_pickle(vwap_file)
        bars = load_pickle(sf).get("bars") or []
        bars_5min = load_5min_bars(ds)
        if bars and bars_5min is not None:
            all_dates.append(fd)
            day_raw[fd] = (ds, bars, bars_5min, vwap_df)

    split = len(all_dates) // 2
    is_dates = set(all_dates[:split])
    print(f"done ({len(all_dates)} days)")
    print(f"IS: {all_dates[0]}->{all_dates[split-1]} ({split}d) | OOS: {all_dates[split]}->{all_dates[-1]} ({len(all_dates)-split}d)")
    sys.stdout.flush()

    # Precompute day_state once (doesn't depend on delta threshold)
    # Using STD1, LB=14 (best known params)
    std_band = 1
    lookback = 14
    print(f"Precomputing day states (STD{std_band}, LB={lookback})...", end=" ", flush=True)
    day_states = {}
    for fd, (ds, bars, bars_5min, vwap_df) in day_raw.items():
        day_state = precompute_day_state(bars_5min, vwap_df, lookback, std_band)
        day_states[fd] = day_state
    print("done")
    sys.stdout.flush()

    # Delta thresholds to sweep
    delta_thresholds = list(range(20, 110, 10))

    for sl_m, tp_m, sl_tp_label in SL_TP_CONFIGS:
        print()
        print("=" * 140)
        print(f"  TRENDING LEVEL DELTA SWEEP: STD{std_band}, LB={lookback}, ADX>=30, SL={sl_m}/TP={tp_m}")
        print("=" * 140)
        print(f"  {'Delta':>10}"
              f" | {'IS n':>5} {'IS PnL':>9} {'IS PF':>6} {'IS DD':>8} {'IS WR%':>6} {'IS Avg':>7}"
              f" | {'OOS n':>5} {'OOS PnL':>9} {'OOS PF':>6} {'OOS DD':>8} {'OOS WR%':>6} {'OOS Avg':>7}")
        print("  " + "-" * 130)
        sys.stdout.flush()

        for min_delta in delta_thresholds:
            is_trades, oos_trades = [], []

            for fd, (ds, bars, bars_5min, vwap_df) in day_raw.items():
                day_state = day_states[fd]
                sigs = detect_signals(bars, day_state, min_delta)

                # ADX >= 30 filter
                filtered = []
                for s in sigs:
                    ct = s["confirm_time"]
                    ct_et = ct.tz_convert(ET) if hasattr(ct, 'tz_convert') else pd.Timestamp(ct, tz='UTC').tz_convert(ET)
                    adx_val = get_adx_at_time(ct_et, adx_lookup)
                    if not np.isnan(adx_val) and adx_val >= 30.0:
                        filtered.append(s)

                if filtered:
                    day_trades = simulate(filtered, bars, sl_m, tp_m)
                    (is_trades if fd in is_dates else oos_trades).extend(day_trades)

            si = stats(is_trades)
            so = stats(oos_trades)
            flag = " *" if si["pf"] > 1.0 and so["pf"] > 1.0 and si["n"] >= 5 and so["n"] >= 5 else "  "
            print(f"  {'>=' + str(min_delta):>10}"
                  f" | {si['n']:>5} ${si['pnl']:>+8,.0f} {si['pf']:>6.2f} ${si['dd']:>+7,.0f} {si['wr']:>6.1f} ${si['avg']:>+6,.0f}"
                  f" | {so['n']:>5} ${so['pnl']:>+8,.0f} {so['pf']:>6.2f} ${so['dd']:>+7,.0f} {so['wr']:>6.1f} ${so['avg']:>+6,.0f}"
                  f"{flag}")
            sys.stdout.flush()

    # Also run without ADX filter for comparison
    print()
    print("=" * 140)
    print(f"  TRENDING LEVEL DELTA SWEEP: STD{std_band}, LB={lookback}, NO ADX FILTER, SL=0.5/TP=1.5")
    print("=" * 140)
    print(f"  {'Delta':>10}"
          f" | {'IS n':>5} {'IS PnL':>9} {'IS PF':>6} {'IS DD':>8} {'IS WR%':>6} {'IS Avg':>7}"
          f" | {'OOS n':>5} {'OOS PnL':>9} {'OOS PF':>6} {'OOS DD':>8} {'OOS WR%':>6} {'OOS Avg':>7}")
    print("  " + "-" * 130)
    sys.stdout.flush()

    for min_delta in delta_thresholds:
        is_trades, oos_trades = [], []
        for fd, (ds, bars, bars_5min, vwap_df) in day_raw.items():
            day_state = day_states[fd]
            sigs = detect_signals(bars, day_state, min_delta)
            if sigs:
                day_trades = simulate(sigs, bars, 0.5, 1.5)
                (is_trades if fd in is_dates else oos_trades).extend(day_trades)
        si = stats(is_trades)
        so = stats(oos_trades)
        flag = " *" if si["pf"] > 1.0 and so["pf"] > 1.0 and si["n"] >= 5 and so["n"] >= 5 else "  "
        print(f"  {'>=' + str(min_delta):>10}"
              f" | {si['n']:>5} ${si['pnl']:>+8,.0f} {si['pf']:>6.2f} ${si['dd']:>+7,.0f} {si['wr']:>6.1f} ${si['avg']:>+6,.0f}"
              f" | {so['n']:>5} ${so['pnl']:>+8,.0f} {so['pf']:>6.2f} ${so['dd']:>+7,.0f} {so['wr']:>6.1f} ${so['avg']:>+6,.0f}"
              f"{flag}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

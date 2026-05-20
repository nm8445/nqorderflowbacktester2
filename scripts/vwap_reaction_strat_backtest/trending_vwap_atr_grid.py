"""
Trending VWAP Band — ATR Grid Optimization

Full SL/TP grid sweep: 0.50 to 2.00 in 0.25 increments.
ADX >= 30 filter. 50/50 in-sample / out-of-sample split.
Shows top 20 configs for STD1 and STD2 separately.
Preference: positive RR (TP >= SL).

Usage:
    python -u scripts/vwap_reaction_strat_backtest/trending_vwap_atr_grid.py
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
# Asymmetric band zone: "above" = side away from VWAP, "below" = side toward VWAP.
#   Upper band (longs):  zone = [band - BAND_ZONE_BELOW, band + BAND_ZONE_ABOVE]
#   Lower band (shorts): zone = [band - BAND_ZONE_ABOVE, band + BAND_ZONE_BELOW]
BAND_ZONE_ABOVE = 3.0  # outside (past the band, away from VWAP)
BAND_ZONE_BELOW = 4.0  # inside  (toward VWAP)
ATR_PERIOD = 14
MIN_DELTA = 30
POINT_VALUE = 20.0

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"

START_DATE = "2025-03-13"
END_DATE = "2026-04-08"
IS_END = "2025-09-25"
OOS_START = "2025-09-26"


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
    # Index at close time (open + 5min) so lookups only see completed bars
    df.index = df.index + pd.Timedelta(minutes=5)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['prev_close']),
                                     abs(df['low'] - df['prev_close'])))
    df['atr'] = df['tr'].rolling(ATR_PERIOD, min_periods=1).mean()
    return df


def has_absorption(bar):
    closed_bearish = bar.close < bar.open
    closed_bullish = bar.close > bar.open
    has_bear = False
    has_bull = False
    for price, lv in bar.levels.items():
        delta = lv.delta
        if closed_bearish and delta >= MIN_DELTA:
            has_bear = True
        elif closed_bullish and delta <= -MIN_DELTA:
            has_bull = True
    return has_bull, has_bear


def get_vwap_bands(vwap_df, ts, std_band):
    idx = vwap_df.index.searchsorted(ts, side='right') - 1
    if idx < 0:
        return None, None, None
    row = vwap_df.iloc[idx]
    return float(row['vwap']), float(row[f'std{std_band}_upper']), float(row[f'std{std_band}_lower'])


def precompute_day_state(bars_5min, vwap_df, lookback, std_band):
    if bars_5min is None or vwap_df is None or bars_5min.empty or vwap_df.empty:
        return None

    vwap_cols = ['vwap', f'std{std_band}_upper', f'std{std_band}_lower']
    vwap_series = vwap_df[vwap_cols].copy().sort_index()
    bs = bars_5min[['close', 'atr']].sort_index()

    # Align tz to UTC
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
    merged.loc[merged['trend'].isna(), ['approach_from_above', 'approach_from_below']] = False

    return merged[['close', 'atr', 'vwap', 'band_upper', 'band_lower', 'trend',
                   'approach_from_above', 'approach_from_below']]


def detect_signals(bars, day_state):
    """Detect absorption signals at std dev bands during trends."""
    if day_state is None or day_state.empty:
        return []

    signals = []
    state_index = day_state.index
    state_values = day_state.values
    # cols: close=0, atr=1, vwap=2, band_upper=3, band_lower=4, trend=5,
    #       approach_from_above=6, approach_from_below=7

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
            band = state[3]  # upper band
            # Upper band: 3 pts above (outside), 4 pts below (inside/toward VWAP)
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
            # Option B approach gate: 5-min bar must have closed at/above band in this trend run
            if not approach_above:
                continue
            signals.append({"bar_index": i, "direction": "long", "confirm_time": cb.close_time,
                            "entry_price": cb.close, "atr": float(atr), "vwap": float(vwap),
                            "band_value": float(band)})

        elif trend == "down":
            band = state[4]  # lower band
            # Lower band: 3 pts below (outside), 4 pts above (inside/toward VWAP)
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
            # Option B approach gate: 5-min bar must have closed at/below band in this trend run
            if not approach_below:
                continue
            signals.append({"bar_index": i, "direction": "short", "confirm_time": cb.close_time,
                            "entry_price": cb.close, "atr": float(atr), "vwap": float(vwap),
                            "band_value": float(band)})

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

        xp, xt, xr = None, None, None
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
                xp, xt, xr = b.close, bct, "session"
                break
            if d == "long":
                if b.low <= sl: xp, xt, xr = sl, bct, "sl"; break
                if b.high >= tp: xp, xt, xr = tp, bct, "tp"; break
            else:
                if b.high >= sl: xp, xt, xr = sl, bct, "sl"; break
                if b.low <= tp: xp, xt, xr = tp, bct, "tp"; break

        if xp is None:
            lb = bars[-1]
            xp, xt, xr = lb.close, lb.close_time, "eod"

        pnl = (xp - ep) if d == "long" else (ep - xp)
        trades.append({"pnl": pnl, "direction": d, "exit": xr, "date": s.get("date","")})
        last_exit = xt

    return trades


def calc_stats(trades):
    if not trades:
        return None
    pnls = [t["pnl"] for t in trades]
    arr = np.array(pnls)
    n = len(arr)
    w = arr[arr > 0]
    l = arr[arr < 0]
    wr = len(w) / n * 100
    avg_w = w.mean() if len(w) > 0 else 0
    avg_l = l.mean() if len(l) > 0 else 0
    exp = (wr/100 * avg_w) + ((1-wr/100) * avg_l)
    gp = w.sum() if len(w) > 0 else 0
    gl = abs(l.sum()) if len(l) > 0 else 1
    pf = gp / gl if gl > 0 else float('inf')
    total = arr.sum()
    cum = arr.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()
    return {"trades": n, "wr": wr, "exp": exp, "pf": pf, "total": total, "dd": dd,
            "avg_w": avg_w, "avg_l": avg_l}


def main():
    print("=" * 80)
    print("  TRENDING VWAP BAND — ATR GRID OPTIMIZATION")
    print("  ADX >= 30 | 50/50 IS/OOS split")
    print(f"  IS: {START_DATE} to {IS_END} | OOS: {OOS_START} to {END_DATE}")
    print("=" * 80)
    sys.stdout.flush()

    print("Building ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done ({len(adx_lookup)} rows)")

    # Pre-load all day data once
    print("Pre-loading all day data...", end=" ", flush=True)
    start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end = datetime.strptime(END_DATE, "%Y-%m-%d").date()
    is_end = datetime.strptime(IS_END, "%Y-%m-%d").date()
    oos_start = datetime.strptime(OOS_START, "%Y-%m-%d").date()

    day_data = {}  # date_str -> (bars, bars_5min, vwap_df)
    for vwap_file in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        ds = vwap_file.stem
        fd = datetime.strptime(ds, "%Y-%m-%d").date()
        if fd < start or fd > end:
            continue
        sf = SIGNAL_CACHE_DIR / f"{ds}.pkl"
        tf = TIMEBARS_5MIN_DIR / f"timebars_5min_{ds.replace('-','_')}.pkl"
        if not sf.exists() or not tf.exists():
            continue
        vwap_df = load_pickle(vwap_file)
        bars = load_pickle(sf)["bars"]
        bars_5min = load_5min_bars(ds)
        if bars and bars_5min is not None:
            day_data[ds] = (bars, bars_5min, vwap_df)
    print(f"done ({len(day_data)} days)")
    sys.stdout.flush()

    # SL/TP grid
    mults = [round(0.50 + i * 0.25, 2) for i in range(7)]  # 0.50 to 2.00
    lookbacks = [10, 14, 20]

    for std_band in [1, 2]:
        print(f"\n{'='*80}")
        print(f"  STD DEV {std_band} — GRID SEARCH")
        print(f"{'='*80}")
        sys.stdout.flush()

        # Precompute signals per day per lookback (signals don't change with SL/TP)
        all_results = []

        for lookback in lookbacks:
            print(f"\n  Computing signals for LB={lookback}...", end=" ", flush=True)

            # Build signals per day
            is_signals_by_day = {}
            oos_signals_by_day = {}

            for ds, (bars, bars_5min, vwap_df) in day_data.items():
                fd = datetime.strptime(ds, "%Y-%m-%d").date()
                day_state = precompute_day_state(bars_5min, vwap_df, lookback, std_band)
                sigs = detect_signals(bars, day_state)

                # ADX filter
                filtered = []
                for s in sigs:
                    s["date"] = ds
                    ct = s["confirm_time"]
                    ct_et = ct.tz_convert("America/New_York") if hasattr(ct, 'tz_convert') else pd.Timestamp(ct, tz='UTC').tz_convert("America/New_York")
                    adx_val = get_adx_at_time(ct_et, adx_lookup)
                    if not np.isnan(adx_val) and adx_val >= 30.0:
                        filtered.append(s)

                if filtered:
                    if fd <= is_end:
                        is_signals_by_day[ds] = filtered
                    else:
                        oos_signals_by_day[ds] = filtered

            is_total = sum(len(v) for v in is_signals_by_day.values())
            oos_total = sum(len(v) for v in oos_signals_by_day.values())
            print(f"IS={is_total} sigs on {len(is_signals_by_day)} days, OOS={oos_total} sigs on {len(oos_signals_by_day)} days")
            sys.stdout.flush()

            # Sweep SL/TP
            for sl_mult in mults:
                for tp_mult in mults:
                    # Skip negative RR (TP < SL)
                    rr = tp_mult / sl_mult

                    # IS
                    is_trades = []
                    for ds in sorted(is_signals_by_day.keys()):
                        bars = day_data[ds][0]
                        is_trades.extend(simulate(is_signals_by_day[ds], bars, sl_mult, tp_mult))
                    is_stats = calc_stats(is_trades)

                    # OOS
                    oos_trades = []
                    for ds in sorted(oos_signals_by_day.keys()):
                        bars = day_data[ds][0]
                        oos_trades.extend(simulate(oos_signals_by_day[ds], bars, sl_mult, tp_mult))
                    oos_stats = calc_stats(oos_trades)

                    all_results.append({
                        "std": std_band, "lb": lookback,
                        "sl": sl_mult, "tp": tp_mult, "rr": rr,
                        "is": is_stats, "oos": oos_stats,
                    })

            print(f"  LB={lookback} grid complete ({len(mults)*len(mults)} combos)")
            sys.stdout.flush()

        # Filter: both IS and OOS must have trades and positive expectancy
        # Sort by OOS expectancy (real edge indicator)
        valid = [r for r in all_results
                 if r["is"] is not None and r["oos"] is not None
                 and r["is"]["exp"] > 0 and r["oos"]["exp"] > 0
                 and r["rr"] >= 1.0]  # positive RR preference

        valid.sort(key=lambda r: r["oos"]["exp"], reverse=True)

        print(f"\n{'='*80}")
        print(f"  TOP 20 CONFIGS — STD{std_band} (positive RR, both IS+OOS profitable)")
        print(f"{'='*80}")
        print(f"  {'LB':>3} {'SL':>5} {'TP':>5} {'RR':>4} | {'IS Trades':>9} {'IS WR':>6} {'IS Exp':>8} {'IS PF':>6} {'IS P&L':>10} | {'OOS Trades':>10} {'OOS WR':>7} {'OOS Exp':>9} {'OOS PF':>7} {'OOS P&L':>10}")
        print(f"  {'---':>3} {'-----':>5} {'-----':>5} {'----':>4} | {'---------':>9} {'------':>6} {'--------':>8} {'------':>6} {'----------':>10} | {'----------':>10} {'-------':>7} {'---------':>9} {'-------':>7} {'----------':>10}")

        for r in valid[:20]:
            i = r["is"]
            o = r["oos"]
            print(f"  {r['lb']:>3} {r['sl']:>5.2f} {r['tp']:>5.2f} {r['rr']:>4.1f} | "
                  f"{i['trades']:>9} {i['wr']:>5.1f}% {i['exp']:>+7.2f} {i['pf']:>6.2f} ${i['total']*POINT_VALUE:>+9,.0f} | "
                  f"{o['trades']:>10} {o['wr']:>6.1f}% {o['exp']:>+8.2f} {o['pf']:>7.2f} ${o['total']*POINT_VALUE:>+9,.0f}")

        sys.stdout.flush()

        # Also show top 20 including sub-1:1 RR for comparison
        valid_all = [r for r in all_results
                     if r["is"] is not None and r["oos"] is not None
                     and r["is"]["exp"] > 0 and r["oos"]["exp"] > 0]
        valid_all.sort(key=lambda r: r["oos"]["exp"], reverse=True)

        if len(valid_all) > len(valid):
            print(f"\n  TOP 20 CONFIGS — STD{std_band} (ALL RR, both IS+OOS profitable)")
            print(f"  {'LB':>3} {'SL':>5} {'TP':>5} {'RR':>4} | {'IS Trades':>9} {'IS WR':>6} {'IS Exp':>8} {'IS PF':>6} {'IS P&L':>10} | {'OOS Trades':>10} {'OOS WR':>7} {'OOS Exp':>9} {'OOS PF':>7} {'OOS P&L':>10}")
            print(f"  {'---':>3} {'-----':>5} {'-----':>5} {'----':>4} | {'---------':>9} {'------':>6} {'--------':>8} {'------':>6} {'----------':>10} | {'----------':>10} {'-------':>7} {'---------':>9} {'-------':>7} {'----------':>10}")
            for r in valid_all[:20]:
                i = r["is"]
                o = r["oos"]
                print(f"  {r['lb']:>3} {r['sl']:>5.2f} {r['tp']:>5.2f} {r['rr']:>4.1f} | "
                      f"{i['trades']:>9} {i['wr']:>5.1f}% {i['exp']:>+7.2f} {i['pf']:>6.2f} ${i['total']*POINT_VALUE:>+9,.0f} | "
                      f"{o['trades']:>10} {o['wr']:>6.1f}% {o['exp']:>+8.2f} {o['pf']:>7.2f} ${o['total']*POINT_VALUE:>+9,.0f}")

        sys.stdout.flush()

    print("\n" + "=" * 80)
    print("  GRID OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

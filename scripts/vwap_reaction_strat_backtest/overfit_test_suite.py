"""
Overfit Test Framework — 5-Test Robustness Suite
Runs all 5 tests from docs/OVERFIT_TEST_FRAMEWORK.md on specified configs.

Tests:
  1. Parameter Stability — are neighboring SL/TP configs profitable?
  2. Walk-Forward — rolling 4-window OOS validation
  3. Monte Carlo Shuffle — trade order resampling (1000x)
  4. Bootstrap CI — resample trades with replacement (10000x)
  5. Direction Permutation — randomize long/short (1000x)

Usage:
    python scripts/vwap_reaction_strat_backtest/overfit_test_suite.py
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import ET, DATA_DIR, LUNCH_START, LUNCH_END, build_adx_lookup, get_adx_at_time

SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_DIR = DATA_DIR / "timebars_5min"

POINT_VALUE = 20.0
TICK_SIZE = 0.25
VWAP_ZONE_POINTS = 3.0
ATR_PERIOD = 14
FORCE_CLOSE = "16:58"

BAND_ZONE_ABOVE = 3.0
BAND_ZONE_BELOW = 4.0
STD_BAND = 1
LOOKBACK = 14

START_DATE = "2025-03-13"
END_DATE = "2026-04-17"

ENTRY_SLIP_TICKS = 0
SL_SLIP_TICKS = 2
TP_SLIP_TICKS = 0

MC_ITERATIONS = 1000
BOOTSTRAP_ITERATIONS = 10000
PERM_ITERATIONS = 1000

np.random.seed(42)


def to_et(ct):
    return ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_5min_bars_fixed(date_str):
    bar_file = TIMEBARS_DIR / f"timebars_5min_{date_str.replace('-','_')}.pkl"
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


# ─── BASE STRATEGY HELPERS ───

def get_vwap_at_time(vwap_df, bar_time):
    if vwap_df is None or vwap_df.empty:
        return None
    prior = vwap_df[vwap_df.index <= bar_time]
    return float(prior['vwap'].iloc[-1]) if not prior.empty else None


def get_atr_at_time(bars_5min, signal_time):
    if bars_5min is None or bars_5min.empty:
        return None
    prior = bars_5min[bars_5min.index <= signal_time]
    return float(prior['atr'].iloc[-1]) if not prior.empty else None


def get_5min_bias_zone(bars_5min, vwap_df, as_of_time):
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


def detect_base_signals(bars, vwap_df, bars_5min, level_delta_min=50):
    signals = []
    for i in range(len(bars) - 1):
        sb = bars[i]
        cb = bars[i + 1]
        if not sb.closed or not cb.closed:
            continue
        bt = sb.close_time
        bt_et = to_et(bt)
        hm = bt_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10":
            continue
        closed_bearish = sb.close < sb.open
        closed_bullish = sb.close > sb.open
        if not closed_bearish and not closed_bullish:
            continue
        has_bear_abs = has_bull_abs = False
        for price, lv in sb.levels.items():
            delta = lv.delta
            if closed_bearish and delta >= level_delta_min:
                has_bear_abs = True; break
            elif closed_bullish and delta <= -level_delta_min:
                has_bull_abs = True; break
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


# ─── TRENDING STRATEGY HELPERS ───

def precompute_day_state(bars_5min, vwap_df):
    if bars_5min is None or vwap_df is None or bars_5min.empty or vwap_df.empty:
        return None
    vwap_cols = ['vwap', f'std{STD_BAND}_upper', f'std{STD_BAND}_lower']
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
    merged['all_above'] = merged['above_int'].rolling(LOOKBACK, min_periods=LOOKBACK).min() == 1
    merged['all_below'] = merged['below_int'].rolling(LOOKBACK, min_periods=LOOKBACK).min() == 1
    merged['trend'] = None
    merged.loc[merged['all_above'], 'trend'] = 'up'
    merged.loc[merged['all_below'], 'trend'] = 'down'
    merged = merged.rename(columns={
        f'std{STD_BAND}_upper': 'band_upper',
        f'std{STD_BAND}_lower': 'band_lower',
    })
    trend_run_id = (merged['trend'] != merged['trend'].shift()).cumsum()
    hit_upper = (merged['trend'] == 'up') & (merged['close'] >= merged['band_upper'])
    hit_lower = (merged['trend'] == 'down') & (merged['close'] <= merged['band_lower'])
    merged['approach_from_above'] = hit_upper.groupby(trend_run_id).cummax().astype(bool)
    merged['approach_from_below'] = hit_lower.groupby(trend_run_id).cummax().astype(bool)
    merged.loc[merged['trend'].isna(), ['approach_from_above', 'approach_from_below']] = False
    return merged[['close', 'atr', 'vwap', 'band_upper', 'band_lower', 'trend',
                   'approach_from_above', 'approach_from_below']]


def detect_trending_signals(bars, day_state, min_delta=30):
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
        bt_et = to_et(bt)
        hm = bt_et.strftime("%H:%M")
        if "17:00" <= hm < "19:00":
            continue
        closed_bearish = sb.close < sb.open
        closed_bullish = sb.close > sb.open
        has_bear = has_bull = False
        for price, lv in sb.levels.items():
            if closed_bearish and lv.delta >= min_delta:
                has_bear = True
            elif closed_bullish and lv.delta <= -min_delta:
                has_bull = True
        if not has_bull and not has_bear:
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
            if not has_bull or cb.close <= cb.open:
                continue
            if last_5m_close < zl:
                continue
            if not approach_above:
                continue
            signals.append({"bar_index": i, "direction": "long", "confirm_time": cb.close_time,
                            "entry_price": cb.close, "atr": float(atr)})
        elif trend == "down":
            band = state[4]
            zh, zl = band + BAND_ZONE_BELOW, band - BAND_ZONE_ABOVE
            if not (sb.high >= zl and sb.low <= zh):
                continue
            cs = lookup(cb.close_time)
            cb_val = cs[4] if cs is not None else band
            if not (cb.high >= cb_val - BAND_ZONE_ABOVE and cb.low <= cb_val + BAND_ZONE_BELOW):
                continue
            if not has_bear or cb.close >= cb.open:
                continue
            if last_5m_close > zh:
                continue
            if not approach_below:
                continue
            signals.append({"bar_index": i, "direction": "short", "confirm_time": cb.close_time,
                            "entry_price": cb.close, "atr": float(atr)})
    return signals


# ─── SIMULATION (returns trade dicts with pnl + direction) ───

def simulate_trades(bars, sigs, sl_mult, tp_mult, skip_lunch=True):
    """Returns list of dicts: {pnl, direction, entry_price, atr, confirm_time, bar_index}"""
    sigs = sorted(sigs, key=lambda x: x["confirm_time"])
    trades, last_exit = [], None
    for sig in sigs:
        ct_et = to_et(sig["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        if skip_lunch and LUNCH_START <= hm < LUNCH_END:
            continue
        if "16:00" <= hm < "19:10":
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
                    ex_price, ex_time = sl - SL_SLIP_TICKS * TICK_SIZE, b.close_time; break
                if b.high >= tp:
                    ex_price, ex_time = tp - TP_SLIP_TICKS * TICK_SIZE, b.close_time; break
            else:
                if b.high >= sl:
                    ex_price, ex_time = sl + SL_SLIP_TICKS * TICK_SIZE, b.close_time; break
                if b.low <= tp:
                    ex_price, ex_time = tp + TP_SLIP_TICKS * TICK_SIZE, b.close_time; break
        if ex_price is None:
            ex_price, ex_time = bars[-1].close, bars[-1].close_time
        pnl = ((ex_price - ep) if d == "long" else (ep - ex_price)) * POINT_VALUE
        trades.append({"pnl": pnl, "direction": d, "entry_price": ep, "atr": atr,
                       "confirm_time": sig["confirm_time"], "bar_index": sig["bar_index"]})
        last_exit = ex_time
    return trades


# ─── SIMULATE WITH FLIPPED DIRECTION (for permutation test) ───

def simulate_with_directions(bars, sigs, sl_mult, tp_mult, directions, skip_lunch=True):
    """Like simulate_trades but uses externally provided directions list."""
    sigs = sorted(sigs, key=lambda x: x["confirm_time"])
    trades, last_exit = [], None
    dir_idx = 0
    for sig in sigs:
        ct_et = to_et(sig["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        if skip_lunch and LUNCH_START <= hm < LUNCH_END:
            dir_idx += 1
            continue
        if "16:00" <= hm < "19:10":
            dir_idx += 1
            continue
        if last_exit is not None and sig["confirm_time"] <= last_exit:
            dir_idx += 1
            continue
        d = directions[dir_idx]
        dir_idx += 1
        ep, atr = sig["entry_price"], sig["atr"]
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
                ex_price, ex_time = b.close, b.close_time; break
            if d == "long":
                if b.low <= sl:
                    ex_price, ex_time = sl - SL_SLIP_TICKS * TICK_SIZE, b.close_time; break
                if b.high >= tp:
                    ex_price, ex_time = tp - TP_SLIP_TICKS * TICK_SIZE, b.close_time; break
            else:
                if b.high >= sl:
                    ex_price, ex_time = sl + SL_SLIP_TICKS * TICK_SIZE, b.close_time; break
                if b.low <= tp:
                    ex_price, ex_time = tp + TP_SLIP_TICKS * TICK_SIZE, b.close_time; break
        if ex_price is None:
            ex_price, ex_time = bars[-1].close, bars[-1].close_time
        pnl = ((ex_price - ep) if d == "long" else (ep - ex_price)) * POINT_VALUE
        trades.append(pnl)
        last_exit = ex_time
    return trades


# ─── THE 5 TESTS ───

def test1_parameter_stability(all_results, sl, tp, step=0.25):
    """Check if all 8 neighboring SL/TP cells are profitable."""
    neighbors = []
    for ds in [-step, 0, step]:
        for dt in [-step, 0, step]:
            if ds == 0 and dt == 0:
                continue
            ns, nt = round(sl + ds, 2), round(tp + dt, 2)
            if ns <= 0 or nt <= 0:
                continue
            key = (ns, nt)
            if key in all_results:
                neighbors.append((ns, nt, all_results[key]))
    profitable = sum(1 for _, _, r in neighbors if r["pnl"] > 0)
    total = len(neighbors)
    return {
        "pass": profitable >= total * 0.75,  # 75%+ neighbors profitable
        "profitable": profitable,
        "total": total,
        "details": [(s, t, r["pnl"], r["pf"]) for s, t, r in neighbors],
    }


def test2_walk_forward(day_signals_ordered, bars_by_date, sl, tp, n_windows=4, skip_lunch=True):
    """Rolling walk-forward: split into n_windows, each one is OOS."""
    dates = list(day_signals_ordered.keys())
    window_size = len(dates) // n_windows
    results = []
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else len(dates)
        window_dates = dates[start:end]
        pnls = []
        for fd in window_dates:
            if fd in day_signals_ordered and fd in bars_by_date:
                trades = simulate_trades(bars_by_date[fd], day_signals_ordered[fd], sl, tp, skip_lunch)
                pnls.extend([t["pnl"] for t in trades])
        total = sum(pnls) if pnls else 0
        results.append({"window": w + 1, "dates": f"{window_dates[0]}->{window_dates[-1]}",
                        "n": len(pnls), "pnl": total})
    profitable_windows = sum(1 for r in results if r["pnl"] > 0)
    return {
        "pass": profitable_windows >= 3,  # 3 of 4 windows profitable
        "profitable_windows": profitable_windows,
        "total_windows": n_windows,
        "windows": results,
    }


def test3_monte_carlo_shuffle(pnls, n_iter=MC_ITERATIONS):
    """Shuffle trade order, compare DD distribution."""
    arr = np.array(pnls)
    real_cum = arr.cumsum()
    real_dd = float((real_cum - np.maximum.accumulate(real_cum)).min())
    worse_count = 0
    for _ in range(n_iter):
        shuffled = np.random.permutation(arr)
        cum = shuffled.cumsum()
        dd = float((cum - np.maximum.accumulate(cum)).min())
        if dd <= real_dd:
            worse_count += 1
    percentile = worse_count / n_iter * 100
    return {
        "pass": percentile >= 70,
        "real_dd": real_dd,
        "percentile": percentile,  # % of random orderings with worse or equal DD
    }


def test4_bootstrap_ci(pnls, n_iter=BOOTSTRAP_ITERATIONS):
    """Bootstrap resample trades, compute PnL distribution."""
    arr = np.array(pnls)
    n = len(arr)
    boot_pnls = np.zeros(n_iter)
    boot_pfs = np.zeros(n_iter)
    for i in range(n_iter):
        sample = arr[np.random.randint(0, n, n)]
        boot_pnls[i] = sample.sum()
        wins = sample[sample > 0].sum()
        losses = abs(sample[sample < 0].sum())
        boot_pfs[i] = wins / losses if losses > 0 else 999
    p_losing = np.mean(boot_pnls < 0) * 100
    ci_low = np.percentile(boot_pnls, 2.5)
    ci_high = np.percentile(boot_pnls, 97.5)
    median_pf = np.median(boot_pfs)
    return {
        "pass": p_losing < 1 and ci_low > 0,
        "p_losing": p_losing,
        "ci_95": (ci_low, ci_high),
        "median_pf": median_pf,
    }


def test5_direction_permutation(day_signals_ordered, bars_by_date, sl, tp,
                                 real_pnl, n_iter=PERM_ITERATIONS, skip_lunch=True):
    """Randomize direction, keep everything else. Compare real PnL."""
    # Collect all signals in order with their original directions
    all_sigs_flat = []
    date_order = []
    for fd in sorted(day_signals_ordered.keys()):
        sigs = sorted(day_signals_ordered[fd], key=lambda x: x["confirm_time"])
        for s in sigs:
            all_sigs_flat.append(s)
            date_order.append(fd)

    perm_pnls = np.zeros(n_iter)
    for i in range(n_iter):
        total = 0
        for fd in sorted(day_signals_ordered.keys()):
            if fd not in bars_by_date:
                continue
            sigs = sorted(day_signals_ordered[fd], key=lambda x: x["confirm_time"])
            rand_dirs = [np.random.choice(["long", "short"]) for _ in sigs]
            day_pnls = simulate_with_directions(bars_by_date[fd], sigs, sl, tp, rand_dirs, skip_lunch)
            total += sum(day_pnls)
        perm_pnls[i] = total

    beats = np.sum(real_pnl > perm_pnls)
    p_value = 1 - beats / n_iter
    return {
        "pass": p_value < 0.01,
        "p_value": p_value,
        "real_pnl": real_pnl,
        "median_perm": float(np.median(perm_pnls)),
        "pct_beaten": beats / n_iter * 100,
    }


def run_all_tests(strategy_name, sl, tp, day_signals_ordered, bars_by_date,
                  all_grid_results, skip_lunch=True):
    """Run all 5 overfit tests and return results dict."""
    print(f"\n  --- {strategy_name} SL={sl:.2f} TP={tp:.2f} ---")

    # Collect all trades
    all_pnls = []
    for fd in sorted(day_signals_ordered.keys()):
        if fd not in bars_by_date:
            continue
        trades = simulate_trades(bars_by_date[fd], day_signals_ordered[fd], sl, tp, skip_lunch)
        all_pnls.extend([t["pnl"] for t in trades])

    if len(all_pnls) < 10:
        print(f"    SKIP: only {len(all_pnls)} trades")
        return None

    total_pnl = sum(all_pnls)
    print(f"    Total trades: {len(all_pnls)}, PnL: ${total_pnl:+,.0f}")

    # Test 1: Parameter Stability
    t1 = test1_parameter_stability(all_grid_results, sl, tp)
    status1 = "PASS" if t1["pass"] else "FAIL"
    print(f"    T1 Param Stability: {status1} ({t1['profitable']}/{t1['total']} neighbors profitable)")

    # Test 2: Walk-Forward
    t2 = test2_walk_forward(day_signals_ordered, bars_by_date, sl, tp, skip_lunch=skip_lunch)
    status2 = "PASS" if t2["pass"] else "FAIL"
    wf_detail = ", ".join(f"W{w['window']}:${w['pnl']:+,.0f}({w['n']}t)" for w in t2["windows"])
    print(f"    T2 Walk-Forward:    {status2} ({t2['profitable_windows']}/{t2['total_windows']} windows profitable) [{wf_detail}]")

    # Test 3: Monte Carlo Shuffle
    t3 = test3_monte_carlo_shuffle(all_pnls)
    status3 = "PASS" if t3["pass"] else "FAIL"
    print(f"    T3 MC Shuffle:      {status3} (real DD ${t3['real_dd']:+,.0f}, better than {t3['percentile']:.1f}% of random)")

    # Test 4: Bootstrap CI
    t4 = test4_bootstrap_ci(all_pnls)
    status4 = "PASS" if t4["pass"] else "FAIL"
    print(f"    T4 Bootstrap:       {status4} (P(losing)={t4['p_losing']:.1f}%, 95% CI=[${t4['ci_95'][0]:+,.0f}, ${t4['ci_95'][1]:+,.0f}], median PF={t4['median_pf']:.2f})")

    # Test 5: Direction Permutation
    t5 = test5_direction_permutation(day_signals_ordered, bars_by_date, sl, tp, total_pnl, skip_lunch=skip_lunch)
    status5 = "PASS" if t5["pass"] else "FAIL"
    print(f"    T5 Dir Permutation: {status5} (p={t5['p_value']:.4f}, beats {t5['pct_beaten']:.1f}% of random, median perm=${t5['median_perm']:+,.0f})")

    passes = sum([t1["pass"], t2["pass"], t3["pass"], t4["pass"], t5["pass"]])
    print(f"    RESULT: {passes}/5 tests passed")

    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5, "passes": passes,
            "sl": sl, "tp": tp, "n_trades": len(all_pnls), "total_pnl": total_pnl}


def main():
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    # ─── LOAD BASE DATA ───
    print("=" * 100)
    print("  LOADING BASE STRATEGY DATA (level_delta >= 50, no ADX)")
    print("=" * 100)

    all_dates = []
    base_day_data = {}
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
            bars = load_pickle(sig_file).get("bars") or []
            vwap_df = load_pickle(vwap_file)
            bars_5min = load_5min_bars_fixed(ds)
            if bars and bars_5min is not None:
                base_day_data[fd] = (bars, vwap_df, bars_5min)

    print(f"Loaded {len(base_day_data)} days for base")

    # Detect base signals
    base_signals = {}  # fd -> list of signals
    base_bars = {}     # fd -> bars
    for fd, (bars, vwap_df, bars_5min) in base_day_data.items():
        sigs = detect_base_signals(bars, vwap_df, bars_5min)
        if sigs:
            base_signals[fd] = sigs
            base_bars[fd] = bars

    print(f"Base: {sum(len(s) for s in base_signals.values())} signals on {len(base_signals)} days")

    # Build grid results for parameter stability (use full dataset)
    print("Building base grid for parameter stability...")
    base_grid = {}
    mults = [round(0.5 + i * 0.25, 2) for i in range(11)]
    for sl_m in mults:
        for tp_m in mults:
            all_pnls = []
            for fd in sorted(base_signals.keys()):
                trades = simulate_trades(base_bars[fd], base_signals[fd], sl_m, tp_m)
                all_pnls.extend([t["pnl"] for t in trades])
            total = sum(all_pnls) if all_pnls else 0
            wins = [p for p in all_pnls if p > 0]
            losses = [p for p in all_pnls if p < 0]
            gp = sum(wins) if wins else 0
            gl = abs(sum(losses)) if losses else 1
            pf = gp / gl if gl > 0 else 999
            base_grid[(sl_m, tp_m)] = {"pnl": total, "pf": pf, "n": len(all_pnls)}

    # Base top 10
    base_top10 = [
        (0.50, 1.50), (0.75, 1.50), (1.00, 1.50), (0.50, 1.75), (0.50, 1.25),
        (1.00, 1.25), (0.75, 1.25), (1.25, 1.50), (0.50, 2.00), (0.75, 1.00),
    ]

    print("\n" + "=" * 100)
    print("  BASE STRATEGY OVERFIT TESTS (level_delta >= 50, no ADX)")
    print("=" * 100)

    base_results = []
    for sl, tp in base_top10:
        r = run_all_tests("BASE", sl, tp, base_signals, base_bars, base_grid, skip_lunch=True)
        if r:
            base_results.append(r)

    # ─── LOAD TRENDING DATA ───
    print("\n\n" + "=" * 100)
    print("  LOADING TRENDING STRATEGY DATA (delta >= 30, ADX >= 30)")
    print("=" * 100)

    print("Building ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done ({len(adx_lookup)} rows)")

    trend_signals = {}
    trend_bars = {}
    for fd, (bars, vwap_df, bars_5min) in base_day_data.items():
        day_state = precompute_day_state(bars_5min, vwap_df)
        sigs = detect_trending_signals(bars, day_state, min_delta=30)
        filtered = []
        for s in sigs:
            ct = s["confirm_time"]
            ct_et = to_et(ct)
            adx_val = get_adx_at_time(ct_et, adx_lookup)
            if not np.isnan(adx_val) and adx_val >= 30.0:
                filtered.append(s)
        if filtered:
            trend_signals[fd] = filtered
            trend_bars[fd] = bars

    print(f"Trending: {sum(len(s) for s in trend_signals.values())} signals on {len(trend_signals)} days")

    # Build trending grid for parameter stability
    print("Building trending grid for parameter stability...")
    trend_grid = {}
    for sl_m in mults:
        for tp_m in mults:
            all_pnls = []
            for fd in sorted(trend_signals.keys()):
                trades = simulate_trades(trend_bars[fd], trend_signals[fd], sl_m, tp_m, skip_lunch=False)
                all_pnls.extend([t["pnl"] for t in trades])
            total = sum(all_pnls) if all_pnls else 0
            wins = [p for p in all_pnls if p > 0]
            losses = [p for p in all_pnls if p < 0]
            gp = sum(wins) if wins else 0
            gl = abs(sum(losses)) if losses else 1
            pf = gp / gl if gl > 0 else 999
            trend_grid[(sl_m, tp_m)] = {"pnl": total, "pf": pf, "n": len(all_pnls)}

    # Trending top 10
    trend_top10 = [
        (0.50, 3.00), (0.75, 3.00), (0.50, 2.75), (0.75, 2.75), (2.00, 3.00),
        (0.50, 2.50), (0.75, 2.50), (1.25, 3.00), (2.25, 3.00), (2.00, 2.75),
    ]

    print("\n" + "=" * 100)
    print("  TRENDING STRATEGY OVERFIT TESTS (delta >= 30, ADX >= 30)")
    print("=" * 100)

    trend_results = []
    for sl, tp in trend_top10:
        r = run_all_tests("TREND", sl, tp, trend_signals, trend_bars, trend_grid, skip_lunch=False)
        if r:
            trend_results.append(r)

    # ─── FINAL SUMMARY ───
    print("\n\n" + "=" * 100)
    print("  FINAL SUMMARY")
    print("=" * 100)

    print(f"\n  {'Strategy':>10} {'SL':>5} {'TP':>5} {'Trades':>7} {'PnL':>10}"
          f" | {'T1':>4} {'T2':>4} {'T3':>4} {'T4':>4} {'T5':>4} | {'Score':>5} {'Verdict':>10}")
    print("  " + "-" * 90)

    all_results = []
    for r in base_results:
        all_results.append(("BASE", r))
    for r in trend_results:
        all_results.append(("TREND", r))

    all_results.sort(key=lambda x: x[1]["passes"], reverse=True)

    for strat, r in all_results:
        t1 = "P" if r["t1"]["pass"] else "F"
        t2 = "P" if r["t2"]["pass"] else "F"
        t3 = "P" if r["t3"]["pass"] else "F"
        t4 = "P" if r["t4"]["pass"] else "F"
        t5 = "P" if r["t5"]["pass"] else "F"
        verdict = "SURVIVES" if r["passes"] >= 4 else "MARGINAL" if r["passes"] == 3 else "REJECT"
        print(f"  {strat:>10} {r['sl']:>5.2f} {r['tp']:>5.2f} {r['n_trades']:>7} ${r['total_pnl']:>+9,.0f}"
              f" |   {t1}    {t2}    {t3}    {t4}    {t5} | {r['passes']:>5}/5 {verdict:>10}")


if __name__ == "__main__":
    main()

"""
5-year VWAP Combined Strategy Backtest.

Uses vwap_reaction_cache_5yr + signal_cache_5yr + vwap_cache_5yr + timebars_5min_5yr
with Databento caches for Apr 2025+ overlap.

Replicates montecarlo_combined.py logic but on 5yr data, outputs trade stats + yearly breakdown.

Usage:
    python -u scripts/vwap_reaction_strat_backtest/backtest_5yr_combined.py
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")

# 5yr caches (MarketTick Dec 2020 - Mar 2025)
MT_SIGNAL_DIR = DATA_DIR / "signal_cache_5yr"
MT_VWAP_DIR = DATA_DIR / "vwap_cache_5yr"
MT_TIMEBARS_DIR = DATA_DIR / "timebars_5min_5yr"
VWAP_REACTION_5YR_DIR = DATA_DIR / "vwap_reaction_cache_5yr"

# Databento caches (Mar 2025 - Apr 2026)
DB_SIGNAL_DIR = DATA_DIR / "signal_cache"
DB_VWAP_DIR = DATA_DIR / "vwap_cache"
DB_TIMEBARS_DIR = DATA_DIR / "timebars_5min"

DB_PRIORITY_FROM = "2025-04-01"

POINT_VALUE = 20.0
MNQ_PV = 2.0

ENTRY_CUTOFF = "16:00"
ENTRY_START = "19:10"
FORCE_CLOSE = "16:58"
LUNCH_START = "12:00"
LUNCH_END = "13:00"

# Strategy configs — matches the previously "locked" combined config:
#   Base:  lvl_delta ≥ 50, SL 0.5x / TP 1.5x @ VWAP, ADX 15-30
#   Trend: lvl_delta ≥ 30, SL 0.75x / TP 2.5x @ STD1, ADX ≥ 30
BASE_SL = 0.50
BASE_TP = 1.50
TREND_SL = 0.75
TREND_TP = 2.50
TREND_STD = 1
TREND_LB = 14
TREND_ADX_MIN = 30.0

ADX_PERIOD = 14
ADX_LOW = 15.0
ADX_HIGH = 30.0

ZONE_POINTS = 3.0
BAND_ZONE_ABOVE = 3.0
BAND_ZONE_BELOW = 4.0
MIN_DELTA = 30           # trend absorption threshold (matches cached signals)
BASE_MIN_DELTA = 50      # post-filter base signals to the stronger delta
                         # (cache was built at min_delta=30; we filter up here)


def find_file(date_str, kind):
    """Find best available file for a date."""
    use_db = date_str >= DB_PRIORITY_FROM
    if kind == "signal":
        fname = f"{date_str}.pkl"
        dirs = ([DB_SIGNAL_DIR, MT_SIGNAL_DIR] if use_db else [MT_SIGNAL_DIR, DB_SIGNAL_DIR])
    elif kind == "vwap":
        fname = f"{date_str}.pkl"
        dirs = ([DB_VWAP_DIR, MT_VWAP_DIR] if use_db else [MT_VWAP_DIR, DB_VWAP_DIR])
    elif kind == "timebars":
        fname = f"timebars_5min_{date_str.replace('-', '_')}.pkl"
        dirs = ([DB_TIMEBARS_DIR, MT_TIMEBARS_DIR] if use_db else [MT_TIMEBARS_DIR, DB_TIMEBARS_DIR])
    else:
        return None
    for d in dirs:
        p = d / fname
        if p.exists():
            return p
    return None


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_5min_bars(date_str):
    path = find_file(date_str, "timebars")
    if path is None:
        return None
    bars = load_pickle(path)
    if not bars:
        return None
    rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
             "low": b["low"], "close": b["close"]} for b in bars]
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df.index = df.index + pd.Timedelta(minutes=5)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df


def compute_adx(df):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    up = high - high.shift()
    dn = low.shift() - low
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.ewm(alpha=1/ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1/ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1/ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/ADX_PERIOD, min_periods=ADX_PERIOD, adjust=False).mean()


def get_adx_at_time(entry_time, adx_lookup):
    prior = adx_lookup[adx_lookup.index <= entry_time]
    if len(prior) == 0:
        return np.nan
    return prior.iloc[-1]["adx"]


def has_absorption(bar):
    closed_bearish = bar.close < bar.open
    closed_bullish = bar.close > bar.open
    has_bull = has_bear = False
    for price, lv in bar.levels.items():
        delta = lv.delta
        if closed_bearish and delta >= MIN_DELTA:
            has_bear = True
        elif closed_bullish and delta <= -MIN_DELTA:
            has_bull = True
    return has_bull, has_bear


def precompute_day_state(bars_5min, vwap_df):
    if bars_5min is None or vwap_df is None or bars_5min.empty or vwap_df.empty:
        return None
    vwap_cols = ['vwap', f'std{TREND_STD}_upper', f'std{TREND_STD}_lower']
    vwap_series = vwap_df[vwap_cols].copy().sort_index()
    bs = bars_5min[['close']].copy()
    # Compute ATR from 5min bars
    df5 = bars_5min.copy()
    df5['prev_close'] = df5['close'].shift(1)
    df5['tr'] = np.maximum(df5['high'] - df5['low'],
                           np.maximum(abs(df5['high'] - df5['prev_close']),
                                      abs(df5['low'] - df5['prev_close'])))
    df5['atr'] = df5['tr'].rolling(14, min_periods=1).mean()
    bs['atr'] = df5['atr']

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
    merged['all_above'] = merged['above_int'].rolling(TREND_LB, min_periods=TREND_LB).min() == 1
    merged['all_below'] = merged['below_int'].rolling(TREND_LB, min_periods=TREND_LB).min() == 1
    merged['trend'] = None
    merged.loc[merged['all_above'], 'trend'] = 'up'
    merged.loc[merged['all_below'], 'trend'] = 'down'

    merged = merged.rename(columns={
        f'std{TREND_STD}_upper': 'band_upper',
        f'std{TREND_STD}_lower': 'band_lower',
    })

    trend_run_id = (merged['trend'] != merged['trend'].shift()).cumsum()
    hit_upper = (merged['trend'] == 'up') & (merged['close'] >= merged['band_upper'])
    hit_lower = (merged['trend'] == 'down') & (merged['close'] <= merged['band_lower'])
    merged['approach_from_above'] = hit_upper.groupby(trend_run_id).cummax().astype(bool)
    merged['approach_from_below'] = hit_lower.groupby(trend_run_id).cummax().astype(bool)
    merged.loc[merged['trend'].isna(), ['approach_from_above', 'approach_from_below']] = False

    return merged[['close', 'atr', 'vwap', 'band_upper', 'band_lower', 'trend',
                   'approach_from_above', 'approach_from_below']]


def detect_trending_signals(bars, day_state):
    if day_state is None or day_state.empty:
        return []
    signals = []
    si = day_state.index
    sv = day_state.values

    def lookup(ts):
        idx = si.searchsorted(ts, side='right') - 1
        return sv[idx] if idx >= 0 else None

    for i in range(len(bars) - 1):
        sb = bars[i]; cb = bars[i + 1]
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
            if not bool(state[6]):
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
            if not is_bear or cb.close >= cb.open:
                continue
            if last_5m_close > zh:
                continue
            if not bool(state[7]):
                continue
            signals.append({"bar_index": i, "direction": "short", "confirm_time": cb.close_time,
                            "entry_price": cb.close, "atr": float(atr)})
    return signals


def main():
    # Collect all dates from vwap_reaction_cache_5yr
    dates = sorted(f.stem for f in VWAP_REACTION_5YR_DIR.glob("*.pkl"))
    print(f"VWAP reaction cache: {len(dates)} dates ({dates[0]} to {dates[-1]})")

    # Build ADX lookup from all 5min bars
    print("Building ADX lookup...", flush=True)
    all_adx = []
    for i, d in enumerate(dates):
        df_5m = load_5min_bars(d)
        if df_5m is None or len(df_5m) <= ADX_PERIOD * 2:
            continue
        adx_s = compute_adx(df_5m)
        all_adx.append(pd.DataFrame({"adx": adx_s}, index=df_5m.index))
        if (i + 1) % 200 == 0:
            print(f"  ADX: {i+1}/{len(dates)}", flush=True)
    adx_lookup = pd.concat(all_adx).sort_index() if all_adx else pd.DataFrame(columns=["adx"])
    print(f"  ADX lookup: {len(adx_lookup)} rows", flush=True)

    # Process each date
    print("Running backtest...", flush=True)
    all_trades = []  # (date_str, pnl_pts, mae_pts, direction, source)

    for di, date_str in enumerate(dates):
        # Load range bars
        sig_file = find_file(date_str, "signal")
        if sig_file is None:
            continue
        signal_data = load_pickle(sig_file)
        bars = signal_data["bars"]
        if not bars:
            continue

        # Base VWAP signals
        vwap_reaction_file = VWAP_REACTION_5YR_DIR / f"{date_str}.pkl"
        vwap_data = load_pickle(vwap_reaction_file)
        base_signals = vwap_data["signals"]

        # Trending band signals
        vwap_file = find_file(date_str, "vwap")
        bars_5min = load_5min_bars(date_str)
        trend_signals = []

        if vwap_file is not None and bars_5min is not None:
            vwap_df = load_pickle(vwap_file)
            day_state = precompute_day_state(bars_5min, vwap_df)
            if day_state is not None:
                trend_signals = detect_trending_signals(bars, day_state)

        # Build unified signal list
        all_signals = []

        for s in base_signals:
            ct = s["confirm_time"]
            ct_et = ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)
            hm = ct_et.strftime("%H:%M")
            if "16:00" <= hm < "19:10":
                continue
            if LUNCH_START <= hm < LUNCH_END:
                continue
            # Post-filter: require the signal bar's strongest absorption level to
            # meet BASE_MIN_DELTA (the cache was built at min_delta=30; we want 50).
            abs_levels = s.get("absorption_levels") or []
            if not abs_levels or max(abs(d) for _, d in abs_levels) < BASE_MIN_DELTA:
                continue
            atr = s["atr"]
            if atr is None or atr <= 0:
                continue
            adx_val = get_adx_at_time(ct_et, adx_lookup)
            if np.isnan(adx_val) or not (ADX_LOW <= adx_val < ADX_HIGH):
                continue
            ep = s["entry_price"]
            d = s["direction"]
            if d == "long":
                sl = ep - atr * BASE_SL; tp = ep + atr * BASE_TP
                if tp <= ep or sl >= ep: continue
            else:
                sl = ep + atr * BASE_SL; tp = ep - atr * BASE_TP
                if tp >= ep or sl <= ep: continue
            all_signals.append({
                "source": "base", "bar_index": s["bar_index"],
                "confirm_time": ct, "confirm_time_et": ct_et,
                "entry_price": ep, "direction": d, "sl": sl, "tp": tp,
            })

        for s in trend_signals:
            ct = s["confirm_time"]
            ct_et = ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)
            hm = ct_et.strftime("%H:%M")
            if "16:00" <= hm < "19:10":
                continue
            if LUNCH_START <= hm < LUNCH_END:
                continue
            atr = s["atr"]
            if atr is None or atr <= 0:
                continue
            adx_val = get_adx_at_time(ct_et, adx_lookup)
            if np.isnan(adx_val) or adx_val < TREND_ADX_MIN:
                continue
            ep = s["entry_price"]
            d = s["direction"]
            if d == "long":
                sl = ep - atr * TREND_SL; tp = ep + atr * TREND_TP
            else:
                sl = ep + atr * TREND_SL; tp = ep - atr * TREND_TP
            all_signals.append({
                "source": "trend", "bar_index": s["bar_index"],
                "confirm_time": ct, "confirm_time_et": ct_et,
                "entry_price": ep, "direction": d, "sl": sl, "tp": tp,
            })

        all_signals.sort(key=lambda x: x["confirm_time"])

        # Simulate with shared position lock
        last_exit_time = None
        for sig in all_signals:
            if last_exit_time is not None and sig["confirm_time"] <= last_exit_time:
                continue

            ep = sig["entry_price"]
            d = sig["direction"]
            sl_price = sig["sl"]
            tp_price = sig["tp"]
            ci = sig["bar_index"] + 1

            exit_price = None
            exit_time = None
            worst_adverse = 0.0

            for j in range(ci + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                if d == "long":
                    adverse = ep - bar.low
                else:
                    adverse = bar.high - ep
                if adverse > worst_adverse:
                    worst_adverse = adverse

                bct = bar.close_time
                bet = bct.tz_convert(ET) if hasattr(bct, "tz_convert") else pd.Timestamp(bct, tz="UTC").tz_convert(ET)
                if bet.strftime("%H:%M") >= FORCE_CLOSE:
                    exit_price = bar.close; exit_time = bct; break
                if d == "long":
                    if bar.low <= sl_price: exit_price = sl_price; exit_time = bct; break
                    if bar.high >= tp_price: exit_price = tp_price; exit_time = bct; break
                else:
                    if bar.high >= sl_price: exit_price = sl_price; exit_time = bct; break
                    if bar.low <= tp_price: exit_price = tp_price; exit_time = bct; break

            if exit_price is None:
                exit_price = bars[-1].close; exit_time = bars[-1].close_time

            pnl = (exit_price - ep) if d == "long" else (ep - exit_price)
            all_trades.append((date_str, pnl, worst_adverse, d, sig["source"]))
            last_exit_time = exit_time

        if (di + 1) % 200 == 0:
            print(f"  [{di+1}/{len(dates)}] {date_str}  trades={len(all_trades)}", flush=True)

    # ── Results ─────────────────────────────────────────────────────────────
    if not all_trades:
        print("No trades generated!")
        return

    dates_arr = np.array([t[0] for t in all_trades])
    pnls = np.array([t[1] for t in all_trades])
    maes = np.array([t[2] for t in all_trades])
    dirs = np.array([t[3] for t in all_trades])
    sources = np.array([t[4] for t in all_trades])

    w = pnls[pnls > 0]; l = pnls[pnls < 0]
    wr = 100 * len(w) / len(pnls)
    pf = w.sum() / abs(l.sum()) if len(l) else 99
    cum = np.cumsum(pnls) * POINT_VALUE
    dd = (cum - np.maximum.accumulate(cum)).min()
    total_pnl = pnls.sum() * POINT_VALUE

    print(f"\n{'='*80}")
    print(f"5-YEAR VWAP COMBINED STRATEGY RESULTS")
    print(f"Base: SL {BASE_SL}x / TP {BASE_TP}x (ADX {ADX_LOW}-{ADX_HIGH})")
    print(f"Trend: SL {TREND_SL}x / TP {TREND_TP}x (ADX {TREND_ADX_MIN}+)")
    print(f"{'='*80}")
    print(f"Trades: {len(pnls):,}  WR: {wr:.1f}%  PF: {pf:.2f}")
    print(f"Total PnL: ${total_pnl:+,.0f} (1 NQ)  Max DD: ${dd:,.0f}")
    print(f"Avg trade: ${pnls.mean()*POINT_VALUE:+,.0f}  Avg MAE: ${maes.mean()*POINT_VALUE:,.0f}")
    print(f"MNQ equiv: PnL ${total_pnl*0.1:+,.0f}  DD ${dd*0.1:,.0f}")

    # By source
    print(f"\nBy source:")
    for src in ["base", "trend"]:
        mask = sources == src
        sp = pnls[mask]
        if len(sp) == 0: continue
        sw = sp[sp > 0]; sl_a = sp[sp < 0]
        spf = sw.sum() / abs(sl_a.sum()) if len(sl_a) else 99
        swr = 100 * len(sw) / len(sp)
        print(f"  {src:>6}: {len(sp):>5} trades  WR={swr:.1f}%  PF={spf:.2f}  PnL=${sp.sum()*POINT_VALUE:+,.0f}")

    # By direction
    print(f"\nBy direction:")
    for dr in ["long", "short"]:
        mask = dirs == dr
        dp = pnls[mask]
        if len(dp) == 0: continue
        dw = dp[dp > 0]; dl = dp[dp < 0]
        dpf = dw.sum() / abs(dl.sum()) if len(dl) else 99
        dwr = 100 * len(dw) / len(dp)
        print(f"  {dr:>6}: {len(dp):>5} trades  WR={dwr:.1f}%  PF={dpf:.2f}  PnL=${dp.sum()*POINT_VALUE:+,.0f}")

    # Yearly breakdown
    years = sorted(set(d[:4] for d in dates_arr))
    print(f"\n{'Year':<6} {'Trades':>7} {'PF':>6} {'WR':>6} {'PnL (1NQ)':>12} {'DD':>10} {'Base':>5} {'Trend':>6}")
    print("-" * 65)
    for yr in years:
        mask = np.array([d[:4] == yr for d in dates_arr])
        yp = pnls[mask]; ys = sources[mask]
        if len(yp) == 0: continue
        yw = yp[yp > 0]; yl = yp[yp < 0]
        ypf = yw.sum() / abs(yl.sum()) if len(yl) else 99
        ywr = 100 * len(yw) / len(yp)
        ycum = yp.cumsum() * POINT_VALUE
        ydd = (ycum - np.maximum.accumulate(ycum)).min() if len(ycum) else 0
        nb = (ys == "base").sum(); nt = (ys == "trend").sum()
        print(f"{yr:<6} {len(yp):>7} {ypf:>6.2f} {ywr:>5.1f}% ${yp.sum()*POINT_VALUE:>+10,.0f} ${ydd:>9,.0f} {nb:>5} {nt:>6}")

    # Monthly
    print(f"\n{'Month':<8} {'Trades':>7} {'PF':>6} {'WR':>6} {'PnL':>12}")
    print("-" * 45)
    months = sorted(set(d[:7] for d in dates_arr))
    for ym in months:
        mask = np.array([d[:7] == ym for d in dates_arr])
        mp = pnls[mask]
        if len(mp) == 0: continue
        mw = mp[mp > 0]; ml = mp[mp < 0]
        mpf = mw.sum() / abs(ml.sum()) if len(ml) else 99
        mwr = 100 * len(mw) / len(mp)
        print(f"{ym:<8} {len(mp):>7} {mpf:>6.2f} {mwr:>5.1f}% ${mp.sum()*POINT_VALUE:>+10,.0f}")


if __name__ == "__main__":
    main()

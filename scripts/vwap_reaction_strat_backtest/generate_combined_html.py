"""
Generate combined equity curve + PnL calendar for:
  1. Base VWAP strat (lvl_delta>=50, no ADX, SL 0.50x, TP 1.50x) at VWAP zone
  2. Trending band strat (delta>=30, ADX>=30, STD1, LB=14, SL 0.75x, TP 2.50x) at std1 bands

Rules: one trade at a time across both strategies. Whichever fires first gets priority.
Signals are merged chronologically per day and simulated with shared position lock.
"""

import pickle
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time,
    LUNCH_START, LUNCH_END,
)

SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_DIR = DATA_DIR / "timebars_5min"
POINT_VALUE = 20.0
TICK_SIZE = 0.25
VWAP_ZONE_POINTS = 3.0
ATR_PERIOD = 14

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"

BAND_ZONE_ABOVE = 3.0
BAND_ZONE_BELOW = 4.0
STD_BAND = 1
LOOKBACK = 14

# Base VWAP strat config
BASE_SL = 0.50
BASE_TP = 1.50
BASE_LEVEL_DELTA = 50

# Trending band strat config
TREND_SL = 0.75
TREND_TP = 2.50
TREND_DELTA = 30
TREND_STD = 1
TREND_LB = 14
TREND_ADX_MIN = 30.0

ENTRY_SLIP_TICKS = 0
SL_SLIP_TICKS = 2
TP_SLIP_TICKS = 0

START_DATE = "2025-03-13"
END_DATE = "2026-04-17"


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


# ─── BASE SIGNAL DETECTION ───

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


def detect_base_signals(bars, vwap_df, bars_5min):
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
            if closed_bearish and lv.delta >= BASE_LEVEL_DELTA:
                has_bear_abs = True; break
            elif closed_bullish and lv.delta <= -BASE_LEVEL_DELTA:
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


# ─── TRENDING SIGNAL DETECTION ───

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
        f'std{STD_BAND}_upper': 'band_upper', f'std{STD_BAND}_lower': 'band_lower'})
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
            if closed_bearish and lv.delta >= TREND_DELTA:
                has_bear = True
            elif closed_bullish and lv.delta <= -TREND_DELTA:
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
            if not (sb.high >= zl and sb.low <= zh): continue
            cs = lookup(cb.close_time)
            cb_val = cs[3] if cs is not None else band
            if not (cb.high >= cb_val - BAND_ZONE_BELOW and cb.low <= cb_val + BAND_ZONE_ABOVE): continue
            if not has_bull or cb.close <= cb.open: continue
            if last_5m_close < zl: continue
            if not approach_above: continue
            signals.append({"bar_index": i, "direction": "long",
                            "confirm_time": cb.close_time, "entry_price": cb.close, "atr": float(atr)})
        elif trend == "down":
            band = state[4]
            zh, zl = band + BAND_ZONE_BELOW, band - BAND_ZONE_ABOVE
            if not (sb.high >= zl and sb.low <= zh): continue
            cs = lookup(cb.close_time)
            cb_val = cs[4] if cs is not None else band
            if not (cb.high >= cb_val - BAND_ZONE_ABOVE and cb.low <= cb_val + BAND_ZONE_BELOW): continue
            if not has_bear or cb.close >= cb.open: continue
            if last_5m_close > zh: continue
            if not approach_below: continue
            signals.append({"bar_index": i, "direction": "short",
                            "confirm_time": cb.close_time, "entry_price": cb.close, "atr": float(atr)})
    return signals


# ─── SIMULATION ───

def simulate_day_combined(bars, base_signals, trend_signals, adx_lookup):
    """
    Simulate one day with both strategies. Merge all signals chronologically.
    One trade at a time — whichever fires first locks position until exit.
    """
    all_signals = []

    for s in base_signals:
        ct = s["confirm_time"]
        ct_et = to_et(ct)
        hm = ct_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10":
            continue
        if LUNCH_START <= hm < LUNCH_END:
            continue
        atr = s["atr"]
        if atr is None or atr <= 0:
            continue
        # No ADX filter for base
        entry_price = s["entry_price"]
        direction = s["direction"]
        if direction == "long":
            sl = entry_price - atr * BASE_SL
            tp = entry_price + atr * BASE_TP
        else:
            sl = entry_price + atr * BASE_SL
            tp = entry_price - atr * BASE_TP
        all_signals.append({
            "source": "base", "bar_index": s["bar_index"],
            "confirm_time": ct, "confirm_time_et": ct_et,
            "entry_price": entry_price, "direction": direction,
            "sl": sl, "tp": tp, "atr": atr, "adx": 0,
        })

    for s in trend_signals:
        ct = s["confirm_time"]
        ct_et = to_et(ct)
        hm = ct_et.strftime("%H:%M")
        if "16:00" <= hm < "19:10":
            continue
        if LUNCH_START <= hm < LUNCH_END:
            continue
        atr = s["atr"]
        if atr is None or atr <= 0:
            continue
        # ADX >= 30 filter
        adx_val = get_adx_at_time(ct_et, adx_lookup)
        if np.isnan(adx_val) or adx_val < TREND_ADX_MIN:
            continue
        entry_price = s["entry_price"]
        direction = s["direction"]
        if direction == "long":
            sl = entry_price - atr * TREND_SL
            tp = entry_price + atr * TREND_TP
        else:
            sl = entry_price + atr * TREND_SL
            tp = entry_price - atr * TREND_TP
        all_signals.append({
            "source": "trend", "bar_index": s["bar_index"],
            "confirm_time": ct, "confirm_time_et": ct_et,
            "entry_price": entry_price, "direction": direction,
            "sl": sl, "tp": tp, "atr": atr, "adx": adx_val,
        })

    all_signals.sort(key=lambda x: x["confirm_time"])

    trades = []
    last_exit_time = None

    for sig in all_signals:
        if last_exit_time is not None and sig["confirm_time"] <= last_exit_time:
            continue

        confirm_bar_idx = sig["bar_index"] + 1
        direction = sig["direction"]
        entry_price = sig["entry_price"]
        sl_price = sig["sl"]
        tp_price = sig["tp"]

        exit_price = exit_time = exit_reason = None

        for j in range(confirm_bar_idx + 1, len(bars)):
            bar = bars[j]
            if not bar.closed:
                continue
            bar_et = to_et(bar.close_time)
            if bar_et.strftime("%H:%M") >= FORCE_CLOSE:
                exit_price, exit_time, exit_reason = bar.close, bar.close_time, "session_close"
                break
            if direction == "long":
                if bar.low <= sl_price:
                    exit_price = sl_price - SL_SLIP_TICKS * TICK_SIZE
                    exit_time, exit_reason = bar.close_time, "stop_loss"; break
                if bar.high >= tp_price:
                    exit_price = tp_price - TP_SLIP_TICKS * TICK_SIZE
                    exit_time, exit_reason = bar.close_time, "target"; break
            else:
                if bar.high >= sl_price:
                    exit_price = sl_price + SL_SLIP_TICKS * TICK_SIZE
                    exit_time, exit_reason = bar.close_time, "stop_loss"; break
                if bar.low <= tp_price:
                    exit_price = tp_price + TP_SLIP_TICKS * TICK_SIZE
                    exit_time, exit_reason = bar.close_time, "target"; break

        if exit_price is None:
            exit_price, exit_time, exit_reason = bars[-1].close, bars[-1].close_time, "eod"

        pnl_pts = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        pnl_dollars = pnl_pts * POINT_VALUE

        exit_et = to_et(exit_time)
        duration = (exit_et - sig["confirm_time_et"]).total_seconds() / 60

        trades.append({
            "source": sig["source"],
            "entry_time": sig["confirm_time_et"].strftime("%Y-%m-%d %H:%M"),
            "entry_price": entry_price, "exit_price": exit_price,
            "exit_reason": exit_reason, "pnl_points": pnl_pts,
            "pnl_dollars": pnl_dollars, "duration_mins": duration,
            "direction": direction, "atr": sig["atr"], "adx": sig["adx"],
        })
        last_exit_time = exit_time

    return trades


def run_combined_backtest(adx_lookup):
    """Run combined backtest across all dates."""
    start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    all_trades = []

    for sig_file in sorted(SIGNAL_CACHE_DIR.glob("*.pkl")):
        date_str = sig_file.stem
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < start or file_date > end:
            continue

        vwap_file = VWAP_CACHE_DIR / f"{date_str}.pkl"
        tb_file = TIMEBARS_DIR / f"timebars_5min_{date_str.replace('-','_')}.pkl"
        if not vwap_file.exists() or not tb_file.exists():
            continue

        # Load range bars
        signal_data = load_pickle(sig_file)
        bars = signal_data.get("bars") or []
        if not bars:
            continue

        vwap_df = load_pickle(vwap_file)
        bars_5min = load_5min_bars_fixed(date_str)
        if bars_5min is None:
            continue

        # Base signals (lvl_delta>=50, no ADX)
        base_signals = detect_base_signals(bars, vwap_df, bars_5min)

        # Trending signals (delta>=30, ADX>=30 checked in simulate)
        day_state = precompute_day_state(bars_5min, vwap_df)
        trend_signals = detect_trending_signals(bars, day_state)

        # Simulate combined
        day_trades = simulate_day_combined(bars, base_signals, trend_signals, adx_lookup)
        for t in day_trades:
            t["date"] = date_str
        all_trades.extend(day_trades)

    return all_trades


def generate_equity_html(trades, output_path):
    """Generate equity curve HTML."""
    if not trades:
        print("No trades."); return

    df = pd.DataFrame(trades)
    total = len(df)
    winners = df[df["pnl_dollars"] > 0]
    losers = df[df["pnl_dollars"] < 0]

    win_rate = len(winners) / total * 100
    avg_win = winners["pnl_dollars"].mean() if len(winners) else 0
    avg_loss = losers["pnl_dollars"].mean() if len(losers) else 0
    avg_pnl = df["pnl_dollars"].mean()
    total_pnl = df["pnl_dollars"].sum()

    gp = winners["pnl_dollars"].sum() if len(winners) else 0
    gl = abs(losers["pnl_dollars"].sum()) if len(losers) else 1
    pf = gp / gl if gl > 0 else 999

    cum = df["pnl_dollars"].cumsum()
    max_dd = float((cum - cum.cummax()).min())
    risk_adj = abs(total_pnl / max_dd) if max_dd != 0 else 0

    avg_dur = df["duration_mins"].mean()
    med_dur = df["duration_mins"].median()

    df["date_parsed"] = pd.to_datetime(df["entry_time"]).dt.date
    daily_pnl = df.groupby("date_parsed")["pnl_dollars"].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0

    tp_exits = df[df["exit_reason"] == "target"]
    sl_exits = df[df["exit_reason"] == "stop_loss"]
    eod_exits = df[df["exit_reason"].isin(["eod", "session_close"])]

    shorts = df[df["direction"] == "short"]
    longs = df[df["direction"] == "long"]

    base_trades = df[df["source"] == "base"]
    trend_trades = df[df["source"] == "trend"]

    # IS/OOS
    sp = total // 2
    is_df = df.iloc[:sp]; oos_df = df.iloc[sp:]
    is_pnl = is_df["pnl_dollars"].sum(); oos_pnl = oos_df["pnl_dollars"].sum()
    is_w = is_df[is_df["pnl_dollars"] > 0]; is_l = is_df[is_df["pnl_dollars"] <= 0]
    oos_w = oos_df[oos_df["pnl_dollars"] > 0]; oos_l = oos_df[oos_df["pnl_dollars"] <= 0]
    is_pf = is_w["pnl_dollars"].sum() / abs(is_l["pnl_dollars"].sum()) if len(is_l) else 999
    oos_pf = oos_w["pnl_dollars"].sum() / abs(oos_l["pnl_dollars"].sum()) if len(oos_l) else 999
    is_wr = len(is_w) / len(is_df) * 100; oos_wr = len(oos_w) / len(oos_df) * 100

    # Buy & hold
    first_price = last_price = None
    bar_files = sorted(TIMEBARS_DIR.glob("timebars_5min_*.pkl"))
    if bar_files:
        first_bars = pickle.load(open(bar_files[0], "rb"))
        last_bars = pickle.load(open(bar_files[-1], "rb"))
        first_price = first_bars[0]["open"] if first_bars else 0
        last_price = last_bars[-1]["close"] if last_bars else 0

    nq_prices = {}
    for td in sorted(set(df["date"])):
        bf = TIMEBARS_DIR / f"timebars_5min_{td.replace('-', '_')}.pkl"
        if bf.exists():
            b5 = pickle.load(open(bf, "rb"))
            if b5: nq_prices[td] = b5[-1]["close"]

    bh_equity = []
    if nq_prices and first_price:
        for d in df["date"]:
            if d in nq_prices: bh_equity.append(round((nq_prices[d] - first_price) * POINT_VALUE, 2))
            elif bh_equity: bh_equity.append(bh_equity[-1])
            else: bh_equity.append(0)
        bh_pnl = bh_equity[-1] if bh_equity else 0
    else:
        bh_equity = [0] * total; bh_pnl = 0
        first_price = first_price or 0; last_price = last_price or 0

    # Monthly
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    monthly = df.groupby("month")["pnl_dollars"].sum()
    monthly_labels = [str(m) for m in monthly.index]
    monthly_values = [round(v, 2) for v in monthly.values]

    strategy_equity = cum.tolist()
    dates = df["entry_time"].tolist()
    dd_series = (cum - cum.cummax()).tolist()

    # Source-split equity curves
    base_cum = []
    trend_cum = []
    b_run = 0; t_run = 0
    for _, row in df.iterrows():
        if row["source"] == "base":
            b_run += row["pnl_dollars"]
        else:
            t_run += row["pnl_dollars"]
        base_cum.append(round(b_run, 2))
        trend_cum.append(round(t_run, 2))

    def fmt(v): return f"${v:,.0f}"
    def cls(v): return "positive" if v >= 0 else "negative"

    # Source stats
    base_pnl = base_trades["pnl_dollars"].sum() if len(base_trades) else 0
    trend_pnl = trend_trades["pnl_dollars"].sum() if len(trend_trades) else 0
    base_wr = len(base_trades[base_trades["pnl_dollars"] > 0]) / len(base_trades) * 100 if len(base_trades) else 0
    trend_wr = len(trend_trades[trend_trades["pnl_dollars"] > 0]) / len(trend_trades) * 100 if len(trend_trades) else 0

    outperf = total_pnl - bh_pnl
    bh_ratio = total_pnl / bh_pnl if bh_pnl != 0 else 0

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Combined VWAP Strategy — Base + Trending Band</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; background: #1e1e1e; color: #e0e0e0; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #4ec9b0; text-align: center; }}
        .subtitle {{ text-align: center; color: #a0a0a0; margin-bottom: 20px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #2d2d2d; padding: 15px; border-radius: 8px; border-left: 4px solid #4ec9b0; }}
        .stat-label {{ color: #999; font-size: 12px; text-transform: uppercase; }}
        .stat-value {{ color: #fff; font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .positive {{ color: #4ec9b0; }}
        .negative {{ color: #f48771; }}
        .highlight {{ background: #2d4a3e; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #4ec9b0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Combined VWAP Strategy</h1>
    <div class="subtitle">Base (lvl_delta&ge;{BASE_LEVEL_DELTA}, SL {BASE_SL}x/TP {BASE_TP}x @ VWAP) + Trending (delta&ge;{TREND_DELTA}, ADX&ge;{TREND_ADX_MIN:.0f}, SL {TREND_SL}x/TP {TREND_TP}x @ STD{TREND_STD})</div>

    <div class="stats">
        <div class="stat-card"><div class="stat-label">Total P&L</div><div class="stat-value {cls(total_pnl)}">{fmt(total_pnl)}</div></div>
        <div class="stat-card"><div class="stat-label">Total Trades</div><div class="stat-value">{total}</div></div>
        <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value {cls(win_rate - 50)}">{win_rate:.1f}%</div></div>
        <div class="stat-card"><div class="stat-label">Profit Factor</div><div class="stat-value {cls(pf - 1)}">{pf:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Max Drawdown</div><div class="stat-value negative">{fmt(abs(max_dd))}</div></div>
        <div class="stat-card"><div class="stat-label">Sharpe Ratio</div><div class="stat-value {cls(sharpe)}">{sharpe:.2f}</div></div>
    </div>

    <div class="stats">
        <div class="stat-card"><div class="stat-label">Avg Winner</div><div class="stat-value positive">{fmt(avg_win)}</div></div>
        <div class="stat-card"><div class="stat-label">Avg Loser</div><div class="stat-value negative">{fmt(avg_loss)}</div></div>
        <div class="stat-card"><div class="stat-label">Avg P&L/Trade</div><div class="stat-value {cls(avg_pnl)}">{fmt(avg_pnl)}</div></div>
        <div class="stat-card"><div class="stat-label">Risk-Adjusted Return</div><div class="stat-value {cls(risk_adj)}">{risk_adj:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Avg Duration</div><div class="stat-value">{avg_dur:.0f} min</div><div style="color:#999;font-size:12px;margin-top:5px;">Median: {med_dur:.0f} min</div></div>
    </div>

    <div class="highlight">
        <h3 style="margin-top:0;color:#4ec9b0;">Strategy Breakdown</h3>
        <div class="stats" style="margin-top:15px;">
            <div class="stat-card" style="border-left-color:#4ec9b0;"><div class="stat-label">Base VWAP (lvl_delta&ge;{BASE_LEVEL_DELTA})</div><div class="stat-value {cls(base_pnl)}">{fmt(base_pnl)}</div><div style="color:#999;font-size:12px;margin-top:5px;">{len(base_trades)} trades | WR: {base_wr:.1f}%</div></div>
            <div class="stat-card" style="border-left-color:#dcdcaa;"><div class="stat-label">Trending (delta&ge;{TREND_DELTA}, ADX&ge;{TREND_ADX_MIN:.0f})</div><div class="stat-value {cls(trend_pnl)}">{fmt(trend_pnl)}</div><div style="color:#999;font-size:12px;margin-top:5px;">{len(trend_trades)} trades | WR: {trend_wr:.1f}%</div></div>
        </div>
    </div>

    <div class="highlight">
        <h3 style="margin-top:0;color:#4ec9b0;">Direction Breakdown</h3>
        <div class="stats" style="margin-top:15px;">
            <div class="stat-card"><div class="stat-label">Shorts</div><div class="stat-value">{len(shorts)}</div><div style="color:#999;font-size:12px;margin-top:5px;">WR: {len(shorts[shorts['pnl_dollars']>0])/max(1,len(shorts))*100:.1f}% | P&L: {fmt(shorts['pnl_dollars'].sum())}</div></div>
            <div class="stat-card"><div class="stat-label">Longs</div><div class="stat-value">{len(longs)}</div><div style="color:#999;font-size:12px;margin-top:5px;">WR: {len(longs[longs['pnl_dollars']>0])/max(1,len(longs))*100:.1f}% | P&L: {fmt(longs['pnl_dollars'].sum())}</div></div>
        </div>
    </div>

    <div class="highlight">
        <h3 style="margin-top:0;color:#4ec9b0;">Exit Breakdown</h3>
        <div class="stats" style="margin-top:15px;">
            <div class="stat-card"><div class="stat-label">Target Hits</div><div class="stat-value positive">{len(tp_exits)} ({len(tp_exits)/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(tp_exits['pnl_dollars'].sum())}</div></div>
            <div class="stat-card"><div class="stat-label">Stop Losses</div><div class="stat-value negative">{len(sl_exits)} ({len(sl_exits)/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(sl_exits['pnl_dollars'].sum())}</div></div>
            <div class="stat-card"><div class="stat-label">EOD / Session</div><div class="stat-value">{len(eod_exits)} ({len(eod_exits)/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(eod_exits['pnl_dollars'].sum())}</div></div>
        </div>
    </div>

    <div class="highlight">
        <h3 style="margin-top:0;color:#4ec9b0;">In-Sample / Out-of-Sample (50/50 by trade count)</h3>
        <div class="stats" style="margin-top:15px;">
            <div class="stat-card"><div class="stat-label">IS ({is_df['date'].iloc[0]} to {is_df['date'].iloc[-1]})</div><div class="stat-value {cls(is_pnl)}">{fmt(is_pnl)}</div><div style="color:#999;font-size:12px;margin-top:5px;">{len(is_df)} trades | WR {is_wr:.1f}% | PF {is_pf:.2f}</div></div>
            <div class="stat-card"><div class="stat-label">OOS ({oos_df['date'].iloc[0]} to {oos_df['date'].iloc[-1]})</div><div class="stat-value {cls(oos_pnl)}">{fmt(oos_pnl)}</div><div style="color:#999;font-size:12px;margin-top:5px;">{len(oos_df)} trades | WR {oos_wr:.1f}% | PF {oos_pf:.2f}</div></div>
        </div>
    </div>

    <div class="highlight">
        <h3 style="margin-top:0;color:#4ec9b0;">Buy & Hold Comparison</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
            <div><p style="margin:5px 0;"><strong>B&H P&L:</strong> {fmt(bh_pnl)}</p><p style="margin:5px 0;color:#999;font-size:13px;">Entry: {first_price:,.2f} | Exit: {last_price:,.2f}</p></div>
            <div><p style="margin:5px 0;"><strong>Outperformance:</strong> <span class="{cls(outperf)}">${outperf:+,.0f}</span></p><p style="margin:5px 0;color:#999;font-size:13px;">Strategy is {bh_ratio:.2f}x buy & hold</p></div>
        </div>
    </div>

    <div id="equity-chart"></div>
    <div id="dd-chart"></div>
    <div id="monthly-chart"></div>

    <script>
        var dates = {json.dumps(dates)};
        var equity = {json.dumps([round(v, 2) for v in strategy_equity])};
        var baseCum = {json.dumps(base_cum)};
        var trendCum = {json.dumps(trend_cum)};
        var bh = {json.dumps(bh_equity)};
        var dd = {json.dumps([round(v, 2) for v in dd_series])};
        var mL = {json.dumps(monthly_labels)};
        var mV = {json.dumps(monthly_values)};
        var sp = {sp};

        Plotly.newPlot('equity-chart', [
            {{ x: dates, y: equity, type:'scatter', mode:'lines', name:'Combined', line:{{color:'#4ec9b0',width:3}}, fill:'tozeroy', fillcolor:'rgba(78,201,176,0.1)' }},
            {{ x: dates, y: baseCum, type:'scatter', mode:'lines', name:'Base VWAP (lvl_delta&ge;{BASE_LEVEL_DELTA})', line:{{color:'#569cd6',width:2,dash:'dot'}} }},
            {{ x: dates, y: trendCum, type:'scatter', mode:'lines', name:'Trending (delta&ge;{TREND_DELTA}, ADX&ge;{TREND_ADX_MIN:.0f})', line:{{color:'#dcdcaa',width:2,dash:'dot'}} }},
            {{ x: dates, y: bh, type:'scatter', mode:'lines', name:'Buy & Hold', line:{{color:'#ce9178',width:2}}, visible:'legendonly' }}
        ], {{
            title:{{text:'Cumulative P&L — Combined Strategy',font:{{color:'#e0e0e0',size:16}}}},
            xaxis:{{gridcolor:'#333',color:'#999'}}, yaxis:{{title:'P&L ($)',gridcolor:'#333',color:'#999',tickformat:'$,.0f'}},
            plot_bgcolor:'#1e1e1e', paper_bgcolor:'#1e1e1e', font:{{color:'#e0e0e0'}}, hovermode:'x unified', height:600,
            shapes:[{{type:'line',x0:dates[sp],x1:dates[sp],y0:0,y1:1,yref:'paper',line:{{color:'#f48771',width:2,dash:'dash'}}}}],
            annotations:[{{x:dates[sp],y:1.05,yref:'paper',text:'IS | OOS',showarrow:false,font:{{color:'#f48771',size:12}}}}],
            legend:{{x:0.02,y:0.98,bgcolor:'rgba(45,45,45,0.8)',bordercolor:'#4ec9b0',borderwidth:1}}
        }}, {{responsive:true}});

        Plotly.newPlot('dd-chart', [{{x:dates,y:dd,type:'scatter',mode:'lines',name:'Drawdown',line:{{color:'#f48771',width:2}},fill:'tozeroy',fillcolor:'rgba(244,135,113,0.15)'}}],
            {{title:{{text:'Drawdown',font:{{color:'#e0e0e0',size:16}}}},xaxis:{{gridcolor:'#333',color:'#999'}},yaxis:{{title:'DD ($)',gridcolor:'#333',color:'#999',tickformat:'$,.0f'}},plot_bgcolor:'#1e1e1e',paper_bgcolor:'#1e1e1e',font:{{color:'#e0e0e0'}},hovermode:'x unified',height:350}},{{responsive:true}});

        var mc = mV.map(v=>v>=0?'#4ec9b0':'#f48771');
        Plotly.newPlot('monthly-chart', [{{x:mL,y:mV,type:'bar',marker:{{color:mc}},text:mV.map(v=>'$'+v.toLocaleString('en-US',{{maximumFractionDigits:0}})),textposition:'outside',textfont:{{color:'#e0e0e0',size:11}}}}],
            {{title:{{text:'Monthly P&L',font:{{color:'#e0e0e0',size:16}}}},xaxis:{{gridcolor:'#333',color:'#999'}},yaxis:{{title:'P&L ($)',gridcolor:'#333',color:'#999',tickformat:'$,.0f'}},plot_bgcolor:'#1e1e1e',paper_bgcolor:'#1e1e1e',font:{{color:'#e0e0e0'}},height:400}},{{responsive:true}});
    </script>

    <div style="margin-top:30px;padding:20px;background:#2d2d2d;border-radius:8px;">
        <h3 style="color:#4ec9b0;">Strategy Configuration</h3>
        <ul style="line-height:1.8;">
            <li><strong>Base:</strong> lvl_delta&ge;{BASE_LEVEL_DELTA}, no ADX | Absorption at VWAP zone (+/-3 pts) | SL {BASE_SL}x / TP {BASE_TP}x ATR</li>
            <li><strong>Trending:</strong> delta&ge;{TREND_DELTA}, ADX&ge;{TREND_ADX_MIN:.0f} | STD{TREND_STD} band, {TREND_LB}-bar lookback | SL {TREND_SL}x / TP {TREND_TP}x ATR</li>
            <li><strong>Priority:</strong> Whichever signal fires first on a given day. One trade at a time across both.</li>
            <li><strong>Entry Window:</strong> 7pm - 4:00pm ET | <strong>Force Close:</strong> 4:58pm ET</li>
        </ul>
    </div>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Equity HTML saved to {output_path}")


def generate_calendar_html(trades, output_path):
    """Generate PnL calendar HTML."""
    # Group by date
    daily_trades = {}
    for t in trades:
        d = t["date"]
        if d not in daily_trades:
            daily_trades[d] = []
        daily_trades[d].append({
            "pnl": round(t["pnl_dollars"], 2),
            "source": t["source"],
        })

    # For the calendar JS: date -> [pnl_dollars, ...]
    cal_data = {}
    for d, ts in daily_trades.items():
        cal_data[d] = [t["pnl"] for t in ts]

    total_trades = sum(len(v) for v in cal_data.values())
    all_pnl = [p for trades in cal_data.values() for p in trades]
    total_pnl = sum(all_pnl)
    winners = [p for p in all_pnl if p > 0]
    win_rate = len(winners) / len(all_pnl) * 100 if all_pnl else 0

    dates = sorted(cal_data.keys())
    start_month = dates[0][:7] if dates else "2025-03"

    # Also build source info per day for tooltip
    source_info = {}
    for d, ts in daily_trades.items():
        base_n = sum(1 for t in ts if t["source"] == "base")
        trend_n = sum(1 for t in ts if t["source"] == "trend")
        source_info[d] = f"{base_n}B {trend_n}T"

    trades_json = json.dumps(cal_data)
    source_json = json.dumps(source_info)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Combined VWAP Strategy - P&L Calendar</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #1e1e1e; color: #fff; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; margin-bottom: 10px; color: #4ec9b0; }}
        .subtitle {{ text-align: center; margin-bottom: 20px; color: #a0a0a0; font-size: 14px; }}
        .strategy-note {{ background: #2d2d30; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4ec9b0; }}
        .strategy-note h3 {{ margin-bottom: 10px; color: #4ec9b0; font-size: 16px; }}
        .strategy-note p {{ color: #a0a0a0; font-size: 13px; margin-bottom: 5px; }}
        .contract-controls {{ background: #2d2d30; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .contract-controls h3 {{ margin-top: 0; margin-bottom: 15px; color: #4ec9b0; }}
        .control-row {{ display: flex; gap: 30px; align-items: center; margin-bottom: 15px; }}
        .radio-group {{ display: flex; gap: 20px; }}
        .radio-option {{ display: flex; align-items: center; gap: 5px; }}
        .slider-container {{ display: flex; align-items: center; gap: 15px; flex: 1; }}
        .slider {{ flex: 1; max-width: 300px; }}
        .slider-value {{ font-size: 18px; font-weight: bold; color: #4ec9b0; min-width: 120px; }}
        .controls {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; background: #2d2d30; padding: 20px; border-radius: 8px; }}
        .nav-buttons button {{ background: #3e3e42; color: #fff; border: none; padding: 10px 20px; margin: 0 5px; border-radius: 4px; cursor: pointer; font-size: 16px; }}
        .nav-buttons button:hover {{ background: #505050; }}
        .month-title {{ font-size: 24px; font-weight: bold; }}
        .month-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: #2d2d30; padding: 15px; border-radius: 8px; }}
        .stat-label {{ font-size: 12px; color: #a0a0a0; margin-bottom: 5px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; }}
        .positive {{ color: #4ec9b0; }}
        .negative {{ color: #f48771; }}
        .neutral {{ color: #a0a0a0; }}
        .calendar {{ background: #2d2d30; border-radius: 8px; padding: 20px; }}
        .calendar-header {{ display: grid; grid-template-columns: repeat(7, 1fr) 120px; gap: 5px; margin-bottom: 10px; font-weight: bold; color: #a0a0a0; }}
        .calendar-row {{ display: grid; grid-template-columns: repeat(7, 1fr) 120px; gap: 5px; margin-bottom: 5px; }}
        .calendar-day {{ background: #3e3e42; padding: 10px; border-radius: 4px; min-height: 80px; position: relative; cursor: pointer; transition: all 0.2s; }}
        .calendar-day:hover {{ transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.3); }}
        .calendar-day.empty {{ background: transparent; cursor: default; }}
        .calendar-day.empty:hover {{ transform: none; box-shadow: none; }}
        .calendar-day.profit {{ background: linear-gradient(135deg, #2d4a3e 0%, #3e3e42 100%); border-left: 3px solid #4ec9b0; }}
        .calendar-day.loss {{ background: linear-gradient(135deg, #4a2d2d 0%, #3e3e42 100%); border-left: 3px solid #f48771; }}
        .day-number {{ font-size: 14px; color: #a0a0a0; margin-bottom: 5px; }}
        .day-pnl {{ font-size: 18px; font-weight: bold; margin-bottom: 3px; }}
        .day-trades {{ font-size: 11px; color: #a0a0a0; }}
        .day-source {{ font-size: 10px; color: #666; margin-top: 2px; }}
        .week-total {{ background: #2d2d30; padding: 10px; border-radius: 4px; display: flex; flex-direction: column; justify-content: center; align-items: center; font-weight: bold; }}
        .week-label {{ font-size: 10px; color: #a0a0a0; margin-bottom: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Combined VWAP Strategy - P&L Calendar</h1>
        <div class="subtitle">Base (lvl_delta&ge;{BASE_LEVEL_DELTA}) + Trending (delta&ge;{TREND_DELTA}, ADX&ge;{TREND_ADX_MIN:.0f}) | {total_trades} trades | {win_rate:.1f}% WR</div>

        <div class="strategy-note">
            <h3>Combined Strategy</h3>
            <p><strong>Base:</strong> lvl_delta&ge;{BASE_LEVEL_DELTA}, no ADX | Absorption at VWAP zone | SL {BASE_SL}x / TP {BASE_TP}x ATR</p>
            <p><strong>Trending:</strong> delta&ge;{TREND_DELTA}, ADX&ge;{TREND_ADX_MIN:.0f} | STD{TREND_STD} band, {TREND_LB}-bar lookback | SL {TREND_SL}x / TP {TREND_TP}x ATR</p>
            <p><strong>Rule:</strong> One trade at a time. Whichever fires first gets priority.</p>
            <p><strong>Total P&L (1 NQ):</strong> ${total_pnl:,.0f}</p>
        </div>

        <div class="contract-controls">
            <h3>Contract Settings</h3>
            <div class="control-row">
                <div>
                    <div class="radio-group">
                        <div class="radio-option"><input type="radio" id="mnq" name="contractType" value="mnq" checked><label for="mnq">MNQ ($2/point)</label></div>
                        <div class="radio-option"><input type="radio" id="nq" name="contractType" value="nq"><label for="nq">NQ ($20/point)</label></div>
                    </div>
                </div>
                <div class="slider-container">
                    <label>Contracts:</label>
                    <input type="range" id="contractSize" class="slider" min="1" max="9" value="1" step="1">
                    <div class="slider-value" id="contractSizeValue">1</div>
                </div>
            </div>
        </div>

        <div class="controls">
            <div class="nav-buttons"><button id="prevMonth">&larr; Previous</button><button id="nextMonth">Next &rarr;</button></div>
            <div class="month-title" id="monthTitle"></div>
        </div>

        <div class="month-stats">
            <div class="stat-card"><div class="stat-label">Month P&L</div><div class="stat-value" id="monthPnl">$0</div></div>
            <div class="stat-card"><div class="stat-label">Trading Days</div><div class="stat-value" id="tradingDays">0</div></div>
            <div class="stat-card"><div class="stat-label">Avg Daily P&L</div><div class="stat-value" id="avgDaily">$0</div></div>
            <div class="stat-card"><div class="stat-label">Year Total</div><div class="stat-value positive" id="yearTotal">$0</div></div>
            <div class="stat-card"><div class="stat-label">Max Daily DD</div><div class="stat-value negative" id="maxDailyDD">$0</div></div>
            <div class="stat-card"><div class="stat-label">Max DD</div><div class="stat-value negative" id="maxDD">$0</div></div>
        </div>

        <div class="calendar">
            <div class="calendar-header"><div>Sun</div><div>Mon</div><div>Tue</div><div>Wed</div><div>Thu</div><div>Fri</div><div>Sat</div><div>Week Total</div></div>
            <div id="calendarBody"></div>
        </div>
    </div>

    <script>
        const tradesData = {trades_json};
        const sourceInfo = {source_json};
        const startMonth = '{start_month}';
        let currentMonth = startMonth;
        let contractType = 'mnq';
        let numContracts = 1;

        function calculatePnl(nqDollarPnl) {{
            return nqDollarPnl * (contractType === 'mnq' ? 0.1 : 1) * numContracts;
        }}

        function renderCalendar() {{
            const [year, month] = currentMonth.split('-').map(Number);
            const firstDay = new Date(year, month - 1, 1);
            const lastDay = new Date(year, month, 0);
            const daysInMonth = lastDay.getDate();
            const startDay = firstDay.getDay();

            document.getElementById('monthTitle').textContent = firstDay.toLocaleString('default', {{ month: 'long', year: 'numeric' }});

            let monthPnl = 0, tradingDays = 0, calendarHtml = '', currentRow = [], weekPnl = 0;

            for (let i = 0; i < startDay; i++) currentRow.push('<div class="calendar-day empty"></div>');

            for (let day = 1; day <= daysInMonth; day++) {{
                const dateStr = `${{year}}-${{String(month).padStart(2, '0')}}-${{String(day).padStart(2, '0')}}`;
                const dayTrades = tradesData[dateStr] || [];

                let dayPnl = 0;
                for (const nqPnl of dayTrades) dayPnl += calculatePnl(nqPnl);

                if (dayTrades.length > 0) {{
                    tradingDays++; monthPnl += dayPnl; weekPnl += dayPnl;
                    const src = sourceInfo[dateStr] || '';
                    currentRow.push(`<div class="calendar-day ${{dayPnl > 0 ? 'profit' : dayPnl < 0 ? 'loss' : ''}}"><div class="day-number">${{day}}</div><div class="day-pnl ${{dayPnl > 0 ? 'positive' : dayPnl < 0 ? 'negative' : 'neutral'}}">$${{dayPnl.toFixed(0)}}</div><div class="day-trades">${{dayTrades.length}} trades</div><div class="day-source">${{src}}</div></div>`);
                }} else {{
                    currentRow.push(`<div class="calendar-day empty"><div class="day-number">${{day}}</div></div>`);
                }}

                if ((startDay + day - 1) % 7 === 6 || day === daysInMonth) {{
                    while (currentRow.length < 7) currentRow.push('<div class="calendar-day empty"></div>');
                    currentRow.push(`<div class="week-total"><div class="week-label">Week</div><div class="${{weekPnl > 0 ? 'positive' : weekPnl < 0 ? 'negative' : 'neutral'}}">$${{weekPnl.toFixed(0)}}</div></div>`);
                    calendarHtml += '<div class="calendar-row">' + currentRow.join('') + '</div>';
                    currentRow = []; weekPnl = 0;
                }}
            }}

            document.getElementById('calendarBody').innerHTML = calendarHtml;
            document.getElementById('monthPnl').innerHTML = `<span class="${{monthPnl > 0 ? 'positive' : monthPnl < 0 ? 'negative' : 'neutral'}}">$${{monthPnl.toFixed(0)}}</span>`;
            document.getElementById('tradingDays').textContent = tradingDays;
            const avg = tradingDays > 0 ? monthPnl / tradingDays : 0;
            document.getElementById('avgDaily').innerHTML = `<span class="${{avg > 0 ? 'positive' : avg < 0 ? 'negative' : 'neutral'}}">$${{avg.toFixed(0)}}</span>`;

            let yearPnl = 0, maxDailyDD = 0, maxDD = 0, peak = 0, run = 0;
            for (const d of Object.keys(tradesData).sort()) {{
                let dp = 0; for (const p of tradesData[d]) dp += calculatePnl(p);
                yearPnl += dp;
                if (dp < maxDailyDD) maxDailyDD = dp;
                run += dp; if (run > peak) peak = run;
                const dd = peak - run; if (dd > maxDD) maxDD = dd;
            }}
            document.getElementById('yearTotal').innerHTML = `<span class="${{yearPnl > 0 ? 'positive' : 'negative'}}">$${{yearPnl.toFixed(0)}}</span>`;
            document.getElementById('maxDailyDD').textContent = `$${{Math.abs(maxDailyDD).toFixed(0)}}`;
            document.getElementById('maxDD').textContent = `$${{maxDD.toFixed(0)}}`;
        }}

        function changeMonth(d) {{
            let [y, m] = currentMonth.split('-').map(Number);
            m += d; if (m < 1) {{ m = 12; y--; }} else if (m > 12) {{ m = 1; y++; }}
            currentMonth = `${{y}}-${{String(m).padStart(2, '0')}}`; renderCalendar();
        }}

        document.getElementById('prevMonth').addEventListener('click', () => changeMonth(-1));
        document.getElementById('nextMonth').addEventListener('click', () => changeMonth(1));
        document.querySelectorAll('input[name="contractType"]').forEach(r => r.addEventListener('change', e => {{ contractType = e.target.value; renderCalendar(); }}));
        document.getElementById('contractSize').addEventListener('input', function() {{ numContracts = parseInt(this.value); document.getElementById('contractSizeValue').textContent = numContracts; renderCalendar(); }});
        renderCalendar();
    </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Calendar HTML saved to {output_path}")


def main():
    print("Building ADX lookup...", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"  {len(adx_lookup)} rows")

    print("Running combined backtest...", flush=True)
    trades = run_combined_backtest(adx_lookup)

    base_n = sum(1 for t in trades if t["source"] == "base")
    trend_n = sum(1 for t in trades if t["source"] == "trend")
    print(f"  {len(trades)} total trades (base={base_n}, trend={trend_n})")

    # Quick IS/OOS summary
    df = pd.DataFrame(trades).sort_values("date")
    half = len(df) // 2
    def _stat(d):
        n = len(d); pnl = d["pnl_dollars"].sum()
        w = d[d["pnl_dollars"] > 0]; l = d[d["pnl_dollars"] < 0]
        wr = 100*len(w)/n if n else 0
        gp = w["pnl_dollars"].sum(); gl = abs(l["pnl_dollars"].sum())
        pf = gp/gl if gl > 0 else float("inf")
        cum = d["pnl_dollars"].cumsum()
        dd = float((cum - cum.cummax()).min()) if n else 0
        daily = d.groupby("date")["pnl_dollars"].sum()
        sh = (daily.mean()/daily.std())*np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0
        return n, pnl, wr, pf, dd, sh
    for lbl, d in [("IS ", df.iloc[:half]), ("OOS", df.iloc[half:]), ("ALL", df)]:
        n, pnl, wr, pf, dd, sh = _stat(d)
        print(f"  {lbl}: n={n:<4} PnL ${pnl:>+9,.0f}  WR {wr:5.1f}%  PF {pf:.2f}  DD ${dd:>+8,.0f}  Sharpe {sh:.2f}")

    eq_path = Path("results/html/vwap_combined_equity.html")
    generate_equity_html(trades, eq_path)

    cal_path = Path("results/html/vwap_combined_calendar.html")
    generate_calendar_html(trades, cal_path)


if __name__ == "__main__":
    main()

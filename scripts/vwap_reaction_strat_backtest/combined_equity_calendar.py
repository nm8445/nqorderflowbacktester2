"""
Combined strategy equity curve + PnL calendar — HTML output.
  Base:     level_delta >= 50, no ADX, SL=0.50, TP=1.50
  Trending: delta >= 30, ADX >= 30, STD1, LB=14, SL=0.75, TP=2.50

Outputs:
  results/html/combined_equity_calendar.html
  results/combined_trades.csv

Usage:
    python scripts/vwap_reaction_strat_backtest/combined_equity_calendar.py
"""
import json
import pickle
import sys
import calendar
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

BASE_SL, BASE_TP = 0.50, 1.50
BASE_LEVEL_DELTA = 50

TREND_SL, TREND_TP = 0.75, 2.50
TREND_DELTA = 30
TREND_ADX_MIN = 30.0

STARTING_EQUITY = 50000.0


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
            signals.append({"bar_index": i, "direction": "long", "strategy": "base",
                            "confirm_time": cb.close_time, "entry_price": cb.close, "atr": atr})
        if bias == "short" and approached_below and has_bear_abs and confirm_bearish:
            signals.append({"bar_index": i, "direction": "short", "strategy": "base",
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
            signals.append({"bar_index": i, "direction": "long", "strategy": "trending",
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
            signals.append({"bar_index": i, "direction": "short", "strategy": "trending",
                            "confirm_time": cb.close_time, "entry_price": cb.close, "atr": float(atr)})
    return signals


# ─── SIMULATION ───

def simulate_day(bars, signals, adx_lookup):
    if not signals:
        return []
    signals = sorted(signals, key=lambda x: x["confirm_time"])
    trades = []
    last_exit = None
    for sig in signals:
        ct_et = to_et(sig["confirm_time"])
        hm = ct_et.strftime("%H:%M")
        strat = sig["strategy"]
        if strat == "base" and LUNCH_START <= hm < LUNCH_END:
            continue
        if "16:00" <= hm < "19:10":
            continue
        if last_exit is not None and sig["confirm_time"] <= last_exit:
            continue
        d, ep, atr = sig["direction"], sig["entry_price"], sig["atr"]
        if strat == "trending":
            adx_val = get_adx_at_time(ct_et, adx_lookup)
            if np.isnan(adx_val) or adx_val < TREND_ADX_MIN:
                continue
            sl_m, tp_m = TREND_SL, TREND_TP
        else:
            sl_m, tp_m = BASE_SL, BASE_TP
        if d == "long":
            sl = ep - atr * sl_m; tp = ep + atr * tp_m
        else:
            sl = ep + atr * sl_m; tp = ep - atr * tp_m
        cbi = sig["bar_index"] + 1
        ex_price = ex_time = ex_reason = None
        for j in range(cbi + 1, len(bars)):
            b = bars[j]
            if not b.closed: continue
            bet = to_et(b.close_time)
            if bet.strftime("%H:%M") >= FORCE_CLOSE:
                ex_price, ex_time, ex_reason = b.close, b.close_time, "eod"; break
            if d == "long":
                if b.low <= sl:
                    ex_price, ex_time, ex_reason = sl - SL_SLIP_TICKS * TICK_SIZE, b.close_time, "sl"; break
                if b.high >= tp:
                    ex_price, ex_time, ex_reason = tp - TP_SLIP_TICKS * TICK_SIZE, b.close_time, "tp"; break
            else:
                if b.high >= sl:
                    ex_price, ex_time, ex_reason = sl + SL_SLIP_TICKS * TICK_SIZE, b.close_time, "sl"; break
                if b.low <= tp:
                    ex_price, ex_time, ex_reason = tp + TP_SLIP_TICKS * TICK_SIZE, b.close_time, "tp"; break
        if ex_price is None:
            ex_price, ex_time, ex_reason = bars[-1].close, bars[-1].close_time, "eod"
        pnl = ((ex_price - ep) if d == "long" else (ep - ex_price)) * POINT_VALUE
        exit_et = to_et(ex_time)
        dur_mins = (exit_et - ct_et).total_seconds() / 60
        trades.append({
            "date": str(ct_et.date()), "time": ct_et.strftime("%H:%M"),
            "strategy": strat, "direction": d,
            "entry": ep, "exit": ex_price, "pnl": pnl,
            "reason": ex_reason, "duration_mins": dur_mins,
        })
        last_exit = ex_time
    return trades


def fmt(v):
    return f"${v:,.0f}"


def pn(v):
    return "positive" if v >= 0 else "negative"


def generate_html(df, output_path):
    total = len(df)
    winners = df[df["pnl"] > 0]
    losers = df[df["pnl"] < 0]
    total_pnl = df["pnl"].sum()
    win_rate = len(winners) / total * 100
    avg_win = winners["pnl"].mean() if len(winners) else 0
    avg_loss = losers["pnl"].mean() if len(losers) else 0
    avg_pnl = df["pnl"].mean()
    gp = winners["pnl"].sum() if len(winners) else 0
    gl = abs(losers["pnl"].sum()) if len(losers) else 1
    pf = gp / gl if gl > 0 else 999
    cum = df["pnl"].cumsum()
    max_dd = float((cum - cum.cummax()).min())
    risk_adj = abs(total_pnl / max_dd) if max_dd != 0 else 0
    avg_dur = df["duration_mins"].mean()

    base_df = df[df["strategy"] == "base"]
    trend_df = df[df["strategy"] == "trending"]
    tp_exits = df[df["reason"] == "tp"]
    sl_exits = df[df["reason"] == "sl"]
    eod_exits = df[df["reason"] == "eod"]

    # IS/OOS split
    dates_sorted = sorted(df["date"].unique())
    split_idx = len(dates_sorted) // 2
    is_end_date = dates_sorted[split_idx - 1] if split_idx > 0 else dates_sorted[0]
    is_df = df[df["date"] <= is_end_date]
    oos_df = df[df["date"] > is_end_date]

    def sub_stats(sub):
        n = len(sub)
        if n == 0:
            return {"n": 0, "pnl": 0, "wr": 0, "pf": 0}
        p = sub["pnl"].sum()
        w = sub[sub["pnl"] > 0]
        l = sub[sub["pnl"] < 0]
        wr = len(w) / n * 100
        gpp = w["pnl"].sum() if len(w) else 0
        gll = abs(l["pnl"].sum()) if len(l) else 1
        return {"n": n, "pnl": p, "wr": wr, "pf": gpp / gll if gll > 0 else 999}

    is_s = sub_stats(is_df)
    oos_s = sub_stats(oos_df)

    # Equity data for chart
    df_sorted = df.sort_values("date").reset_index(drop=True)
    daily = df_sorted.groupby("date")["pnl"].sum().reset_index()
    daily.columns = ["date", "pnl"]
    daily["equity"] = STARTING_EQUITY + daily["pnl"].cumsum()

    dates_json = json.dumps(daily["date"].tolist())
    equity_json = json.dumps([round(v, 2) for v in daily["equity"].tolist()])

    # Per-strategy equity
    base_daily = df_sorted[df_sorted["strategy"] == "base"].groupby("date")["pnl"].sum().cumsum()
    trend_daily = df_sorted[df_sorted["strategy"] == "trending"].groupby("date")["pnl"].sum().cumsum()

    # Align to all dates
    all_trade_dates = daily["date"].tolist()
    base_eq = []
    trend_eq = []
    for d in all_trade_dates:
        base_eq.append(round(STARTING_EQUITY + (base_daily.get(d, base_daily[base_daily.index <= d].iloc[-1] if len(base_daily[base_daily.index <= d]) else 0)), 2))
        trend_eq.append(round(STARTING_EQUITY + (trend_daily.get(d, trend_daily[trend_daily.index <= d].iloc[-1] if len(trend_daily[trend_daily.index <= d]) else 0)), 2))

    base_eq_json = json.dumps(base_eq)
    trend_eq_json = json.dumps(trend_eq)

    # PnL calendar HTML
    daily_pnl = df_sorted.groupby("date")["pnl"].sum().to_dict()
    all_months = sorted(set((int(d[:4]), int(d[5:7])) for d in daily_pnl.keys()))
    max_abs = max(abs(v) for v in daily_pnl.values()) if daily_pnl else 1

    cal_html = ""
    for year, month in all_months:
        cal_data = calendar.monthcalendar(year, month)
        month_total = 0
        cells = ""
        for week in cal_data:
            cells += "<tr>"
            for day_idx, day in enumerate(week):
                if day == 0:
                    cells += '<td class="cal-empty"></td>'
                    continue
                ds = f"{year}-{month:02d}-{day:02d}"
                if ds in daily_pnl:
                    pnl = daily_pnl[ds]
                    month_total += pnl
                    intensity = min(abs(pnl) / max_abs, 1.0) * 0.7 + 0.3
                    if pnl > 0:
                        bg = f"rgba(78, 201, 176, {intensity})"
                    else:
                        bg = f"rgba(244, 135, 113, {intensity})"
                    cells += f'<td class="cal-day" style="background:{bg}"><div class="cal-num">{day}</div><div class="cal-pnl">${pnl:+,.0f}</div></td>'
                else:
                    cells += f'<td class="cal-day cal-nodata"><div class="cal-num" style="color:#555">{day}</div></td>'
            cells += "</tr>"

        total_color = "#4ec9b0" if month_total > 0 else "#f48771"
        cal_html += f"""
        <div class="cal-month">
            <div class="cal-title">{calendar.month_abbr[month]} {year}</div>
            <table class="cal-table">
                <thead><tr><th>M</th><th>T</th><th>W</th><th>T</th><th>F</th><th>S</th><th>S</th></tr></thead>
                <tbody>{cells}</tbody>
            </table>
            <div class="cal-total" style="color:{total_color}">Total: ${month_total:+,.0f}</div>
        </div>"""

    # Monthly summary table
    monthly_rows = ""
    for year, month in all_months:
        mask = df_sorted["date"].str.startswith(f"{year}-{month:02d}")
        m = df_sorted[mask]
        if m.empty:
            continue
        n = len(m)
        pnl = m["pnl"].sum()
        nb = len(m[m["strategy"] == "base"])
        nt = len(m[m["strategy"] == "trending"])
        wr = len(m[m["pnl"] > 0]) / n * 100
        c = "#4ec9b0" if pnl > 0 else "#f48771"
        monthly_rows += f'<tr><td>{calendar.month_abbr[month]} {year}</td><td>{n}</td><td>{nb}</td><td>{nt}</td><td style="color:{c}">${pnl:+,.0f}</td><td>{wr:.1f}%</td></tr>'

    # Base/Trend sub-stats
    def strat_card(label, sub):
        n = len(sub)
        if n == 0:
            return ""
        p = sub["pnl"].sum()
        w = sub[sub["pnl"] > 0]
        l = sub[sub["pnl"] < 0]
        wr = len(w) / n * 100
        gpp = w["pnl"].sum() if len(w) else 0
        gll = abs(l["pnl"].sum()) if len(l) else 1
        spf = gpp / gll if gll > 0 else 999
        return f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{n} trades</div>
                <div style="color: #999; font-size: 12px; margin-top: 5px;">
                    PnL: <span class="{pn(p)}">{fmt(p)}</span> | WR: {wr:.1f}% | PF: {spf:.2f}
                </div>
            </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Combined Strategy - Equity Curve & Calendar</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #1e1e1e;
            color: #e0e0e0;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #4ec9b0; text-align: center; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #a0a0a0; margin-bottom: 20px; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4ec9b0;
        }}
        .stat-label {{ color: #999; font-size: 12px; text-transform: uppercase; }}
        .stat-value {{ color: #fff; font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .positive {{ color: #4ec9b0; }}
        .negative {{ color: #f48771; }}
        .highlight {{
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #4ec9b0;
        }}
        .cal-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .cal-month {{
            background: #2d2d2d;
            border-radius: 8px;
            padding: 10px;
        }}
        .cal-title {{
            text-align: center;
            font-weight: bold;
            color: #4ec9b0;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .cal-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }}
        .cal-table th {{
            color: #888;
            font-size: 10px;
            padding: 2px;
            text-align: center;
        }}
        .cal-day {{
            text-align: center;
            padding: 4px 2px;
            border-radius: 4px;
        }}
        .cal-empty {{ }}
        .cal-nodata {{ }}
        .cal-num {{ font-size: 10px; color: #ccc; }}
        .cal-pnl {{ font-size: 9px; font-weight: bold; color: #fff; }}
        .cal-total {{
            text-align: center;
            font-weight: bold;
            margin-top: 8px;
            font-size: 13px;
        }}
        table.monthly {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        table.monthly th, table.monthly td {{
            padding: 8px 12px;
            text-align: right;
            border-bottom: 1px solid #333;
        }}
        table.monthly th {{ color: #999; font-size: 12px; text-transform: uppercase; }}
        table.monthly td:first-child {{ text-align: left; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Combined Strategy</h1>
        <div class="subtitle">
            Base (lvl_delta>=50, SL={BASE_SL}/TP={BASE_TP}) + Trending (delta>=30, ADX>=30, SL={TREND_SL}/TP={TREND_TP})<br>
            {START_DATE} to {END_DATE} | IS: {dates_sorted[0]} -> {is_end_date} | OOS: {dates_sorted[split_idx]} -> {dates_sorted[-1]}
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total P&L</div>
                <div class="stat-value {pn(total_pnl)}">{fmt(total_pnl)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value">{win_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Profit Factor</div>
                <div class="stat-value {pn(pf - 1)}">{pf:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Drawdown</div>
                <div class="stat-value negative">{fmt(abs(max_dd))}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Risk-Adjusted Return</div>
                <div class="stat-value {pn(risk_adj)}">{risk_adj:.2f}</div>
            </div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Avg Winner</div>
                <div class="stat-value positive">{fmt(avg_win)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Loser</div>
                <div class="stat-value negative">{fmt(avg_loss)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg P&L/Trade</div>
                <div class="stat-value {pn(avg_pnl)}">{fmt(avg_pnl)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Duration</div>
                <div class="stat-value">{avg_dur:.0f} min</div>
            </div>
        </div>

        <div class="highlight">
            <h3 style="margin-top: 0; color: #4ec9b0;">Strategy Breakdown</h3>
            <div class="stats">
                {strat_card("Base (VWAP Zone)", base_df)}
                {strat_card("Trending (Std Band)", trend_df)}
            </div>
        </div>

        <div class="highlight">
            <h3 style="margin-top: 0; color: #4ec9b0;">IS / OOS Split</h3>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-label">In-Sample</div>
                    <div class="stat-value {pn(is_s['pnl'])}">{fmt(is_s['pnl'])}</div>
                    <div style="color:#999;font-size:12px;margin-top:5px;">{is_s['n']} trades | WR: {is_s['wr']:.1f}% | PF: {is_s['pf']:.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Out-of-Sample</div>
                    <div class="stat-value {pn(oos_s['pnl'])}">{fmt(oos_s['pnl'])}</div>
                    <div style="color:#999;font-size:12px;margin-top:5px;">{oos_s['n']} trades | WR: {oos_s['wr']:.1f}% | PF: {oos_s['pf']:.2f}</div>
                </div>
            </div>
        </div>

        <div class="highlight">
            <h3 style="margin-top: 0; color: #4ec9b0;">Exit Breakdown</h3>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-label">Target Hits</div>
                    <div class="stat-value positive">{len(tp_exits)} ({len(tp_exits)/total*100:.1f}%)</div>
                    <div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(tp_exits['pnl'].sum())}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Stop Losses</div>
                    <div class="stat-value negative">{len(sl_exits)} ({len(sl_exits)/total*100:.1f}%)</div>
                    <div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(sl_exits['pnl'].sum())}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">EOD Exits</div>
                    <div class="stat-value">{len(eod_exits)} ({len(eod_exits)/total*100:.1f}%)</div>
                    <div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(eod_exits['pnl'].sum())}</div>
                </div>
            </div>
        </div>

        <div id="equity-chart"></div>

        <script>
            var dates = {dates_json};
            var equity = {equity_json};
            var baseEq = {base_eq_json};
            var trendEq = {trend_eq_json};

            var combined = {{
                x: dates, y: equity, type: 'scatter', mode: 'lines',
                name: 'Combined', line: {{ color: '#4ec9b0', width: 3 }},
                fill: 'tozeroy', fillcolor: 'rgba(78, 201, 176, 0.1)'
            }};
            var baseLine = {{
                x: dates, y: baseEq, type: 'scatter', mode: 'lines',
                name: 'Base Only', line: {{ color: '#dcdcaa', width: 1.5, dash: 'dot' }},
                visible: 'legendonly'
            }};
            var trendLine = {{
                x: dates, y: trendEq, type: 'scatter', mode: 'lines',
                name: 'Trending Only', line: {{ color: '#ce9178', width: 1.5, dash: 'dot' }},
                visible: 'legendonly'
            }};
            var splitLine = {{
                x: ['{is_end_date}', '{is_end_date}'],
                y: [{STARTING_EQUITY - abs(max_dd)}, {STARTING_EQUITY + total_pnl + 5000}],
                type: 'scatter', mode: 'lines',
                name: 'IS/OOS Split', line: {{ color: '#ff8c00', width: 2, dash: 'dash' }}
            }};

            Plotly.newPlot('equity-chart', [combined, baseLine, trendLine, splitLine], {{
                title: {{ text: 'Equity Curve (Starting $50,000)', font: {{ color: '#e0e0e0', size: 16 }} }},
                xaxis: {{ title: 'Date', gridcolor: '#333', color: '#999' }},
                yaxis: {{ title: 'Equity ($)', gridcolor: '#333', color: '#999', tickformat: '$,.0f' }},
                plot_bgcolor: '#1e1e1e', paper_bgcolor: '#1e1e1e',
                font: {{ color: '#e0e0e0' }}, hovermode: 'x unified', height: 600,
                showlegend: true,
                legend: {{ x: 0.02, y: 0.98, bgcolor: 'rgba(45,45,45,0.8)', bordercolor: '#4ec9b0', borderwidth: 1 }}
            }}, {{responsive: true}});
        </script>

        <h2 style="color: #4ec9b0; text-align: center; margin-top: 40px;">PnL Calendar</h2>
        <div class="cal-grid">
            {cal_html}
        </div>

        <div class="highlight">
            <h3 style="margin-top: 0; color: #4ec9b0;">Monthly Summary</h3>
            <table class="monthly">
                <thead><tr><th style="text-align:left">Month</th><th>Trades</th><th>Base</th><th>Trend</th><th>PnL</th><th>WR%</th></tr></thead>
                <tbody>{monthly_rows}</tbody>
            </table>
        </div>

        <div style="margin-top: 30px; padding: 20px; background: #2d2d2d; border-radius: 8px;">
            <h3 style="color: #4ec9b0;">Strategy Configuration</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4 style="color: #dcdcaa;">Base Strategy</h4>
                    <ul style="line-height: 1.8;">
                        <li>Signal: Absorption at session VWAP zone (+/-3 pts)</li>
                        <li>Absorption: Per-level delta >= {BASE_LEVEL_DELTA}</li>
                        <li>Bias: Zone-based (5m close outside VWAP +/-3)</li>
                        <li>SL: {BASE_SL}x ATR | TP: {BASE_TP}x ATR</li>
                        <li>Slippage: Entry 0 ticks, SL 2 ticks</li>
                        <li>Lunch filter: {LUNCH_START}-{LUNCH_END} ET</li>
                    </ul>
                </div>
                <div>
                    <h4 style="color: #ce9178;">Trending Strategy</h4>
                    <ul style="line-height: 1.8;">
                        <li>Signal: Absorption at STD{STD_BAND} band during trend</li>
                        <li>Absorption: Per-level delta >= {TREND_DELTA}</li>
                        <li>Trend: {LOOKBACK} consecutive 5m closes above/below VWAP</li>
                        <li>ADX >= {TREND_ADX_MIN:.0f} filter</li>
                        <li>SL: {TREND_SL}x ATR | TP: {TREND_TP}x ATR</li>
                        <li>Approach gate: 5m close must have reached band</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved to {output_path}")


def main():
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    print("Building ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print("done")

    print("Loading data...", end=" ", flush=True)
    all_dates = []
    day_raw = {}
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
        if not vwap_file.exists() or not tb_file.exists():
            continue
        bars = load_pickle(sig_file).get("bars") or []
        vwap_df = load_pickle(vwap_file)
        bars_5min = load_5min_bars_fixed(ds)
        if bars and bars_5min is not None:
            all_dates.append(fd)
            day_raw[fd] = (ds, bars, vwap_df, bars_5min)
    print(f"done ({len(all_dates)} days)")

    print("Detecting signals...", end=" ", flush=True)
    all_trades = []
    for fd, (ds, bars, vwap_df, bars_5min) in sorted(day_raw.items()):
        base_sigs = detect_base_signals(bars, vwap_df, bars_5min)
        day_state = precompute_day_state(bars_5min, vwap_df)
        trend_sigs = detect_trending_signals(bars, day_state)
        combined = base_sigs + trend_sigs
        day_trades = simulate_day(bars, combined, adx_lookup)
        all_trades.extend(day_trades)
    print(f"done ({len(all_trades)} trades)")

    df = pd.DataFrame(all_trades)

    # Stats
    base_trades = df[df["strategy"] == "base"]
    trend_trades = df[df["strategy"] == "trending"]
    for label, sub in [("BASE", base_trades), ("TRENDING", trend_trades), ("COMBINED", df)]:
        n = len(sub)
        if n == 0: continue
        pnl = sub["pnl"].sum()
        wr = len(sub[sub["pnl"] > 0]) / n * 100
        w = sub[sub["pnl"] > 0]["pnl"].sum()
        l = abs(sub[sub["pnl"] < 0]["pnl"].sum()) or 1
        print(f"  {label:>10}: {n} trades, PnL ${pnl:+,.0f}, WR {wr:.1f}%, PF {w/l:.2f}")

    # Save CSV
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "combined_trades.csv", index=False)
    print(f"Saved trades to results/combined_trades.csv")

    # Generate HTML
    generate_html(df, Path("results/html/combined_equity_calendar.html"))


if __name__ == "__main__":
    main()

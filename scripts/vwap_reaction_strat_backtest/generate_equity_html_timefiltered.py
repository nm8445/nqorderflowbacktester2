"""
Generate equity curve HTML for VWAP Reaction strategy with time filters:
  - 9:30am - 4pm ET only
  - No lunch hour (12pm-1pm)
  - SL=1.0x / TP=2.0x ATR
"""

import pickle
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0
MAX_MARTINGALE = 8

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"

SL_MULT = 1.0
TP_MULT = 2.0


def get_nq_prices_for_buy_hold():
    timebars_dir = DATA_DIR / "timebars_5min"
    bar_files = sorted(timebars_dir.glob("timebars_5min_*.pkl"))
    if not bar_files:
        return None, None, None, None
    with open(bar_files[0], "rb") as f:
        first_bars = pickle.load(f)
    first_price = first_bars[0]["open"] if first_bars else None
    first_date = bar_files[0].stem.replace("timebars_5min_", "").replace("_", "-")
    with open(bar_files[-1], "rb") as f:
        last_bars = pickle.load(f)
    last_price = last_bars[-1]["close"] if last_bars else None
    last_date = bar_files[-1].stem.replace("timebars_5min_", "").replace("_", "-")
    return first_price, last_price, first_date, last_date


def run_backtest_with_time_filter():
    start = datetime.strptime("2025-03-13", "%Y-%m-%d").date()
    end = datetime.strptime("2026-04-08", "%Y-%m-%d").date()

    all_trades = []

    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue

        with open(cache_file, "rb") as f:
            vwap_data = pickle.load(f)

        signals = vwap_data["signals"]
        if not signals:
            continue

        signal_cache_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_cache_file.exists():
            continue

        with open(signal_cache_file, "rb") as f:
            signal_data = pickle.load(f)

        bars = signal_data["bars"]

        last_exit_time = None
        for signal in signals:
            entry_price = signal["entry_price"]
            direction = signal["direction"]
            atr = signal["atr"]
            confirm_bar_idx = signal["bar_index"] + 1

            if atr is None or atr <= 0:
                continue

            confirm_time = signal["confirm_time"]
            if hasattr(confirm_time, "tz_convert"):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz="UTC").tz_convert(ET)

            # Time filter: 9:30am-4pm ET, skip 12-1pm lunch
            entry_min = confirm_et.hour * 60 + confirm_et.minute
            if entry_min < 570 or entry_min >= 960:  # before 9:30 or after 4pm
                continue
            if 720 <= entry_min < 780:  # 12pm-1pm lunch
                continue

            if "16:00" <= confirm_et.strftime("%H:%M") < "19:10":
                continue

            if last_exit_time is not None and confirm_time <= last_exit_time:
                continue

            if direction == "long":
                sl_price = entry_price - atr * SL_MULT
                tp_price = entry_price + atr * TP_MULT
                if tp_price <= entry_price or sl_price >= entry_price:
                    continue
            else:
                sl_price = entry_price + atr * SL_MULT
                tp_price = entry_price - atr * TP_MULT
                if tp_price >= entry_price or sl_price <= entry_price:
                    continue

            exit_price = None
            exit_time = None
            exit_reason = None

            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue

                bar_ct = bar.close_time
                if hasattr(bar_ct, "tz_convert"):
                    bar_et = bar_ct.tz_convert(ET)
                else:
                    bar_et = pd.Timestamp(bar_ct, tz="UTC").tz_convert(ET)
                if bar_et.strftime("%H:%M") >= FORCE_CLOSE:
                    exit_price = bar.close
                    exit_time = bar.close_time
                    exit_reason = "session_close"
                    break

                if direction == "short":
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
                else:
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

            if exit_price is None:
                last_bar = bars[-1]
                exit_price = last_bar.close
                exit_time = last_bar.close_time
                exit_reason = "eod"

            if direction == "short":
                pnl_points = entry_price - exit_price
            else:
                pnl_points = exit_price - entry_price

            pnl_dollars = pnl_points * POINT_VALUE

            if hasattr(exit_time, "tz_convert"):
                exit_et = exit_time.tz_convert(ET)
            else:
                exit_et = pd.Timestamp(exit_time, tz="UTC").tz_convert(ET)

            duration_mins = (exit_et - confirm_et).total_seconds() / 60

            all_trades.append({
                "date": date_str,
                "entry_time": confirm_et.strftime("%Y-%m-%d %H:%M"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_points": pnl_points,
                "pnl_dollars": pnl_dollars,
                "duration_mins": duration_mins,
                "direction": direction,
                "atr": atr,
            })
            last_exit_time = exit_time

    return all_trades


def generate_html(trades, output_path):
    if not trades:
        print("No trades to chart.")
        return

    df = pd.DataFrame(trades)
    total = len(df)
    winners = df[df["pnl_dollars"] > 0]
    losers = df[df["pnl_dollars"] < 0]

    win_rate = len(winners) / total * 100
    avg_win = winners["pnl_dollars"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl_dollars"].mean() if len(losers) > 0 else 0
    avg_pnl = df["pnl_dollars"].mean()
    total_pnl = df["pnl_dollars"].sum()

    gross_profit = winners["pnl_dollars"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_dollars"].sum()) if len(losers) > 0 else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    cum = df["pnl_dollars"].cumsum()
    max_dd = float((cum - cum.cummax()).min())
    risk_adj = abs(total_pnl / max_dd) if max_dd != 0 else 0

    avg_dur = df["duration_mins"].mean()
    med_dur = df["duration_mins"].median()

    # Daily P&L for Sharpe
    df["date_parsed"] = pd.to_datetime(df["entry_time"]).dt.date
    daily_pnl = df.groupby("date_parsed")["pnl_dollars"].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0

    tp_exits = df[df["exit_reason"] == "target"]
    sl_exits = df[df["exit_reason"] == "stop_loss"]
    eod_exits = df[df["exit_reason"].isin(["eod", "session_close"])]

    shorts = df[df["direction"] == "short"]
    longs = df[df["direction"] == "long"]

    # Martingale equity
    marty_equity = []
    marty_contracts = 1
    marty_cum = 0.0
    marty_base_trades = 0
    marty_scaled_trades = 0
    marty_base_pnl = 0.0
    marty_scaled_pnl = 0.0

    for _, t in df.iterrows():
        pnl = t["pnl_dollars"] * marty_contracts
        marty_cum += pnl
        marty_equity.append(marty_cum)
        if marty_contracts == 1:
            marty_base_trades += 1
            marty_base_pnl += pnl
        else:
            marty_scaled_trades += 1
            marty_scaled_pnl += pnl
        if t["pnl_dollars"] > 0:
            marty_contracts = 1
        else:
            marty_contracts = min(marty_contracts * 2, MAX_MARTINGALE)

    marty_total = marty_cum
    marty_arr = np.array(marty_equity)
    marty_max_dd = float((marty_arr - np.maximum.accumulate(marty_arr)).min())

    strategy_equity = cum.tolist()
    marty_equity_list = [round(v, 2) for v in marty_equity]
    dates = df["entry_time"].tolist()

    # Buy & hold
    first_price, last_price, first_date, last_date = get_nq_prices_for_buy_hold()
    bh_equity = []
    timebars_dir = DATA_DIR / "timebars_5min"
    trade_dates = sorted(set(df["date"]))

    nq_prices = {}
    for td in trade_dates:
        td_fmt = td.replace("-", "_")
        bar_file = timebars_dir / f"timebars_5min_{td_fmt}.pkl"
        if bar_file.exists():
            with open(bar_file, "rb") as f:
                bars_5m = pickle.load(f)
            if bars_5m:
                nq_prices[td] = bars_5m[-1]["close"]

    if nq_prices and first_price:
        for d in df["date"]:
            if d in nq_prices:
                bh_equity.append(round((nq_prices[d] - first_price) * POINT_VALUE, 2))
            elif bh_equity:
                bh_equity.append(bh_equity[-1])
            else:
                bh_equity.append(0)
        bh_pnl = bh_equity[-1] if bh_equity else 0
    else:
        bh_equity = [0] * len(dates)
        bh_pnl = 0
        first_price = first_price or 0
        last_price = last_price or 0

    outperformance = total_pnl - bh_pnl
    bh_ratio = total_pnl / bh_pnl if bh_pnl != 0 else 0

    # Monthly P&L
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    monthly = df.groupby("month")["pnl_dollars"].sum()
    monthly_labels = [str(m) for m in monthly.index]
    monthly_values = [round(v, 2) for v in monthly.values]

    # Drawdown series
    dd_series = (cum - cum.cummax()).tolist()

    # IS/OOS split stats
    sp = total // 2
    is_df = df.iloc[:sp]
    oos_df = df.iloc[sp:]
    is_pnl = is_df["pnl_dollars"].sum()
    oos_pnl = oos_df["pnl_dollars"].sum()
    is_w = is_df[is_df["pnl_dollars"] > 0]
    is_l = is_df[is_df["pnl_dollars"] <= 0]
    oos_w = oos_df[oos_df["pnl_dollars"] > 0]
    oos_l = oos_df[oos_df["pnl_dollars"] <= 0]
    is_pf = is_w["pnl_dollars"].sum() / abs(is_l["pnl_dollars"].sum()) if len(is_l) else 999
    oos_pf = oos_w["pnl_dollars"].sum() / abs(oos_l["pnl_dollars"].sum()) if len(oos_l) else 999
    is_wr = len(is_w) / len(is_df) * 100
    oos_wr = len(oos_w) / len(oos_df) * 100

    def fmt(v):
        return f"${v:,.0f}"

    def cls(v):
        return "positive" if v >= 0 else "negative"

    dir_html = ""
    if len(shorts) > 0:
        s_wr = len(shorts[shorts["pnl_dollars"] > 0]) / len(shorts) * 100
        s_pnl = shorts["pnl_dollars"].sum()
        dir_html += f'<div class="stat-card"><div class="stat-label">Shorts</div><div class="stat-value">{len(shorts)}</div><div style="color:#999;font-size:12px;margin-top:5px;">WR: {s_wr:.1f}% | P&L: {fmt(s_pnl)}</div></div>'
    if len(longs) > 0:
        l_wr = len(longs[longs["pnl_dollars"] > 0]) / len(longs) * 100
        l_pnl = longs["pnl_dollars"].sum()
        dir_html += f'<div class="stat-card"><div class="stat-label">Longs</div><div class="stat-value">{len(longs)}</div><div style="color:#999;font-size:12px;margin-top:5px;">WR: {l_wr:.1f}% | P&L: {fmt(l_pnl)}</div></div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VWAP Reaction Strategy — Time Filtered</title>
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
    <h1>VWAP Reaction Continuation Strategy</h1>
    <div class="subtitle">40-range bars | VWAP zone: ±3 pts | SL: {SL_MULT}x ATR | TP: {TP_MULT}x ATR | 9:30am-4pm ET | No Lunch (12-1pm)</div>

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
        {dir_html}
    </div>

    <div class="highlight">
        <h3 style="margin-top:0;color:#4ec9b0;">Exit Breakdown</h3>
        <div class="stats" style="margin-top:15px;">
            <div class="stat-card"><div class="stat-label">Target Hits</div><div class="stat-value positive">{len(tp_exits)} ({len(tp_exits)/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(tp_exits['pnl_dollars'].sum())}</div></div>
            <div class="stat-card"><div class="stat-label">Stop Losses</div><div class="stat-value negative">{len(sl_exits)} ({len(sl_exits)/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(sl_exits['pnl_dollars'].sum())}</div></div>
            <div class="stat-card"><div class="stat-label">EOD / Session Close</div><div class="stat-value">{len(eod_exits)} ({len(eod_exits)/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(eod_exits['pnl_dollars'].sum())}</div></div>
        </div>
    </div>

    <div class="highlight">
        <h3 style="margin-top:0;color:#4ec9b0;">In-Sample / Out-of-Sample (50/50 Chronological)</h3>
        <div class="stats" style="margin-top:15px;">
            <div class="stat-card"><div class="stat-label">IS ({is_df['date'].iloc[0]} → {is_df['date'].iloc[-1]})</div><div class="stat-value {cls(is_pnl)}">{fmt(is_pnl)}</div><div style="color:#999;font-size:12px;margin-top:5px;">{len(is_df)} trades | WR {is_wr:.1f}% | PF {is_pf:.2f}</div></div>
            <div class="stat-card"><div class="stat-label">OOS ({oos_df['date'].iloc[0]} → {oos_df['date'].iloc[-1]})</div><div class="stat-value {cls(oos_pnl)}">{fmt(oos_pnl)}</div><div style="color:#999;font-size:12px;margin-top:5px;">{len(oos_df)} trades | WR {oos_wr:.1f}% | PF {oos_pf:.2f}</div></div>
        </div>
    </div>

    <div class="highlight">
        <h3 style="margin-top:0;color:#4ec9b0;">Martingale Breakdown (2x after loss, cap {MAX_MARTINGALE}x)</h3>
        <div class="stats" style="margin-top:15px;">
            <div class="stat-card"><div class="stat-label">Base Trades (1x)</div><div class="stat-value">{marty_base_trades} ({marty_base_trades/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(marty_base_pnl)}</div></div>
            <div class="stat-card"><div class="stat-label">Scaled Trades (2x+)</div><div class="stat-value">{marty_scaled_trades} ({marty_scaled_trades/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(marty_scaled_pnl)}</div></div>
            <div class="stat-card"><div class="stat-label">Martingale Total P&L</div><div class="stat-value {cls(marty_total)}">{fmt(marty_total)}</div></div>
            <div class="stat-card"><div class="stat-label">Martingale Max DD</div><div class="stat-value negative">{fmt(abs(marty_max_dd))}</div></div>
        </div>
    </div>

    <div class="highlight">
        <h3 style="margin-top:0;color:#4ec9b0;">Buy & Hold Comparison</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
            <div>
                <p style="margin:5px 0;"><strong>Buy & Hold P&L:</strong> {fmt(bh_pnl)}</p>
                <p style="margin:5px 0;color:#999;font-size:13px;">Entry: {first_price:,.2f} | Exit: {last_price:,.2f}</p>
            </div>
            <div>
                <p style="margin:5px 0;"><strong>Strategy Outperformance:</strong> <span class="{cls(outperformance)}">${outperformance:+,.0f}</span></p>
                <p style="margin:5px 0;color:#999;font-size:13px;">Strategy is {bh_ratio:.2f}x buy & hold</p>
            </div>
        </div>
    </div>

    <div id="equity-chart"></div>
    <div id="dd-chart"></div>
    <div id="monthly-chart"></div>

    <script>
        var dates = {json.dumps(dates)};
        var equity = {json.dumps([round(v, 2) for v in strategy_equity])};
        var martyEquity = {json.dumps(marty_equity_list)};
        var bh = {json.dumps(bh_equity)};
        var dd = {json.dumps([round(v, 2) for v in dd_series])};
        var monthLabels = {json.dumps(monthly_labels)};
        var monthValues = {json.dumps(monthly_values)};

        // IS/OOS split line
        var splitIdx = {sp};
        var splitDate = dates[splitIdx];

        Plotly.newPlot('equity-chart', [
            {{ x: dates, y: equity, type: 'scatter', mode: 'lines', name: 'Strategy (1x)',
               line: {{ color: '#4ec9b0', width: 3 }}, fill: 'tozeroy', fillcolor: 'rgba(78, 201, 176, 0.1)' }},
            {{ x: dates, y: martyEquity, type: 'scatter', mode: 'lines', name: 'Martingale',
               line: {{ color: '#dcdcaa', width: 2, dash: 'dot' }}, visible: 'legendonly' }},
            {{ x: dates, y: bh, type: 'scatter', mode: 'lines', name: 'Buy & Hold (1 NQ)',
               line: {{ color: '#ce9178', width: 2 }}, visible: 'legendonly' }}
        ], {{
            title: {{ text: 'Cumulative P&L', font: {{ color: '#e0e0e0', size: 16 }} }},
            xaxis: {{ gridcolor: '#333', color: '#999' }},
            yaxis: {{ title: 'P&L ($)', gridcolor: '#333', color: '#999', tickformat: '$,.0f' }},
            plot_bgcolor: '#1e1e1e', paper_bgcolor: '#1e1e1e',
            font: {{ color: '#e0e0e0' }}, hovermode: 'x unified', height: 600,
            shapes: [{{ type: 'line', x0: splitDate, x1: splitDate, y0: 0, y1: 1, yref: 'paper',
                        line: {{ color: '#f48771', width: 2, dash: 'dash' }} }}],
            annotations: [{{ x: splitDate, y: 1.05, yref: 'paper', text: 'IS | OOS', showarrow: false,
                             font: {{ color: '#f48771', size: 12 }} }}],
            legend: {{ x: 0.02, y: 0.98, bgcolor: 'rgba(45,45,45,0.8)', bordercolor: '#4ec9b0', borderwidth: 1 }}
        }}, {{ responsive: true }});

        Plotly.newPlot('dd-chart', [
            {{ x: dates, y: dd, type: 'scatter', mode: 'lines', name: 'Drawdown',
               line: {{ color: '#f48771', width: 2 }}, fill: 'tozeroy', fillcolor: 'rgba(244,135,113,0.15)' }}
        ], {{
            title: {{ text: 'Drawdown', font: {{ color: '#e0e0e0', size: 16 }} }},
            xaxis: {{ gridcolor: '#333', color: '#999' }},
            yaxis: {{ title: 'Drawdown ($)', gridcolor: '#333', color: '#999', tickformat: '$,.0f' }},
            plot_bgcolor: '#1e1e1e', paper_bgcolor: '#1e1e1e',
            font: {{ color: '#e0e0e0' }}, hovermode: 'x unified', height: 350
        }}, {{ responsive: true }});

        var monthColors = monthValues.map(v => v >= 0 ? '#4ec9b0' : '#f48771');
        Plotly.newPlot('monthly-chart', [
            {{ x: monthLabels, y: monthValues, type: 'bar', marker: {{ color: monthColors }},
               text: monthValues.map(v => '$' + v.toLocaleString('en-US', {{maximumFractionDigits: 0}})),
               textposition: 'outside', textfont: {{ color: '#e0e0e0', size: 11 }} }}
        ], {{
            title: {{ text: 'Monthly P&L', font: {{ color: '#e0e0e0', size: 16 }} }},
            xaxis: {{ gridcolor: '#333', color: '#999' }},
            yaxis: {{ title: 'P&L ($)', gridcolor: '#333', color: '#999', tickformat: '$,.0f' }},
            plot_bgcolor: '#1e1e1e', paper_bgcolor: '#1e1e1e',
            font: {{ color: '#e0e0e0' }}, height: 400
        }}, {{ responsive: true }});
    </script>

    <div style="margin-top:30px;padding:20px;background:#2d2d2d;border-radius:8px;">
        <h3 style="color:#4ec9b0;">Strategy Configuration</h3>
        <ul style="line-height:1.8;">
            <li><strong>Strategy:</strong> VWAP Reaction Continuation — Both Directions</li>
            <li><strong>Entry:</strong> Absorption at session VWAP zone (±3 pts), 5-min bias confirms direction</li>
            <li><strong>VWAP:</strong> Session VWAP anchored at 6pm ET, cumulative tick-by-tick</li>
            <li><strong>Stop Loss:</strong> {SL_MULT}x ATR (14-period, 5-min bars)</li>
            <li><strong>Take Profit:</strong> {TP_MULT}x ATR (14-period, 5-min bars)</li>
            <li><strong>Entry Window:</strong> 9:30am - 4:00pm ET (no lunch 12-1pm)</li>
            <li><strong>Bar Type:</strong> 40-range volumetric bars</li>
            <li><strong>Absorption:</strong> Delta >= 30 per level</li>
            <li><strong>Martingale:</strong> 2x after loss, cap at {MAX_MARTINGALE}x</li>
        </ul>
    </div>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved to {output_path}")


def main():
    print(f"Running time-filtered backtest: SL={SL_MULT}x TP={TP_MULT}x | 9:30-4pm | no lunch...")
    trades = run_backtest_with_time_filter()
    print(f"Got {len(trades)} trades")

    output = Path("results/html/vwap_reaction_timefiltered.html")
    generate_html(trades, output)


if __name__ == "__main__":
    main()

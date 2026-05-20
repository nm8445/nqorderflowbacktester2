"""
Generate equity curve HTML for Rough Vol 15-min strategy.
Reads from rough_vol_15min_trades.csv.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

import config_15min as cfg

DATA_DIR = cfg.CACHE_DIR
TRADES_CSV = DATA_DIR / "rough_vol_15min_trades.csv"
OUTPUT = Path("results/html/rough_vol_15min_equity.html")


def generate_html():
    df = pd.read_csv(TRADES_CSV)
    df["entry_dt"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert("America/New_York")
    df["exit_dt"] = pd.to_datetime(df["exit_time"], utc=True).dt.tz_convert("America/New_York")

    total = len(df)
    winners = df[df["total_pnl"] > 0]
    losers = df[df["total_pnl"] <= 0]

    win_rate = len(winners) / total * 100
    avg_win = winners["total_pnl"].mean() if len(winners) else 0
    avg_loss = losers["total_pnl"].mean() if len(losers) else 0
    avg_pnl = df["total_pnl"].mean()
    total_pnl = df["total_pnl"].sum()

    gross_profit = winners["total_pnl"].sum() if len(winners) else 0
    gross_loss = abs(losers["total_pnl"].sum()) if len(losers) else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    cum = df["total_pnl"].cumsum()
    max_dd = float((cum - cum.cummax()).min())
    risk_adj = abs(total_pnl / max_dd) if max_dd != 0 else 0

    duration_mins = (df["exit_dt"] - df["entry_dt"]).dt.total_seconds() / 60
    avg_dur = duration_mins.mean()
    med_dur = duration_mins.median()

    # Sharpe
    df["date"] = df["entry_dt"].dt.date
    daily_sum = df.groupby("date")["total_pnl"].sum()
    sharpe = (daily_sum.mean() / daily_sum.std()) * np.sqrt(252) if daily_sum.std() > 0 else 0

    # Exit breakdown
    tp_exits = df[df["exit_reason"] == "TP"]
    sl_exits = df[df["exit_reason"] == "SL"]
    sess_exits = df[df["exit_reason"] == "SESSION"]

    # Direction breakdown
    shorts = df[df["direction"] == "SHORT"]
    longs = df[df["direction"] == "LONG"]

    # Monthly P&L
    df["month"] = df["entry_dt"].dt.to_period("M")
    monthly = df.groupby("month")["total_pnl"].sum()
    monthly_labels = [str(m) for m in monthly.index]
    monthly_values = [round(v, 2) for v in monthly.values]

    # Drawdown series
    dd_series = (cum - cum.cummax()).tolist()

    dates = df["entry_dt"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    strategy_equity = [round(v, 2) for v in cum.tolist()]

    def fmt(v):
        return f"${v:,.0f}"

    def cls(v):
        return "positive" if v >= 0 else "negative"

    dir_html = ""
    if len(shorts):
        s_wr = len(shorts[shorts["total_pnl"] > 0]) / len(shorts) * 100
        s_pnl = shorts["total_pnl"].sum()
        dir_html += f'<div class="stat-card"><div class="stat-label">Shorts</div><div class="stat-value">{len(shorts)}</div><div style="color:#999;font-size:12px;margin-top:5px;">WR: {s_wr:.1f}% | P&L: {fmt(s_pnl)}</div></div>'
    if len(longs):
        l_wr = len(longs[longs["total_pnl"] > 0]) / len(longs) * 100
        l_pnl = longs["total_pnl"].sum()
        dir_html += f'<div class="stat-card"><div class="stat-label">Longs</div><div class="stat-value">{len(longs)}</div><div style="color:#999;font-size:12px;margin-top:5px;">WR: {l_wr:.1f}% | P&L: {fmt(l_pnl)}</div></div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Rough Vol 15-Min Strategy</title>
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
    <h1>Rough Volatility 15-Min Strategy</h1>
    <div class="subtitle">H={cfg.H} | ETA={cfg.ETA} | HIGH_Z={cfg.HIGH_Z} | EMA={cfg.EMA_LEN} | ATR SL={cfg.ATR_SL}x / TP={cfg.ATR_TP}x | Session: {cfg.SESSION_START}h-{cfg.SESSION_END}h ET</div>

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
        <div class="stat-card"><div class="stat-label">Risk-Adj Return</div><div class="stat-value {cls(risk_adj)}">{risk_adj:.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Avg Duration</div><div class="stat-value">{avg_dur:.0f} min</div><div style="color:#999;font-size:12px;margin-top:5px;">Median: {med_dur:.0f} min</div></div>
        {dir_html}
    </div>

    <div class="highlight">
        <h3 style="margin-top:0;color:#4ec9b0;">Exit Breakdown</h3>
        <div class="stats" style="margin-top:15px;">
            <div class="stat-card"><div class="stat-label">Take Profit</div><div class="stat-value positive">{len(tp_exits)} ({len(tp_exits)/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(tp_exits['total_pnl'].sum())}</div></div>
            <div class="stat-card"><div class="stat-label">Stop Loss</div><div class="stat-value negative">{len(sl_exits)} ({len(sl_exits)/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(sl_exits['total_pnl'].sum())}</div></div>
            <div class="stat-card"><div class="stat-label">Session Close</div><div class="stat-value">{len(sess_exits)} ({len(sess_exits)/total*100:.1f}%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: {fmt(sess_exits['total_pnl'].sum())}</div></div>
        </div>
    </div>

    <div id="equity-chart"></div>
    <div id="dd-chart"></div>
    <div id="monthly-chart"></div>

    <script>
        var dates = {json.dumps(dates)};
        var equity = {json.dumps(strategy_equity)};
        var dd = {json.dumps([round(v, 2) for v in dd_series])};
        var monthLabels = {json.dumps(monthly_labels)};
        var monthValues = {json.dumps(monthly_values)};

        Plotly.newPlot('equity-chart', [{{
            x: dates, y: equity, type: 'scatter', mode: 'lines',
            name: 'Equity', line: {{ color: '#4ec9b0', width: 3 }},
            fill: 'tozeroy', fillcolor: 'rgba(78, 201, 176, 0.1)'
        }}], {{
            title: {{ text: 'Cumulative P&L', font: {{ color: '#e0e0e0', size: 16 }} }},
            xaxis: {{ gridcolor: '#333', color: '#999' }},
            yaxis: {{ title: 'P&L ($)', gridcolor: '#333', color: '#999', tickformat: '$,.0f' }},
            plot_bgcolor: '#1e1e1e', paper_bgcolor: '#1e1e1e',
            font: {{ color: '#e0e0e0' }}, hovermode: 'x unified', height: 500
        }}, {{ responsive: true }});

        Plotly.newPlot('dd-chart', [{{
            x: dates, y: dd, type: 'scatter', mode: 'lines',
            name: 'Drawdown', line: {{ color: '#f48771', width: 2 }},
            fill: 'tozeroy', fillcolor: 'rgba(244, 135, 113, 0.15)'
        }}], {{
            title: {{ text: 'Drawdown', font: {{ color: '#e0e0e0', size: 16 }} }},
            xaxis: {{ gridcolor: '#333', color: '#999' }},
            yaxis: {{ title: 'Drawdown ($)', gridcolor: '#333', color: '#999', tickformat: '$,.0f' }},
            plot_bgcolor: '#1e1e1e', paper_bgcolor: '#1e1e1e',
            font: {{ color: '#e0e0e0' }}, hovermode: 'x unified', height: 350
        }}, {{ responsive: true }});

        var monthColors = monthValues.map(v => v >= 0 ? '#4ec9b0' : '#f48771');
        Plotly.newPlot('monthly-chart', [{{
            x: monthLabels, y: monthValues, type: 'bar',
            marker: {{ color: monthColors }},
            text: monthValues.map(v => '$' + v.toLocaleString('en-US', {{maximumFractionDigits: 0}})),
            textposition: 'outside', textfont: {{ color: '#e0e0e0', size: 11 }}
        }}], {{
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
            <li><strong>Model:</strong> Rough Volatility (Hurst H={cfg.H}, ETA={cfg.ETA}, V0={cfg.V0})</li>
            <li><strong>Entry:</strong> z_vol > {cfg.HIGH_Z} + EMA({cfg.EMA_LEN}) trend filter</li>
            <li><strong>Exit:</strong> ATR({cfg.ATR_LEN}) SL={cfg.ATR_SL}x / TP={cfg.ATR_TP}x, session close at {cfg.SESSION_END}h ET</li>
            <li><strong>Session:</strong> {cfg.SESSION_START}h - {cfg.SESSION_END}h ET</li>
            <li><strong>Bars:</strong> 15-minute OHLC</li>
            <li><strong>Martingale:</strong> streak={cfg.MARTINGALE_STREAK}, mult={cfg.MARTINGALE_MULTIPLE}x, max={cfg.MAX_DOUBLES}</li>
        </ul>
    </div>
</div>
</body>
</html>"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    generate_html()

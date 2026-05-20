"""
Generate equity curve HTML for VWAP Reaction Continuation strategy.
Format: matches equity_curve_locked.html template with buy & hold comparison.
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


def get_nq_prices_for_buy_hold():
    """Get NQ price at first and last trade dates for buy & hold calc."""
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


def run_backtest(sl_mult, tp_mult, direction_filter=None):
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

        filtered_signals = signals
        if direction_filter:
            filtered_signals = [s for s in signals if s["direction"] == direction_filter]

        if not filtered_signals:
            continue

        last_exit_time = None
        for signal in filtered_signals:
            entry_price = signal["entry_price"]
            direction = signal["direction"]
            atr = signal["atr"]
            confirm_bar_idx = signal["bar_index"] + 1

            if atr is None or atr <= 0:
                continue

            # No new entries after 4:50 PM ET
            confirm_time = signal["confirm_time"]
            if hasattr(confirm_time, "tz_convert"):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz="UTC").tz_convert(ET)
            if "16:00" <= confirm_et.strftime("%H:%M") < "19:10":
                continue

            # Skip if previous trade hasn't exited yet
            if last_exit_time is not None and confirm_time <= last_exit_time:
                continue

            if direction == "long":
                sl_price = entry_price - atr * sl_mult
                tp_price = entry_price + atr * tp_mult
                if tp_price <= entry_price or sl_price >= entry_price:
                    continue
            else:
                sl_price = entry_price + atr * sl_mult
                tp_price = entry_price - atr * tp_mult
                if tp_price >= entry_price or sl_price <= entry_price:
                    continue

            exit_price = None
            exit_time = None
            exit_reason = None

            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue

                # Force close at 4:58 PM ET
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

            if exit_price is None and exit_time is None:
                last_bar = bars[-1]
                exit_price = last_bar.close
                exit_time = last_bar.close_time
                exit_reason = "eod"

            if direction == "short":
                pnl_points = entry_price - exit_price
            else:
                pnl_points = exit_price - entry_price

            pnl_dollars = pnl_points * POINT_VALUE

            confirm_time = signal["confirm_time"]
            if hasattr(confirm_time, "tz_convert"):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz="UTC").tz_convert(ET)

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


def generate_html(trades, sl_mult, tp_mult, output_path, direction_filter=None):
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

    tp_exits = df[df["exit_reason"] == "target"]
    sl_exits = df[df["exit_reason"] == "stop_loss"]
    eod_exits = df[df["exit_reason"] == "eod"]

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
        has_buy_hold = True
    else:
        bh_equity = [0] * len(dates)
        bh_pnl = 0
        has_buy_hold = False
        first_price = first_price or 0
        last_price = last_price or 0

    outperformance = total_pnl - bh_pnl
    bh_ratio = total_pnl / bh_pnl if bh_pnl != 0 else 0

    def fmt_dollar(v):
        return f"${v:,.0f}"

    def pos_neg_class(v):
        return "positive" if v >= 0 else "negative"

    dir_label = direction_filter.upper() if direction_filter else "Both Directions"
    title = f"VWAP Reaction Continuation Strategy - {dir_label}"

    # Direction stats
    dir_html = ""
    if len(shorts) > 0:
        s_wr = len(shorts[shorts['pnl_dollars'] > 0]) / len(shorts) * 100
        s_pnl = shorts['pnl_dollars'].sum()
        dir_html += f"""
            <div class="stat-card">
                <div class="stat-label">Shorts</div>
                <div class="stat-value">{len(shorts)}</div>
                <div style="color: #999; font-size: 12px; margin-top: 5px;">WR: {s_wr:.1f}% | P&L: {fmt_dollar(s_pnl)}</div>
            </div>"""
    if len(longs) > 0:
        l_wr = len(longs[longs['pnl_dollars'] > 0]) / len(longs) * 100
        l_pnl = longs['pnl_dollars'].sum()
        dir_html += f"""
            <div class="stat-card">
                <div class="stat-label">Longs</div>
                <div class="stat-value">{len(longs)}</div>
                <div style="color: #999; font-size: 12px; margin-top: 5px;">WR: {l_wr:.1f}% | P&L: {fmt_dollar(l_pnl)}</div>
            </div>"""

    dates_json = json.dumps(dates)
    strategy_json = json.dumps([round(v, 2) for v in strategy_equity])
    bh_json = json.dumps(bh_equity)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #1e1e1e;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #4ec9b0;
            text-align: center;
        }}
        .subtitle {{
            text-align: center;
            color: #a0a0a0;
            margin-bottom: 20px;
        }}
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
        .stat-label {{
            color: #999;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .stat-value {{
            color: #fff;
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .positive {{ color: #4ec9b0; }}
        .negative {{ color: #f48771; }}
        .highlight {{
            background: #2d4a3e;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #4ec9b0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="subtitle">40-range bars | VWAP zone: +/-3 pts | SL: {sl_mult}x ATR | TP: {tp_mult}x ATR | 7pm-4:59pm ET</div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total P&L</div>
                <div class="stat-value {pos_neg_class(total_pnl)}">{fmt_dollar(total_pnl)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value {pos_neg_class(win_rate - 50)}">{win_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Profit Factor</div>
                <div class="stat-value {pos_neg_class(pf - 1)}">{pf:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Drawdown</div>
                <div class="stat-value negative">{fmt_dollar(abs(max_dd))}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Risk-Adjusted Return</div>
                <div class="stat-value {pos_neg_class(risk_adj)}">{risk_adj:.2f}</div>
            </div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Avg Winner</div>
                <div class="stat-value positive">{fmt_dollar(avg_win)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Loser</div>
                <div class="stat-value negative">{fmt_dollar(avg_loss)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg P&L/Trade</div>
                <div class="stat-value {pos_neg_class(avg_pnl)}">{fmt_dollar(avg_pnl)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Duration</div>
                <div class="stat-value">{avg_dur:.0f} min</div>
                <div style="color: #999; font-size: 12px; margin-top: 5px;">Median: {med_dur:.0f} min</div>
            </div>
            {dir_html}
        </div>

        <div class="highlight">
            <h3 style="margin-top: 0; color: #4ec9b0;">Exit Breakdown</h3>
            <div class="stats" style="margin-top: 15px;">
                <div class="stat-card">
                    <div class="stat-label">Target Hits</div>
                    <div class="stat-value positive">{len(tp_exits)} ({len(tp_exits)/total*100:.1f}%)</div>
                    <div style="color: #999; font-size: 12px; margin-top: 5px;">P&L: {fmt_dollar(tp_exits['pnl_dollars'].sum())}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Stop Losses</div>
                    <div class="stat-value negative">{len(sl_exits)} ({len(sl_exits)/total*100:.1f}%)</div>
                    <div style="color: #999; font-size: 12px; margin-top: 5px;">P&L: {fmt_dollar(sl_exits['pnl_dollars'].sum())}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">EOD Exits</div>
                    <div class="stat-value">{len(eod_exits)} ({len(eod_exits)/total*100:.1f}%)</div>
                    <div style="color: #999; font-size: 12px; margin-top: 5px;">P&L: {fmt_dollar(eod_exits['pnl_dollars'].sum())}</div>
                </div>
            </div>
        </div>

        <div class="highlight">
            <h3 style="margin-top: 0; color: #4ec9b0;">Martingale Breakdown</h3>
            <div class="stats" style="margin-top: 15px;">
                <div class="stat-card">
                    <div class="stat-label">Base Trades (1x)</div>
                    <div class="stat-value">{marty_base_trades} ({marty_base_trades/total*100:.1f}%)</div>
                    <div style="color: #999; font-size: 12px; margin-top: 5px;">P&L: {fmt_dollar(marty_base_pnl)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Martingale (2x+)</div>
                    <div class="stat-value">{marty_scaled_trades} ({marty_scaled_trades/total*100:.1f}%)</div>
                    <div style="color: #999; font-size: 12px; margin-top: 5px;">P&L: {fmt_dollar(marty_scaled_pnl)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Martingale Total P&L</div>
                    <div class="stat-value {pos_neg_class(marty_total)}">{fmt_dollar(marty_total)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Martingale Max DD</div>
                    <div class="stat-value negative">{fmt_dollar(abs(marty_max_dd))}</div>
                </div>
            </div>
        </div>

        <div class="highlight">
            <h3 style="margin-top: 0; color: #4ec9b0;">Buy & Hold Comparison</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <p style="margin: 5px 0;"><strong>Buy & Hold P&L:</strong> {fmt_dollar(bh_pnl)}</p>
                    <p style="margin: 5px 0; color: #999; font-size: 13px;">Entry: {first_price:,.2f} | Exit: {last_price:,.2f}</p>
                </div>
                <div>
                    <p style="margin: 5px 0;"><strong>Strategy Outperformance:</strong> <span class="{pos_neg_class(outperformance)}">${outperformance:+,.0f}</span></p>
                    <p style="margin: 5px 0; color: #999; font-size: 13px;">Strategy is {bh_ratio:.2f}x buy & hold</p>
                </div>
            </div>
            <p style="margin-top: 15px; color: #a0a0a0; font-size: 13px;"><em>Click legend items to toggle strategy vs buy & hold on the chart</em></p>
        </div>

        <div id="equity-chart"></div>

        <script>
            var dates = {dates_json};
            var strategyEquity = {strategy_json};
            var buyHoldEquity = {bh_json};
            var buyHoldTotal = {bh_pnl:.2f};

            var strategyTrace = {{
                x: dates,
                y: strategyEquity,
                type: 'scatter',
                mode: 'lines',
                name: 'Strategy',
                line: {{
                    color: '#4ec9b0',
                    width: 3
                }},
                fill: 'tozeroy',
                fillcolor: 'rgba(78, 201, 176, 0.1)',
                visible: true
            }};

            var buyHoldTrace = {{
                x: dates,
                y: buyHoldEquity,
                type: 'scatter',
                mode: 'lines',
                name: 'Buy & Hold (1 NQ)',
                line: {{
                    color: '#ce9178',
                    width: 2
                }},
                visible: 'legendonly'
            }};

            var traces = [strategyTrace, buyHoldTrace];

            var annotations = [{{
                x: 0.5,
                y: 1.08,
                xref: 'paper',
                yref: 'paper',
                text: 'Strategy: {fmt_dollar(total_pnl)} | Buy & Hold: $' + buyHoldTotal.toLocaleString('en-US', {{maximumFractionDigits: 0}}) + ' | Outperformance: ${outperformance:+,.0f}',
                showarrow: false,
                font: {{
                    size: 14,
                    color: '#4ec9b0'
                }}
            }}];

            var layout = {{
                title: {{
                    text: 'Cumulative P&L Over Time',
                    font: {{ color: '#e0e0e0', size: 16 }}
                }},
                xaxis: {{
                    title: 'Date',
                    gridcolor: '#333',
                    color: '#999'
                }},
                yaxis: {{
                    title: 'Cumulative P&L ($)',
                    gridcolor: '#333',
                    color: '#999',
                    tickformat: '$,.0f'
                }},
                plot_bgcolor: '#1e1e1e',
                paper_bgcolor: '#1e1e1e',
                font: {{ color: '#e0e0e0' }},
                hovermode: 'x unified',
                height: 600,
                showlegend: true,
                legend: {{
                    x: 0.02,
                    y: 0.98,
                    bgcolor: 'rgba(45, 45, 45, 0.8)',
                    bordercolor: '#4ec9b0',
                    borderwidth: 1
                }},
                annotations: annotations
            }};

            Plotly.newPlot('equity-chart', traces, layout, {{responsive: true}});
        </script>

        <div style="margin-top: 30px; padding: 20px; background: #2d2d2d; border-radius: 8px;">
            <h3 style="color: #4ec9b0;">Strategy Configuration</h3>
            <ul style="line-height: 1.8;">
                <li><strong>Strategy:</strong> VWAP Reaction Continuation - {dir_label}</li>
                <li><strong>Entry:</strong> Absorption at session VWAP zone (+/-3 pts), 5-min bias confirms direction</li>
                <li><strong>VWAP:</strong> Session VWAP anchored at 6pm ET, cumulative tick-by-tick</li>
                <li><strong>Bias Filter:</strong> Most recent 5-min close vs VWAP determines long/short bias</li>
                <li><strong>Confirmation:</strong> Next bar closes in direction of bias</li>
                <li><strong>Stop Loss:</strong> {sl_mult}x ATR (14-period, 5-min bars) from entry</li>
                <li><strong>Take Profit:</strong> {tp_mult}x ATR (14-period, 5-min bars) from entry</li>
                <li><strong>Martingale:</strong> 2x after loss, cap at {MAX_MARTINGALE}x</li>
                <li><strong>Entry Window:</strong> 7pm - 4:59pm ET</li>
                <li><strong>Bar Type:</strong> 40-range volumetric bars</li>
                <li><strong>Absorption Threshold:</strong> Delta >= 30 per level</li>
            </ul>
        </div>
    </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    SL_MULT = 1.0
    TP_MULT = 2.0

    print(f"Running backtest: SL={SL_MULT}x ATR, TP={TP_MULT}x ATR, both directions")
    trades = run_backtest(SL_MULT, TP_MULT)
    print(f"Got {len(trades)} trades")

    output = Path("results/html/vwap_reaction_sl1_tp2.html")
    generate_html(trades, SL_MULT, TP_MULT, output)

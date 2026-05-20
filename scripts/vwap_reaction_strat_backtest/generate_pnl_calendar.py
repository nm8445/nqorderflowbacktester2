"""
Generate P&L Calendar HTML for VWAP Reaction strategy.
Config: SL 1.0x / TP 2.0x ATR, no martingale.
Output format matches pnl_calendar.html template with MNQ/NQ switcher.
"""

import pickle
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"
SL_MULT = 1.0
TP_MULT = 2.0


def collect_daily_trades():
    """Return dict of date_str -> [pnl_points, ...] for each trade."""
    start = datetime.strptime("2025-03-13", "%Y-%m-%d").date()
    end = datetime.strptime("2026-04-08", "%Y-%m-%d").date()

    daily = {}

    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue

        with open(cache_file, 'rb') as f:
            vwap_data = pickle.load(f)
        signals = vwap_data["signals"]
        if not signals:
            continue

        signal_cache_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_cache_file.exists():
            continue
        with open(signal_cache_file, 'rb') as f:
            signal_data = pickle.load(f)
        bars = signal_data["bars"]

        day_pnls = []
        last_exit_time = None

        for signal in signals:
            entry_price = signal["entry_price"]
            direction = signal["direction"]
            atr = signal["atr"]
            confirm_bar_idx = signal["bar_index"] + 1

            if atr is None or atr <= 0:
                continue

            confirm_time = signal["confirm_time"]
            if hasattr(confirm_time, 'tz_convert'):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz='UTC').tz_convert(ET)
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
            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                bar_ct = bar.close_time
                if hasattr(bar_ct, 'tz_convert'):
                    bar_et = bar_ct.tz_convert(ET)
                else:
                    bar_et = pd.Timestamp(bar_ct, tz='UTC').tz_convert(ET)
                if bar_et.strftime("%H:%M") >= FORCE_CLOSE:
                    exit_price = bar.close
                    exit_time = bar.close_time
                    break
                if direction == "short":
                    if bar.high >= sl_price:
                        exit_price = sl_price; exit_time = bar.close_time; break
                    if bar.low <= tp_price:
                        exit_price = tp_price; exit_time = bar.close_time; break
                else:
                    if bar.low <= sl_price:
                        exit_price = sl_price; exit_time = bar.close_time; break
                    if bar.high >= tp_price:
                        exit_price = tp_price; exit_time = bar.close_time; break

            if exit_price is None:
                last_bar = bars[-1]
                exit_price = last_bar.close
                exit_time = last_bar.close_time

            if direction == "short":
                pnl = entry_price - exit_price
            else:
                pnl = exit_price - entry_price

            # Store as NQ dollar P&L (1 NQ contract = $20/point)
            day_pnls.append(round(pnl * POINT_VALUE, 2))
            last_exit_time = exit_time

        if day_pnls:
            daily[date_str] = day_pnls

    return daily


def generate_html(daily_trades, output_path):
    """Generate the calendar HTML."""
    # Compute summary stats for 1 NQ
    total_trades = sum(len(v) for v in daily_trades.values())
    all_pnl = [p for trades in daily_trades.values() for p in trades]
    total_pnl = sum(all_pnl)
    winners = [p for p in all_pnl if p > 0]
    win_rate = len(winners) / len(all_pnl) * 100 if all_pnl else 0

    dates = sorted(daily_trades.keys())
    start_month = dates[0][:7] if dates else "2025-03"

    trades_json = json.dumps(daily_trades)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>VWAP Reaction Strategy - P&L Calendar</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1e1e1e;
            color: #fff;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #4ec9b0;
        }}

        .subtitle {{
            text-align: center;
            margin-bottom: 20px;
            color: #a0a0a0;
            font-size: 14px;
        }}

        .strategy-note {{
            background: #2d2d30;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #4ec9b0;
        }}

        .strategy-note h3 {{
            margin-bottom: 10px;
            color: #4ec9b0;
            font-size: 16px;
        }}

        .strategy-note p {{
            color: #a0a0a0;
            font-size: 13px;
            margin-bottom: 5px;
        }}

        .contract-controls {{
            background: #2d2d30;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}

        .contract-controls h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            color: #4ec9b0;
        }}

        .control-row {{
            display: flex;
            gap: 30px;
            align-items: center;
            margin-bottom: 15px;
        }}

        .radio-group {{
            display: flex;
            gap: 20px;
        }}

        .radio-option {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        .slider-container {{
            display: flex;
            align-items: center;
            gap: 15px;
            flex: 1;
        }}

        .slider {{
            flex: 1;
            max-width: 300px;
        }}

        .slider-value {{
            font-size: 18px;
            font-weight: bold;
            color: #4ec9b0;
            min-width: 120px;
        }}

        .controls {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            background: #2d2d30;
            padding: 20px;
            border-radius: 8px;
        }}

        .nav-buttons button {{
            background: #3e3e42;
            color: #fff;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }}

        .nav-buttons button:hover {{
            background: #505050;
        }}

        .nav-buttons button:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}

        .month-title {{
            font-size: 24px;
            font-weight: bold;
        }}

        .month-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: #2d2d30;
            padding: 15px;
            border-radius: 8px;
        }}

        .stat-label {{
            font-size: 12px;
            color: #a0a0a0;
            margin-bottom: 5px;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: bold;
        }}

        .positive {{ color: #4ec9b0; }}
        .negative {{ color: #f48771; }}
        .neutral {{ color: #a0a0a0; }}

        .calendar {{
            background: #2d2d30;
            border-radius: 8px;
            padding: 20px;
        }}

        .calendar-header {{
            display: grid;
            grid-template-columns: repeat(7, 1fr) 120px;
            gap: 5px;
            margin-bottom: 10px;
            font-weight: bold;
            color: #a0a0a0;
        }}

        .calendar-row {{
            display: grid;
            grid-template-columns: repeat(7, 1fr) 120px;
            gap: 5px;
            margin-bottom: 5px;
        }}

        .calendar-day {{
            background: #3e3e42;
            padding: 10px;
            border-radius: 4px;
            min-height: 80px;
            position: relative;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .calendar-day:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}

        .calendar-day.empty {{
            background: transparent;
            cursor: default;
        }}

        .calendar-day.empty:hover {{
            transform: none;
            box-shadow: none;
        }}

        .calendar-day.profit {{
            background: linear-gradient(135deg, #2d4a3e 0%, #3e3e42 100%);
            border-left: 3px solid #4ec9b0;
        }}

        .calendar-day.loss {{
            background: linear-gradient(135deg, #4a2d2d 0%, #3e3e42 100%);
            border-left: 3px solid #f48771;
        }}

        .day-number {{
            font-size: 14px;
            color: #a0a0a0;
            margin-bottom: 5px;
        }}

        .day-pnl {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 3px;
        }}

        .day-trades {{
            font-size: 11px;
            color: #a0a0a0;
        }}

        .week-total {{
            background: #2d2d30;
            padding: 10px;
            border-radius: 4px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-weight: bold;
        }}

        .week-label {{
            font-size: 10px;
            color: #a0a0a0;
            margin-bottom: 3px;
        }}

        input[type="range"] {{
            width: 100%;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>VWAP Reaction Strategy - P&L Calendar</h1>
        <div class="subtitle">SL 1.0x ATR / TP 2.0x ATR | {total_trades} trades | {win_rate:.1f}% WR | Entry cutoff 4:50 PM, Force close 4:58 PM</div>

        <div class="strategy-note">
            <h3>Strategy: VWAP Reaction Continuation</h3>
            <p><strong>Entry:</strong> Absorption signal + confirmation at VWAP zone (+/-3 pts), 5-min bias confirms direction</p>
            <p><strong>ATR:</strong> 14-period ATR from 5-min bars</p>
            <p><strong>Stop:</strong> 1.0x ATR</p>
            <p><strong>Target:</strong> 2.0x ATR</p>
            <p><strong>Session:</strong> 7 PM - 4:50 PM ET (force close 4:58 PM)</p>
            <p><strong>Total P&L (1 NQ):</strong> ${total_pnl:,.0f} ({total_trades} trades)</p>
        </div>

        <div class="contract-controls">
            <h3>Contract Settings</h3>
            <div class="control-row">
                <div>
                    <div class="radio-group">
                        <div class="radio-option">
                            <input type="radio" id="mnq" name="contractType" value="mnq" checked>
                            <label for="mnq">MNQ ($2/point)</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" id="nq" name="contractType" value="nq">
                            <label for="nq">NQ ($20/point)</label>
                        </div>
                    </div>
                </div>
                <div class="slider-container">
                    <label>Contracts:</label>
                    <input type="range" id="contractSize" class="slider" min="1" max="5" value="2" step="1">
                    <div class="slider-value" id="contractSizeValue">2</div>
                </div>
            </div>
        </div>

        <div class="controls">
            <div class="nav-buttons">
                <button id="prevMonth">&larr; Previous</button>
                <button id="nextMonth">Next &rarr;</button>
            </div>
            <div class="month-title" id="monthTitle"></div>
        </div>

        <div class="month-stats">
            <div class="stat-card">
                <div class="stat-label">Month P&L</div>
                <div class="stat-value" id="monthPnl">$0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Trading Days</div>
                <div class="stat-value" id="tradingDays">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Daily P&L</div>
                <div class="stat-value" id="avgDaily">$0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Year Total</div>
                <div class="stat-value positive" id="yearTotal">$0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Daily DD</div>
                <div class="stat-value negative" id="maxDailyDD">$0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max DD</div>
                <div class="stat-value negative" id="maxDD">$0</div>
            </div>
        </div>

        <div class="calendar">
            <div class="calendar-header">
                <div>Sunday</div>
                <div>Monday</div>
                <div>Tuesday</div>
                <div>Wednesday</div>
                <div>Thursday</div>
                <div>Friday</div>
                <div>Saturday</div>
                <div>Week Total</div>
            </div>
            <div id="calendarBody"></div>
        </div>
    </div>

    <script>
        const tradesData = {trades_json};

        const startMonth = '{start_month}';

        let currentMonth = startMonth;
        let contractType = 'mnq';
        let numContracts = 2;

        function calculatePnl(nqDollarPnl) {{
            // nqDollarPnl is P&L for 1 NQ contract in dollars
            // MNQ is 1/10th the size
            const isMNQ = contractType === 'mnq';
            const multiplier = isMNQ ? 0.1 : 1;
            return nqDollarPnl * multiplier * numContracts;
        }}

        function renderCalendar() {{
            const [year, month] = currentMonth.split('-').map(Number);
            const firstDay = new Date(year, month - 1, 1);
            const lastDay = new Date(year, month, 0);
            const daysInMonth = lastDay.getDate();
            const startDay = firstDay.getDay();

            document.getElementById('monthTitle').textContent =
                firstDay.toLocaleString('default', {{ month: 'long', year: 'numeric' }});

            let monthPnl = 0;
            let tradingDays = 0;

            let calendarHtml = '';
            let currentRow = [];
            let weekPnl = 0;

            for (let i = 0; i < startDay; i++) {{
                currentRow.push('<div class="calendar-day empty"></div>');
            }}

            for (let day = 1; day <= daysInMonth; day++) {{
                const dateStr = `${{year}}-${{String(month).padStart(2, '0')}}-${{String(day).padStart(2, '0')}}`;
                const dayTrades = tradesData[dateStr] || [];

                let dayPnl = 0;
                for (const nqPnl of dayTrades) {{
                    dayPnl += calculatePnl(nqPnl);
                }}

                if (dayTrades.length > 0) {{
                    tradingDays++;
                    monthPnl += dayPnl;
                    weekPnl += dayPnl;

                    const pnlClass = dayPnl > 0 ? 'profit' : dayPnl < 0 ? 'loss' : '';
                    const pnlColor = dayPnl > 0 ? 'positive' : dayPnl < 0 ? 'negative' : 'neutral';

                    currentRow.push(`
                        <div class="calendar-day ${{pnlClass}}">
                            <div class="day-number">${{day}}</div>
                            <div class="day-pnl ${{pnlColor}}">$${{dayPnl.toFixed(0)}}</div>
                            <div class="day-trades">${{dayTrades.length}} trades</div>
                        </div>
                    `);
                }} else {{
                    currentRow.push(`
                        <div class="calendar-day empty">
                            <div class="day-number">${{day}}</div>
                        </div>
                    `);
                }}

                const dayOfWeek = (startDay + day - 1) % 7;
                if (dayOfWeek === 6 || day === daysInMonth) {{
                    while (currentRow.length < 7) {{
                        currentRow.push('<div class="calendar-day empty"></div>');
                    }}

                    const weekClass = weekPnl > 0 ? 'positive' : weekPnl < 0 ? 'negative' : 'neutral';
                    currentRow.push(`
                        <div class="week-total">
                            <div class="week-label">Week</div>
                            <div class="${{weekClass}}">$${{weekPnl.toFixed(0)}}</div>
                        </div>
                    `);

                    calendarHtml += '<div class="calendar-row">' + currentRow.join('') + '</div>';
                    currentRow = [];
                    weekPnl = 0;
                }}
            }}

            document.getElementById('calendarBody').innerHTML = calendarHtml;

            const monthPnlClass = monthPnl > 0 ? 'positive' : monthPnl < 0 ? 'negative' : 'neutral';
            document.getElementById('monthPnl').innerHTML =
                `<span class="${{monthPnlClass}}">$${{monthPnl.toFixed(0)}}</span>`;

            document.getElementById('tradingDays').textContent = tradingDays;

            const avgDailyVal = tradingDays > 0 ? monthPnl / tradingDays : 0;
            const avgDailyClass = avgDailyVal > 0 ? 'positive' : avgDailyVal < 0 ? 'negative' : 'neutral';
            document.getElementById('avgDaily').innerHTML =
                `<span class="${{avgDailyClass}}">$${{avgDailyVal.toFixed(0)}}</span>`;

            let yearPnl = 0;
            let maxDailyDrawdown = 0;
            let maxDrawdown = 0;
            let peak = 0;
            let runningPnl = 0;

            const allDates = Object.keys(tradesData).sort();

            for (const dateStr of allDates) {{
                let dailyPnl = 0;
                for (const nqPnl of tradesData[dateStr]) {{
                    dailyPnl += calculatePnl(nqPnl);
                }}

                yearPnl += dailyPnl;

                if (dailyPnl < maxDailyDrawdown) {{
                    maxDailyDrawdown = dailyPnl;
                }}

                runningPnl += dailyPnl;
                if (runningPnl > peak) {{
                    peak = runningPnl;
                }}
                const drawdown = peak - runningPnl;
                if (drawdown > maxDrawdown) {{
                    maxDrawdown = drawdown;
                }}
            }}

            const yearClass = yearPnl > 0 ? 'positive' : yearPnl < 0 ? 'negative' : 'neutral';
            document.getElementById('yearTotal').innerHTML = `<span class="${{yearClass}}">$${{yearPnl.toFixed(0)}}</span>`;
            document.getElementById('maxDailyDD').textContent = `$${{Math.abs(maxDailyDrawdown).toFixed(0)}}`;
            document.getElementById('maxDD').textContent = `$${{maxDrawdown.toFixed(0)}}`;
        }}

        function changeMonth(delta) {{
            let [year, month] = currentMonth.split('-').map(Number);
            month += delta;

            if (month < 1) {{
                month = 12;
                year--;
            }} else if (month > 12) {{
                month = 1;
                year++;
            }}

            currentMonth = `${{year}}-${{String(month).padStart(2, '0')}}`;
            renderCalendar();
        }}

        document.getElementById('prevMonth').addEventListener('click', () => changeMonth(-1));
        document.getElementById('nextMonth').addEventListener('click', () => changeMonth(1));

        document.querySelectorAll('input[name="contractType"]').forEach(radio => {{
            radio.addEventListener('change', (e) => {{
                contractType = e.target.value;
                renderCalendar();
            }});
        }});

        document.getElementById('contractSize').addEventListener('input', function() {{
            numContracts = parseInt(this.value);
            document.getElementById('contractSizeValue').textContent = numContracts;
            renderCalendar();
        }});

        renderCalendar();
    </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Calendar saved to {output_path}")


def main():
    print("Collecting daily trades...")
    daily = collect_daily_trades()
    print(f"  {sum(len(v) for v in daily.values())} trades across {len(daily)} trading days")

    output = Path("results/html/vwap_reaction_pnl_calendar.html")
    generate_html(daily, output)


if __name__ == "__main__":
    main()

"""
Generate P&L Calendar HTML for optimized rough vol strategy.
Config: H=0.40, KERNEL=80, NORM=250, ZLOOK=100, HIGH_Z=1.5, EMA=50,
        SL=1.5x, TP=2.5x ATR(14), Session 09:30-15:30 (force close 15:27),
        Martingale streak=1, mult=3.0, max_doubles=4, daily reset.
"""
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

H=0.40; KERNEL_LEN=80; ETA=1.0; V0=0.0001
NORM_LEN=250; Z_LOOKBACK=100; HIGH_Z=1.5; EMA_LEN=50; ATR_LEN=14
ATR_SL=1.5; ATR_TP=2.5; ET="America/New_York"; POINT_VALUE=20.0
SS="09:30"; SE="15:30"; FORCE_CLOSE="15:27"; MT=5
MART_STREAK=1; MART_MULT=3.0; MART_MAX_DOUBLES=4

OUT = Path("C:/trading/nqorderflowbacktester/results/html/roughvol_calendar.html")


def build_combined_bars():
    print("Loading data...", flush=True)
    df_mt = pd.read_parquet(MARKETTICK_PARQUET)
    frames = []
    for f in sorted(TIMEBARS_DIR.glob("timebars_5min_202*.pkl")):
        with open(f, "rb") as fh:
            bars = pickle.load(fh)
        if not bars: continue
        rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
                 "low": b["low"], "close": b["close"]} for b in bars]
        df5 = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df5["group"] = df5.index.floor("15min")
        agg = df5.groupby("group").agg(open=("open", "first"), high=("high", "max"),
                                        low=("low", "min"), close=("close", "last"))
        agg.index += pd.Timedelta(minutes=15)
        frames.append(agg)
    df_pkl = pd.concat(frames).sort_index()
    df_pkl = df_pkl[~df_pkl.index.duplicated(keep="first")]
    cutoff = df_mt.index[-1]
    df_pkl_new = df_pkl[df_pkl.index > cutoff]
    df = pd.concat([df_mt[["open", "high", "low", "close"]], df_pkl_new[["open", "high", "low", "close"]]]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df.index
    ])
    print(f"Combined: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})", flush=True)
    return df


def main():
    df = build_combined_bars()
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)

    # Rough vol z-score
    ret = np.log(closes / closes.shift(1)); ret.iloc[0] = 0.0
    mean_ret = ret.rolling(NORM_LEN).mean()
    std_ret = ret.rolling(NORM_LEN).std(ddof=0)
    shock = np.nan_to_num(np.where(std_ret > 0, (ret - mean_ret) / std_ret, 0.0), nan=0.0)
    ks = np.arange(1, KERNEL_LEN + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = ks ** (H - 0.5) - (ks - 1) ** (H - 0.5)
    kernel[0] = 1.0
    xH = np.convolve(shock, kernel, mode="full")[:n]
    v_model = np.minimum(V0 * np.exp(ETA * xH), V0 * 1e4)
    v_s = pd.Series(v_model, index=df.index)
    v_m = v_s.rolling(Z_LOOKBACK).mean(); v_d = v_s.rolling(Z_LOOKBACK).std(ddof=0)
    df["z_vol"] = np.nan_to_num(np.where(v_d > 0, (v_s - v_m) / v_d, 0.0), nan=0.0)

    # EMA + RMA ATR
    df["ema"] = closes.ewm(span=EMA_LEN, adjust=False).mean()
    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(), (lows - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n); atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i+1].mean()
        else: atr_vals[i] = (atr_vals[i-1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    # Backtest
    trades = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
    qty = 1; loss_streak = 0
    for i in range(len(df)):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE
        # Force close at 15:27
        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE * qty
            trades.append({"pnl": round(pnl, 2), "date": str(td), "qty": qty, "reason": "CLOSE"})
            if pnl > 0: loss_streak = 0
            else: loss_streak += 1
            pos = 0
            continue
        if not ins: continue
        if td != cd: cd = td; dt = 0; loss_streak = 0
        if loss_streak >= MART_STREAK:
            steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
            qty = max(1, round(MART_MULT ** steps))
        else: qty = 1
        atr_v = row["atr"]
        if atr_v <= 0: continue
        z = row["z_vol"]; cl = row["close"]; ema = row["ema"]
        if pos != 0:
            xp = None
            if d == "long":
                if row["low"] <= sl: xp = sl
                elif row["high"] >= tp: xp = tp
            else:
                if row["high"] >= sl: xp = sl
                elif row["low"] <= tp: xp = tp
            if xp:
                pnl = ((xp - ep) if d == "long" else (ep - xp)) * POINT_VALUE * qty
                trades.append({"pnl": round(pnl, 2), "date": str(td), "qty": qty,
                               "reason": "TP" if ((d == "long" and xp >= tp) or (d == "short" and xp <= tp)) else "SL"})
                if pnl > 0: loss_streak = 0
                else: loss_streak += 1
                if loss_streak >= MART_STREAK:
                    steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                    qty = max(1, round(MART_MULT ** steps))
                else: qty = 1
                pos = 0; continue
        if pos == 0 and dt < MT:
            if z > HIGH_Z and cl > ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
            elif z > HIGH_Z and cl < ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1

    # Group by date
    daily_trades = {}
    for t in trades:
        daily_trades.setdefault(t["date"], []).append(t["pnl"])

    # Stats
    all_pnl = [t["pnl"] for t in trades]
    total_trades = len(all_pnl)
    total_pnl = sum(all_pnl)
    winners = [p for p in all_pnl if p > 0]
    win_rate = len(winners) / len(all_pnl) * 100 if all_pnl else 0

    dates = sorted(daily_trades.keys())
    start_month = dates[0][:7] if dates else "2021-01"

    # Exit reason info per day
    daily_reasons = {}
    for t in trades:
        daily_reasons.setdefault(t["date"], []).append(t["reason"])
    reason_info = {}
    for d, rs in daily_reasons.items():
        tp_n = sum(1 for r in rs if r == "TP")
        sl_n = sum(1 for r in rs if r == "SL")
        cl_n = sum(1 for r in rs if r == "CLOSE")
        parts = []
        if tp_n: parts.append(f"{tp_n}TP")
        if sl_n: parts.append(f"{sl_n}SL")
        if cl_n: parts.append(f"{cl_n}CL")
        reason_info[d] = " ".join(parts)

    trades_json = json.dumps(daily_trades)
    reason_json = json.dumps(reason_info)

    pf_val = sum(p for p in all_pnl if p > 0) / abs(sum(p for p in all_pnl if p < 0)) if sum(p for p in all_pnl if p < 0) else 0
    print(f"Trades: {total_trades}  PF: {pf_val:.2f}  PnL: ${total_pnl:+,.0f}  WR: {win_rate:.1f}%")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Rough Vol Strategy - P&L Calendar (Optimized)</title>
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
        <h1>Rough Vol Strategy - P&L Calendar (Optimized)</h1>
        <div class="subtitle">H={H} | NORM={NORM_LEN} | ZLOOK={Z_LOOKBACK} | HIGH_Z={HIGH_Z} | Marti s={MART_STREAK} m={MART_MULT}x d={MART_MAX_DOUBLES} daily | {total_trades} trades | {win_rate:.1f}% WR</div>

        <div class="strategy-note">
            <h3>Strategy: Rough Vol (Phase 1-5 Optimized)</h3>
            <p><strong>Entry:</strong> z_vol &gt; {HIGH_Z} — long if close &gt; EMA({EMA_LEN}), short if close &lt; EMA({EMA_LEN})</p>
            <p><strong>Risk:</strong> SL {ATR_SL}x ATR / TP {ATR_TP}x ATR | Martingale streak={MART_STREAK}, mult={MART_MULT}x, max_doubles={MART_MAX_DOUBLES}, daily reset</p>
            <p><strong>Session:</strong> {SS}-{SE} ET | Force close {FORCE_CLOSE} | Max {MT} trades/day | 15-min bars</p>
            <p><strong>Total P&L (1 NQ):</strong> ${total_pnl:,.0f} ({total_trades} trades, PF {pf_val:.2f})</p>
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
        const reasonInfo = {reason_json};
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
                    const info = reasonInfo[dateStr] || '';
                    currentRow.push(`<div class="calendar-day ${{dayPnl > 0 ? 'profit' : dayPnl < 0 ? 'loss' : ''}}"><div class="day-number">${{day}}</div><div class="day-pnl ${{dayPnl > 0 ? 'positive' : dayPnl < 0 ? 'negative' : 'neutral'}}">$${{dayPnl.toFixed(0)}}</div><div class="day-trades">${{dayTrades.length}} trades</div><div class="day-source">${{info}}</div></div>`);
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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"Calendar saved to {OUT}")


if __name__ == "__main__":
    main()

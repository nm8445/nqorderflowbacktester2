"""
Generate the interactive PnL calendar HTML for the 3-strategy combined portfolio,
in the same dark-themed format as scripts/overnight range strat/scripts/gen_filtered_pnl_calendar.py.

Source: scripts/rough vol orderflow/results/combined_4way_trades.csv
Output:
  - scripts/rough vol orderflow/results/combined_pnl_calendar.html
  - live/combined deployment plan/combined_pnl_calendar.html
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLAN_DIR = HERE.parent.parent / "live" / "combined deployment plan"

NQ_PT = 20.0  # pnl_$ stored at NQ basis; divide to get points

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>4-Strategy Combined — PnL Calendar &amp; Equity</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1e1e1e; color: #fff; padding: 20px; }
  .container { max-width: 1500px; margin: 0 auto; }
  h1 { text-align: center; margin-bottom: 6px; color: #4ec9b0; }
  .subtitle { text-align: center; margin-bottom: 16px; color: #a0a0a0; font-size: 13px; }
  .strategy-note { background: #2d2d30; padding: 14px 18px; border-radius: 8px; margin-bottom: 18px; border-left: 4px solid #4ec9b0; font-size: 12px; line-height: 1.5; }
  .strategy-note h3 { color: #4ec9b0; font-size: 14px; margin-bottom: 6px; }
  .strategy-note span.k { color: #d7ba7d; }
  .controls-row { background: #2d2d30; padding: 16px; border-radius: 8px; margin-bottom: 14px; display: flex; gap: 30px; align-items: center; flex-wrap: wrap; }
  .controls-row h3 { color: #4ec9b0; font-size: 14px; margin-right: 12px; }
  .radio-group { display: flex; gap: 14px; }
  .slider-container { display: flex; align-items: center; gap: 12px; min-width: 380px; }
  .slider { flex: 1; max-width: 260px; accent-color: #4ec9b0; }
  .slider-value { font-size: 18px; font-weight: bold; color: #4ec9b0; min-width: 80px; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 14px; }
  .stat-card { background: #2d2d30; padding: 12px; border-radius: 8px; }
  .stat-label { font-size: 11px; color: #a0a0a0; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-value { font-size: 20px; font-weight: bold; }
  .positive { color: #4ec9b0; }
  .negative { color: #f48771; }
  .neutral  { color: #a0a0a0; }
  .chart-card { background: #2d2d30; padding: 14px; border-radius: 8px; margin-bottom: 14px; }
  .chart-card h3 { color: #4ec9b0; font-size: 14px; margin-bottom: 8px; }
  .controls { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; background: #2d2d30; padding: 14px; border-radius: 8px; }
  .nav-buttons button { background: #3e3e42; color: #fff; border: none; padding: 8px 16px; margin: 0 4px; border-radius: 4px; cursor: pointer; font-size: 14px; }
  .nav-buttons button:hover { background: #505050; }
  .month-title { font-size: 22px; font-weight: bold; }
  .calendar { background: #2d2d30; border-radius: 8px; padding: 16px; }
  .calendar-header { display: grid; grid-template-columns: repeat(7, 1fr) 110px; gap: 4px; margin-bottom: 8px; font-weight: bold; color: #a0a0a0; font-size: 12px; }
  .calendar-row { display: grid; grid-template-columns: repeat(7, 1fr) 110px; gap: 4px; margin-bottom: 4px; }
  .calendar-day { background: #3e3e42; padding: 8px; border-radius: 4px; min-height: 70px; }
  .calendar-day.empty { background: transparent; }
  .calendar-day.profit { background: linear-gradient(135deg, #2d4a3e 0%, #3e3e42 100%); border-left: 3px solid #4ec9b0; }
  .calendar-day.loss   { background: linear-gradient(135deg, #4a2d2d 0%, #3e3e42 100%); border-left: 3px solid #f48771; }
  .day-number { font-size: 12px; color: #a0a0a0; margin-bottom: 3px; }
  .day-pnl   { font-size: 16px; font-weight: bold; margin-bottom: 2px; }
  .day-trades { font-size: 10px; color: #a0a0a0; }
  .week-total { background: #2d2d30; padding: 8px; border-radius: 4px; display: flex; flex-direction: column; justify-content: center; align-items: center; font-weight: bold; font-size: 12px; }
  .week-label { font-size: 9px; color: #a0a0a0; margin-bottom: 2px; }
</style>
</head>
<body>
<div class="container">
  <h1>4-Strategy Combined &mdash; PnL Calendar &amp; Equity</h1>
  <div class="subtitle">__SUBTITLE__</div>

  <div class="strategy-note">
    <h3>Strategies (running on the same account)</h3>
    <p>
      <span class="k">Rough Vol v3:</span> 20m bars, z_vol &gt; 2.0, EMA-80 bias, window_N8_D150 absorption, session 09:00-12:59 + 14:00-14:44 ET, no martingale<br>
      <span class="k">B2 OHI/OLO:</span> 5m pinbar + windowed delta, ratchet SL + MFE-guard, fixed TP 2&times;ATR, session 09:00-14:59 ET, force-close 16:00 ET, FC-only martingale<br>
      <span class="k">Overnight Drift:</span> long-only at 19:00 ET, yellow ratchet / green target, force-close 08:00 ET, locked martingale (s1-L2)<br>
      <span class="k">Fabio ORB:</span> 5m bars, long-only ORB breakout (08:30-09:00 ET range), N=4 confirm closes + delta&ge;300, static SL=ORB_Low, 4R TP, EOD exit 14:00 ET, no martingale<br>
      <span class="k">Conflict rule:</span> intraday RV/B2/FB can all fire if same direction; opposing direction = SECOND entry blocked (no hedging). OD never overlaps with intraday.
    </p>
  </div>

  <div class="controls-row">
    <h3>Risk</h3>
    <div class="radio-group">
      <label><input type="radio" name="contractType" value="mnq" checked> MNQ ($2/pt)</label>
      <label><input type="radio" name="contractType" value="nq"> NQ ($20/pt)</label>
    </div>
    <div class="slider-container">
      <label>Base contracts:</label>
      <input type="range" id="contractSize" class="slider" min="1" max="10" value="1" step="1">
      <div class="slider-value" id="contractSizeValue">1</div>
    </div>
    <div style="font-size: 11px; color: #a0a0a0; max-width: 380px;">
      Per-strategy martingale (B2 + OD) is baked into each trade's points already.
      The base-contracts slider multiplies every trade size by N (so a mart-2x trade at base=3 fires 6 contracts).
    </div>
  </div>

  <div class="stats-grid">
    <div class="stat-card"><div class="stat-label">Total P&amp;L</div><div class="stat-value" id="totalPnl">$0</div></div>
    <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value neutral" id="winRate">0%</div></div>
    <div class="stat-card"><div class="stat-label">Profit Factor</div><div class="stat-value neutral" id="pf">0</div></div>
    <div class="stat-card"><div class="stat-label">Trades</div><div class="stat-value neutral" id="totalTrades">0</div></div>
    <div class="stat-card"><div class="stat-label">Max DD</div><div class="stat-value negative" id="maxDD">$0</div></div>
    <div class="stat-card"><div class="stat-label">Sharpe (daily)</div><div class="stat-value neutral" id="sharpe">0</div></div>
    <div class="stat-card"><div class="stat-label">Best Day</div><div class="stat-value positive" id="bestDay">$0</div></div>
    <div class="stat-card"><div class="stat-label">Worst Day</div><div class="stat-value negative" id="worstDay">$0</div></div>
  </div>

  <div class="chart-card">
    <h3>Combined Equity Curve</h3>
    <div id="equityChart" style="height: 380px;"></div>
  </div>

  <div class="controls">
    <div class="nav-buttons">
      <button id="firstMonth">|&larr; First</button>
      <button id="prevMonth">&larr; Prev</button>
      <button id="nextMonth">Next &rarr;</button>
      <button id="lastMonth">Last &rarr;|</button>
    </div>
    <div class="month-title" id="monthTitle"></div>
    <div id="monthSubtitle" style="color:#a0a0a0; font-size: 12px;"></div>
  </div>

  <div class="calendar">
    <div class="calendar-header">
      <div>Sunday</div><div>Monday</div><div>Tuesday</div><div>Wednesday</div>
      <div>Thursday</div><div>Friday</div><div>Saturday</div><div>Week</div>
    </div>
    <div id="calendarBody"></div>
  </div>
</div>

<script>
// trades_data: { "YYYY-MM-DD": [pts_per_trade_already_scaled_by_mart_size, ...] }
const tradesData = __TRADES_DATA__;
const allDates = Object.keys(tradesData).sort();
const startMonth = '__START_MONTH__';
const endMonth = '__END_MONTH__';

let currentMonth = startMonth;
let contractType = 'mnq';
let baseContracts = 1;

function dollarsPerPt() { return (contractType === 'mnq') ? 2 : 20; }

function dollarsForTrade(pts) { return pts * dollarsPerPt() * baseContracts; }

function dailyDollars(dateStr) {
  const trades = tradesData[dateStr] || [];
  let s = 0;
  for (const p of trades) s += dollarsForTrade(p);
  return s;
}

function fmt$(n) {
  const sign = n < 0 ? '-' : '';
  const abs = Math.abs(n);
  if (abs >= 1000) return sign + '$' + abs.toLocaleString('en-US', {maximumFractionDigits: 0});
  return sign + '$' + abs.toFixed(0);
}

function recomputeAggregates() {
  let total = 0, peak = 0, mdd = 0, runEq = 0;
  let wins = 0, losses = 0, sumW = 0, sumL = 0;
  let totalTrades = 0;
  let bestDay = 0, worstDay = 0;
  const dailyDollarsArr = [];
  for (const d of allDates) {
    const trades = tradesData[d] || [];
    totalTrades += trades.length;
    let dDollars = 0;
    for (const p of trades) {
      const v = dollarsForTrade(p);
      dDollars += v;
      if (p > 0) { wins++; sumW += v; }
      else if (p < 0) { losses++; sumL += -v; }
    }
    dailyDollarsArr.push(dDollars);
    runEq += dDollars;
    if (runEq > peak) peak = runEq;
    const dd = peak - runEq;
    if (dd > mdd) mdd = dd;
    if (dDollars > bestDay) bestDay = dDollars;
    if (dDollars < worstDay) worstDay = dDollars;
    total += dDollars;
  }
  const wr = (wins + losses) > 0 ? wins / (wins + losses) * 100 : 0;
  const pf = sumL > 0 ? sumW / sumL : (sumW > 0 ? Infinity : 0);
  const m = dailyDollarsArr.reduce((a, b) => a + b, 0) / dailyDollarsArr.length;
  const v = dailyDollarsArr.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(dailyDollarsArr.length - 1, 1);
  const sd = Math.sqrt(v);
  const sharpe = sd > 0 ? (m / sd) * Math.sqrt(252) : 0;
  return { total, mdd, wr, pf, totalTrades, bestDay, worstDay, sharpe };
}

function renderHeadline() {
  const a = recomputeAggregates();
  const totEl = document.getElementById('totalPnl');
  totEl.textContent = fmt$(a.total);
  totEl.className = 'stat-value ' + (a.total > 0 ? 'positive' : (a.total < 0 ? 'negative' : 'neutral'));
  document.getElementById('winRate').textContent = a.wr.toFixed(1) + '%';
  document.getElementById('pf').textContent = (isFinite(a.pf) ? a.pf.toFixed(2) : '∞');
  document.getElementById('totalTrades').textContent = a.totalTrades;
  document.getElementById('maxDD').textContent = '-' + fmt$(a.mdd).replace(/^-/,'');
  document.getElementById('sharpe').textContent = a.sharpe.toFixed(2);
  document.getElementById('bestDay').textContent = fmt$(a.bestDay);
  document.getElementById('worstDay').textContent = fmt$(a.worstDay);
}

function renderEquityCurve() {
  let runEq = 0; const xs = []; const ys = []; const dds = []; let peak = 0;
  for (const d of allDates) {
    runEq += dailyDollars(d);
    xs.push(d);
    ys.push(runEq);
    if (runEq > peak) peak = runEq;
    dds.push(runEq - peak);
  }
  const eqTrace = {
    x: xs, y: ys, type: 'scatter', mode: 'lines',
    name: 'Equity ($)',
    line: { color: '#4ec9b0', width: 1.6 },
    fill: 'tozeroy', fillcolor: 'rgba(78,201,176,0.10)',
  };
  const ddTrace = {
    x: xs, y: dds, type: 'scatter', mode: 'lines',
    name: 'Drawdown ($)',
    yaxis: 'y2',
    line: { color: '#f48771', width: 1 },
    fill: 'tozeroy', fillcolor: 'rgba(244,135,113,0.18)',
  };
  Plotly.react('equityChart', [eqTrace, ddTrace], {
    paper_bgcolor: '#2d2d30', plot_bgcolor: '#2d2d30',
    margin: { l: 70, r: 70, t: 10, b: 40 },
    xaxis: { color: '#a0a0a0', gridcolor: '#3e3e42' },
    yaxis: { color: '#4ec9b0', gridcolor: '#3e3e42', title: 'Equity ($)', tickprefix: '$' },
    yaxis2: { color: '#f48771', overlaying: 'y', side: 'right', title: 'Drawdown ($)', tickprefix: '$', showgrid: false },
    legend: { font: { color: '#a0a0a0' }, orientation: 'h', y: -0.18 },
    showlegend: true,
  }, { displayModeBar: false, responsive: true });
}

function renderCalendar() {
  const [year, month] = currentMonth.split('-').map(Number);
  const firstDay = new Date(year, month - 1, 1);
  const lastDay  = new Date(year, month, 0);
  const daysInMonth = lastDay.getDate();
  const startDay = firstDay.getDay();

  document.getElementById('monthTitle').textContent =
    firstDay.toLocaleString('default', { month: 'long', year: 'numeric' });

  let calendarHtml = '';
  let currentRow = [];
  let weekPnl = 0;
  let monthPnl = 0;
  let monthDays = 0;

  for (let i = 0; i < startDay; i++) {
    currentRow.push('<div class="calendar-day empty"></div>');
  }
  for (let day = 1; day <= daysInMonth; day++) {
    const dStr = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
    const trades = tradesData[dStr] || [];
    if (trades.length > 0) {
      const dDollars = dailyDollars(dStr);
      monthPnl += dDollars; monthDays++;
      weekPnl += dDollars;
      const cls = dDollars > 0 ? 'profit' : (dDollars < 0 ? 'loss' : '');
      const pColor = dDollars > 0 ? 'positive' : (dDollars < 0 ? 'negative' : 'neutral');
      currentRow.push(`
        <div class="calendar-day ${cls}">
          <div class="day-number">${day}</div>
          <div class="day-pnl ${pColor}">${fmt$(dDollars)}</div>
          <div class="day-trades">${trades.length} trade${trades.length>1?'s':''}</div>
        </div>`);
    } else {
      currentRow.push(`<div class="calendar-day empty"><div class="day-number">${day}</div></div>`);
    }
    const dayOfWeek = (startDay + day - 1) % 7;
    if (dayOfWeek === 6 || day === daysInMonth) {
      while (currentRow.length < 7) currentRow.push('<div class="calendar-day empty"></div>');
      const wkClass = weekPnl > 0 ? 'positive' : (weekPnl < 0 ? 'negative' : 'neutral');
      currentRow.push(`<div class="week-total"><div class="week-label">Week</div><div class="${wkClass}">${fmt$(weekPnl)}</div></div>`);
      calendarHtml += '<div class="calendar-row">' + currentRow.join('') + '</div>';
      currentRow = []; weekPnl = 0;
    }
  }
  document.getElementById('calendarBody').innerHTML = calendarHtml;

  const sub = document.getElementById('monthSubtitle');
  if (monthDays > 0) {
    const cls = monthPnl > 0 ? 'positive' : (monthPnl < 0 ? 'negative' : 'neutral');
    sub.innerHTML = `<span class="${cls}">${fmt$(monthPnl)}</span> &nbsp; · &nbsp; ${monthDays} trading days`;
  } else {
    sub.textContent = 'no trades';
  }
}

function changeMonth(delta) {
  let [y, m] = currentMonth.split('-').map(Number);
  m += delta;
  if (m < 1) { m = 12; y--; }
  else if (m > 12) { m = 1; y++; }
  currentMonth = `${y}-${String(m).padStart(2, '0')}`;
  renderCalendar();
}

document.getElementById('prevMonth').addEventListener('click', () => changeMonth(-1));
document.getElementById('nextMonth').addEventListener('click', () => changeMonth(1));
document.getElementById('firstMonth').addEventListener('click', () => { currentMonth = startMonth; renderCalendar(); });
document.getElementById('lastMonth').addEventListener('click',  () => { currentMonth = endMonth;   renderCalendar(); });
document.querySelectorAll('input[name="contractType"]').forEach(r =>
  r.addEventListener('change', e => { contractType = e.target.value; renderHeadline(); renderEquityCurve(); renderCalendar(); }));
document.getElementById('contractSize').addEventListener('input', function() {
  baseContracts = parseInt(this.value, 10);
  document.getElementById('contractSizeValue').textContent = baseContracts;
  renderHeadline(); renderEquityCurve(); renderCalendar();
});

renderHeadline();
renderEquityCurve();
renderCalendar();
</script>
</body>
</html>
"""


def main():
    df = pd.read_csv(RESULTS_DIR / "combined_4way_trades.csv")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    # Convert $ -> points (template multiplies pts by $2 for MNQ or $20 for NQ)
    df["pts"] = df["pnl_$"] / NQ_PT

    print(f"trades: {len(df)}   total pts: {df['pts'].sum():+.1f}   "
          f"= ${df['pnl_$'].sum():+,.0f} NQ basis (= ${df['pnl_$'].sum()/10:+,.0f} MNQ basis)")

    trades_data = {}
    for d, sub in df.groupby("date"):
        trades_data[str(d)] = [round(float(x), 4) for x in sub["pts"].values]

    n_trades = len(df)
    total_pts = df["pts"].sum()
    subtitle = (f"Rough Vol v3 (no mart) + B2 OHI/OLO (FC-only mart) + Overnight Drift (locked) + Fabio ORB (locked)  ·  "
                f"{n_trades} trades  ·  2020-12 → 2026-05  ·  "
                f"locked MNQ basis: {total_pts*2:+,.0f}$ at base=1   |   NQ basis: {total_pts*20:+,.0f}$ at base=1")

    all_dates = sorted(trades_data.keys())
    start_month = all_dates[0][:7]
    end_month = all_dates[-1][:7]

    html = (HTML_TEMPLATE
            .replace("__TRADES_DATA__", json.dumps(trades_data))
            .replace("__START_MONTH__", start_month)
            .replace("__END_MONTH__", end_month)
            .replace("__SUBTITLE__", subtitle))

    for out in (RESULTS_DIR / "combined_pnl_calendar.html",
                PLAN_DIR / "combined_pnl_calendar.html"):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()

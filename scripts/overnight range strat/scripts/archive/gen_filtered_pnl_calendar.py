"""Render an interactive PnL calendar + equity curve HTML for the locked
filtered config (V2 K=0.8 lock=0.45 + mart-fc-only + drop POS+SHORT + 9-14 ET).

Output: results/html/locked_filtered_pnl_calendar.html

Page features:
  - Risk slider (contracts 1-10), MNQ/NQ contract toggle
  - Headline stats card (total $, max DD $, PF, Sharpe, WR)
  - Equity curve (Plotly), full data, updates with slider
  - Monthly calendar grid with day-by-day PnL color-coded, week totals
  - Month nav (prev/next) -- updates calendar without reloading
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sweep_ratchet_sl_fixed_tp import filter_pre_dedupe
from test_pure_ratchet_exits import build_20min_bars, FORCE_CLOSE_TIME

PARQUET_DIR = Path(__file__).parent / "parquets"
EOD_MQ      = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
OUT_HTML    = Path(__file__).parent.parent.parent.parent / "results" / "html" / "locked_filtered_pnl_calendar.html"

VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
CONF_N, CONF_D = 5, 75
YMULT, TPMULT  = 2.50, 2.00
MFE_K, MFE_LOCK = 0.8, 0.45
ALLOWED_HOURS = {9, 10, 11, 12, 13, 14}
DROP_POS_SHORT = True


def simulate_exit_v2(direction, entry_ts, entry_price, bars20):
    sign = 1 if direction == "LONG" else -1
    bars_idx = bars20.index
    start = bars_idx.searchsorted(entry_ts, side="right")
    if start >= len(bars_idx): return None
    ent_date = entry_ts.date()
    end = start
    while end < len(bars_idx) and bars_idx[end].date() == ent_date: end += 1
    if end == start: return None
    init_idx = start - 1
    if init_idx < 0 or np.isnan(bars20["atr_y"].iloc[init_idx]): return None
    init_atr_y = float(bars20["atr_y"].iloc[init_idx])
    yellow_val = entry_price - sign * YMULT * init_atr_y
    prev_yellow = yellow_val
    o = bars20["open"].values[start:end]; h = bars20["high"].values[start:end]
    l = bars20["low"].values[start:end];  c = bars20["close"].values[start:end]
    ay = bars20["atr_y"].values[start:end]; ts_arr = bars_idx[start:end]
    n = end - start
    green_val = entry_price + sign * TPMULT * init_atr_y
    tp_dist = abs(green_val - entry_price)
    mfe_so_far = 0.0
    for i in range(n):
        bar_close_ts = ts_arr[i] + pd.Timedelta(minutes=20)
        cur_mfe = (h[i] - entry_price) if sign > 0 else (entry_price - l[i])
        if cur_mfe > mfe_so_far: mfe_so_far = cur_mfe
        if not np.isnan(ay[i]):
            raw_yellow = c[i] - sign * YMULT * ay[i]
            yellow_val = max(prev_yellow, raw_yellow) if sign > 0 else min(prev_yellow, raw_yellow)
        if mfe_so_far >= MFE_K * tp_dist:
            mfe_stop = entry_price + sign * MFE_LOCK * mfe_so_far
            stop_level = max(yellow_val, mfe_stop) if sign > 0 else min(yellow_val, mfe_stop)
        else:
            stop_level = yellow_val
        if sign > 0 and h[i] >= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED", bar_close_ts, i + 1)
        if sign < 0 and l[i] <= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED", bar_close_ts, i + 1)
        if sign > 0 and c[i] <= stop_level and c[i] < o[i]:
            return (c[i] - entry_price, "SL_TRAIL", bar_close_ts, i + 1)
        if sign < 0 and c[i] >= stop_level and c[i] > o[i]:
            return (entry_price - c[i], "SL_TRAIL", bar_close_ts, i + 1)
        if ts_arr[i].time() >= FORCE_CLOSE_TIME:
            return (sign * (c[i] - entry_price), "FORCE_CLOSE", bar_close_ts, i + 1)
        prev_yellow = yellow_val
    return (sign * (c[-1] - entry_price), "EOD", ts_arr[-1] + pd.Timedelta(minutes=20), n)


def load_gamma_lookup():
    if not EOD_MQ.exists():
        return None, None
    eod = pd.read_parquet(EOD_MQ)
    eod["date"] = pd.to_datetime(eod["date"]).dt.date
    eod = eod.set_index("date").sort_index()
    eod_dates = sorted(eod.index.tolist())
    def prior_mq(d):
        prev = None
        for md in eod_dates:
            if md < d: prev = md
            else: break
        return prev
    return eod, prior_mq


def attach_gamma(cands):
    eod, prior_mq = load_gamma_lookup()
    cands = cands.copy()
    if eod is None:
        cands["gamma_sign"] = np.nan
        return cands
    col = None
    for c in ("qqq_gamma_sign", "gamma_sign"):
        if c in eod.columns:
            col = c; break
    cands["entry_date"] = pd.to_datetime(cands["entry_time_et"]).dt.date
    if col is None:
        cands["gamma_sign"] = np.nan
        return cands
    g_lookup = {}
    for d in cands["entry_date"].unique():
        p = prior_mq(d)
        g_lookup[d] = eod.loc[p, col] if p in eod.index else np.nan
    cands["gamma_sign"] = cands["entry_date"].map(g_lookup)
    return cands


def filter_candidates(cands):
    cands = cands.copy()
    cands["entry_hour"] = pd.to_datetime(cands["entry_time_et"]).dt.hour
    keep = cands["entry_hour"].isin(ALLOWED_HOURS)
    if DROP_POS_SHORT:
        keep &= ~((cands["gamma_sign"] == 1) & (cands["direction"] == "SHORT"))
    return cands[keep].reset_index(drop=True)


def run_trades(cands, bars20, period_label):
    rows = []
    last_exit = pd.Timestamp(0, tz="America/New_York")
    for i, row in cands.iterrows():
        ex = simulate_exit_v2(row["direction"], row["entry_time_et"],
                              float(row["entry_price"]), bars20)
        if ex is None: continue
        pnl, reason, exit_ts, bars_held = ex
        if row["entry_time_et"] > last_exit:
            rows.append({
                "entry_ts": row["entry_time_et"], "exit_ts": exit_ts,
                "direction": row["direction"], "reason": reason, "pnl": pnl,
                "period": period_label, "date": row["entry_time_et"].date(),
            })
            last_exit = exit_ts
    return pd.DataFrame(rows)


def apply_mart_fc(df):
    sizes = []
    cur = 1
    for pnl, reason in zip(df["pnl"].values, df["reason"].values):
        sizes.append(cur)
        if cur == 2:
            cur = 1
        else:
            cur = 2 if (pnl < 0 and reason == "FORCE_CLOSE") else 1
    return np.array(sizes)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Locked Filtered Config — PnL Calendar & Equity</title>
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
  <h1>Locked Filtered Config — PnL Calendar &amp; Equity</h1>
  <div class="subtitle">__SUBTITLE__</div>

  <div class="strategy-note">
    <h3>Strategy</h3>
    <p>
      <span class="k">Entry:</span> B2 X=0.75 N=15 D=70 strict, BAND_K=0.25 + conf_N=5 D=75 HALF
      &nbsp;|&nbsp; <span class="k">Exit:</span> ratchet SL ymult=2.5 + V2 MFE-guard (K=0.8, lock=0.45) + fixed TP 2.0×ATR_at_entry + force-close 16:00 ET
      &nbsp;|&nbsp; <span class="k">Sizing:</span> martingale FC-only (loss-by-FC → 2x next; max size 2)
      &nbsp;|&nbsp; <span class="k">Filters:</span> entries 09:00-14:59 ET only, drop POS-gamma SHORTs
      &nbsp;|&nbsp; <span class="k">Dedupe:</span> chained Mode 1
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
      Note: martingale is already baked in (each trade has its own size 1 or 2).
      The slider multiplies that size — at base=1 a mart-2x trade trades 2 contracts;
      at base=3 a mart-2x trade trades 6 contracts.
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
    <h3>Equity Curve (full IS + OOS)</h3>
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

// Per-pt to dollars (NQ basis $20/pt; MNQ = 1/10).
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
  // Daily Sharpe (annualized x sqrt(252)).
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
    dds.push(runEq - peak);  // negative drawdown
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
    print("loading 20-min bars + entry candidates...")
    bars20 = build_20min_bars()
    is_df  = pd.read_parquet(PARQUET_DIR / "entry_signal_trades.parquet")
    oos_df = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")
    is_cands  = filter_pre_dedupe(is_df)
    oos_cands = filter_pre_dedupe(oos_df)

    is_cands  = attach_gamma(is_cands)
    oos_cands = attach_gamma(oos_cands)
    is_kept   = filter_candidates(is_cands)
    oos_kept  = filter_candidates(oos_cands)

    print("simulating exits...")
    is_t  = run_trades(is_kept,  bars20, "IS")
    oos_t = run_trades(oos_kept, bars20, "OOS")
    comb  = pd.concat([is_t, oos_t], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)

    sizes = apply_mart_fc(comb)
    comb["size"] = sizes
    comb["scaled_pnl_pts"] = comb["pnl"] * sizes  # in NQ pts already × mart-size

    print(f"  trades: {len(comb)}  total pts: {comb['scaled_pnl_pts'].sum():+.1f}  "
          f"({comb['scaled_pnl_pts'].sum()*2:+,.0f} MNQ$ at base=1)")

    # Group per date
    trades_data = {}
    for d, sub in comb.groupby("date"):
        trades_data[str(d)] = [round(float(x), 4) for x in sub["scaled_pnl_pts"].values]

    # Build subtitle
    n_trades = len(comb)
    total_pts = comb["scaled_pnl_pts"].sum()
    subtitle = (f"V2 K=0.8 lock=0.45 + mart-fc-only + filtered (drop POS+SHORT, hours 9-14 ET)  ·  "
                f"{n_trades} trades  ·  IS 2020-12 → 2024-12 + OOS 2025-01 → 2025-11  ·  "
                f"locked-in MNQ basis: {total_pts*2:+,.0f}$ at base=1")

    all_dates = sorted(trades_data.keys())
    start_month = all_dates[0][:7]
    end_month   = all_dates[-1][:7]

    html = (HTML_TEMPLATE
            .replace("__TRADES_DATA__", json.dumps(trades_data))
            .replace("__START_MONTH__", start_month)
            .replace("__END_MONTH__",   end_month)
            .replace("__SUBTITLE__",    subtitle))

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nwrote {OUT_HTML}")
    print(f"file size: {OUT_HTML.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()

"""
Monte Carlo: Combined VWAP Strategy — Full Apex Trader Funding Simulation.

Apex rules:
- Challenge: $50k account, $2k trailing DD, $3k profit target
- Daily loss limit: ~$1k (implied by DD mechanics)
- Funded: DD locks at $50k, consistency rule = 50% (no single day > 50% of total profit)
- Payout buffer: $2,100 above account size
- Payout = profits above buffer
- Account cost: $34

Sections:
  Part 1: Challenge pass rates — fixed MNQ 1-10
  Part 2: Funded payouts with consistency rule — fixed MNQ 1-10
  Part 3: Fullport multi-account rotation — 1-10 accounts ($1k risk/trade)
  Part 4: Slippage resistance — challenge + funded + fullport
  Part 5: Recommendations + CSV export

HTML output: slippage resistance charts

Usage:
    python -u scripts/vwap_reaction_strat_backtest/montecarlo_apex_full.py
"""

import csv
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time, passes_adx_filter,
    LUNCH_START, LUNCH_END,
)
from trending_vwap_atr_grid import (
    load_pickle, load_5min_bars,
    precompute_day_state, detect_signals as detect_trending_signals,
)

VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_PRICE_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_DIR = DATA_DIR / "timebars_5min"

POINT_VALUE = 20.0
MNQ_PV = 2.0

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"

# Strategy configs
BASE_SL = 0.50
BASE_TP = 1.90
TREND_SL = 1.00
TREND_TP = 1.00
TREND_STD = 1
TREND_LB = 14
TREND_ADX_MIN = 30.0

# ── Apex Trader Funding rules ──
ACCOUNT_SIZE = 50_000
TRAIL_DD = 2_000
PROFIT_TARGET = 3_000
DAILY_LOSS_LIM = 1_000       # per-account daily loss lock
FUNDED_BUFFER = 2_100        # must keep $2,100 above account size before payout
CONSISTENCY_RULE = 0.50      # no single day > 50% of total realized profit
ACCOUNT_COST = 34            # challenge account cost
TARGET_RISK = 1_000          # fullport: dollars risked per trade

N_SIMS = 10_000
MAX_DAYS_CHALLENGE = 500
MAX_DAYS_FUNDED = 1500
SEED = 42

START_DATE = "2025-03-13"
END_DATE = "2026-04-08"

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "montecarlo"


# ═══════════════════════════════════════════════════════════════════════
# Trade collection (combined base + trending)
# ═══════════════════════════════════════════════════════════════════════

def collect_combined_trades(adx_lookup):
    """
    Collect combined trades. Returns:
      pnl_pool  (N,) — pnl in points
      mae_pool  (N,) — max adverse excursion in points
      sl_pool   (N,) — stop loss distance in points (for dynamic sizing)
      day_counts (D,) — trades per calendar day (including 0-trade days)
    """
    start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    daily_pnls = {}
    daily_maes = {}
    daily_sls = {}
    all_dates = set()

    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        all_dates.add(date_str)

        signal_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_file.exists():
            continue
        signal_data = load_pickle(signal_file)
        bars = signal_data["bars"]
        if not bars:
            continue

        vwap_data = load_pickle(cache_file)
        base_signals = vwap_data["signals"]

        # Trending signals
        vwap_price_file = VWAP_PRICE_CACHE_DIR / f"{date_str}.pkl"
        tb_file = TIMEBARS_DIR / f"timebars_5min_{date_str.replace('-', '_')}.pkl"
        trend_signals = []
        if vwap_price_file.exists() and tb_file.exists():
            vwap_df = load_pickle(vwap_price_file)
            bars_5min = load_5min_bars(date_str)
            if bars_5min is not None:
                day_state = precompute_day_state(bars_5min, vwap_df, TREND_LB, TREND_STD)
                trend_signals = detect_trending_signals(bars, day_state)
                for s in trend_signals:
                    s["date"] = date_str

        # Build unified signal list
        all_signals = []

        for s in base_signals:
            ct = s["confirm_time"]
            ct_et = ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)
            if "16:00" <= ct_et.strftime("%H:%M") < "19:10":
                continue
            if LUNCH_START <= ct_et.strftime("%H:%M") < LUNCH_END:
                continue
            atr = s["atr"]
            if atr is None or atr <= 0:
                continue
            adx_val = get_adx_at_time(ct_et, adx_lookup)
            if not passes_adx_filter(adx_val):
                continue
            ep = s["entry_price"]; d = s["direction"]
            sl_pts = atr * BASE_SL; tp_pts = atr * BASE_TP
            if d == "long":
                sl = ep - sl_pts; tp = ep + tp_pts
                if tp <= ep or sl >= ep: continue
            else:
                sl = ep + sl_pts; tp = ep - tp_pts
                if tp >= ep or sl <= ep: continue
            all_signals.append({
                "bar_index": s["bar_index"], "confirm_time": ct,
                "entry_price": ep, "direction": d, "sl": sl, "tp": tp, "sl_pts": sl_pts,
            })

        for s in trend_signals:
            ct = s["confirm_time"]
            ct_et = ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)
            if "16:00" <= ct_et.strftime("%H:%M") < "19:10":
                continue
            if LUNCH_START <= ct_et.strftime("%H:%M") < LUNCH_END:
                continue
            atr = s["atr"]
            if atr is None or atr <= 0:
                continue
            adx_val = get_adx_at_time(ct_et, adx_lookup)
            if np.isnan(adx_val) or adx_val < TREND_ADX_MIN:
                continue
            ep = s["entry_price"]; d = s["direction"]
            sl_pts = atr * TREND_SL; tp_pts = atr * TREND_TP
            if d == "long":
                sl = ep - sl_pts; tp = ep + tp_pts
            else:
                sl = ep + sl_pts; tp = ep - tp_pts
            all_signals.append({
                "bar_index": s["bar_index"], "confirm_time": ct,
                "entry_price": ep, "direction": d, "sl": sl, "tp": tp, "sl_pts": sl_pts,
            })

        all_signals.sort(key=lambda x: x["confirm_time"])

        # Simulate with shared position lock
        d_pnls, d_maes, d_sls = [], [], []
        last_exit_time = None

        for sig in all_signals:
            if last_exit_time is not None and sig["confirm_time"] <= last_exit_time:
                continue
            ep = sig["entry_price"]; d = sig["direction"]
            sl_price = sig["sl"]; tp_price = sig["tp"]
            ci = sig["bar_index"] + 1
            exit_price = None; exit_time = None; worst_adverse = 0.0

            for j in range(ci + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                adverse = (ep - bar.low) if d == "long" else (bar.high - ep)
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
            d_pnls.append(pnl); d_maes.append(worst_adverse); d_sls.append(sig["sl_pts"])
            last_exit_time = exit_time

        if d_pnls:
            daily_pnls[date_str] = d_pnls
            daily_maes[date_str] = d_maes
            daily_sls[date_str] = d_sls

    # Build arrays
    all_pnls, all_maes, all_sls, day_counts = [], [], [], []
    for d in sorted(all_dates):
        if d in daily_pnls:
            day_counts.append(len(daily_pnls[d]))
            all_pnls.extend(daily_pnls[d])
            all_maes.extend(daily_maes[d])
            all_sls.extend(daily_sls[d])
        else:
            day_counts.append(0)

    return (np.array(all_pnls), np.array(all_maes),
            np.array(all_sls), np.array(day_counts, dtype=np.int32))


# ═══════════════════════════════════════════════════════════════════════
# Challenge sim — vectorized pre-allocation
# ═══════════════════════════════════════════════════════════════════════

def sim_challenge(pnl_pool, mae_pool, day_counts, contracts, rng, slippage=0.0):
    pnl_adj = pnl_pool - slippage
    pv = MNQ_PV

    # Pre-generate all random draws
    trades_per_day = rng.choice(day_counts, size=(N_SIMS, MAX_DAYS_CHALLENGE))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(pnl_pool), size=total_needed)
    all_pnls = pnl_adj[idx] * pv * contracts
    all_maes = mae_pool[idx] * pv * contracts

    passes = 0; blowups = 0; pass_days = []
    cursor = 0

    for sim in range(N_SIMS):
        equity = float(ACCOUNT_SIZE); hwm = equity; dd_level = equity - TRAIL_DD
        passed = False; blown = False; day = 0

        for di in range(MAX_DAYS_CHALLENGE):
            nt = int(trades_per_day[sim, di])
            if nt == 0:
                day += 1; continue
            for _ in range(nt):
                if equity - all_maes[cursor] <= dd_level:
                    blown = True; cursor += 1; break
                equity += all_pnls[cursor]; cursor += 1
                if equity <= dd_level:
                    blown = True; break
            if blown:
                # Skip remaining days
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break
            day += 1
            if equity > hwm:
                hwm = equity; dd_level = hwm - TRAIL_DD
            if equity <= dd_level:
                blown = True
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break
            if equity - ACCOUNT_SIZE >= PROFIT_TARGET:
                passed = True; pass_days.append(day)
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break

        if passed: passes += 1
        if blown: blowups += 1

    return {
        "pass_pct": passes / N_SIMS * 100,
        "avg_days": np.mean(pass_days) if pass_days else 0,
        "med_days": np.median(pass_days) if pass_days else 0,
        "blow_pct": blowups / N_SIMS * 100,
    }


# ═══════════════════════════════════════════════════════════════════════
# Funded sim with Apex consistency rule (50%) — optimized
# ═══════════════════════════════════════════════════════════════════════

def sim_funded(pnl_pool, mae_pool, day_counts, contracts, rng, slippage=0.0):
    """
    Funded sim with:
    - Trailing DD that locks at account size
    - Consistency rule: no single day P&L > 50% of total realized profit
    - Payout: withdraw profits above $2,100 buffer, only when consistency passes
    Optimized: track max_day_pnl instead of iterating full history.
    """
    pnl_adj = pnl_pool - slippage
    pv = MNQ_PV

    # Pre-generate all random draws
    trades_per_day = rng.choice(day_counts, size=(N_SIMS, MAX_DAYS_FUNDED))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(pnl_pool), size=total_needed)
    all_pnls = pnl_adj[idx] * pv * contracts
    all_maes = mae_pool[idx] * pv * contracts

    all_payouts = []
    all_payout_amounts = []
    all_dpp = []
    blowups = 0
    consistency_blocks = 0
    cursor = 0

    for sim in range(N_SIMS):
        equity = float(ACCOUNT_SIZE); hwm = equity
        dd_level = equity - TRAIL_DD; dd_locked = False
        payouts = 0; payout_total = 0.0
        cycle_start = 0; blown = False
        # Optimized consistency: just track max single-day positive P&L in current cycle
        max_day_pnl = 0.0

        for di in range(MAX_DAYS_FUNDED):
            nt = int(trades_per_day[sim, di])
            if nt == 0:
                continue
            day_pnl = 0.0
            for _ in range(nt):
                if equity - all_maes[cursor] <= dd_level:
                    blown = True; cursor += 1; break
                equity += all_pnls[cursor]
                day_pnl += all_pnls[cursor]
                cursor += 1
                if equity <= dd_level:
                    blown = True; break
            if blown:
                for rd in range(di + 1, MAX_DAYS_FUNDED):
                    cursor += int(trades_per_day[sim, rd])
                break

            if equity > hwm:
                hwm = equity
                if not dd_locked:
                    dd_level = hwm - TRAIL_DD
                    if dd_level >= ACCOUNT_SIZE:
                        dd_level = float(ACCOUNT_SIZE)
                        dd_locked = True
            if equity <= dd_level:
                blown = True
                for rd in range(di + 1, MAX_DAYS_FUNDED):
                    cursor += int(trades_per_day[sim, rd])
                break

            if day_pnl > max_day_pnl:
                max_day_pnl = day_pnl

            # Check payout eligibility
            buffer_eq = ACCOUNT_SIZE + FUNDED_BUFFER
            profit_above_buffer = equity - buffer_eq
            if profit_above_buffer >= PROFIT_TARGET:
                total_profit = equity - ACCOUNT_SIZE
                # Consistency: no single day > 50% of total profit
                consistency_ok = total_profit <= 0 or max_day_pnl / total_profit <= CONSISTENCY_RULE

                if consistency_ok:
                    payout_amount = profit_above_buffer
                    payouts += 1
                    payout_total += payout_amount
                    all_dpp.append(di + 1 - cycle_start)
                    all_payout_amounts.append(payout_amount)
                    equity = buffer_eq; hwm = buffer_eq
                    cycle_start = di + 1
                    max_day_pnl = 0.0  # reset for new cycle
                else:
                    consistency_blocks += 1

        all_payouts.append(payouts)
        if blown: blowups += 1

    payout_arr = np.array(all_payouts)
    got_payout = np.sum(payout_arr > 0)

    return {
        "payout_chance_pct": got_payout / N_SIMS * 100,
        "avg_payouts": np.mean(payout_arr),
        "med_payouts": np.median(payout_arr),
        "avg_dpp": np.mean(all_dpp) if all_dpp else 0,
        "med_dpp": np.median(all_dpp) if all_dpp else 0,
        "blow_pct": blowups / N_SIMS * 100,
        "avg_payout_amount": np.mean(all_payout_amounts) if all_payout_amounts else 0,
        "net_lifetime": np.mean(payout_arr) * (np.mean(all_payout_amounts) if all_payout_amounts else PROFIT_TARGET) - TRAIL_DD,
        "consistency_blocks": consistency_blocks,
    }


# ═══════════════════════════════════════════════════════════════════════
# Fullport multi-account rotation — pre-allocated
# ═══════════════════════════════════════════════════════════════════════

def sim_fullport(pnl_pool, mae_pool, sl_pool, day_counts, n_accounts, rng, slippage=0.0):
    pnl_adj = pnl_pool - slippage

    # Pre-compute dynamic sizing for all trades
    sl_dollars = sl_pool * MNQ_PV
    fp_contracts = np.where(sl_dollars > 0, np.maximum(1, np.floor(TARGET_RISK / sl_dollars)).astype(int), 1)
    fp_pnl = pnl_adj * MNQ_PV * fp_contracts
    fp_mae = mae_pool * MNQ_PV * fp_contracts

    # Pre-generate random draws
    trades_per_day = rng.choice(day_counts, size=(N_SIMS, MAX_DAYS_CHALLENGE))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(pnl_pool), size=total_needed)

    passed_any = 0; all_blown = 0; pass_days_list = []
    cursor = 0

    for sim in range(N_SIMS):
        equities = np.full(n_accounts, float(ACCOUNT_SIZE))
        hwms = np.full(n_accounts, float(ACCOUNT_SIZE))
        dd_floors = np.full(n_accounts, float(ACCOUNT_SIZE - TRAIL_DD))
        alive = np.ones(n_accounts, dtype=bool)
        daily_loss = np.zeros(n_accounts)
        daily_locked = np.zeros(n_accounts, dtype=bool)
        n_passed = 0; sim_passed = False; sim_pass_day = 0

        for day in range(MAX_DAYS_CHALLENGE):
            daily_loss[:] = 0; daily_locked[:] = False
            nt = int(trades_per_day[sim, day])
            if nt == 0: continue

            for _ in range(nt):
                ti = idx[cursor]; cursor += 1
                t_pnl = fp_pnl[ti]; t_mae = fp_mae[ti]

                acct = -1
                for a in range(n_accounts):
                    if alive[a] and not daily_locked[a]:
                        acct = a; break
                if acct == -1: break

                if equities[acct] - t_mae <= dd_floors[acct]:
                    alive[acct] = False; continue
                if daily_loss[acct] + t_mae >= DAILY_LOSS_LIM:
                    daily_locked[acct] = True
                    equities[acct] += t_pnl
                    daily_loss[acct] += max(0, -t_pnl)
                    if equities[acct] <= dd_floors[acct]:
                        alive[acct] = False
                    elif equities[acct] > hwms[acct]:
                        hwms[acct] = equities[acct]
                        dd_floors[acct] = hwms[acct] - TRAIL_DD
                    if equities[acct] - ACCOUNT_SIZE >= PROFIT_TARGET and alive[acct]:
                        n_passed += 1; alive[acct] = False
                    continue

                equities[acct] += t_pnl
                if t_pnl < 0: daily_loss[acct] += abs(t_pnl)
                if equities[acct] <= dd_floors[acct]:
                    alive[acct] = False; continue
                if equities[acct] > hwms[acct]:
                    hwms[acct] = equities[acct]
                    dd_floors[acct] = hwms[acct] - TRAIL_DD
                if daily_loss[acct] >= DAILY_LOSS_LIM:
                    daily_locked[acct] = True
                if equities[acct] - ACCOUNT_SIZE >= PROFIT_TARGET:
                    n_passed += 1; alive[acct] = False

            if not any(alive):
                for rd in range(day + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break
            if n_passed > 0 and not sim_passed:
                sim_passed = True; sim_pass_day = day + 1

        # Skip remaining days if sim ended early
        # (cursor already advanced in break above)
        if sim_passed or n_passed > 0:
            passed_any += 1
            pass_days_list.append(sim_pass_day if sim_pass_day > 0 else day + 1)
        if not any(alive) and n_passed == 0:
            all_blown += 1

    return {
        "pass_pct": passed_any / N_SIMS * 100,
        "avg_days": np.mean(pass_days_list) if pass_days_list else 0,
        "all_blown_pct": all_blown / N_SIMS * 100,
    }


# ═══════════════════════════════════════════════════════════════════════
# HTML report generation
# ═══════════════════════════════════════════════════════════════════════

def generate_slippage_html(challenge_data, funded_data, fullport_data):
    """Generate interactive HTML with slippage resistance charts."""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Apex Monte Carlo — Slippage Resistance</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; }
  h1 { text-align: center; color: #00d4ff; margin-bottom: 5px; font-size: 1.6em; }
  .subtitle { text-align: center; color: #888; margin-bottom: 25px; font-size: 0.9em; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  .chart-box { background: #12121a; border: 1px solid #2a2a3a; border-radius: 8px; padding: 15px; }
  .chart-box.full { grid-column: 1 / -1; }
  .chart-title { color: #00d4ff; font-size: 1.1em; margin-bottom: 10px; font-weight: 600; }
  .summary-box { background: #12121a; border: 1px solid #2a2a3a; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
  .summary-box h2 { color: #00d4ff; margin-bottom: 12px; font-size: 1.2em; }
  table { width: 100%%; border-collapse: collapse; font-size: 0.85em; }
  th { background: #1a1a2e; color: #00d4ff; padding: 8px 10px; text-align: right; border-bottom: 2px solid #2a2a3a; }
  td { padding: 7px 10px; text-align: right; border-bottom: 1px solid #1a1a2e; }
  tr:hover td { background: #1a1a28; }
  th:first-child, td:first-child { text-align: left; }
  .good { color: #00e676; }
  .warn { color: #ffab40; }
  .bad { color: #ff5252; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 600; }
  .tag-green { background: #00e67622; color: #00e676; }
  .tag-yellow { background: #ffab4022; color: #ffab40; }
  .tag-red { background: #ff525222; color: #ff5252; }
  .note { color: #888; font-size: 0.82em; margin-top: 8px; }
</style>
</head>
<body>
<h1>Apex Trader Funding — Monte Carlo Slippage Resistance</h1>
<div class="subtitle">Combined VWAP Strategy (Base SL 0.50/TP 1.90 + Trending SL 1.00/TP 1.00) | """ + f"{N_SIMS:,}" + """ sims | Apex rules: $2k DD, $3k target, $2,100 buffer, 50%% consistency</div>

"""

    slippages = challenge_data["slippages"]
    mnq_range = challenge_data["mnq_range"]

    # ── Chart 1: Challenge pass rate vs slippage ──
    html += '<div class="grid">\n<div class="chart-box"><div id="ch_pass" style="height:400px;"></div></div>\n'
    html += '<div class="chart-box"><div id="ch_blow" style="height:400px;"></div></div>\n</div>\n'

    # ── Chart 2: Funded payout chance + avg payouts vs slippage ──
    html += '<div class="grid">\n<div class="chart-box"><div id="fd_chance" style="height:400px;"></div></div>\n'
    html += '<div class="chart-box"><div id="fd_payouts" style="height:400px;"></div></div>\n</div>\n'

    # ── Chart 3: Funded net lifetime vs slippage ──
    html += '<div class="grid">\n<div class="chart-box"><div id="fd_net" style="height:400px;"></div></div>\n'
    html += '<div class="chart-box"><div id="fp_pass" style="height:400px;"></div></div>\n</div>\n'

    # ── Summary tables ──
    # Challenge table
    html += '<div class="summary-box"><h2>Challenge Pass Rate by Slippage</h2>\n<table><tr><th>MNQ</th>'
    for s in slippages:
        html += f'<th>{s:.2f}pt</th>'
    html += '</tr>\n'
    for mi, m in enumerate(mnq_range):
        html += f'<tr><td>{m} MNQ</td>'
        for si, s in enumerate(slippages):
            v = challenge_data["pass"][si][mi]
            cls = "good" if v >= 90 else ("warn" if v >= 70 else "bad")
            html += f'<td class="{cls}">{v:.1f}%</td>'
        html += '</tr>\n'
    html += '</table>\n<p class="note">Green >= 90% | Yellow >= 70% | Red < 70%</p></div>\n'

    # Funded table
    html += '<div class="summary-box"><h2>Funded Payout Chance by Slippage</h2>\n<table><tr><th>MNQ</th>'
    for s in slippages:
        html += f'<th>{s:.2f}pt</th>'
    html += '</tr>\n'
    for mi, m in enumerate(mnq_range):
        html += f'<tr><td>{m} MNQ</td>'
        for si, s in enumerate(slippages):
            v = funded_data["chance"][si][mi]
            cls = "good" if v >= 90 else ("warn" if v >= 70 else "bad")
            html += f'<td class="{cls}">{v:.1f}%</td>'
        html += '</tr>\n'
    html += '</table></div>\n'

    # Funded avg payouts table
    html += '<div class="summary-box"><h2>Avg Payouts by Slippage</h2>\n<table><tr><th>MNQ</th>'
    for s in slippages:
        html += f'<th>{s:.2f}pt</th>'
    html += '</tr>\n'
    for mi, m in enumerate(mnq_range):
        html += f'<tr><td>{m} MNQ</td>'
        for si, s in enumerate(slippages):
            v = funded_data["avg_payouts"][si][mi]
            html += f'<td>{v:.1f}</td>'
        html += '</tr>\n'
    html += '</table></div>\n'

    # Net lifetime table
    html += '<div class="summary-box"><h2>Net Lifetime Value by Slippage</h2>\n<table><tr><th>MNQ</th>'
    for s in slippages:
        html += f'<th>{s:.2f}pt</th>'
    html += '</tr>\n'
    for mi, m in enumerate(mnq_range):
        html += f'<tr><td>{m} MNQ</td>'
        for si, s in enumerate(slippages):
            v = funded_data["net"][si][mi]
            cls = "good" if v > 50000 else ("warn" if v > 10000 else "bad")
            html += f'<td class="{cls}">${v:+,.0f}</td>'
        html += '</tr>\n'
    html += '</table></div>\n'

    # Fullport table
    html += '<div class="summary-box"><h2>Fullport Pass Rate by Slippage ($1k risk/trade, dynamic sizing)</h2>\n<table><tr><th>Accounts</th>'
    for s in slippages:
        html += f'<th>{s:.2f}pt</th>'
    html += '</tr>\n'
    for ai, a in enumerate(fullport_data["account_counts"]):
        html += f'<tr><td>{a} accounts (${a * ACCOUNT_COST})</td>'
        for si, s in enumerate(slippages):
            v = fullport_data["pass"][si][ai]
            cls = "good" if v >= 90 else ("warn" if v >= 70 else "bad")
            html += f'<td class="{cls}">{v:.1f}%</td>'
        html += '</tr>\n'
    html += '</table></div>\n'

    # ── Plotly charts ──
    html += '<script>\n'
    colors = ['#00d4ff', '#00e676', '#ffab40', '#ff5252', '#bb86fc',
              '#03dac6', '#ff7043', '#42a5f5', '#ab47bc', '#26a69a']

    # Challenge pass rate chart
    html += 'var ch_traces = [\n'
    for si, s in enumerate(slippages):
        y = json.dumps(challenge_data["pass"][si])
        html += f'  {{x: {json.dumps(mnq_range)}, y: {y}, name: "{s:.2f}pt slip", type: "scatter", mode: "lines+markers", line: {{color: "{colors[si % len(colors)]}", width: 2}}}},\n'
    html += '];\n'
    html += """Plotly.newPlot('ch_pass', ch_traces, {
        title: {text: 'Challenge Pass Rate vs Slippage', font: {color: '#00d4ff', size: 14}},
        xaxis: {title: 'MNQ Contracts', color: '#888', gridcolor: '#1a1a2e'},
        yaxis: {title: 'Pass %', color: '#888', gridcolor: '#1a1a2e', range: [0, 105]},
        paper_bgcolor: '#12121a', plot_bgcolor: '#12121a', legend: {font: {color: '#ccc'}},
        font: {color: '#ccc'}
    }, {responsive: true});\n"""

    # Challenge blowup chart
    html += 'var cb_traces = [\n'
    for si, s in enumerate(slippages):
        y = json.dumps(challenge_data["blow"][si])
        html += f'  {{x: {json.dumps(mnq_range)}, y: {y}, name: "{s:.2f}pt slip", type: "scatter", mode: "lines+markers", line: {{color: "{colors[si % len(colors)]}", width: 2}}}},\n'
    html += '];\n'
    html += """Plotly.newPlot('ch_blow', cb_traces, {
        title: {text: 'Challenge Blowup Rate vs Slippage', font: {color: '#ff5252', size: 14}},
        xaxis: {title: 'MNQ Contracts', color: '#888', gridcolor: '#1a1a2e'},
        yaxis: {title: 'Blowup %', color: '#888', gridcolor: '#1a1a2e'},
        paper_bgcolor: '#12121a', plot_bgcolor: '#12121a', legend: {font: {color: '#ccc'}},
        font: {color: '#ccc'}
    }, {responsive: true});\n"""

    # Funded payout chance chart
    html += 'var fc_traces = [\n'
    for si, s in enumerate(slippages):
        y = json.dumps(funded_data["chance"][si])
        html += f'  {{x: {json.dumps(mnq_range)}, y: {y}, name: "{s:.2f}pt slip", type: "scatter", mode: "lines+markers", line: {{color: "{colors[si % len(colors)]}", width: 2}}}},\n'
    html += '];\n'
    html += """Plotly.newPlot('fd_chance', fc_traces, {
        title: {text: 'Funded Payout Chance vs Slippage', font: {color: '#00d4ff', size: 14}},
        xaxis: {title: 'MNQ Contracts', color: '#888', gridcolor: '#1a1a2e'},
        yaxis: {title: 'Chance of any payout %', color: '#888', gridcolor: '#1a1a2e', range: [0, 105]},
        paper_bgcolor: '#12121a', plot_bgcolor: '#12121a', legend: {font: {color: '#ccc'}},
        font: {color: '#ccc'}
    }, {responsive: true});\n"""

    # Funded avg payouts chart
    html += 'var fp_traces = [\n'
    for si, s in enumerate(slippages):
        y = json.dumps(funded_data["avg_payouts"][si])
        html += f'  {{x: {json.dumps(mnq_range)}, y: {y}, name: "{s:.2f}pt slip", type: "scatter", mode: "lines+markers", line: {{color: "{colors[si % len(colors)]}", width: 2}}}},\n'
    html += '];\n'
    html += """Plotly.newPlot('fd_payouts', fp_traces, {
        title: {text: 'Avg Payouts Over Funded Lifetime vs Slippage', font: {color: '#00e676', size: 14}},
        xaxis: {title: 'MNQ Contracts', color: '#888', gridcolor: '#1a1a2e'},
        yaxis: {title: 'Avg Payouts', color: '#888', gridcolor: '#1a1a2e'},
        paper_bgcolor: '#12121a', plot_bgcolor: '#12121a', legend: {font: {color: '#ccc'}},
        font: {color: '#ccc'}
    }, {responsive: true});\n"""

    # Net lifetime chart
    html += 'var fn_traces = [\n'
    for si, s in enumerate(slippages):
        y = json.dumps(funded_data["net"][si])
        html += f'  {{x: {json.dumps(mnq_range)}, y: {y}, name: "{s:.2f}pt slip", type: "scatter", mode: "lines+markers", line: {{color: "{colors[si % len(colors)]}", width: 2}}}},\n'
    html += '];\n'
    html += """Plotly.newPlot('fd_net', fn_traces, {
        title: {text: 'Net Lifetime Value vs Slippage', font: {color: '#00e676', size: 14}},
        xaxis: {title: 'MNQ Contracts', color: '#888', gridcolor: '#1a1a2e'},
        yaxis: {title: 'Net Lifetime $', color: '#888', gridcolor: '#1a1a2e'},
        paper_bgcolor: '#12121a', plot_bgcolor: '#12121a', legend: {font: {color: '#ccc'}},
        font: {color: '#ccc'}
    }, {responsive: true});\n"""

    # Fullport pass rate chart
    acct_counts = fullport_data["account_counts"]
    html += 'var fpp_traces = [\n'
    for si, s in enumerate(slippages):
        y = json.dumps(fullport_data["pass"][si])
        html += f'  {{x: {json.dumps(acct_counts)}, y: {y}, name: "{s:.2f}pt slip", type: "scatter", mode: "lines+markers", line: {{color: "{colors[si % len(colors)]}", width: 2}}}},\n'
    html += '];\n'
    html += """Plotly.newPlot('fp_pass', fpp_traces, {
        title: {text: 'Fullport Pass Rate vs Slippage ($1k risk/trade)', font: {color: '#bb86fc', size: 14}},
        xaxis: {title: 'Number of Accounts', color: '#888', gridcolor: '#1a1a2e'},
        yaxis: {title: 'Pass >= 1 account %', color: '#888', gridcolor: '#1a1a2e', range: [0, 105]},
        paper_bgcolor: '#12121a', plot_bgcolor: '#12121a', legend: {font: {color: '#ccc'}},
        font: {color: '#ccc'}
    }, {responsive: true});\n"""

    html += '</script>\n</body>\n</html>'
    return html


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("  APEX TRADER FUNDING — COMBINED VWAP MONTE CARLO")
    print(f"  Base: SL {BASE_SL}x / TP {BASE_TP}x (ADX 15-30) + Trend: SL {TREND_SL}x / TP {TREND_TP}x (ADX 30+)")
    print(f"  Apex rules: ${ACCOUNT_SIZE:,} acct | ${TRAIL_DD:,} trail DD | ${PROFIT_TARGET:,} target")
    print(f"  Funded: ${FUNDED_BUFFER:,} buffer | {CONSISTENCY_RULE*100:.0f}% consistency rule | ${ACCOUNT_COST} per account")
    print(f"  {N_SIMS:,} sims | MNQ $2/pt")
    print("=" * 110)
    sys.stdout.flush()

    print("\nBuilding ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done ({len(adx_lookup)} rows)")

    print("Collecting combined trades...", end=" ", flush=True)
    pnl_pool, mae_pool, sl_pool, day_counts = collect_combined_trades(adx_lookup)
    print(f"done")

    n = len(pnl_pool)
    w = pnl_pool[pnl_pool > 0]; l = pnl_pool[pnl_pool < 0]
    wr = len(w) / n * 100
    pf = abs(w.sum()) / abs(l.sum()) if len(l) else 999
    trading_days = np.sum(day_counts > 0)

    print(f"\n  Trade pool: {n} trades | {len(day_counts)} calendar days ({trading_days} trading)")
    print(f"  WR: {wr:.1f}% | PF: {pf:.2f} | Avg: {pnl_pool.mean():+.1f} pts/trade")
    print(f"  NQ total: ${pnl_pool.sum() * POINT_VALUE:+,.0f} | DD: ${(np.cumsum(pnl_pool) * POINT_VALUE - np.maximum.accumulate(np.cumsum(pnl_pool) * POINT_VALUE)).min():+,.0f}")
    sys.stdout.flush()

    mnq_range = list(range(1, 11))
    slippages = [0.0, 0.25, 0.50, 0.75, 1.00, 1.50]
    account_counts = [1, 2, 3, 4, 5, 6, 8, 10]

    # Storage for HTML
    ch_pass_data = []; ch_blow_data = []
    fd_chance_data = []; fd_payouts_data = []; fd_net_data = []
    fp_pass_data = []

    # CSV rows
    csv_challenge = []
    csv_funded = []
    csv_fullport = []

    # ===================================================================
    # PART 1: Challenge pass rate — fixed MNQ 1-10
    # ===================================================================
    print(f"\n{'=' * 110}")
    print(f"  PART 1: CHALLENGE PASS RATE (single account, MNQ 1-10)")
    print(f"{'=' * 110}")
    print(f"  {'MNQ':>4} {'Pass%':>7} {'AvgDays':>8} {'MedDays':>8} {'Blow%':>7}")
    print(f"  {'-'*42}")
    sys.stdout.flush()

    rng = np.random.default_rng(SEED)
    for c in mnq_range:
        r = sim_challenge(pnl_pool, mae_pool, day_counts, c, rng)
        print(f"  {c:>4} {r['pass_pct']:>6.1f}% {r['avg_days']:>8.0f} {r['med_days']:>8.0f} {r['blow_pct']:>6.1f}%", flush=True)
        csv_challenge.append({"mnq": c, "slippage": 0, **r})

    # ===================================================================
    # PART 2: Funded payouts — fixed MNQ 1-10
    # ===================================================================
    print(f"\n{'=' * 110}")
    print(f"  PART 2: FUNDED PAYOUTS (Apex: ${FUNDED_BUFFER} buffer, {CONSISTENCY_RULE*100:.0f}% consistency)")
    print(f"{'=' * 110}")
    print(f"  {'MNQ':>4} {'PayChance':>10} {'AvgPay':>7} {'MedPay':>7} {'AvgD/P':>7} {'Blow%':>7} {'NetLife':>10} {'$/mo':>8} {'ConBlocks':>10}")
    print(f"  {'-'*82}")
    sys.stdout.flush()

    rng = np.random.default_rng(SEED)
    for c in mnq_range:
        r = sim_funded(pnl_pool, mae_pool, day_counts, c, rng)
        monthly = (r["avg_payout_amount"] / r["avg_dpp"] * 21) if r["avg_dpp"] > 0 else 0
        print(f"  {c:>4} {r['payout_chance_pct']:>9.1f}% {r['avg_payouts']:>7.1f} {r['med_payouts']:>7.0f} "
              f"{r['avg_dpp']:>7.0f} {r['blow_pct']:>6.1f}% ${r['net_lifetime']:>+9,.0f} ${monthly:>7,.0f} {r['consistency_blocks']:>10}",
              flush=True)
        csv_funded.append({"mnq": c, "slippage": 0, "monthly": monthly, **r})

    # ===================================================================
    # PART 3: Fullport multi-account
    # ===================================================================
    print(f"\n{'=' * 110}")
    print(f"  PART 3: FULLPORT ROTATION ($1k risk/trade, dynamic MNQ sizing)")
    print(f"{'=' * 110}")
    print(f"  {'Accts':>5} {'Cost':>6} {'Pass>=1':>8} {'AvgDays':>8} {'AllBlown':>9} {'EV/attempt':>12} {'Att2Pass':>9}")
    print(f"  {'-'*62}")
    sys.stdout.flush()

    rng = np.random.default_rng(SEED)
    for na in account_counts:
        r = sim_fullport(pnl_pool, mae_pool, sl_pool, day_counts, na, rng)
        cost = na * ACCOUNT_COST
        pr = r["pass_pct"] / 100
        ev = pr * PROFIT_TARGET - cost if pr > 0 else -cost
        att = 1 / pr if pr > 0 else float('inf')
        print(f"  {na:>5} ${cost:>4} {r['pass_pct']:>7.1f}% {r['avg_days']:>7.1f}d {r['all_blown_pct']:>8.1f}% "
              f"${ev:>+10,.0f} {att:>8.1f}x", flush=True)
        csv_fullport.append({"accounts": na, "slippage": 0, "cost": cost, "ev": ev, **r})

    # ===================================================================
    # PART 4: Slippage resistance (all three modes)
    # ===================================================================
    print(f"\n{'=' * 110}")
    print(f"  PART 4: SLIPPAGE RESISTANCE")
    print(f"{'=' * 110}")
    sys.stdout.flush()

    for si, slip in enumerate(slippages):
        print(f"\n  --- Slippage: {slip:.2f} pt roundtrip ---")

        # Challenge
        ch_pass_row = []; ch_blow_row = []
        print(f"    Challenge:  ", end="", flush=True)
        rng = np.random.default_rng(SEED)
        for c in mnq_range:
            r = sim_challenge(pnl_pool, mae_pool, day_counts, c, rng, slippage=slip)
            ch_pass_row.append(round(r["pass_pct"], 1))
            ch_blow_row.append(round(r["blow_pct"], 1))
            csv_challenge.append({"mnq": c, "slippage": slip, **r})
            print(f"{c}MNQ={r['pass_pct']:.0f}% ", end="", flush=True)
        print()
        ch_pass_data.append(ch_pass_row)
        ch_blow_data.append(ch_blow_row)

        # Funded
        fd_chance_row = []; fd_pay_row = []; fd_net_row = []
        print(f"    Funded:     ", end="", flush=True)
        rng = np.random.default_rng(SEED)
        for c in mnq_range:
            r = sim_funded(pnl_pool, mae_pool, day_counts, c, rng, slippage=slip)
            fd_chance_row.append(round(r["payout_chance_pct"], 1))
            fd_pay_row.append(round(r["avg_payouts"], 1))
            fd_net_row.append(round(r["net_lifetime"], 0))
            csv_funded.append({"mnq": c, "slippage": slip, **r})
            print(f"{c}MNQ={r['payout_chance_pct']:.0f}%/{r['avg_payouts']:.1f}p ", end="", flush=True)
        print()
        fd_chance_data.append(fd_chance_row)
        fd_payouts_data.append(fd_pay_row)
        fd_net_data.append(fd_net_row)

        # Fullport
        fp_pass_row = []
        print(f"    Fullport:   ", end="", flush=True)
        rng = np.random.default_rng(SEED)
        for na in account_counts:
            r = sim_fullport(pnl_pool, mae_pool, sl_pool, day_counts, na, rng, slippage=slip)
            fp_pass_row.append(round(r["pass_pct"], 1))
            csv_fullport.append({"accounts": na, "slippage": slip, "cost": na * ACCOUNT_COST, **r})
            print(f"{na}acct={r['pass_pct']:.0f}% ", end="", flush=True)
        print()
        fp_pass_data.append(fp_pass_row)

        sys.stdout.flush()

    # ===================================================================
    # Save results
    # ===================================================================
    print(f"\n{'=' * 110}")
    print(f"  SAVING RESULTS")
    print(f"{'=' * 110}")

    # CSV: challenge
    ch_path = RESULTS_DIR / "apex_challenge_results.csv"
    with open(ch_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_challenge[0].keys())
        w.writeheader(); w.writerows(csv_challenge)
    print(f"  Saved: {ch_path}")

    # CSV: funded
    fd_path = RESULTS_DIR / "apex_funded_results.csv"
    with open(fd_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_funded[0].keys())
        w.writeheader(); w.writerows(csv_funded)
    print(f"  Saved: {fd_path}")

    # CSV: fullport
    fp_path = RESULTS_DIR / "apex_fullport_results.csv"
    with open(fp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fullport[0].keys())
        w.writeheader(); w.writerows(csv_fullport)
    print(f"  Saved: {fp_path}")

    # HTML
    html_content = generate_slippage_html(
        challenge_data={"slippages": slippages, "mnq_range": mnq_range,
                        "pass": ch_pass_data, "blow": ch_blow_data},
        funded_data={"slippages": slippages, "mnq_range": mnq_range,
                     "chance": fd_chance_data, "avg_payouts": fd_payouts_data,
                     "net": fd_net_data},
        fullport_data={"slippages": slippages, "account_counts": account_counts,
                       "pass": fp_pass_data},
    )
    html_path = RESULTS_DIR / "apex_slippage_resistance.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  Saved: {html_path}")

    # ===================================================================
    # PART 5: Recommendations
    # ===================================================================
    print(f"\n{'=' * 110}")
    print(f"  RECOMMENDATIONS")
    print(f"{'=' * 110}")
    print(f"\n  Account cost: ${ACCOUNT_COST} | Apex rules")
    print(f"  Consistency rule (50%) may delay some payouts — check ConBlocks column")
    print(f"\n  See HTML report for interactive slippage charts:")
    print(f"    {html_path}")
    sys.stdout.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()

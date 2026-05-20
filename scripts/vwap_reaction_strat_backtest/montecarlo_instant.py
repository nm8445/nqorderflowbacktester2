"""
Monte Carlo — Instant Funding account simulation.

Rules:
  - $50k account, $2k EOD trailing DD (stops trailing at $50k)
  - Daily loss limit: $1,250 (account blown if daily unrealized loss hits this)
  - 20% consistency: max single day P&L <= 20% of total profit at payout time
  - Minimum payout: $1,000 (withdraw all profit above buffer)
  - Buffer: $2k above DD floor (after payout, equity resets to $52k)
  - No challenge — start funded immediately
  - MNQ only (point value $2/pt)
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"

ACCOUNT_SIZE     = 50_000
TRAIL_DD         = 2_000
DAILY_LOSS_LIMIT = 1_250
CONSISTENCY      = 0.20
MIN_PAYOUT       = 1_000
MAX_PAYOUT       = 2_000
BUFFER           = 2_000   # keep $2k above DD floor after withdrawal

MNQ_PV = 2.0
SL_MULT = 1.0
TP_MULT = 2.0

N_SIMS = 10_000
MAX_DAYS = 1500
SEED = 42


def load_days():
    start = datetime.strptime("2025-03-13", "%Y-%m-%d").date()
    end = datetime.strptime("2026-04-08", "%Y-%m-%d").date()
    days = []
    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        with open(cache_file, 'rb') as f:
            vd = pickle.load(f)
        signals = vd["signals"]
        if not signals:
            continue
        scf = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not scf.exists():
            continue
        with open(scf, 'rb') as f:
            sd = pickle.load(f)
        days.append((date_str, signals, sd["bars"]))
    return days


def collect_trades(days):
    """Return (pnl_array, mae_array, day_counts) for default 1.0/2.0 config."""
    all_dates = set()
    daily_pnls = {}
    daily_maes = {}

    for date_str, signals, bars in days:
        all_dates.add(date_str)
        day_trades = []
        day_maes = []
        last_exit_time = None

        for signal in signals:
            ep = signal["entry_price"]; d = signal["direction"]; atr = signal["atr"]
            cbi = signal["bar_index"] + 1
            if atr is None or atr <= 0:
                continue
            ct = signal["confirm_time"]
            if hasattr(ct, 'tz_convert'):
                cet = ct.tz_convert(ET)
            else:
                cet = pd.Timestamp(ct, tz='UTC').tz_convert(ET)
            if "16:00" <= cet.strftime("%H:%M") < "19:10":
                continue
            if last_exit_time is not None and ct <= last_exit_time:
                continue

            sl = ep - atr * SL_MULT if d == "long" else ep + atr * SL_MULT
            tp = ep + atr * TP_MULT if d == "long" else ep - atr * TP_MULT
            if d == "long" and (tp <= ep or sl >= ep): continue
            if d == "short" and (tp >= ep or sl <= ep): continue

            exp = None; ext = None
            worst_adverse = 0.0
            for j in range(cbi + 1, len(bars)):
                bar = bars[j]
                if not bar.closed: continue
                if d == "long":
                    adverse = ep - bar.low
                else:
                    adverse = bar.high - ep
                if adverse > worst_adverse:
                    worst_adverse = adverse

                bct = bar.close_time
                if hasattr(bct, 'tz_convert'): bet = bct.tz_convert(ET)
                else: bet = pd.Timestamp(bct, tz='UTC').tz_convert(ET)
                if bet.strftime("%H:%M") >= FORCE_CLOSE:
                    exp = bar.close; ext = bct; break
                if d == "short":
                    if bar.high >= sl: exp = sl; ext = bct; break
                    if bar.low <= tp: exp = tp; ext = bct; break
                else:
                    if bar.low <= sl: exp = sl; ext = bct; break
                    if bar.high >= tp: exp = tp; ext = bct; break
            if exp is None:
                exp = bars[-1].close; ext = bars[-1].close_time

            pnl = (ep - exp) if d == "short" else (exp - ep)
            day_trades.append(pnl)
            day_maes.append(worst_adverse)
            last_exit_time = ext

        if day_trades:
            daily_pnls[date_str] = day_trades
            daily_maes[date_str] = day_maes

    all_pnl = []
    all_mae = []
    day_counts = []
    for dd in sorted(all_dates):
        if dd in daily_pnls:
            day_counts.append(len(daily_pnls[dd]))
            all_pnl.extend(daily_pnls[dd])
            all_mae.extend(daily_maes[dd])
        else:
            day_counts.append(0)

    return (np.array(all_pnl, dtype=np.float64),
            np.array(all_mae, dtype=np.float64),
            np.array(day_counts, dtype=np.int32))


def run_instant_funded(trade_pool, mae_pool, day_count_dist, contracts, rng):
    """Simulate instant funded accounts. Returns per-sim results."""
    pv = MNQ_PV
    trades_per_day = rng.choice(day_count_dist, size=(N_SIMS, MAX_DAYS))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(trade_pool), size=total_needed)
    all_pnls = trade_pool[idx]
    all_maes = mae_pool[idx]

    results = []
    cursor = 0

    for sim in range(N_SIMS):
        equity = float(ACCOUNT_SIZE)
        hwm = float(ACCOUNT_SIZE)
        dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
        dd_locked = False
        payouts = 0
        total_withdrawn = 0.0
        max_day_pnl = 0.0
        days_to_first_payout = 0
        first_payout_found = False

        for di in range(MAX_DAYS):
            nt = int(trades_per_day[sim, di])
            if nt == 0:
                if not first_payout_found:
                    days_to_first_payout += 1
                continue

            day_open_equity = equity
            day_pnl = 0.0
            blown = False

            for _ in range(nt):
                trade_pnl = all_pnls[cursor] * pv * contracts
                trade_mae = all_maes[cursor] * pv * contracts
                cursor += 1

                # Check unrealized DD breach
                if equity - trade_mae <= dd_level:
                    blown = True; break

                # Check daily loss limit (unrealized)
                unrealized_day_loss = (day_open_equity - (equity - trade_mae))
                if unrealized_day_loss >= DAILY_LOSS_LIMIT:
                    blown = True; break

                equity += trade_pnl
                day_pnl += trade_pnl

                # Check realized DD breach
                if equity <= dd_level:
                    blown = True; break

                # Check daily loss limit (realized)
                if day_open_equity - equity >= DAILY_LOSS_LIMIT:
                    blown = True; break

            if blown:
                for rd in range(di + 1, MAX_DAYS):
                    cursor += int(trades_per_day[sim, rd])
                break

            # EOD: update trailing DD
            if equity > hwm:
                hwm = equity
                if not dd_locked:
                    dd_level = hwm - TRAIL_DD
                    if dd_level >= ACCOUNT_SIZE:
                        dd_level = float(ACCOUNT_SIZE)
                        dd_locked = True

            if equity <= dd_level:
                for rd in range(di + 1, MAX_DAYS):
                    cursor += int(trades_per_day[sim, rd])
                break

            if day_pnl > 0 and day_pnl > max_day_pnl:
                max_day_pnl = day_pnl

            if not first_payout_found:
                days_to_first_payout += 1

            # Check payout eligibility
            buffer_eq = ACCOUNT_SIZE + BUFFER
            profit = equity - buffer_eq
            if profit >= MIN_PAYOUT:
                # 20% consistency check
                can_withdraw = True
                if max_day_pnl > CONSISTENCY * profit:
                    can_withdraw = False
                if can_withdraw:
                    withdraw = min(profit, MAX_PAYOUT)
                    payouts += 1
                    total_withdrawn += withdraw
                    if not first_payout_found:
                        first_payout_found = True
                    equity -= withdraw
                    hwm = equity
                    max_day_pnl = 0.0

        results.append({
            "payouts": payouts,
            "total_withdrawn": total_withdrawn,
            "days_to_first": days_to_first_payout if first_payout_found else -1,
            "got_payout": first_payout_found,
        })

    return results


def main():
    print("Loading data...", flush=True)
    days = load_days()
    print(f"  {len(days)} trading days loaded\n", flush=True)

    trade_pool, mae_pool, day_counts = collect_trades(days)
    n = len(trade_pool)
    w = trade_pool[trade_pool > 0]
    wr = len(w) / n * 100
    exp = trade_pool.mean()
    print(f"  Trade pool: {n} trades | WR: {wr:.1f}% | Exp: {exp:+.2f} pts", flush=True)
    print(f"  Avg MAE: {mae_pool.mean():.2f} pts | Max MAE: {mae_pool.max():.2f} pts\n", flush=True)

    rng = np.random.default_rng(SEED)

    print("=" * 100)
    print("INSTANT FUNDED — MNQ Risk (1.0 SL / 2.0 TP ATR)")
    print(f"Account: ${ACCOUNT_SIZE:,} | DD: ${TRAIL_DD:,} trailing (locks at ${ACCOUNT_SIZE:,})")
    print(f"Daily loss limit: ${DAILY_LOSS_LIMIT:,} | Consistency: {CONSISTENCY:.0%}")
    print(f"Min payout: ${MIN_PAYOUT:,} | Max payout: ${MAX_PAYOUT:,} | Buffer: ${BUFFER:,} | {N_SIMS:,} sims")
    print(f"Unrealized P&L tracked for DD and daily loss limit")
    print("=" * 100)

    header = (f"{'Contracts':>10s} | {'Got Payout%':>11s} | {'Avg Days':>8s} | "
              f"{'Avg Payouts':>11s} | {'Avg $/Pay':>9s} | {'Avg $/Life':>10s} | "
              f"{'Med Days':>8s}")
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    for contracts in range(1, 6):
        results = run_instant_funded(trade_pool, mae_pool, day_counts, contracts, rng)

        got_payout = [r for r in results if r["got_payout"]]
        pct_payout = len(got_payout) / len(results) * 100
        avg_days = np.mean([r["days_to_first"] for r in got_payout]) if got_payout else 0
        med_days = np.median([r["days_to_first"] for r in got_payout]) if got_payout else 0
        avg_payouts = np.mean([r["payouts"] for r in results])
        avg_total = np.mean([r["total_withdrawn"] for r in results])
        avg_per_pay = avg_total / avg_payouts if avg_payouts > 0 else 0

        print(f"  MNQ {contracts:>2d}   | {pct_payout:10.1f}% | {avg_days:7.1f}d | "
              f"{avg_payouts:11.1f} | ${avg_per_pay:>7,.0f} | ${avg_total:>9,.0f} | "
              f"{med_days:7.0f}d", flush=True)

    # Also show what happens WITHOUT consistency for comparison
    print()
    print("=" * 100)
    print("SAME BUT WITHOUT CONSISTENCY (for comparison)")
    print("=" * 100)
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    # Temporarily disable consistency
    import types
    original_consistency = CONSISTENCY

    for contracts in range(1, 6):
        # Run with consistency=None by modifying the check inline
        pv = MNQ_PV
        trades_per_day = rng.choice(day_counts, size=(N_SIMS, MAX_DAYS))
        total_needed = int(trades_per_day.sum())
        idx = rng.integers(0, len(trade_pool), size=total_needed)
        all_pnls_arr = trade_pool[idx]
        all_maes_arr = mae_pool[idx]

        sim_results = []
        cursor = 0

        for sim in range(N_SIMS):
            equity = float(ACCOUNT_SIZE)
            hwm = float(ACCOUNT_SIZE)
            dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
            dd_locked = False
            payouts = 0
            total_withdrawn = 0.0
            max_day_pnl = 0.0
            days_to_first = 0
            first_found = False

            for di in range(MAX_DAYS):
                nt = int(trades_per_day[sim, di])
                if nt == 0:
                    if not first_found:
                        days_to_first += 1
                    continue

                day_open = equity
                day_pnl = 0.0
                blown = False

                for _ in range(nt):
                    tp_val = all_pnls_arr[cursor] * pv * contracts
                    tm_val = all_maes_arr[cursor] * pv * contracts
                    cursor += 1
                    if equity - tm_val <= dd_level:
                        blown = True; break
                    if day_open - (equity - tm_val) >= DAILY_LOSS_LIMIT:
                        blown = True; break
                    equity += tp_val
                    day_pnl += tp_val
                    if equity <= dd_level:
                        blown = True; break
                    if day_open - equity >= DAILY_LOSS_LIMIT:
                        blown = True; break

                if blown:
                    for rd in range(di + 1, MAX_DAYS):
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
                    for rd in range(di + 1, MAX_DAYS):
                        cursor += int(trades_per_day[sim, rd])
                    break

                if day_pnl > 0 and day_pnl > max_day_pnl:
                    max_day_pnl = day_pnl

                if not first_found:
                    days_to_first += 1

                buffer_eq = ACCOUNT_SIZE + BUFFER
                profit = equity - buffer_eq
                if profit >= MIN_PAYOUT:
                    # NO consistency check
                    withdraw = min(profit, MAX_PAYOUT)
                    payouts += 1
                    total_withdrawn += withdraw
                    if not first_found:
                        first_found = True
                    equity -= withdraw
                    hwm = equity
                    max_day_pnl = 0.0

            sim_results.append({
                "payouts": payouts,
                "total_withdrawn": total_withdrawn,
                "days_to_first": days_to_first if first_found else -1,
                "got_payout": first_found,
            })

        got_payout = [r for r in sim_results if r["got_payout"]]
        pct_payout = len(got_payout) / len(sim_results) * 100
        avg_days = np.mean([r["days_to_first"] for r in got_payout]) if got_payout else 0
        med_days = np.median([r["days_to_first"] for r in got_payout]) if got_payout else 0
        avg_payouts = np.mean([r["payouts"] for r in sim_results])
        avg_total = np.mean([r["total_withdrawn"] for r in sim_results])
        avg_per_pay = avg_total / avg_payouts if avg_payouts > 0 else 0

        print(f"  MNQ {contracts:>2d}   | {pct_payout:10.1f}% | {avg_days:7.1f}d | "
              f"{avg_payouts:11.1f} | ${avg_per_pay:>7,.0f} | ${avg_total:>9,.0f} | "
              f"{med_days:7.0f}d", flush=True)


if __name__ == "__main__":
    main()

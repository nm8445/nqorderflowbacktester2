"""
Monte Carlo Prop Firm Simulation — VWAP Reaction Strategy (SL 1.0x / TP 2.0x).

Prop firm rules:
  - $50k account, $2k EOD trailing drawdown, $3k profit target
  - Challenge: trailing DD never stops
  - Funded: DD stops trailing once DD level reaches starting balance ($50k)
  - Consistency: max single day P&L <= X% of total profit (50%, 40%, or none)
  - Martingale: 2 contracts after 2 consecutive losses, else 1

Instruments:
  - MNQ: $2/point per contract (1–5 contracts)
  - NQ: $20/point per contract (1–3 contracts)
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"
SL_MULT = 1.0
TP_MULT = 2.0

ACCOUNT_SIZE   = 50_000
TRAIL_DD       = 2_000
PROFIT_TARGET  = 3_000
FUNDED_BUFFER  = 2_000

N_SIMS       = 10_000
MAX_DAYS_CHALLENGE = 500
MAX_DAYS_FUNDED    = 1500
SEED         = 42

MNQ_PV = 2.0
NQ_PV  = 20.0


def collect_trades():
    """Return trade P&L array (points) and per-day trade count array."""
    start = datetime.strptime("2025-03-13", "%Y-%m-%d").date()
    end = datetime.strptime("2026-04-08", "%Y-%m-%d").date()

    daily_pnls = {}
    all_dates = set()

    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        all_dates.add(date_str)

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

        day_trades = []
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
                    exit_price = bar.close; exit_time = bar.close_time; break
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
                exit_price = last_bar.close; exit_time = last_bar.close_time
            pnl = (entry_price - exit_price) if direction == "short" else (exit_price - entry_price)
            day_trades.append(pnl)
            last_exit_time = exit_time

        if day_trades:
            daily_pnls[date_str] = day_trades

    all_pnl = []
    day_counts = []
    for d in sorted(all_dates):
        if d in daily_pnls:
            day_counts.append(len(daily_pnls[d]))
            all_pnl.extend(daily_pnls[d])
        else:
            day_counts.append(0)

    return np.array(all_pnl, dtype=np.float64), np.array(day_counts, dtype=np.int32)


def _pregenerate_days(trade_pool, day_count_dist, max_days, n_sims, rng):
    """Pre-generate all random data for simulations.
    Returns: trades_per_day (n_sims, max_days), trade_pnl_flat, day_offsets
    """
    # For each sim and day, how many trades
    trades_per_day = rng.choice(day_count_dist, size=(n_sims, max_days))
    # Pre-generate all trade indices
    total_trades_needed = int(trades_per_day.sum())
    trade_indices = rng.integers(0, len(trade_pool), size=total_trades_needed)
    trade_pnls = trade_pool[trade_indices]
    return trades_per_day, trade_pnls


def run_challenge_batch(trade_pool, day_count_dist, point_value, base_contracts,
                        use_martingale, consistency, n_sims, rng):
    """Run n_sims challenge simulations. Returns (pass_rate, avg_days_to_pass)."""
    trades_per_day, all_trade_pnls = _pregenerate_days(
        trade_pool, day_count_dist, MAX_DAYS_CHALLENGE, n_sims, rng
    )

    passes = 0
    pass_days_list = []
    trade_cursor = 0

    for sim in range(n_sims):
        equity = float(ACCOUNT_SIZE)
        hwm = float(ACCOUNT_SIZE)
        dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
        loss_streak = 0
        max_day_pnl = 0.0
        blown = False
        passed = False
        day_count = 0
        total_profit_for_consistency = 0.0

        for day_idx in range(MAX_DAYS_CHALLENGE):
            n_trades = int(trades_per_day[sim, day_idx])
            if n_trades == 0:
                day_count += 1
                continue

            day_pnl = 0.0
            for _ in range(n_trades):
                pnl_pts = all_trade_pnls[trade_cursor]
                trade_cursor += 1

                if use_martingale and loss_streak >= 2:
                    contracts = base_contracts * 2
                else:
                    contracts = base_contracts

                trade_pnl = pnl_pts * point_value * contracts
                equity += trade_pnl
                day_pnl += trade_pnl

                if equity <= dd_level:
                    blown = True
                    break

                if pnl_pts < 0:
                    loss_streak += 1
                else:
                    loss_streak = 0

            if blown:
                # Still need to consume remaining pre-generated trades for this sim
                for remaining_day in range(day_idx + 1, MAX_DAYS_CHALLENGE):
                    trade_cursor += int(trades_per_day[sim, remaining_day])
                break

            day_count += 1

            if equity > hwm:
                hwm = equity
                dd_level = hwm - TRAIL_DD

            if equity <= dd_level:
                for remaining_day in range(day_idx + 1, MAX_DAYS_CHALLENGE):
                    trade_cursor += int(trades_per_day[sim, remaining_day])
                break

            total_profit = equity - ACCOUNT_SIZE
            if day_pnl > 0 and day_pnl > max_day_pnl:
                max_day_pnl = day_pnl

            if total_profit >= PROFIT_TARGET:
                if consistency is None:
                    passed = True
                else:
                    if max_day_pnl <= consistency * total_profit:
                        passed = True

                if passed:
                    for remaining_day in range(day_idx + 1, MAX_DAYS_CHALLENGE):
                        trade_cursor += int(trades_per_day[sim, remaining_day])
                    break

        if passed:
            passes += 1
            pass_days_list.append(day_count)

    rate = passes / n_sims * 100
    avg_days = np.mean(pass_days_list) if pass_days_list else 0
    return rate, avg_days


def run_funded_batch(trade_pool, day_count_dist, point_value, base_contracts,
                     use_martingale, consistency, n_sims, rng):
    """Run n_sims funded simulations. Returns (avg_payouts, avg_days_per_payout)."""
    trades_per_day, all_trade_pnls = _pregenerate_days(
        trade_pool, day_count_dist, MAX_DAYS_FUNDED, n_sims, rng
    )

    all_payouts = []
    all_days_per_payout = []
    trade_cursor = 0

    for sim in range(n_sims):
        equity = float(ACCOUNT_SIZE)
        hwm = float(ACCOUNT_SIZE)
        dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
        dd_locked = False
        loss_streak = 0
        payouts = 0
        cycle_start_day = 0
        max_day_pnl = 0.0

        for day_idx in range(MAX_DAYS_FUNDED):
            n_trades = int(trades_per_day[sim, day_idx])
            if n_trades == 0:
                continue

            day_pnl = 0.0
            blown = False
            for _ in range(n_trades):
                pnl_pts = all_trade_pnls[trade_cursor]
                trade_cursor += 1

                if use_martingale and loss_streak >= 2:
                    contracts = base_contracts * 2
                else:
                    contracts = base_contracts

                trade_pnl = pnl_pts * point_value * contracts
                equity += trade_pnl
                day_pnl += trade_pnl

                if equity <= dd_level:
                    blown = True
                    break

                if pnl_pts < 0:
                    loss_streak += 1
                else:
                    loss_streak = 0

            if blown:
                for remaining_day in range(day_idx + 1, MAX_DAYS_FUNDED):
                    trade_cursor += int(trades_per_day[sim, remaining_day])
                break

            if equity > hwm:
                hwm = equity
                if not dd_locked:
                    dd_level = hwm - TRAIL_DD
                    if dd_level >= ACCOUNT_SIZE:
                        dd_level = float(ACCOUNT_SIZE)
                        dd_locked = True

            if equity <= dd_level:
                for remaining_day in range(day_idx + 1, MAX_DAYS_FUNDED):
                    trade_cursor += int(trades_per_day[sim, remaining_day])
                break

            if day_pnl > 0 and day_pnl > max_day_pnl:
                max_day_pnl = day_pnl

            buffer_equity = ACCOUNT_SIZE + FUNDED_BUFFER
            profit_above_buffer = equity - buffer_equity

            if profit_above_buffer >= PROFIT_TARGET:
                can_payout = True
                if consistency is not None:
                    if max_day_pnl > consistency * profit_above_buffer:
                        can_payout = False

                if can_payout:
                    payouts += 1
                    all_days_per_payout.append(day_idx + 1 - cycle_start_day)
                    equity = buffer_equity
                    hwm = buffer_equity
                    max_day_pnl = 0.0
                    cycle_start_day = day_idx + 1

        all_payouts.append(payouts)

    avg_payouts = np.mean(all_payouts)
    avg_dpp = np.mean(all_days_per_payout) if all_days_per_payout else 0
    return avg_payouts, avg_dpp


def test_path_dependency(trade_pool, day_count_dist, point_value, base_contracts, rng):
    """Compare pass rates: random vs wins-first vs losses-first."""
    n_test = 2000

    # Random
    random_rate, _ = run_challenge_batch(
        trade_pool, day_count_dist, point_value, base_contracts,
        False, None, n_test, rng
    )

    # Wins first / losses first: sort the trade pool, draw sequentially
    results = {"random": random_rate}

    for label, sorted_pool in [
        ("wins_first", np.sort(trade_pool)[::-1]),
        ("losses_first", np.sort(trade_pool)),
    ]:
        passes = 0
        for _ in range(n_test):
            shuffled = rng.permutation(sorted_pool)
            # Bias: partially sorted — mix 70% sorted order with 30% random
            n = len(shuffled)
            n_sorted = int(n * 0.7)
            pool = np.concatenate([sorted_pool[:n_sorted], rng.permutation(sorted_pool[n_sorted:])])

            equity = float(ACCOUNT_SIZE)
            hwm = float(ACCOUNT_SIZE)
            dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
            passed = False
            idx = 0
            for day in range(MAX_DAYS_CHALLENGE):
                n_trades = int(rng.choice(day_count_dist))
                blown = False
                for _ in range(n_trades):
                    pnl_pts = pool[idx % len(pool)]
                    idx += 1
                    trade_pnl = pnl_pts * point_value * base_contracts
                    equity += trade_pnl
                    if equity <= dd_level:
                        blown = True
                        break
                if blown:
                    break
                if equity > hwm:
                    hwm = equity
                    dd_level = hwm - TRAIL_DD
                if equity <= dd_level:
                    break
                if (equity - ACCOUNT_SIZE) >= PROFIT_TARGET:
                    passed = True
                    break
            if passed:
                passes += 1
        results[label] = passes / n_test * 100

    return results


def main():
    print("Collecting trade data...", flush=True)
    trade_pool, day_count_dist = collect_trades()
    print(f"  {len(trade_pool)} trades, {len(day_count_dist)} trading days")
    print(f"  Avg trades/day: {day_count_dist.mean():.1f} (including 0-trade days)")
    dc = Counter(day_count_dist.tolist())
    print(f"  Day distribution: {dict(sorted(dc.items()))}")
    print(f"  Avg P&L: {trade_pool.mean():+.2f} pts | WR: {(trade_pool > 0).sum()/len(trade_pool)*100:.1f}%")
    print(flush=True)

    rng = np.random.default_rng(SEED)

    configs = []
    for c in range(1, 6):
        configs.append(("MNQ", c, MNQ_PV))
    for c in range(1, 4):
        configs.append(("NQ", c, NQ_PV))

    consistency_options = [("none", None), ("50%", 0.50), ("40%", 0.40)]
    martingale_options = [False, True]

    # ── CHALLENGE ──────────────────────────────────────────────────────
    print("=" * 120)
    print("CHALLENGE PHASE — PASS RATES & AVG DAYS TO PASS")
    print(f"Account: ${ACCOUNT_SIZE:,} | Trailing DD: ${TRAIL_DD:,} | Target: ${PROFIT_TARGET:,} | Sims: {N_SIMS:,}")
    print("=" * 120)

    header = (
        f"{'Instr':>5s} {'Con':>3s} {'Mart':>4s} | "
        f"{'No Cons':>8s} {'Days':>5s} | "
        f"{'50% Cons':>8s} {'Days':>5s} | "
        f"{'40% Cons':>8s} {'Days':>5s} | "
        f"{'Risk/Trade':>12s}"
    )
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    avg_sl_pts = abs(trade_pool[trade_pool < 0].mean())

    for instr, base_con, pv in configs:
        for mart in martingale_options:
            row = []
            max_con = base_con * 2 if mart else base_con
            risk_str = f"${avg_sl_pts * pv * base_con:,.0f}"
            if mart:
                risk_str += f"-${avg_sl_pts * pv * max_con:,.0f}"

            for _, cons_val in consistency_options:
                rate, avg_days = run_challenge_batch(
                    trade_pool, day_count_dist, pv, base_con, mart, cons_val, N_SIMS, rng
                )
                row.append((rate, avg_days))

            mart_str = "YES" if mart else "NO"
            print(
                f"{instr:>5s} {base_con:>3d} {mart_str:>4s} | "
                f"{row[0][0]:7.1f}% {row[0][1]:5.1f} | "
                f"{row[1][0]:7.1f}% {row[1][1]:5.1f} | "
                f"{row[2][0]:7.1f}% {row[2][1]:5.1f} | "
                f"{risk_str:>12s}",
                flush=True,
            )

    # ── FUNDED ─────────────────────────────────────────────────────────
    print()
    print("=" * 120)
    print("FUNDED PHASE — AVG PAYOUTS BEFORE BLOW & AVG DAYS PER PAYOUT")
    print(f"Account: ${ACCOUNT_SIZE:,} | DD stops at ${ACCOUNT_SIZE:,} | Buffer: ${FUNDED_BUFFER:,} | Payout target: ${PROFIT_TARGET:,}")
    print("=" * 120)

    header2 = (
        f"{'Instr':>5s} {'Con':>3s} {'Mart':>4s} | "
        f"{'--- No Consistency ---':>22s} | "
        f"{'--- 50% Consistency --':>22s} | "
        f"{'--- 40% Consistency --':>22s}"
    )
    sub_header = (
        f"{'':>14s} | "
        f"{'Payouts':>7s} {'Days/P':>7s} {'$/Blow':>7s} | "
        f"{'Payouts':>7s} {'Days/P':>7s} {'$/Blow':>7s} | "
        f"{'Payouts':>7s} {'Days/P':>7s} {'$/Blow':>7s}"
    )
    print(header2)
    print(sub_header)
    print("-" * len(header2))
    sys.stdout.flush()

    for instr, base_con, pv in configs:
        for mart in martingale_options:
            row = []
            for _, cons_val in consistency_options:
                avg_payouts, avg_dpp = run_funded_batch(
                    trade_pool, day_count_dist, pv, base_con, mart, cons_val, N_SIMS, rng
                )
                net_per_blow = avg_payouts * PROFIT_TARGET - TRAIL_DD
                row.append((avg_payouts, avg_dpp, net_per_blow))

            mart_str = "YES" if mart else "NO"
            print(
                f"{instr:>5s} {base_con:>3d} {mart_str:>4s} | "
                f"{row[0][0]:7.1f} {row[0][1]:7.1f} {row[0][2]:+7.0f} | "
                f"{row[1][0]:7.1f} {row[1][1]:7.1f} {row[1][2]:+7.0f} | "
                f"{row[2][0]:7.1f} {row[2][1]:7.1f} {row[2][2]:+7.0f}",
                flush=True,
            )

    # ── PATH DEPENDENCY ────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("PATH DEPENDENCY TEST (no consistency, no martingale)")
    print("=" * 80)

    test_configs = [("MNQ", 2, MNQ_PV), ("MNQ", 4, MNQ_PV), ("NQ", 1, NQ_PV), ("NQ", 2, NQ_PV)]

    header3 = f"{'Instr':>5s} {'Con':>3s} | {'Random':>8s} {'Wins 1st':>9s} {'Loss 1st':>9s} | {'Verdict':>10s}"
    print(header3)
    print("-" * len(header3))

    for instr, base_con, pv in test_configs:
        pd_results = test_path_dependency(trade_pool, day_count_dist, pv, base_con, rng)
        spread = max(pd_results.values()) - min(pd_results.values())
        dep = "YES" if spread > 10 else "MILD" if spread > 3 else "NO"
        print(
            f"{instr:>5s} {base_con:>3d} | "
            f"{pd_results['random']:7.1f}% {pd_results['wins_first']:8.1f}% {pd_results['losses_first']:8.1f}% | "
            f"{dep:>10s}",
            flush=True,
        )

    print()
    print("Path dependency: NO = <3% spread, MILD = 3-10%, YES = >10%")
    print("$/Blow = (avg_payouts * $3k_target) - $2k_DD = net $ earned per account lifecycle")


if __name__ == "__main__":
    main()

"""
Monte Carlo prop firm comparison across top positive R:R configs.
Tests challenge pass rate and funded payouts for MNQ 2-3 and NQ 1.
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
POINT_VALUE = 20.0

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"

ACCOUNT_SIZE   = 50_000
TRAIL_DD       = 2_000
PROFIT_TARGET  = 3_000
FUNDED_BUFFER  = 2_000

N_SIMS = 5_000
MAX_DAYS_CHALLENGE = 500
MAX_DAYS_FUNDED = 1500
SEED = 42

MNQ_PV = 2.0
NQ_PV  = 20.0

# Top positive R:R configs to test
CONFIGS = [
    (0.50, 0.75),
    (0.50, 1.00),
    (0.50, 1.50),
    (0.50, 1.75),
    (0.50, 2.00),
    (0.75, 1.00),
    (0.75, 1.50),
    (0.75, 1.75),
    (0.75, 2.00),
    (1.00, 1.25),
    (1.00, 1.50),
    (1.00, 1.75),
    (1.00, 2.00),  # current default
    (1.25, 1.75),
    (1.25, 2.00),
]


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


def collect_trades_for_config(days, sl_mult, tp_mult):
    """Run backtest, return (trade_pnl_array, mae_array, day_count_array).

    mae_array stores the max adverse excursion per trade in points (always >= 0).
    This is the worst unrealized loss the trade experienced before resolving.
    """
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

            if d == "long":
                sl = ep - atr * sl_mult; tp = ep + atr * tp_mult
                if tp <= ep or sl >= ep: continue
            else:
                sl = ep + atr * sl_mult; tp = ep - atr * tp_mult
                if tp >= ep or sl <= ep: continue

            exp = None; ext = None
            worst_adverse = 0.0  # max adverse excursion in points (positive = loss)
            for j in range(cbi + 1, len(bars)):
                bar = bars[j]
                if not bar.closed: continue
                # Track worst unrealized loss during this bar
                if d == "long":
                    adverse = ep - bar.low  # how far below entry
                else:
                    adverse = bar.high - ep  # how far above entry
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
    for d in sorted(all_dates):
        if d in daily_pnls:
            day_counts.append(len(daily_pnls[d]))
            all_pnl.extend(daily_pnls[d])
            all_mae.extend(daily_maes[d])
        else:
            day_counts.append(0)

    return (np.array(all_pnl, dtype=np.float64),
            np.array(all_mae, dtype=np.float64),
            np.array(day_counts, dtype=np.int32))


def sim_challenge(trade_pool, mae_pool, day_count_dist, pv, contracts, consistency, rng):
    trades_per_day = rng.choice(day_count_dist, size=(N_SIMS, MAX_DAYS_CHALLENGE))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(trade_pool), size=total_needed)
    all_pnls = trade_pool[idx]
    all_maes = mae_pool[idx]

    passes = 0
    pass_days = []
    cursor = 0

    for sim in range(N_SIMS):
        equity = float(ACCOUNT_SIZE)
        hwm = float(ACCOUNT_SIZE)
        dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
        max_day_pnl = 0.0
        passed = False
        day = 0

        for di in range(MAX_DAYS_CHALLENGE):
            nt = int(trades_per_day[sim, di])
            if nt == 0:
                day += 1
                continue

            day_pnl = 0.0
            blown = False
            for _ in range(nt):
                trade_pnl = all_pnls[cursor] * pv * contracts
                trade_mae = all_maes[cursor] * pv * contracts
                cursor += 1
                # Check unrealized DD — account blows if equity - MAE breaches DD level
                if equity - trade_mae <= dd_level:
                    blown = True; break
                equity += trade_pnl
                day_pnl += trade_pnl
                if equity <= dd_level:
                    blown = True; break

            if blown:
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break

            day += 1
            if equity > hwm:
                hwm = equity; dd_level = hwm - TRAIL_DD
            if equity <= dd_level:
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break

            if day_pnl > max_day_pnl:
                max_day_pnl = day_pnl

            total_profit = equity - ACCOUNT_SIZE
            if total_profit >= PROFIT_TARGET:
                ok = True
                if consistency is not None and max_day_pnl > consistency * total_profit:
                    ok = False
                if ok:
                    passed = True
                    for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                        cursor += int(trades_per_day[sim, rd])
                    break

        if passed:
            passes += 1
            pass_days.append(day)

    return passes / N_SIMS * 100, (np.mean(pass_days) if pass_days else 0)


def sim_funded(trade_pool, mae_pool, day_count_dist, pv, contracts, consistency, rng):
    trades_per_day = rng.choice(day_count_dist, size=(N_SIMS, MAX_DAYS_FUNDED))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(trade_pool), size=total_needed)
    all_pnls = trade_pool[idx]
    all_maes = mae_pool[idx]

    all_payouts = []
    all_dpp = []
    cursor = 0

    for sim in range(N_SIMS):
        equity = float(ACCOUNT_SIZE)
        hwm = float(ACCOUNT_SIZE)
        dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
        dd_locked = False
        payouts = 0
        cycle_start = 0
        max_day_pnl = 0.0

        for di in range(MAX_DAYS_FUNDED):
            nt = int(trades_per_day[sim, di])
            if nt == 0:
                continue

            day_pnl = 0.0
            blown = False
            for _ in range(nt):
                trade_pnl = all_pnls[cursor] * pv * contracts
                trade_mae = all_maes[cursor] * pv * contracts
                cursor += 1
                # Check unrealized DD — account blows if equity - MAE breaches DD level
                if equity - trade_mae <= dd_level:
                    blown = True; break
                equity += trade_pnl
                day_pnl += trade_pnl
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
                for rd in range(di + 1, MAX_DAYS_FUNDED):
                    cursor += int(trades_per_day[sim, rd])
                break

            if day_pnl > max_day_pnl:
                max_day_pnl = day_pnl

            buffer_eq = ACCOUNT_SIZE + FUNDED_BUFFER
            profit = equity - buffer_eq
            if profit >= PROFIT_TARGET:
                can = True
                if consistency is not None and max_day_pnl > consistency * profit:
                    can = False
                if can:
                    payouts += 1
                    all_dpp.append(di + 1 - cycle_start)
                    equity = buffer_eq
                    hwm = buffer_eq
                    max_day_pnl = 0.0
                    cycle_start = di + 1

        all_payouts.append(payouts)

    return np.mean(all_payouts), (np.mean(all_dpp) if all_dpp else 0)


def main():
    print("Loading data...", flush=True)
    days = load_days()
    print(f"  {len(days)} trading days loaded\n", flush=True)

    rng = np.random.default_rng(SEED)

    # Contract configs to test
    risk_configs = [
        ("MNQ 2", 2, MNQ_PV),
        ("MNQ 3", 3, MNQ_PV),
        ("NQ 1",  1, NQ_PV),
    ]

    # Consistency: none only for speed, then show 40% for funded
    print("=" * 140)
    print("CHALLENGE PASS RATE & DAYS (no consistency)")
    print(f"Account: ${ACCOUNT_SIZE:,} | DD: ${TRAIL_DD:,} | Target: ${PROFIT_TARGET:,} | {N_SIMS} sims")
    print("=" * 140)

    header = f"{'SL':>5s} {'TP':>5s} {'R:R':>5s} {'Trades':>6s} {'WR%':>5s} {'PF':>5s} {'Exp':>7s} |"
    for label, _, _ in risk_configs:
        header += f" {label+' Pass':>10s} {'Days':>5s} |"
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    config_results = {}

    for sl, tp in CONFIGS:
        trade_pool, mae_pool, day_counts = collect_trades_for_config(days, sl, tp)

        if len(trade_pool) < 30:
            continue

        arr = trade_pool
        n = len(arr)
        w = arr[arr > 0]; l = arr[arr < 0]
        wr = len(w) / n * 100
        gp = w.sum() if len(w) else 0
        gl = abs(l.sum()) if len(l) else 1
        pf = gp / gl if gl > 0 else 999
        exp = arr.mean()
        rr = tp / sl

        line = f"{sl:5.2f} {tp:5.2f} {rr:5.2f} {n:6d} {wr:4.1f}% {pf:5.2f} {exp:+7.2f} |"

        for label, contracts, pv in risk_configs:
            rate, avg_days = sim_challenge(trade_pool, mae_pool, day_counts, pv, contracts, None, rng)
            line += f" {rate:9.1f}% {avg_days:5.1f} |"

        print(line, flush=True)
        config_results[(sl, tp)] = (trade_pool, mae_pool, day_counts)

    # Funded phase
    print()
    print("=" * 140)
    print("FUNDED: AVG PAYOUTS & DAYS/PAYOUT (no consistency)")
    print(f"DD stops at ${ACCOUNT_SIZE:,} | Buffer: ${FUNDED_BUFFER:,} | Payout: ${PROFIT_TARGET:,}")
    print("=" * 140)

    header2 = f"{'SL':>5s} {'TP':>5s} {'R:R':>5s} |"
    for label, _, _ in risk_configs:
        header2 += f" {label+' Pay':>9s} {'D/P':>5s} {'$/Life':>8s} |"
    print(header2)
    print("-" * len(header2))
    sys.stdout.flush()

    for sl, tp in CONFIGS:
        if (sl, tp) not in config_results:
            continue
        trade_pool, mae_pool, day_counts = config_results[(sl, tp)]
        rr = tp / sl

        line = f"{sl:5.2f} {tp:5.2f} {rr:5.2f} |"

        for label, contracts, pv in risk_configs:
            avg_pay, avg_dpp = sim_funded(trade_pool, mae_pool, day_counts, pv, contracts, None, rng)
            net = avg_pay * PROFIT_TARGET - TRAIL_DD
            line += f" {avg_pay:9.1f} {avg_dpp:5.1f} {net:+8.0f} |"

        print(line, flush=True)

    # 40% consistency funded
    print()
    print("=" * 140)
    print("FUNDED: AVG PAYOUTS & DAYS/PAYOUT (40% consistency)")
    print("=" * 140)

    header3 = f"{'SL':>5s} {'TP':>5s} {'R:R':>5s} |"
    for label, _, _ in risk_configs:
        header3 += f" {label+' Pay':>9s} {'D/P':>5s} {'$/Life':>8s} |"
    print(header3)
    print("-" * len(header3))
    sys.stdout.flush()

    for sl, tp in CONFIGS:
        if (sl, tp) not in config_results:
            continue
        trade_pool, mae_pool, day_counts = config_results[(sl, tp)]
        rr = tp / sl

        line = f"{sl:5.2f} {tp:5.2f} {rr:5.2f} |"

        for label, contracts, pv in risk_configs:
            avg_pay, avg_dpp = sim_funded(trade_pool, mae_pool, day_counts, pv, contracts, 0.40, rng)
            net = avg_pay * PROFIT_TARGET - TRAIL_DD
            line += f" {avg_pay:9.1f} {avg_dpp:5.1f} {net:+8.0f} |"

        print(line, flush=True)


if __name__ == "__main__":
    main()

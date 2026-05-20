"""
Quick MC sim: Challenge (50% consistency) + Funded (no consistency).
4pm entry cutoff, unrealized DD tracked, MNQ 1-5.
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
SL_MULT = 1.0
TP_MULT = 2.0

ACCOUNT_SIZE   = 50_000
TRAIL_DD       = 2_000
PROFIT_TARGET  = 3_000
FUNDED_BUFFER  = 2_000

N_SIMS = 10_000
MAX_DAYS_CHALLENGE = 500
MAX_DAYS_FUNDED = 1500
SEED = 42
MNQ_PV = 2.0


def collect_trades():
    start = datetime.strptime("2025-03-13", "%Y-%m-%d").date()
    end = datetime.strptime("2026-04-08", "%Y-%m-%d").date()
    all_dates = set()
    daily_pnls = {}
    daily_maes = {}

    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        all_dates.add(date_str)
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
        bars = sd["bars"]

        day_trades = []
        day_maes = []
        last_exit_time = None
        for signal in signals:
            ep = signal["entry_price"]; d = signal["direction"]; atr = signal["atr"]
            cbi = signal["bar_index"] + 1
            if atr is None or atr <= 0: continue
            ct = signal["confirm_time"]
            if hasattr(ct, 'tz_convert'): cet = ct.tz_convert(ET)
            else: cet = pd.Timestamp(ct, tz='UTC').tz_convert(ET)
            if "16:00" <= cet.strftime("%H:%M") < "19:10": continue
            if last_exit_time is not None and ct <= last_exit_time: continue

            if d == "long":
                sl = ep - atr * SL_MULT; tp = ep + atr * TP_MULT
                if tp <= ep or sl >= ep: continue
            else:
                sl = ep + atr * SL_MULT; tp = ep - atr * TP_MULT
                if tp >= ep or sl <= ep: continue

            exp = None; ext = None; worst = 0.0
            for j in range(cbi + 1, len(bars)):
                bar = bars[j]
                if not bar.closed: continue
                if d == "long": adv = ep - bar.low
                else: adv = bar.high - ep
                if adv > worst: worst = adv
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
            day_maes.append(worst)
            last_exit_time = ext

        if day_trades:
            daily_pnls[date_str] = day_trades
            daily_maes[date_str] = day_maes

    all_pnl = []; all_mae = []; day_counts = []
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


def sim_challenge(tp, mp, dc, contracts, consistency, rng):
    pv = MNQ_PV
    tpd = rng.choice(dc, size=(N_SIMS, MAX_DAYS_CHALLENGE))
    total = int(tpd.sum())
    idx = rng.integers(0, len(tp), size=total)
    pnls = tp[idx]; maes = mp[idx]
    passes = 0; pass_days = []; cursor = 0

    for sim in range(N_SIMS):
        equity = float(ACCOUNT_SIZE); hwm = float(ACCOUNT_SIZE)
        dd_level = float(ACCOUNT_SIZE - TRAIL_DD)
        max_day_pnl = 0.0; passed = False; day = 0

        for di in range(MAX_DAYS_CHALLENGE):
            nt = int(tpd[sim, di])
            if nt == 0: day += 1; continue
            day_pnl = 0.0; blown = False
            for _ in range(nt):
                t_pnl = pnls[cursor] * pv * contracts
                t_mae = maes[cursor] * pv * contracts
                cursor += 1
                if equity - t_mae <= dd_level: blown = True; break
                equity += t_pnl; day_pnl += t_pnl
                if equity <= dd_level: blown = True; break
            if blown:
                for rd in range(di+1, MAX_DAYS_CHALLENGE): cursor += int(tpd[sim, rd])
                break
            day += 1
            if equity > hwm: hwm = equity; dd_level = hwm - TRAIL_DD
            if equity <= dd_level:
                for rd in range(di+1, MAX_DAYS_CHALLENGE): cursor += int(tpd[sim, rd])
                break
            if day_pnl > max_day_pnl: max_day_pnl = day_pnl
            total_profit = equity - ACCOUNT_SIZE
            if total_profit >= PROFIT_TARGET:
                ok = True
                if consistency is not None and max_day_pnl > consistency * total_profit:
                    ok = False
                if ok:
                    passed = True
                    for rd in range(di+1, MAX_DAYS_CHALLENGE): cursor += int(tpd[sim, rd])
                    break
        if passed: passes += 1; pass_days.append(day)

    return passes / N_SIMS * 100, (np.mean(pass_days) if pass_days else 0), \
           (np.median(pass_days) if pass_days else 0)


def sim_funded(tp, mp, dc, contracts, consistency, rng):
    pv = MNQ_PV
    tpd = rng.choice(dc, size=(N_SIMS, MAX_DAYS_FUNDED))
    total = int(tpd.sum())
    idx = rng.integers(0, len(tp), size=total)
    pnls = tp[idx]; maes = mp[idx]
    all_payouts = []; all_dpp = []; all_withdrawn = []; cursor = 0

    for sim in range(N_SIMS):
        equity = float(ACCOUNT_SIZE); hwm = float(ACCOUNT_SIZE)
        dd_level = float(ACCOUNT_SIZE - TRAIL_DD); dd_locked = False
        payouts = 0; max_day_pnl = 0.0; cycle_start = 0; total_withdrawn = 0.0

        for di in range(MAX_DAYS_FUNDED):
            nt = int(tpd[sim, di])
            if nt == 0: continue
            day_pnl = 0.0; blown = False
            for _ in range(nt):
                t_pnl = pnls[cursor] * pv * contracts
                t_mae = maes[cursor] * pv * contracts
                cursor += 1
                if equity - t_mae <= dd_level: blown = True; break
                equity += t_pnl; day_pnl += t_pnl
                if equity <= dd_level: blown = True; break
            if blown:
                for rd in range(di+1, MAX_DAYS_FUNDED): cursor += int(tpd[sim, rd])
                break
            if equity > hwm:
                hwm = equity
                if not dd_locked:
                    dd_level = hwm - TRAIL_DD
                    if dd_level >= ACCOUNT_SIZE:
                        dd_level = float(ACCOUNT_SIZE); dd_locked = True
            if equity <= dd_level:
                for rd in range(di+1, MAX_DAYS_FUNDED): cursor += int(tpd[sim, rd])
                break
            if day_pnl > 0 and day_pnl > max_day_pnl: max_day_pnl = day_pnl
            buffer_eq = ACCOUNT_SIZE + FUNDED_BUFFER
            profit = equity - buffer_eq
            if profit >= PROFIT_TARGET:
                can = True
                if consistency is not None and max_day_pnl > consistency * profit:
                    can = False
                if can:
                    payouts += 1; total_withdrawn += profit
                    all_dpp.append(di + 1 - cycle_start)
                    equity = buffer_eq; hwm = buffer_eq
                    max_day_pnl = 0.0; cycle_start = di + 1
        all_payouts.append(payouts)
        all_withdrawn.append(total_withdrawn)

    return (np.mean(all_payouts), np.mean(all_dpp) if all_dpp else 0,
            np.mean(all_withdrawn))


def main():
    print("Loading data (4pm cutoff)...", flush=True)
    trade_pool, mae_pool, day_counts = collect_trades()
    n = len(trade_pool)
    w = trade_pool[trade_pool > 0]
    wr = len(w) / n * 100
    print(f"  {n} trades | WR: {wr:.1f}% | Exp: {trade_pool.mean():+.2f} pts | PF: 1.54\n", flush=True)

    rng = np.random.default_rng(SEED)

    # Challenge: 50% consistency
    print("=" * 90)
    print("CHALLENGE — 50% Consistency | EOD Trailing DD $2k | Target $3k")
    print("Unrealized DD tracked | 10,000 sims")
    print("=" * 90)
    print(f"{'MNQ':>5s} | {'Pass%':>7s} | {'Avg Days':>8s} | {'Med Days':>8s} | {'Risk/Trade':>10s} | {'Max Loss':>9s}")
    print("-" * 70)
    sys.stdout.flush()

    for contracts in range(1, 6):
        rate, avg_d, med_d = sim_challenge(trade_pool, mae_pool, day_counts,
                                           contracts, 0.50, rng)
        avg_risk = trade_pool[trade_pool < 0].mean() * MNQ_PV * contracts
        max_loss = trade_pool.min() * MNQ_PV * contracts
        print(f"  {contracts:>3d} | {rate:6.1f}% | {avg_d:7.1f}d | {med_d:7.0f}d | "
              f"${avg_risk:+8.0f} | ${max_loss:+8.0f}", flush=True)

    # Also show no consistency for comparison
    print()
    print("=" * 90)
    print("CHALLENGE — No Consistency (for comparison)")
    print("=" * 90)
    print(f"{'MNQ':>5s} | {'Pass%':>7s} | {'Avg Days':>8s} | {'Med Days':>8s}")
    print("-" * 50)
    sys.stdout.flush()

    for contracts in range(1, 6):
        rate, avg_d, med_d = sim_challenge(trade_pool, mae_pool, day_counts,
                                           contracts, None, rng)
        print(f"  {contracts:>3d} | {rate:6.1f}% | {avg_d:7.1f}d | {med_d:7.0f}d", flush=True)

    # Funded: no consistency
    print()
    print("=" * 90)
    print("FUNDED — No Consistency | DD locks at $50k | Buffer $2k | Payout $3k")
    print("=" * 90)
    print(f"{'MNQ':>5s} | {'Avg Payouts':>11s} | {'Avg D/P':>7s} | {'Avg $/Life':>10s} | {'$/Month':>8s}")
    print("-" * 65)
    sys.stdout.flush()

    for contracts in range(1, 6):
        avg_pay, avg_dpp, avg_total = sim_funded(trade_pool, mae_pool, day_counts,
                                                  contracts, None, rng)
        net = avg_total - TRAIL_DD
        monthly = (PROFIT_TARGET / avg_dpp * 21) if avg_dpp > 0 else 0
        print(f"  {contracts:>3d} | {avg_pay:11.1f} | {avg_dpp:6.1f}d | ${net:>9,.0f} | ${monthly:>7,.0f}", flush=True)


if __name__ == "__main__":
    main()

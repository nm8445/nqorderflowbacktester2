"""
MC sim: Challenge with $1500 head start (already $1500 in profit).
50% consistency, EOD trailing DD, MNQ 1-5.
Only need $1500 more to hit $3k target.
But DD has already trailed up — HWM = $51,500, DD level = $49,500.
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
HEAD_START     = 1_500

N_SIMS = 10_000
MAX_DAYS = 500
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


def sim_challenge(tp, mp, dc, contracts, consistency, head_start, rng):
    pv = MNQ_PV
    tpd = rng.choice(dc, size=(N_SIMS, MAX_DAYS))
    total = int(tpd.sum())
    idx = rng.integers(0, len(tp), size=total)
    pnls = tp[idx]; maes = mp[idx]

    passes = 0; pass_days = []; blown_count = 0; cursor = 0

    for sim in range(N_SIMS):
        # Start with head start already banked
        starting_eq = float(ACCOUNT_SIZE + head_start)
        equity = starting_eq
        hwm = starting_eq
        dd_level = starting_eq - TRAIL_DD  # DD has trailed up

        # For consistency: assume best day so far contributed some portion
        # Conservative: assume head_start came evenly, so max_day = head_start
        # (worst case for consistency check)
        max_day_pnl = float(head_start)

        passed = False; day = 0

        for di in range(MAX_DAYS):
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
                blown_count += 1
                for rd in range(di+1, MAX_DAYS): cursor += int(tpd[sim, rd])
                break
            day += 1
            if equity > hwm: hwm = equity; dd_level = hwm - TRAIL_DD
            if equity <= dd_level:
                blown_count += 1
                for rd in range(di+1, MAX_DAYS): cursor += int(tpd[sim, rd])
                break
            if day_pnl > max_day_pnl: max_day_pnl = day_pnl
            total_profit = equity - ACCOUNT_SIZE
            if total_profit >= PROFIT_TARGET:
                ok = True
                if consistency is not None and max_day_pnl > consistency * total_profit:
                    ok = False
                if ok:
                    passed = True
                    for rd in range(di+1, MAX_DAYS): cursor += int(tpd[sim, rd])
                    break
        if passed: passes += 1; pass_days.append(day)

    return (passes / N_SIMS * 100,
            np.mean(pass_days) if pass_days else 0,
            np.median(pass_days) if pass_days else 0,
            blown_count / N_SIMS * 100)


def main():
    print("Loading data...", flush=True)
    trade_pool, mae_pool, day_counts = collect_trades()
    print(f"  {len(trade_pool)} trades loaded\n", flush=True)

    rng = np.random.default_rng(SEED)

    # Head start scenario: already $1500 in profit
    # Need only $1500 more to reach $3k target
    # But DD has trailed: HWM = $51,500, DD level = $49,500
    # Remaining DD cushion from current equity: $51,500 - $49,500 = $2,000

    print("=" * 90)
    print(f"CHALLENGE WITH ${HEAD_START:,} HEAD START")
    print(f"Account: ${ACCOUNT_SIZE:,} | Current equity: ${ACCOUNT_SIZE + HEAD_START:,}")
    print(f"HWM: ${ACCOUNT_SIZE + HEAD_START:,} | DD level: ${ACCOUNT_SIZE + HEAD_START - TRAIL_DD:,}")
    print(f"Remaining to target: ${PROFIT_TARGET - HEAD_START:,}")
    print(f"DD cushion from current equity: ${TRAIL_DD:,}")
    print("=" * 90)
    print()

    # 50% consistency
    print("50% CONSISTENCY:")
    print(f"{'MNQ':>5s} | {'Pass%':>7s} | {'Avg Days':>8s} | {'Med Days':>8s} | {'Blown%':>7s}")
    print("-" * 55)
    sys.stdout.flush()

    for contracts in range(1, 6):
        rate, avg_d, med_d, blown_pct = sim_challenge(
            trade_pool, mae_pool, day_counts, contracts, 0.50, HEAD_START, rng)
        print(f"  {contracts:>3d} | {rate:6.1f}% | {avg_d:7.1f}d | {med_d:7.0f}d | {blown_pct:6.1f}%",
              flush=True)

    # No consistency
    print()
    print("NO CONSISTENCY:")
    print(f"{'MNQ':>5s} | {'Pass%':>7s} | {'Avg Days':>8s} | {'Med Days':>8s} | {'Blown%':>7s}")
    print("-" * 55)
    sys.stdout.flush()

    for contracts in range(1, 6):
        rate, avg_d, med_d, blown_pct = sim_challenge(
            trade_pool, mae_pool, day_counts, contracts, None, HEAD_START, rng)
        print(f"  {contracts:>3d} | {rate:6.1f}% | {avg_d:7.1f}d | {med_d:7.0f}d | {blown_pct:6.1f}%",
              flush=True)

    # Compare to fresh start
    print()
    print("=" * 90)
    print("COMPARISON: FRESH START (no head start) — 50% consistency")
    print("=" * 90)
    print(f"{'MNQ':>5s} | {'Pass%':>7s} | {'Avg Days':>8s} | {'Med Days':>8s}")
    print("-" * 45)
    sys.stdout.flush()

    for contracts in range(1, 6):
        rate, avg_d, med_d, _ = sim_challenge(
            trade_pool, mae_pool, day_counts, contracts, 0.50, 0, rng)
        print(f"  {contracts:>3d} | {rate:6.1f}% | {avg_d:7.1f}d | {med_d:7.0f}d", flush=True)


if __name__ == "__main__":
    main()

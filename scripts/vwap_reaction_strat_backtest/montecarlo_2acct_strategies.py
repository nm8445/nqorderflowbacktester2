"""
Monte Carlo: 2 x $50k Apex accounts — best strategy to pass 1 fast.

Tests 3 approaches:
1. MIRROR   — both accounts take every trade (same exposure)
2. ALTERNATE — A gets odd trades, B gets even (split exposure)
3. ROTATE   — A takes trades until daily-locked/blown, then B takes over

All with 50% consistency rule enforced.
MNQ 1-10.

Usage:
    python -u scripts/vwap_reaction_strat_backtest/montecarlo_2acct_strategies.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from montecarlo_apex_full import (
    collect_combined_trades, build_adx_lookup,
    MNQ_PV, ACCOUNT_SIZE, TRAIL_DD, PROFIT_TARGET, DAILY_LOSS_LIM,
    CONSISTENCY_RULE, ACCOUNT_COST, N_SIMS, MAX_DAYS_CHALLENGE, SEED,
)

N = N_SIMS


def sim_mirror(pnl_pool, mae_pool, day_counts, contracts, rng):
    """Both accounts take every trade simultaneously."""
    trades_per_day = rng.choice(day_counts, size=(N, MAX_DAYS_CHALLENGE))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(pnl_pool), size=total_needed)
    pnls = pnl_pool[idx] * MNQ_PV * contracts
    maes = mae_pool[idx] * MNQ_PV * contracts

    passes = 0
    pass_days = []
    cursor = 0

    for sim in range(N):
        eq = [float(ACCOUNT_SIZE), float(ACCOUNT_SIZE)]
        hwm = [float(ACCOUNT_SIZE), float(ACCOUNT_SIZE)]
        dd = [float(ACCOUNT_SIZE - TRAIL_DD), float(ACCOUNT_SIZE - TRAIL_DD)]
        alive = [True, True]
        max_dp = [0.0, 0.0]
        passed = False

        for di in range(MAX_DAYS_CHALLENGE):
            nt = int(trades_per_day[sim, di])
            if nt == 0:
                continue
            day_pnl = [0.0, 0.0]
            for _ in range(nt):
                tp = pnls[cursor]
                tm = maes[cursor]
                cursor += 1
                for a in range(2):
                    if not alive[a]:
                        continue
                    if eq[a] - tm <= dd[a]:
                        alive[a] = False
                        continue
                    eq[a] += tp
                    day_pnl[a] += tp
                    if eq[a] <= dd[a]:
                        alive[a] = False

            for a in range(2):
                if not alive[a]:
                    continue
                if day_pnl[a] > max_dp[a]:
                    max_dp[a] = day_pnl[a]
                if eq[a] > hwm[a]:
                    hwm[a] = eq[a]
                    dd[a] = hwm[a] - TRAIL_DD
                if eq[a] <= dd[a]:
                    alive[a] = False
                    continue
                profit = eq[a] - ACCOUNT_SIZE
                if profit >= PROFIT_TARGET:
                    if profit > 0 and max_dp[a] / profit <= CONSISTENCY_RULE:
                        passed = True

            if passed:
                pass_days.append(di + 1)
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break
            if not any(alive):
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break

        if passed:
            passes += 1

    return passes / N * 100, np.mean(pass_days) if pass_days else 0, np.median(pass_days) if pass_days else 0


def sim_alternate(pnl_pool, mae_pool, day_counts, contracts, rng):
    """A gets odd trades, B gets even trades."""
    trades_per_day = rng.choice(day_counts, size=(N, MAX_DAYS_CHALLENGE))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(pnl_pool), size=total_needed)
    pnls = pnl_pool[idx] * MNQ_PV * contracts
    maes = mae_pool[idx] * MNQ_PV * contracts

    passes = 0
    pass_days = []
    cursor = 0

    for sim in range(N):
        eq = [float(ACCOUNT_SIZE), float(ACCOUNT_SIZE)]
        hwm = [float(ACCOUNT_SIZE), float(ACCOUNT_SIZE)]
        dd = [float(ACCOUNT_SIZE - TRAIL_DD), float(ACCOUNT_SIZE - TRAIL_DD)]
        alive = [True, True]
        max_dp = [0.0, 0.0]
        passed = False
        trade_count = 0

        for di in range(MAX_DAYS_CHALLENGE):
            nt = int(trades_per_day[sim, di])
            if nt == 0:
                continue
            day_pnl = [0.0, 0.0]
            for _ in range(nt):
                tp = pnls[cursor]
                tm = maes[cursor]
                cursor += 1
                # Alternate between accounts
                a = trade_count % 2
                trade_count += 1
                if not alive[a]:
                    a = 1 - a  # try the other
                    if not alive[a]:
                        continue
                if eq[a] - tm <= dd[a]:
                    alive[a] = False
                    continue
                eq[a] += tp
                day_pnl[a] += tp
                if eq[a] <= dd[a]:
                    alive[a] = False

            for a in range(2):
                if not alive[a]:
                    continue
                if day_pnl[a] > max_dp[a]:
                    max_dp[a] = day_pnl[a]
                if eq[a] > hwm[a]:
                    hwm[a] = eq[a]
                    dd[a] = hwm[a] - TRAIL_DD
                if eq[a] <= dd[a]:
                    alive[a] = False
                    continue
                profit = eq[a] - ACCOUNT_SIZE
                if profit >= PROFIT_TARGET:
                    if profit > 0 and max_dp[a] / profit <= CONSISTENCY_RULE:
                        passed = True

            if passed:
                pass_days.append(di + 1)
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break
            if not any(alive):
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break

        if passed:
            passes += 1

    return passes / N * 100, np.mean(pass_days) if pass_days else 0, np.median(pass_days) if pass_days else 0


def sim_rotate(pnl_pool, mae_pool, day_counts, contracts, rng):
    """A takes trades until daily-locked/blown, then B takes over."""
    trades_per_day = rng.choice(day_counts, size=(N, MAX_DAYS_CHALLENGE))
    total_needed = int(trades_per_day.sum())
    idx = rng.integers(0, len(pnl_pool), size=total_needed)
    pnls = pnl_pool[idx] * MNQ_PV * contracts
    maes = mae_pool[idx] * MNQ_PV * contracts

    passes = 0
    pass_days = []
    cursor = 0

    for sim in range(N):
        eq = [float(ACCOUNT_SIZE), float(ACCOUNT_SIZE)]
        hwm = [float(ACCOUNT_SIZE), float(ACCOUNT_SIZE)]
        dd = [float(ACCOUNT_SIZE - TRAIL_DD), float(ACCOUNT_SIZE - TRAIL_DD)]
        alive = [True, True]
        max_dp = [0.0, 0.0]
        daily_loss = [0.0, 0.0]
        daily_locked = [False, False]
        passed = False

        for di in range(MAX_DAYS_CHALLENGE):
            daily_loss[0] = 0.0; daily_loss[1] = 0.0
            daily_locked[0] = False; daily_locked[1] = False

            nt = int(trades_per_day[sim, di])
            if nt == 0:
                continue
            day_pnl = [0.0, 0.0]

            for _ in range(nt):
                tp = pnls[cursor]
                tm = maes[cursor]
                cursor += 1
                # Pick first available
                a = -1
                for x in range(2):
                    if alive[x] and not daily_locked[x]:
                        a = x
                        break
                if a == -1:
                    break

                if eq[a] - tm <= dd[a]:
                    alive[a] = False
                    continue

                if daily_loss[a] + tm >= DAILY_LOSS_LIM:
                    daily_locked[a] = True
                    eq[a] += tp
                    day_pnl[a] += tp
                    if tp < 0:
                        daily_loss[a] += abs(tp)
                    if eq[a] <= dd[a]:
                        alive[a] = False
                    elif eq[a] > hwm[a]:
                        hwm[a] = eq[a]
                        dd[a] = hwm[a] - TRAIL_DD
                    continue

                eq[a] += tp
                day_pnl[a] += tp
                if tp < 0:
                    daily_loss[a] += abs(tp)
                if eq[a] <= dd[a]:
                    alive[a] = False
                    continue
                if eq[a] > hwm[a]:
                    hwm[a] = eq[a]
                    dd[a] = hwm[a] - TRAIL_DD
                if daily_loss[a] >= DAILY_LOSS_LIM:
                    daily_locked[a] = True

            for a in range(2):
                if not alive[a]:
                    continue
                if day_pnl[a] > max_dp[a]:
                    max_dp[a] = day_pnl[a]
                profit = eq[a] - ACCOUNT_SIZE
                if profit >= PROFIT_TARGET:
                    if profit > 0 and max_dp[a] / profit <= CONSISTENCY_RULE:
                        passed = True

            if passed:
                pass_days.append(di + 1)
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break
            if not any(alive):
                for rd in range(di + 1, MAX_DAYS_CHALLENGE):
                    cursor += int(trades_per_day[sim, rd])
                break

        if passed:
            passes += 1

    return passes / N * 100, np.mean(pass_days) if pass_days else 0, np.median(pass_days) if pass_days else 0


def main():
    print("Building ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done")
    print("Collecting trades...", end=" ", flush=True)
    pnl_pool, mae_pool, sl_pool, day_counts = collect_combined_trades(adx_lookup)
    print(f"done ({len(pnl_pool)} trades)")

    print()
    print("=" * 110)
    print("  2 ACCOUNTS x $50k APEX — FASTEST WAY TO PASS 1 (50% consistency rule)")
    print(f"  Cost: 2 x ${ACCOUNT_COST} = ${2 * ACCOUNT_COST}")
    print("=" * 110)

    strategies = [
        ("MIRROR (both take every trade)", sim_mirror),
        ("ALTERNATE (A=odd, B=even trades)", sim_alternate),
        ("ROTATE (A first, B when A locked/blown)", sim_rotate),
    ]

    best_results = {}

    for name, fn in strategies:
        print(f"\n  --- {name} ---")
        print(f"  {'MNQ':>4} {'Pass%':>7} {'AvgDays':>8} {'MedDays':>8}")
        print(f"  {'-' * 32}")
        sys.stdout.flush()
        strat_results = []
        for c in range(1, 11):
            rng = np.random.default_rng(SEED)
            rate, avg, med = fn(pnl_pool, mae_pool, day_counts, c, rng)
            print(f"  {c:>4} {rate:>6.1f}% {avg:>8.0f} {med:>8.0f}", flush=True)
            strat_results.append((c, rate, avg, med))
        best_results[name] = strat_results

    # Summary comparison
    print()
    print("=" * 110)
    print("  HEAD-TO-HEAD COMPARISON (best MNQ for speed with >= 90% pass rate)")
    print("=" * 110)
    for name, results in best_results.items():
        safe = [(c, r, a, m) for c, r, a, m in results if r >= 90]
        if safe:
            fastest = min(safe, key=lambda x: x[3])  # min median days
            print(f"  {name}")
            print(f"    Best: {fastest[0]} MNQ -> {fastest[1]:.1f}% pass, {fastest[3]:.0f} median days")
        else:
            # Relax to >= 80%
            safe80 = [(c, r, a, m) for c, r, a, m in results if r >= 80]
            if safe80:
                fastest = min(safe80, key=lambda x: x[3])
                print(f"  {name}")
                print(f"    Best (>= 80%): {fastest[0]} MNQ -> {fastest[1]:.1f}% pass, {fastest[3]:.0f} median days")
            else:
                best = max(results, key=lambda x: x[1])
                print(f"  {name}")
                print(f"    Best available: {best[0]} MNQ -> {best[1]:.1f}% pass, {best[3]:.0f} median days")

    # Also show: for each MNQ, which strategy wins?
    print()
    print("=" * 110)
    print("  WHICH STRATEGY WINS AT EACH MNQ SIZE?")
    print("=" * 110)
    print(f"  {'MNQ':>4}  {'Mirror':>14}  {'Alternate':>14}  {'Rotate':>14}  {'Winner':>20}")
    print(f"  {'-' * 70}")

    strat_names = list(best_results.keys())
    for i in range(10):
        c = i + 1
        rates = []
        meds = []
        for name in strat_names:
            _, r, a, m = best_results[name][i]
            rates.append(r)
            meds.append(m)

        labels = []
        for r, m in zip(rates, meds):
            labels.append(f"{r:.0f}%/{m:.0f}d")

        # Winner = highest pass rate, tiebreak by median days
        best_idx = max(range(3), key=lambda x: (rates[x], -meds[x]))
        short_names = ["Mirror", "Alternate", "Rotate"]
        winner = short_names[best_idx]

        print(f"  {c:>4}  {labels[0]:>14}  {labels[1]:>14}  {labels[2]:>14}  {winner:>20}")

    sys.stdout.flush()
    print("\nDone.")


if __name__ == "__main__":
    main()

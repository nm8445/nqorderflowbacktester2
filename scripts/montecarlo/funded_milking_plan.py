"""MC: User's full milking plan.

Phase 1: Lock $2K cushion at 1 NQ (or via hedge — compare both).
Phase 2: At 1 MNQ, grind to +$1K profit AND >=5 winning days, with $2K cushion.
Phase 3: Withdraw $1K. Balance back to $2K cushion. Repeat phase 2.
Continue until blow.

Key innovation: $1K withdrawal at $3K total profit -> always restore $2K buffer.
DD floor locked at starting balance after first $2K crossing -> doesn't trail down with withdrawals.

Compare cushion-build approaches:
  A. HEDGE (cross-firm 1:2 RR): 80% pair pass, 4 avg survivors of 10
  B. COPY-PAIR 1 NQ (correlated 2-at-a-time): 40.7% pair pass, 4 avg survivors but high correlation
  C. INDEPENDENT 1 NQ (each account separate): 40.7% per account, 4 avg survivors but uncorrelated
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import binom

TRADES_CSV = "live/combined deployment plan/combined_trades_with_mae.csv"
RNG = np.random.default_rng(42)
N_TRIALS = 5_000

# Phase 1 (cushion build)
PHASE1_TARGET = 2000
PHASE1_FLOOR  = -2000

# Phase 2 (milking)
MILK_TARGET_PROFIT = 1000   # +$1K profit per cycle
MILK_FLOOR         = -2000  # $2K cushion (account blows if drops below)
MILK_WIN_DAYS_MIN  = 5
WITHDRAWAL_PER_CYCLE = 1000


def load_trades(strats=None):
    df = pd.read_csv(TRADES_CSV)
    if strats is not None:
        df = df[df["strat"].isin(strats)]
    return df[["pnl_$", "mae_$"]].values, df.groupby("date")["pnl_$"].sum().values


def simulate_phase1(trades, max_trades=200):
    """Single account: trade at 1 NQ until +$2K lock or blow (MAE-aware)."""
    cum = 0.0
    for _ in range(max_trades):
        idx = RNG.integers(0, len(trades))
        pnl, mae = trades[idx]
        if cum + mae <= PHASE1_FLOOR:
            return False, cum
        cum += pnl
        if cum >= PHASE1_TARGET:
            return True, cum
        if cum <= PHASE1_FLOOR:
            return False, cum
    return False, cum


def simulate_milk_cycle(daily_pnl_mnq, max_days=120):
    """Single milking cycle at 1 MNQ: grind +$1K AND >=5 winning days, with $2K cushion floor.
    Returns (success, days_taken, max_drawdown_during_cycle, profit_achieved)."""
    cum = 0.0
    win_days = 0
    min_cum = 0.0
    for d in range(1, max_days + 1):
        daily = RNG.choice(daily_pnl_mnq)
        cum += daily
        if cum < min_cum:
            min_cum = cum
        if daily > 0:
            win_days += 1
        if cum <= MILK_FLOOR:
            return False, d, min_cum, cum  # blown
        if cum >= MILK_TARGET_PROFIT and win_days >= MILK_WIN_DAYS_MIN:
            return True, d, min_cum, cum  # success
    return False, max_days, min_cum, cum  # timed out


def simulate_account_lifetime(daily_pnl_mnq, max_cycles=50):
    """One funded account post-cushion-lock: keep milking until it blows."""
    total_extracted = 0
    total_days = 0
    cycles = 0
    for _ in range(max_cycles):
        ok, days, _, _ = simulate_milk_cycle(daily_pnl_mnq)
        total_days += days
        if not ok:
            return total_extracted, cycles, total_days
        total_extracted += WITHDRAWAL_PER_CYCLE
        cycles += 1
    return total_extracted, cycles, total_days


def main():
    trades, daily_pnl_nq = load_trades()
    daily_pnl_mnq = daily_pnl_nq / 10
    print(f"Loaded {len(trades)} trades, {len(daily_pnl_nq)} trading days")
    print(f"1 MNQ daily: mean=${daily_pnl_mnq.mean():.0f} std=${daily_pnl_mnq.std():.0f}")

    # === PHASE 1: cushion build comparison ===
    print("\n" + "=" * 90)
    print("PHASE 1: CUSHION BUILD ($2K lock)")
    print("=" * 90)

    # 1 NQ MAE-aware (per account)
    p1_results = [simulate_phase1(trades) for _ in range(N_TRIALS)]
    p_lock_1nq = sum(1 for ok, _ in p1_results if ok) / N_TRIALS
    print(f"\n  1 NQ MAE-aware per-account lock rate: {p_lock_1nq*100:.1f}%")

    # 10-account distributions
    p_hedge_pair = 0.80
    print(f"\n  {'Approach':<35} {'Avg survivors':>14} {'P(<=2)':>10} {'P(0 survive)':>14}")
    # Hedge: 5 pairs binom(5, 0.80) → survivors = 1 per locked pair
    h_avg = 5 * p_hedge_pair
    h_p2 = binom.cdf(2, 5, p_hedge_pair) * 100
    h_p0 = binom.pmf(0, 5, p_hedge_pair) * 100
    print(f"  {'A. HEDGE (5 pairs, 1:2 RR)':<35} {h_avg:>14.2f} {h_p2:>9.2f}% {h_p0:>13.3f}%")

    # Copy-pair 1 NQ: 5 pairs binom(5, p_lock), 2 accounts per locked pair → correlated
    cp_avg = 5 * p_lock_1nq * 2
    # P(<=2 survivors) = P(0 or 1 pair locks)
    cp_p2 = (binom.pmf(0, 5, p_lock_1nq) + binom.pmf(1, 5, p_lock_1nq)) * 100
    cp_p0 = binom.pmf(0, 5, p_lock_1nq) * 100
    print(f"  {'B. COPY-PAIR 1 NQ (correlated)':<35} {cp_avg:>14.2f} {cp_p2:>9.2f}% {cp_p0:>13.3f}%")

    # Independent 1 NQ: 10 accounts binom(10, p_lock)
    ind_avg = 10 * p_lock_1nq
    ind_p2 = binom.cdf(2, 10, p_lock_1nq) * 100
    ind_p0 = binom.pmf(0, 10, p_lock_1nq) * 100
    print(f"  {'C. INDEPENDENT 1 NQ (uncorrelated)':<35} {ind_avg:>14.2f} {ind_p2:>9.2f}% {ind_p0:>13.3f}%")

    # === PHASE 2: milking economics per surviving account ===
    print("\n" + "=" * 90)
    print("PHASE 2: MILKING (1 MNQ, +$1K target, 5 win days, $2K cushion floor)")
    print("=" * 90)

    # Per-cycle stats
    cycle_results = [simulate_milk_cycle(daily_pnl_mnq) for _ in range(N_TRIALS)]
    p_cycle_success = sum(1 for ok, _, _, _ in cycle_results if ok) / N_TRIALS
    days_to_success = [d for ok, d, _, _ in cycle_results if ok]
    drawdowns = [mc for _, _, mc, _ in cycle_results]
    print(f"\n  Per-milk-cycle stats:")
    print(f"    P(success per cycle): {p_cycle_success*100:.1f}%")
    print(f"    P(blow per cycle):    {(1-p_cycle_success)*100:.1f}%")
    print(f"    Median days to $1K + 5 wins: {np.median(days_to_success):.0f}")
    print(f"    Mean days: {np.mean(days_to_success):.1f}")
    print(f"    Worst DD during cycle: mean=${np.mean(drawdowns):.0f}  p5=${np.percentile(drawdowns, 5):.0f}")

    # Lifetime value
    lifetimes = [simulate_account_lifetime(daily_pnl_mnq) for _ in range(N_TRIALS)]
    extracted = [x[0] for x in lifetimes]
    n_cycles = [x[1] for x in lifetimes]
    n_days = [x[2] for x in lifetimes]
    print(f"\n  Per-cushion-locked-account lifetime:")
    print(f"    Avg total extracted: ${np.mean(extracted):,.0f}")
    print(f"    Median extracted:    ${np.median(extracted):,.0f}")
    print(f"    Avg cycles before blow: {np.mean(n_cycles):.1f}")
    print(f"    Avg trading days alive: {np.mean(n_days):.0f}  (={np.mean(n_days)/250:.1f} years)")
    print(f"    Distribution of $ extracted:")
    print(f"      p25=${np.percentile(extracted, 25):,.0f}  p50=${np.percentile(extracted, 50):,.0f}")
    print(f"      p75=${np.percentile(extracted, 75):,.0f}  p95=${np.percentile(extracted, 95):,.0f}")

    # === FULL CYCLE ECONOMICS ===
    print("\n" + "=" * 90)
    print("FULL CYCLE: 10 evals -> funded -> cushion -> milking lifetime")
    print("=" * 90)
    avg_lifetime_per_locked = np.mean(extracted)
    eval_cost_per_cycle = 20 * 100  # ~20 evals at $100 each
    print(f"  Eval cost per 10-funded-cycle: ${eval_cost_per_cycle:,.0f}")

    for label, avg_survivors in [("A. HEDGE", h_avg), ("B. COPY-PAIR 1 NQ", cp_avg), ("C. INDEPENDENT 1 NQ", ind_avg)]:
        total_extracted_per_cycle = avg_survivors * avg_lifetime_per_locked
        net = total_extracted_per_cycle - eval_cost_per_cycle
        print(f"  {label:<30} survivors={avg_survivors:.2f}  total_extracted=${total_extracted_per_cycle:>10,.0f}  net=${net:>10,.0f}")


if __name__ == "__main__":
    main()

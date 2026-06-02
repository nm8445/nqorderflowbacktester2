"""Month-by-month income progression for the milking + reinvest cycle.

User's plan (per their description):
1. Start with 1 funded account fresh ($50K, $2K DD)
2. Grind to +$3K profit (71% pass rate per user) without breaching DD
3. Withdraw $1.5K from $3K profit; balance $51.5K, cushion $1.5K... or
   withdraw $1K leaving $2K cushion (cleaner math, used in prior MC)
4. Spend $1K of payout on 10 new evals (~35% pass = 3.5 funded)
5. Hedge new funded accounts at $2K SL / $3K TP (88% pair pass) for cushion lock
6. New cushion-locked accounts join milking pool
7. Each milker grinds +$1K, withdraws $1K, restores $2K cushion, repeats
8. Per-cycle: 92.9% success, 7.1% blow (from prior MC)
9. Per-cycle duration: median 16 trading days

Sweep over months 1..24. Track:
- Active milker count
- Monthly income $
- Cumulative profit
- New milkers added per cycle
- Mortality per month
"""
from __future__ import annotations
import numpy as np

# Parameters from prior MCs and user spec
TRADING_DAYS_PER_MONTH   = 21
DAYS_PER_MILK_CYCLE      = 16    # median from prior MC
CYCLE_SUCCESS_PROB       = 0.929  # P(extract $1K per cycle) from prior MC
PER_CYCLE_PAYOUT         = 1000   # $ extracted per successful cycle
FIRST_CYCLE_PASS_PROB    = 0.71   # P(fresh account hits $3K before blow) from user

# Reinvestment math
EVAL_COST                = 100
EVALS_PER_BATCH          = 10
BATCH_COST               = EVALS_PER_BATCH * EVAL_COST   # $1000
EVAL_PASS_RATE           = 0.35   # gambler's ruin coinflip rough estimate
HEDGE_PAIR_PASS          = 0.88   # 1:1.5 RR hedge (2K SL / 3K TP)
SOLO_PASS                = 0.71   # solo account same as first cycle

# Cap from firm account limits
MAX_ACTIVE_MILKERS       = 25
PAYOUT_PER_BATCH_CASH    = 500    # withdraw $1.5K, $1K to evals, $500 to pocket

N_TRIALS                 = 5000

def simulate_one_path(n_months: int = 24) -> dict:
    rng = np.random.default_rng()
    active_milkers = 0     # cushion-locked, milking
    fresh_accounts = 1     # one fresh funded account at month 0
    monthly_income   = []
    monthly_active   = []
    monthly_new      = []
    monthly_blown    = []
    cumulative_cash  = 0
    cumulative_pocket = 0   # net cash to keep (after eval reinvest)

    for month in range(1, n_months + 1):
        income_this_month = 0
        new_this_month    = 0
        blown_this_month  = 0

        # === Step 1: process FRESH accounts (need to hit $3K first) ===
        # These accounts haven't locked cushion yet
        new_milkers_from_fresh = 0
        blown_fresh = 0
        for _ in range(fresh_accounts):
            if rng.random() < FIRST_CYCLE_PASS_PROB:
                new_milkers_from_fresh += 1
                # User extracts $1K immediately upon reaching $3K (becomes milker)
                income_this_month += PER_CYCLE_PAYOUT
            else:
                blown_fresh += 1
        active_milkers += new_milkers_from_fresh
        new_this_month  += new_milkers_from_fresh
        blown_this_month += blown_fresh
        fresh_accounts = 0   # all processed

        # === Step 2: each active milker runs cycles ===
        # Average cycles per month = trading_days_per_month / days_per_cycle
        avg_cycles_per_month = TRADING_DAYS_PER_MONTH / DAYS_PER_MILK_CYCLE
        # Each milker gets a Poisson-distributed number of cycle attempts
        for _ in range(active_milkers):
            n_cycles = rng.poisson(avg_cycles_per_month)
            n_cycles = max(0, n_cycles)
            survived = True
            for c in range(n_cycles):
                if rng.random() < CYCLE_SUCCESS_PROB:
                    income_this_month += PER_CYCLE_PAYOUT
                else:
                    survived = False
                    blown_this_month += 1
                    break
            if not survived:
                active_milkers -= 1   # remove blown milker

        # === Step 3: reinvest payouts into new evals ===
        # Smart logic: only buy enough to grow toward MAX or replace mortality
        room = MAX_ACTIVE_MILKERS - active_milkers
        # Each batch produces ~1.7 new locked accounts on average (10 * 0.35 * ~0.5 lock rate)
        avg_new_per_batch = EVALS_PER_BATCH * EVAL_PASS_RATE * (HEDGE_PAIR_PASS * 0.5 + SOLO_PASS * 0.5)
        # Want to fill the room + replace expected mortality next month (~7% × n_cycles × active)
        expected_mortality = active_milkers * (1 - CYCLE_SUCCESS_PROB) * avg_cycles_per_month
        target_new = room + expected_mortality
        batches_target = max(0, int(np.ceil(target_new / max(avg_new_per_batch, 0.1))))
        # Cap by available cash (each batch needs $1K)
        max_affordable = income_this_month // BATCH_COST
        batches_to_buy = int(min(batches_target, max_affordable))

        # Each batch: 10 evals * 35% pass = 3.5 new funded
        new_funded_from_batches = 0
        for _ in range(batches_to_buy):
            n_passed = rng.binomial(EVALS_PER_BATCH, EVAL_PASS_RATE)
            # Of those passed, hedge them in pairs
            n_pairs = n_passed // 2
            n_solo = n_passed % 2
            n_locked_pairs = rng.binomial(n_pairs, HEDGE_PAIR_PASS)
            n_locked_solo  = rng.binomial(n_solo, SOLO_PASS)
            n_locked = n_locked_pairs + n_locked_solo
            new_funded_from_batches += n_locked
        # Cap to firm limit
        room = MAX_ACTIVE_MILKERS - active_milkers
        new_funded_from_batches = min(new_funded_from_batches, room)
        active_milkers += new_funded_from_batches
        new_this_month  += new_funded_from_batches

        # Cash tracking
        eval_spend     = batches_to_buy * BATCH_COST
        pocket_cash    = batches_to_buy * PAYOUT_PER_BATCH_CASH
        net_income     = income_this_month - eval_spend
        cumulative_cash  += income_this_month
        cumulative_pocket += net_income

        monthly_income.append(net_income)
        monthly_active.append(active_milkers)
        monthly_new.append(new_this_month)
        monthly_blown.append(blown_this_month)

    return dict(
        monthly_income=monthly_income,
        monthly_active=monthly_active,
        monthly_new=monthly_new,
        monthly_blown=monthly_blown,
        final_cum=cumulative_pocket,
    )


def main():
    n_months = 24
    print("=" * 100)
    print("MILKING + REINVEST MONTHLY PROGRESSION (MC: 5000 paths)")
    print("=" * 100)
    print(f"Starting state: 1 fresh funded account")
    print(f"First-cycle pass prob: {FIRST_CYCLE_PASS_PROB:.0%}, milk cycle success: {CYCLE_SUCCESS_PROB:.1%}")
    print(f"Days per cycle: {DAYS_PER_MILK_CYCLE} (median), trading days/mo: {TRADING_DAYS_PER_MONTH}")
    print(f"Eval pass rate: {EVAL_PASS_RATE:.0%}, hedge pair pass: {HEDGE_PAIR_PASS:.0%}")
    print(f"Max active milkers cap: {MAX_ACTIVE_MILKERS}")
    print(f"$/batch reinvest: ${BATCH_COST} (10 evals), ${PAYOUT_PER_BATCH_CASH} kept as cash")

    results = [simulate_one_path(n_months) for _ in range(N_TRIALS)]
    income_matrix = np.array([r["monthly_income"] for r in results])    # shape (N_TRIALS, n_months)
    active_matrix = np.array([r["monthly_active"] for r in results])
    new_matrix    = np.array([r["monthly_new"]    for r in results])
    blown_matrix  = np.array([r["monthly_blown"]  for r in results])

    print(f"\n{'Mon':>4} | {'Mean income':>12} {'Median':>10} {'p10':>8} {'p90':>8} | {'Avg active':>11} {'Avg new':>8} {'Avg blown':>10} | {'Cum mean':>11}")
    print("-" * 120)
    cum_mean = 0
    cum_median = 0
    for m in range(n_months):
        mean_i  = income_matrix[:, m].mean()
        median_i = np.median(income_matrix[:, m])
        p10     = np.percentile(income_matrix[:, m], 10)
        p90     = np.percentile(income_matrix[:, m], 90)
        mean_a  = active_matrix[:, m].mean()
        mean_n  = new_matrix[:, m].mean()
        mean_b  = blown_matrix[:, m].mean()
        cum_mean   += mean_i
        cum_median += median_i
        print(f"M{m+1:>2}  | ${mean_i:>10,.0f} ${median_i:>8,.0f} ${p10:>6,.0f} ${p90:>6,.0f} | {mean_a:>10.1f} {mean_n:>7.1f} {mean_b:>9.1f} | ${cum_mean:>9,.0f}")

    # Summary
    total_income = income_matrix.sum(axis=1)
    print(f"\n{'=' * 100}")
    print(f"24-MONTH TOTAL NET INCOME (after eval reinvest):")
    print(f"  Mean:    ${total_income.mean():>10,.0f}")
    print(f"  Median:  ${np.median(total_income):>10,.0f}")
    print(f"  p10:     ${np.percentile(total_income, 10):>10,.0f}")
    print(f"  p90:     ${np.percentile(total_income, 90):>10,.0f}")
    print(f"  P(go bust early): {(active_matrix[:, -1] == 0).mean()*100:.1f}%")

    # Time to saturation
    saturation_month = []
    for r in range(N_TRIALS):
        for m in range(n_months):
            if active_matrix[r, m] >= MAX_ACTIVE_MILKERS * 0.8:
                saturation_month.append(m + 1)
                break
    if saturation_month:
        print(f"\n  Months to reach 80% saturation ({int(MAX_ACTIVE_MILKERS*0.8)} milkers):")
        print(f"    Median: {int(np.median(saturation_month))}")
        print(f"    Mean:   {np.mean(saturation_month):.1f}")
    else:
        print(f"\n  Never reaches saturation in 24 months in any path")


if __name__ == "__main__":
    main()

"""Milking progression v2 — DAY-BY-DAY tracking, not lumped per month.

Each account has a state and time-in-state. Day-stepped simulation.

States:
  FRESH       - newly funded, grinding to first $3K (lock cushion). Takes ~30 days, 71% pass.
  IN_EVAL     - eval batch in progress (10 evals fired together, ~5 days to resolve).
  CUSHIONING  - just passed eval, in hedge phase. 1 trading day to resolve.
  MILKING     - cushion locked. Cycling +$1K every ~16 days at 92.9% success.
  BLOWN       - dead, removed from pool.

Per day:
  - Each FRESH account advances; resolves at day 30 (binary: become MILKING or BLOWN).
  - Each MILKING account: cycle completes after 16 days. 92.9% extract $1K, then restart;
    7.1% BLOWN.
  - Payouts trigger reinvest:
      * Need $1.5K cash to start eval batch
      * Each batch: 10 evals × 35% pass = 3.5 funded
      * Then hedge resolves day later: 88% pair lock + 71% solo
      * So ~1.7 new MILKERS per batch on average

Time scale: 21 trading days = 1 month. 504 trading days = 24 months.
"""
from __future__ import annotations
import numpy as np

# Account states
FRESH       = "FRESH"
IN_EVAL     = "IN_EVAL"     # batch being fired
CUSHIONING  = "CUSHIONING"  # post-eval, hedge resolving
MILKING     = "MILKING"
BLOWN       = "BLOWN"

# Timing
DAYS_FIRST_CYCLE_AVG  = 30   # fresh to cushion-locked (grind $3K from $0)
DAYS_EVAL_BATCH       = 5    # to pass eval batch
DAYS_CUSHION_HEDGE    = 1    # hedge resolves in 1 day
DAYS_MILK_CYCLE_AVG   = 16   # cushion to next $1K withdrawal

# Probabilities
P_FRESH_PASS          = 0.71  # fresh account passes first cycle (user-stated)
P_MILK_SUCCESS        = 0.929 # per milk cycle
P_EVAL_PASS           = 0.35  # per individual eval (gambler's ruin)
P_HEDGE_PAIR_PASS     = 0.88
P_SOLO_PASS           = 0.71

# Constants
PAYOUT_PER_CYCLE      = 1000
PAYOUT_PER_FIRST_LOCK = 1500   # withdraw $1.5K on first lock (keeps $1.5K cushion)
BATCH_COST            = 1000   # 10 evals × $100
BATCH_CASH_NEEDED     = 1500   # batch + $500 reserve (user keeps $500 per batch)
EVALS_PER_BATCH       = 10

MAX_ACTIVE_MILKERS    = 25
N_DAYS                = 21 * 24  # 24 months
N_TRIALS              = 2000


class Account:
    __slots__ = ("state", "days_left")
    def __init__(self, state, days_left):
        self.state = state
        self.days_left = days_left


def step_account(acct: Account, rng) -> tuple[str, float]:
    """Step one trading day. Returns (event, payout_$)."""
    acct.days_left -= 1
    if acct.days_left > 0:
        return ("ONGOING", 0)
    # Resolve
    if acct.state == FRESH:
        if rng.random() < P_FRESH_PASS:
            acct.state = MILKING
            acct.days_left = int(rng.poisson(DAYS_MILK_CYCLE_AVG)) + 1
            return ("LOCKED", PAYOUT_PER_FIRST_LOCK)
        else:
            acct.state = BLOWN
            return ("BLOWN", 0)
    elif acct.state == IN_EVAL:
        # Returns LIST of fresh accounts to spawn (handled in caller)
        return ("EVAL_DONE", 0)
    elif acct.state == CUSHIONING:
        if rng.random() < P_HEDGE_PAIR_PASS:
            acct.state = MILKING
            acct.days_left = int(rng.poisson(DAYS_MILK_CYCLE_AVG)) + 1
            return ("LOCKED", 0)   # no immediate payout (already at $3K, milking begins next cycle)
        else:
            acct.state = BLOWN
            return ("BLOWN", 0)
    elif acct.state == MILKING:
        if rng.random() < P_MILK_SUCCESS:
            acct.days_left = int(rng.poisson(DAYS_MILK_CYCLE_AVG)) + 1
            return ("CYCLE_OK", PAYOUT_PER_CYCLE)
        else:
            acct.state = BLOWN
            return ("BLOWN", 0)
    return ("NOTHING", 0)


def run_one(n_days: int) -> dict:
    rng = np.random.default_rng()
    # Track accounts as list of Account objects
    accounts = [Account(FRESH, DAYS_FIRST_CYCLE_AVG)]
    cash_pool = 0   # cash available to spend on evals
    monthly_income = [0] * (n_days // 21 + 1)
    eval_batches_pending = []   # list of (resolve_day, n_evals)
    cumulative_paid = 0

    for day in range(1, n_days + 1):
        month_idx = (day - 1) // 21
        day_income = 0

        # 1. Resolve pending eval batches that finish today
        new_pending = []
        for resolve_day, n_evals in eval_batches_pending:
            if resolve_day == day:
                # Evaluate evals: each is independent
                n_passed = rng.binomial(n_evals, P_EVAL_PASS)
                # New funded accounts go into CUSHIONING (will hedge)
                for _ in range(n_passed):
                    if len([a for a in accounts if a.state != BLOWN]) >= MAX_ACTIVE_MILKERS:
                        break
                    accounts.append(Account(CUSHIONING, DAYS_CUSHION_HEDGE))
            else:
                new_pending.append((resolve_day, n_evals))
        eval_batches_pending = new_pending

        # 2. Step each active account
        for acct in accounts:
            if acct.state == BLOWN:
                continue
            evt, payout = step_account(acct, rng)
            day_income += payout

        # 3. Add income to cash pool
        cash_pool += day_income
        cumulative_paid += day_income

        # 4. Reinvest: buy as many batches as we can afford (cap by room + replenishment need)
        n_milking = sum(1 for a in accounts if a.state == MILKING)
        n_locking_in = sum(1 for a in accounts if a.state in (FRESH, CUSHIONING, IN_EVAL))
        in_pipeline = n_milking + n_locking_in + sum(n for _, n in eval_batches_pending) * P_EVAL_PASS * 0.7
        room = MAX_ACTIVE_MILKERS - n_milking - n_locking_in
        # Want to keep pipeline filled; buy if room exists and we have cash
        if room > 0 and cash_pool >= BATCH_CASH_NEEDED:
            # Buy 1 batch (avoid over-buying same day)
            cash_pool -= BATCH_COST
            eval_batches_pending.append((day + DAYS_EVAL_BATCH, EVALS_PER_BATCH))

        monthly_income[month_idx] += day_income

    return dict(
        monthly_income=monthly_income[:24],
        n_milking_final=sum(1 for a in accounts if a.state == MILKING),
        n_blown_final=sum(1 for a in accounts if a.state == BLOWN),
        cumulative_paid=cumulative_paid,
    )


def main():
    print("=" * 100)
    print("MILKING PROGRESSION v2 — Day-by-day tracking")
    print("=" * 100)
    print(f"Trials: {N_TRIALS}, days: {N_DAYS} ({N_DAYS//21} months)")
    print(f"First cycle: {DAYS_FIRST_CYCLE_AVG} days avg, {P_FRESH_PASS:.0%} pass")
    print(f"Eval batch: {DAYS_EVAL_BATCH} days, {P_EVAL_PASS:.0%} pass per eval")
    print(f"Hedge: {DAYS_CUSHION_HEDGE} day, {P_HEDGE_PAIR_PASS:.0%} lock")
    print(f"Milk cycle: {DAYS_MILK_CYCLE_AVG} days avg, {P_MILK_SUCCESS:.1%} success, ${PAYOUT_PER_CYCLE}/cycle")
    print(f"Max active: {MAX_ACTIVE_MILKERS}")
    print()

    results = [run_one(N_DAYS) for _ in range(N_TRIALS)]
    income_mat = np.array([r["monthly_income"] for r in results])   # (N_TRIALS, 24)
    final_milking = np.array([r["n_milking_final"] for r in results])
    bust = (final_milking == 0).mean()

    # Per-month stats
    print(f"{'Mon':>3} | {'Mean inc':>10} {'Median':>9} {'p10':>7} {'p90':>7} {'p50_active*':>11} | {'Cum mean':>10}")
    print("-" * 95)
    cum = 0
    # Active milkers per month estimate: simulate average
    # For now, derive from average net adds
    for m in range(24):
        col = income_mat[:, m]
        mean_inc = col.mean()
        med = np.median(col)
        p10 = np.percentile(col, 10)
        p90 = np.percentile(col, 90)
        cum += mean_inc
        # active estimate from monthly income / per-milker income
        approx_active = mean_inc / (1.31 * P_MILK_SUCCESS * PAYOUT_PER_CYCLE) if mean_inc > 0 else 0
        print(f"M{m+1:>2} | ${mean_inc:>8,.0f} ${med:>7,.0f} ${p10:>5,.0f} ${p90:>5,.0f} {approx_active:>10.1f}     | ${cum:>8,.0f}")

    total = income_mat.sum(axis=1)
    print(f"\n24-month TOTAL")
    print(f"  Mean:   ${total.mean():>10,.0f}")
    print(f"  Median: ${np.median(total):>10,.0f}")
    print(f"  p10:    ${np.percentile(total, 10):>10,.0f}")
    print(f"  p90:    ${np.percentile(total, 90):>10,.0f}")
    print(f"  P(bust early, 0 active by M24): {bust*100:.1f}%")
    print(f"  Avg active milkers at M24: {final_milking.mean():.1f}")


if __name__ == "__main__":
    main()

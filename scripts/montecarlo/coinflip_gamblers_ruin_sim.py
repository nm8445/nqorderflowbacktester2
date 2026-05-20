"""
Coinflip / Gambler's Ruin payout extraction sim.

Hypothesis: With 50/50 EV-neutral trades but asymmetric prop-firm cost structure
(cheap $100 evals vs $1,350 cash payouts), can you extract positive expected $?

Setup (user-specified):
  - 10 funded accounts simultaneously (target)
  - 3 accounts trade per day (random selection)
  - Each trade: 50/50 win/lose, symmetric R:R, $1,500 stake
  - Funded account = Lucid Flex 50K rules ($2K trailing DD, +$3K target for first payout)
  - When bust: replace with new eval at $100, gambler's-ruin pass with 30% rate

Variants tested:
  A) Pure 50/50 (no edge, no commission drag)
  B) Realistic: 50/50 with $30 round-trip slippage on each trade
  C) Slight edge: 52/48 (small edge from B2/RV-style entries)

Tracks: net cash extracted, total evals bought, accounts active, bust counts.
"""
from __future__ import annotations
import numpy as np

N_SIMS = 5_000
HORIZON_DAYS = 252        # trading days (1 year)
TARGET_SLOTS = 10         # active funded accounts
TRADES_PER_DAY = 3        # number of accounts that trade each day
TRADE_RISK = 1_500.0      # $ risk per trade
START_BAL = 50_000.0
TRAIL_DD = 2_000.0        # $ trailing DD
LOCK_AT = 53_000.0        # balance HWM that locks floor
LOCK_FLOOR = 50_000.0
PROFIT_TGT_1ST = 3_000.0  # +$3K to first payout
PROFIT_TGT_NEXT = 2_000.0 # +$2K for subsequent payouts (per user's stagger plan)
PAYOUT_1ST_GROSS = 1_500.0
PAYOUT_NEXT_GROSS = 1_000.0
SPLIT = 0.90              # Lucid 90/10
MIN_WIN_DAY = 150.0
QUAL_DAYS = 5
MAX_PAYOUTS = 6
EVAL_COST = 100.0
EVAL_PASS_RATE = 0.30
EVAL_DAYS_PER_ATTEMPT = 3  # gambler's ruin: 3 days median per attempt


class Account:
    __slots__ = ("balance","floor","hwm","locked","qual_days","cycle_profit",
                 "stagger_first_done","payouts","cash","busted")
    def __init__(self):
        self.balance = START_BAL
        self.floor = START_BAL - TRAIL_DD
        self.hwm = START_BAL
        self.locked = False
        self.qual_days = 0
        self.cycle_profit = 0.0
        self.stagger_first_done = False
        self.payouts = 0
        self.cash = 0.0
        self.busted = False


def step_account(acc, won, win_prob, slip_per_trade):
    """One trade applied. Returns trader_cash extracted this step (if payout)."""
    if won:
        pnl = TRADE_RISK - slip_per_trade
    else:
        pnl = -TRADE_RISK - slip_per_trade
    acc.balance += pnl
    # Bust check
    if acc.balance < acc.floor:
        acc.busted = True
        return 0.0
    # EOD-style trailing update (since one trade per day for that account)
    if not acc.locked:
        if acc.balance > acc.hwm:
            acc.hwm = acc.balance
        acc.floor = max(START_BAL - TRAIL_DD, acc.hwm - TRAIL_DD)
        if acc.hwm >= LOCK_AT:
            acc.locked = True
            acc.floor = LOCK_FLOOR
    # Qualifying day
    if pnl >= MIN_WIN_DAY:
        acc.qual_days += 1
    acc.cycle_profit += pnl
    # Payout?
    extracted = 0.0
    if acc.payouts < MAX_PAYOUTS and acc.qual_days >= QUAL_DAYS and acc.cycle_profit > 0:
        gross = 0.0
        if not acc.stagger_first_done:
            if acc.cycle_profit >= PROFIT_TGT_1ST:
                gross = PAYOUT_1ST_GROSS
                acc.stagger_first_done = True
        else:
            if acc.cycle_profit >= PROFIT_TGT_NEXT:
                gross = PAYOUT_NEXT_GROSS
        if gross >= 500:
            trader = gross * SPLIT
            acc.balance -= gross
            if not acc.locked:
                acc.hwm = max(START_BAL, acc.hwm - gross)
                acc.floor = max(START_BAL - TRAIL_DD, acc.hwm - TRAIL_DD)
            acc.payouts += 1
            acc.cash += trader
            extracted = trader
            acc.qual_days = 0
            acc.cycle_profit = 0.0
            if acc.payouts >= MAX_PAYOUTS:
                acc.busted = True  # graduate, slot ends
    return extracted


def simulate_one_year(rng, win_prob=0.50, slip_per_trade=0.0):
    """Returns (gross_cash, eval_cost, evals_bought, busts, payouts_total)."""
    # Need to maintain TARGET_SLOTS active funded accounts.
    # Track also accounts in 'eval' state (replacement pipeline).
    accounts = [Account() for _ in range(TARGET_SLOTS)]
    eval_pipeline = []  # list of remaining-days for each in-progress eval
    total_cash = 0.0
    total_eval_cost = 0.0
    total_evals = 0
    total_busts = 0
    total_payouts = 0

    for day in range(HORIZON_DAYS):
        # 1) Pick TRADES_PER_DAY random ACTIVE accounts to trade
        active_idxs = [i for i, a in enumerate(accounts) if not a.busted]
        if active_idxs:
            n_trade = min(TRADES_PER_DAY, len(active_idxs))
            chosen = rng.choice(active_idxs, size=n_trade, replace=False)
            for i in chosen:
                won = rng.random() < win_prob
                cash = step_account(accounts[i], won, win_prob, slip_per_trade)
                total_cash += cash
                if cash > 0:
                    total_payouts += 1
                if accounts[i].busted:
                    total_busts += 1

        # 2) Advance eval pipeline; passes refill busted slots
        new_pipeline = []
        for days_remaining in eval_pipeline:
            days_remaining -= 1
            if days_remaining <= 0:
                # eval attempt resolves
                if rng.random() < EVAL_PASS_RATE:
                    # passed — find busted slot to refill
                    for i, a in enumerate(accounts):
                        if a.busted:
                            accounts[i] = Account()
                            break
                # if no busted slot to fill (shouldn't happen often), just discard
                # if failed, eval is gone, will need new one
            else:
                new_pipeline.append(days_remaining)
        eval_pipeline = new_pipeline

        # 3) For each busted slot without a pending eval, start a new eval
        n_busted = sum(1 for a in accounts if a.busted)
        n_pending = len(eval_pipeline)
        n_to_start = max(0, n_busted - n_pending)
        for _ in range(n_to_start):
            total_eval_cost += EVAL_COST
            total_evals += 1
            eval_pipeline.append(EVAL_DAYS_PER_ATTEMPT)

    return total_cash, total_eval_cost, total_evals, total_busts, total_payouts


def run_scenario(name, win_prob, slip):
    rng = np.random.default_rng(seed=hash(name) % 9973)
    sims = [simulate_one_year(rng, win_prob, slip) for _ in range(N_SIMS)]
    gross = np.array([s[0] for s in sims])
    evals_cost = np.array([s[1] for s in sims])
    n_evals = np.array([s[2] for s in sims])
    n_busts = np.array([s[3] for s in sims])
    n_pmts = np.array([s[4] for s in sims])
    net = gross - evals_cost
    print(f"\n--- {name} ---")
    print(f"  win_prob = {win_prob:.2%}, slippage per trade = ${slip:.0f}")
    print(f"  Mean gross cash:     ${gross.mean():>9,.0f}")
    print(f"  Mean eval cost:      ${evals_cost.mean():>9,.0f}  ({n_evals.mean():.0f} evals/yr)")
    print(f"  Mean NET cash:       ${net.mean():>9,.0f}    (p25 ${np.percentile(net,25):,.0f} / p75 ${np.percentile(net,75):,.0f})")
    print(f"  Mean #busts/yr:      {n_busts.mean():.1f}")
    print(f"  Mean #payouts/yr:    {n_pmts.mean():.1f}")
    print(f"  Monthly mean NET:    ${net.mean()/12:,.0f}")
    return net


def main():
    print("=" * 70)
    print("COINFLIP / GAMBLER'S RUIN PAYOUT EXTRACTION SIM")
    print(f"10 accounts max, {TRADES_PER_DAY} trades/day across all, ${TRADE_RISK:.0f} risk/trade")
    print(f"Lucid Flex 50K rules, stagger payouts ($1500 first / $1000 subsequent)")
    print(f"Eval: $100, {EVAL_PASS_RATE:.0%} pass rate, {EVAL_DAYS_PER_ATTEMPT} days/attempt")
    print(f"{N_SIMS} sims over {HORIZON_DAYS}-day year")
    print("=" * 70)

    run_scenario("PURE coin flip (no friction)", win_prob=0.50, slip=0.0)
    run_scenario("Realistic 50/50 w/ $30 slip/trade",  win_prob=0.50, slip=30.0)
    run_scenario("Slight edge 52/48 w/ $30 slip",      win_prob=0.52, slip=30.0)
    run_scenario("Strong edge 55/45 w/ $30 slip",      win_prob=0.55, slip=30.0)
    run_scenario("Reverse edge 48/52 w/ $30 slip",     win_prob=0.48, slip=30.0)


if __name__ == "__main__":
    main()

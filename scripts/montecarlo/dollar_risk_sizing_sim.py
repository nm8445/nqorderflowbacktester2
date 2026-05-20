"""
Fixed dollar-risk per trade — gambler's ruin payout extraction analysis.

User idea: ATR-based fixed SL/TP, position-sized to fixed $X risk per trade.
Tests how P(pass eval) and funded throughput change with:
  - Risk per trade: $250, $500, $750, $1000, $1500
  - Win rate: 45%, 48%, 50%, 52%, 55%, 60%
  - 1:1 R:R assumed (symmetric)

Key insight from gambler's ruin:
  P(reach $3K target before $2K trail bust) = (1 - r^b) / (1 - r^(a+b))
  where a = target/step, b = bust/step, r = q/p, with p = win rate
"""
from __future__ import annotations
import numpy as np

N_SIMS = 10_000
HORIZON_DAYS = 252
TRADES_PER_DAY = 1        # one trade per day per account
PROFIT_TGT = 3_000.0
TRAIL_DD = 2_000.0
LOCK_AT = 53_000.0
LOCK_FLOOR = 50_000.0
START_BAL = 50_000.0
MIN_WIN_DAY = 150.0
QUAL_DAYS = 5
PAYOUT_1ST = 1_500.0
PAYOUT_NEXT = 1_000.0
MAX_PAYOUTS = 6
SPLIT = 0.90
TRADE_FRICTION = 15.0      # $/trade commission+slip


def gamblers_ruin_p(win_prob, risk, target=PROFIT_TGT, bust=TRAIL_DD):
    """Closed-form P(reach +target before -bust). Step size = risk."""
    a = int(round(target / risk))
    b = int(round(bust / risk))
    if a == 0 or b == 0:
        return 0.0
    p = win_prob
    q = 1 - p
    if abs(p - 0.5) < 1e-9:
        return b / (a + b)
    r = q / p
    return (1 - r**b) / (1 - r**(a + b))


def simulate_funded_extraction(rng, win_prob, risk, friction=TRADE_FRICTION):
    """Simulate one funded account over a year with fixed-$ risk per trade.
    Track payouts cashed and bust.
    Returns (cash, busted, bust_day, payouts_count)."""
    balance = START_BAL
    floor = START_BAL - TRAIL_DD
    hwm = balance
    locked = False
    qual_days = 0
    cycle_profit = 0.0
    stagger_first_done = False
    payouts = 0
    cash = 0.0
    for d in range(HORIZON_DAYS):
        # 1 trade per day
        won = rng.random() < win_prob
        pnl = (risk if won else -risk) - friction
        balance += pnl
        if balance < floor:
            return cash, True, d, payouts
        if not locked:
            if balance > hwm:
                hwm = balance
            floor = max(START_BAL - TRAIL_DD, hwm - TRAIL_DD)
            if hwm >= LOCK_AT:
                locked = True
                floor = LOCK_FLOOR
        if pnl >= MIN_WIN_DAY:
            qual_days += 1
        cycle_profit += pnl
        if payouts < MAX_PAYOUTS and qual_days >= QUAL_DAYS and cycle_profit > 0:
            gross = 0.0
            if not stagger_first_done:
                if cycle_profit >= PROFIT_TGT:
                    gross = PAYOUT_1ST
                    stagger_first_done = True
            else:
                if cycle_profit >= 2000:
                    gross = PAYOUT_NEXT
            if gross > 0:
                trader = gross * SPLIT
                balance -= gross
                if not locked:
                    hwm = max(START_BAL, hwm - gross)
                    floor = max(START_BAL - TRAIL_DD, hwm - TRAIL_DD)
                payouts += 1
                cash += trader
                qual_days = 0
                cycle_profit = 0.0
                if payouts >= MAX_PAYOUTS:
                    break
    return cash, False, None, payouts


def main():
    print("=" * 75)
    print("FIXED DOLLAR-RISK GAMBLER'S RUIN: P(pass) and FUNDED EXTRACTION")
    print(f"Target +$3K, Bust -$2K trail, 1:1 R:R, {TRADE_FRICTION} friction/trade")
    print("=" * 75)

    risks = [250, 500, 750, 1000, 1500]
    win_rates = [0.45, 0.48, 0.50, 0.52, 0.55, 0.60]

    # === Closed-form P(pass) eval ===
    print("\n=== P(reach +$3K before -$2K trail) — closed-form (no friction) ===\n")
    header = "Win rate |" + "|".join(f" ${r}".rjust(8) for r in risks) + "|"
    print(header)
    print("-" * len(header))
    for p in win_rates:
        row = f"  {p:.0%}    |"
        for r in risks:
            P = gamblers_ruin_p(p, r)
            row += f"  {P*100:>5.1f}% |"
        print(row)

    # === Simulated funded-account extraction (1 yr, w/ friction) ===
    print("\n=== Mean NET cash per funded account-year (with $15 friction, 1 trade/day) ===\n")
    print(header.replace("Win rate", "Win rate"))
    print("-" * len(header))
    for p in win_rates:
        row = f"  {p:.0%}    |"
        for r in risks:
            rng = np.random.default_rng(seed=int(p*1000) * 17 + r)
            sims = [simulate_funded_extraction(rng, p, r) for _ in range(N_SIMS)]
            mean_cash = np.mean([s[0] for s in sims])
            row += f" ${mean_cash:>6,.0f} |"
        print(row)

    # === Bust rate ===
    print("\n=== Bust rate per funded year ===\n")
    print(header)
    print("-" * len(header))
    for p in win_rates:
        row = f"  {p:.0%}    |"
        for r in risks:
            rng = np.random.default_rng(seed=int(p*1000) * 19 + r + 7)
            sims = [simulate_funded_extraction(rng, p, r) for _ in range(N_SIMS)]
            bust_rate = np.mean([s[1] for s in sims])
            row += f"  {bust_rate*100:>5.1f}% |"
        print(row)

    # === Annual evals needed (P(pass) drives this) ===
    print("\n=== Avg evals needed per funded pass (1/P_pass × $100 = eval cost per slot) ===\n")
    print(header)
    print("-" * len(header))
    for p in win_rates:
        row = f"  {p:.0%}    |"
        for r in risks:
            P = gamblers_ruin_p(p, r)
            evals = 1 / P if P > 0 else 999
            cost = evals * 100
            row += f" ${cost:>6,.0f} |"
        print(row)

    # === 5-Winning-Day rule analysis ===
    print("\n=== 5-winning-day rule check (binding constraint) ===")
    print("With fixed $X risk + 1:1 R:R, each win is a 'winning day' (>$150).")
    print("Need 5 winning days AND +$3K cycle profit. So minimum cycle = 5W + (W-L)*X = +$3K")
    print()
    for r in risks:
        # need W-L >= 3000/r winning vs losing days
        net_wins_needed = int(np.ceil(PROFIT_TGT / r))
        # to also have 5 wins: min wins = max(5, net_wins_needed)
        # Actually since each win = winning day, need W >= 5
        min_W = max(5, net_wins_needed)
        max_L = min_W - net_wins_needed
        print(f"  ${r:>5} risk: need >= {min_W} wins, <= {max_L} losses. "
              f"Min cycle length = {min_W + max_L} trades. "
              f"5-day rule {'BINDING' if min_W > net_wins_needed else 'auto-satisfied'}")


if __name__ == "__main__":
    main()

"""MC: Funded phase — straight 1 NQ vs cross-firm hedge.

User has 10 funded accounts. Two approaches to build $2K cushion (DD lock):

  APPROACH A: HEDGE
    Pair into 5 pairs. 1:2 RR ($2K SL / $4K TP).
    Random-walk per pair: 80% land with $2-4K cushion, 20% both blow.
    Survivors: ~4 (per 10 funded).

  APPROACH B: STRAIGHT 1 NQ
    Each account runs 4-strat at 1 NQ independently.
    Stop when cum P&L hits +$2K (cushion locked) or -$2K (blown).
    Survivors: bootstrap from historical daily P&L.

Compare:
  - Avg # survivors out of 10 funded
  - Total cushion locked across survivors
  - Days-to-cushion-lock
  - Risk profile (path-dependent vs structural)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADES_CSV = "live/combined deployment plan/combined_trades.csv"
N_TRIALS = 20_000
N_FUNDED = 10
TARGET = 2000     # +$2K to lock cushion
BLOW   = -2000    # -$2K to blow account
HEDGE_PAIR_PASS = 0.80   # from random-walk analysis
HEDGE_TP_PROB   = 0.30   # P(survivor lands at +$4K vs +$2K)
RNG = np.random.default_rng(42)


def load_daily_pnl_1nq():
    df = pd.read_csv(TRADES_CSV)
    return df.groupby("date")["pnl_$"].sum().values


def bootstrap_single_account(daily_pnl, n_trials, target=TARGET, blow=BLOW, max_days=30):
    """Per-account: sample days until +$2K or -$2K."""
    rows = []
    for _ in range(n_trials):
        cum = 0.0
        days = 0
        while days < max_days:
            cum += RNG.choice(daily_pnl)
            days += 1
            if cum >= target:
                rows.append((True, days, cum)); break
            if cum <= blow:
                rows.append((False, days, cum)); break
        else:
            rows.append((False, days, cum))
    return pd.DataFrame(rows, columns=["locked", "days", "final_cum"])


def main():
    daily = load_daily_pnl_1nq()
    print(f"Loaded {len(daily)} live trading days")
    print(f"  Daily P&L (1 NQ): mean=${daily.mean():.0f}  std=${daily.std():.0f}")

    print("\n" + "=" * 90)
    print("APPROACH A: HEDGE (5 pairs, 1:2 RR)")
    print("=" * 90)
    # Per-pair: 80% land 1 survivor with cushion, 20% both blow
    # 30% of survivors at +$4K TP, 70% at +$2K
    survivors_per_cycle = []
    cushion_per_cycle = []
    n_pairs = N_FUNDED // 2
    for _ in range(N_TRIALS):
        n_hedge_survive = RNG.binomial(n_pairs, HEDGE_PAIR_PASS)
        n_tp = RNG.binomial(n_hedge_survive, HEDGE_TP_PROB)
        n_sl = n_hedge_survive - n_tp
        cushion = n_tp * 4000 + n_sl * 2000
        survivors_per_cycle.append(n_hedge_survive)
        cushion_per_cycle.append(cushion)

    survivors_a = np.array(survivors_per_cycle)
    cushion_a = np.array(cushion_per_cycle)
    print(f"  Avg survivors:        {survivors_a.mean():.2f} / 10 funded")
    print(f"  Median survivors:     {int(np.median(survivors_a))}")
    print(f"  Survivor distribution: " +
          " ".join(f"{k}c:{(survivors_a == k).sum()/N_TRIALS*100:.0f}%" for k in range(n_pairs + 1)))
    print(f"  Avg cushion locked:   ${cushion_a.mean():,.0f}")
    print(f"  Hedge phase duration: ~1 day (set up, let one side blow)")
    print(f"  Accounts permanently lost in hedge: 5 (one per pair)")

    # First payout potential: $2K per survivor
    payout_a = survivors_a * 2000
    print(f"  First payout revenue: ${payout_a.mean():,.0f}  (range ${payout_a.min():,.0f}-${payout_a.max():,.0f})")
    print(f"  Surviving accounts after first payout: {survivors_a.mean():.2f}")

    print("\n" + "=" * 90)
    print("APPROACH B: STRAIGHT 1 NQ (each account independently)")
    print("=" * 90)
    df_single = bootstrap_single_account(daily, N_TRIALS, max_days=30)
    p_lock = df_single["locked"].mean()
    locked_days = df_single[df_single["locked"]]["days"]
    blown_days  = df_single[~df_single["locked"]]["days"]

    print(f"  Per-account P(lock $2K cushion before blowing $2K): {p_lock*100:.1f}%")
    print(f"  Per-account P(blow): {(1-p_lock)*100:.1f}%")
    print(f"  Days to lock cushion: median={locked_days.median():.0f}  avg={locked_days.mean():.1f}  p95={locked_days.quantile(0.95):.0f}")
    print(f"  Days to blow:         median={blown_days.median():.0f}  avg={blown_days.mean():.1f}")

    # Simulate 10 independent accounts
    survivors_b = []
    for _ in range(N_TRIALS):
        n_survive = RNG.binomial(N_FUNDED, p_lock)
        survivors_b.append(n_survive)
    survivors_b = np.array(survivors_b)
    cushion_b = survivors_b * 2000  # 1 NQ approach: $2K cushion per survivor

    print(f"\n  Avg survivors:        {survivors_b.mean():.2f} / 10 funded")
    print(f"  Median survivors:     {int(np.median(survivors_b))}")
    print(f"  Survivor distribution: " +
          " ".join(f"{k}c:{(survivors_b == k).sum()/N_TRIALS*100:.0f}%" for k in range(N_FUNDED + 1)
                  if (survivors_b == k).sum() / N_TRIALS > 0.01))
    print(f"  Avg cushion locked:   ${cushion_b.mean():,.0f}")

    payout_b = survivors_b * 2000
    print(f"  First payout revenue: ${payout_b.mean():,.0f}  (range ${payout_b.min():,.0f}-${payout_b.max():,.0f})")

    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 90)
    print(f"{'Metric':<40} {'Hedge':>15} {'Straight 1 NQ':>15}")
    print("-" * 75)
    print(f"{'Avg survivors / 10 funded':<40} {survivors_a.mean():>15.2f} {survivors_b.mean():>15.2f}")
    print(f"{'Avg total cushion locked':<40} {f'${cushion_a.mean():,.0f}':>15} {f'${cushion_b.mean():,.0f}':>15}")
    print(f"{'Avg first payout revenue':<40} {f'${payout_a.mean():,.0f}':>15} {f'${payout_b.mean():,.0f}':>15}")
    print(f"{'P(survivors >= 5)':<40} {(survivors_a >= 5).mean()*100:>14.1f}% {(survivors_b >= 5).mean()*100:>14.1f}%")
    print(f"{'P(survivors >= 7)':<40} {(survivors_a >= 7).mean()*100:>14.1f}% {(survivors_b >= 7).mean()*100:>14.1f}%")
    print(f"{'Setup time':<40} {'~1 day':>15} {f'{locked_days.median():.0f}-{locked_days.quantile(0.95):.0f} days':>15}")
    print(f"{'Detection risk':<40} {'Moderate':>15} {'None':>15}")

    print("\n" + "=" * 90)
    print("NET CYCLE COMPARISON (10 funded, 30-day cycle)")
    print("=" * 90)
    # Both add $2K milking after first payout (conservative)
    milk_revenue = 2000  # avg from earlier MC

    hedge_total = payout_a.mean() + milk_revenue
    nq_total    = payout_b.mean() + milk_revenue * survivors_b.mean() / survivors_a.mean()  # scale milk to survivors

    print(f"  Hedge:        first_payout=${payout_a.mean():,.0f}  +milk=${milk_revenue:,.0f}  =${hedge_total:,.0f} / cycle")
    print(f"  Straight 1NQ: first_payout=${payout_b.mean():,.0f}  +milk=~${milk_revenue * survivors_b.mean() / survivors_a.mean():,.0f}  =${nq_total:,.0f} / cycle")

    eval_cost = 100 * 18  # ~18 evals/cycle from earlier
    print(f"\n  Net per cycle (after $1.8K eval cost):")
    print(f"    Hedge:        ${hedge_total - eval_cost:,.0f}")
    print(f"    Straight 1NQ: ${nq_total - eval_cost:,.0f}")
    print(f"\n  Annual (8.3 cycles/yr):")
    print(f"    Hedge:        ${(hedge_total - eval_cost) * 8.3:,.0f}")
    print(f"    Straight 1NQ: ${(nq_total - eval_cost) * 8.3:,.0f}")


if __name__ == "__main__":
    main()

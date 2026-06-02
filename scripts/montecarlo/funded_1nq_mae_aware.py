"""MC: Funded phase 1 NQ — WITH unrealized MAE checks.

User's critical point: at 1 NQ with $2K trailing DD, unrealized intraday losses
can blow the account before realized P&L ever materializes. My earlier MC
only used realized pnl_$ — UNDERSTATED bust risk significantly.

Real bust check: account_state + current_unrealized < floor.
Per trade: if account_state + trade.MAE < floor → BLOWN.

Models:
  A. Sequential trades, MAE-aware bust check
  B. Compare to hedge (guaranteed cushion via 1:2 mechanic)
  C. Show how per-strategy choice changes bust rate (skip OD vs all 4)

Floor = -$2000 (fresh funded, trailing DD)
Lock cushion = +$2000 (after which floor trails to start, account safe)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADES_CSV = "live/combined deployment plan/combined_trades_with_mae.csv"
TARGET_CUSHION = 2000
DD_FLOOR = -2000
N_TRIALS = 20_000
N_FUNDED = 10
RNG = np.random.default_rng(42)


def load_trades(strats=None):
    df = pd.read_csv(TRADES_CSV)
    if strats is not None:
        df = df[df["strat"].isin(strats)]
    return df[["pnl_$", "mae_$", "strat"]].values


def simulate_account(trades_arr, target=TARGET_CUSHION, floor=DD_FLOOR, max_trades=200):
    """Single account: sample trades, bust on MAE touching floor, lock at target."""
    cum_pnl = 0.0
    n = 0
    while n < max_trades:
        idx = RNG.integers(0, len(trades_arr))
        pnl = trades_arr[idx][0]
        mae = trades_arr[idx][1]
        # Bust check: low point of this trade = cum_pnl + mae
        # mae is negative, so cum_pnl + mae = lowest equity reached during trade
        low = cum_pnl + mae
        if low <= floor:
            return False, n + 1
        cum_pnl += pnl
        n += 1
        if cum_pnl >= target:
            return True, n
        if cum_pnl <= floor:  # realized blow (rare since MAE check happens first)
            return False, n
    return False, n


def run_mc(strats_label, strats, n_trials=N_TRIALS):
    arr = load_trades(strats)
    print(f"\n  Loaded {len(arr)} trades for {strats_label}")
    print(f"  Mean MAE: ${np.mean([t[1] for t in arr]):.0f}")
    print(f"  Trades touching -$2K MAE: {sum(1 for t in arr if t[1] <= -2000)}/{len(arr)} "
          f"({100*sum(1 for t in arr if t[1] <= -2000)/len(arr):.1f}%)")

    results = [simulate_account(arr) for _ in range(n_trials)]
    locked = [r[0] for r in results]
    trades_taken = [r[1] for r in results]

    p_lock = np.mean(locked)
    locked_trades = [t for r, t in zip(locked, trades_taken) if r]
    blown_trades  = [t for r, t in zip(locked, trades_taken) if not r]

    return dict(
        label=strats_label,
        p_lock=p_lock,
        n_trades=len(arr),
        median_trades_to_lock=np.median(locked_trades) if locked_trades else np.nan,
        median_trades_to_blow=np.median(blown_trades) if blown_trades else np.nan,
    )


def main():
    print("=" * 90)
    print("1 NQ FUNDED PHASE: MAE-AWARE BUST CHECK")
    print("=" * 90)
    print(f"Floor: ${DD_FLOOR}    Target cushion lock: +${TARGET_CUSHION}")
    print(f"Bust rule: account_state + trade.MAE <= floor -> BLOWN")

    # Variants
    variants = [
        ("4-strat (all: OD+B2+RV)", None),
        ("OD only",                  ["OD"]),
        ("B2+RV only (skip OD)",     ["B2", "RV"]),
        ("OD+B2 (skip RV)",          ["OD", "B2"]),
        ("OD+RV (skip B2)",          ["OD", "RV"]),
    ]

    summaries = []
    for label, strats in variants:
        s = run_mc(label, strats)
        summaries.append(s)

    print("\n" + "=" * 90)
    print("PER-ACCOUNT LOCK RATE AT 1 NQ (MAE-AWARE)")
    print("=" * 90)
    print(f"{'Variant':<35} {'P(lock $2K)':>14} {'P(blow)':>10} {'Med trades to lock':>20}")
    print("-" * 85)
    for s in summaries:
        med_lock = f"{s['median_trades_to_lock']:.0f}" if not np.isnan(s['median_trades_to_lock']) else "n/a"
        print(f"  {s['label']:<33} {s['p_lock']*100:>13.1f}% {(1-s['p_lock'])*100:>9.1f}% {med_lock:>20}")

    # 10-account simulation per variant
    from scipy.stats import binom
    print("\n" + "=" * 90)
    print("10-ACCOUNT FUNDED CYCLE OUTCOMES (each variant)")
    print("=" * 90)
    print(f"{'Variant':<35} {'Avg survivors':>14} {'P(>=5)':>10} {'P(<=2 disaster)':>17} {'First payout':>14}")
    print("-" * 95)
    for s in summaries:
        avg = 10 * s['p_lock']
        p_5 = (1 - binom.cdf(4, 10, s['p_lock'])) * 100
        p_2 = binom.cdf(2, 10, s['p_lock']) * 100
        revenue = avg * 2000
        print(f"  {s['label']:<33} {avg:>13.2f}  {p_5:>9.1f}% {p_2:>16.2f}% {f'${revenue:,.0f}':>14}")

    # Hedge baseline
    p_hedge_pair = 0.80
    avg_hedge = 5 * p_hedge_pair
    revenue_hedge = avg_hedge * 2000
    print(f"\n  {'HEDGE (5 pairs, 80%)':<33} {avg_hedge:>13.2f}  {binom.pmf(5,5,0.80)*100:>9.1f}% {binom.cdf(2,5,0.80)*100:>16.2f}% {f'${revenue_hedge:,.0f}':>14}")

    print("\n" + "=" * 90)
    print("ANNUAL NET COMPARISON (30-day cycle, ~$1.8K eval cost)")
    print("=" * 90)
    eval_cost = 1800
    milk_factor = 0.25  # $2K milk per surviving account roughly
    for s in summaries:
        survivors = 10 * s['p_lock']
        first_payout = survivors * 2000
        milk = survivors * 500  # conservative
        net_cycle = first_payout + milk - eval_cost
        annual = net_cycle * 8.3
        print(f"  {s['label']:<33} survivors={survivors:>5.2f}  payout=${first_payout:>6,.0f}  net/cycle=${net_cycle:>6,.0f}  annual=${annual:>8,.0f}")
    # Hedge
    survivors = avg_hedge
    first_payout = survivors * 2000
    milk = survivors * 500
    net_cycle = first_payout + milk - eval_cost
    annual = net_cycle * 8.3
    print(f"  {'HEDGE':<33} survivors={survivors:>5.2f}  payout=${first_payout:>6,.0f}  net/cycle=${net_cycle:>6,.0f}  annual=${annual:>8,.0f}")


if __name__ == "__main__":
    main()

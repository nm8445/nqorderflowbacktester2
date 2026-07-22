"""50k funded: which gamble->milk->payout PATH makes the most money, and the best gamble risk.

Paths (all: gamble 1:1 at risk R to a target, $2k trailing-lock floor; then milk 1 MNQ to the withdraw
target; withdraw 50% of profit capped $2k after 5 winning days (>=$150), 90% split, 2 payouts max):
  A  gamble -> +3k, withdraw at +3k
  B  gamble -> +4k, withdraw at +4k          (the 'fat buffer via gambling' idea)
  C  gamble -> +2k lock, MILK up to +4k, withdraw at +4k   (the 'lock then milk' idea)

Also: P(>=1 payout) across a 10-eval buy (34% MAE-aware eval pass -> ~3-4 funded).
Run: python scripts/farm_income/funded_path_compare.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from farm_income_mc import load_eval_R, load_daily_1mnq

Rd = load_eval_R(); daily = load_daily_1mnq()
DD, WIN, WINS, SPLIT, CAP, MINW, MAXP, MILK = 2000., 150., 5, 0.90, 2000., 500., 2, 1
P_PASS = 0.34        # 50k eval pass rate — MAE-aware (real firm floating rule; old 0.43 was realized-only/optimistic)


def life(R, G, W, rng):
    """Returns (cash extracted after split, n_payouts). G=gamble target, W=withdraw target."""
    p = 0.0; peak = 0.0
    while p < G:                                    # gamble 1:1 at risk R to +G
        p += Rd[rng.integers(Rd.size)] * R
        peak = max(peak, p)
        if p <= min(0.0, peak - DD) + 1e-9:
            return 0.0, 0                            # blew gambling
    cash = 0.0; pays = 0                             # locked (G>=2000 -> floor at 0); milk + payouts
    for _ in range(MAXP):
        wins = 0
        while p < W or wins < WINS:
            pnl = daily[rng.integers(daily.size)] * MILK
            p += pnl
            if p <= 1e-9:
                return cash, pays                    # milk blew (back to start)
            if pnl >= WIN:
                wins += 1
        w = min(0.5 * p, CAP)
        if w < MINW:
            return cash, pays
        cash += w * SPLIT; p -= w; pays += 1
    return cash, pays


def stats(R, G, W, n=40000, seed=3):
    rng = np.random.default_rng(seed)
    res = [life(R, G, W, rng) for _ in range(n)]
    cash = np.array([r[0] for r in res]); pays = np.array([r[1] for r in res])
    return cash.mean(), (pays >= 1).mean()


def cohort_any_payout(p_pay, n_evals=10, n=300000, seed=5):
    """P(>=1 payout) and avg funded count buying n_evals (each passes p_pass -> each funded pays p_pay)."""
    rng = np.random.default_rng(seed)
    funded = rng.random((n, n_evals)) < P_PASS
    paid = (rng.random((n, n_evals)) < p_pay) & funded
    return paid.any(axis=1).mean(), funded.sum(axis=1).mean()


def main():
    paths = {"A  gamble->3k          ": (3000, 3000),
             "B  gamble->4k          ": (4000, 4000),
             "C  gamble->2k, milk->4k": (2000, 4000)}
    print("E[$/funded]  and  P(>=1 payout per funded)  by path x gamble risk (milk 1 MNQ):\n")
    print(f"  {'path':<24} {'risk':>6} {'E[$/funded]':>12} {'P(pay)/funded':>14}")
    best = {}
    for name, (G, W) in paths.items():
        for R in (500, 750, 1000, 1500, 2000):
            ef, pp = stats(R, G, W)
            tag = ""
            if name not in best or ef > best[name][1]:
                best[name] = (R, ef, pp)
            print(f"  {name:<24} {R:>6} {ef:>12,.0f} {pp*100:>13.0f}%")
        print()
    print("=" * 70)
    print("BEST risk per path (by E[$/funded]) + cohort P(>=1 payout) buying 10 evals:")
    print("=" * 70)
    print(f"  {'path':<24} {'best risk':>9} {'E[$/funded]':>12} {'P(pay)/fund':>12} {'P(>=1 pay of 10 evals)':>24}")
    for name, (R, ef, pp) in best.items():
        coh, avg_funded = cohort_any_payout(pp)
        print(f"  {name:<24} {R:>9} {ef:>12,.0f} {pp*100:>11.0f}% {coh*100:>22.0f}%")
    print(f"\n  (10 evals x {P_PASS*100:.0f}% pass = ~{10*P_PASS:.1f} funded on average)")


if __name__ == "__main__":
    main()

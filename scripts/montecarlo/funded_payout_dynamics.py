"""Funded payout dynamics — real strategy (green-touch TP uncapped, soft trailing yellow, force-close),
sized to the initial yellow = $2k risk, one signal at a time (firing-order, freq-weighted). 50k acct,
$2k trailing-then-lock DD. Milk: reach +$3k -> withdraw to +$2k, repeat until blow. 1-yr horizon.

KEY vs the challenge: TP is NOT capped at $1,500 here, so a single green win (OD avg +1.68R = +$3.4k)
can hit a payout in ONE trade -> reaching $3k is far likelier than the capped 35-45% challenge.

Uses _risknorm pnl_R/mae_R (real exits), RV filtered at ATR>150 (stop>300). Run:
    python scripts/montecarlo/funded_payout_dynamics.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

RN = Path(__file__).resolve().parents[2] / "scripts" / "cfd prop firms" / "_risknorm_trades.csv"
RISK, GCOST = 2000., 30.


def pool():
    d = pd.read_csv(RN)
    d = d[~((d.strat == "RV") & (d.stop_pts > 300))]            # RV ATR-150 filter
    return np.column_stack([(d.pnl_pts / d.stop_pts).values, (d.mae_pts / d.stop_pts).values])


def account(P, rng, days=252):
    n = len(P); bal = 50000.; peak = 50000.; floor = 48000.; locked = False; pays = 0; wd = 0.
    first_pay_day = None
    for day in range(days):
        pr, ma = P[rng.integers(0, n)]
        if ma >= 1.0 or pr <= -1.0:                              # float touched yellow OR realized >= buffer -> blow
            return pays, wd, first_pay_day, "blow"
        bal += pr * RISK - GCOST
        if bal > peak: peak = bal
        if not locked:
            floor = min(50000., peak - 2000.)
            if floor >= 50000.: locked = True; floor = 50000.
        if bal >= 53000.:                                        # +$3k -> withdraw to +$2k
            w = bal - 52000.; wd += w; bal -= w; pays += 1
            if first_pay_day is None: first_pay_day = day + 1
    return pays, wd, first_pay_day, "survive"


def main():
    P = pool(); rng = np.random.default_rng(7); N = 100000
    res = [account(P, rng) for _ in range(N)]
    pays = np.array([r[0] for r in res]); wd = np.array([r[1] for r in res])
    fpd = [r[2] for r in res if r[2] is not None]
    reached = np.mean(pays >= 1)
    print("FUNDED (real exits, $2k-yellow, milk to +$3k->+$2k, 1yr):\n")
    print(f"  P(reach $3k = >=1 payout): {reached*100:.0f}%   (vs capped challenge 35-45%)")
    print(f"  median days to first payout: {int(np.median(fpd))}")
    print(f"  avg # payouts / funded acct (its lifetime): {pays.mean():.2f}  (median {int(np.median(pays))})")
    print(f"  avg $ withdrawn / funded acct: ${wd.mean():,.0f}")
    paid = pays[pays >= 1]
    print(f"  among accounts that pay AT ALL: avg {paid.mean():.1f} payouts, "
          f"avg ${wd[pays>=1].mean()/paid.mean():,.0f}/payout")
    print(f"\n  Of 10 funded accounts:")
    print(f"    ~{10*reached:.1f} pay out at all; ~{10*(1-reached):.1f} blow before any payout")
    print(f"    total payouts across the 10: ~{10*pays.mean():.0f}/yr  (~${10*wd.mean():,.0f} withdrawn)")


if __name__ == "__main__":
    main()

"""Capped-spend cohort: buy 30 evals ($3k @ $100), ~45% pass -> ~14 funded. Each funded gambles to a
buffer -> de-risks to 1 MNQ -> milks (5 winning days >=$150 -> withdraw 50% of profit capped $2k,
80% split, leave 50%+DD, MAX 4 payouts then retire). NO reinvestment (keep the profit). Milk is
copy-traded (shared pack = correlated). Run: python scripts/montecarlo/cohort_harvest.py
"""
from __future__ import annotations
import numpy as np
from farm_firing_order import gamble_pool, milk_packs

P_PASS, N_EVAL, EVAL = 0.452, 30, 100.
WIN_DAY, MAXPAY, SPLIT, MAXW = 150., 4, 0.80, 2000.


def newg():
    return dict(st="gamble", bal=50000., peak=50000., floor=48000., locked=False, paid=0, wdays=0)


def run(G, M, rng, days=252):
    ng, nm = len(G), len(M)
    n_funded = int(np.sum(rng.random(N_EVAL) < P_PASS))     # ~14 of 30 pass
    accts = []; born = False
    take = 0.; npay = 0; n_paid = 0; reach = 0
    for day in range(days):
        if day == 4 and not born:                            # funded come online after the ~4-day challenge gamble
            accts = [newg() for _ in range(n_funded)]; born = True
        pack = M[rng.integers(0, nm)]; alive = []
        for a in accts:
            if a["st"] == "gamble":                          # independent gamble to a buffer
                pr, ma = G[rng.integers(0, ng)]
                if ma >= 1.0 or pr <= -1.0: continue          # blow
                a["bal"] += pr * 2000. - 30.
                if a["bal"] > a["peak"]: a["peak"] = a["bal"]
                if not a["locked"]:
                    a["floor"] = min(50000., a["peak"] - 2000.)
                    if a["floor"] >= 50000.: a["locked"] = True; a["floor"] = 50000.
                if a["bal"] > 50000.: a["st"] = "milk"; reach += 1
                alive.append(a)
            else:                                            # milk 1 MNQ, shared pack (correlated)
                d0 = a["bal"]; dead = False
                for pn, fl in pack:
                    if a["bal"] - fl <= a["floor"]: dead = True; break
                    a["bal"] += pn - 2.
                if dead: continue
                if a["bal"] > a["peak"]: a["peak"] = a["bal"]
                if not a["locked"]:
                    a["floor"] = min(50000., a["peak"] - 2000.)
                    if a["floor"] >= 50000.: a["locked"] = True; a["floor"] = 50000.
                if a["bal"] - d0 >= WIN_DAY: a["wdays"] += 1
                profit = a["bal"] - 50000.
                if a["wdays"] >= 5 and profit > 0.:
                    w = min(0.5 * profit, MAXW); t = SPLIT * w
                    take += t; a["bal"] -= w; a["wdays"] = 0; npay += 1
                    if a["paid"] == 0: n_paid += 1
                    a["paid"] += 1
                    if a["paid"] >= MAXPAY: continue          # retire at 4 payouts (don't re-add)
                alive.append(a)
        accts = alive
    return take, npay, n_funded, reach, n_paid


def main():
    G = gamble_pool(); M = milk_packs(); N = 5000
    rng = np.random.default_rng(11)
    res = [run(G, M, rng) for _ in range(N)]
    take = np.array([r[0] for r in res]); npay = np.array([r[1] for r in res])
    nf = np.array([r[2] for r in res]); rc = np.array([r[3] for r in res]); npd = np.array([r[4] for r in res])
    net = take - N_EVAL * EVAL
    print("CAPPED COHORT: $3k -> 30 evals -> milk, NO reinvestment (keep the profit)\n")
    print(f"  funded (of 30 evals): {nf.mean():.0f}   reached buffer & de-risked: {rc.mean():.0f}   paid >=1: {npd.mean():.0f}")
    print(f"  total payouts: ~{npay.mean():.0f}\n")
    print(f"  GROSS take-home: mean ${take.mean():,.0f}  median ${np.median(take):,.0f}  "
          f"p25 ${np.percentile(take,25):,.0f}  p10 ${np.percentile(take,10):,.0f}")
    print(f"  NET (minus $3k evals): mean ${net.mean():,.0f}  median ${np.median(net):,.0f}  "
          f"p25 ${np.percentile(net,25):,.0f}  P(net<=0) {np.mean(net<=0)*100:.0f}%")
    print(f"\n  This is ONE cohort (plays out in ~3 months, then idle).")
    print(f"  If you re-buy $3k each time it retires (~3-4x/yr): annual ~${net.mean()*3.5:,.0f} net "
          f"(spending ~${3000*3.5:,.0f}/yr on evals).")


if __name__ == "__main__":
    main()

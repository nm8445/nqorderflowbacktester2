"""Lane frontier under the CORRECTED fully-MAE model (2026-07-10): how income variance (CV) and
P(losing year) scale with the number of decorrelated lanes at a FIXED 30 accounts. Re-fits
CV(L) ~ a + b/sqrt(L) for the calculator -- the old 0.26 + 0.51/sqrt(L) was fit to the optimistic
43%-payout MC; the lower 34% pass / 33% payout rates make each lane lumpier, so CV shifts UP and
extra lanes buy more decorrelation. Reuses a_vs_b.funded() (MAE gamble + milk).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scripts.futurespropmc.challenge_gamble_pass import build_outcomes
from farm_income_mc import load_daily_1mnq_mae
from a_vs_b import funded

ACCOUNTS = 30; SIG = 3.04; FEE = 100.0; PASS = 0.34; TDY = 251
STRUCT = {"A": dict(gtr=11, milk=15, cap=2), "B": dict(gtr=8, milk=20, cap=2)}
LANES = [1, 2, 3, 5, 6, 10, 15]


def cash_pool(struct, pool, d1, cap, n=200_000, seed=3):
    """i.i.d. sample of $ pocketed per FUNDED account (0 for the ~66% that blow)."""
    rng = np.random.default_rng(seed)
    return np.array([funded(struct, pool, d1, rng, cap)[0] for _ in range(n)])


def frontier(cp, gtr, milk, n_years=6000, seed=9):
    rng = np.random.default_rng(seed)
    res = {}
    for L in LANES:
        firms = ACCOUNTS / L                      # correlated copies per lane (firm diversification)
        cyc_days = gtr * (L / SIG) + milk         # throughput cycle (more lanes -> longer)
        nc = max(1, int(round(TDY / cyc_days)))
        yrs = np.empty(n_years)
        for y in range(n_years):
            tot = 0.0
            for _ in range(nc):
                passed = rng.random(L) < PASS                 # eval pass per lane (correlated firms)
                cash = cp[rng.integers(cp.size, size=L)]      # funded $ per lane (one draw x firms)
                tot += np.sum(firms * np.where(passed, cash, 0.0)) - firms * L * FEE
            yrs[y] = tot
        res[L] = dict(mean=yrs.mean(), cv=yrs.std() / yrs.mean() if yrs.mean() else float("nan"),
                      ploss=(yrs < 0).mean(), p10=np.percentile(yrs, 10),
                      p90=np.percentile(yrs, 90), cyc_yr=TDY / cyc_days)
    return res


def fit_cv(res):
    """Least-squares CV(L) = a + b / sqrt(L)."""
    L = np.array(LANES, float); y = np.array([res[k]["cv"] for k in LANES])
    x = 1.0 / np.sqrt(L)
    b, a = np.polyfit(x, y, 1)
    return a, b


def main():
    df = build_outcomes(1500., 1500.); pool = df[["g", "mae"]].values
    d1 = load_daily_1mnq_mae()
    for s in ("A", "B"):
        c = STRUCT[s]
        cp = cash_pool(s, pool, d1, c["cap"])
        res = frontier(cp, c["gtr"], c["milk"])
        a, b = fit_cv(res)
        print(f"\n=== Structure {s} (cap {c['cap']}, E[$/funded]=${cp.mean():,.0f}) ===")
        print(f"  {'lanes':>5} {'copies':>6} {'cyc/yr':>6} {'mean$':>10} {'p10':>10} {'p90':>10} "
              f"{'CV':>5} {'P(loss)':>8}")
        for L in LANES:
            r = res[L]
            print(f"  {L:>5} {ACCOUNTS/L:>6.1f} {r['cyc_yr']:>6.1f} {r['mean']:>10,.0f} {r['p10']:>10,.0f} "
                  f"{r['p90']:>10,.0f} {r['cv']:>5.2f} {r['ploss']*100:>7.1f}%")
        print(f"  FIT: CV(L) = {a:.2f} + {b:.2f}/sqrt(L)   (old calc: 0.26 + 0.51/sqrt(L))")


if __name__ == "__main__":
    main()

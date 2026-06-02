"""Sweep the per-trade RISK used in the funded reach-$3k phase.

At ~$1k risk a 2-loss day busts a fresh funded account (-$2k = floor), so only ~27%
of funded accounts reach their first payout. Lowering reach-phase risk survives 2-loss
days but reaches +$3k slower (more cumulative blow exposure). This finds the sweet spot.

For each reach-risk level we scale the eval-bracket per-trade pnl linearly (win/loss both
scale with position size), run the two-phase funded sim, and report reach-$3k rate,
$/funded acct, payouts/acct, and resulting net $/cycle (using the RV+FB+MEC challenge
E[#passed]).

Run:  python -m scripts.propfirm_milking.sweep_funded_risk
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import scripts.propfirm_milking.mc_portfolio as M
from scripts.propfirm_milking.common import RESULTS_DIR

STRATS = ["RV", "FB", "MEC"]
REACH_RISK = [1000, 900, 800, 700, 600, 500, 400, 300]   # $ per trade
N_FUNDED = 30_000
N_CHAL = 4_000
ACCOUNT_COST = M.ACCOUNT_COST
N_ACCTS = M.N_ACCTS


def _init(chal_days, funded_daily):
    M._CHAL_DAYS = chal_days
    M._FUNDED_DAILY = funded_daily


def _sweep_risk(risk):
    scale = risk / 1000.0
    rng = np.random.default_rng(1000 + risk)
    withdrawn = np.empty(N_FUNDED)
    reached = 0
    payouts = 0
    for i in range(N_FUNDED):
        w, p, r = M.run_funded(rng, reach_scale=scale)
        withdrawn[i] = w
        reached += r
        payouts += p
    return {
        "reach_risk_$": risk,
        "reach_rate": round(reached / N_FUNDED, 4),
        "$/funded_acct": round(float(withdrawn.mean()), 0),
        "payouts/acct": round(payouts / N_FUNDED, 3),
        "median_$ (reachers)": round(float(np.median(withdrawn[withdrawn > 0])) if (withdrawn > 0).any() else 0, 0),
    }


def main():
    pd.set_option("display.width", 200)
    print("Building day-packs (RV+FB+MEC) + funded daily...")
    chal_days = M.build_challenge_day_packs(STRATS)
    funded_daily = M.build_funded_daily()
    _init(chal_days, funded_daily)

    # Challenge E[#passed of 30] for RV+FB+MEC (for net/cycle)
    rng = np.random.default_rng(7)
    n_passed = np.mean([M.run_challenge(rng)[0] for _ in range(N_CHAL)])
    print(f"  challenge E[#passed of 30] = {n_passed:.2f}\n")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(8, len(REACH_RISK)),
                             initializer=_init, initargs=(chal_days, funded_daily)) as ex:
        rows = list(ex.map(_sweep_risk, REACH_RISK))
    df = pd.DataFrame(rows)
    df["E[#passed]"] = round(n_passed, 2)
    df["E[net $/cycle]"] = (df["$/funded_acct"] * n_passed - N_ACCTS * ACCOUNT_COST).round(0)
    df["cycle_mult"] = (df["$/funded_acct"] * n_passed / (N_ACCTS * ACCOUNT_COST)).round(2)
    df.to_csv(RESULTS_DIR / "funded_risk_sweep.csv", index=False)
    print(f"done in {time.time()-t0:.1f}s\n")
    print("=== FUNDED REACH-$3k RISK SWEEP (RV+FB+MEC) ===")
    print(df.to_string(index=False))
    best = df.loc[df["E[net $/cycle]"].idxmax()]
    print(f"\nBest net/cycle at reach-risk ${int(best['reach_risk_$'])}: "
          f"reach {best['reach_rate']:.0%}, ${best['$/funded_acct']:,.0f}/acct, "
          f"net ${best['E[net $/cycle]']:,.0f} ({best['cycle_mult']}x)")


if __name__ == "__main__":
    main()

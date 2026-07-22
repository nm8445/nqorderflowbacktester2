"""Funded gamble: optimal (target T, risk R, RR) for $ EXTRACTED per funded. NO consistency rule on
funded (only evals) -> the only goal is reach T before the trailing $2k floor, then milk 5 winning days
and withdraw half (T/2), capped at 2 payouts. WR per RR from the REAL combined brackets.

Win = R*RR. A win/loss is one winning/losing day. Floor = min(0, peak-2000) (trails, locks at 0 once
peak>=2000). Bigger R blows faster pre-lock (a single -R can breach -2000). Milk = real 1-MNQ daily.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scripts.propfirm_milking.common import load_1min_bars
from funded_rr_sweep import build_all_packs, combined_R
from farm_income_mc import _milk, load_daily_1mnq, SPLIT, MIN_WD, WINS_NEEDED, MAX_WD

R_GRID = [1000, 1500, 2000]
RR_GRID = [1.0, 1.33, 1.5, 2.0]
T_GRID = [3000, 4000]


def _gamble(pool_d, target, rng, locked, start=0.0):
    """Random-walk `pool_d` ($-per-trade samples) from `start` to +target before the floor.
    Returns (reached_profit|None, winning_days)."""
    p = start; peak = max(start, 0.0); wins = 0
    for _ in range(300):
        step = pool_d[rng.integers(pool_d.size)]
        p += step
        if step > 0:
            wins += 1
        peak = max(peak, p)
        floor = 0.0 if locked else min(0.0, peak - 2000.0)
        if p <= floor + 1e-9:
            return None, wins
        if p >= target:
            return p, wins
    return None, wins


def funded_life(pool_d, target, daily, rng):
    """Structure A (re-gamble), 2-payout cap. Returns $ the user pockets (after split)."""
    p, gw = _gamble(pool_d, target, rng, locked=False)
    if p is None:
        return 0.0
    cash = 0.0; nwd = 0
    while nwd < MAX_WD:
        if nwd > 0:
            p, gw = _gamble(pool_d, target, rng, locked=True, start=p)
            if p is None:
                return cash
        need = max(0, WINS_NEEDED - gw)
        if need > 0:
            p, _ = _milk(p, need, daily, rng, floor=0.0)
            if p is None:
                return cash
        w = p / 2.0
        if w < MIN_WD:
            return cash
        cash += w * SPLIT; p -= w; nwd += 1
    return cash


def main():
    df1m = load_1min_bars()
    packs = build_all_packs(df1m)
    daily = load_daily_1mnq()
    pools = {rr: combined_R(packs, rr) for rr in RR_GRID}     # R-units; scale by risk below
    print("Funded extraction sweep (structure A, 2-payout cap, no consistency). E[$/funded] = user take.\n")
    print(f"  {'T':>5} {'risk':>5} {'RR':>5} {'win$':>5} {'WR':>5} {'E[$/funded]':>12} {'reach%':>7}")
    best = (None, -1)
    for T in T_GRID:
        for R in R_GRID:
            for rr in RR_GRID:
                pool_d = pools[rr] * R
                rng = np.random.default_rng(5)
                vals = np.empty(40000)
                reach = 0
                for i in range(vals.size):
                    # reach-flag via a quick standalone gamble (reuse first leg)
                    vals[i] = funded_life(pool_d, T, daily, rng)
                ef = vals.mean()
                # reach% (first-leg) — separate quick estimate
                rng2 = np.random.default_rng(9)
                rr_reach = np.mean([_gamble(pool_d, T, rng2, locked=False)[0] is not None for _ in range(20000)])
                wr = (pools[rr] > 0).mean() * 100
                tag = "  <-" if ef > best[1] else ""
                if ef > best[1]:
                    best = ((T, R, rr), ef)
                print(f"  {T:>5} {R:>5} {rr:>5.2f} {R*rr:>5.0f} {wr:>4.1f}% {ef:>12.0f} {rr_reach*100:>6.1f}%{tag}")
    (T, R, rr), ef = best
    print(f"\n  OPTIMAL: target ${T}, risk ${R}, RR {rr} (win ${R*rr:.0f}) -> E[$/funded] ${ef:.0f}")


if __name__ == "__main__":
    main()

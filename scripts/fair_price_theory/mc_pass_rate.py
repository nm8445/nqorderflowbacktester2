"""Fair Price Theory -> 50k futures eval pass-rate Monte Carlo.

50k eval: $2k trailing-then-lock DD (floor=min(50000,max(48000,peak-2000))), +$3k target,
floating-blowable, no martingale.  Sized risk-normalized: each trade risks $R (contracts =
R/(SL_pts*$2)), so worst floating per trade = ~1R (hard SL) -> clean floating check.

Per-trade pnl in R-units = pnl_pts / SL_pts (SL=-1R, TP=+1.52R).  Costs = contracts*$4 RT.
Day-packs from the single-account serial trade logs (1 position at a time).  Sweeps R and mode.

Run (after fair_price_strategy.py builds the edge):  python scripts/fair_price_theory/mc_pass_rate.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from fair_price_strategy import run    # same folder

MNQ_PT, COST = 2.0, 4.0                 # $/pt, $/contract RT
START, DD, LOCK, TARGET, CAP = 50_000., 2000., 50_000., 3000., 504
N = 30_000


def day_packs(t: pd.DataFrame):
    """date -> list of (pnl_R, sl_pts) for that day's serial trades."""
    t = t.sort_values(["date", "fill_idx"])
    packs = []
    for _, g in t.groupby("date", sort=True):
        packs.append(list(zip((g["pnl_pts"] / g["sl"]).values, g["sl"].values)))
    return packs


def sim(packs, R, rng):
    bal = START; peak = START; floor = START - DD; n = len(packs)
    for d in range(CAP):
        real = 0.; bust = False
        for pnl_r, sl in packs[rng.integers(0, n)]:
            contracts = max(1, round(R / (sl * MNQ_PT)))
            cost = contracts * COST
            worst = -R - cost                      # floating bottoms at ~ -1R (hard SL) + cost
            if bal + real + worst < floor:
                bust = True; break
            real += pnl_r * R - cost
        if bust:
            return "bust", d + 1
        bal += real
        if bal - START >= TARGET:
            return "pass", d + 1
        if bal > peak: peak = bal
        floor = min(LOCK, max(START - DD, peak - DD))
    return "timeout", CAP


def main():
    print("Fair Price Theory -> 50k futures eval ($2k trailing-lock, $3k target, floating, marti off)\n")
    print(f"{'mode':>10} {'R':>6} {'pass%':>7} {'bust%':>7} {'to%':>6} {'med d':>6} {'p90 d':>6}")
    for mode in ("cont_only", "candle", "bos", "combined"):
        t = run(mode)
        packs = day_packs(t)
        for R in (500, 750, 1000, 1500):
            rng = np.random.default_rng(7)
            outs, dys = [], []
            for _ in range(N):
                o, d = sim(packs, R, rng); outs.append(o); dys.append(d)
            outs = np.array(outs); dys = np.array(dys)
            pa = outs == "pass"; bu = outs == "bust"; to = outs == "timeout"
            pdd = dys[pa]
            med = int(np.median(pdd)) if pa.any() else 0
            p90 = int(np.percentile(pdd, 90)) if pa.any() else 0
            print(f"{mode:>10} {R:>6} {pa.mean()*100:6.1f}% {bu.mean()*100:6.1f}% "
                  f"{to.mean()*100:5.1f}% {med:>6} {p90:>6}")
        print()


if __name__ == "__main__":
    main()

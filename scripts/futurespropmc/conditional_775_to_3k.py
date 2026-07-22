"""One-off: currently +$775 profit on the MFF funded, DD still $2k (NOT locked yet).
P(reach +$3k profit) before breaching the $2k DD, + days.

MFF DD = EOD-trailing: floor = min(start+$100, high-water-EOD-balance - $2000). Trails up at each EOD
new high, LOCKS at +$100 once you've closed a day >= +$2,100. Intraday floating-blowable (1-min MAE
vs the frozen floor). 4-way combined (OD+RV+B2+FB), martingale OFF. Data: combined_4way_with_mae_1min.csv.

Profit terms (start balance = 0): start +$775, target +$3000, floor starts at 775 - 2000 = -$1225.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parent / "results" / "combined_4way_with_mae_1min.csv"
START, TARGET, DD, LOCK, COST, CAP, N = 775., 3000., 2000., 100., 2.0, 504, 50000


def packs():
    df = pd.read_csv(CSV).sort_values("ts")
    return [list(zip(g["pnl_1c"], g["mae_1c"])) for _, g in df.groupby("date", sort=True)]


def sim(P, rng, mnq):
    s = mnq / 10.; bal = START; hw = START; floor = min(LOCK, hw - DD); n = len(P)
    for d in range(CAP):
        tr = P[rng.integers(0, n)]; real = 0.
        for p, m in tr:                          # m = worst floating (1NQ, negative)
            if bal + real + (m * s - COST * mnq) < floor:   # intraday equity breaches the frozen floor
                return "bust", d + 1
            real += p * s - COST * mnq
        bal += real
        if bal >= TARGET:
            return "pass", d + 1
        if bal > hw:                             # EOD ratchet (trails up only; locks at +$100)
            hw = bal
            floor = min(LOCK, hw - DD)
    return "timeout", CAP


def main():
    P = packs()
    print(f"From +${START:,.0f} -> +${TARGET:,.0f}  |  $2k EOD-trailing DD (locks at +$100)  |  "
          f"4-way, marti OFF  |  {N:,} sims\n")
    print(f"{'size':>6} {'PASS':>7} {'BUST':>7} {'timeout':>8} {'mean d':>8} {'median d':>9} {'p90 d':>7}")
    for mnq in (1, 2, 3):
        rng = np.random.default_rng(7)
        outs, dys = [], []
        for _ in range(N):
            o, d = sim(P, rng, mnq); outs.append(o); dys.append(d)
        outs = np.array(outs); dys = np.array(dys)
        pa = outs == "pass"; bu = outs == "bust"; to = outs == "timeout"
        pd_ = dys[pa] if pa.any() else np.array([np.nan])
        print(f"{mnq:>5}M {pa.mean()*100:6.1f}% {bu.mean()*100:6.1f}% {to.mean()*100:7.1f}% "
              f"{pd_.mean():8.1f} {np.median(pd_):9.0f} {np.percentile(pd_,90):7.0f}")


if __name__ == "__main__":
    main()

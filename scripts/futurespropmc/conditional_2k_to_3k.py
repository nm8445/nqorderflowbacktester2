"""Conditional MC: already +$2k on a 50k futures eval -> P(reach +$3k) without busting, + days.

Rule (user's firm): once up $2k the trailing floor LOCKS at $50k (start balance) and stops
trailing, so the $ cushion just grows with profit. So from here: balance starts $52k,
floor fixed $50k, target $53k, floating-blowable (intraday 1-min MAE checked vs floor),
4-way combined (OD+RV+B2+FB), martingale OFF.  Data: combined_4way_with_mae_1min.csv.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parent / "results" / "combined_4way_with_mae_1min.csv"
START, FLOOR, TARGET, COST, CAP, N = 52000., 50000., 53000., 2.0, 504, 50000


def packs():
    df = pd.read_csv(CSV).sort_values("ts")
    return [list(zip(g["pnl_1c"], g["mae_1c"])) for _, g in df.groupby("date", sort=True)]


def sim(P, rng, mnq):
    s = mnq / 10.; bal = START; n = len(P)
    for d in range(CAP):
        tr = P[rng.integers(0, n)]; real = 0.
        for p, m in tr:                      # m = worst floating (1NQ, negative); check each trade
            if bal + real + (m * s - COST * mnq) < FLOOR:
                return "bust", d + 1
            real += p * s - COST * mnq
        bal += real
        if bal >= TARGET:
            return "pass", d + 1
    return "timeout", CAP


def main():
    P = packs()
    print(f"conditional: start ${START:,.0f}  floor ${FLOOR:,.0f} (locked)  target ${TARGET:,.0f}  "
          f"marti OFF  |  {N:,} sims\n")
    print(f"{'size':>6} {'PASS':>7} {'BUST':>7} {'timeout':>8} {'mean d':>8} {'median d':>9} {'p90 d':>7}")
    for mnq in (1, 2):
        rng = np.random.default_rng(7)
        outs, dys = [], []
        for _ in range(N):
            o, d = sim(P, rng, mnq); outs.append(o); dys.append(d)
        outs = np.array(outs); dys = np.array(dys)
        pa = outs == "pass"; bu = outs == "bust"; to = outs == "timeout"
        pd_ = dys[pa]
        print(f"{mnq:>5}M {pa.mean()*100:6.1f}% {bu.mean()*100:6.1f}% {to.mean()*100:7.1f}% "
              f"{pd_.mean():8.1f} {np.median(pd_):9.0f} {np.percentile(pd_,90):7.0f}")


if __name__ == "__main__":
    main()

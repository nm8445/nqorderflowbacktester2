"""Futures 50k ($2k DD) vs 150k ($4.5k DD) funded milking + challenge — does the bigger DD fix the
floating-blow problem that made 50k a burn-and-churn lane?

Both: trailing DD that LOCKS at start once you've banked +DD profit; floating-blowable below floor;
no RPTI / no daily limit. Milk = bank-to-lock, then withdraw down to start+cushion every 10 td (90%).
4-way combined day-packs. Run: python scripts/montecarlo/futures_50k_vs_150k.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parent / "results" / "combined_4way_with_mae_1min.csv"
COST, CYCLE, SPLIT, HORIZON, N = 2.0, 10, 0.90, 252, 20_000

ACCTS = {  # name: (start, dd, eval_target, eval_cost)
    "50k ($2.0k DD)":  (50_000.,  2000., 3000., 150.),
    "150k ($4.5k DD)": (150_000., 4500., 9000., 300.),
}


def packs():
    df = pd.read_csv(CSV).sort_values("ts")
    return [list(zip(g["pnl_1c"].astype(float), (-g["mae_1c"]).astype(float)))
            for _, g in df.groupby("date", sort=True)]


def sim_funded(P, rng, start, dd, mnq, cushion):
    s = mnq / 10.; n = len(P); bal = start; peak = start; floor = start - dd
    locked = False; dic = 0; cash = 0.; pays = 0
    for d in range(HORIZON):
        real = 0.; bust = False
        for p, m in P[rng.integers(0, n)]:
            if bal + real - m * s <= floor: bust = True; break
            real += p * s - mnq * COST
        if bust: return cash, pays, True
        bal += real
        if bal > peak: peak = bal
        if not locked:
            floor = min(start, peak - dd)
            if floor >= start: locked = True; floor = start
        dic += 1
        if dic >= CYCLE:
            pay = bal - (start + cushion)
            if pay >= 200.: cash += pay * SPLIT; pays += 1; bal -= pay
            dic = 0
    return cash, pays, False


def sim_challenge(P, rng, start, dd, target, mnq, max_days=200):
    s = mnq / 10.; n = len(P); bal = start; peak = start; floor = start - dd; goal = start + target
    for d in range(max_days):
        real = 0.; bust = False
        for p, m in P[rng.integers(0, n)]:
            if bal + real - m * s <= floor: bust = True; break
            real += p * s - mnq * COST
        if bust: return False, d + 1
        bal += real
        if bal > peak: peak = bal
        floor = min(start, peak - dd)
        if bal >= goal: return True, d + 1
    return False, max_days


def main():
    P = packs()
    print("FUNDED milking (bank-to-lock then withdraw to start+cushion, 90% split, /10td):\n")
    print(f"{'account':>16} {'mnq':>4} {'cushion':>8} {'net$/yr':>9} {'payouts':>8} {'blow%':>7}")
    plans = {"50k ($2.0k DD)": [(1, 1000), (2, 1000)],
             "150k ($4.5k DD)": [(1, 2000), (2, 2500), (3, 3000)]}
    for name, (start, dd, _, _) in ACCTS.items():
        for mnq, cush in plans[name]:
            rng = np.random.default_rng(7)
            r = [sim_funded(P, rng, start, dd, mnq, cush) for _ in range(N)]
            print(f"{name:>16} {mnq:>4} {cush:>8} {np.mean([x[0] for x in r]):>9.0f} "
                  f"{np.mean([x[1] for x in r]):>8.1f} {100*np.mean([x[2] for x in r]):>6.1f}%")
    print(f"\nCHALLENGE pass (combined edge, swept size; futures = no RPTI):")
    print(f"{'account':>16} {'mnq':>4} {'pass%':>7} {'med days':>9} {'$/funded':>9}")
    for name, (start, dd, target, cost) in ACCTS.items():
        for mnq in ([1, 2] if start < 1e5 else [1, 2, 3]):
            rng = np.random.default_rng(7)
            r = [sim_challenge(P, rng, start, dd, target, mnq) for _ in range(N)]
            pa = np.mean([x[0] for x in r]); dys = [x[1] for x in r if x[0]]
            med = int(np.median(dys)) if dys else 999
            print(f"{name:>16} {mnq:>4} {pa*100:>6.1f}% {med:>9} {cost/pa if pa else 0:>9.0f}")


if __name__ == "__main__":
    main()

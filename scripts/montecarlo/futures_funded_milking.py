"""Futures 50k FUNDED milking mechanics — how to actually extract payouts without blowing.

Rule: $2k trailing DD that LOCKS at $50k once you're +$2k (floor stays $50k forever after, even
through withdrawals). Floating-blowable below the floor. No RPTI, no daily limit.

Tests two things that determine durability:
  (1) WITHDRAWAL CUSHION: after locking, milk down to $50k+cushion (not all the way to $50k).
      Leaving a cushion = a normal floating day can't punch the floor.
  (2) the lock itself: once locked the floor never re-arms (the earlier farm sim re-armed it -> too
      harsh).  Here it's fixed correctly.

4-way combined day-packs, 1 MNQ.  Run: python scripts/montecarlo/futures_funded_milking.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parent / "results" / "combined_4way_with_mae_1min.csv"
START, DD, LOCK, COST, CYCLE, SPLIT = 50_000., 2000., 50_000., 2.0, 10, 0.90
HORIZON, N = 252, 20_000


def packs():
    df = pd.read_csv(CSV).sort_values("ts")
    return [list(zip(g["pnl_1c"].astype(float), (-g["mae_1c"]).astype(float)))
            for _, g in df.groupby("date", sort=True)]


def sim(P, rng, mnq, cushion, bank_first):
    """bank_first: hold trades until floor is locked before allowing any withdrawal."""
    s = mnq / 10.; n = len(P); bal = START; peak = START; floor = START - DD
    locked = False; dic = 0; cash = 0.; pays = 0
    for d in range(HORIZON):
        real = 0.; bust = False
        for p, m in P[rng.integers(0, n)]:
            flo = m * s
            if bal + real - flo <= floor: bust = True; break
            real += p * s - mnq * COST
        if bust: return cash, pays, True, d + 1
        bal += real
        if bal > peak: peak = bal
        if not locked:
            floor = min(LOCK, peak - DD)
            if floor >= LOCK: locked = True; floor = LOCK
        dic += 1
        if dic >= CYCLE:
            can_withdraw = locked or not bank_first
            if can_withdraw:
                pay = bal - (START + cushion)
                if pay >= 200.:
                    cash += pay * SPLIT; pays += 1; bal -= pay   # floor stays locked
            dic = 0
    return cash, pays, False, HORIZON


def main():
    P = packs()
    print("Futures 50k funded milking (1 MNQ, $2k trailing-then-LOCK @50k, 90% split, payout/10td)\n")
    print(f"{'mode':>26} {'net$/yr':>9} {'payouts':>8} {'blow%':>7}")
    configs = [
        ("milk to floor (cushion $0)", 0,    False),
        ("leave $500 cushion",          500,  False),
        ("leave $1000 cushion",         1000, False),
        ("leave $1500 cushion",         1500, False),
        ("bank-to-lock, then $1k cush", 1000, True),
        ("bank-to-lock, then $1.5k",    1500, True),
    ]
    for label, cush, bank in configs:
        rng = np.random.default_rng(7)
        r = [sim(P, rng, 1, cush, bank) for _ in range(N)]
        cash = np.mean([x[0] for x in r]); pays = np.mean([x[1] for x in r])
        blow = np.mean([x[2] for x in r])
        print(f"{label:>26} {cash:>9.0f} {pays:>8.1f} {blow*100:>6.1f}%")


if __name__ == "__main__":
    main()

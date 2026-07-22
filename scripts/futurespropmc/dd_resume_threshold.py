"""Two-tier eval sizing: risk $750 while tight, resume $1500 at DD >= T. Find the best T.

Question: an account sitting at $350-650 DD risks $750; at what remaining DD is it safe/optimal to
go back to full $1500? Key mechanic: at $1500 risk a full stop-out is -$1500, so you must have
DD > $1500 to survive it, and enough extra that the post-loss DD drops back into the $750 zone
(not straight to a blow). MAE-aware; floor frozen at $48k below the $50k peak. No marti, decorrelated.

Run: python scripts/futurespropmc/dd_resume_threshold.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from challenge_gamble_pass import build_outcomes   # noqa: E402

START, DD, TARGET = 50000.0, 2000.0, 3000.0
R_MIN = 100.0
N = 40000
HORIZON = 250
TDY = 251.0


def sim(R, maeR, rng, T, start_bal, tight=750.0, full=1500.0):
    bal = start_bal; peak = START; floor = min(START, peak - DD)
    for d in range(HORIZON):
        dd = bal - floor
        if dd <= 0:
            return 0, d
        risk = full if dd >= T else tight
        risk = max(R_MIN, min(1500.0, risk))
        i = rng.integers(R.size)
        if bal + maeR[i] * risk <= floor:
            return 0, d + 1
        bal += R[i] * risk
        if bal <= floor:
            return 0, d + 1
        if bal - START >= TARGET:
            return 1, d + 1
        if bal > peak:
            peak = bal
        floor = min(START, peak - DD)
    return 0, HORIZON


def stat(R, maeR, T, start_bal):
    rng = np.random.default_rng(7)
    res = [sim(R, maeR, rng, T, start_bal) for _ in range(N)]
    passed = np.array([r[0] for r in res]); days = np.array([r[1] for r in res])
    p = passed.mean()
    med = int(np.median(days[passed == 1])) if passed.any() else 0
    mt = days.mean()
    thru = p * (TDY / mt) if mt > 0 else 0.0
    return p, med, thru


def main():
    df = build_outcomes(1500.0, 1500.0)
    R = (df.g / 1500.0).values
    maeR = (df.mae / 1500.0).values
    print(f"pool n={R.size}  WR {(R>0).mean()*100:.1f}%   rule: risk $750 if DD<T else $1500\n")
    for dd_now in (350.0, 500.0, 650.0):
        sb = 48000.0 + dd_now
        print(f"=== starting DD ${dd_now:.0f} (bal ${sb:,.0f}) ===")
        print(f"{'resume T':>9} | {'pass':>6} {'med d->pass':>11} {'passes/yr/slot':>14}")
        # include 'always $750' and 'always $1500' as bookends
        p, med, thru = stat(R, maeR, 1e9, sb)
        print(f"{'$750 all':>9} | {p*100:>5.1f}% {med:>11} {thru:>14.1f}")
        for T in (1200, 1400, 1600, 1800, 2000, 2200):
            p, med, thru = stat(R, maeR, float(T), sb)
            print(f"{('$'+str(T)):>9} | {p*100:>5.1f}% {med:>11} {thru:>14.1f}")
        p, med, thru = stat(R, maeR, 0.0, sb)
        print(f"{'$1500 all':>9} | {p*100:>5.1f}% {med:>11} {thru:>14.1f}")
        print()


if __name__ == "__main__":
    main()

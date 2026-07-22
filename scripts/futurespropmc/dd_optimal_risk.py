"""Optimal DD->risk schedule for the $1500 1:1 eval (decorrelated, 1 signal/account, no marti).

Answers three things directly:
  1. optimal risk for an account sitting at tight DD ($350-$650),
  2. the DD level at which you resume full $1500,
  3. the pass rate for an account that just lost its first ~$1500 trade, under the optimal schedule.

MAE-aware (bal + worst_floating_$ <= floor => blown mid-trade). Below the $50k peak the floor is
FROZEN at $48k, so small wins GROW the DD cushion. Pool = real 1-min first-touch R/mae_R from
challenge_gamble_pass (all 4 strats, no martingale), sized in R so it's scale-free.

Run: python scripts/futurespropmc/dd_optimal_risk.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from challenge_gamble_pass import build_outcomes   # noqa: E402

START, DD, TARGET = 50000.0, 2000.0, 3000.0
R_MIN = 100.0          # 1-MNQ minimum risk
N = 40000
HORIZON = 120          # ~6 months at 1 decorrelated trade/day (evals have no hard limit; long = generous)


def sim(R, maeR, rng, policy, start_bal, start_peak=START, max_days=HORIZON):
    bal = start_bal; peak = start_peak; floor = min(START, peak - DD)
    for d in range(max_days):
        dd = bal - floor
        if dd <= 0:
            return 0, d, "blow"
        risk = max(R_MIN, min(1500.0, policy(dd)))
        i = rng.integers(R.size)
        if bal + maeR[i] * risk <= floor:
            return 0, d + 1, "blow"
        bal += R[i] * risk
        if bal <= floor:
            return 0, d + 1, "blow"
        if bal - START >= TARGET:
            return 1, d + 1, "pass"
        if bal > peak:
            peak = bal
        floor = min(START, peak - DD)
    return 0, max_days, "timeout"


def stat(R, maeR, policy, start_bal, start_peak=START):
    rng = np.random.default_rng(7)
    res = [sim(R, maeR, rng, policy, start_bal, start_peak) for _ in range(N)]
    p = np.mean([r[0] for r in res])
    outs = np.array([r[2] for r in res]); dys = np.array([r[1] for r in res])
    pd_ = dys[outs == "pass"]
    return p, np.mean(outs == "blow"), np.mean(outs == "timeout"), (int(np.median(pd_)) if pd_.size else 0)


def main():
    df = build_outcomes(1500.0, 1500.0)
    R = (df.g / 1500.0).values
    maeR = (df.mae / 1500.0).values
    print(f"pool n={R.size}  avg R {R.mean():+.3f}  WR {(R>0).mean()*100:.1f}%  horizon {HORIZON} trades\n")

    fresh = START                         # $2000 DD
    loss = START - 1500.0                 # bal $48,500 after a first ~$1500 loss -> DD $500

    # ---- 1. FIXED risk sweep: what single risk is best from a fresh-loss ($500 DD) account? ----
    print("=== Fixed risk (constant, no DD-awareness) ===")
    print(f"{'risk$':>6} | {'fresh-loss($500 DD) pass':>26} {'blow':>6} {'med d':>6} | {'fresh pass':>11} {'blow':>6}")
    for r in (100, 200, 300, 500, 750, 1000, 1500):
        pl, bl, tl, dl = stat(R, maeR, (lambda rr: (lambda dd: rr))(r), loss)
        pf, bf, tf, dfd = stat(R, maeR, (lambda rr: (lambda dd: rr))(r), fresh)
        print(f"{r:>6} | {pl*100:>17.1f}% {' ':>7} {bl*100:>5.0f}% {dl:>6} | {pf*100:>10.1f}% {bf*100:>5.0f}%")

    # ---- 2. Buffer schedule risk = clip(DD - B, R_MIN, 1500): sweep the reserve B ----
    print("\n=== DD schedule: risk = clip(remaining_DD - B, $100, $1500) — sweep reserve B ===")
    print(f"{'B$':>6} {'resume$1500 at DD':>18} | {'fresh-loss pass':>16} {'blow':>6} {'med d':>6} | {'fresh pass':>11} {'timeout':>8}")
    best = None
    for B in (0, 200, 400, 600, 800, 1000, 1200):
        pol = (lambda bb: (lambda dd: dd - bb))(B)
        pl, bl, tl, dl = stat(R, maeR, pol, loss)
        pf, bf, tf, dfd = stat(R, maeR, pol, fresh)
        resume = 1500 + B
        print(f"{B:>6} {resume:>16} | {pl*100:>13.1f}% {' ':>2} {bl*100:>5.0f}% {dl:>6} | {pf*100:>10.1f}% {tf*100:>7.1f}%")
        if best is None or pl > best[1]:
            best = (B, pl, resume)

    Bopt, popt, resume = best
    print(f"\nBEST reserve B=${Bopt} (fresh-loss pass {popt*100:.1f}%). Resulting DD->risk schedule:")
    print(f"{'remaining DD':>13} -> {'risk':>7}")
    for dd in (350, 500, 650, 1000, 1500, 2000, 2500, 3000):
        risk = max(R_MIN, min(1500.0, dd - Bopt))
        tag = "  <- resume full $1500" if abs(risk - 1500.0) < 1e-9 and dd - 100 < resume <= dd else ""
        print(f"{dd:>11}$ -> {risk:>6.0f}${tag}")
    print(f"\n=> Resume full $1500 once remaining DD >= ${resume}.")


if __name__ == "__main__":
    main()

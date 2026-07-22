"""Optimal risk for SPEED (throughput), not just pass rate. $1500 1:1 eval, decorrelated, no marti.

Throughput = passes per YEAR per account slot = pass_rate * (251 / mean_days_to_terminate).
A blow/timeout frees the slot to re-buy an eval, so fast-resolving policies cycle more attempts.
Also reports evals-per-pass (= 1/pass_rate = fee efficiency). Speed favors big risk; fees favor small.

Tests fixed risks AND proportional risk = c*DD (c~1 = 'stop at the floor, only die on a real stop-out').
MAE-aware; below the $50k peak the floor is frozen at $48k so wins grow the cushion.
Pool = real 1-min first-touch R/mae_R (dynamic ATR SL/TP already baked in), no martingale.

Run: python scripts/futurespropmc/dd_speed_risk.py
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
TDY = 251.0        # trading days/yr


def sim(R, maeR, rng, policy, start_bal, start_peak=START, max_days=HORIZON):
    bal = start_bal; peak = start_peak; floor = min(START, peak - DD)
    for d in range(max_days):
        dd = bal - floor
        if dd <= 0:
            return 0, d
        risk = max(R_MIN, min(1500.0, policy(dd)))
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
    return 0, max_days


def stat(R, maeR, policy, start_bal):
    rng = np.random.default_rng(7)
    res = [sim(R, maeR, rng, policy, start_bal) for _ in range(N)]
    passed = np.array([r[0] for r in res]); days = np.array([r[1] for r in res])
    p = passed.mean()
    mean_term = days.mean()                       # days to pass OR blow OR timeout (slot occupied)
    med_pass = int(np.median(days[passed == 1])) if passed.any() else 0
    throughput = p * (TDY / mean_term) if mean_term > 0 else 0.0   # passes/yr/slot
    evals_per_pass = (1.0 / p) if p > 0 else float("inf")
    return p, med_pass, mean_term, throughput, evals_per_pass


def block(R, maeR, label, start_bal):
    print(f"\n=== {label} ===")
    print(f"{'policy':>16} | {'pass':>6} {'med d->pass':>11} {'avg d resolve':>13} "
          f"{'passes/yr/slot':>14} {'evals/pass':>10}")
    policies = [
        ("fixed $200", lambda dd: 200.0), ("fixed $300", lambda dd: 300.0),
        ("fixed $500", lambda dd: 500.0), ("fixed $750", lambda dd: 750.0),
        ("fixed $1000", lambda dd: 1000.0), ("fixed $1500", lambda dd: 1500.0),
        ("risk=0.8*DD", lambda dd: 0.8 * dd), ("risk=1.0*DD", lambda dd: 1.0 * dd),
        ("risk=1.2*DD", lambda dd: 1.2 * dd),
    ]
    rows = []
    for name, pol in policies:
        p, mdp, mt, thru, epp = stat(R, maeR, pol, start_bal)
        rows.append((name, p, mdp, mt, thru, epp))
        print(f"{name:>16} | {p*100:>5.1f}% {mdp:>11} {mt:>12.0f}d {thru:>13.1f} {epp:>10.1f}")
    best_thru = max(rows, key=lambda r: r[4])
    best_pass = max(rows, key=lambda r: r[1])
    print(f"  -> most passes/yr/slot: {best_thru[0]} ({best_thru[4]:.1f}); highest pass%: {best_pass[0]} ({best_pass[1]*100:.1f}%)")


def main():
    df = build_outcomes(1500.0, 1500.0)
    R = (df.g / 1500.0).values
    maeR = (df.mae / 1500.0).values
    print(f"pool n={R.size}  avg R {R.mean():+.3f}  WR {(R>0).mean()*100:.1f}%  (dynamic ATR SL/TP baked in)")
    block(R, maeR, "TIGHT account: remaining DD $500 (just lost first ~$1500 trade)", START - 1500.0)
    block(R, maeR, "FRESH account: remaining DD $2000", START)


if __name__ == "__main__":
    main()

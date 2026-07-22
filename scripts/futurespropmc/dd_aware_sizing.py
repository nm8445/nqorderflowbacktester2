"""DD-aware sizing vs flat $1500: how much does sizing to remaining-DD lift the MAE-aware pass rate?

Rule: risk = min($1500, k * remaining_DD).  When the account is below its $50k peak the trailing
floor is FIXED, so a small win GROWS the cushion -> you can size back up. When DD is comfortable
(k*DD >= 1500) you trade full size. TP = risk (1:1), so each trade's outcome + worst dip are
scale-invariant in R -> reuse the real 1-min first-touch pool from challenge_gamble_pass.

MAE-aware throughout (bal + worst_floating_$ <= floor => blown mid-trade). Pass at +$3,000
(same rule as the 34% baseline; NOTE: ignores the 50% consistency rule, which DOWN-sizing only
HELPS -> the reported improvement is conservative). No hard time limit on real futures evals, so
we show a near-term (60-trade) and a patient (250-trade) horizon plus median days-to-pass.

Run: python scripts/futurespropmc/dd_aware_sizing.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from challenge_gamble_pass import build_outcomes   # noqa: E402

START, FLOOR0, DD, TARGET = 50000.0, 48000.0, 2000.0, 3000.0
R_MIN = 100.0     # 1-MNQ minimum risk (~stop*$2); you can't size below this -> tight DD can still blow
N = 50000


def sim(R, maeR, rng, sizing, max_days, start_bal=START, start_peak=START, mae_aware=True):
    """Returns (passed, days, reason). reason: pass | blow (floor breached) | timeout (stalled -
    ground down to tiny size without reaching +$3k). With risk >= R_MIN a real breach IS possible
    once remaining DD < R_MIN (forced 1-MNQ minimum)."""
    bal = start_bal; peak = start_peak; floor = min(START, peak - DD)
    for d in range(max_days):
        dd = bal - floor
        if dd <= 0:
            return 0, d, "blow"
        risk = max(R_MIN, sizing(dd))                # can't size below 1 MNQ
        i = rng.integers(R.size)
        m = maeR[i] * risk
        if mae_aware and bal + m <= floor:           # floating dip breaches the floor mid-trade
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


def run(R, maeR, label, sizing, max_days, start_bal=START, start_peak=START):
    rng = np.random.default_rng(7)
    res = [sim(R, maeR, rng, sizing, max_days, start_bal, start_peak) for _ in range(N)]
    p = np.mean([r[0] for r in res])
    outs = np.array([r[2] for r in res])
    dys = np.array([r[1] for r in res])
    pass_days = dys[outs == "pass"]
    md = int(np.median(pass_days)) if pass_days.size else 0
    print(f"  {label:22s}  PASS {p*100:5.1f}%  BLOW {np.mean(outs=='blow')*100:5.1f}%  "
          f"TIMEOUT {np.mean(outs=='timeout')*100:5.1f}%  med days {md:>3}")
    return p


def main():
    df = build_outcomes(1500.0, 1500.0)              # 1:1 pool, real 1-min first-touch
    R = (df.g / 1500.0).values
    maeR = (df.mae / 1500.0).values
    print(f"pool: n={R.size}  avg R {R.mean():+.3f}  avg MAE_R {maeR.mean():+.3f}  WR {(R>0).mean()*100:.1f}%")

    full = lambda dd: 1500.0
    # Rule shapes (all capped at $1500; sim floors at R_MIN):
    #   prop k   = risk k*dd (proportional; downsizes winners too when below peak)
    #   buffer b = risk dd-b (bet everything except a $b cushion; stays full-size when comfortable)
    prop = lambda k: (lambda dd: min(1500.0, k * dd))
    buf = lambda b: (lambda dd: min(1500.0, dd - b))
    thr = lambda t: (lambda dd: 1500.0 if dd >= t else R_MIN)   # full size only above a DD threshold
    rules = [
        ("flat $1500 (baseline)", full),
        ("prop k=0.5", prop(0.5)),
        ("prop k=0.7", prop(0.7)),
        ("buffer $300", buf(300.0)),
        ("buffer $900 (=avg MAE)", buf(900.0)),
        ("threshold dd>=$1000", thr(1000.0)),
    ]

    print("\n===== FRESH accounts, horizon 250 trades =====")
    for label, rule in rules:
        run(R, maeR, label, rule, 250)

    # The user's live situation: 6 accounts currently at ~$350-650 remaining DD (early-loss state,
    # floor still $48k, peak still $50k). Conditional pass prob from HERE, full-size vs each rule.
    print("\n===== CONDITIONAL: an account sitting at tight DD right now (peak $50k, floor $48k) =====")
    for dd_now in (350.0, 500.0, 650.0):
        sb = FLOOR0 + dd_now                          # bal = floor + remaining DD
        print(f"  -- remaining DD ${dd_now:.0f} (bal ${sb:,.0f}, profit ${sb-START:+,.0f}) | horizon 250 --")
        for label, rule in rules:
            run(R, maeR, label, rule, 250, start_bal=sb)


if __name__ == "__main__":
    main()

"""FundingPips 2-step challenge MC — CONFIGURABLE rule-set + per-phase targets.

Edit CONFIG for any FP/CFD 2-step variant. Reports flat-MNQ and per-strat-MAE sizing.
Live 4-way combined (OD/B2/RV/FB), intraday-aware (walk trades within day by exit time).

Run:  python "scripts/cfd prop firms/fundingpips_2step_configurable.py"
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"

# ======================= CONFIG =======================
START_BAL     = 100_000.0
DLL_PCT       = 0.05          # daily loss limit, % of START-OF-DAY balance
MAXLOSS_PCT   = 0.10          # static max loss, % of initial
P1_TARGET_PCT = 0.10          # phase-1 profit target, % of initial
P2_TARGET_PCT = 0.05          # phase-2 profit target, % of initial
# ======================================================
STATIC_FLOOR  = START_BAL * (1 - MAXLOSS_PCT)
P1_TARGET_BAL = START_BAL * (1 + P1_TARGET_PCT)
P2_TARGET_BAL = START_BAL * (1 + P2_TARGET_PCT)

CAP_DAYS, N_SIMS = 300, 20_000
NQ_PT, COST_RT_PER_MNQ = 20.0, 4.0
SL_PTS = {"OD": 600.0, "B2": 600.0, "RV": 200.0, "FB": 150.0}
RISK_GRID = [500, 750, 1000, 1500, 2000, 2500, 3000]
MNQ_GRID = [1, 2, 3, 4, 5, 6, 8, 10]


def load_packs():
    df = pd.read_csv(TRADES_CSV)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    df["d"] = df["exit_ts"].dt.date
    df = df.sort_values(["d", "exit_ts"])
    return [list(zip(g["strat"], g["pnl_$"].astype(float))) for _, g in df.groupby("d", sort=True)]


def simulate_phase(packs, scale_fn, target_bal, rng):
    balance = START_BAL
    n = len(packs)
    for day in range(CAP_DAYS):
        day_start = balance
        dll = DLL_PCT * day_start
        running = 0.0
        for strat, pnl in packs[rng.integers(0, n)]:
            sc, cost = scale_fn(strat)
            running += pnl * sc - cost
            eq = day_start + running
            if running <= -dll:
                return (False, day + 1, "DLL")
            if eq <= STATIC_FLOOR:
                return (False, day + 1, "MaxLoss")
            if eq >= target_bal:
                return (True, day + 1, "Target")
        balance = day_start + running
        if balance >= target_bal:
            return (True, day + 1, "Target")
    return (False, CAP_DAYS, "Timeout")


def run_grid(label, packs, scale_fns, key_name):
    rows = []
    for k, fn in scale_fns.items():
        rng = np.random.default_rng(13 + hash(str(k)) % 1000)
        p1 = both = 0; d1 = []; d2 = []; dtot = []; fdll = fml = 0
        for _ in range(N_SIMS):
            ok1, days1, why1 = simulate_phase(packs, fn, P1_TARGET_BAL, rng)
            if ok1:
                p1 += 1; d1.append(days1)
                ok2, days2, _ = simulate_phase(packs, fn, P2_TARGET_BAL, rng)
                if ok2:
                    both += 1; d2.append(days2); dtot.append(days1 + days2)
            else:
                if why1 == "DLL": fdll += 1
                elif why1 == "MaxLoss": fml += 1
        rows.append({
            key_name: k,
            "P(pass p1)": round(p1 / N_SIMS, 3),
            "P(pass BOTH)": round(both / N_SIMS, 3),
            "med_d_p1": int(np.median(d1)) if d1 else None,
            "med_d_total": int(np.median(dtot)) if dtot else None,
            "p25_tot": int(np.percentile(dtot, 25)) if dtot else None,
            "p75_tot": int(np.percentile(dtot, 75)) if dtot else None,
            "fail_DLL%": round(fdll / N_SIMS, 3),
            "fail_Max%": round(fml / N_SIMS, 3),
        })
    df = pd.DataFrame(rows)
    print(f"\n=== {label} ===")
    print(df.to_string(index=False))
    return df


def main():
    pd.set_option("display.width", 240)
    packs = load_packs()
    print(f"$100k | DLL {DLL_PCT:.0%} of day-start | static floor ${STATIC_FLOOR:,.0f} "
          f"({MAXLOSS_PCT:.0%}) | P1 +${P1_TARGET_PCT*START_BAL:,.0f} | P2 +${P2_TARGET_PCT*START_BAL:,.0f}")

    def perstrat(R):
        def f(s):
            sc = R / (SL_PTS[s] * NQ_PT)
            return sc, sc * 10.0 * COST_RT_PER_MNQ
        return f

    def flat(m):
        def f(s):
            return m / 10.0, COST_RT_PER_MNQ * m
        return f

    run_grid("FLAT MNQ (4-way)", packs, {m: flat(m) for m in MNQ_GRID}, "mnq")
    run_grid("PER-STRAT MAE risk (4-way)", packs, {R: perstrat(R) for R in RISK_GRID}, "risk_$/trade")


if __name__ == "__main__":
    main()

"""FundingPips-style 100K TWO-STEP CHALLENGE — pass rate + days-to-pass Monte Carlo.

Rules (user spec):
  - Account: $100,000
  - Daily loss limit (DLL): 3% of START-OF-DAY balance (intraday-aware)
  - Static max loss: 6% of initial -> floor $94,000 (never moves)
  - Profit target: +$6,000 EACH phase (balance -> $106,000)
  - Two steps: Phase 1, then Phase 2 (fresh $100k). Pass = clear BOTH.

Strategy: the LIVE 4-way combined (OD+RV+B2+FB) daily P&L, bootstrapped by trading day.
Source: scripts/rough vol orderflow/results/combined_4way_trades.csv (1 NQ/strat = 10 MNQ base).
Sizing swept as MNQ count (scale = mnq/10). Within each day we walk trades in EXIT-time order so
a mid-day DLL or max-loss breach is caught (not just EOD). Realized-PnL based (true intraday
floating MAE would be a touch stricter — noted).

Run:  python "scripts/cfd prop firms/fundingpips_2step_100k_mc.py"
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
OUT_CSV = Path(__file__).resolve().parent / "fundingpips_2step_100k_mc.csv"

START_BAL    = 100_000.0
DLL_PCT      = 0.03            # 3% of start-of-day balance
MAXLOSS_PCT  = 0.06           # 6% of initial -> static floor
STATIC_FLOOR = START_BAL * (1 - MAXLOSS_PCT)   # 94,000
TARGET       = 6_000.0        # +$6k each phase
TARGET_BAL   = START_BAL + TARGET              # 106,000
MIN_DAYS     = 0              # min trading days before a pass counts (FP varies; 0 = none)
CAP_DAYS     = 250           # per-phase compute cap (FP challenge has no time limit)
COST_RT_PER_MNQ = 4.0        # commission+spread analog $/round-turn/MNQ
N_SIMS       = 20_000
MNQ_GRID     = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]


def load_daily_packs() -> list[np.ndarray]:
    """Per trading day: array of trade pnl (1 NQ basis), ordered by exit time."""
    df = pd.read_csv(TRADES_CSV)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    df["d"] = df["exit_ts"].dt.date
    df = df.sort_values(["d", "exit_ts"])
    return [g["pnl_$"].to_numpy(float) for _, g in df.groupby("d", sort=True)]


def simulate_phase(packs, scale, cost, rng) -> tuple[bool, int, str]:
    """One challenge phase. Returns (passed, days_used, reason)."""
    balance = START_BAL
    n = len(packs)
    for day in range(CAP_DAYS):
        day_start = balance
        dll = DLL_PCT * day_start
        trades = packs[rng.integers(0, n)]
        running = 0.0
        for raw in trades:
            running += raw * scale - cost
            equity = day_start + running
            if running <= -dll:
                return (False, day + 1, "DLL")
            if equity <= STATIC_FLOOR:
                return (False, day + 1, "MaxLoss")
            if equity >= TARGET_BAL and (day + 1) >= MIN_DAYS:
                return (True, day + 1, "Target")
        balance = day_start + running
        if balance >= TARGET_BAL and (day + 1) >= MIN_DAYS:
            return (True, day + 1, "Target")
    return (False, CAP_DAYS, "Timeout")


def run():
    packs = load_daily_packs()
    allpnl = np.array([p.sum() for p in packs])
    print(f"Loaded {len(packs)} trading days (4-way @1NQ): mean ${allpnl.mean():.0f}/day "
          f"std ${allpnl.std():.0f}  worst ${allpnl.min():.0f}  best ${allpnl.max():.0f}")
    print(f"Rules: $100k | DLL 3% of day-start | static floor ${STATIC_FLOOR:,.0f} | "
          f"target +${TARGET:,.0f}/phase x2 | cost ${COST_RT_PER_MNQ}/MNQ RT\n")

    rows = []
    for mnq in MNQ_GRID:
        scale = mnq / 10.0
        cost = COST_RT_PER_MNQ * mnq
        rng = np.random.default_rng(1234 + mnq)
        p1_pass = p2_pass = 0
        d1 = []; d2 = []
        both = 0; tot_days = []
        r1 = {"DLL": 0, "MaxLoss": 0, "Timeout": 0}
        for _ in range(N_SIMS):
            ok1, days1, why1 = simulate_phase(packs, scale, cost, rng)
            if ok1:
                p1_pass += 1; d1.append(days1)
                ok2, days2, why2 = simulate_phase(packs, scale, cost, rng)
                if ok2:
                    p2_pass += 1; d2.append(days2); both += 1
                    tot_days.append(days1 + days2)
                else:
                    r1[why2 if why2 in r1 else "Timeout"] += 1
            else:
                r1[why1 if why1 in r1 else "Timeout"] += 1
        rows.append({
            "mnq": mnq,
            "P(pass phase1)": round(p1_pass / N_SIMS, 3),
            "P(pass phase2|p1)": round(p2_pass / max(p1_pass, 1), 3),
            "P(pass BOTH)": round(both / N_SIMS, 3),
            "med_days_p1": int(np.median(d1)) if d1 else None,
            "med_days_p2": int(np.median(d2)) if d2 else None,
            "med_days_total": int(np.median(tot_days)) if tot_days else None,
            "p25_total": int(np.percentile(tot_days, 25)) if tot_days else None,
            "p75_total": int(np.percentile(tot_days, 75)) if tot_days else None,
            "fail_DLL%": round(r1["DLL"] / N_SIMS, 3),
            "fail_MaxLoss%": round(r1["MaxLoss"] / N_SIMS, 3),
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print(df.to_string(index=False))
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    best = df.loc[df["P(pass BOTH)"].idxmax()]
    print(f"\nBest combined pass rate: {best['mnq']:.0f} MNQ -> "
          f"P(both)={best['P(pass BOTH)']:.0%}, median {best['med_days_total']} td to clear both phases.")


if __name__ == "__main__":
    run()

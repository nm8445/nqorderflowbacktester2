"""FundingPips 100K 2-step challenge — SIZING VARIANTS.

Same rules as fundingpips_2step_100k_mc.py ($100k, 3% DLL of day-start, 6% static floor,
+$6k each phase, 2 steps). Compares smarter sizing than flat MNQ:

  A) PER-STRAT RISK sizing — each strategy sized to an equal $-risk-per-trade using its
     historical worst-MAE (from mt5_executor.sl_pts_per_strat: OD 600pt, B2 600pt, RV 200pt,
     FB 150pt). OD (huge overnight MAE) gets small size; RV/FB (tight) get large size. This
     caps each strat's per-trade $ risk so OD tail days stop nuking the 3% daily limit.
        scale_strat(R) = R / (sl_pts_strat * $20)   # in NQ units
  B) PER-STRAT RISK, OD DROPPED — RV+B2+FB only.
  C) FLAT MNQ, OD DROPPED — for apples-to-apples vs the base flat-MNQ run.

Run:  python "scripts/cfd prop firms/fundingpips_2step_variants.py"
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
OUT_CSV = Path(__file__).resolve().parent / "fundingpips_2step_variants.csv"

START_BAL, DLL_PCT, STATIC_FLOOR = 100_000.0, 0.03, 94_000.0
TARGET_BAL = 106_000.0
CAP_DAYS, N_SIMS = 250, 20_000
COST_RT_PER_MNQ = 4.0
NQ_PT = 20.0

# Per-strat worst-MAE-based risk basis (NQ points) — from live mt5_executor.sl_pts_per_strat
SL_PTS = {"OD": 600.0, "B2": 600.0, "RV": 200.0, "FB": 150.0}
RISK_GRID = [250, 500, 750, 1000, 1250, 1500, 2000]     # target $ risk per trade
MNQ_GRID = [1, 2, 3, 4, 5, 6, 8, 10]


def load_packs(drop_od=False):
    """Per day: list of (strat, pnl_1nq) ordered by exit time."""
    df = pd.read_csv(TRADES_CSV)
    if drop_od:
        df = df[df["strat"] != "OD"]
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    df["d"] = df["exit_ts"].dt.date
    df = df.sort_values(["d", "exit_ts"])
    return [list(zip(g["strat"], g["pnl_$"].astype(float))) for _, g in df.groupby("d", sort=True)]


def simulate_phase(packs, scale_fn, rng):
    """scale_fn(strat) -> (nq_scale, cost_per_trade). Returns (passed, days, reason)."""
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
            if eq >= TARGET_BAL:
                return (True, day + 1, "Target")
        balance = day_start + running
        if balance >= TARGET_BAL:
            return (True, day + 1, "Target")
    return (False, CAP_DAYS, "Timeout")


def run_grid(label, packs, scale_fns: dict, key_name: str):
    rows = []
    for k, scale_fn in scale_fns.items():
        rng = np.random.default_rng(99 + hash(str(k)) % 1000)
        p1 = both = 0
        d1 = []; dtot = []; fdll = fml = 0
        for _ in range(N_SIMS):
            ok1, days1, why1 = simulate_phase(packs, scale_fn, rng)
            if ok1:
                p1 += 1; d1.append(days1)
                ok2, days2, _ = simulate_phase(packs, scale_fn, rng)
                if ok2:
                    both += 1; dtot.append(days1 + days2)
            else:
                if why1 == "DLL": fdll += 1
                elif why1 == "MaxLoss": fml += 1
        rows.append({
            key_name: k,
            "P(pass p1)": round(p1 / N_SIMS, 3),
            "P(pass BOTH)": round(both / N_SIMS, 3),
            "med_days_total": int(np.median(dtot)) if dtot else None,
            "p25_total": int(np.percentile(dtot, 25)) if dtot else None,
            "p75_total": int(np.percentile(dtot, 75)) if dtot else None,
            "fail_DLL%": round(fdll / N_SIMS, 3),
            "fail_MaxLoss%": round(fml / N_SIMS, 3),
        })
    df = pd.DataFrame(rows)
    print(f"\n=== {label} ===")
    print(df.to_string(index=False))
    return df


def main():
    pd.set_option("display.width", 240)
    packs_4w = load_packs(drop_od=False)
    packs_no_od = load_packs(drop_od=True)
    print(f"4-way days: {len(packs_4w)} | OD-dropped days: {len(packs_no_od)}")
    print(f"Per-strat risk sizing basis (NQ pts): {SL_PTS}\n")

    def perstrat(R):
        def f(strat):
            sc = R / (SL_PTS[strat] * NQ_PT)          # NQ-equivalent size
            return sc, sc * 10.0 * COST_RT_PER_MNQ    # cost: sc*10 MNQ
        return f

    def flat(mnq):
        def f(strat):
            sc = mnq / 10.0
            return sc, COST_RT_PER_MNQ * mnq
        return f

    dfs = {}
    dfs["A_perstrat_4way"] = run_grid(
        "A) PER-STRAT RISK sizing — 4-way (OD/B2/RV/FB)",
        packs_4w, {R: perstrat(R) for R in RISK_GRID}, "risk_$/trade")
    dfs["B_perstrat_noOD"] = run_grid(
        "B) PER-STRAT RISK sizing — OD DROPPED (RV/B2/FB)",
        packs_no_od, {R: perstrat(R) for R in RISK_GRID}, "risk_$/trade")
    dfs["C_flat_noOD"] = run_grid(
        "C) FLAT MNQ — OD DROPPED (RV/B2/FB)",
        packs_no_od, {m: flat(m) for m in MNQ_GRID}, "mnq")

    big = pd.concat([d.assign(variant=k) for k, d in dfs.items()], ignore_index=True)
    big.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    for k, d in dfs.items():
        b = d.loc[d["P(pass BOTH)"].idxmax()]
        kn = "risk_$/trade" if "perstrat" in k else "mnq"
        print(f"  {k}: best P(both)={b['P(pass BOTH)']:.0%} at {kn}={b[kn]}, "
              f"median {b['med_days_total']} td")


if __name__ == "__main__":
    main()

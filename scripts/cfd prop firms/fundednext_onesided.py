"""FundedNext 100k FUNDED EV under the ONE-SIDED-BET compliant regime:
  - no hedging + only ONE position open at a time (overlapping RTH signals are ignored until flat)
  - each strat sized so its WORST historical MAE <= $2,800 (under the 3% = $3,000 RPTI) -> RPTI
    can never trip by construction; OD (overnight) never overlaps RTH so it keeps its own budget.

Rules: $100k, 5% daily loss (equity/floating, of day-start), 10% static max ($90k floor),
RPTI $3,000 (won't bind), 80% split, payout every 10 td, min $200.  Data:
combined_4way_with_mae_1min.csv (ts, exit_ts, strat, dir, pnl_1c, mae_1c at 1 NQ = 10 MNQ).

Run:  python "scripts/cfd prop firms/fundednext_onesided.py"
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from math import floor

CSV = Path(__file__).resolve().parents[2] / "scripts" / "montecarlo" / "results" / "combined_4way_with_mae_1min.csv"
START, DLL, FLOOR, RPTI = 100_000., 0.05, 90_000., 3000.
SPLIT, CYCLE, MINPAY, COST = 0.80, 10, 200., 4.0
BUDGET = 2800.
HORIZON, N = 252, 20_000


def build(one_at_a_time=True, budget=BUDGET):
    df = pd.read_csv(CSV)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    # per-strat whole-MNQ sizing so worst MAE (1 MNQ = mae_1c/10) * mnq <= budget
    worst = df.groupby("strat")["mae_1c"].min().abs() / 10.   # $ per 1 MNQ
    mnq = {s: max(1, floor(budget / worst[s])) for s in worst.index}
    # one-position-at-a-time global filter (take if flat)
    keep = []
    last_exit = pd.Timestamp.min.tz_localize("UTC")
    for _, r in df.iterrows():
        if one_at_a_time and r["ts"] < last_exit:
            continue
        keep.append(r.name); last_exit = r["exit_ts"]
    f = df.loc[keep].copy()
    f["mnq"] = f["strat"].map(mnq)
    f["pnl"] = f["pnl_1c"] * f["mnq"] / 10.       # scale 1-NQ -> mnq MNQ
    f["flo"] = (-f["mae_1c"]) * f["mnq"] / 10.
    f["date"] = pd.to_datetime(f["date"]).dt.date
    packs = [list(zip(g["pnl"], g["flo"], g["mnq"])) for _, g in f.groupby("date", sort=True)]
    return packs, mnq, worst, f


def sim(packs, rng):
    bal = START; dic = 0; cash = 0.; pays = 0; n = len(packs)
    for d in range(HORIZON):
        base = bal; dfloor = base * (1 - DLL); real = 0.; bust = None
        for pnl, flo, m in packs[rng.integers(0, n)]:
            if flo >= RPTI: bust = "RPTI"; break
            eq = bal + real - flo
            if eq <= FLOOR: bust = "MaxLoss"; break
            if eq <= dfloor: bust = "DLL"; break
            real += pnl - m * COST
        if bust: return cash, pays, bust, d + 1
        bal += real; dic += 1
        if dic >= CYCLE:
            profit = bal - START
            if profit >= MINPAY: cash += profit * SPLIT; pays += 1; bal = START
            dic = 0
    return cash, pays, None, HORIZON


def run(label, one_at_a_time):
    packs, mnq, worst, f = build(one_at_a_time)
    rng = np.random.default_rng(7)
    res = [sim(packs, rng) for _ in range(N)]
    cash = np.array([r[0] for r in res]); pays = np.array([r[1] for r in res])
    reasons = [r[2] for r in res]; blow = np.mean([r[2] is not None for r in res])
    print(f"--- {label} ---")
    if one_at_a_time:
        print("  sizing (worst MAE <= $2,800):  " +
              "  ".join(f"{s}={mnq[s]}MNQ(worst ${worst[s]*mnq[s]:.0f})" for s in ["OD","RV","B2","FB"]))
        print(f"  trades kept after 1-at-a-time filter: {len(f)} "
              f"(per-strat: {dict(f.strat.value_counts())})")
    print(f"  E[$ withdrawn/yr] = ${cash.mean():,.0f}   median ${np.median(cash):,.0f}   "
          f"payouts/yr {pays.mean():.1f}")
    print(f"  blow(1yr) = {blow*100:.1f}%  (RPTI {np.mean([x=='RPTI' for x in reasons])*100:.1f}, "
          f"DLL {np.mean([x=='DLL' for x in reasons])*100:.1f}, "
          f"MaxLoss {np.mean([x=='MaxLoss' for x in reasons])*100:.1f})\n")


def budget_sweep():
    print("Budget knob (1-at-a-time, MAE<=budget/strat) — trade EV for survival:")
    print(f"{'budget':>7}  {'sizes (OD/RV/B2/FB)':>22}  {'E[$/yr]':>9}  {'median':>8}  {'blow%':>6}")
    for bud in (2800, 2000, 1500, 1000, 700):
        packs, mnq, worst, f = build(True, bud)
        rng = np.random.default_rng(7)
        res = [sim(packs, rng) for _ in range(N)]
        cash = np.array([r[0] for r in res]); blow = np.mean([r[2] is not None for r in res])
        sizes = "/".join(str(mnq[s]) for s in ["OD", "RV", "B2", "FB"])
        print(f"  ${bud:>5}  {sizes:>22}  ${cash.mean():>8,.0f}  ${np.median(cash):>7,.0f}  {blow*100:>5.1f}%")
    print()


def main():
    print(f"FundedNext 100k funded — one-sided-bet compliant (1 position at a time, MAE<=$2,800/strat)\n")
    run("ONE-SIDED COMPLIANT (1-at-a-time + MAE budget)", True)
    run("reference: same sizing, NO 1-at-a-time filter (overlaps allowed)", False)
    budget_sweep()


if __name__ == "__main__":
    main()

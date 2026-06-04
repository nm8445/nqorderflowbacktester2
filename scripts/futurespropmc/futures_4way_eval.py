"""PROPER futures 50k eval Monte Carlo — FULL 4-way (OD+RV+B2+FB), consistent 1-min MAE, marti OFF.

Account rule (the painstaking one):
  $50k start, $3k profit target, $2k EOD drawdown that TRAILS then LOCKS at $50k:
      floor = min(50000, max(48000, peak_EOD_balance - 2000))
  FLOATING-blowable: intraday equity (balance + realized-so-far + open-trade MAE) checked vs floor.
  Trades processed in fill-time order within a day (OD overnight first, then RTH RV/B2/FB).

Data: scripts/montecarlo/results/combined_4way_with_mae_1min.csv (built by build_4way_mae.py;
pnl/mae per contract = marti OFF; MAE = worst 1-min low/high over each trade's hold).

Run:  python scripts/montecarlo/futures_4way_eval.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "scripts" / "montecarlo" / "results" / "combined_4way_with_mae_1min.csv"
START, TARGET = 50000.0, 3000.0
DD, LOCK = 2000.0, 50000.0
COST = 2.0
N_SIMS, CAP = 15000, 504


def load_day_packs(strats):
    df = pd.read_csv(CSV)
    df = df[df.strat.isin(strats)].sort_values("ts")
    return [list(zip(g["pnl_1c"], g["mae_1c"])) for _, g in df.groupby("date", sort=True)]


def sim(packs, mnq, rng, floating=True):
    s = mnq / 10.0
    bal = START; peak = START; floor = START - DD
    n = len(packs)
    for d in range(CAP):
        tr = packs[rng.integers(0, n)]
        realized = 0.0; bust = False
        for pnl, mae in tr:
            adj = (mae if floating else pnl) * s - COST * mnq    # mae is negative
            if bal + realized + adj < floor:
                bust = True; break
            realized += pnl * s - COST * mnq
        if bust:
            return (0, d + 1)
        bal += realized
        if bal - START >= TARGET:
            return (1, d + 1)
        if bal > peak: peak = bal
        floor = min(LOCK, max(START - DD, peak - DD))
    return (0, CAP)


def run(label, strats):
    packs = load_day_packs(strats)
    print(f"\n=== {label} ({len(packs)} trading days) ===")
    print("%4s  %8s  %8s  %10s  %12s" % ("MNQ", "P(pass)", "P(pass)*", "med days", "p25/p75"))
    for mnq in [1, 2, 3, 4]:
        rng = np.random.default_rng(11 + mnq)
        res = [sim(packs, mnq, rng, True) for _ in range(N_SIMS)]
        rng = np.random.default_rng(11 + mnq)
        res_r = [sim(packs, mnq, rng, False) for _ in range(N_SIMS)]
        pa = np.mean([r[0] for r in res]); par = np.mean([r[0] for r in res_r])
        dd = [r[1] for r in res if r[0] == 1]
        md = int(np.median(dd)) if dd else 0
        p25 = int(np.percentile(dd, 25)) if dd else 0; p75 = int(np.percentile(dd, 75)) if dd else 0
        print("%4d  %7.1f%%  %7.1f%%  %8dd  %6d/%d" % (mnq, 100*pa, 100*par, md, p25, p75))
    print("  (P(pass) = floating/MAE-aware = real rule; P(pass)* = realized-only, for reference)")


def main():
    pd.set_option("display.width", 200)
    print("PROPER futures 50k eval: 4-way (OD+RV+B2+FB), 1-min MAE, marti OFF,")
    print("trailing-then-lock @50k floor, $3k target, FLOATING-blowable.")
    run("FULL 4-way (your live combine)", ["OD", "RV", "B2", "FB"])
    run("drop B2 (OD+RV+FB)", ["OD", "RV", "FB"])
    run("RV+FB only", ["RV", "FB"])


if __name__ == "__main__":
    main()

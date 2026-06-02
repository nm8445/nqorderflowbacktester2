"""FundingPips FUNDED (Master) EV — RULE-ACCURATE: floating DLL + floating Max Loss + $2k RPTI.

Real FundingPips funded rules:
  - Daily Loss: 5% of max(opening balance, opening equity) — EQUITY based (floating counts).
    Reset 00:00 UTC+3 = 5pm ET (bundles OD-overnight with the next day's session).
  - Max Loss: 10% static floor ($90k) — equity based (floating counts).
  - Risk Per Trade Idea (RPTI): $100k -> 2% = $2,000 max loss on ONE trade idea, realized +
    UNREALISED. Master/funded ONLY (not eval). Since the combined trades ONE instrument (NAS100),
    concurrent same-direction positions + the 10-min reopen rule COMBINE into one idea -> we model
    each trade's floating MAE against $2k (a LOWER BOUND on breach risk; concurrency would be worse).

No hard stops (would truncate the edge) -> floating MAE is what trips RPTI/DLL/MaxLoss.
Data: combined_trades_with_mae.csv (OD/RV/B2 + mae_$ at 1 NQ). FB excluded (no MAE in file; FB is
long-only ORB w/ ORB-low stop ~100pt MAE, low RPTI risk — would add profit, little extra bust).

Run:  python "scripts/cfd prop firms/fundingpips_funded_rpti.py"
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAE_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
FB_CSV = ROOT / "scripts" / "fabio_orb" / "results" / "paper_replication_delta300_trades.csv"

START = 100_000.0
DLL_PCT, FLOOR = 0.05, 90_000.0
RPTI = 2_000.0
NQ_PT, COST = 20.0, 4.0
CYCLE_TD, SPLIT, MIN_PAYOUT = 10, 0.80, 200.0
HORIZON_TD, N_SIMS = 252, 20_000
MNQ_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


def load_packs():
    df = pd.read_csv(MAE_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["date", "entry_ts"])
    # per day: list of (pnl_1nq, floating_loss_mag_1nq)
    return [list(zip(g["pnl_$"].astype(float), (-g["mae_$"]).astype(float)))
            for _, g in df.groupby("date", sort=True)]


def simulate(packs, mnq, rng):
    scale = mnq / 10.0
    bal = START
    dic = 0
    cash = 0.0
    n = len(packs)
    for day in range(HORIZON_TD):
        baseline = bal                       # flat at FP day open (OD enters after the 5pm reset)
        dll_floor = baseline - DLL_PCT * baseline
        realized = 0.0
        bust = None
        for pnl1, flo1 in packs[rng.integers(0, n)]:
            flo = flo1 * scale               # this trade's floating loss magnitude
            if flo >= RPTI:                  # single-trade-idea floating kill
                bust = "RPTI"; break
            eq_low = bal + realized - flo     # worst equity during this trade
            if eq_low <= FLOOR:
                bust = "MaxLoss"; break
            if eq_low <= dll_floor:
                bust = "DLL"; break
            realized += pnl1 * scale - mnq * COST
        if bust:
            return cash, bust, day + 1
        bal += realized
        dic += 1
        if dic >= CYCLE_TD:
            profit = bal - START
            if profit >= MIN_PAYOUT:
                cash += profit * SPLIT
                bal = START
            dic = 0
    return cash, None, HORIZON_TD


def main():
    pd.set_option("display.width", 220)
    packs = load_packs()
    print(f"FundingPips funded (RULE-ACCURATE): floating 5% DLL, 10% floor, $2k RPTI/trade-idea.")
    print(f"{len(packs)} FP-days (OD/RV/B2 + MAE; FB excluded). Flat MNQ, native exits (no hard stop).\n")
    rows = []
    for mnq in MNQ_GRID:
        rng = np.random.default_rng(11 + int(mnq * 100))
        res = [simulate(packs, mnq, rng) for _ in range(N_SIMS)]
        cash = np.array([r[0] for r in res])
        reasons = [r[1] for r in res]
        bust = np.array([r[1] is not None for r in res])
        rows.append({
            "mnq": mnq,
            "E[$ withdrawn]": round(cash.mean(), 0),
            "median_$": round(np.median(cash), 0),
            "bust(1yr)": round(bust.mean(), 3),
            "via_RPTI": round(np.mean([x == "RPTI" for x in reasons]), 3),
            "via_DLL": round(np.mean([x == "DLL" for x in reasons]), 3),
            "via_MaxLoss": round(np.mean([x == "MaxLoss" for x in reasons]), 3),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(Path(__file__).resolve().parent / "fundingpips_funded_rpti.csv", index=False)
    print("\nRPTI is funded-only, floating-based, per-trade-idea. No hard stop => the wide-ATR floating")
    print("excursions (RV/B2) + OD overnight regularly tag $2k once scaled, terminating accounts.")


if __name__ == "__main__":
    main()

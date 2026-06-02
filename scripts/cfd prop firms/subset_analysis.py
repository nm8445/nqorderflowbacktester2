"""Strategy-subset analysis for BOTH the futures 50k eval and the FundingPips funded account.

Question: does dropping RTH strats (B2) or OD let me size up / extract more?
Builds 4-way MAE packs (marti OFF): OD/RV/B2 from combined_trades_with_mae.csv (OD & B2 qty
reconstructed and divided to 1c), FB from paper_replication_delta300 (MAE = risk_pts*20, a
conservative cap on FB's float). Runs each subset through:
  A) Futures 50k eval: trailing-then-lock @50k floor, floating -> pass rate + median days.
  B) FundingPips funded: 5% daily / 10% static / $2k RPTI per-trade-idea floating -> E[$ withdrawn]/yr.

Run:  python "scripts/cfd prop firms/subset_analysis.py"
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAE = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
FBC = ROOT / "scripts" / "fabio_orb" / "results" / "paper_replication_delta300_trades.csv"
PNL = "pnl_" + chr(36); M = "mae_" + chr(36)
COST = 4.0


def per_strat_trades():
    """Return dict strat -> list of (date, pnl_1nq, mae_1nq), marti OFF for OD/B2."""
    df = pd.read_csv(MAE)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("entry_ts")
    out = {}
    # OD marti-off
    od = df[df.strat == "OD"].copy(); st = 0; q = []
    for p in od[PNL]:
        q.append(2 if st == 1 else 1); loss = p < 0
        st = (1 if loss else 0) if st == 0 else (2 if st == 1 else (1 if loss else 0))
    od["qty"] = q
    out["OD"] = list(zip(od["date"], od[PNL] / od["qty"], od[M] / od["qty"]))
    # B2 marti-off (FC inferred from ~16:00 ET exit)
    b2 = df[df.strat == "B2"].copy(); b2["fc"] = b2["exit_ts"].dt.hour == 16; ns = 1; q = []
    for p, fc in zip(b2[PNL], b2["fc"]):
        q.append(ns); ns = 1 if ns == 2 else (2 if (p < 0 and fc) else 1)
    b2["qty"] = q
    out["B2"] = list(zip(b2["date"], b2[PNL] / b2["qty"], b2[M] / b2["qty"]))
    # RV (no marti)
    rv = df[df.strat == "RV"]
    out["RV"] = list(zip(rv["date"], rv[PNL], rv[M]))
    # FB from paper_replication (MAE = risk_pts * 20)
    fb = pd.read_csv(FBC)
    fb["date"] = pd.to_datetime(fb["session_date"]).dt.date
    # FB float stored NEGATIVE (adverse), to match OD/RV/B2 mae_$ sign convention.
    # risk_pts = entry-ORB_low = full stop distance -> conservative (overstates winners' float).
    out["FB"] = list(zip(fb["date"], fb["net_dollars"].astype(float), -(fb["risk_pts"].astype(float) * 20.0)))
    return out


def build_packs(strats, per):
    by_date = {}
    for s in strats:
        for d, p, m in per[s]:
            by_date.setdefault(d, []).append((p, m))
    return [by_date[d] for d in sorted(by_date)]


# ---- A) Futures 50k eval: trailing-then-lock floating ----
def fut_eval(packs, mnq, rng):
    s = mnq / 10.; bal = 50000.; peak = 50000.; floor = 48000.; n = len(packs)
    for d in range(504):
        tr = packs[rng.integers(0, n)]; dr = 0.; bust = False
        for p, m in tr:
            if bal + dr + (m * s - COST * mnq) < floor: bust = True; break
            dr += p * s - COST * mnq
        if bust: return (0, d + 1)
        bal += dr
        if bal - 50000 >= 3000: return (1, d + 1)
        if bal > peak: peak = bal
        floor = min(50000., max(48000., peak - 2000.))
    return (0, 504)


# ---- B) FundingPips funded: floating DLL + static floor + $2k RPTI ----
def fp_funded(packs, mnq, rng):
    s = mnq / 10.; bal = 100000.; dic = 0; cash = 0.; n = len(packs)
    for d in range(252):
        base = bal; dll = 0.05 * base; realized = 0.; bust = False
        for p, m in packs[rng.integers(0, n)]:
            flo = (-m) * s if m < 0 else 0.0
            if flo >= 2000.: bust = True; break                      # RPTI
            eqlow = bal + realized - flo
            if eqlow <= 90000. or eqlow <= base - dll: bust = True; break
            realized += p * s - mnq * COST
        if bust: return cash
        bal += realized; dic += 1
        if dic >= 10:
            prof = bal - 100000.
            if prof >= 200.: cash += prof * 0.80; bal = 100000.
            dic = 0
    return cash


def main():
    pd.set_option("display.width", 240)
    per = per_strat_trades()
    subsets = {
        "ALL (OD+RV+B2+FB)": ["OD", "RV", "B2", "FB"],
        "drop B2 (OD+RV+FB)": ["OD", "RV", "FB"],
        "drop OD (RV+B2+FB)": ["RV", "B2", "FB"],
        "RV+FB only": ["RV", "FB"],
    }
    MNQ = [1, 2, 3, 4, 5]
    print("=== A) FUTURES 50k eval — trailing-then-lock floating, marti OFF — P(pass) / median days ===")
    print("%-20s " % "subset" + "  ".join("%-13s" % f"{m}MNQ" for m in MNQ))
    for name, strats in subsets.items():
        packs = build_packs(strats, per)
        cells = []
        for m in MNQ:
            rng = np.random.default_rng(3 + m)
            res = [fut_eval(packs, m, rng) for _ in range(8000)]
            pa = np.mean([r[0] for r in res]); dd = [r[1] for r in res if r[0] == 1]
            cells.append("%4.0f%%/%3dd" % (100 * pa, int(np.median(dd)) if dd else 0))
        print("%-20s " % name + "  ".join("%-13s" % c for c in cells))

    print("\n=== B) FUNDINGPIPS funded — RPTI floating, marti OFF — E[$ withdrawn]/yr (bust%) ===")
    print("%-20s " % "subset" + "  ".join("%-15s" % f"{m}MNQ" for m in MNQ))
    for name, strats in subsets.items():
        packs = build_packs(strats, per)
        cells = []
        for m in MNQ:
            rng = np.random.default_rng(50 + m)
            cash = np.array([fp_funded(packs, m, rng) for _ in range(8000)])
            cells.append("$%5.0f" % cash.mean())
        print("%-20s " % name + "  ".join("%-15s" % c for c in cells))


if __name__ == "__main__":
    main()

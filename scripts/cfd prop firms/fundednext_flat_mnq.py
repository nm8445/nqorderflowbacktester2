"""FundedNext 2-step challenge pass rate + days-to-pass with FLAT MNQ across all 4 strats
(same contract count for OD/RV/B2/FB), swept 1-10 MNQ, for 100k and 200k accounts.

Challenge = +8% then +5%, 5% daily loss, 10% static max, RPTI-exempt (eval), floating-blowable.
One-position-at-a-time compliance filter kept on.  NOTE: flat MNQ is half the %-risk on a 200k vs a
100k, so 200k needs ~2x the MNQ for the same speed.  Data: combined_4way_with_mae_1min.csv.

Run:  python "scripts/cfd prop firms/fundednext_flat_mnq.py"
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "scripts" / "montecarlo" / "results" / "combined_4way_with_mae_1min.csv"
DLL, MAXDD, COST, P1, P2 = 0.05, 0.10, 4.0, 0.08, 0.05
N = 8000


def base_trades():
    d = pd.read_csv(CSV)
    d["ts"] = pd.to_datetime(d["ts"], utc=True); d["exit_ts"] = pd.to_datetime(d["exit_ts"], utc=True)
    d = d.sort_values("ts").reset_index(drop=True)
    keep = []; last = pd.Timestamp.min.tz_localize("UTC")
    for _, r in d.iterrows():
        if r["ts"] < last: continue
        keep.append(r.name); last = r["exit_ts"]
    f = d.loc[keep].copy(); f["date"] = pd.to_datetime(f["date"]).dt.date
    return f


def packs_flat(f, m):
    pnl = f["pnl_1c"].values * m / 10.; flo = (-f["mae_1c"]).values * m / 10.
    g = pd.DataFrame({"date": f["date"].values, "pnl": pnl, "flo": flo, "m": m})
    return [list(zip(x["pnl"], x["flo"], x["m"])) for _, x in g.groupby("date", sort=True)]


def challenge(packs, size, rng, max_days=400):
    n = len(packs); total = 0
    for tgt in (P1, P2):
        bal = size; goal = size * (1 + tgt); floor_ = size * (1 - MAXDD); done = False
        for d in range(max_days):
            total += 1; dfl = bal * (1 - DLL); real = 0.; bust = False
            for pnl, flo, m in packs[rng.integers(0, n)]:
                if bal + real - flo <= floor_ or bal + real - flo <= dfl: bust = True; break
                real += pnl - m * COST
            if bust: return False, total
            bal += real
            if bal >= goal: done = True; break
        if not done: return False, total
    return True, total


def main():
    f = base_trades()
    print("FundedNext 2-step challenge — FLAT MNQ (all 4 strats same size), one-at-a-time, RPTI-exempt\n")
    for size in (100_000, 200_000):
        print(f"=== {size//1000}k account  (5% daily / 10% max DD; target +8% then +5%) ===")
        print(f"{'MNQ':>4} {'pass%':>7} {'med days':>9} {'p90 days':>9} {'worst RV float$':>15}")
        for m in range(1, 11):
            packs = packs_flat(f, m); rng = np.random.default_rng(7)
            r = [challenge(packs, size, rng) for _ in range(N)]
            pa = np.mean([x[0] for x in r]); dys = [x[1] for x in r if x[0]]
            med = int(np.median(dys)) if dys else 999; p90 = int(np.percentile(dys, 90)) if dys else 999
            rv_worst = 1622.5 * m   # RV worst MAE per MNQ ~ $1,622
            print(f"{m:>4} {pa*100:>6.1f}% {med:>9} {p90:>9} {rv_worst:>14,.0f}")
        print()


if __name__ == "__main__":
    main()

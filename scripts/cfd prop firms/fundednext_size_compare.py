"""FundedNext 100k vs 200k — one-sided-bet compliant (1-at-a-time + per-strat MAE budget).
Shows (A) funded EV and (B) 2-step challenge pass TIME, to test whether bigger accounts take longer.

Everything is %-based (target +8%/+5%, daily 5%, max 10%, RPTI 3%), so sizing to the SAME %-of-account
gives identical pass time/blow at 2x the dollars (scale-invariant).  Sizing 200k more aggressively
(higher % budget) passes faster.  Data: combined_4way_with_mae_1min.csv.

Run:  python "scripts/cfd prop firms/fundednext_size_compare.py"
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from math import floor

CSV = Path(__file__).resolve().parents[2] / "scripts" / "montecarlo" / "results" / "combined_4way_with_mae_1min.csv"
DLL, MAXDD, RPTI_PCT, SPLIT, CYCLE, MINPAY, COST = 0.05, 0.10, 0.03, 0.80, 10, 200., 4.0
P1, P2 = 0.08, 0.05
N_FUND, N_CHAL = 20_000, 10_000

_DF = None
def df():
    global _DF
    if _DF is None:
        d = pd.read_csv(CSV)
        d["ts"] = pd.to_datetime(d["ts"], utc=True); d["exit_ts"] = pd.to_datetime(d["exit_ts"], utc=True)
        _DF = d.sort_values("ts").reset_index(drop=True)
    return _DF


def packs_for(budget):
    d = df()
    worst = d.groupby("strat")["mae_1c"].min().abs() / 10.
    mnq = {s: max(1, floor(budget / worst[s])) for s in worst.index}
    keep = []; last = pd.Timestamp.min.tz_localize("UTC")
    for _, r in d.iterrows():
        if r["ts"] < last: continue
        keep.append(r.name); last = r["exit_ts"]
    f = d.loc[keep].copy(); f["mnq"] = f["strat"].map(mnq)
    f["pnl"] = f["pnl_1c"] * f["mnq"] / 10.; f["flo"] = (-f["mae_1c"]) * f["mnq"] / 10.
    f["date"] = pd.to_datetime(f["date"]).dt.date
    return [list(zip(g["pnl"], g["flo"], g["mnq"])) for _, g in f.groupby("date", sort=True)], mnq


def funded(packs, size, rng, horizon=252):
    rpti = RPTI_PCT * size; floor_ = size * (1 - MAXDD); bal = size; dic = 0; cash = 0.; n = len(packs); bust = False
    for d in range(horizon):
        base = bal; dfl = base * (1 - DLL); real = 0.
        for pnl, flo, m in packs[rng.integers(0, n)]:
            if flo >= rpti or bal + real - flo <= floor_ or bal + real - flo <= dfl: bust = True; break
            real += pnl - m * COST
        if bust: return cash, True
        bal += real; dic += 1
        if dic >= CYCLE:
            pr = bal - size
            if pr >= MINPAY: cash += pr * SPLIT; bal = size
            dic = 0
    return cash, False


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
    print("FundedNext one-sided-compliant — 100k vs 200k\n")
    print("(A) FUNDED EV (1yr), sized to the SAME 2.8% MAE budget on each:")
    print(f"{'acct':>6} {'budget$':>8} {'sizes OD/RV/B2/FB':>18} {'E[$/yr]':>9} {'blow%':>7}")
    for size, bud in [(100_000, 2800), (200_000, 5600)]:
        packs, mnq = packs_for(bud); rng = np.random.default_rng(7)
        r = [funded(packs, size, rng) for _ in range(N_FUND)]
        cash = np.mean([x[0] for x in r]); blow = np.mean([x[1] for x in r])
        sizes = "/".join(str(mnq[s]) for s in ["OD", "RV", "B2", "FB"])
        print(f"{size//1000:>5}k {bud:>8} {sizes:>18} ${cash:>8,.0f} {blow*100:>6.1f}%")

    print("\n(B) 2-step CHALLENGE pass time (+8% then +5%, 5% daily / 10% max, RPTI-exempt):")
    print(f"{'acct':>6} {'budget %':>9} {'sizes':>14} {'pass%':>7} {'med days':>9} {'p90 days':>9}")
    for label, size, pct in [("100k", 100_000, 0.028), ("200k", 200_000, 0.028),
                             ("100k", 100_000, 0.020), ("200k", 200_000, 0.040),
                             ("200k", 200_000, 0.050)]:
        packs, mnq = packs_for(pct * size); rng = np.random.default_rng(7)
        r = [challenge(packs, size, rng) for _ in range(N_CHAL)]
        pa = np.mean([x[0] for x in r]); dys = [x[1] for x in r if x[0]]
        med = int(np.median(dys)) if dys else 999; p90 = int(np.percentile(dys, 90)) if dys else 999
        sizes = "/".join(str(mnq[s]) for s in ["OD", "RV", "B2", "FB"])
        print(f"{label:>6} {pct*100:>8.1f}% {sizes:>14} {pa*100:>6.1f}% {med:>9} {p90:>9}")


if __name__ == "__main__":
    main()

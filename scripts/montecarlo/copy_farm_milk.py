"""Copy-trade farm milking EV — 50k futures funded, 1 MNQ 4-way combined.

Per account: $2k trailing-then-lock DD (floor locks at $50k once +$2k). Milk = grind to +$3k, withdraw
down to +$2k buffer (~$1k/payout), repeat. Max 5 payouts then the account is retired. Account blows
if floating equity (per-trade MAE) touches the floor.

Farm: start 10 funded; all live accounts take the SAME daily draw (copy-trade => correlated). Each
$1k payout -> cash; whenever cash >= $2k, spend $2k on evals (-> new funded after a pass lag), up to
a 30-account cap. NET $/yr = total withdrawn - total eval spend.

Run:  python scripts/montecarlo/copy_farm_milk.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parent / "results" / "combined_4way_with_mae_1min.csv"
START, DD, LOCK, COST = 50_000., 2000., 50_000., 2.0
TARGET, BUFFER, MAXPAY = 3000., 2000., 5
CAP_DAYS, N_ACCT = 504, 60_000


def packs():
    df = pd.read_csv(CSV).sort_values("ts")
    # 1 MNQ = 1/10 NQ; pnl_1c / mae_1c are at 1 NQ
    return [list(zip(g["pnl_1c"].values * .1, (-g["mae_1c"]).values * .1))
            for _, g in df.groupby("date", sort=True)]


# ---------- per-account building block ----------
def account_life(P, rng):
    n = len(P); bal = START; peak = START; floor = START - DD; locked = False; pays = 0; wd = 0.
    for day in range(CAP_DAYS):
        for p, flo in P[rng.integers(0, n)]:
            if bal - flo <= floor:
                return pays, wd, "blow", day + 1
            bal += p - COST
        if bal > peak: peak = bal
        if not locked:
            floor = min(LOCK, peak - DD)
            if floor >= LOCK: locked = True; floor = LOCK
        if bal >= START + TARGET:
            take = bal - (START + BUFFER); wd += take; bal -= take; pays += 1
            if pays >= MAXPAY:
                return pays, wd, "maxed", day + 1
    return pays, wd, "timeout", CAP_DAYS


def per_account(P):
    rng = np.random.default_rng(7)
    pays, wd, res = [], [], []
    for _ in range(N_ACCT):
        p, w, r, d = account_life(P, rng); pays.append(p); wd.append(w); res.append(r)
    pays = np.array(pays); wd = np.array(wd); res = np.array(res)
    print("PER-ACCOUNT (1 MNQ 4-way, $2k cushion, max 5 payouts):")
    print(f"  E[payouts]={pays.mean():.2f}  E[$ withdrawn]=${wd.mean():,.0f}  "
          f"P(reach 5)={np.mean(res=='maxed')*100:.1f}%  P(blow first)={np.mean(res=='blow')*100:.1f}%")
    return wd.mean(), pays.mean()


# ---------- farm with reinvestment ----------
def fresh():
    return dict(bal=START, peak=START, floor=START - DD, locked=False, pays=0)


def step(a, pack):
    """advance one account one day on a shared pack; return ('alive'|'blow'|'maxed', withdrawn)."""
    for p, flo in pack:
        if a["bal"] - flo <= a["floor"]:
            return "blow", 0.
        a["bal"] += p - COST
    if a["bal"] > a["peak"]: a["peak"] = a["bal"]
    if not a["locked"]:
        a["floor"] = min(LOCK, a["peak"] - DD)
        if a["floor"] >= LOCK: a["locked"] = True; a["floor"] = LOCK
    w = 0.
    if a["bal"] >= START + TARGET:
        w = a["bal"] - (START + BUFFER); a["bal"] -= w; a["pays"] += 1
        if a["pays"] >= MAXPAY:
            return "maxed", w
    return "alive", w


def farm_year(P, rng, evals_per_2k, lag, cap=30, start_n=10, days=252):
    n = len(P)
    accts = [fresh() for _ in range(start_n)]
    pending = []          # (ready_day, count)
    cash = 0.; withdrawn = 0.; eval_spend = 0.
    for day in range(days):
        # activate matured evals (respect cap)
        still = []
        for rd, c in pending:
            if rd <= day:
                room = cap - len(accts)
                add = min(c, max(0, room))
                accts += [fresh() for _ in range(add)]
            else:
                still.append((rd, c))
        pending = still
        pack = P[rng.integers(0, n)]           # one shared draw -> correlated
        alive = []
        for a in accts:
            st, w = step(a, pack)
            if w: cash += w; withdrawn += w
            if st == "alive": alive.append(a)
        accts = alive
        # reinvest
        inflight = len(accts) + sum(c for _, c in pending)
        while cash >= 2000. and inflight < cap:
            cash -= 2000.; eval_spend += 2000.
            pending.append((day + lag, evals_per_2k)); inflight += evals_per_2k
    return withdrawn - eval_spend, withdrawn, eval_spend, len(accts)


def run_farm(P, evals_per_2k, lag):
    rng = np.random.default_rng(11)
    nets, wds, evs, ends = [], [], [], []
    for _ in range(4000):
        net, wd, ev, end = farm_year(P, rng, evals_per_2k, lag)
        nets.append(net); wds.append(wd); evs.append(ev); ends.append(end)
    nets = np.array(nets)
    print(f"  evals/$2k={evals_per_2k}, pass-lag={lag}td:  NET ${nets.mean():,.0f}/yr "
          f"(median ${np.median(nets):,.0f}, p25 ${np.percentile(nets,25):,.0f}, "
          f"p75 ${np.percentile(nets,75):,.0f})  | gross withdrawn ${np.mean(wds):,.0f}, "
          f"eval spend ${np.mean(evs):,.0f}, end-accts {np.mean(ends):.0f}")


def main():
    P = packs()
    per_account(P)
    print("\nFARM (start 10 funded, reinvest $2k->evals, 30-acct cap, correlated daily draws):")
    for epk, lag in [(10, 45), (7, 45), (10, 30), (7, 60)]:
        run_farm(P, epk, lag)
    print("\n(NET = withdrawn - eval spend. 'evals/$2k' & 'pass-lag' are the eval economics;")
    print(" 10/$2k is your stated plan (optimistic); ~7/$2k is realistic at ~$200/funded.)")


if __name__ == "__main__":
    main()

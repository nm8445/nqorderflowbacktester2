"""RV ATR-60 entry filter -> re-run (a) FundedNext 100k funded EV, (b) 50k copy-farm milking,
(c) 50k futures eval pass rate, sweeping RV 1-3 MNQ.

RV filter: drop any RV trade whose 20-min ATR(14) at entry > 60 pt (caps SL=2xATR at 120pt,
risk ~$240/MNQ, removes the tariff-crash tail). OD/B2/FB unchanged. Worst RV float collapses from
$1,622 -> ~$300/MNQ so RV can size up.  Run: python scripts/montecarlo/rv_atr60_rerun.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "scripts" / "montecarlo" / "results" / "combined_4way_with_mae_1min.csv"
PARQ = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
ET = "America/New_York"
ATR_CAP = 60.0


def filtered():
    df = pd.read_csv(CSV)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    d = pd.read_parquet(PARQ, columns=["high", "low", "close"])
    if d.index.tz is None: d.index = d.index.tz_localize("UTC")
    d.index = d.index.tz_convert(ET); d = d.sort_index()
    b = d.resample("20min", label="right", closed="right").agg(
        {"high": "max", "low": "min", "close": "last"}).dropna()
    pc = b.close.shift(1)
    tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    a_idx = atr.index.values.astype("int64"); a_val = atr.values
    ent_et = df["ts"].dt.tz_convert(ET).dt.tz_localize(None).values.astype("datetime64[ns]").astype("int64")
    ai = np.searchsorted(a_idx, ent_et, "right") - 1
    df["atr"] = np.where(ai >= 0, a_val[np.clip(ai, 0, len(a_val) - 1)], np.nan)
    rv_mask = df.strat == "RV"
    drop = rv_mask & (df["atr"] > ATR_CAP)
    f = df[~drop].copy()
    rvw = (-f[f.strat == "RV"].mae_1c).max() / 10.   # worst RV float at 1 MNQ
    print(f"RV ATR-{ATR_CAP:.0f} filter: dropped {int(drop.sum())} RV trades "
          f"({f[f.strat=='RV'].shape[0]} kept).  RV worst float now ${rvw:,.0f}/MNQ (was $1,622)\n")
    return f


def packs(f, sizing, one_at_a_time, cost):
    g = f.sort_values("ts")
    if one_at_a_time:
        keep = []; last = pd.Timestamp.min.tz_localize("UTC")
        for _, r in g.iterrows():
            if r["ts"] < last: continue
            keep.append(r.name); last = r["exit_ts"]
        g = g.loc[keep]
    g = g.copy()
    g["mnq"] = g["strat"].map(sizing)
    g["pnl"] = g["pnl_1c"] * g["mnq"] / 10.
    g["flo"] = (-g["mae_1c"]) * g["mnq"] / 10.
    return [list(zip(x["pnl"], x["flo"], x["mnq"])) for _, x in g.groupby("date", sort=True)]


# ---------- (a) FundedNext 100k funded one-sided ----------
def funded_fn(P, rng, horizon=252):
    START, DLL, FLOOR, RPTI, SPLIT, CYC, COST = 100000., .05, 90000., 3000., .80, 10, 4.
    bal = START; dic = 0; cash = 0.; n = len(P)
    for d in range(horizon):
        base = bal; dfl = base * (1 - DLL); real = 0.; bust = False
        for pnl, flo, m in P[rng.integers(0, n)]:
            if flo >= RPTI or bal + real - flo <= FLOOR or bal + real - flo <= dfl: bust = True; break
            real += pnl - m * COST
        if bust: return cash, True
        bal += real; dic += 1
        if dic >= CYC:
            pr = bal - START
            if pr >= 200.: cash += pr * SPLIT; bal = START
            dic = 0
    return cash, False


# ---------- (c) 50k futures eval ----------
def eval_50k(P, rng):
    START, DD, LOCK, TGT, COST, CAP = 50000., 2000., 50000., 3000., 2., 504
    bal = START; peak = START; floor = START - DD; n = len(P)
    for d in range(CAP):
        real = 0.; bust = False
        for pnl, flo, m in P[rng.integers(0, n)]:
            if bal + real - flo <= floor: bust = True; break
            real += pnl - m * COST
        if bust: return "bust", d + 1
        bal += real
        if bal - START >= TGT: return "pass", d + 1
        if bal > peak: peak = bal
        floor = min(LOCK, max(START - DD, peak - DD))
    return "timeout", CAP


# ---------- (b) copy-farm milking (50k futures) ----------
def fresh(): return dict(bal=50000., peak=50000., floor=48000., locked=False, pays=0)
def step(a, pack):
    COST = 2.
    for pnl, flo, m in pack:
        if a["bal"] - flo <= a["floor"]: return "blow", 0.
        a["bal"] += pnl - m * COST
    if a["bal"] > a["peak"]: a["peak"] = a["bal"]
    if not a["locked"]:
        a["floor"] = min(50000., a["peak"] - 2000.)
        if a["floor"] >= 50000.: a["locked"] = True; a["floor"] = 50000.
    w = 0.
    if a["bal"] >= 53000.:
        w = a["bal"] - 52000.; a["bal"] -= w; a["pays"] += 1
        if a["pays"] >= 5: return "maxed", w
    return "alive", w
def farm_year(P, rng, evals_per_2k=7, lag=45, cap=30, start_n=10, days=252):
    n = len(P); accts = [fresh() for _ in range(start_n)]; pending = []
    cash = 0.; wd = 0.; ev = 0.
    for day in range(days):
        still = []
        for rd, c in pending:
            if rd <= day:
                room = cap - len(accts); accts += [fresh() for _ in range(min(c, max(0, room)))]
            else: still.append((rd, c))
        pending = still
        pack = P[rng.integers(0, n)]; alive = []
        for a in accts:
            st, w = step(a, pack)
            if w: cash += w; wd += w
            if st == "alive": alive.append(a)
        accts = alive
        inflight = len(accts) + sum(c for _, c in pending)
        while cash >= 2000. and inflight < cap:
            cash -= 2000.; ev += 2000.; pending.append((day + lag, evals_per_2k)); inflight += evals_per_2k
    return wd - ev


def main():
    f = filtered()
    print("RV sizing sweep (OD/B2/FB fixed; RV = 1..3 MNQ)\n")

    print("(a) FundedNext 100k funded one-sided (OD5/B2 4/FB 4, one-at-a-time):")
    for rv in (1, 2, 3):
        P = packs(f, {"OD": 5, "RV": rv, "B2": 4, "FB": 4}, True, 4.)
        rng = np.random.default_rng(7)
        r = [funded_fn(P, rng) for _ in range(20000)]
        cash = np.mean([x[0] for x in r]); blow = np.mean([x[1] for x in r])
        print(f"   RV@{rv}: E[$/yr]=${cash:>7,.0f}  blow={blow*100:4.1f}%")

    print("\n(c) 50k futures eval pass (OD/B2/FB @1, RV swept), trailing-lock $2k/$3k:")
    for rv in (1, 2, 3):
        P = packs(f, {"OD": 1, "RV": rv, "B2": 1, "FB": 1}, False, 2.)
        rng = np.random.default_rng(7)
        res = [eval_50k(P, rng) for _ in range(20000)]
        outs = np.array([x[0] for x in res]); dys = np.array([x[1] for x in res])
        pa = outs == "pass"; med = int(np.median(dys[pa])) if pa.any() else 0
        print(f"   RV@{rv}: pass={pa.mean()*100:4.1f}%  bust={np.mean(outs=='bust')*100:4.1f}%  "
              f"med days={med}")

    print("\n(b) 50k copy-farm milking (OD/B2/FB @1, RV swept; start 10, 30-cap, 7 evals/$2k):")
    for rv in (1, 2, 3):
        P = packs(f, {"OD": 1, "RV": rv, "B2": 1, "FB": 1}, False, 2.)
        # per-account
        rng = np.random.default_rng(7); pays = []; wd = []
        for _ in range(40000):
            a = fresh(); p = 0; w = 0.; n = len(P); blew = False
            for d in range(504):
                st, x = step(a, P[rng.integers(0, n)])
                if x: w += x; p += 1
                if st in ("blow", "maxed"): blew = (st == "blow"); break
            pays.append(p); wd.append(w)
        rng = np.random.default_rng(11)
        nets = [farm_year(P, rng) for _ in range(3000)]
        print(f"   RV@{rv}: per-acct E[$]=${np.mean(wd):>6,.0f} ({np.mean(pays):.2f} payouts)  |  "
              f"FARM net=${np.mean(nets):>8,.0f}/yr (median ${np.median(nets):>7,.0f}, "
              f"p25 ${np.percentile(nets,25):>6,.0f})")


if __name__ == "__main__":
    main()

"""Copy-farm milking DYNAMICS — how many accounts are live over time, blow frequency, payout
concurrency, and when (if) the farm reaches the 30-cap. 50k futures, all legs @1 MNQ, RV ATR-70.

Run: python scripts/montecarlo/farm_dynamics.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parent / "results" / "combined_4way_with_mae_1min.csv"
PARQ = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
ET = "America/New_York"; COST = 2.0


def packs():
    df = pd.read_csv(CSV); df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    d = pd.read_parquet(PARQ, columns=["high", "low", "close"])
    if d.index.tz is None: d.index = d.index.tz_localize("UTC")
    d.index = d.index.tz_convert(ET); d = d.sort_index()
    b = d.resample("20min", label="right", closed="right").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    pc = b.close.shift(1); tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean(); ai = atr.index.values.astype("int64"); av = atr.values
    ent = df["ts"].dt.tz_convert(ET).dt.tz_localize(None).values.astype("datetime64[ns]").astype("int64")
    ix = np.searchsorted(ai, ent, "right") - 1
    df["atr"] = np.where(ix >= 0, av[np.clip(ix, 0, len(av) - 1)], np.nan)
    df = df[~((df.strat == "RV") & (df.atr > 70))]
    df["pnl"] = df["pnl_1c"] / 10.; df["flo"] = (-df["mae_1c"]) / 10.
    return [list(zip(g["pnl"], g["flo"])) for _, g in df.groupby("date", sort=True)]


def fresh(): return dict(bal=50000., peak=50000., floor=48000., locked=False, pays=0)
def step(a, pack):
    for pnl, flo in pack:
        if a["bal"] - flo <= a["floor"]: return "blow", 0.
        a["bal"] += pnl - COST
    if a["bal"] > a["peak"]: a["peak"] = a["bal"]
    if not a["locked"]:
        a["floor"] = min(50000., a["peak"] - 2000.)
        if a["floor"] >= 50000.: a["locked"] = True; a["floor"] = 50000.
    if a["bal"] >= 53000.:
        w = a["bal"] - 52000.; a["bal"] -= w; a["pays"] += 1
        return ("maxed" if a["pays"] >= 5 else "alive"), w
    return "alive", 0.


def run(P, rng, days=252, cap=30, start_n=10, evals_per_2k=7, lag=45):
    n = len(P); accts = [fresh() for _ in range(start_n)]; pending = []
    cash = 0.; live = []; pod = []; blows = 0; maxed = 0; r20 = None; r30 = None
    for day in range(days):
        still = []
        for rd, c in pending:
            if rd <= day:
                room = cap - len(accts); accts += [fresh() for _ in range(min(c, max(0, room)))]
            else: still.append((rd, c))
        pending = still
        pack = P[rng.integers(0, n)]; alive = []; npay = 0
        for a in accts:
            st, w = step(a, pack)
            if w: cash += w; npay += 1
            if st == "alive": alive.append(a)
            elif st == "blow": blows += 1
            else: maxed += 1
        accts = alive
        live.append(len(accts)); pod.append(npay)
        if r20 is None and len(accts) >= 20: r20 = day + 1
        if r30 is None and len(accts) >= 30: r30 = day + 1
        inflight = len(accts) + sum(c for _, c in pending)
        while cash >= 2000. and inflight < cap:
            cash -= 2000.; pending.append((day + lag, evals_per_2k)); inflight += evals_per_2k
    return np.array(live), np.array(pod), blows, maxed, r20, r30


def main():
    P = packs(); rng = np.random.default_rng(11); N = 3000
    LIVE = np.zeros((N, 252)); POD = []; blows = []; maxed = []; r20 = []; r30 = []
    for i in range(N):
        lv, pod, bl, mx, a20, a30 = run(P, rng)
        LIVE[i] = lv; POD.append(pod); blows.append(bl); maxed.append(mx)
        r20.append(a20); r30.append(a30)
    print("COPY-FARM DYNAMICS (start 10, 30-cap, all legs @1 MNQ, RV ATR-70; 3000 sims)\n")
    print("Live funded accounts over time (median [p25-p75]):")
    for mo, d in [(1, 20), (2, 41), (3, 62), (6, 125), (9, 188), (12, 251)]:
        col = LIVE[:, d]
        print(f"   month {mo:>2}: {np.median(col):>4.0f}  [{np.percentile(col,25):.0f}-{np.percentile(col,75):.0f}]")
    print(f"\nPeak live (median across sims): {np.median(LIVE.max(axis=1)):.0f}   "
          f"(p25 {np.percentile(LIVE.max(axis=1),25):.0f}, p75 {np.percentile(LIVE.max(axis=1),75):.0f})")
    frac20 = np.mean([x is not None for x in r20]); frac30 = np.mean([x is not None for x in r30])
    md20 = np.median([x for x in r20 if x]) if any(r20) else None
    md30 = np.median([x for x in r30 if x]) if any(r30) else None
    print(f"reach >=20 accts: {frac20*100:.0f}% of sims (median day {md20})")
    print(f"reach  =30 (cap): {frac30*100:.0f}% of sims (median day {md30})")
    print(f"\nBlows/yr: mean {np.mean(blows):.1f}  |  Retired-at-5-payouts/yr: mean {np.mean(maxed):.1f}")
    print(f"  -> of accounts that leave, {np.mean(blows)/(np.mean(blows)+np.mean(maxed))*100:.0f}% blow, "
          f"{np.mean(maxed)/(np.mean(blows)+np.mean(maxed))*100:.0f}% retire at 5 payouts")
    # payout concurrency
    allpod = np.concatenate([np.array(p) for p in POD])
    pay_days = allpod[allpod > 0]
    tot_payouts = allpod.sum()
    print(f"\nPayout concurrency (on days a payout happens):")
    print(f"   accounts paying out per payout-day: mean {pay_days.mean():.1f}, median {np.median(pay_days):.0f}, max {pay_days.max():.0f}")
    big = (allpod >= 5)
    print(f"   {100*allpod[big].sum()/tot_payouts:.0f}% of all $ payouts happen on days where >=5 accounts pay out together")
    print(f"   (copy-trade = same trades -> same-age accounts pay/blow in lockstep)")


if __name__ == "__main__":
    main()

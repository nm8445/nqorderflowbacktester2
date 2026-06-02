"""Faithful firing-order farm: each signal -> one fresh account (gamble, sized so a stop-out blows it,
risk = current buffer). Gamble draws are INDEPENDENT + frequency-weighted (draw from all trades ->
OD-heavy). Profit-survivor -> Phase 2 (1 MNQ 4-way, SHARED pack = correlated, 5 winning days ->
withdraw down to +$2k). Small-loss survivor keeps gambling. Blow -> dead, re-bought (reinvest).

Reports the real mean/median/p25/p10 and gamble-vs-milk dynamics.
Run: python scripts/montecarlo/farm_firing_order.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

RN = Path(__file__).resolve().parents[2] / "scripts" / "cfd prop firms" / "_risknorm_trades.csv"
MAE = Path(__file__).resolve().parent / "results" / "combined_4way_with_mae_1min.csv"
PARQ = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
ET = "America/New_York"
EVAL, GCOST, WIN_DAY = 165., 30., 50.


def gamble_pool():
    d = pd.read_csv(RN)
    d = d[~((d.strat == "RV") & (d.stop_pts > 300))]   # match live RV ATR-150 filter (RV stop = 2xATR)
    return np.column_stack([(d.pnl_pts / d.stop_pts).values, (d.mae_pts / d.stop_pts).values])  # freq-weighted


def milk_packs():
    df = pd.read_csv(MAE); df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    b = pd.read_parquet(PARQ, columns=["high", "low", "close"])
    if b.index.tz is None: b.index = b.index.tz_localize("UTC")
    b.index = b.index.tz_convert(ET); b = b.sort_index()
    bb = b.resample("20min", label="right", closed="right").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    pc = bb.close.shift(1); tr = pd.concat([bb.high - bb.low, (bb.high - pc).abs(), (bb.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean(); ai = atr.index.values.astype("int64"); av = atr.values
    ent = df["ts"].values.astype("int64")        # UTC ns (FIX: was ET wall-clock vs UTC bars -> wrong ATR)
    ix = np.searchsorted(ai, ent, "right") - 1
    df["atr"] = np.where(ix >= 0, av[np.clip(ix, 0, len(av) - 1)], np.nan)
    df = df[~((df.strat == "RV") & (df.atr > 150))]   # corrected ATR + corrected threshold
    df["pnl"] = df["pnl_1c"] / 10.; df["flo"] = (-df["mae_1c"]) / 10.
    return [list(zip(g["pnl"], g["flo"])) for _, g in df.groupby("date", sort=True)]


def newg(): return dict(st="gamble", bal=50000., peak=50000., floor=48000., locked=False, wd=0)


def run(G, P, rng, days=252, cap=30, start=10, evals_per_2k=7, lag=45):
    ng, npk = len(G), len(P)
    accts = [newg() for _ in range(start)]; pending = []
    cash = 0.; wd = 0.; ev = 0.; gam = []; mlk = []
    for day in range(days):
        still = []
        for rd, c in pending:
            if rd <= day:
                room = cap - len(accts); accts += [newg() for _ in range(min(c, max(0, room)))]
            else: still.append((rd, c))
        pending = still
        pack = P[rng.integers(0, npk)]; alive = []; ngam = 0; nmlk = 0
        for a in accts:
            if a["st"] == "gamble":
                ngam += 1
                pnl_r, mae_r = G[rng.integers(0, ng)]
                risk = a["bal"] - a["floor"]
                # hard stop at yellow: touched yellow (mae>=1R) OR realized >= buffer -> blow, loss capped at -risk
                if mae_r >= 1.0 or pnl_r <= -1.0: continue
                a["bal"] += pnl_r * risk - GCOST
                if a["bal"] > a["peak"]: a["peak"] = a["bal"]
                if not a["locked"]:
                    a["floor"] = min(50000., a["peak"] - 2000.)
                    if a["floor"] >= 50000.: a["locked"] = True; a["floor"] = 50000.
                if a["bal"] > 50000.: a["st"] = "milk"; a["wd"] = 0   # in profit -> de-risk
                alive.append(a)
            else:
                nmlk += 1; d0 = a["bal"]; dead = False
                for pnl, flo in pack:
                    if a["bal"] - flo <= a["floor"]: dead = True; break
                    a["bal"] += pnl - 2.
                if dead: continue
                if a["bal"] > a["peak"]: a["peak"] = a["bal"]
                if not a["locked"]:
                    a["floor"] = min(50000., a["peak"] - 2000.)
                    if a["floor"] >= 50000.: a["locked"] = True; a["floor"] = 50000.
                if a["bal"] - d0 >= WIN_DAY: a["wd"] += 1
                if a["wd"] >= 5 and a["bal"] > 52000.:
                    w = a["bal"] - 52000.; wd += w; cash += w; a["bal"] -= w; a["wd"] = 0
                alive.append(a)
        accts = alive; gam.append(ngam); mlk.append(nmlk)
        inflight = len(accts) + sum(c for _, c in pending)
        while cash >= 2000. and inflight < cap:
            cash -= 2000.; ev += 2000.; pending.append((day + lag, evals_per_2k)); inflight += evals_per_2k
    return wd - ev, np.mean(gam), np.mean(mlk)


def main():
    G = gamble_pool(); P = milk_packs(); N = 2500
    rng = np.random.default_rng(11); nets = []; gs = []; ms = []
    for _ in range(N):
        net, g, m = run(G, P, rng); nets.append(net); gs.append(g); ms.append(m)
    nets = np.array(nets)
    print("FIRING-ORDER farm (gamble independent, milk 4-way correlated; start 10 -> 30 cap)\n")
    print(f"  NET ${nets.mean():,.0f}/yr  median ${np.median(nets):,.0f}  "
          f"p25 ${np.percentile(nets,25):,.0f}  p10 ${np.percentile(nets,10):,.0f}  P(net<=0) {np.mean(nets<=0)*100:.0f}%")
    print(f"  avg accounts gambling/day {np.mean(gs):.1f}, milking/day {np.mean(ms):.1f}")
    print("\n  vs copy-milk farm: p25 $0, P(<=0) 31%  |  vs fully-independent gamble-milk: p25 ~$49k")


if __name__ == "__main__":
    main()

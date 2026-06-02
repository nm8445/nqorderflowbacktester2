"""Per-TRADE staggering (the user's actual rule): pairs activate at trades 1,2,3,4,5 — processed
trade-by-trade — vs fully-synced (all at trade 1). Does it lift the p25 off $0 / spread the payouts?
50k futures milking, all legs @1 MNQ, RV ATR-70.  Run: python scripts/montecarlo/farm_stagger_test.py
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


def newacct(start): return dict(bal=50000., peak=50000., floor=48000., locked=False, pays=0,
                                start=start, active=(start == 0), dead=None)


def eod(a):
    """end-of-day floor/payout update; returns withdrawal (0 if none)."""
    if a["bal"] > a["peak"]: a["peak"] = a["bal"]
    if not a["locked"]:
        a["floor"] = min(50000., a["peak"] - 2000.)
        if a["floor"] >= 50000.: a["locked"] = True; a["floor"] = 50000.
    if a["bal"] >= 53000.:
        w = a["bal"] - 52000.; a["bal"] -= w; a["pays"] += 1
        if a["pays"] >= 5: a["dead"] = "maxed"
        return w
    return 0.


def run(P, rng, stagger, days=252, cap=30, evals_per_2k=7, lag=45):
    n = len(P)
    starts = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] if stagger else [1] * 10
    accts = [newacct(s) for s in starts]   # initial 10 (activate at their trade)
    pending = []; cash = 0.; wd = 0.; ev = 0.; pod = []; gtc = 0
    for day in range(days):
        # reinvested evals mature -> active immediately
        still = []
        for rd, c in pending:
            if rd <= day:
                room = cap - sum(1 for a in accts if not a["dead"])
                for _ in range(min(c, max(0, room))): accts.append(newacct(0))
            else: still.append((rd, c))
        pending = still
        pack = P[rng.integers(0, n)]; npay = 0
        for pnl, flo in pack:
            gtc += 1
            for a in accts:                       # activate initials reaching their trade
                if not a["active"] and a["start"] == gtc: a["active"] = True
            for a in accts:
                if a["active"] and not a["dead"]:
                    if a["bal"] - flo <= a["floor"]: a["dead"] = "blow"
                    else: a["bal"] += pnl - COST
        for a in accts:                            # EOD
            if a["active"] and not a["dead"]:
                w = eod(a)
                if w: cash += w; wd += w; npay += 1
        accts = [a for a in accts if not a["dead"]]
        pod.append(npay)
        inflight = sum(1 for a in accts if not a["dead"]) + sum(c for _, c in pending)
        while cash >= 2000. and inflight < cap:
            cash -= 2000.; ev += 2000.; pending.append((day + lag, evals_per_2k)); inflight += evals_per_2k
    return wd - ev, np.array(pod)


def main():
    P = packs(); N = 3000
    print("Per-TRADE stagger (pairs @ trades 1-5) vs SYNCED (all @ trade 1) — 50k milking farm\n")
    print(f"{'config':>16} {'mean net':>9} {'median':>8} {'p25':>7} {'p10':>7} {'P(net<=0)':>10} "
          f"{'pay/payout-day':>15} {'%$ in >=5-clusters':>18}")
    for label, stag in [("SYNCED", False), ("PER-TRADE stagger", True)]:
        rng = np.random.default_rng(11); nets = []; pods = []
        for _ in range(N):
            net, pod = run(P, rng, stag); nets.append(net); pods.append(pod)
        nets = np.array(nets); allp = np.concatenate(pods); tot = allp.sum(); payd = allp[allp > 0]
        clustered = allp[allp >= 5].sum() / tot * 100
        print(f"{label:>16} ${nets.mean():>8,.0f} ${np.median(nets):>7,.0f} ${np.percentile(nets,25):>6,.0f} "
              f"${np.percentile(nets,10):>6,.0f} {np.mean(nets<=0)*100:>9.0f}% {payd.mean():>14.1f} {clustered:>17.0f}%")
    print("\nAfter trade 5 (~day 2) ALL 10 take every trade identically for the remaining ~250 days,")
    print("and each payout resets to +$2k -> the 5-trade head start is tiny vs 250 days of lockstep.")


if __name__ == "__main__":
    main()

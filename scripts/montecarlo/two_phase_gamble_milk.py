"""Two-phase 'gamble then milk', 1 strat per account (DE-CORRELATED):
  Phase 1: risk ~$2k/trade (one stop-out blows the 50k account), one trade/day, until either blow
           or the account has built a >= $2k cushion (a 'thousands' win).
  Phase 2: drop to 1 MNQ, trade the 4-way combined, take payouts at +$3k (down to +$2k buffer),
           max 5 payouts. If the Phase-1 cushion already >= $3k, take an immediate payout.

Phase-1 trade outcomes use risk-normalized per-strat data (_risknorm_trades.csv: pnl_R, mae_R).
De-correlation: each account runs a different strat on different days -> independent paths, so the
farm p25 should clear $0 (unlike copy-trading the same trade). Run:
    python scripts/montecarlo/two_phase_gamble_milk.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

RN = Path(__file__).resolve().parents[2] / "scripts" / "cfd prop firms" / "_risknorm_trades.csv"
MAE = Path(__file__).resolve().parent / "results" / "combined_4way_with_mae_1min.csv"
PARQ = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
ET = "America/New_York"
RISK, SWITCH, COST_P1, EVAL = 2000., 2000., 30., 165.


def strat_trades():
    d = pd.read_csv(RN)
    d["pnl_R"] = d["pnl_pts"] / d["stop_pts"]; d["mae_R"] = d["mae_pts"] / d["stop_pts"]
    return {s: d[d.strat == s][["pnl_R", "mae_R"]].values for s in ["OD", "RV", "B2", "FB"]}


def fourway_packs():
    df = pd.read_csv(MAE); df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    b = pd.read_parquet(PARQ, columns=["high", "low", "close"])
    if b.index.tz is None: b.index = b.index.tz_localize("UTC")
    b.index = b.index.tz_convert(ET); b = b.sort_index()
    bb = b.resample("20min", label="right", closed="right").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    pc = bb.close.shift(1); tr = pd.concat([bb.high - bb.low, (bb.high - pc).abs(), (bb.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean(); ai = atr.index.values.astype("int64"); av = atr.values
    ent = df["ts"].dt.tz_convert(ET).dt.tz_localize(None).values.astype("datetime64[ns]").astype("int64")
    ix = np.searchsorted(ai, ent, "right") - 1
    df["atr"] = np.where(ix >= 0, av[np.clip(ix, 0, len(av) - 1)], np.nan)
    df = df[~((df.strat == "RV") & (df.atr > 70))]
    df["pnl"] = df["pnl_1c"] / 10.; df["flo"] = (-df["mae_1c"]) / 10.
    return [list(zip(g["pnl"], g["flo"])) for _, g in df.groupby("date", sort=True)]


def phase1(T, rng, cap=40):
    bal = 50000.; peak = 50000.; floor = 48000.; n = len(T)
    for _ in range(cap):
        pnl_r, mae_r = T[rng.integers(0, n)]
        if bal - mae_r * RISK <= floor:
            return None                      # blew the gamble
        bal += pnl_r * RISK - COST_P1
        if bal > peak: peak = bal
        floor = min(50000., peak - 2000.)
        if bal - 50000. >= SWITCH:
            return bal - 50000.              # cushion -> Phase 2
    return max(bal - 50000., 0.) if bal - 50000. >= SWITCH else None


def phase2(P, rng, cushion):
    bal = 50000. + cushion; peak = bal; floor = 50000. if cushion >= 2000. else bal - 2000.
    locked = cushion >= 2000.; pays = 0; wd = 0.; n = len(P)
    if cushion >= 3000.:                      # immediate payout
        w = bal - 52000.; wd += w; bal -= w; pays += 1
    for _ in range(504):
        for pnl, flo in P[rng.integers(0, n)]:
            if bal - flo <= floor: return wd
            bal += pnl - 2.
        if bal > peak: peak = bal
        if not locked:
            floor = min(50000., peak - 2000.)
            if floor >= 50000.: locked = True; floor = 50000.
        if bal >= 53000.:
            w = bal - 52000.; wd += w; bal -= w; pays += 1
            if pays >= 5: return wd
    return wd


def main():
    ST = strat_trades(); P4 = fourway_packs()
    print("TWO-PHASE gamble->milk, 1 strat/account (risk $2k -> switch at +$2k -> milk 1MNQ)\n")
    print(f"{'strat':>6} {'P(reach P2)':>12} {'avg cushion':>12} {'P2 milk$':>9} {'$/funded acct':>14} {'$/eval spent':>13}")
    perstrat = {}
    for s, T in ST.items():
        rng = np.random.default_rng(7); reach = 0; cush = []; ext = []
        for _ in range(40000):
            c = phase1(T, rng)
            if c is None: ext.append(0.); continue
            reach += 1; cush.append(c)
            ext.append(phase2(P4, rng, c))
        pr = reach / 40000; perA = np.mean(ext)
        # $/eval: each attempt costs EVAL; perA is per funded acct that entered phase1
        per_eval = perA - EVAL
        perstrat[s] = (pr, perA)
        print(f"{s:>6} {pr*100:>11.0f}% ${np.mean(cush):>10,.0f} ${np.mean([phase2(P4,np.random.default_rng(3),c) for c in cush[:2000]]):>8,.0f} "
              f"${perA:>13,.0f} ${per_eval:>12,.0f}")
    # ---- farm-level p25 (de-correlated): 30 accts split across strats, independent ----
    print("\nFarm-level variance check (30 accts, evenly split OD/RV/B2/FB, INDEPENDENT draws):")
    rng = np.random.default_rng(11); years = []
    str016 = list(ST.items())
    for _ in range(3000):
        tot = 0.
        for i in range(30):
            s, T = str016[i % 4]
            c = phase1(T, rng)
            tot += (phase2(P4, rng, c) - EVAL) if c is not None else -EVAL
        years.append(tot)
    years = np.array(years)
    print(f"   30-acct yr (single pass, no re-buy): mean ${years.mean():,.0f}  median ${np.median(years):,.0f}  "
          f"p25 ${np.percentile(years,25):,.0f}  p10 ${np.percentile(years,10):,.0f}  P(<=0) {np.mean(years<=0)*100:.0f}%")
    print("   (copy-milk farm p25 was $0 / P(<=0) ~31% — compare the de-correlated p25 above)")


if __name__ == "__main__":
    main()

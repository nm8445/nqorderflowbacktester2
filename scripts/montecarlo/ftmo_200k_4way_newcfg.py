"""FTMO 200k FUNDED income MC for the 4-way combined, NEW configs (OD-1hr + FB-giveback).

Account rules modelled
  balance          $200,000
  max daily loss   5%  = $10,000, measured on EQUITY (floating counts) vs day-start balance
  max total loss   10% = $20,000 static -> hard floor $180,000, equity-based
  per-trade risk    1% = $2,000  (the user's own rule -> becomes the SIZING BUDGET)
  one position at a time, no hedging  (this also makes "one-sided betting" un-trippable)
  profit split     80%, payout cycle every PAYOUT_CYCLE trading days, $ min payout

Sizing: CONTINUOUS (fractional) MNQ, because the user trades CFD lots and is not restricted to
whole futures contracts. size_MNQ = BUDGET / worst_MAE_per_MNQ.  MAE is per-contract so it
scales linearly. Three budget bases are reported (worst / p99.5 / p99 observed MAE).

Data: results/combined_4way_newcfg_with_mae.csv  (built by build_4way_mae_newcfg.py; MAE anchored
on 1-min bars with per-leg calibrated fill offsets, RV ATR_MAX=150 applied).

Run:  python scripts/montecarlo/ftmo_200k_4way_newcfg.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "scripts" / "montecarlo" / "results" / "combined_4way_newcfg_with_mae.csv"

START = 200_000.0
DLL_PCT = 0.05                 # 5% daily, equity-based, vs day-start balance
FLOOR = START * 0.90           # 10% static max loss
BUDGET = 2_000.0               # 1% per-trade risk rule
SPLIT = 0.80
PAYOUT_CYCLE = 20              # trading days between payouts
MIN_PAYOUT = 200.0
COST_PER_MNQ = 4.0             # round-turn commission+slippage, $/MNQ
HORIZON = 252
NSIM = 20_000
STRATS = ["OD", "RV", "B2", "FB"]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    # Legs cover slightly different spans; restrict to the window where ALL FOUR are live so
    # the day-bootstrap isn't diluted by days on which a leg simply had no data.
    lo = max(df[df.strat == s]["ts"].min() for s in STRATS)
    hi = min(df[df.strat == s]["exit_ts"].max() for s in STRATS)
    out = df[(df.ts >= lo) & (df.exit_ts <= hi)].copy()
    print(f"common window: {lo.date()} .. {hi.date()}   {len(out)} trades "
          f"(dropped {len(df)-len(out)} outside)")
    return out


def sizes(df: pd.DataFrame, basis: str) -> pd.Series:
    """MNQ per strat so the chosen MAE quantile == BUDGET. mae_1c is $ per 1 NQ = 10 MNQ."""
    m = df.groupby("strat")["mae_1c"].apply(
        lambda x: x.abs().max() if basis == "worst" else x.abs().quantile(float(basis)))
    return BUDGET / (m / 10.0)


def conflict_filter(df: pd.DataFrame, priority=None) -> pd.DataFrame:
    """One position at a time. FCFS by default; `priority` lets a listed strat pre-empt nothing
    (we never close early) but lets us DROP a leg entirely to test blocking cost."""
    d = df if priority is None else df[df.strat.isin(priority)]
    keep, last_exit = [], pd.Timestamp.min.tz_localize("UTC")
    for idx, ts, xt in zip(d.index, d["ts"], d["exit_ts"]):
        if ts < last_exit:
            continue
        keep.append(idx)
        last_exit = xt
    return d.loc[keep].copy()


def ftmo_day(ts: pd.Series) -> pd.Series:
    """FTMO's daily-loss window resets at 00:00 CE(S)T == 18:00 ET year-round.
    So an OD fill at 19:59 ET on day D belongs to FTMO-day D+1 -- the SAME daily budget as the
    RTH trades it overlaps the next morning. Shifting ET by +6h maps 18:00 -> midnight."""
    return (ts.dt.tz_convert("America/New_York") + pd.Timedelta(hours=6)).dt.date


def make_packs(f: pd.DataFrame, mnq: pd.Series):
    g = f.copy()
    g["n"] = g["strat"].map(mnq)
    g["pnl"] = g["pnl_1c"] * g["n"] / 10.0 - g["n"] * COST_PER_MNQ
    g["flo"] = g["mae_1c"].abs() * g["n"] / 10.0
    g["d"] = ftmo_day(g["ts"])
    return [list(zip(x["pnl"].values, x["flo"].values)) for _, x in g.groupby("d", sort=True)]


def sim(packs, rng, horizon=HORIZON):
    n = len(packs)
    bal, cash, since, pays = START, 0.0, 0, 0
    for day in range(horizon):
        dfloor = bal * (1 - DLL_PCT)
        real = 0.0
        for pnl, flo in packs[rng.integers(0, n)]:
            eq = bal + real - flo                 # worst floating point of the trade
            if eq <= FLOOR:
                return cash, pays, "MaxLoss", day + 1
            if eq <= dfloor:
                return cash, pays, "DLL", day + 1
            real += pnl
        bal += real
        since += 1
        if since >= PAYOUT_CYCLE:
            prof = bal - START
            if prof > MIN_PAYOUT:
                cash += prof * SPLIT
                bal = START                       # withdraw profit, reset to nominal
                pays += 1
            since = 0
    return cash, pays, None, horizon


def run(packs, label, nsim=NSIM):
    rng = np.random.default_rng(7)
    res = [sim(packs, rng) for _ in range(nsim)]
    cash = np.array([r[0] for r in res])
    busts = [r[2] for r in res]
    blow = np.mean([b is not None for b in busts])
    print(f"{label:<34}{cash.mean():>11,.0f}{np.median(cash):>11,.0f}"
          f"{np.percentile(cash,10):>10,.0f}{np.percentile(cash,90):>11,.0f}{blow*100:>9.1f}%"
          f"   DLL {np.mean([b=='DLL' for b in busts])*100:4.1f}%  "
          f"Max {np.mean([b=='MaxLoss' for b in busts])*100:4.1f}%")
    return cash.mean(), blow


def main():
    df = load()

    print("\n=== per-strat floating MAE, $ per 1 MNQ ===")
    hdr = f"{'':<4}{'n':>6}{'med':>8}{'p95':>8}{'p99':>8}{'p99.5':>9}{'worst':>9}"
    print(hdr); print("-" * len(hdr))
    for s in STRATS:
        m = df[df.strat == s]["mae_1c"].abs() / 10.0
        print(f"{s:<4}{len(m):>6}{m.median():>8,.0f}{m.quantile(.95):>8,.0f}"
              f"{m.quantile(.99):>8,.0f}{m.quantile(.995):>9,.0f}{m.max():>9,.0f}")

    print(f"\n=== sizing so MAE == ${BUDGET:,.0f} (1% of ${START:,.0f}) ===")
    print(f"{'basis':<10}" + "".join(f"{s:>10}" for s in STRATS))
    print("-" * 50)
    tbl = {}
    for basis in ["worst", "0.995", "0.99"]:
        z = sizes(df, basis)
        tbl[basis] = z
        print(f"{basis:<10}" + "".join(f"{z[s]:>10.2f}" for s in STRATS))

    # How often would the 1% rule actually be breached at each sizing basis?
    print(f"\n=== residual breaches of the ${BUDGET:,.0f} cap (in-sample, {len(df)} trades) ===")
    yrs = (df["ts"].max() - df["ts"].min()).days / 365.25
    print(f"{'basis':<10}{'breaches':>10}{'of trades':>11}{'per year':>10}   worst breach")
    for basis in ["worst", "0.995", "0.99"]:
        z = tbl[basis]
        fl = df["mae_1c"].abs() / 10.0 * df["strat"].map(z)
        nb = int((fl > BUDGET + 1e-6).sum())
        print(f"{basis:<10}{nb:>10}{100*nb/len(df):>10.2f}%{nb/yrs:>10.1f}   "
              f"${fl.max():>7,.0f}  ({fl.max()/BUDGET:.2f}x cap)")

    # how much does the one-position rule cost?
    print("\n=== one-position-at-a-time conflict cost (FCFS) ===")
    raw = df.groupby("strat").size()
    filt = conflict_filter(df)
    kept = filt.groupby("strat").size()
    print(f"{'':<5}{'raw':>7}{'kept':>7}{'kept%':>8}{'net_raw$':>12}{'net_kept$':>12}")
    for s in STRATS:
        nr, nk = raw.get(s, 0), kept.get(s, 0)
        print(f"{s:<5}{nr:>7}{nk:>7}{100*nk/max(nr,1):>7.1f}%"
              f"{df[df.strat==s]['pnl_1c'].sum():>12,.0f}{filt[filt.strat==s]['pnl_1c'].sum():>12,.0f}")

    print(f"\n{'config':<34}{'mean$':>11}{'median$':>11}{'p10':>10}{'p90':>11}{'blow':>9}")
    print("-" * 96)
    for basis in ["worst", "0.995", "0.99"]:
        packs = make_packs(filt, tbl[basis])
        run(packs, f"4-way 1-at-a-time  MAE={basis}")

    # Is OD-1hr worth its blocking cost under the 1-position rule?
    print()
    for subset in [["OD"], ["RV", "B2", "FB"], ["OD", "RV", "B2", "FB"]]:
        f2 = conflict_filter(df, priority=subset)
        packs = make_packs(f2, sizes(df, "worst"))
        run(packs, "subset " + "+".join(subset))


if __name__ == "__main__":
    main()

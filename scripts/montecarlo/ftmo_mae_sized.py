"""FTMO EV with MAE-BASED sizing (the user's actual idea).

Instead of sizing each strat so its STOP risks a fixed $, size each strat so its historical MAE
(worst adverse excursion) only floats ~$BUDGET against you. Then a single trade essentially can't
threaten the account, and breaching the daily/max DD needs a pileup that almost never happens ->
super-low blow rate, while still extracting a good amount.

Per strat: C = BUDGET / (mae_ref_pts * $2/pt[MNQ]), where mae_ref_pts = the chosen MAE percentile.
  - p95  = "likely worst" (5% of trades float a bit more)
  - max  = no historical trade ever floated past $BUDGET (ultra-safe)

Account rules (FTMO, static floor, floating-aware, biweekly 80%, no RPTI):
  100K: $5k daily / $10k max DD, $1,000 MAE budget.
  200K: $10k daily / $20k max DD, $2,000 MAE budget.
Withdraw profit above (start + buffer) each cycle; leave `buffer` working to stand off the static floor.

Run: python scripts/montecarlo/ftmo_mae_sized.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RN = ROOT / "scripts" / "cfd prop firms" / "_risknorm_trades.csv"
TM = ROOT / "scripts" / "futurespropmc" / "results" / "combined_4way_with_mae_1min.csv"
ET = "America/New_York"
INTRADAY = {"FB", "RV", "B2"}
MNQ_PT = 2.0           # $/pt per MNQ
CYCLE_TD, HORIZON, SPLIT, N_SIMS = 10, 252, 0.80, 20_000


def build_packs():
    """Daily packs of (strat, pnl_pts, mae_pts) after 1-trade-at-a-time (FB/RV/B2); OD always in."""
    rn = pd.read_csv(RN); tm = pd.read_csv(TM)
    rn["date"] = pd.to_datetime(rn["date"]).dt.date
    tm["entry"] = pd.to_datetime(tm["ts"], utc=True).dt.tz_convert(ET)
    tm["exit"] = pd.to_datetime(tm["exit_ts"], utc=True).dt.tz_convert(ET)
    tm["date"] = tm["entry"].dt.date
    rn = rn.sort_values(["date", "strat"]).reset_index(drop=True)
    tm = tm.sort_values(["date", "strat", "entry"]).reset_index(drop=True)
    rn["k"] = rn.groupby(["date", "strat"]).cumcount()
    tm["k"] = tm.groupby(["date", "strat"]).cumcount()
    m = tm.merge(rn[["date", "strat", "k", "pnl_pts", "mae_pts"]], on=["date", "strat", "k"], how="left")
    assert m["pnl_pts"].notna().all()
    m["mae_abs"] = m["mae_pts"].abs()
    packs = []
    for _, g in m.groupby("date"):
        g = g.sort_values("entry"); taken = []; busy = None
        for _, r in g.iterrows():
            if r["strat"] in INTRADAY:
                if busy is not None and r["entry"] < busy:
                    continue
                busy = r["exit"]
            taken.append((r["entry"], r["strat"], float(r["pnl_pts"]), float(r["mae_abs"])))
        taken.sort(key=lambda x: x[0])
        packs.append([(s, p, mae) for _, s, p, mae in taken])
    return packs, m


def strat_sizes(m, budget, pct):
    """Contracts per strat so the chosen MAE percentile floats ~$budget."""
    sizes = {}
    for s, g in m.groupby("strat"):
        ref = g["mae_abs"].max() if pct == "max" else g["mae_abs"].quantile(pct)
        sizes[s] = budget / (ref * MNQ_PT)
    return sizes


def precompute(packs, sizes):
    """Convert each pack to ($pnl, $mae-floating) using fixed per-strat contracts."""
    out = []
    for pk in packs:
        out.append([(p * sizes[s] * MNQ_PT, -mae * sizes[s] * MNQ_PT) for s, p, mae in pk])
    return out


def simulate(dpacks, start, dll, maxdd, buffer, rng):
    floor = start - maxdd; bal = start; since = 0; pay = 0; cash = 0.0; n = len(dpacks)
    for d in range(HORIZON):
        pack = dpacks[rng.integers(0, n)]; ds = bal; realized = 0.0
        for pnl_d, mae_d in pack:
            fl = bal + realized + mae_d
            if fl < ds - dll:  return dict(b=1, r="DLL", day=d, pay=pay, cash=cash)
            if fl < floor:     return dict(b=1, r="MAX", day=d, pay=pay, cash=cash)
            realized += pnl_d
        bal += realized; since += 1
        if since >= CYCLE_TD:
            wd = bal - (start + buffer)
            if wd > 0:
                cash += wd * SPLIT; bal -= wd; pay += 1
            since = 0
    return dict(b=0, r=None, day=None, pay=pay, cash=cash)


def run(name, start, dll, maxdd, packs, m, budget, buffer):
    print(f"\n############ {name}  |  ${budget:,.0f} MAE-budget/trade  |  "
          f"${dll:,.0f} daily / ${maxdd:,.0f} max  |  leave ${buffer:,.0f} cushion ############")
    print(f"{'MAE pctile':>11} {'sizes (MNQ)':>34} {'blow%':>7} {'DLL%':>6} {'medday':>7} "
          f"{'EVcash':>9} {'medcash':>9} {'pays':>6}")
    for pct in (0.90, 0.95, 0.99, "max"):
        sizes = strat_sizes(m, budget, pct)
        dpacks = precompute(packs, sizes)
        rng = np.random.default_rng(42)
        S = [simulate(dpacks, start, dll, maxdd, buffer, rng) for _ in range(N_SIMS)]
        b = np.array([s["b"] for s in S])
        dll_sh = sum(s["r"] == "DLL" for s in S) / max(b.sum(), 1)
        days = [s["day"] for s in S if s["b"]]
        cash = np.array([s["cash"] for s in S]); pays = np.array([s["pay"] for s in S])
        ss = "  ".join(f"{k}:{v:.1f}" for k, v in sizes.items())
        plabel = "max" if pct == "max" else f"p{int(pct*100)}"
        print(f"{plabel:>11} {ss:>34} {b.mean()*100:>6.1f}% {dll_sh*100:>5.0f}% "
              f"{(np.median(days) if days else 0):>7.0f} {cash.mean():>9,.0f} "
              f"{np.median(cash):>9,.0f} {pays.mean():>6.1f}")


def main():
    packs, m = build_packs()
    print(f"Built {len(packs)} daily packs. MAE-based sizing (cap floating, not stop).")
    run("FTMO 200K", 200_000, 10_000, 20_000, packs, m, budget=2_000, buffer=10_000)
    run("FTMO 100K", 100_000, 5_000, 10_000, packs, m, budget=1_000, buffer=5_000)


if __name__ == "__main__":
    main()

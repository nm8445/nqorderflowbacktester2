"""FTMO 100K & 200K EV — risk-normalized, 1-trade-at-a-time.

User's exact rules:
  100K: $5,000 daily DD, $10,000 max DD (STATIC floor at start-10k), $1,000 risk/trade.
  200K: $10,000 daily DD, $20,000 max DD (static floor at start-20k), $2,000 risk/trade.
  One trade at a time: FB / RV / B2 only ONE active at once (greedy non-overlap by entry time).
  OD is overnight (19:00->08:00) and never clashes with the intraday session -> always taken.
  Payout: FTMO bi-weekly (~10 trading days/cycle), 80% split, withdraw all profit above start,
          balance resets to start. NO RPTI / floating-profit kill (unlike FundingPips/FundedNext).
  DD is floating-aware: intraday MAE counts toward both the daily and the static-max limits.

Note on R-geometry: 100K has 5R daily / 10R max DD at $1k/R; 200K has 5R daily / 10R max DD at
$2k/R. Identical risk geometry -> 200K EV is exactly 2x the 100K EV (only the dollar scale differs).

Data: risk-normalized trades (pnl_R, mae_R, stop) joined to the timed 4-way combined (entry/exit).

Run: python scripts/montecarlo/ftmo_ev_onetrade.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RN = ROOT / "scripts" / "cfd prop firms" / "_risknorm_trades.csv"
TM = ROOT / "scripts" / "futurespropmc" / "results" / "combined_4way_with_mae_1min.csv"
ET = "America/New_York"

CYCLE_TD = 10          # trading days per bi-weekly payout cycle
HORIZON = 252          # trading days simulated (1 year)
SPLIT = 0.80           # trader take-home share
N_SIMS = 20_000
INTRADAY = {"FB", "RV", "B2"}   # mutually exclusive (1 active at a time); OD is separate

CONFIGS = {
    "FTMO 100K": dict(start=100_000.0, dll=5_000.0, maxdd=10_000.0, risk=1_000.0),
    "FTMO 200K": dict(start=200_000.0, dll=10_000.0, maxdd=20_000.0, risk=2_000.0),
}


def build_daily_packs() -> list[list[tuple[float, float]]]:
    """Each pack = one historical day's time-ordered list of taken (pnl_R, mae_R)."""
    rn = pd.read_csv(RN)
    tm = pd.read_csv(TM)
    rn["date"] = pd.to_datetime(rn["date"]).dt.date
    tm["entry"] = pd.to_datetime(tm["ts"], utc=True).dt.tz_convert(ET)
    tm["exit"] = pd.to_datetime(tm["exit_ts"], utc=True).dt.tz_convert(ET)
    tm["date"] = tm["entry"].dt.date

    # join row-by-row within each (date, strat) group (counts verified identical, ET dates align)
    rn = rn.sort_values(["date", "strat"]).reset_index(drop=True)
    tm = tm.sort_values(["date", "strat", "entry"]).reset_index(drop=True)
    rn["k"] = rn.groupby(["date", "strat"]).cumcount()
    tm["k"] = tm.groupby(["date", "strat"]).cumcount()
    m = tm.merge(rn[["date", "strat", "k", "pnl_pts", "mae_pts", "stop_pts"]],
                 on=["date", "strat", "k"], how="left")
    assert m["stop_pts"].notna().all(), "join left some trades unmatched"
    m["pnl_R"] = m["pnl_pts"] / m["stop_pts"]
    m["mae_R"] = -(m["mae_pts"].abs() / m["stop_pts"])   # force negative (worst adverse, in R)

    packs = []
    for _, g in m.groupby("date"):
        g = g.sort_values("entry")
        taken = []
        busy_until = None
        for _, r in g.iterrows():
            if r["strat"] in INTRADAY:
                if busy_until is not None and r["entry"] < busy_until:
                    continue                      # an intraday trade is already live -> skip
                busy_until = r["exit"]
            taken.append((r["entry"], float(r["pnl_R"]), float(r["mae_R"])))
        taken.sort(key=lambda x: x[0])
        packs.append([(p, mae) for _, p, mae in taken])
    return packs


def simulate(packs, cfg, rng):
    start, dll, maxdd, risk = cfg["start"], cfg["dll"], cfg["maxdd"], cfg["risk"]
    floor = start - maxdd                 # STATIC max-loss floor (never moves)
    bal = start
    since = 0
    payouts = 0
    take_home = 0.0
    n = len(packs)
    for d in range(HORIZON):
        pack = packs[rng.integers(0, n)]
        day_start = bal
        realized = 0.0
        for pnl_R, mae_R in pack:
            float_low = bal + realized + mae_R * risk     # worst floating equity this trade
            if float_low < day_start - dll:               # daily loss limit
                return dict(busted=True, reason="DLL", day=d, payouts=payouts, cash=take_home)
            if float_low < floor:                         # static max DD
                return dict(busted=True, reason="MAX", day=d, payouts=payouts, cash=take_home)
            realized += pnl_R * risk
        bal += realized
        since += 1
        if since >= CYCLE_TD:
            profit = bal - start
            if profit > 0:
                take_home += profit * SPLIT
                bal = start
                payouts += 1
            since = 0
    return dict(busted=False, reason=None, day=None, payouts=payouts, cash=take_home)


def main():
    packs = build_daily_packs()
    tcount = [len(p) for p in packs]
    rsum = [sum(x[0] for x in p) for p in packs]
    print(f"Built {len(packs)} daily packs (1-trade-at-a-time for FB/RV/B2, OD overnight separate).")
    print(f"  trades/day: mean {np.mean(tcount):.2f}  max {max(tcount)}   "
          f"daily R: mean {np.mean(rsum):+.3f}  std {np.std(rsum):.3f}\n")

    for name, cfg in CONFIGS.items():
        rng = np.random.default_rng(42)
        sims = [simulate(packs, cfg, rng) for _ in range(N_SIMS)]
        busted = np.mean([s["busted"] for s in sims])
        dll_share = (sum(s["reason"] == "DLL" for s in sims) /
                     max(sum(s["busted"] for s in sims), 1))
        cash = np.array([s["cash"] for s in sims])
        pays = np.array([s["payouts"] for s in sims])
        any_pay = np.mean(pays >= 1)
        print(f"=== {name} | DD {cfg['dll']:,.0f} daily / {cfg['maxdd']:,.0f} max | "
              f"${cfg['risk']:,.0f} risk/trade | 80% split | 1yr ===")
        print(f"  blow rate (1yr) : {busted*100:5.1f}%   ({dll_share*100:.0f}% of blows = daily-limit)")
        print(f"  any payout      : {any_pay*100:5.1f}%   median payouts {int(np.median(pays))}  "
              f"mean {pays.mean():.2f}")
        print(f"  EV take-home/yr : ${cash.mean():,.0f}   (median ${np.median(cash):,.0f})")
        print(f"  p25 / p75       : ${np.percentile(cash,25):,.0f} / ${np.percentile(cash,75):,.0f}")
        print(f"  p10 / p90       : ${np.percentile(cash,10):,.0f} / ${np.percentile(cash,90):,.0f}\n")


if __name__ == "__main__":
    main()

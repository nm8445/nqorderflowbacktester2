"""FundingPips 100K FUNDED account EV — with a $2k max cumulative exposure cap.

After passing the 2-step challenge you get a $100k funded account. This sims its lifetime
value (expected $ withdrawn) using the live 4-way combined with PER-STRAT MAE risk sizing,
where the firm caps your max cumulative open exposure (risk) at $2,000.

Interpretation of the $2k cap: total open-position risk <= $2k at any time. The combined
rarely has >1-2 strats open at once (no-hedge coordinator; OD overnight vs RV/B2/FB daytime),
so we size each trade to a per-strat risk R and report EV across R; the $2k cap means
R ~= $1,000 (room for ~2 concurrent) up to ~$2,000 (single position).

Funded rules: $100k, 3% daily loss (of day-start), 6% static floor ($94k). Payout: FundingPips
bi-weekly 80% split (cycle ~10 td); withdraw all profit above $100k each cycle, balance resets.
Run until bust or 1-year horizon. EV = expected total trader cash per funded account.

Run:  python "scripts/cfd prop firms/fundingpips_funded_ev.py"
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"

START_BAL = 100_000.0
NQ_PT, COST_RT_PER_MNQ = 20.0, 4.0
SL_PTS = {"OD": 600.0, "B2": 600.0, "RV": 200.0, "FB": 150.0}

HORIZON_TD = 252
CYCLE_TD = 10                 # bi-weekly payout (~10 trading days)
SPLIT = 0.80                  # FundingPips 80%
MIN_PAYOUT = 200.0            # don't bother withdrawing trivial profit
N_SIMS = 20_000

EXPOSURE_CAP = 2_000.0
RISK_GRID = [500, 750, 1000, 1500, 2000]    # per-trade $ risk (capped by $2k exposure)
CHALLENGE_FEE = 500.0         # ~FP 100k 2-step fee (adjust to what you pay)

# Two account rule-sets (funded DD + that account's 2-step pass rate at R~$1k)
ACCOUNTS = {
    "A: 3% daily / 6% max ($2k-exposure acct)": dict(dll=0.03, maxloss=0.06, pass2=0.79),
    "B: 5% daily / 10% max (10%/5% targets)":   dict(dll=0.05, maxloss=0.10, pass2=0.86),
}


def load_packs():
    df = pd.read_csv(TRADES_CSV)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    df["d"] = df["exit_ts"].dt.date
    df = df.sort_values(["d", "exit_ts"])
    return [list(zip(g["strat"], g["pnl_$"].astype(float))) for _, g in df.groupby("d", sort=True)]


def simulate_funded(packs, R, dll_pct, floor, rng):
    """Returns total trader cash extracted, n_payouts, busted(bool), days_survived."""
    balance = START_BAL
    days_in_cycle = 0
    cash = 0.0
    payouts = 0
    n = len(packs)
    for day in range(HORIZON_TD):
        day_start = balance
        dll = dll_pct * day_start
        running = 0.0
        busted = False
        for strat, pnl in packs[rng.integers(0, n)]:
            sc = R / (SL_PTS[strat] * NQ_PT)
            running += pnl * sc - sc * 10.0 * COST_RT_PER_MNQ
            eq = day_start + running
            if running <= -dll or eq <= floor:
                busted = True
                break
        if busted:
            return cash, payouts, True, day + 1
        balance = day_start + running
        days_in_cycle += 1
        if days_in_cycle >= CYCLE_TD:
            profit = balance - START_BAL
            if profit >= MIN_PAYOUT:
                cash += profit * SPLIT
                balance = START_BAL
                payouts += 1
            days_in_cycle = 0
    return cash, payouts, False, HORIZON_TD


def main():
    pd.set_option("display.width", 220)
    packs = load_packs()
    print("Funded EV: $100k, bi-weekly 80% payout, 1-yr horizon, per-strat MAE risk sizing.")
    print("$2k max-exposure cap -> R ~ $1,000 (room for ~2 concurrent) .. $2,000 (single position).\n")
    for label, cfg in ACCOUNTS.items():
        floor = START_BAL * (1 - cfg["maxloss"])
        rows = []
        for R in RISK_GRID:
            rng = np.random.default_rng(7 + R)
            res = [simulate_funded(packs, R, cfg["dll"], floor, rng) for _ in range(N_SIMS)]
            cash = np.array([r[0] for r in res])
            busts = np.array([r[2] for r in res])
            net = cfg["pass2"] * cash.mean() - CHALLENGE_FEE
            rows.append({
                "risk_$/trade": R,
                "E[$ withdrawn]/funded": round(cash.mean(), 0),
                "median_$": round(np.median(cash), 0),
                "bust_rate(1yr)": round(busts.mean(), 3),
                "acct_EV(pass+fee)": round(net, 0),
            })
        df = pd.DataFrame(rows)
        print(f"=== {label} | funded floor ${floor:,.0f}, 2-step pass {cfg['pass2']:.0%} ===")
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()

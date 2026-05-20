"""
FTMO & FundingPips 100K funded account Monte Carlo
====================================================
Inputs: live/combined deployment plan/combined_trades.csv (3-way combined, 1 NQ per strat)

Common rules ($100K funded, static DD):
  - Start: $100,000
  - Static max loss: $10,000 (floor at $90,000, NEVER moves up)
  - Daily loss limit: $5,000 (closed + unrealized at midnight; we model as daily realized PnL)
  - Payout cycle: 14 calendar days ~ 10 trading days
  - First payout: only after first full cycle (no early eligibility)
  - No payout cap, no max payout count, no consistency on FTMO funded
  - Trader receives 80% of cycle profit; account balance drops by full withdrawn amount

FundingPips bi-weekly = same 80% split as FTMO so we report one row.
FundingPips also offers Monthly @ 100% — included as a separate scenario.

NOTE on instruments: both firms offer NAS100 CFD (not NQ futures).
1 MNQ ≈ 2 NAS100 lots at $1/pt — daily PnL in $ is unchanged at the MNQ level.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades.csv"

START_BAL = 100_000.0
STATIC_FLOOR = 90_000.0
DLL = 5_000.0                          # daily loss limit
CYCLE_TD = 10                          # trading days per cycle (~14 calendar days)
HORIZON_DAYS = 252
N_SIMS = 5_000

# Per-firm config
FIRMS = {
    "FTMO_biweekly_80":     {"split": 0.80, "cycle_td": 10, "min_withdrawal": 50.0,    "label": "FTMO bi-weekly 80%"},
    "FP_biweekly_80":       {"split": 0.80, "cycle_td": 10, "min_withdrawal": 2_000.0, "label": "FundingPips bi-weekly 80% (2% min)"},
    "FP_monthly_100":       {"split": 1.00, "cycle_td": 22, "min_withdrawal": 2_000.0, "label": "FundingPips monthly 100% (2% min)"},
    "FP_ondemand_90":       {"split": 0.90, "cycle_td": 1,  "min_withdrawal": 2_000.0, "label": "FundingPips on-demand 90% (no consistency model)"},
}


def load_daily_pnl_for_1nq() -> np.ndarray:
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["pnl_$"].sum().sort_index()
    return daily.values.astype(float)


def simulate_one(daily_1nq, mnq, firm_cfg, rng):
    mult = mnq / 10.0
    split = firm_cfg["split"]
    cycle_td = firm_cfg["cycle_td"]
    min_wd = firm_cfg["min_withdrawal"]

    balance = START_BAL
    days_since_cycle_start = 0
    busted = False
    bust_day = None
    bust_reason = None
    payouts = 0
    total_trader_cash = 0.0
    days_to_first_payout = None
    n_pool = len(daily_1nq)

    for d in range(HORIZON_DAYS):
        daily_pnl = daily_1nq[rng.integers(0, n_pool)] * mult

        # DLL check (intraday/EOD realized)
        if daily_pnl <= -DLL:
            busted = True
            bust_day = d
            bust_reason = "DLL"
            break

        balance += daily_pnl

        # Static DD check
        if balance < STATIC_FLOOR:
            busted = True
            bust_day = d
            bust_reason = "MaxLoss"
            break

        days_since_cycle_start += 1

        # Payout at end of cycle
        if days_since_cycle_start >= cycle_td:
            cycle_profit = balance - START_BAL
            if cycle_profit >= min_wd:
                # withdraw entire profit above start
                gross_withdrawn = cycle_profit
                trader_cash = gross_withdrawn * split
                balance -= gross_withdrawn
                payouts += 1
                total_trader_cash += trader_cash
                if days_to_first_payout is None:
                    days_to_first_payout = d
            days_since_cycle_start = 0

    return {
        "busted": busted,
        "bust_day": bust_day,
        "bust_reason": bust_reason,
        "payouts": payouts,
        "total_trader_cash": total_trader_cash,
        "days_to_first_payout": days_to_first_payout,
        "end_balance": balance,
        "avg_per_payout": (total_trader_cash / payouts) if payouts > 0 else 0.0,
    }


def run_grid():
    daily = load_daily_pnl_for_1nq()
    print(f"Loaded {len(daily)} historical days. "
          f"mean=${daily.mean():.0f}/day std=${daily.std():.0f} "
          f"min=${daily.min():.0f} max=${daily.max():.0f} (at 1 NQ per strat)\n")

    rows = []
    for firm_key, cfg in FIRMS.items():
        for mnq in [1, 2, 3, 4, 5]:
            rng = np.random.default_rng(seed=hash(firm_key) % 10000 + mnq)
            sims = [simulate_one(daily, mnq, cfg, rng) for _ in range(N_SIMS)]

            bust_rate = np.mean([s["busted"] for s in sims])
            dll_share = (sum(1 for s in sims if s["bust_reason"] == "DLL") /
                         max(sum(1 for s in sims if s["busted"]), 1))
            payouts = [s["payouts"] for s in sims]
            cash = [s["total_trader_cash"] for s in sims]
            t1 = [s["days_to_first_payout"] for s in sims if s["days_to_first_payout"] is not None]

            any_payout = np.mean([s["payouts"] >= 1 for s in sims])
            six_plus = np.mean([s["payouts"] >= 6 for s in sims])
            avg_per_pmt = np.mean([s["avg_per_payout"] for s in sims if s["payouts"] >= 1]) if any_payout > 0 else 0

            rows.append({
                "firm": firm_key,
                "mnq": mnq,
                "bust_rate": bust_rate,
                "dll_share_of_busts": dll_share if bust_rate > 0 else 0,
                "any_payout_rate": any_payout,
                "ge6_payouts_rate": six_plus,
                "median_payouts": int(np.median(payouts)),
                "mean_payouts": np.mean(payouts),
                "median_total_$": np.median(cash),
                "mean_total_$": np.mean(cash),
                "p25_total_$": np.percentile(cash, 25),
                "p75_total_$": np.percentile(cash, 75),
                "avg_per_payout_$": avg_per_pmt,
                "median_days_to_1st": int(np.median(t1)) if t1 else None,
                "p25_days_to_1st": int(np.percentile(t1, 25)) if t1 else None,
                "p75_days_to_1st": int(np.percentile(t1, 75)) if t1 else None,
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)

    for firm_key, cfg in FIRMS.items():
        sub = df[df.firm == firm_key].drop(columns=["firm"])
        print(f"\n=== {cfg['label']} | $100K static DD ===")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    out_csv = ROOT / "live" / "combined deployment plan" / "ftmo_fundingpips_100k_montecarlo.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    run_grid()

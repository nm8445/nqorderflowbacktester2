"""
Hola Prime $100k funded sim with PROPER payout rules:
  - Bi-weekly: 3 profitable days (>=$500/day net) in 14 calendar days, 80% split
  - Monthly:   7 profitable days in 30 calendar days, 95% split
  - On-demand: $2k+ total profit, biggest day <= 40% of total, 80% split

Account rules:
  - $10k trailing DD (locks at $100k floor after +$5k profit)
  - $5k daily loss
  - 2% risk-per-trade limit (auto-satisfied at MNQ<=15)

Slippage: MT5 CFD (RV/B2 $28, OD $70x1.25 mart) — using 2025+ regime data.
"""
from __future__ import annotations
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
COMBINED_MAE = RESULTS_DIR / "combined_3way_trades_with_mae.csv"

ACCT_SIZE       = 100_000
DAILY_LOSS_LIM  = 5_000
TRAILING_DD     = 10_000
LOCK_PROFIT     = 5_000
HORIZON_DAYS    = 252
N_SIMS          = 5000

# 2025+ regime
START_DATE = pd.Timestamp("2025-01-01", tz="America/New_York").date()

SLIPPAGE = {"RV": 28.0, "B2": 28.0, "OD": 70.0}
OD_MART_MULT = 1.25

PROFITABLE_DAY_THRESH = 500   # $500/day = 0.5% of $100k

# Payout cadences to compare
CADENCES = {
    "bi_weekly": {"window_days": 14, "required_profitable_days": 3, "split": 0.80, "min_profit": 0},
    "monthly":   {"window_days": 30, "required_profitable_days": 7, "split": 0.95, "min_profit": 0},
    "on_demand": {"window_days": None, "required_profitable_days": 0, "split": 0.80, "min_profit": 2_000,
                  "consistency_max": 0.40},
}


def load_daily(start_date=None):
    df = pd.read_csv(COMBINED_MAE)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    if start_date is not None:
        df = df[df["date"] >= start_date].copy()
    df["slip_$"] = df["strat"].map(SLIPPAGE).fillna(0.0)
    od_mask = df["strat"] == "OD"
    df.loc[od_mask, "slip_$"] *= OD_MART_MULT
    df["pnl_after_slip"] = df["pnl_$"] - df["slip_$"]
    df["mae_after_slip"] = df["mae_$"] - df["slip_$"]
    df = df.sort_values(["date", "entry_ts"])
    out = []
    for d, g in df.groupby("date", sort=True):
        out.append(list(zip(g["pnl_after_slip"].astype(float),
                            g["mae_after_slip"].astype(float))))
    return out


def sim_funded_with_cadence(daily, mnq, cadence_name, rng):
    """Simulate funded account with given payout cadence."""
    cadence = CADENCES[cadence_name]
    scale = mnq * 0.1
    balance = ACCT_SIZE; peak = ACCT_SIZE
    locked = False; prev_eod = ACCT_SIZE
    cumulative_profit_since_payout = 0.0
    daily_profits_since_payout = []  # list of (day_idx, net_pnl)
    payouts = []  # (day, gross_amount, cash_received)
    bust_day = -1

    for day in range(HORIZON_DAYS):
        idx = rng.integers(0, len(daily))
        if not daily[idx]:
            continue
        day_pnl = 0.0
        busted = False
        for pnl_nq, mae_nq in daily[idx]:
            mae_d = mae_nq * scale; pnl_d = pnl_nq * scale
            eq_dip = balance + day_pnl + mae_d
            if eq_dip > peak: peak = eq_dip
            if peak >= ACCT_SIZE + LOCK_PROFIT:
                cur_floor = max(ACCT_SIZE, peak - TRAILING_DD)
            else:
                cur_floor = peak - TRAILING_DD
            if eq_dip <= cur_floor or (prev_eod - eq_dip) >= DAILY_LOSS_LIM:
                busted = True; break
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur > peak: peak = cur
            if peak >= ACCT_SIZE + LOCK_PROFIT:
                cur_floor = max(ACCT_SIZE, peak - TRAILING_DD)
            else:
                cur_floor = peak - TRAILING_DD
            if cur <= cur_floor or (prev_eod - cur) >= DAILY_LOSS_LIM:
                busted = True; break
        if busted:
            bust_day = day; break

        balance += day_pnl
        prev_eod = balance
        cumulative_profit_since_payout += day_pnl
        daily_profits_since_payout.append((day, day_pnl))

        # Check payout eligibility
        eligible = False
        if cadence_name in ("bi_weekly", "monthly"):
            # Count profitable days within the cadence window (from start of cycle or last payout)
            window_days = cadence["window_days"]
            required = cadence["required_profitable_days"]
            cycle_start_day = daily_profits_since_payout[0][0] if daily_profits_since_payout else day
            days_in_cycle = day - cycle_start_day + 1
            if days_in_cycle >= window_days:
                profitable_count = sum(1 for d, p in daily_profits_since_payout if p >= PROFITABLE_DAY_THRESH)
                if profitable_count >= required and cumulative_profit_since_payout > 0:
                    eligible = True
                else:
                    # Failed to meet requirement — reset cycle without payout (forfeit)
                    # Realistically: keep trading, profits accumulate, but the cycle resets
                    daily_profits_since_payout = []  # rolling window restart
                    # Don't reset cumulative profit — it stays in the account
        else:  # on_demand
            if cumulative_profit_since_payout >= cadence["min_profit"]:
                pos_days = [p for d, p in daily_profits_since_payout if p > 0]
                if pos_days:
                    biggest_day = max(pos_days)
                    if biggest_day <= cadence["consistency_max"] * cumulative_profit_since_payout:
                        eligible = True

        if eligible:
            gross = cumulative_profit_since_payout
            cash = gross * cadence["split"]
            payouts.append((day, gross, cash))
            balance = ACCT_SIZE
            peak = ACCT_SIZE; locked = False; prev_eod = ACCT_SIZE
            cumulative_profit_since_payout = 0.0
            daily_profits_since_payout = []

    total_cash = sum(p[2] for p in payouts)
    first_pay_day = payouts[0][0] if payouts else -1
    return dict(payouts=payouts, total_cash=total_cash, bust_day=bust_day,
                n_payouts=len(payouts), first_pay_day=first_pay_day)


def run_set(label, daily, n_sims=N_SIMS):
    print(f"\n=== {label} ({len(daily)} historical days) ===")
    rng = np.random.default_rng(2026)
    rows = []
    print(f"{'MNQ':>4} {'cadence':>12}  {'bust%':>6}  {'med_cash':>10}  {'mean_cash':>10}  "
          f"{'med_pay#':>9}  {'mean_pay#':>10}  {'med_d_1st':>10}  {'mean_d_1st':>11}")
    for mnq in [1, 2, 3, 4]:
        for cadence_name in CADENCES.keys():
            sims = [sim_funded_with_cadence(daily, mnq, cadence_name, rng) for _ in range(n_sims)]
            cash = np.array([s["total_cash"] for s in sims])
            payouts = np.array([s["n_payouts"] for s in sims])
            first_pay = np.array([s["first_pay_day"] for s in sims if s["first_pay_day"] >= 0])
            n_bust = sum(1 for s in sims if s["bust_day"] >= 0)
            med_first = int(np.median(first_pay)) if len(first_pay) else -1
            mean_first = float(first_pay.mean()) if len(first_pay) else -1
            print(f"{mnq:>4} {cadence_name:>12}  {n_bust/n_sims*100:>5.1f}%  "
                  f"${np.median(cash):>9,.0f}  ${cash.mean():>9,.0f}  "
                  f"{int(np.median(payouts)):>9}  {payouts.mean():>10.2f}  "
                  f"{med_first:>10}  {mean_first:>11.1f}")
            rows.append(dict(slice=label, mnq=mnq, cadence=cadence_name,
                              bust_rate=n_bust/n_sims,
                              median_cash=float(np.median(cash)),
                              mean_cash=float(cash.mean()),
                              mean_payouts=float(payouts.mean()),
                              median_days_first_payout=med_first,
                              mean_days_first_payout=mean_first))
    return rows


def main():
    print("Hola Prime $100k funded with proper payout rules — FULL vs 2025+ regime\n")
    daily_full = load_daily(start_date=None)
    daily_2025 = load_daily(start_date=START_DATE)

    rows = []
    rows += run_set("FULL (2020-12 -> 2026-05)", daily_full)
    rows += run_set("2025+ regime", daily_2025)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "holaprime_funded_proper_rules.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR / 'holaprime_funded_proper_rules.csv'}")


if __name__ == "__main__":
    main()

"""
Pair/Triple rotation strategy across 10 funded accounts — Monte Carlo.

User's design:
  10 funded accounts total
  Active pool: 2 (or 3) accounts at a time
  Rotation: pool active for 1 trading week (5 days), then dormant
  Each account sees 1/5 (pair) or 1/3.3 (triple) duty cycle
  Each active account runs the full 3-strategy combo (RV+B2+OD)

Compares vs:
  - Continuous copy-trade all 10 accounts (synchronized busts)
  - Continuous run 2 accounts (no rotation, no dormancy)

Tests at 1, 2, 3 MNQ per account.
Bootstrap from combined_trades_with_mae.csv (full stack MAE-aware).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"

N_SIMS = 5_000
TRADING_DAYS_YEAR = 252
DAYS_PER_WEEK = 5
FUTURES_COST = 2.0

# Lucid Flex 50K
START_BAL = 50_000.0
FLOOR_INIT = 48_000.0
LOCK_AT = 53_000.0
LOCK_FLOOR = 50_000.0
TRAIL_DD = 2_000.0
PROFIT_TGT_1ST = 3_000.0
PROFIT_TGT_NEXT = 2_000.0
PAYOUT_1ST_GROSS = 1_500.0
PAYOUT_NEXT_GROSS = 1_000.0
SPLIT = 0.90
MIN_WIN_DAY = 150.0
QUAL_DAYS_NEEDED = 5
MAX_PAYOUTS = 6


def load_packs():
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"])
    packs = []
    for date, grp in df.groupby("date", sort=True):
        trades = [(r["pnl_$"], r["mae_$"]) for _, r in grp.iterrows()]
        packs.append(trades)
    return packs


class Account:
    __slots__ = ("balance","floor","hwm","locked","qual_days","cycle_profit",
                 "stagger_first_done","payouts","cash","busted")
    def __init__(self):
        self.balance = START_BAL
        self.floor = FLOOR_INIT
        self.hwm = START_BAL
        self.locked = False
        self.qual_days = 0
        self.cycle_profit = 0.0
        self.stagger_first_done = False
        self.payouts = 0
        self.cash = 0.0
        self.busted = False


def run_trading_day(acc, trades, mnq):
    """Apply one trading day's trades to one account. Returns trader_cash extracted."""
    if acc.busted or acc.payouts >= MAX_PAYOUTS:
        return 0.0
    scale = mnq / 10.0
    cost = FUTURES_COST
    daily_realized = 0.0
    # MAE-aware intraday bust check
    for pnl, mae in trades:
        pnl_scaled = pnl * scale - cost * mnq
        mae_scaled = mae * scale - cost * mnq
        if acc.balance + daily_realized + mae_scaled < acc.floor:
            acc.busted = True
            return 0.0
        daily_realized += pnl_scaled
    acc.balance += daily_realized
    if not acc.locked:
        if acc.balance > acc.hwm:
            acc.hwm = acc.balance
        acc.floor = max(FLOOR_INIT, acc.hwm - TRAIL_DD)
        if acc.hwm >= LOCK_AT:
            acc.locked = True
            acc.floor = LOCK_FLOOR
    if daily_realized >= MIN_WIN_DAY:
        acc.qual_days += 1
    acc.cycle_profit += daily_realized
    extracted = 0.0
    if acc.qual_days >= QUAL_DAYS_NEEDED and acc.cycle_profit > 0:
        gross = 0.0
        if not acc.stagger_first_done:
            if acc.cycle_profit >= PROFIT_TGT_1ST:
                gross = PAYOUT_1ST_GROSS
                acc.stagger_first_done = True
        else:
            if acc.cycle_profit >= PROFIT_TGT_NEXT:
                gross = PAYOUT_NEXT_GROSS
        if gross > 0:
            trader = gross * SPLIT
            acc.balance -= gross
            if not acc.locked:
                acc.hwm = max(START_BAL, acc.hwm - gross)
                acc.floor = max(FLOOR_INIT, acc.hwm - TRAIL_DD)
            acc.payouts += 1
            acc.cash += trader
            extracted = trader
            acc.qual_days = 0
            acc.cycle_profit = 0.0
    return extracted


def simulate_rotation(packs, mnq, n_active, n_total, rng):
    """Rotation: n_active accounts trade each week, rotating through n_total accounts."""
    accounts = [Account() for _ in range(n_total)]
    n_packs = len(packs)
    monthly_cash = [0.0] * 12
    total_cash = 0.0
    n_groups = n_total // n_active  # number of rotation slots
    if n_total % n_active != 0:
        n_groups = (n_total + n_active - 1) // n_active

    for day_idx in range(TRADING_DAYS_YEAR):
        week_idx = day_idx // DAYS_PER_WEEK
        active_group = week_idx % n_groups
        active_ids = list(range(active_group * n_active, min((active_group + 1) * n_active, n_total)))
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        for aid in active_ids:
            cash = run_trading_day(accounts[aid], trades, mnq)
            total_cash += cash
            month = day_idx // 21  # ~21 trading days per month
            if month >= 12: month = 11
            monthly_cash[month] += cash

    total_busted = sum(1 for a in accounts if a.busted)
    return total_cash, total_busted, monthly_cash


def simulate_copy_trade(packs, mnq, n_total, rng):
    """All n_total accounts trade every day (copy-trade)."""
    accounts = [Account() for _ in range(n_total)]
    n_packs = len(packs)
    monthly_cash = [0.0] * 12
    total_cash = 0.0
    for day_idx in range(TRADING_DAYS_YEAR):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        for aid in range(n_total):
            cash = run_trading_day(accounts[aid], trades, mnq)
            total_cash += cash
            month = day_idx // 21
            if month >= 12: month = 11
            monthly_cash[month] += cash
    total_busted = sum(1 for a in accounts if a.busted)
    return total_cash, total_busted, monthly_cash


def report(label, results):
    cash_arr = np.array([r[0] for r in results])
    bust_arr = np.array([r[1] for r in results])
    monthly = np.array([r[2] for r in results])
    mean_cash = cash_arr.mean()
    mean_busts = bust_arr.mean()
    median_cash = np.median(cash_arr)
    p25 = np.percentile(cash_arr, 25)
    p75 = np.percentile(cash_arr, 75)
    monthly_mean = monthly.mean(axis=0).mean()
    monthly_std = monthly.mean(axis=0).std()
    months_with_payout = (monthly > 0).mean(axis=0)
    pct_months_with_payout = months_with_payout.mean() * 100

    print(f"  {label}")
    print(f"    Annual NET (mean): ${mean_cash:>9,.0f}    (median ${median_cash:,.0f}, p25 ${p25:,.0f}, p75 ${p75:,.0f})")
    print(f"    Mean accounts busted of 10: {mean_busts:.1f}")
    print(f"    Monthly mean: ${monthly_mean:,.0f}   month-to-month std: ${monthly_std:,.0f}")
    print(f"    Pct months with >=1 payout: {pct_months_with_payout:.0f}%")


def main():
    packs = load_packs()
    print(f"Loaded {len(packs)} trading days / {sum(len(p) for p in packs)} trades (full stack MAE-aware)\n")

    for mnq in [1, 2, 3]:
        print(f"\n========== {mnq} MNQ PER ACCOUNT ==========\n")

        # Rotation: 2 accounts at a time (5 pairs)
        rng = np.random.default_rng(seed=mnq * 100 + 1)
        res_p2 = [simulate_rotation(packs, mnq, n_active=2, n_total=10, rng=rng) for _ in range(N_SIMS)]
        report(f"ROTATION (2 active / 10 accts, weekly rotation)", res_p2)

        rng = np.random.default_rng(seed=mnq * 100 + 2)
        res_p3 = [simulate_rotation(packs, mnq, n_active=3, n_total=10, rng=rng) for _ in range(N_SIMS)]
        report(f"ROTATION (3 active / 10 accts, weekly rotation)", res_p3)

        rng = np.random.default_rng(seed=mnq * 100 + 3)
        res_continuous_10 = [simulate_copy_trade(packs, mnq, n_total=10, rng=rng) for _ in range(N_SIMS)]
        report(f"CONTINUOUS copy-trade ALL 10 accts", res_continuous_10)

        rng = np.random.default_rng(seed=mnq * 100 + 4)
        res_continuous_2 = [simulate_copy_trade(packs, mnq, n_total=2, rng=rng) for _ in range(N_SIMS)]
        report(f"CONTINUOUS copy-trade just 2 accts (no rotation)", res_continuous_2)


if __name__ == "__main__":
    main()

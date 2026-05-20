"""
Staggered-start scaling vs continuous copy-trade vs coinflip
=============================================================
User's idea: copy-trade 3 accounts week 1, then add 3 more each subsequent week,
all on same signals. Use OD martingale DISABLED (1c always).

Tests:
  A) Staggered: 3 accts week 1, 6 by w2, 9 by w3, 10 by w4. Copy-traded.
  B) All-10 continuous copy-trade from day 1.
  C) Staggered same setup but with martingale OFF.
  D) Coinflip baseline.

All at 1 MNQ per account. Full 3-stack strategy (RV+B2+OD).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
OD_RAW_CSV = ROOT / "live" / "overnight drift" / "trades.csv"

N_SIMS = 5_000
HORIZON = 252
FUTURES_COST = 2.0

START_BAL = 50_000.0
FLOOR_INIT = 48_000.0
LOCK_AT = 53_000.0
LOCK_FLOOR = 50_000.0
TRAIL_DD = 2_000.0


def load_packs(disable_od_marti=False):
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"]).reset_index(drop=True)
    if disable_od_marti:
        od_raw = pd.read_csv(OD_RAW_CSV)
        od_raw["entry_time"] = pd.to_datetime(od_raw["entry_time"], utc=True, format="mixed")
        qty_map = dict(zip(od_raw["entry_time"], od_raw["qty"]))
        df["qty"] = 1
        for i in df.index[df["strat"] == "OD"]:
            df.at[i, "qty"] = qty_map.get(df.at[i, "entry_ts"], 1)
        scale = np.where((df["strat"] == "OD") & (df["qty"] == 2), 0.5, 1.0)
        df["pnl_$"] = df["pnl_$"] * scale
        df["mae_$"] = df["mae_$"] * scale
    packs = []
    for date, grp in df.groupby("date", sort=True):
        trades = [(r["pnl_$"], r["mae_$"]) for _, r in grp.iterrows()]
        packs.append(trades)
    return packs


class Account:
    __slots__ = ("balance","floor","hwm","locked","qual_days","cycle_profit",
                 "stagger_first_done","payouts","cash","busted","started_day")
    def __init__(self, started_day=0):
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
        self.started_day = started_day


def step_day(acc, trades, mnq, day_idx):
    """Apply one day's trades to one account."""
    if acc.busted or acc.payouts >= 6 or day_idx < acc.started_day:
        return 0.0
    scale = mnq / 10.0
    cost = FUTURES_COST
    daily_realized = 0.0
    for pnl, mae in trades:
        pnl_s = pnl * scale - cost * mnq
        mae_s = mae * scale - cost * mnq
        if acc.balance + daily_realized + mae_s < acc.floor:
            acc.busted = True
            return 0.0
        daily_realized += pnl_s
    acc.balance += daily_realized
    if not acc.locked:
        if acc.balance > acc.hwm: acc.hwm = acc.balance
        acc.floor = max(FLOOR_INIT, acc.hwm - TRAIL_DD)
        if acc.hwm >= LOCK_AT:
            acc.locked = True
            acc.floor = LOCK_FLOOR
    if daily_realized >= 150:
        acc.qual_days += 1
    acc.cycle_profit += daily_realized
    extracted = 0.0
    if acc.qual_days >= 5 and acc.cycle_profit > 0:
        gross = 0.0
        if not acc.stagger_first_done:
            if acc.cycle_profit >= 3000:
                gross = 1500
                acc.stagger_first_done = True
        else:
            if acc.cycle_profit >= 2000:
                gross = 1000
        if gross > 0:
            trader = gross * 0.9
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


def sim_continuous(packs, mnq, n_total, rng):
    """All n_total accounts copy-trade from day 0."""
    accounts = [Account(started_day=0) for _ in range(n_total)]
    n_packs = len(packs)
    total = 0.0
    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        for a in accounts:
            total += step_day(a, trades, mnq, d)
    busted = sum(1 for a in accounts if a.busted)
    return total, busted


def sim_staggered(packs, mnq, total_accts, add_per_week, weeks_between, rng):
    """Staggered start: add `add_per_week` accounts every `weeks_between` weeks."""
    accounts = []
    next_add_day = 0
    n_packs = len(packs)
    total = 0.0
    days_per_week = 5
    while len(accounts) < total_accts and next_add_day < HORIZON:
        n_new = min(add_per_week, total_accts - len(accounts))
        for _ in range(n_new):
            accounts.append(Account(started_day=next_add_day))
        next_add_day += weeks_between * days_per_week
    # If we didn't fill all by end of year, fill remaining
    while len(accounts) < total_accts:
        accounts.append(Account(started_day=HORIZON))  # will never activate

    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        for a in accounts:
            total += step_day(a, trades, mnq, d)
    busted = sum(1 for a in accounts if a.busted)
    return total, busted


def report(label, results):
    cash = np.array([r[0] for r in results])
    busts = np.array([r[1] for r in results])
    print(f"  {label}")
    print(f"    Annual NET mean: ${cash.mean():>9,.0f}   median ${np.median(cash):>9,.0f}   "
          f"p25 ${np.percentile(cash,25):>7,.0f}   p75 ${np.percentile(cash,75):>7,.0f}")
    print(f"    Monthly mean:    ${cash.mean()/12:>9,.0f}   mean busts of total: {busts.mean():.1f}")


def main():
    print("=" * 90)
    print("STAGGERED-START vs CONTINUOUS COPY-TRADE — All 1 MNQ, full 3-stack")
    print("=" * 90)

    for label, disable_marti in [("OD marti ON (original)", False), ("OD marti OFF (1c always)", True)]:
        print(f"\n========== {label} ==========")
        packs = load_packs(disable_od_marti=disable_marti)

        rng = np.random.default_rng(seed=hash(label) % 9973 + 1)
        cont10 = [sim_continuous(packs, 1, 10, rng) for _ in range(N_SIMS)]
        report("Continuous copy-trade ALL 10 from day 0", cont10)

        rng = np.random.default_rng(seed=hash(label) % 9973 + 2)
        stag10 = [sim_staggered(packs, 1, 10, add_per_week=3, weeks_between=1, rng=rng) for _ in range(N_SIMS)]
        report("Staggered: 3 accts/wk, ramp to 10 over 4 weeks", stag10)

        rng = np.random.default_rng(seed=hash(label) % 9973 + 3)
        stag_slow = [sim_staggered(packs, 1, 10, add_per_week=2, weeks_between=2, rng=rng) for _ in range(N_SIMS)]
        report("Staggered: 2 accts every 2 weeks, ramp to 10 over 10 weeks", stag_slow)

        rng = np.random.default_rng(seed=hash(label) % 9973 + 4)
        cont15 = [sim_continuous(packs, 1, 15, rng) for _ in range(N_SIMS)]
        report("Continuous copy-trade ALL 15 from day 0 (max across 3 firms)", cont15)

    # === Compare to coinflip ===
    print("\n" + "=" * 90)
    print("COINFLIP BASELINE for comparison")
    print("=" * 90)
    print("\nCoinflip 10 accts, 50/50, $1500 risk, with $30 slip:")
    print("  Annual NET mean: ~$18,346   Monthly: ~$1,529")
    print("Coinflip 15 accts (scaled): ~$27,500   Monthly: ~$2,290")


if __name__ == "__main__":
    main()

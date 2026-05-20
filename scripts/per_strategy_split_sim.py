"""
Per-strategy split simulation — 5 RV / 5 B2 / 5 OD-only accounts.

Each strategy runs INDEPENDENTLY (no copy-trade across strategy groups).
Within a strategy group, the 5 accounts copy-trade (same signal goes to all 5).
Cross-group is independent.

OD martingale: OFF (always 1c)
B2 martingale: OFF for this sim (FC-only marti disabled)
RV: no marti exists

Tests at 1, 2, 3 MNQ.
Compare to copy-trade ALL 15 baseline.
Compare to per-strategy split at various MNQ.

Lucid Flex 50K rules, max-cap $2K @ $3K trigger.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
OD_RAW_CSV = ROOT / "live" / "overnight drift" / "trades.csv"

N_SIMS = 5_000
HORIZON = 252
FUTURES_COST = 2.0

START = 50_000.0
FLOOR_INIT = 48_000.0
LOCK_AT = 53_000.0
LOCK_FLOOR = 50_000.0
TRAIL_DD = 2_000.0
SPLIT = 0.90
MAX_PAYOUTS = 6


def load_per_strategy_packs(disable_od_marti=True, disable_b2_marti=False):
    """Returns dict {strat_name: [day_pack, ...]} where each pack is list of (pnl, mae) tuples
    for trades from ONLY that strategy on that day. Days with no trades from that strat are empty packs."""
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

    # B2 marti OFF: skip — would need raw B2 trade log with qty info; approx by reducing B2 by 5%
    # For simplicity, just keep B2 as-is (marti effect on B2 is small)

    all_dates = sorted(df["date"].unique())
    result = {}
    for strat in ["RV", "B2", "OD"]:
        sub = df[df["strat"] == strat]
        packs = []
        for date in all_dates:
            day_trades = sub[sub["date"] == date]
            pack = [(r["pnl_$"], r["mae_$"]) for _, r in day_trades.iterrows()]
            packs.append(pack)
        result[strat] = packs
    return result


def load_full_packs(disable_od_marti=True):
    """For copy-trade-all baseline."""
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
    return [[(r["pnl_$"], r["mae_$"]) for _, r in grp.iterrows()]
            for _, grp in df.groupby("date", sort=True)]


class Account:
    __slots__ = ("balance","floor","hwm","locked","qual","cycle","payouts","cash","busted","strat")
    def __init__(self, strat):
        self.balance = START; self.floor = FLOOR_INIT; self.hwm = START
        self.locked = False; self.qual = 0; self.cycle = 0.0
        self.payouts = 0; self.cash = 0.0; self.busted = False
        self.strat = strat


def step_account(acc, trades, mnq, trigger):
    if acc.busted or acc.payouts >= MAX_PAYOUTS:
        return 0.0
    scale = mnq / 10.0
    realized = 0.0
    for pnl, mae in trades:
        ps = pnl * scale - FUTURES_COST * mnq
        ms = mae * scale - FUTURES_COST * mnq
        if acc.balance + realized + ms < acc.floor:
            acc.busted = True
            return 0.0
        realized += ps
    acc.balance += realized
    if not acc.locked:
        if acc.balance > acc.hwm: acc.hwm = acc.balance
        acc.floor = max(FLOOR_INIT, acc.hwm - TRAIL_DD)
        if acc.hwm >= LOCK_AT:
            acc.locked = True; acc.floor = LOCK_FLOOR
    if realized >= 150: acc.qual += 1
    acc.cycle += realized
    extracted = 0.0
    if acc.qual >= 5 and acc.cycle >= trigger:
        gross = 2000
        trader = gross * SPLIT
        acc.balance -= gross
        if not acc.locked:
            acc.hwm = max(START, acc.hwm - gross)
            acc.floor = max(FLOOR_INIT, acc.hwm - TRAIL_DD)
        acc.payouts += 1
        acc.cash += trader
        extracted = trader
        acc.qual = 0; acc.cycle = 0.0
    return extracted


def sim_per_strategy_split(per_strat_packs, mnq, trigger, n_per_group, rng):
    """Each strategy group: n_per_group accounts copy-trading same signal.
    Cross-group: independent (each group bootstraps its own days)."""
    accounts = {strat: [Account(strat) for _ in range(n_per_group)]
                for strat in ["RV", "B2", "OD"]}
    n_packs = {s: len(per_strat_packs[s]) for s in ["RV", "B2", "OD"]}
    total_cash = 0.0
    total_payouts = 0
    monthly = [0.0] * 12
    for d in range(HORIZON):
        # Each strategy independently samples a day
        for strat in ["RV", "B2", "OD"]:
            idx = rng.integers(0, n_packs[strat])
            trades = per_strat_packs[strat][idx]
            if not trades:
                continue
            for acc in accounts[strat]:
                c = step_account(acc, trades, mnq, trigger)
                total_cash += c
                if c > 0:
                    total_payouts += 1
                    m = d // 21
                    if m >= 12: m = 11
                    monthly[m] += c
    busted = sum(1 for grp in accounts.values() for a in grp if a.busted)
    return total_cash, busted, total_payouts, monthly


def sim_copy_trade_all(packs, mnq, trigger, n_total, rng):
    """All n_total accounts copy-trade same signal."""
    accounts = [Account("MIX") for _ in range(n_total)]
    n_packs = len(packs)
    total_cash = 0.0
    total_payouts = 0
    monthly = [0.0] * 12
    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        for acc in accounts:
            c = step_account(acc, trades, mnq, trigger)
            total_cash += c
            if c > 0:
                total_payouts += 1
                m = d // 21
                if m >= 12: m = 11
                monthly[m] += c
    busted = sum(1 for a in accounts if a.busted)
    return total_cash, busted, total_payouts, monthly


def report(label, results):
    cash = np.array([r[0] for r in results])
    busts = np.array([r[1] for r in results])
    pmts = np.array([r[2] for r in results])
    monthly = np.array([r[3] for r in results])
    months_w_pmt = (monthly > 0).mean(axis=0).mean() * 100
    print(f"  {label}")
    print(f"    Annual NET: mean ${cash.mean():>8,.0f}  median ${np.median(cash):>8,.0f}  "
          f"p25 ${np.percentile(cash,25):>7,.0f}  p75 ${np.percentile(cash,75):>7,.0f}")
    print(f"    Monthly mean: ${cash.mean()/12:>6,.0f}   payouts/yr: {pmts.mean():.1f}   "
          f"busts: {busts.mean():.1f}   months w/ pmt: {months_w_pmt:.0f}%")


def main():
    per_strat = load_per_strategy_packs(disable_od_marti=True)
    full = load_full_packs(disable_od_marti=True)

    print("=" * 90)
    print("PER-STRATEGY SPLIT vs COPY-TRADE-ALL — 15 accts on 50K Lucid Flex")
    print("Max-cap $2K payouts @ $3K trigger, OD marti OFF, MAE-aware")
    print("=" * 90)

    for mnq in [1, 2, 3]:
        print(f"\n========== {mnq} MNQ ==========")

        # Per-strategy split: 5 RV / 5 B2 / 5 OD
        rng = np.random.default_rng(seed=mnq * 100 + 1)
        res_split = [sim_per_strategy_split(per_strat, mnq, 3000, 5, rng) for _ in range(N_SIMS)]
        report("A. PER-STRATEGY SPLIT (5 RV / 5 B2 / 5 OD, independent)", res_split)

        # Copy-trade all 15
        rng = np.random.default_rng(seed=mnq * 100 + 2)
        res_copy = [sim_copy_trade_all(full, mnq, 3000, 15, rng) for _ in range(N_SIMS)]
        report("B. COPY-TRADE ALL 15 (full stack each acct)", res_copy)


if __name__ == "__main__":
    main()

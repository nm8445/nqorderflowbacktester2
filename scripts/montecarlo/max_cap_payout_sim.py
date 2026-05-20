"""
$2,000 max-cap payouts vs stagger A — copy-trade 10 accts.

Premise: instead of stagger ($1500/$1000), wait for cycle profit >= $2,000
THEN withdraw $2K gross (= $1,800 trader) every cycle. Take advantage of
the fact that once floor locks at $50K, extra profit = pure cushion.

Variants:
  - Stagger A baseline ($1500/$1000)
  - Max-cap with cycle trigger at $2,000
  - Max-cap with cycle trigger at $3,000 (more cushion before payout)
  - Max-cap with staggered start: 5 accts day 0, 10 accts from day 1

Full 3-stack with OD martingale OFF. 1 MNQ. MAE-aware.
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


def load_packs():
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"]).reset_index(drop=True)
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
    __slots__ = ("balance","floor","hwm","locked","qual","cycle","payouts","cash","busted","start_day")
    def __init__(self, start_day=0):
        self.balance = START; self.floor = FLOOR_INIT; self.hwm = START
        self.locked = False; self.qual = 0; self.cycle = 0.0
        self.payouts = 0; self.cash = 0.0; self.busted = False
        self.start_day = start_day


def step(acc, trades, mnq, day, payout_mode, trigger_profit):
    """payout_mode: 'stagger' or 'maxcap'"""
    if acc.busted or acc.payouts >= MAX_PAYOUTS or day < acc.start_day:
        return 0.0
    scale = mnq / 10.0
    cost = FUTURES_COST
    realized = 0.0
    for pnl, mae in trades:
        ps = pnl * scale - cost * mnq
        ms = mae * scale - cost * mnq
        if acc.balance + realized + ms < acc.floor:
            acc.busted = True
            return 0.0
        realized += ps
    acc.balance += realized
    if not acc.locked:
        if acc.balance > acc.hwm: acc.hwm = acc.balance
        acc.floor = max(FLOOR_INIT, acc.hwm - TRAIL_DD)
        if acc.hwm >= LOCK_AT:
            acc.locked = True
            acc.floor = LOCK_FLOOR
    if realized >= 150:
        acc.qual += 1
    acc.cycle += realized

    extracted = 0.0
    if acc.qual >= 5 and acc.cycle > 0:
        if payout_mode == "stagger":
            if acc.payouts == 0 and acc.cycle >= 3000:
                gross = 1500
            elif acc.payouts >= 1 and acc.cycle >= 2000:
                gross = 1000
            else:
                gross = 0
        elif payout_mode == "maxcap":
            # Wait for trigger_profit before withdrawing $2K
            if acc.cycle >= trigger_profit:
                gross = 2000
            else:
                gross = 0
        else:
            gross = 0
        if gross > 0:
            trader = gross * SPLIT
            acc.balance -= gross
            if not acc.locked:
                acc.hwm = max(START, acc.hwm - gross)
                acc.floor = max(FLOOR_INIT, acc.hwm - TRAIL_DD)
            acc.payouts += 1
            acc.cash += trader
            extracted = trader
            acc.qual = 0
            acc.cycle = 0.0
    return extracted


def sim(packs, mnq, n_total, payout_mode, trigger, stagger_start_days, rng):
    accounts = []
    if stagger_start_days is None:
        # all start at day 0
        accounts = [Account(0) for _ in range(n_total)]
    else:
        # stagger_start_days = list of day indices when each account starts
        for sd in stagger_start_days:
            accounts.append(Account(sd))
    n_packs = len(packs)
    total_cash = 0.0
    total_payouts = 0
    monthly = [0.0] * 12
    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        for a in accounts:
            c = step(a, trades, mnq, d, payout_mode, trigger)
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
    months_with_pmt = (monthly > 0).mean(axis=0).mean() * 100
    print(f"  {label}")
    print(f"    Mean NET/yr: ${cash.mean():>9,.0f}   p25 ${np.percentile(cash,25):>7,.0f}   p75 ${np.percentile(cash,75):>7,.0f}")
    print(f"    Monthly mean: ${cash.mean()/12:>6,.0f}   mean payouts/yr: {pmts.mean():.1f}   busts: {busts.mean():.1f}")
    print(f"    Pct months w/ payout: {months_with_pmt:.0f}%")


def main():
    packs = load_packs()
    print("=" * 80)
    print("$2K MAX-CAP PAYOUTS vs STAGGER A — 10 copy-traded accts, 1 MNQ, marti OFF")
    print(f"Lucid Flex 50K rules, MAE-aware, {N_SIMS} sims")
    print("=" * 80)

    rng = np.random.default_rng(seed=1001)
    res_stag = [sim(packs, 1, 10, "stagger", 0, None, rng) for _ in range(N_SIMS)]
    report("A. STAGGER A baseline ($1,500/$1,000)", res_stag)

    rng = np.random.default_rng(seed=1002)
    res_2k_2k = [sim(packs, 1, 10, "maxcap", 2000, None, rng) for _ in range(N_SIMS)]
    report("B. MAX-CAP $2K, trigger at +$2K cycle profit (tightest cushion)", res_2k_2k)

    rng = np.random.default_rng(seed=1003)
    res_2k_3k = [sim(packs, 1, 10, "maxcap", 3000, None, rng) for _ in range(N_SIMS)]
    report("C. MAX-CAP $2K, trigger at +$3K cycle profit (safer)", res_2k_3k)

    rng = np.random.default_rng(seed=1004)
    res_2k_4k = [sim(packs, 1, 10, "maxcap", 4000, None, rng) for _ in range(N_SIMS)]
    report("D. MAX-CAP $2K, trigger at +$4K cycle profit (max cushion before pulling)", res_2k_4k)

    # Staggered start: 5 day 0, 5 day 1
    stagger_days = [0]*5 + [1]*5
    rng = np.random.default_rng(seed=1005)
    res_split = [sim(packs, 1, 10, "maxcap", 3000, stagger_days, rng) for _ in range(N_SIMS)]
    report("E. MAX-CAP $2K @ +$3K, staggered start (5 day 0, 5 day 1)", res_split)

    print("\n" + "=" * 80)
    print("COINFLIP $2K-cap variant for comparison (10 accts, 50/50, $1500 stake)")
    print("=" * 80)
    print("Coinflip 50/50 with $2K payouts (estimated by scaling prior result by 2k/1500):")
    print("  Annual NET: ~$24,500   Monthly: ~$2,040")


if __name__ == "__main__":
    main()

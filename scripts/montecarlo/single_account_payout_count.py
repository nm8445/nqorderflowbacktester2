"""How many payouts per year on ONE account at different cycle triggers?"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
OD_RAW_CSV = ROOT / "live" / "overnight drift" / "trades.csv"

N_SIMS = 10_000
HORIZON = 252


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


def sim_one_account(packs, mnq, trigger, rng, max_payouts=6):
    """Single account, max-cap $2K payouts when cycle profit >= trigger.
    Returns (n_payouts, total_cash, busted, days_per_payout_list)."""
    bal = 50_000.0
    floor = 48_000.0
    hwm = bal
    locked = False
    qual = 0
    cycle = 0.0
    payouts = 0
    cash = 0.0
    days_at_payouts = []
    n_packs = len(packs)
    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        realized = 0.0
        for pnl, mae in trades:
            ps = pnl * (mnq/10) - 2.0 * mnq
            ms = mae * (mnq/10) - 2.0 * mnq
            if bal + realized + ms < floor:
                return payouts, cash, True, days_at_payouts
            realized += ps
        bal += realized
        if not locked:
            if bal > hwm: hwm = bal
            floor = max(48_000, hwm - 2000)
            if hwm >= 53_000:
                locked = True; floor = 50_000
        if realized >= 150: qual += 1
        cycle += realized
        if qual >= 5 and cycle >= trigger:
            gross = 2000
            trader = gross * 0.9
            bal -= gross
            if not locked:
                hwm = max(50_000, hwm - gross)
                floor = max(48_000, hwm - 2000)
            payouts += 1
            cash += trader
            days_at_payouts.append(d + 1)
            qual = 0; cycle = 0.0
            if payouts >= max_payouts: break
    return payouts, cash, False, days_at_payouts


def report(label, trigger, packs):
    rng = np.random.default_rng(seed=hash(label) % 9973)
    results = [sim_one_account(packs, 1, trigger, rng) for _ in range(N_SIMS)]
    payouts = np.array([r[0] for r in results])
    cash = np.array([r[1] for r in results])
    busts = np.array([r[2] for r in results])

    p_any = np.mean(payouts >= 1)
    p_2plus = np.mean(payouts >= 2)
    p_3plus = np.mean(payouts >= 3)
    p_5plus = np.mean(payouts >= 5)
    p_6 = np.mean(payouts >= 6)

    # days between payouts (for accounts that got multiple)
    days_btwn = []
    days_to_first = []
    for r in results:
        days_list = r[3]
        if days_list:
            days_to_first.append(days_list[0])
            for i in range(1, len(days_list)):
                days_btwn.append(days_list[i] - days_list[i-1])

    print(f"\n=== {label} (trigger = cycle profit >= ${trigger:,}) ===")
    print(f"  Mean payouts/yr per account: {payouts.mean():.2f}")
    print(f"  Median payouts/yr per account: {int(np.median(payouts))}")
    print(f"  Distribution: 0={1-p_any:.0%}  1+={p_any:.0%}  2+={p_2plus:.0%}  "
          f"3+={p_3plus:.0%}  5+={p_5plus:.0%}  6={p_6:.0%}")
    print(f"  Mean annual cash: ${cash.mean():,.0f}")
    print(f"  Bust rate: {busts.mean():.0%}")
    if days_to_first:
        print(f"  Days to 1st payout (when achieved): median {int(np.median(days_to_first))}d, "
              f"mean {np.mean(days_to_first):.0f}d")
    if days_btwn:
        print(f"  Days BETWEEN payouts: median {int(np.median(days_btwn))}d, "
              f"mean {np.mean(days_btwn):.0f}d")


def main():
    packs = load_packs()
    print("SINGLE ACCOUNT payout analysis — 1 MNQ full stack marti OFF MAE-aware")
    print(f"Max-cap $2K payouts ($1,800 trader). Lucid Flex 50K. {N_SIMS} sims.\n")

    report("Aggressive @ $2K trigger (instant-bust risk)", 2000, packs)
    report("Balanced @ $3K trigger (~$1K cushion after pull)", 3000, packs)
    report("Safe @ $4K trigger (~$2K cushion after pull)", 4000, packs)


if __name__ == "__main__":
    main()

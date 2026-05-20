"""
Days to next $2K payout — fresh account ($50K) vs post-payout state ($52K).
1 MNQ full stack, OD marti OFF, MAE-aware, Lucid Flex 50K rules.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
OD_RAW_CSV = ROOT / "live" / "overnight drift" / "trades.csv"

N_SIMS = 10_000
HORIZON = 252
FUTURES_COST = 2.0


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


def sim_days_to_payout(packs, mnq, start_bal, start_floor, locked, rng):
    """Returns (days_to_payout, busted) for one trial.
    Needs 5 winning days + cycle profit >= $2K."""
    balance = start_bal
    floor = start_floor
    hwm = start_bal
    qual = 0
    cycle = 0.0
    scale = mnq / 10.0
    n_packs = len(packs)
    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        realized = 0.0
        for pnl, mae in trades:
            ps = pnl * scale - FUTURES_COST * mnq
            ms = mae * scale - FUTURES_COST * mnq
            if balance + realized + ms < floor:
                return None, True  # busted
            realized += ps
        balance += realized
        if not locked:
            if balance > hwm: hwm = balance
            floor = max(start_floor, hwm - 2000)
            if hwm >= 53_000:
                locked = True
                floor = 50_000
        if realized >= 150:
            qual += 1
        cycle += realized
        if qual >= 5 and cycle >= 2000:
            return d + 1, False
    return None, False  # neither bust nor payout


def report(label, results):
    days = [r[0] for r in results if r[0] is not None]
    busts = sum(1 for r in results if r[1])
    timeouts = sum(1 for r in results if r[0] is None and not r[1])
    n = len(results)
    p_payout = len(days) / n * 100
    p_bust = busts / n * 100
    print(f"\n{label}")
    print(f"  P(reach payout): {p_payout:.1f}%   P(bust first): {p_bust:.1f}%   P(neither in 1yr): {100-p_payout-p_bust:.1f}%")
    if days:
        print(f"  Days to payout (when achieved):")
        print(f"    median: {int(np.median(days))}d   mean: {np.mean(days):.0f}d")
        print(f"    p25: {int(np.percentile(days, 25))}d   p75: {int(np.percentile(days, 75))}d")
        print(f"    p10: {int(np.percentile(days, 10))}d   p90: {int(np.percentile(days, 90))}d")


def main():
    packs = load_packs()
    print("=" * 70)
    print("DAYS TO NEXT $2K PAYOUT — Single account, 1 MNQ, full stack")
    print(f"Trigger: 5 winning days + cycle profit >= $2K. {N_SIMS} sims.")
    print("=" * 70)

    # Scenario A: Fresh account, $50K balance, $48K floor (trailing not yet locked)
    rng = np.random.default_rng(seed=4001)
    res_a = [sim_days_to_payout(packs, 1, 50_000, 48_000, False, rng) for _ in range(N_SIMS)]
    report("A. FRESH ACCOUNT: $50K balance, $48K floor (trailing), $2K cushion", res_a)

    # Scenario B: $52K balance, $50K floor (locked), $2K cushion
    rng = np.random.default_rng(seed=4002)
    res_b = [sim_days_to_payout(packs, 1, 52_000, 50_000, True, rng) for _ in range(N_SIMS)]
    report("B. AT $52K BALANCE: $50K floor (locked), $2K cushion", res_b)

    # Scenario C (for context): $50K balance, $50K floor (locked post-withdrawal), $0 cushion
    rng = np.random.default_rng(seed=4003)
    res_c = [sim_days_to_payout(packs, 1, 50_000, 50_000, True, rng) for _ in range(N_SIMS)]
    report("C. POST-MAX-CAP-PAYOUT: $50K balance, $50K floor locked, $0 cushion", res_c)

    # Scenario D: $51K balance, $50K floor locked (stagger-A post-payout state)
    rng = np.random.default_rng(seed=4004)
    res_d = [sim_days_to_payout(packs, 1, 51_500, 50_000, True, rng) for _ in range(N_SIMS)]
    report("D. POST-STAGGER-PAYOUT: $51.5K bal, $50K floor locked, $1.5K cushion", res_d)


if __name__ == "__main__":
    main()

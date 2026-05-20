"""
P(reach +$3K profit before -$2K trailing DD bust) — per strategy + combined, by MNQ size.

Simulates the Lucid Flex 50K eval (or any $50K / $2K-trail / $3K-target eval).
Two bust modes:
  - 'realized' : EOD balance vs trailing floor (lenient)
  - 'mae'      : INTRADAY equity (= balance + open-trade MAE) vs trailing floor (real rule)

Tracks first-passage to +$3K profit OR bust. No time limit (Lucid Flex eval has none).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"

N_SIMS = 10_000
MAX_DAYS = 504           # 2 yrs cap — eval has no time limit but we cap for compute
START_BAL = 50_000.0
PROFIT_TGT = 3_000.0     # +$3K above start
INITIAL_FLOOR = 48_000.0 # $2K trailing DD
FUTURES_COST = 2.0       # $/RT/MNQ


def load_packs(strats):
    df = pd.read_csv(TRADES_CSV)
    df = df[df["strat"].isin(strats)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"])
    packs = []
    for date, grp in df.groupby("date", sort=True):
        trades = [(r["pnl_$"], r["mae_$"]) for _, r in grp.iterrows()]
        packs.append(trades)
    return packs


def simulate_one(packs, mnq, rng, use_mae=True, cost=FUTURES_COST):
    """Run one eval until pass (+$3K) or bust (-$2K trail).
    Returns (outcome, days_taken). outcome in {'pass', 'bust', 'timeout'}.
    """
    scale = mnq / 10.0
    balance = START_BAL
    hwm = balance
    floor = INITIAL_FLOOR    # $2K trailing
    n_packs = len(packs)
    if n_packs == 0:
        return "timeout", 0
    for d in range(MAX_DAYS):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        daily_realized = 0.0
        busted = False
        for pnl, mae in trades:
            pnl_scaled = pnl * scale - cost * mnq
            mae_scaled = mae * scale - cost * mnq
            check = balance + daily_realized + (mae_scaled if use_mae else pnl_scaled)
            if check < floor:
                busted = True
                break
            daily_realized += pnl_scaled
        if busted:
            return "bust", d + 1
        balance += daily_realized
        # Check pass condition (EOD profit >= $3K)
        if balance - START_BAL >= PROFIT_TGT:
            return "pass", d + 1
        # Update trailing floor EOD
        if balance > hwm:
            hwm = balance
            floor = max(INITIAL_FLOOR, hwm - 2_000.0)
    return "timeout", MAX_DAYS


def run_scenario(name, strats):
    packs = load_packs(strats)
    n_days = len(packs)
    n_trades = sum(len(p) for p in packs)
    print(f"\n--- {name} ({n_days} days / {n_trades} trades) ---")
    rows = []
    for mnq in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed=hash(name) % 9973 + mnq * 17)
        results = [simulate_one(packs, mnq, rng, use_mae=True) for _ in range(N_SIMS)]
        outcomes = [r[0] for r in results]
        days = [r[1] for r in results]
        p_pass = outcomes.count("pass") / N_SIMS
        p_bust = outcomes.count("bust") / N_SIMS
        p_timeout = outcomes.count("timeout") / N_SIMS
        pass_days = [d for o, d in zip(outcomes, days) if o == "pass"]
        bust_days = [d for o, d in zip(outcomes, days) if o == "bust"]
        rows.append({
            "mnq": mnq,
            "P(pass)": p_pass,
            "P(bust)": p_bust,
            "P(timeout_2yr)": p_timeout,
            "median_days_to_pass": int(np.median(pass_days)) if pass_days else None,
            "p25_d_pass": int(np.percentile(pass_days, 25)) if pass_days else None,
            "p75_d_pass": int(np.percentile(pass_days, 75)) if pass_days else None,
            "median_days_to_bust": int(np.median(bust_days)) if bust_days else None,
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    return df


def main():
    print("Lucid Flex 50K eval: P(reach +$3K profit before -$2K trailing DD breach)")
    print(f"Bust check is MAE-aware (intraday floating equity vs trailing floor)")
    print(f"{N_SIMS} sims per cell, max horizon 2 years\n")

    scenarios = [
        ("RV only",        ["RV"]),
        ("B2 only",        ["B2"]),
        ("OD only",        ["OD"]),
        ("RV + B2",        ["RV", "B2"]),
        ("RV + OD",        ["RV", "OD"]),
        ("B2 + OD",        ["B2", "OD"]),
        ("Full stack (RV+B2+OD)", ["RV", "B2", "OD"]),
    ]
    all_dfs = {}
    for name, strats in scenarios:
        all_dfs[name] = run_scenario(name, strats)

    print("\n\n=== HEADLINE: P(pass) by strategy and MNQ size ===")
    print(f"{'Strategy':<28} " + " ".join(f"{m}MNQ".rjust(8) for m in [1, 2, 3, 4, 5]))
    for name, df in all_dfs.items():
        cells = " ".join(f"{r['P(pass)']*100:>7.1f}%" for _, r in df.iterrows())
        print(f"{name:<28} {cells}")

    print("\n=== Median days to pass ===")
    print(f"{'Strategy':<28} " + " ".join(f"{m}MNQ".rjust(8) for m in [1, 2, 3, 4, 5]))
    for name, df in all_dfs.items():
        cells = " ".join(f"{r['median_days_to_pass'] if r['median_days_to_pass'] else 'n/a':>7}d" for _, r in df.iterrows())
        print(f"{name:<28} {cells}")


if __name__ == "__main__":
    main()

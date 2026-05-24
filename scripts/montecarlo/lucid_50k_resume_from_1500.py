"""Lucid Flex 50K MC — starting from CURRENT state of +$1,500 profit.

Account state at sim start:
  Balance:   $51,500  (you already made $1,500 profit)
  HWM:       $51,500  (peak balance to date)
  Floor:     max($48,000, HWM - $2,000) = $49,500  (trailing DD)
  Target:    $53,000  ($3,000 total profit)
  Lock:      Floor locks at $50,000 once HWM >= $52,000

Sizing: 1 MNQ AND 2 MNQ on 4-way combined stack — sweep both.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"

START_BAL = 51_500.0       # current balance (started 50K, made +1500)
START_HWM = 51_500.0       # HWM = current (assume monotone up to here)
TRAIL_DD = 2_000.0
INITIAL_FLOOR = 50_000.0 - TRAIL_DD     # 48,000 (the original floor)
LOCK_TRIGGER = 52_000.0
LOCKED_FLOOR = 50_000.0
TARGET = 53_000.0          # +$3K total profit
MIN_QUAL_DAY = 150.0       # for context, not bust check

MAX_DAYS = 120             # cap each sim
N_SIMS = 20_000


def load_daily_pnl_1mnq() -> np.ndarray:
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed").dt.date
    daily = df.groupby("date")["pnl_$"].sum().sort_index()
    # Convert from 1 NQ basis to 1 MNQ basis
    return daily.values.astype(np.float64) * 0.1


def simulate(daily_1mnq: np.ndarray, mnq: float, rng: np.random.Generator) -> dict:
    bal = START_BAL
    hwm = START_HWM
    # Initial floor reflects current HWM
    floor = max(INITIAL_FLOOR, hwm - TRAIL_DD)
    n_pool = len(daily_1mnq)
    start_idx = int(rng.integers(0, n_pool - MAX_DAYS))

    for d in range(MAX_DAYS):
        pnl = daily_1mnq[(start_idx + d) % n_pool] * mnq
        bal += pnl
        if bal > hwm:
            hwm = bal
        if hwm >= LOCK_TRIGGER:
            floor = LOCKED_FLOOR
        else:
            floor = max(INITIAL_FLOOR, hwm - TRAIL_DD)

        if bal <= floor:
            return {"hit": False, "bust": True, "days": d + 1, "final_bal": bal}
        if bal >= TARGET:
            return {"hit": True, "bust": False, "days": d + 1, "final_bal": bal}

    return {"hit": False, "bust": False, "days": MAX_DAYS, "final_bal": bal}


def run(mnq: float):
    daily = load_daily_pnl_1mnq()
    print(f"\n{'='*70}")
    print(f"  LUCID 50K — RESUMING FROM +$1,500 — sizing {mnq} MNQ")
    print(f"  Current bal: ${START_BAL:,.0f}  HWM: ${START_HWM:,.0f}  "
          f"Floor: ${max(INITIAL_FLOOR, START_HWM - TRAIL_DD):,.0f}  Target: ${TARGET:,.0f}")
    print(f"{'='*70}")
    rng = np.random.default_rng(2026 + int(mnq * 100))
    sims = [simulate(daily, mnq, rng) for _ in range(N_SIMS)]

    n_hit  = sum(1 for s in sims if s["hit"])
    n_bust = sum(1 for s in sims if s["bust"])
    n_to   = sum(1 for s in sims if not s["hit"] and not s["bust"])

    print(f"\n  Hit $53K target:    {n_hit/N_SIMS*100:5.1f}%  ({n_hit:,} sims)")
    print(f"  Bust (hit floor):   {n_bust/N_SIMS*100:5.1f}%  ({n_bust:,} sims)")
    print(f"  Timeout (>{MAX_DAYS}d):    {n_to/N_SIMS*100:5.1f}%  ({n_to:,} sims)")

    days_hit = [s["days"] for s in sims if s["hit"]]
    if days_hit:
        print(f"\n  Days to reach $53K (conditional on hit, n={len(days_hit):,}):")
        print(f"    p10:    {int(np.percentile(days_hit, 10)):>3d} days")
        print(f"    p25:    {int(np.percentile(days_hit, 25)):>3d} days")
        print(f"    MEDIAN: {int(np.median(days_hit)):>3d} days")
        print(f"    p75:    {int(np.percentile(days_hit, 75)):>3d} days")
        print(f"    p90:    {int(np.percentile(days_hit, 90)):>3d} days")
        print(f"    MEAN:   {np.mean(days_hit):.1f} days")

    days_bust = [s["days"] for s in sims if s["bust"]]
    if days_bust:
        print(f"\n  Days to bust (conditional on bust, n={len(days_bust):,}):")
        print(f"    p25={int(np.percentile(days_bust, 25))}d  "
              f"median={int(np.median(days_bust))}d  "
              f"p75={int(np.percentile(days_bust, 75))}d")

    # Cumulative timeline
    print(f"\n  Cumulative % by day X:")
    for day in [5, 7, 10, 14, 21, 30, 45, 60, 90, 120]:
        hit_pct = sum(1 for s in sims if s["hit"] and s["days"] <= day) / N_SIMS * 100
        bust_pct = sum(1 for s in sims if s["bust"] and s["days"] <= day) / N_SIMS * 100
        print(f"    by day {day:>3d}: hit={hit_pct:5.1f}%   bust={bust_pct:4.1f}%")

    return n_hit / N_SIMS, days_hit


def main():
    print("Lucid Flex 50K — resuming from +$1,500, need $1,500 more to hit $3K target")
    print(f"Trail DD: ${TRAIL_DD:,.0f}  (current floor ${START_HWM-TRAIL_DD:,.0f}, locks at $50K once HWM >= $52K)")

    pass_rate_1, days_1 = run(1.0)
    pass_rate_2, days_2 = run(2.0)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  1 MNQ:  {pass_rate_1*100:.1f}% pass, median {int(np.median(days_1)) if days_1 else '-'} days")
    print(f"  2 MNQ:  {pass_rate_2*100:.1f}% pass, median {int(np.median(days_2)) if days_2 else '-'} days")


if __name__ == "__main__":
    main()

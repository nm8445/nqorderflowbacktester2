"""Days-to-$3K-target sim for Lucid Flex 50K @ 1 MNQ on the 4-strat stack.

Lucid Flex 50K rules (modeled):
  - Start $50,000 balance
  - Trailing DD = $2,000 from HWM, FLOOR LOCKS at $50K once HWM hits $52K
  - First payout target: balance >= $53,000  (+$3K profit)
  - No daily loss limit assumed (Lucid Flex doesn't have one)
  - Min qualifying days: 5 days with +$150 net (we report this stat too)
  - 1 MNQ = 1/10 of NQ basis, so PnL scaled by 0.1

Method:
  - Daily-PnL sequences from combined_4way_trades.csv (1414 sessions)
  - Bootstrap sims: randomly sample contiguous sequences of days
  - Track: hit_target (yes/no), days_to_target, bust (trail DD hit first), qualifying days
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"

START = 50_000.0
TRAIL_DD = 2_000.0
INITIAL_FLOOR = 48_000.0  # start - trail_dd
LOCK_TRIGGER = 52_000.0   # HWM at which floor locks at 50K
LOCKED_FLOOR = 50_000.0
TARGET = 53_000.0         # +$3K profit
MIN_QUAL_DAY = 150.0
QUAL_DAYS_NEEDED = 5

MNQ_SCALE = 0.1  # 1 MNQ vs 1 NQ basis in CSV
N_SIMS = 20_000
MAX_DAYS = 120   # cap each sim


def load_daily_pnl():
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed").dt.date
    daily = df.groupby("date")["pnl_$"].sum().sort_index()
    return daily.values * MNQ_SCALE


def simulate_one(daily_pnl: np.ndarray, rng: np.random.Generator) -> dict:
    """One bootstrap path: contiguous start index, walk forward."""
    n = len(daily_pnl)
    start_idx = int(rng.integers(0, n - MAX_DAYS))
    bal = START
    hwm = START
    floor = INITIAL_FLOOR
    qual_days = 0
    days_to_target = None
    bust = False
    for d in range(MAX_DAYS):
        pnl = daily_pnl[(start_idx + d) % n]
        bal += pnl
        if pnl >= MIN_QUAL_DAY:
            qual_days += 1
        # Update HWM and floor
        if bal > hwm:
            hwm = bal
        # Floor mechanics
        if hwm >= LOCK_TRIGGER and floor < LOCKED_FLOOR:
            floor = LOCKED_FLOOR
        elif hwm < LOCK_TRIGGER:
            floor = max(INITIAL_FLOOR, hwm - TRAIL_DD)
        # Bust check
        if bal <= floor:
            bust = True
            return {"hit": False, "bust": True, "days": d + 1,
                    "qual_days": qual_days, "final_bal": bal}
        # Target check
        if bal >= TARGET:
            days_to_target = d + 1
            return {"hit": True, "bust": False, "days": days_to_target,
                    "qual_days": qual_days, "final_bal": bal,
                    "qual_ok": qual_days >= QUAL_DAYS_NEEDED}
    return {"hit": False, "bust": False, "days": MAX_DAYS,
            "qual_days": qual_days, "final_bal": bal}


def main():
    daily = load_daily_pnl()
    print(f"Loaded {len(daily)} trading days")
    print(f"  Mean daily PnL @ 1 MNQ: ${daily.mean():+.2f}")
    print(f"  Std:                    ${daily.std():.2f}")
    print(f"  Median daily:           ${np.median(daily):+.2f}")
    print(f"  Best day:               ${daily.max():+.2f}")
    print(f"  Worst day:              ${daily.min():+.2f}")

    rng = np.random.default_rng(42)
    sims = [simulate_one(daily, rng) for _ in range(N_SIMS)]

    hits = [s for s in sims if s["hit"]]
    busts = [s for s in sims if s["bust"]]
    timeouts = [s for s in sims if not s["hit"] and not s["bust"]]

    print(f"\n{'='*70}")
    print(f"Results over {N_SIMS:,} sims ({MAX_DAYS}-day horizon)")
    print(f"{'='*70}")
    print(f"  Hit $3K target:    {len(hits)/N_SIMS*100:5.1f}%  ({len(hits):,} sims)")
    print(f"  Bust before:       {len(busts)/N_SIMS*100:5.1f}%  ({len(busts):,} sims)")
    print(f"  Timeout (>{MAX_DAYS}d): {len(timeouts)/N_SIMS*100:5.1f}%  ({len(timeouts):,} sims)")

    if hits:
        days_to_hit = np.array([s["days"] for s in hits])
        print(f"\nDAYS TO $3K TARGET (conditional on hit, n={len(hits):,}):")
        for pct, lbl in [(10, "p10"), (25, "p25"), (50, "MEDIAN"), (75, "p75"), (90, "p90")]:
            print(f"    {lbl:>6}: {int(np.percentile(days_to_hit, pct)):>3} days")
        print(f"  mean: {days_to_hit.mean():.1f} days")
        qual_ok = sum(1 for s in hits if s.get("qual_ok", False))
        print(f"\n  Of those that hit target, {qual_ok}/{len(hits)} ({qual_ok/len(hits)*100:.0f}%) "
              f"had >=5 qualifying days ($150+ winners)")

    if busts:
        days_to_bust = np.array([s["days"] for s in busts])
        print(f"\nDAYS TO BUST (conditional on bust, n={len(busts):,}):")
        print(f"    p25={int(np.percentile(days_to_bust, 25))}d, "
              f"median={int(np.percentile(days_to_bust, 50))}d, "
              f"p75={int(np.percentile(days_to_bust, 75))}d")

    # Day-by-day cumulative success rate
    print(f"\n{'='*70}")
    print("CUMULATIVE % of sims that have hit $3K target by day X:")
    print(f"{'='*70}")
    for d_check in [5, 7, 10, 14, 21, 30, 45, 60, 90, 120]:
        hit_by = sum(1 for s in sims if s["hit"] and s["days"] <= d_check)
        bust_by = sum(1 for s in sims if s["bust"] and s["days"] <= d_check)
        print(f"  by day {d_check:>3}: hit={hit_by/N_SIMS*100:5.1f}%   "
              f"bust={bust_by/N_SIMS*100:5.1f}%   still alive=     "
              f"{(N_SIMS - hit_by - bust_by)/N_SIMS*100:5.1f}%")


if __name__ == "__main__":
    main()

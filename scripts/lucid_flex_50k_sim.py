"""
LucidFlex 50K Funded Account Monte Carlo
==========================================
Inputs: live/combined deployment plan/combined_trades.csv (3-way combined log, 1 NQ per strat)

Lucid Flex 50K rules (sourced 2026-05):
  - Starting balance: $50,000
  - EOD trailing drawdown: $2,000 below highest EOD balance
  - DD locks at $50,000 once EOD balance hits Initial Trail Balance $53,000
  - No daily loss limit
  - Payout cycle: need >=5 days with profit >=$150 AND positive cycle PnL
  - Max payout per request: $2,000 (cap)
  - Min payout: $500
  - Max 6 payouts then account auto-graduates to LucidLive
  - 90/10 split (the $2,000 cap is the trader's 90% share, so really need $2,222 cycle profit)
  - Max position size at 50K tier: 4 minis / 40 micros (well above 1-5 MNQ)

Sizing: backtest is 1 NQ per strategy = 10 MNQ per strategy.
We scale daily PnL by (mnq_per_strat / 10) for each test level.

Withdrawal modes (amounts are GROSS deducted from the account; Lucid sends trader 90%):
  A) "stagger": at cycle_profit >= $3,000 → withdraw $1,500 gross (account drops to floor+$1.5k).
     Thereafter, at cycle_profit >= $2,000 → withdraw $1,000 gross (leave $1k cycle buffer).
  B) "buffer2k": every payout opportunity, withdraw down to leave $2,000 above the DD floor,
     capped at $2,000 gross per request, min $500.

DD handling on payout: payout reduces balance AND reduces HWM by same amount
(standard Lucid behavior — trail floor moves down with you, no penalty for taking profit).
After lock (HWM >= $53,000), floor is permanent at $50,000 and payouts don't touch it.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades.csv"

START_BAL = 50_000.0
TRAIL_DD = 2_000.0
LOCK_AT_BAL = 53_000.0          # once EOD >= this, floor locks
LOCK_FLOOR = 50_000.0
QUAL_DAY_MIN = 150.0            # min profit to count toward 5-day requirement
QUAL_DAYS_NEEDED = 5
MAX_PAYOUTS = 6
MAX_PAYOUT_AMT = 2_000.0
MIN_PAYOUT_AMT = 500.0
SPLIT_RETAINED = 0.90           # trader keeps 90% of withdrawal (10% to Lucid)
HORIZON_DAYS = 252              # 1 year
N_SIMS = 5_000


def load_daily_pnl_for_1nq() -> np.ndarray:
    """Daily PnL series at 1 NQ per strategy from the combined trade log."""
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["pnl_$"].sum().sort_index()
    return daily.values.astype(float)


def simulate_one(daily_1nq: np.ndarray, mnq_per_strat: float, mode: str,
                 rng: np.random.Generator) -> dict:
    """Run a single 1-year sim. Returns dict of stats."""
    mult = mnq_per_strat / 10.0
    balance = START_BAL
    hwm = START_BAL
    floor = START_BAL - TRAIL_DD     # 48,000
    locked = False

    cycle_qual_days = 0
    cycle_profit = 0.0               # cumulative profit since last payout request
    payouts_taken = 0
    payouts_log = []                 # (day_idx, amount_withdrawn_gross)
    stagger_first_payout_done = False
    busted = False
    bust_day = None
    days_traded = 0
    total_gross_withdrawn = 0.0
    days_to_first_payout = None

    # bootstrap day-by-day
    n_pool = len(daily_1nq)
    for d in range(HORIZON_DAYS):
        pnl = daily_1nq[rng.integers(0, n_pool)] * mult
        balance += pnl
        days_traded += 1

        # bust check (EOD)
        if balance < floor:
            busted = True
            bust_day = d
            break

        # update HWM and floor (only if not locked)
        if not locked:
            if balance > hwm:
                hwm = balance
                floor = max(START_BAL - TRAIL_DD, hwm - TRAIL_DD)
            if hwm >= LOCK_AT_BAL:
                locked = True
                floor = LOCK_FLOOR

        # update qualifying day counter (only positive days >=$150)
        if pnl >= QUAL_DAY_MIN:
            cycle_qual_days += 1
        cycle_profit += pnl

        # ---- Withdrawal logic ----
        if payouts_taken < MAX_PAYOUTS and cycle_qual_days >= QUAL_DAYS_NEEDED and cycle_profit > 0:

            # All withdrawal amounts here are GROSS deductions from the account.
            # Trader receives 90% (SPLIT_RETAINED) in cash; the other 10% is Lucid's fee.
            target_gross = 0.0
            if mode == "stagger":
                if not stagger_first_payout_done:
                    if cycle_profit >= 3_000.0:
                        target_gross = 1_500.0
                        stagger_first_payout_done = True
                else:
                    if cycle_profit >= 2_000.0:
                        target_gross = 1_000.0
            elif mode == "buffer2k":
                # withdraw down to floor + $2k, capped at $2,000 gross per request
                effective_floor = LOCK_FLOOR if locked else floor
                excess_over_buffer = balance - (effective_floor + 2_000.0)
                if excess_over_buffer >= MIN_PAYOUT_AMT:
                    target_gross = min(MAX_PAYOUT_AMT, excess_over_buffer)

            # apply cap and min, then deduct
            if target_gross >= MIN_PAYOUT_AMT:
                target_gross = min(target_gross, MAX_PAYOUT_AMT)
                trader_share = target_gross * SPLIT_RETAINED

                balance -= target_gross
                if not locked:
                    hwm = max(START_BAL, hwm - target_gross)
                    floor = max(START_BAL - TRAIL_DD, hwm - TRAIL_DD)

                payouts_taken += 1
                payouts_log.append((d, trader_share))
                total_gross_withdrawn += trader_share
                if days_to_first_payout is None:
                    days_to_first_payout = d
                # reset cycle counters
                cycle_qual_days = 0
                cycle_profit = 0.0

                # if 6 payouts taken, auto-graduate (stop sim)
                if payouts_taken >= MAX_PAYOUTS:
                    break

    return {
        "busted": busted,
        "bust_day": bust_day,
        "payouts": payouts_taken,
        "total_withdrawn": total_gross_withdrawn,
        "days_to_first_payout": days_to_first_payout,
        "end_balance": balance,
        "graduated": payouts_taken >= MAX_PAYOUTS,
        "avg_payout": (total_gross_withdrawn / payouts_taken) if payouts_taken > 0 else 0.0,
        "days_traded": days_traded,
    }


def run_grid():
    daily = load_daily_pnl_for_1nq()
    print(f"Loaded {len(daily)} trading days from combined log. "
          f"mean={daily.mean():.2f} std={daily.std():.2f} "
          f"min={daily.min():.2f} max={daily.max():.2f}")

    rows = []
    for mnq in [1, 2, 3, 4, 5]:
        for mode in ["stagger", "buffer2k"]:
            rng = np.random.default_rng(seed=2026 + mnq * 7 + (0 if mode == "stagger" else 1))
            sims = [simulate_one(daily, mnq, mode, rng) for _ in range(N_SIMS)]

            bust_rate = np.mean([s["busted"] for s in sims])
            payouts = [s["payouts"] for s in sims]
            withdrawn = [s["total_withdrawn"] for s in sims]
            t1 = [s["days_to_first_payout"] for s in sims if s["days_to_first_payout"] is not None]
            any_payout_rate = np.mean([s["payouts"] >= 1 for s in sims])
            grad_rate = np.mean([s["graduated"] for s in sims])

            # avg payout *amount per withdrawal* (across sims where >=1 payout)
            avg_pmt = np.mean([s["avg_payout"] for s in sims if s["payouts"] >= 1]) if any_payout_rate > 0 else 0

            rows.append({
                "mnq": mnq,
                "mode": mode,
                "bust_rate": bust_rate,
                "any_payout_rate": any_payout_rate,
                "graduate_rate": grad_rate,
                "median_payouts": int(np.median(payouts)),
                "mean_payouts": np.mean(payouts),
                "p10_payouts": int(np.percentile(payouts, 10)),
                "p90_payouts": int(np.percentile(payouts, 90)),
                "median_total_$": np.median(withdrawn),
                "mean_total_$": np.mean(withdrawn),
                "p25_total_$": np.percentile(withdrawn, 25),
                "p75_total_$": np.percentile(withdrawn, 75),
                "avg_per_payout_$": avg_pmt,
                "median_days_to_first_payout": int(np.median(t1)) if t1 else None,
                "p25_days_to_first": int(np.percentile(t1, 25)) if t1 else None,
                "p75_days_to_first": int(np.percentile(t1, 75)) if t1 else None,
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== LucidFlex 50K Monte Carlo (N=5,000 / horizon=252 trading days) ===\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    out_path = ROOT / "live" / "combined deployment plan" / "lucid_flex_50k_montecarlo.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Also dump a clean text summary
    txt = ROOT / "live" / "combined deployment plan" / "lucid_flex_50k_montecarlo.txt"
    with open(txt, "w") as f:
        f.write("LucidFlex 50K — Monte Carlo on combined 3-strategy log\n")
        f.write(f"{N_SIMS} sims/cell, {HORIZON_DAYS}-day horizon, bootstrap from "
                f"{len(daily)} historical days (2020-12 .. 2026-05)\n")
        f.write("Rules: $2K EOD trailing DD (locks at $50K floor once balance reaches $53K),\n")
        f.write("       5 days >= $150 profit per payout cycle, $2K cap per payout (trader's 90%),\n")
        f.write("       6 payouts max then auto-graduate to LucidLive.\n\n")
        f.write(df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
        f.write("\n")
    print(f"Saved: {txt}")


if __name__ == "__main__":
    run_grid()

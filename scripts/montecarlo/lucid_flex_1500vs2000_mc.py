"""LucidFlex 50K Monte Carlo — exact two payout patterns.

Patterns tested:
  A) "$1500 @ $3K"   payout = $1,500 GROSS whenever cycle profit >= $3,000.
  B) "$2000 @ $4K"   payout = $2,000 GROSS whenever cycle profit >= $4,000.

Both patterns repeat per payout cycle (5 qualifying days >= $150 required between payouts).
Trader keeps 90% of gross (Lucid takes 10%).

Sizing: 1 MNQ vs 2 MNQ (multiplier on the 4-strategy combined backtest at 1-NQ basis).

Lucid Flex 50K rules:
  - Start $50K, $2K trailing DD floor.
  - Floor LOCKS at $50K once HWM >= $52K (no more downward trail after that).
  - MAE-aware: an open trade whose adverse excursion would breach the floor BUSTS the account.
  - No daily loss limit.
  - 5 qualifying days (>= $150 profit) per payout cycle.
  - Max 6 payouts then auto-graduate (account dies, you start new one).

Output:
  - For each (pattern, mnq): bust%, any-payout%, days-to-first-payout (median + p25/p75),
    annual payouts under "refresh-after-bust-or-graduate" policy.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRADES_CSV   = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
B2_TRADES    = ROOT / "scripts" / "overnight range strat" / "tradelogs" / "robust_configs" / "locked_v2_k08_lock045_mart_fc_filtered_trades.csv"
# Per-trade MAE source (the same backtest log used by other sims)
FAB_TRADES   = Path("D:/trading_pythonbacktest_data/fabio orb/trades_final_modeA.csv")

# Lucid Flex 50K
START_BAL     = 50_000.0
TRAIL_DD      = 2_000.0
LOCK_AT_BAL   = 52_000.0       # HWM trigger to lock floor
LOCK_FLOOR    = 50_000.0
QUAL_DAY_MIN  = 150.0
QUAL_DAYS_REQ = 5
MAX_PAYOUTS   = 6
SPLIT_RETAIN  = 0.90
HORIZON_DAYS  = 252
N_SIMS        = 10_000

# Payout patterns
PATTERNS = {
    "$1500@$3K": (3_000.0, 1_500.0),
    "$2000@$4K": (4_000.0, 2_000.0),
}


def load_daily_pnl_with_mae() -> tuple[np.ndarray, np.ndarray]:
    """Per-trading-day: (daily realized PnL @ 1-NQ basis,
                         daily MAX MAE in dollars @ 1-NQ basis).

    MAE = the worst single intraday equity dip across all trades that day.
    Approximated by the largest single-trade MAE in dollars on that day.
    """
    df = pd.read_csv(TRADES_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  utc=True).dt.tz_convert("America/New_York")
    df["date"]     = df["exit_ts"].dt.date

    # Per-strat MAE in dollars at 1-NQ basis (multiply by 20 if needed)
    # For B2: read mae from backtest log; for FB: use mae_pts from log; for RV/OD: no MAE col,
    # we'll proxy by trade's realized loss (lower bound — actual MAE is worse).
    b2 = pd.read_csv(B2_TRADES)
    b2["entry_ts"] = pd.to_datetime(b2["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    b2["date"] = b2["entry_ts"].dt.date
    # B2 backtest log doesn't have MAE directly, but peak_mfe and pnl give bounds.
    # Use the trade's realized PnL as a lower-bound MAE for sim purposes:
    #   for losers, MAE >= |pnl|; for winners, MAE >= 0 (we can't know without bars).
    b2["mae_$"] = (-b2["scaled_pnl"]).clip(lower=0) * 20

    fb = pd.read_csv(FAB_TRADES)
    fb = fb[fb["mode"] == "A"].copy()
    fb["entry_ts"] = pd.to_datetime(fb["entry_time"], utc=True).dt.tz_convert("America/New_York")
    fb["date"] = fb["entry_ts"].dt.date
    fb["mae_$"] = fb["mae_pts"] * 20

    # RV + OD: proxy MAE = |trade_pnl| for losers (lower bound)
    df["mae_$"] = (-df["pnl_$"]).clip(lower=0)
    # Replace MAE for B2 trades with actual computed values
    b2_keyed = b2.set_index("entry_ts")["mae_$"].to_dict()
    fb_keyed = fb.set_index("entry_ts")["mae_$"].to_dict()
    df["mae_$"] = df.apply(
        lambda r: (b2_keyed.get(r["entry_ts"], r["mae_$"]) if r["strat"] == "B2"
                   else fb_keyed.get(r["entry_ts"], r["mae_$"]) if r["strat"] == "FB"
                   else r["mae_$"]),
        axis=1,
    )

    daily = df.groupby("date").agg(pnl=("pnl_$", "sum"), mae=("mae_$", "max"))
    return daily["pnl"].values.astype(float), daily["mae"].values.astype(float)


def simulate_one_account(pnl_pool: np.ndarray, mae_pool: np.ndarray, mnq: int,
                          profit_trigger: float, payout_amt: float,
                          rng: np.random.Generator) -> dict:
    """Simulate ONE account life. Ends at first payout OR bust.
    (User refreshes funded accounts after each payout AND after each bust.)
    """
    mult = mnq / 10.0
    balance = START_BAL
    hwm = START_BAL
    floor = START_BAL - TRAIL_DD
    locked = False
    cycle_qual = 0
    cycle_profit = 0.0
    busted = False
    days_lived = 0
    days_to_payout = None
    got_payout = False
    trader_received = 0.0

    n_pool = len(pnl_pool)
    for d in range(HORIZON_DAYS):
        idx = rng.integers(0, n_pool)
        day_pnl = pnl_pool[idx] * mult
        day_mae = mae_pool[idx] * mult

        if balance - day_mae < floor:
            busted = True
            days_lived = d + 1
            break
        balance += day_pnl
        days_lived = d + 1
        if balance < floor:
            busted = True
            break

        if not locked:
            if balance > hwm:
                hwm = balance
                floor = max(START_BAL - TRAIL_DD, hwm - TRAIL_DD)
            if hwm >= LOCK_AT_BAL:
                locked = True
                floor = LOCK_FLOOR

        if day_pnl >= QUAL_DAY_MIN:
            cycle_qual += 1
        cycle_profit += day_pnl

        if cycle_qual >= QUAL_DAYS_REQ and cycle_profit >= profit_trigger:
            got_payout = True
            days_to_payout = d + 1
            trader_received = payout_amt * SPLIT_RETAIN
            break

    return {
        "busted": busted,
        "got_payout": got_payout,
        "days_lived": days_lived,
        "days_to_payout": days_to_payout,
        "trader_$": trader_received,
    }


# Back-compat alias for the run_pattern function
simulate_one = simulate_one_account


def run_pattern(pnl_pool, mae_pool, label, profit_trigger, payout_amt):
    print(f"\n========================================================================")
    print(f"  Pattern: {label}  (trigger=${profit_trigger:,.0f}  payout=${payout_amt:,.0f} gross)")
    print(f"  Policy: refresh account after EACH payout AND after bust")
    print(f"========================================================================")
    rows = []
    for mnq in [1, 2]:
        rng = np.random.default_rng(seed=2026 + mnq)
        sims = [simulate_one_account(pnl_pool, mae_pool, mnq, profit_trigger, payout_amt, rng)
                for _ in range(N_SIMS)]
        bust_rate    = np.mean([s["busted"]    for s in sims])
        payout_rate  = np.mean([s["got_payout"] for s in sims])

        # Days to payout (only when achieved)
        d_pay = [s["days_to_payout"] for s in sims if s["got_payout"]]
        d_med = int(np.median(d_pay)) if d_pay else None
        d_mean = float(np.mean(d_pay)) if d_pay else None
        d_p25 = int(np.percentile(d_pay, 25)) if d_pay else None
        d_p75 = int(np.percentile(d_pay, 75)) if d_pay else None

        # Avg account lifespan (any reason for closing)
        avg_lifespan = float(np.mean([s["days_lived"] for s in sims]))

        # Accounts/year, expected trader payouts/year
        accts_per_year   = HORIZON_DAYS / avg_lifespan if avg_lifespan > 0 else 0
        # Each account gets 0 or 1 payout. Expected payouts/yr = payout_rate * accts/yr
        payouts_per_year = payout_rate * accts_per_year
        trader_per_year  = payouts_per_year * payout_amt * SPLIT_RETAIN
        gross_acct_fees  = accts_per_year * 150.0   # $150 eval per account
        net_per_year     = trader_per_year - gross_acct_fees

        rows.append({
            "mnq": mnq,
            "bust%": bust_rate * 100,
            "payout%": payout_rate * 100,
            "d_to_pay_med": d_med,
            "d_to_pay_p25": d_p25,
            "d_to_pay_p75": d_p75,
            "avg_life_d": avg_lifespan,
            "accts/yr": accts_per_year,
            "payouts/yr": payouts_per_year,
            "trader_$/yr": trader_per_year,
            "fees_$/yr": gross_acct_fees,
            "NET_$/yr": net_per_year,
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
    return df


def main():
    print(f"Loading combined 4-way trades from {TRADES_CSV.name}...")
    pnl_pool, mae_pool = load_daily_pnl_with_mae()
    print(f"  {len(pnl_pool)} trading days  "
          f"mean PnL=${pnl_pool.mean():.0f}  std=${pnl_pool.std():.0f}  "
          f"max MAE=${mae_pool.max():.0f}")

    all_rows = []
    for label, (trig, amt) in PATTERNS.items():
        df = run_pattern(pnl_pool, mae_pool, label, trig, amt)
        df["pattern"] = label
        all_rows.append(df)

    summary = pd.concat(all_rows).reset_index(drop=True)
    out = ROOT / "live" / "combined deployment plan" / "lucid_flex_50k_1500vs2000_mc.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

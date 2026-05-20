"""FundedNext 2-Step Challenge Monte Carlo on 4-strategy combined.

Strategy stack: RV + B2 + OD + Fabio ORB (combined_4way_trades.csv)

Challenge rules (user-confirmed):
  - Phase 1: hit +8% profit target
  - Phase 2: hit +5% profit target (after passing Phase 1; balance resets to start)
  - Daily loss > 5% of starting balance -> account FAILS
  - Total drawdown (static, from starting balance) > 10% -> account FAILS
  - NO per-trade-idea rule (only daily DD and max DD apply during challenges)
  - No time limit (run until pass or fail)
  - No minimum trading days (assume FundedNext's 0-3 day min is not binding)

Costs:
  - 200K challenge: $1,100
  - 100K challenge: $500
  - Funded payout split: 95% (NOT modeled here — this script is the challenge phase only)

Sweep: MNQ 1 .. 10
Sims per cell: 5000, horizon 252 trading days (caps unlikely-to-pass tails)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"

NQ_PT = 20.0
HORIZON_DAYS = 252
N_SIMS = 5000

ACCOUNTS = {
    "100K": dict(start=100_000, eval_fee=500),
    "200K": dict(start=200_000, eval_fee=1_100),
}

PHASE1_TARGET = 0.08    # +8%
PHASE2_TARGET = 0.05    # +5%
DAILY_CAP_PCT = 0.05    # 5%
TOTAL_CAP_PCT = 0.10    # 10%, static floor


def load_daily_pnl_1nq() -> np.ndarray:
    df = pd.read_csv(TRADES_CSV)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"]    = df["exit_ts"].dt.date
    daily = df.groupby("date")["pnl_$"].sum()
    return daily.values.astype(float)


def simulate_phase(pnl_pool, mnq, start_bal, target_dollars, daily_cap, total_cap,
                   rng, max_days=HORIZON_DAYS) -> tuple[bool, int, float]:
    """Run ONE phase to completion. Returns (passed, days_used, end_balance).

    Static floor: balance < start_bal - total_cap => FAIL
    Daily cap: day_pnl < -daily_cap => FAIL
    Target: balance >= start_bal + target_dollars => PASS
    """
    mult = mnq / 10.0
    balance = start_bal
    floor = start_bal - total_cap
    n_pool = len(pnl_pool)
    for d in range(max_days):
        day_pnl = pnl_pool[rng.integers(0, n_pool)] * mult
        # Daily cap kill (checked BEFORE adding to balance — instant kill)
        if day_pnl < -daily_cap:
            return False, d + 1, balance + day_pnl
        balance += day_pnl
        # Static total DD kill
        if balance < floor:
            return False, d + 1, balance
        # Profit target hit (EOD check)
        if balance >= start_bal + target_dollars:
            return True, d + 1, balance
    # Ran out of horizon — count as fail (challenge wasn't completed in 1 year)
    return False, max_days, balance


def simulate_full_challenge(pnl_pool, mnq, start_bal, rng) -> dict:
    """Run Phase 1 → (if passed) Phase 2. Return dict of results."""
    daily_cap = start_bal * DAILY_CAP_PCT
    total_cap = start_bal * TOTAL_CAP_PCT
    p1_target = start_bal * PHASE1_TARGET
    p2_target = start_bal * PHASE2_TARGET

    p1_pass, p1_days, p1_end = simulate_phase(pnl_pool, mnq, start_bal, p1_target,
                                                daily_cap, total_cap, rng)
    if not p1_pass:
        return {"p1_pass": False, "p2_pass": False, "p1_days": p1_days,
                "p2_days": 0, "total_days": p1_days, "end_balance": p1_end}

    # Phase 2: balance resets to starting balance (FundedNext convention)
    p2_pass, p2_days, p2_end = simulate_phase(pnl_pool, mnq, start_bal, p2_target,
                                                daily_cap, total_cap, rng)
    return {"p1_pass": True, "p2_pass": p2_pass, "p1_days": p1_days,
            "p2_days": p2_days, "total_days": p1_days + p2_days, "end_balance": p2_end}


def run_grid(label, rules, pnl_pool):
    print(f"\n========================================================================")
    print(f"  {label} 2-Step Challenge  start=${rules['start']:,}  eval_fee=${rules['eval_fee']:,}")
    print(f"  Phase 1: +{PHASE1_TARGET*100:.0f}% (${rules['start']*PHASE1_TARGET:,.0f})  "
          f"Phase 2: +{PHASE2_TARGET*100:.0f}% (${rules['start']*PHASE2_TARGET:,.0f})")
    print(f"  Daily cap: {DAILY_CAP_PCT*100:.0f}% (${rules['start']*DAILY_CAP_PCT:,.0f})  "
          f"Max DD: {TOTAL_CAP_PCT*100:.0f}% static (${rules['start']*TOTAL_CAP_PCT:,.0f})")
    print(f"========================================================================")
    rows = []
    for mnq in range(1, 11):
        rng = np.random.default_rng(seed=2026 + mnq + (1 if label == "200K" else 0))
        sims = [simulate_full_challenge(pnl_pool, mnq, rules["start"], rng) for _ in range(N_SIMS)]

        p1_rate = np.mean([s["p1_pass"] for s in sims])
        p2_rate = np.mean([s["p2_pass"] for s in sims])
        overall = p2_rate     # only passes if both phases pass

        # Days stats — only over sims that REACHED that phase
        p1_days_all = [s["p1_days"] for s in sims]
        p1_days_pass = [s["p1_days"] for s in sims if s["p1_pass"]]
        p2_days_pass = [s["p2_days"] for s in sims if s["p2_pass"]]
        total_pass = [s["total_days"] for s in sims if s["p2_pass"]]

        # Cost-per-funded-account: 1 / overall pass rate accounts on average
        avg_attempts = 1.0 / overall if overall > 0 else float("inf")
        cost_per_funded = rules["eval_fee"] * avg_attempts

        rows.append({
            "mnq": mnq,
            "p1_pass%":   p1_rate * 100,
            "p2_pass%":   p2_rate * 100 / max(p1_rate, 1e-9),  # conditional
            "overall%":   overall * 100,
            "p1_days_med":  int(np.median(p1_days_pass)) if p1_days_pass else None,
            "p1_days_p25":  int(np.percentile(p1_days_pass, 25)) if p1_days_pass else None,
            "p1_days_p75":  int(np.percentile(p1_days_pass, 75)) if p1_days_pass else None,
            "p2_days_med":  int(np.median(p2_days_pass)) if p2_days_pass else None,
            "total_days_med": int(np.median(total_pass)) if total_pass else None,
            "attempts_to_fund": avg_attempts if avg_attempts != float("inf") else None,
            "cost_to_fund_$": cost_per_funded if cost_per_funded != float("inf") else None,
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
    return df


def main():
    print("Loading combined 4-strategy daily PnL...")
    pnl_pool = load_daily_pnl_1nq()
    print(f"  {len(pnl_pool)} trading days, mean=${pnl_pool.mean():.0f}, "
          f"std=${pnl_pool.std():.0f}")

    all_dfs = []
    for label, rules in ACCOUNTS.items():
        df = run_grid(label, rules, pnl_pool)
        df["account"] = label
        all_dfs.append(df)

    summary = pd.concat(all_dfs).reset_index(drop=True)
    out = ROOT / "live" / "combined deployment plan" / "fundednext_2step_4strat_mc.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

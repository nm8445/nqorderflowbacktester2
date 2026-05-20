"""
Coinflip prop-firm sim v2 — multi-trade-per-day, EOD-locked trailing DD.

Setup the user described:
- Trade 9am-4pm ET. Take a new trade as soon as the prior one resolves.
- TP/SL fixed points, 1:1 RR. Per-trade outcome = +R or -R (fair coin).
- Profit target: $3000.
- Trailing DD: $2000 below the running PEAK OF EOD BALANCE. Intraday
  excursions don't update the floor; the floor only ratchets up at EOD.
- Consistency rule: at the moment of passing, no single day's profit may
  exceed 40% of the cumulative P&L. If hit at +$3K with too-fat a day, the
  trader keeps trading more days to dilute that day's share.
- 10 accounts at $100 each ($1,000 buy-in). Goal: maximize P(>=3 funded).

We sweep two dimensions:
  - R per trade ($, fixed)
  - Trades per day (back-to-back; user has multiple strategies running)

Bust check is intraday (per trade), pass check is at EOD only. Max 30 days.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

TARGET = 3_000.0
TRAILING_DD = 2_000.0
CONSISTENCY = 0.40
MAX_DAYS = 30
N_ACCOUNTS = 10
COST_PER_ACCT = 100.0
TOTAL_COST = N_ACCOUNTS * COST_PER_ACCT


def simulate_eval(R: float, N_per_day: int, rng: np.random.Generator) -> tuple[str, float, int]:
    """One eval simulation.

    Returns (outcome, final_cum, days_used).
    outcome in {'pass_consistent', 'pass_inconsistent', 'bust', 'timeout'}
    """
    cum = 0.0
    peak_eod = 0.0
    dd_floor = peak_eod - TRAILING_DD
    day_profits: list[float] = []

    for day in range(MAX_DAYS):
        day_start_cum = cum
        for _ in range(N_per_day):
            win = rng.random() < 0.5
            cum += R if win else -R
            if cum <= dd_floor:
                return "bust", cum, day + 1
        day_profit = cum - day_start_cum
        day_profits.append(day_profit)
        # EOD: update peak and DD floor
        if cum > peak_eod:
            peak_eod = cum
        dd_floor = peak_eod - TRAILING_DD
        # Pass check
        if cum >= TARGET:
            max_day = max(day_profits)
            ratio = max_day / cum if cum > 0 else 1.0
            if ratio <= CONSISTENCY + 1e-9:
                return "pass_consistent", cum, day + 1
            # else: keep trading to dilute - fall through
    # Timeout
    if cum >= TARGET:
        return "pass_inconsistent", cum, MAX_DAYS
    return "timeout", cum, MAX_DAYS


def evaluate(R: float, N: int, n_sims: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    counts = {"pass_consistent": 0, "pass_inconsistent": 0, "bust": 0, "timeout": 0}
    days_to_pass: list[int] = []
    for _ in range(n_sims):
        outcome, _final, days = simulate_eval(R, N, rng)
        counts[outcome] += 1
        if outcome == "pass_consistent":
            days_to_pass.append(days)
    p_funded = counts["pass_consistent"] / n_sims
    p_pass_any = (counts["pass_consistent"] + counts["pass_inconsistent"]) / n_sims
    return {
        "R": R,
        "N_per_day": N,
        "p_funded": p_funded,
        "p_pass_any": p_pass_any,
        "p_bust": counts["bust"] / n_sims,
        "p_timeout": counts["timeout"] / n_sims,
        "avg_days_to_pass": float(np.mean(days_to_pass)) if days_to_pass else float("nan"),
    }


def binomial_geq_k(p: float, n: int, k: int) -> float:
    return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))


def main() -> None:
    print(f"Settings: target=${TARGET:.0f}, EOD trailing DD=${TRAILING_DD:.0f}, "
          f"consistency={CONSISTENCY*100:.0f}%, max {MAX_DAYS} days")
    print(f"{N_ACCOUNTS} accounts @ ${COST_PER_ACCT:.0f} = ${TOTAL_COST:.0f}\n")

    R_grid = [200, 300, 400, 500, 600, 750, 1000, 1200]
    N_grid = [1, 3, 5, 8, 10, 15, 20]
    n_sims = 40_000

    grid = []
    for R in R_grid:
        for N in N_grid:
            r = evaluate(R, N, n_sims, seed=42)
            r["P>=3/10"] = binomial_geq_k(r["p_funded"], N_ACCOUNTS, 3)
            r["P>=4/10"] = binomial_geq_k(r["p_funded"], N_ACCOUNTS, 4)
            r["E[funded]"] = r["p_funded"] * N_ACCOUNTS
            grid.append(r)
    df = pd.DataFrame(grid)
    df.to_csv(OUT_DIR / "coinflip_propfirm_v2_grid.csv", index=False)

    # Pivot: p_funded grid
    print("=== P(funded) per account (%) — rows=R$, cols=trades/day ===")
    pivot = (df.pivot(index="R", columns="N_per_day", values="p_funded") * 100).round(1)
    print(pivot.to_string())

    print("\n=== P(>= 3 of 10 funded) (%) — rows=R$, cols=trades/day ===")
    pivot3 = (df.pivot(index="R", columns="N_per_day", values="P>=3/10") * 100).round(1)
    print(pivot3.to_string())

    print("\n=== P(>= 4 of 10 funded) (%) — rows=R$, cols=trades/day ===")
    pivot4 = (df.pivot(index="R", columns="N_per_day", values="P>=4/10") * 100).round(1)
    print(pivot4.to_string())

    print("\n=== Average days to pass (consistency-clean passes only) ===")
    pivot_days = df.pivot(index="R", columns="N_per_day", values="avg_days_to_pass").round(1)
    print(pivot_days.to_string())

    # Best overall
    best = df.sort_values("P>=3/10", ascending=False).head(10)
    print("\n=== Top 10 by P(>= 3 of 10 funded) ===")
    show = best[["R", "N_per_day", "p_funded", "p_bust", "p_timeout",
                 "avg_days_to_pass", "E[funded]", "P>=3/10", "P>=4/10"]].copy()
    show["p_funded"] = (show["p_funded"] * 100).round(1)
    show["p_bust"] = (show["p_bust"] * 100).round(1)
    show["p_timeout"] = (show["p_timeout"] * 100).round(1)
    show["P>=3/10"] = (show["P>=3/10"] * 100).round(1)
    show["P>=4/10"] = (show["P>=4/10"] * 100).round(1)
    show.rename(columns={"R": "R_$", "N_per_day": "trd/day", "p_funded": "P(fund)%",
                         "p_bust": "P(bust)%", "p_timeout": "P(t/o)%",
                         "avg_days_to_pass": "avg_days"}, inplace=True)
    with pd.option_context("display.width", 200):
        print(show.to_string(index=False))

    print(f"\nSaved -> {OUT_DIR / 'coinflip_propfirm_v2_grid.csv'}")


if __name__ == "__main__":
    main()

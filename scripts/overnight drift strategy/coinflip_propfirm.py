"""
Fair-coin prop firm Monte Carlo.

Setup (Apex-style $50K eval):
- One trade at a time per account.
- Entry direction is random (50/50 buy or sell).
- TP and SL are fixed points, 1:1 RR. Per-trade outcome is +R (prob 0.5) or -R (prob 0.5).
- Profit target: $3,000.
- Trailing EOD drawdown: $2,000 below the running peak of END-OF-DAY balance.
- Consistency rule: largest single-day profit must be <= 40% of total profit at payout.
- Max trades per eval: 60 (about 3 months of 1-trade-per-day).
- 1 trade per day (so EOD = per-trade for trailing-DD update purposes).

We sweep R from $100..$2000 and report:
  - Raw pass rate (hits +$3K before busting OR timing out)
  - Consistency-compliant pass rate (excludes passes that violate 40% rule)
  - P(>= k of N accounts funded) for N=10
  - EV at various payout assumptions

For a fair coin with NO trailing DD, gambler's ruin gives pass = D/(U+D) = 2000/5000 = 40% (R-independent).
Trailing DD makes pass rate strictly lower because the lower barrier follows the equity peak up.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# Account params
TARGET = 3_000.0
TRAILING_DD = 2_000.0
MAX_TRADES = 60
CONSISTENCY = 0.40
N_ACCOUNTS = 10
COST_PER_ACCT = 100.0
TOTAL_COST = N_ACCOUNTS * COST_PER_ACCT


def simulate_one(R: float, rng: np.random.Generator) -> tuple[str, float, float, int]:
    """One eval. Returns (outcome, final_pnl, max_day_profit, n_trades).

    outcome: 'pass_consistent' | 'pass_violates_consistency' | 'bust' | 'timeout'
    Assumes 1 trade per day so EOD == per-trade for trailing-DD updates.
    Stops trading immediately when cum >= TARGET (won't keep trading after pass).
    """
    cum = 0.0
    peak = 0.0
    daily_profits: list[float] = []  # one per trade, since 1 trade/day
    for i in range(MAX_TRADES):
        # Fair coin
        win = rng.random() < 0.5
        pnl = R if win else -R
        cum += pnl
        daily_profits.append(pnl)
        if cum > peak:
            peak = cum
        # Bust check: cum dropped >$2K from peak
        if cum <= peak - TRAILING_DD:
            return "bust", cum, max(daily_profits), i + 1
        # Pass check
        if cum >= TARGET:
            max_day = max(daily_profits)
            # Consistency: max single day profit <= 40% of total at payout
            if max_day / cum <= CONSISTENCY + 1e-9:
                return "pass_consistent", cum, max_day, i + 1
            else:
                return "pass_violates_consistency", cum, max_day, i + 1
    return "timeout", cum, max(daily_profits) if daily_profits else 0.0, MAX_TRADES


def evaluate_R(R: float, n_sims: int = 100_000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    counts = {"pass_consistent": 0, "pass_violates_consistency": 0, "bust": 0, "timeout": 0}
    final_pnls = []
    for _ in range(n_sims):
        outcome, final, max_day, n_trades = simulate_one(R, rng)
        counts[outcome] += 1
        final_pnls.append(final)
    # "Funded" = pass and complies with consistency
    p_funded = counts["pass_consistent"] / n_sims
    p_pass_any = (counts["pass_consistent"] + counts["pass_violates_consistency"]) / n_sims
    return {
        "R": R,
        "p_funded": p_funded,
        "p_pass_any": p_pass_any,
        "p_bust": counts["bust"] / n_sims,
        "p_timeout": counts["timeout"] / n_sims,
        "expected_pnl": np.mean(final_pnls),
        "consistency_pass_lands_at_3k_exactly": (R <= TARGET * CONSISTENCY + 1e-9),
        **counts,
    }


def binomial_k_of_n(p: float, n: int = N_ACCOUNTS) -> dict:
    """P(exactly k passes) and P(>= k passes) for k in 0..n."""
    out = {}
    for k in range(n + 1):
        out[f"P_exact_{k}"] = math.comb(n, k) * p**k * (1 - p) ** (n - k)
    for k in range(n + 1):
        out[f"P_atleast_{k}"] = sum(out[f"P_exact_{j}"] for j in range(k, n + 1))
    return out


def main() -> None:
    print(f"Settings: target=${TARGET:.0f}, trailing DD=${TRAILING_DD:.0f}, "
          f"max {MAX_TRADES} trades, consistency={CONSISTENCY*100:.0f}%, "
          f"{N_ACCOUNTS} accounts @ ${COST_PER_ACCT:.0f} = ${TOTAL_COST:.0f}")
    print()

    R_grid = [100, 150, 200, 250, 300, 400, 500, 600, 750, 800, 1000, 1100, 1200, 1300, 1500, 1750, 2000]
    rows = []
    for R in R_grid:
        r = evaluate_R(R, n_sims=80_000, seed=42)
        rows.append(r)
    df = pd.DataFrame(rows)

    # Append binomial outcomes for 10 accounts
    extras = []
    for r in rows:
        b = binomial_k_of_n(r["p_funded"], n=N_ACCOUNTS)
        extras.append(b)
    df_b = pd.DataFrame(extras)
    df = pd.concat([df, df_b], axis=1)

    print("=== Pass-rate sweep (n=80,000 sims per R) ===")
    show = df[
        [
            "R",
            "p_funded",
            "p_pass_any",
            "p_bust",
            "p_timeout",
            "P_atleast_1",
            "P_atleast_3",
            "P_atleast_4",
            "P_atleast_5",
        ]
    ].copy()
    for c in show.columns[1:]:
        show[c] = (show[c] * 100).round(1)
    show.rename(
        columns={
            "R": "R_$",
            "p_funded": "P(funded)",
            "p_pass_any": "P(hit+3K)",
            "p_bust": "P(bust)",
            "p_timeout": "P(timeout)",
            "P_atleast_1": "P>=1/10",
            "P_atleast_3": "P>=3/10",
            "P_atleast_4": "P>=4/10",
            "P_atleast_5": "P>=5/10",
        },
        inplace=True,
    )
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(show.to_string(index=False))

    # Expected count
    df["E_funded"] = df["p_funded"] * N_ACCOUNTS

    print("\n=== Expected funded accounts (10 evals) + payout-sensitivity ===")
    table_rows = []
    for _, r in df.iterrows():
        row = {
            "R_$": int(r["R"]),
            "P(funded)": f"{r['p_funded']*100:.1f}%",
            "E[funded/10]": round(r["E_funded"], 2),
            "P>=3/10": f"{r['P_atleast_3']*100:.1f}%",
            "P>=4/10": f"{r['P_atleast_4']*100:.1f}%",
        }
        for payout in [1000, 1500, 2000, 3000, 5000]:
            ev = N_ACCOUNTS * r["p_funded"] * payout - TOTAL_COST
            row[f"EV@${payout}"] = f"${ev:,.0f}"
        table_rows.append(row)
    with pd.option_context("display.width", 240):
        print(pd.DataFrame(table_rows).to_string(index=False))

    print("\n=== Notes ===")
    print(f"- For R > ${int(TARGET*CONSISTENCY)} (= 40% of $3000 target), passes that land")
    print(f"  at exactly $3K violate the 40% consistency rule. P(funded) excludes those.")
    print(f"- 'P(funded)' is the consistency-compliant pass rate. 'P(hit+3K)' is the raw pass rate.")
    print(f"- Direction (buy/sell) is irrelevant since the coin is fair; only $-risk matters.")
    print(f"- Lower R = more chances to recover from drawdown -> higher pass rate, but more trades to fund.")
    print(f"- Higher R > $1200 -> consistency rule disqualifies most passes.")

    df.to_csv(OUT_DIR / "coinflip_propfirm_sweep.csv", index=False)
    print(f"\nSaved -> {OUT_DIR / 'coinflip_propfirm_sweep.csv'}")


if __name__ == "__main__":
    main()

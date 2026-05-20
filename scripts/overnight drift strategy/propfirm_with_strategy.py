"""
Prop-firm simulator using the actual overnight drift trade stream.

For each ATR config (sl_mult, tp_mult):
  - Generate 1357 historical trades over 2020-12 to 2026-05.
  - Scale per-trade $-P&L to a fixed risk-per-SL ($R) via fractional contracts.
    Real-world: MNQ is $2/pt, so fractional sizing is achievable in
    increments of 1 MNQ = 1/10 of an NQ contract. We treat contracts as
    fully fractional here (the user explicitly allowed MNQ-style sizing).
  - For every possible START DATE, simulate an eval:
      cum = sum of trades; track EOD peak; trailing DD = peak - $2000;
      pass when cum >= $3000; bust when cum <= peak - $2000 (intraday=EOD
      because we're 1 trade/day per account).
  - Compute per-account pass rate, mean days to pass, P(>=3 of 10 fund).
  - "2 at a time" affects calendar time, not per-account pass rate; we
    compute expected wall-clock time to reach 3 fundeds assuming a
    rolling 2-account pipeline.

Configs tested:
  - 1.25 / 1.50 (best PF)
  - 1.50 / 1.50 (1:1 reference)
  - 1.50 / 0.50 (high WR / low RR)
  - 1.25 / 3.00 (best gross $)
  - 1.00 / 1.50 (small SL)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from asym_atr_grid import run_asym  # noqa: E402
from overnight_drift_strategy import build_full_20min_series  # noqa: E402

NQ_POINT = 20.0
PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLES = "D:/trading_pythonbacktest_data/timebars_5min"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
TZ = "America/New_York"

TARGET = 3_000.0
TRAILING_DD = 2_000.0
CONSISTENCY = 0.40
MAX_DAYS = 30
N_ACCOUNTS = 10


def scale_trades(trades: pd.DataFrame, R: float) -> np.ndarray:
    """Per-trade $-P&L when sizing contracts so SL-hit = -R.

    contracts = R / (sl_pts * NQ_POINT)
    pnl_$ = pnl_pts * NQ_POINT * contracts = pnl_pts * R / sl_pts
    """
    pnl_pts = trades["pnl_pts"].to_numpy()
    sl_pts = trades["sl_pts"].to_numpy()
    return pnl_pts * (R / sl_pts)


def simulate_account(pnls: np.ndarray, max_days: int = MAX_DAYS) -> tuple[str, int]:
    """One eval, 1 trade/day. Returns (outcome, days_used)."""
    cum = 0.0
    peak = 0.0
    daily_profits: list[float] = []
    for i, p in enumerate(pnls[:max_days]):
        cum += p
        daily_profits.append(p)
        # EOD update: peak/DD floor
        if cum > peak:
            peak = cum
        dd_floor = peak - TRAILING_DD
        if cum <= dd_floor:
            return "bust", i + 1
        if cum >= TARGET:
            max_day = max(daily_profits)
            if max_day / cum <= CONSISTENCY + 1e-9:
                return "pass_consistent", i + 1
            # otherwise continue to dilute
    if cum >= TARGET:
        return "pass_inconsistent", min(len(pnls), max_days)
    return "timeout", min(len(pnls), max_days)


def evaluate_R(pnls: np.ndarray) -> dict:
    """Roll start through trade stream. Each start = a fresh account."""
    n = len(pnls)
    counts = {"pass_consistent": 0, "pass_inconsistent": 0, "bust": 0, "timeout": 0}
    days_pass: list[int] = []
    days_bust: list[int] = []
    days_any: list[int] = []
    for s in range(n - 5):
        out, d = simulate_account(pnls[s:])
        counts[out] += 1
        days_any.append(d)
        if out == "pass_consistent":
            days_pass.append(d)
        elif out == "bust":
            days_bust.append(d)
    tot = sum(counts.values())
    return {
        "starts": tot,
        "p_funded": counts["pass_consistent"] / tot,
        "p_pass_any": (counts["pass_consistent"] + counts["pass_inconsistent"]) / tot,
        "p_bust": counts["bust"] / tot,
        "p_timeout": counts["timeout"] / tot,
        "avg_days_pass": float(np.mean(days_pass)) if days_pass else float("nan"),
        "avg_days_bust": float(np.mean(days_bust)) if days_bust else float("nan"),
        "avg_days_any": float(np.mean(days_any)) if days_any else float("nan"),
    }


def binomial_geq_k(p: float, n: int, k: int) -> float:
    return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))


def expected_days_to_3_fundeds(p: float, avg_days_per_account: float, concurrent: int = 2) -> float:
    """Approximate wall-clock days to get 3 fundeds with `concurrent` accounts
    running and 1 finishing -> 1 new starting (rolling pipeline).

    Each account-slot finishes in avg_days_per_account days on average; with
    `concurrent` slots, the system resolves accounts at rate
    `concurrent / avg_days_per_account` per day. To get 3 fundeds, we need
    on average 3/p account resolutions (since each resolution is a Bernoulli
    trial with success prob p). Wall-clock days = (3/p) * avg_days_per_account
    / concurrent.
    """
    if p <= 0:
        return float("inf")
    expected_resolutions = 3.0 / p
    return expected_resolutions * avg_days_per_account / concurrent


def main() -> None:
    print("Loading bars...", flush=True)
    bars = build_full_20min_series(PARQUET, PICKLES)
    print(f"  bars: {len(bars):,}  range: {bars.index.min()} -> {bars.index.max()}\n", flush=True)

    configs = [
        ("Best-PF      ", 1.25, 1.50),
        ("Symmetric 1:1", 1.50, 1.50),
        ("HighWR LowRR ", 1.50, 0.50),
        ("Best-Gross   ", 1.25, 3.00),
        ("Tight SL     ", 1.00, 1.50),
    ]

    R_grid = [200, 300, 400, 500, 600, 750, 1000]

    rows = []
    for name, sl, tp in configs:
        trades = run_asym(bars, sl, tp)
        trades = trades.sort_values("entry_time").reset_index(drop=True)
        for R in R_grid:
            pnls = scale_trades(trades, R)
            r = evaluate_R(pnls)
            p = r["p_funded"]
            d3_2 = expected_days_to_3_fundeds(p, r["avg_days_any"], concurrent=2)
            rows.append({
                "config": name,
                "sl_mult": sl,
                "tp_mult": tp,
                "R_$": R,
                "p_funded": p,
                "p_pass_any": r["p_pass_any"],
                "p_bust": r["p_bust"],
                "p_timeout": r["p_timeout"],
                "avg_days_pass": r["avg_days_pass"],
                "avg_days_any": r["avg_days_any"],
                "P>=3/10": binomial_geq_k(p, N_ACCOUNTS, 3),
                "P>=4/10": binomial_geq_k(p, N_ACCOUNTS, 4),
                "E_days_to_3_at_2_slots": d3_2,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "propfirm_strategy_sweep.csv", index=False)

    # Pivot: pass rate per config x R
    for metric, title in [
        ("p_funded", "P(funded) per account (%)"),
        ("P>=3/10", "P(>=3 of 10 funded) (%)"),
        ("P>=4/10", "P(>=4 of 10 funded) (%)"),
        ("avg_days_pass", "Avg days per PASSING eval"),
        ("E_days_to_3_at_2_slots", "Expected wall-clock days to 3 fundeds (2 concurrent)"),
    ]:
        piv = df.pivot(index="config", columns="R_$", values=metric)
        if "%" in title:
            piv = (piv * 100).round(1)
        else:
            piv = piv.round(1)
        print(f"\n=== {title} ===")
        with pd.option_context("display.width", 240):
            print(piv.to_string())

    # Best overall
    print("\n=== Top configs by P(>=3/10) ===")
    top = df.sort_values("P>=3/10", ascending=False).head(10)
    show = top[["config", "sl_mult", "tp_mult", "R_$", "p_funded", "P>=3/10", "P>=4/10",
                "avg_days_pass", "p_bust", "E_days_to_3_at_2_slots"]].copy()
    show["p_funded"] = (show["p_funded"] * 100).round(1)
    show["P>=3/10"] = (show["P>=3/10"] * 100).round(1)
    show["P>=4/10"] = (show["P>=4/10"] * 100).round(1)
    show["p_bust"] = (show["p_bust"] * 100).round(1)
    show["avg_days_pass"] = show["avg_days_pass"].round(1)
    show["E_days_to_3_at_2_slots"] = show["E_days_to_3_at_2_slots"].round(1)
    with pd.option_context("display.width", 200):
        print(show.to_string(index=False))

    # Sweet spot: maximize EV per calendar day
    print("\n=== Top configs by 'fastest 3 fundeds' (lowest E_days, requiring P>=3/10 >= 50%) ===")
    sub = df[df["P>=3/10"] >= 0.50].copy()
    sub = sub.sort_values("E_days_to_3_at_2_slots").head(10)
    show = sub[["config", "sl_mult", "tp_mult", "R_$", "p_funded", "P>=3/10", "P>=4/10",
                "avg_days_pass", "E_days_to_3_at_2_slots"]].copy()
    show["p_funded"] = (show["p_funded"] * 100).round(1)
    show["P>=3/10"] = (show["P>=3/10"] * 100).round(1)
    show["P>=4/10"] = (show["P>=4/10"] * 100).round(1)
    show["avg_days_pass"] = show["avg_days_pass"].round(1)
    show["E_days_to_3_at_2_slots"] = show["E_days_to_3_at_2_slots"].round(1)
    with pd.option_context("display.width", 200):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Hola Prime 2-step challenge — phase 1 and phase 2 separately.
Both phases use STATIC $8k DD from initial $100k balance.

Phase 1: target $8,000 (8%), daily loss $5k, $8k static DD
Phase 2: target $5,000 (5%), daily loss $5k, $8k static DD (account resets to $100k)

MT5 CFD slippage applied (Hola Prime NAS100).
Test MNQ sizes 1-5, report:
  - P(phase pass), median days, p25/p75
  - P(both phases pass), expected total days
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
COMBINED_MAE = RESULTS_DIR / "combined_3way_trades_with_mae.csv"

ACCT_SIZE       = 100_000
DAILY_LOSS_LIM  = 5_000
STATIC_DD       = 10_000  # $10k buffer (corrected per user)

P1_TARGET = 8_000
P2_TARGET = 5_000

HORIZON_PER_PHASE = 365  # generous (no minimum days)
N_SIMS = 5000

# MT5 CFD slippage
SLIPPAGE = {"RV": 28.0, "B2": 28.0, "OD": 70.0}
OD_MART_MULT = 1.25


def load_daily():
    df = pd.read_csv(COMBINED_MAE)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    df["slip_$"] = df["strat"].map(SLIPPAGE).fillna(0.0)
    od_mask = df["strat"] == "OD"
    df.loc[od_mask, "slip_$"] *= OD_MART_MULT
    df["pnl_after_slip"] = df["pnl_$"] - df["slip_$"]
    df["mae_after_slip"] = df["mae_$"] - df["slip_$"]
    df = df.sort_values(["date", "entry_ts"])
    out = []
    for d, g in df.groupby("date", sort=True):
        out.append(list(zip(g["pnl_after_slip"].astype(float),
                            g["mae_after_slip"].astype(float))))
    return out


def sim_phase(daily, mnq, rng, target):
    """One phase. Returns (passed, days_used, fail_reason)."""
    scale = mnq * 0.1
    balance = ACCT_SIZE
    prev_eod = ACCT_SIZE
    floor = ACCT_SIZE - STATIC_DD
    target_bal = ACCT_SIZE + target

    for day in range(HORIZON_PER_PHASE):
        idx = rng.integers(0, len(daily))
        if not daily[idx]:
            continue
        day_pnl = 0.0
        for pnl_nq, mae_nq in daily[idx]:
            mae_d = mae_nq * scale; pnl_d = pnl_nq * scale
            eq_dip = balance + day_pnl + mae_d
            if eq_dip <= floor:
                return (False, day + 1, "dd")
            if (prev_eod - eq_dip) >= DAILY_LOSS_LIM:
                return (False, day + 1, "daily")
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur <= floor:
                return (False, day + 1, "dd")
            if (prev_eod - cur) >= DAILY_LOSS_LIM:
                return (False, day + 1, "daily")
            if cur >= target_bal:
                return (True, day + 1, "passed")
        balance += day_pnl
        prev_eod = balance
    return (False, HORIZON_PER_PHASE, "horizon")


def main():
    print("Setup: Hola Prime 2-step challenge, STATIC $10k DD, MT5 CFD slippage")
    print(f"  Phase 1: 8% target ($8k), $5k daily, $10k static DD")
    print(f"  Phase 2: 5% target ($5k), $5k daily, $10k static DD (account resets)")
    print(f"  N sims: {N_SIMS}\n")

    daily = load_daily()
    rng = np.random.default_rng(2026)

    print(f"{'MNQ':>4}  {'P1_pass%':>9}  {'P1_med_d':>9}  {'P1_p25/p75':>11}  "
          f"{'P2_pass%':>9}  {'P2_med_d':>9}  {'P2_p25/p75':>11}  "
          f"{'BOTH_pass%':>11}  {'Total_med_d':>12}")
    rows = []
    for mnq in [1, 2, 3, 4, 5]:
        # Phase 1
        p1_results = [sim_phase(daily, mnq, rng, P1_TARGET) for _ in range(N_SIMS)]
        p1_passes = [d for p, d, _ in p1_results if p]
        p1_pass_rate = len(p1_passes) / N_SIMS

        # Phase 2 (assume passed P1, account resets — same distribution)
        p2_results = [sim_phase(daily, mnq, rng, P2_TARGET) for _ in range(N_SIMS)]
        p2_passes = [d for p, d, _ in p2_results if p]
        p2_pass_rate = len(p2_passes) / N_SIMS

        # Combined
        both_pass_rate = p1_pass_rate * p2_pass_rate

        # Days
        p1_med = int(np.median(p1_passes)) if p1_passes else -1
        p1_p25 = int(np.percentile(p1_passes, 25)) if p1_passes else -1
        p1_p75 = int(np.percentile(p1_passes, 75)) if p1_passes else -1
        p2_med = int(np.median(p2_passes)) if p2_passes else -1
        p2_p25 = int(np.percentile(p2_passes, 25)) if p2_passes else -1
        p2_p75 = int(np.percentile(p2_passes, 75)) if p2_passes else -1
        total_med = p1_med + p2_med if p1_med > 0 and p2_med > 0 else -1

        print(f"{mnq:>4}  {p1_pass_rate*100:>8.1f}%  {p1_med:>9}  {p1_p25:>4}/{p1_p75:>5}  "
              f"{p2_pass_rate*100:>8.1f}%  {p2_med:>9}  {p2_p25:>4}/{p2_p75:>5}  "
              f"{both_pass_rate*100:>10.1f}%  {total_med:>12}")
        rows.append(dict(mnq=mnq, p1_pass=p1_pass_rate, p2_pass=p2_pass_rate,
                          both_pass=both_pass_rate, p1_med_days=p1_med, p2_med_days=p2_med,
                          total_med_days=total_med))

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "holaprime_2step_phases.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR / 'holaprime_2step_phases.csv'}")


if __name__ == "__main__":
    main()

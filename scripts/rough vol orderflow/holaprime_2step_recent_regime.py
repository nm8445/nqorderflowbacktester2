"""
Hola Prime 2-step challenge pass rate on the RECENT REGIME (mid-2024 to 2026 only).
Same rules as full-sample test: $10k static DD, $5k daily, MT5 CFD slippage.
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
STATIC_DD       = 10_000
P1_TARGET = 8_000
P2_TARGET = 5_000
HORIZON_PER_PHASE = 365
N_SIMS = 5000

SLIPPAGE = {"RV": 28.0, "B2": 28.0, "OD": 70.0}
OD_MART_MULT = 1.25

# Recent regime: 2025 onwards
START_DATE = pd.Timestamp("2025-01-01", tz="America/New_York").date()


def load_daily(start_date=None):
    df = pd.read_csv(COMBINED_MAE)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    if start_date is not None:
        df = df[df["date"] >= start_date].copy()
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
    return out, df


def sim_phase(daily, mnq, rng, target):
    scale = mnq * 0.1
    balance = ACCT_SIZE; prev_eod = ACCT_SIZE
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
            if eq_dip <= floor: return (False, day + 1, "dd")
            if (prev_eod - eq_dip) >= DAILY_LOSS_LIM: return (False, day + 1, "daily")
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur <= floor: return (False, day + 1, "dd")
            if (prev_eod - cur) >= DAILY_LOSS_LIM: return (False, day + 1, "daily")
            if cur >= target_bal: return (True, day + 1, "passed")
        balance += day_pnl
        prev_eod = balance
    return (False, HORIZON_PER_PHASE, "horizon")


def run_set(daily_set, label):
    rng = np.random.default_rng(2026)
    print(f"\n=== {label} ({len(daily_set)} historical trading days) ===")
    print(f"{'MNQ':>4}  {'P1_pass%':>9}  {'P1_med_d':>9}  {'P2_pass%':>9}  {'P2_med_d':>9}  "
          f"{'BOTH_pass%':>11}  {'Total_med_d':>12}")
    rows = []
    for mnq in [1, 2, 3, 4, 5]:
        p1_results = [sim_phase(daily_set, mnq, rng, P1_TARGET) for _ in range(N_SIMS)]
        p1_passes = [d for p, d, _ in p1_results if p]
        p1_pass_rate = len(p1_passes) / N_SIMS
        p2_results = [sim_phase(daily_set, mnq, rng, P2_TARGET) for _ in range(N_SIMS)]
        p2_passes = [d for p, d, _ in p2_results if p]
        p2_pass_rate = len(p2_passes) / N_SIMS
        both_pass_rate = p1_pass_rate * p2_pass_rate
        p1_med = int(np.median(p1_passes)) if p1_passes else -1
        p2_med = int(np.median(p2_passes)) if p2_passes else -1
        total_med = p1_med + p2_med if p1_med > 0 and p2_med > 0 else -1
        print(f"{mnq:>4}  {p1_pass_rate*100:>8.1f}%  {p1_med:>9}  {p2_pass_rate*100:>8.1f}%  {p2_med:>9}  "
              f"{both_pass_rate*100:>10.1f}%  {total_med:>12}")
        rows.append(dict(label=label, mnq=mnq, p1_pass=p1_pass_rate, p2_pass=p2_pass_rate,
                          both_pass=both_pass_rate, p1_med_days=p1_med, p2_med_days=p2_med,
                          total_med_days=total_med))
    return rows


def main():
    print("Setup: Hola Prime 2-step, STATIC $10k DD, MT5 CFD slippage")
    print(f"  Comparing: full sample vs recent regime (from {START_DATE})\n")

    daily_full, df_full = load_daily()
    daily_recent, df_recent = load_daily(start_date=START_DATE)

    print(f"Full sample:    {len(daily_full)} days,  total PnL (post-slip): ${df_full['pnl_after_slip'].sum():,.0f}")
    print(f"Recent (from {START_DATE}): {len(daily_recent)} days,  total PnL (post-slip): ${df_recent['pnl_after_slip'].sum():,.0f}")
    print(f"  Recent PnL per day: ${df_recent['pnl_after_slip'].sum()/len(daily_recent):,.0f} (full sample: ${df_full['pnl_after_slip'].sum()/len(daily_full):,.0f})")

    rows = []
    rows += run_set(daily_full, "FULL SAMPLE (2020-12 -> 2026-05)")
    rows += run_set(daily_recent, f"RECENT REGIME ({START_DATE} -> 2026-05)")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "holaprime_2step_recent_regime.csv", index=False)


if __name__ == "__main__":
    main()

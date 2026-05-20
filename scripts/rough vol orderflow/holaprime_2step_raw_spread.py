"""
Hola Prime 2-step challenge — compare Standard vs Raw Spread + Swap-Free slippage.
2025+ regime only.
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
P1_TARGET       = 8_000
P2_TARGET       = 5_000
HORIZON_PER_PHASE = 365
N_SIMS          = 5000

START_DATE = pd.Timestamp("2025-01-01", tz="America/New_York").date()

# Slippage scenarios (NQ basis $ per trade)
SCENARIOS = {
    "Standard CFD":              {"RV": 28.0, "B2": 28.0, "OD": 70.0, "OD_mart_mult": 1.25},
    "Raw spread + swap-free CONSERVATIVE":   {"RV": 20.0, "B2": 20.0, "OD": 45.0, "OD_mart_mult": 1.25},
    "Raw spread + swap-free OPTIMISTIC":     {"RV": 14.0, "B2": 14.0, "OD": 30.0, "OD_mart_mult": 1.25},
    "CME futures (for reference)":           {"RV": 8.0,  "B2": 8.0,  "OD": 12.0, "OD_mart_mult": 1.25},
}


def load_daily(slippage_dict, start_date=None):
    df = pd.read_csv(COMBINED_MAE)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    if start_date is not None:
        df = df[df["date"] >= start_date].copy()
    df["slip_$"] = df["strat"].map(slippage_dict).fillna(0.0)
    od_mask = df["strat"] == "OD"
    df.loc[od_mask, "slip_$"] *= slippage_dict["OD_mart_mult"]
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
        if not daily[idx]: continue
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


def main():
    print(f"Setup: 2-step challenge, STATIC $10k DD, 2025+ regime only ({START_DATE} onwards)\n")
    rng = np.random.default_rng(2026)
    all_rows = []
    for sc_name, slip in SCENARIOS.items():
        daily, df_slice = load_daily(slip, start_date=START_DATE)
        avg_per_day = df_slice["pnl_after_slip"].sum() / len(daily)
        print(f"\n=== {sc_name} (PnL post-slip: ${avg_per_day:.0f}/day) ===")
        print(f"{'MNQ':>4}  {'P1_pass%':>9}  {'P1_med_d':>9}  {'P2_pass%':>9}  {'P2_med_d':>9}  "
              f"{'BOTH%':>7}  {'Total_d':>8}")
        for mnq in [2, 3, 4, 5]:
            p1 = [sim_phase(daily, mnq, rng, P1_TARGET) for _ in range(N_SIMS)]
            p1_passes = [d for p, d, _ in p1 if p]
            p1_rate = len(p1_passes) / N_SIMS
            p2 = [sim_phase(daily, mnq, rng, P2_TARGET) for _ in range(N_SIMS)]
            p2_passes = [d for p, d, _ in p2 if p]
            p2_rate = len(p2_passes) / N_SIMS
            both = p1_rate * p2_rate
            p1_med = int(np.median(p1_passes)) if p1_passes else -1
            p2_med = int(np.median(p2_passes)) if p2_passes else -1
            tot = p1_med + p2_med if p1_med > 0 and p2_med > 0 else -1
            print(f"{mnq:>4}  {p1_rate*100:>8.1f}%  {p1_med:>9}  {p2_rate*100:>8.1f}%  {p2_med:>9}  "
                  f"{both*100:>6.1f}%  {tot:>8}")
            all_rows.append(dict(scenario=sc_name, mnq=mnq,
                                  p1_pass=p1_rate, p2_pass=p2_rate, both_pass=both,
                                  p1_med=p1_med, p2_med=p2_med, total_med=tot))
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "holaprime_2step_raw_spread.csv", index=False)


if __name__ == "__main__":
    main()

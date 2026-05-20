"""
Tradeify Select 50k funded — test what happens if we DROP OD from the funded combo.
OD has the catastrophic MAE that busts the $2k DD; RV+B2 are RTH with much smaller MAE.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
COMBINED_MAE = RESULTS_DIR / "combined_3way_trades_with_mae.csv"

ACCT_SIZE       = 50_000
DAILY_LOSS_LIM  = 1_000
TRAILING_DD     = 2_000
LOCK_BALANCE    = 52_100
LOCK_FLOOR      = 50_100
PAYOUT_THRESH   = 1_000
HORIZON_DAYS    = 252
N_SIMS          = 5000
SLIPPAGE        = {"RV": 8.0, "B2": 8.0, "OD": 10.0}


def load_daily(strats_to_keep):
    df = pd.read_csv(COMBINED_MAE)
    df = df[df["strat"].isin(strats_to_keep)].copy()
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    df["slip_$"] = df["strat"].map(SLIPPAGE).fillna(0.0)
    od_mask = df["strat"] == "OD"
    df.loc[od_mask, "slip_$"] = df.loc[od_mask, "slip_$"] * 1.25
    df["pnl_after_slip"] = df["pnl_$"] - df["slip_$"]
    df["mae_after_slip"] = df["mae_$"] - df["slip_$"]
    df = df.sort_values(["date", "entry_ts"])
    out = []
    for d, g in df.groupby("date", sort=True):
        out.append(list(zip(g["pnl_after_slip"].astype(float),
                            g["mae_after_slip"].astype(float))))
    return out


def sim_funded(daily, mnq, rng, horizon=HORIZON_DAYS):
    scale = mnq * 0.1
    balance = ACCT_SIZE; peak = ACCT_SIZE; locked = False
    prev_eod = ACCT_SIZE
    total_extracted = 0.0; n_payouts = 0; days_alive = 0
    bust_day = -1
    for day in range(horizon):
        idx = rng.integers(0, len(daily))
        if not daily[idx]:
            days_alive = day + 1
            continue
        day_pnl = 0.0; busted = False
        for pnl_nq, mae_nq in daily[idx]:
            mae_d = mae_nq * scale; pnl_d = pnl_nq * scale
            eq_dip = balance + day_pnl + mae_d
            if eq_dip > peak: peak = eq_dip
            cur_floor = LOCK_FLOOR if locked else (peak - TRAILING_DD)
            if eq_dip <= cur_floor or (prev_eod - eq_dip) >= DAILY_LOSS_LIM:
                busted = True; break
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur > peak: peak = cur
            if not locked and cur >= LOCK_BALANCE: locked = True
            cur_floor = LOCK_FLOOR if locked else (peak - TRAILING_DD)
            if cur <= cur_floor or (prev_eod - cur) >= DAILY_LOSS_LIM:
                busted = True; break
        if busted:
            bust_day = day; days_alive = day; break
        days_alive = day + 1
        balance += day_pnl
        prev_eod = balance
        if balance >= ACCT_SIZE + PAYOUT_THRESH:
            total_extracted += balance - ACCT_SIZE
            n_payouts += 1
            balance = ACCT_SIZE; peak = ACCT_SIZE; locked = False; prev_eod = ACCT_SIZE
    return dict(total_extracted=total_extracted, n_payouts=n_payouts,
                days_alive=days_alive, busted=bust_day >= 0)


def run_scenario(daily, label):
    print(f"\n=== {label} ===")
    print(f"{'MNQ':>4}  {'bust%':>6}  {'median_$':>10}  {'mean_$':>10}  "
          f"{'p25_$':>9}  {'p75_$':>9}  {'med_payouts':>11}  {'med_days_alive':>15}")
    rng = np.random.default_rng(2026)
    rows = []
    for mnq in [1, 2, 3, 4]:
        sims = [sim_funded(daily, mnq, rng) for _ in range(N_SIMS)]
        ex = np.array([s["total_extracted"] for s in sims])
        n_bust = sum(1 for s in sims if s["busted"])
        n_payouts = np.array([s["n_payouts"] for s in sims])
        days_alive = np.array([s["days_alive"] for s in sims])
        print(f"{mnq:>4}  {n_bust/N_SIMS*100:>5.1f}%  {np.median(ex):>+10,.0f}  {ex.mean():>+10,.0f}  "
              f"{np.percentile(ex, 25):>+9,.0f}  {np.percentile(ex, 75):>+9,.0f}  "
              f"{int(np.median(n_payouts)):>11}  {int(np.median(days_alive)):>15}")
        rows.append(dict(label=label, mnq=mnq, bust_rate=n_bust/N_SIMS,
                          median_extracted=float(np.median(ex)),
                          mean_extracted=float(ex.mean()),
                          p25=float(np.percentile(ex, 25)),
                          p75=float(np.percentile(ex, 75))))
    return rows


def main():
    print("Loading...")
    daily_all = load_daily(["RV", "B2", "OD"])
    daily_no_od = load_daily(["RV", "B2"])
    print(f"  All-3 days: {len(daily_all)}; RV+B2 only days: {len(daily_no_od)}")

    rows = []
    rows += run_scenario(daily_all, "Full combo (RV+B2+OD)")
    rows += run_scenario(daily_no_od, "RV + B2 only (no OD)")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tradeify_funded_rvbtwo.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR / 'tradeify_funded_rvbtwo.csv'}")


if __name__ == "__main__":
    main()

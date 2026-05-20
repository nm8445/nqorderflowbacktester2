"""
Tradeify Select 50k funded: 1 account at MNQ=1.
Compare:
  - Full combo (RV+B2+OD)
  - RV + B2 only (no OD)
Report:
  - P(at least 1 payout before bust)
  - P(at least 2 payouts)
  - Days to first payout
  - Mean total extracted
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
N_SIMS          = 10_000
SLIPPAGE        = {"RV": 8.0, "B2": 8.0, "OD": 10.0}
MNQ             = 1


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


def sim_one(daily, mnq, rng):
    scale = mnq * 0.1
    balance = ACCT_SIZE; peak = ACCT_SIZE; locked = False
    prev_eod = ACCT_SIZE
    payouts = []   # (day, amount)
    bust_day = -1
    for day in range(HORIZON_DAYS):
        idx = rng.integers(0, len(daily))
        if not daily[idx]:
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
            bust_day = day; break
        balance += day_pnl
        prev_eod = balance
        if balance >= ACCT_SIZE + PAYOUT_THRESH:
            payouts.append((day, balance - ACCT_SIZE))
            balance = ACCT_SIZE; peak = ACCT_SIZE; locked = False; prev_eod = ACCT_SIZE
    return dict(payouts=payouts, bust_day=bust_day,
                total_extracted=sum(p[1] for p in payouts))


def report(label, sims):
    n = len(sims)
    first_payout_days = []
    second_payout_days = []
    n_with_1 = 0; n_with_2 = 0; n_with_3 = 0
    total_extr = []
    n_payouts_list = []
    for s in sims:
        if s["payouts"]:
            first_payout_days.append(s["payouts"][0][0])
            n_with_1 += 1
            if len(s["payouts"]) >= 2:
                second_payout_days.append(s["payouts"][1][0])
                n_with_2 += 1
            if len(s["payouts"]) >= 3:
                n_with_3 += 1
        n_payouts_list.append(len(s["payouts"]))
        total_extr.append(s["total_extracted"])

    print(f"\n=== {label} ===")
    print(f"  n_sims: {n}")
    print(f"  P(>=1 payout):           {100*n_with_1/n:.1f}%")
    print(f"  P(>=2 payouts):          {100*n_with_2/n:.1f}%")
    print(f"  P(>=3 payouts):          {100*n_with_3/n:.1f}%")
    print(f"  P(bust before any payout): {100*(1-n_with_1/n):.1f}%")
    if first_payout_days:
        fp = np.array(first_payout_days)
        print(f"\n  Days to FIRST payout (among the {n_with_1} that achieved one):")
        print(f"    median:    {int(np.median(fp))} business days")
        print(f"    mean:      {fp.mean():.1f}")
        print(f"    p25 / p75: {int(np.percentile(fp, 25))} / {int(np.percentile(fp, 75))}")
        print(f"    p10 / p90: {int(np.percentile(fp, 10))} / {int(np.percentile(fp, 90))}")
    if second_payout_days:
        sp = np.array(second_payout_days)
        print(f"\n  Days to SECOND payout (among the {n_with_2} that achieved 2+):")
        print(f"    median:    {int(np.median(sp))} business days")
        print(f"    mean:      {sp.mean():.1f}")

    extr = np.array(total_extr)
    npay = np.array(n_payouts_list)
    print(f"\n  Total $ extracted across full year:")
    print(f"    median: ${np.median(extr):,.0f}")
    print(f"    mean:   ${extr.mean():,.0f}")
    print(f"    p25 / p75: ${np.percentile(extr, 25):,.0f} / ${np.percentile(extr, 75):,.0f}")
    print(f"  Mean #payouts: {npay.mean():.2f}  Median: {int(np.median(npay))}")


def main():
    rng = np.random.default_rng(2026)
    print("Loading...")
    daily_all = load_daily(["RV", "B2", "OD"])
    daily_no_od = load_daily(["RV", "B2"])
    print(f"  Full-combo days: {len(daily_all)}; RV+B2-only days: {len(daily_no_od)}")
    print(f"  1 funded account, MNQ=1, $1k payout threshold, 252-day horizon, 10k sims")

    print("\nRunning full combo (RV+B2+OD)...")
    sims_full = [sim_one(daily_all, MNQ, rng) for _ in range(N_SIMS)]
    report("Full combo (RV + B2 + OD)", sims_full)

    print("\nRunning RV+B2 only...")
    sims_rvb = [sim_one(daily_no_od, MNQ, rng) for _ in range(N_SIMS)]
    report("RV + B2 only (no OD)", sims_rvb)


if __name__ == "__main__":
    main()

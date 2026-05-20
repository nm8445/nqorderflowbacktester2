"""
For MNQ=1.5 vs MNQ=4 (on-demand $2k threshold payouts):
  - Days to bust (for busts) / total trading days (for survivors)
  - Days to FIRST payout
  - Cash extracted BEFORE bust (per busted run)
  - Cash extracted (per survivor run)
  - Payout count and avg size
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
COMBINED_MAE = RESULTS_DIR / "combined_3way_trades_with_mae.csv"

ACCT = 100_000
MAX_DD = 10_000
LOCK_PROFIT = 5_000
DAILY_LOSS = 5_000
HORIZON = 252
DOWNTIME = 2
PAYOUT_THRESHOLD = 2_000  # on-demand $2k
N_SIMS = 10_000


def load_daily():
    df = pd.read_csv(COMBINED_MAE)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    df = df.sort_values(["date", "entry_ts"])
    return [list(zip(g["pnl_$"].astype(float), g["mae_$"].astype(float)))
            for _, g in df.groupby("date", sort=True)]


def simulate(daily, mnq, rng):
    scale = mnq * 0.1
    balance = ACCT; peak = ACCT; prev_eod = ACCT
    downtime = 0
    payouts = []
    busted_day = -1
    first_payout_day = -1
    days_traded = 0
    for day in range(HORIZON):
        if downtime > 0:
            downtime -= 1
            continue
        data = daily[rng.integers(0, len(daily))]
        days_traded += 1
        day_pnl = 0.0
        busted_today = False
        for pnl_nq, mae_nq in data:
            mae_d = mae_nq * scale; pnl_d = pnl_nq * scale
            eq_dip = balance + day_pnl + mae_d
            if eq_dip > peak: peak = eq_dip
            if peak >= ACCT + LOCK_PROFIT:
                floor = max(ACCT, peak - MAX_DD)
            else:
                floor = peak - MAX_DD
            if eq_dip <= floor or (prev_eod - eq_dip) >= DAILY_LOSS:
                busted_today = True; break
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur > peak: peak = cur
            if peak >= ACCT + LOCK_PROFIT:
                floor = max(ACCT, peak - MAX_DD)
            else:
                floor = peak - MAX_DD
            if cur <= floor or (prev_eod - cur) >= DAILY_LOSS:
                busted_today = True; break
        if busted_today:
            busted_day = day
            break
        balance += day_pnl
        prev_eod = balance
        if balance >= ACCT + PAYOUT_THRESHOLD:
            payouts.append((day, balance - ACCT))
            if first_payout_day < 0:
                first_payout_day = day
            balance = ACCT; peak = ACCT; prev_eod = ACCT
            downtime = DOWNTIME
    return dict(
        busted=busted_day >= 0,
        busted_day=busted_day,
        days_traded=days_traded,
        payouts=payouts,
        first_payout_day=first_payout_day,
        final_excess=(balance - ACCT) if busted_day < 0 else 0,
    )


def report(label, sims):
    n = len(sims)
    n_bust = sum(1 for s in sims if s["busted"])
    n_surv = n - n_bust
    print(f"\n{'='*80}\n{label}\n{'='*80}")
    print(f"Total sims: {n}  |  Bust: {n_bust} ({n_bust/n*100:.1f}%)  |  Survived: {n_surv}")

    # Days to first payout (across ALL sims that had >=1 payout)
    fp = [s["first_payout_day"] for s in sims if s["first_payout_day"] >= 0]
    fp_arr = np.array(fp) if fp else np.array([])
    print(f"\nDays to FIRST payout (sims that achieved >=1 payout):")
    print(f"  n with first payout:  {len(fp)} ({len(fp)/n*100:.1f}% of sims)")
    if len(fp):
        print(f"  median:               {int(np.median(fp_arr))} business days")
        print(f"  mean:                 {fp_arr.mean():.1f}")
        print(f"  p25 / p75:            {int(np.percentile(fp_arr, 25))} / {int(np.percentile(fp_arr, 75))}")

    # Bust path: payouts before bust
    bust_sims = [s for s in sims if s["busted"]]
    if bust_sims:
        days_to_bust = np.array([s["busted_day"] for s in bust_sims])
        n_payouts_pre_bust = np.array([len(s["payouts"]) for s in bust_sims])
        cash_pre_bust = np.array([sum(p[1] for p in s["payouts"]) for s in bust_sims])
        print(f"\nBUSTED runs ({n_bust}):")
        print(f"  Days to bust:               median {int(np.median(days_to_bust))}  mean {days_to_bust.mean():.1f}  p25/p75 {int(np.percentile(days_to_bust,25))}/{int(np.percentile(days_to_bust,75))}")
        print(f"  Payouts before bust:        median {int(np.median(n_payouts_pre_bust))}  mean {n_payouts_pre_bust.mean():.2f}")
        print(f"  Cash extracted before bust: median ${np.median(cash_pre_bust):,.0f}  mean ${cash_pre_bust.mean():,.0f}")
        print(f"  $0 extracted (bust before any payout): {(n_payouts_pre_bust == 0).sum()} runs ({(n_payouts_pre_bust == 0).mean()*100:.1f}% of busts)")

    # Survivor path
    surv = [s for s in sims if not s["busted"]]
    if surv:
        surv_payouts = np.array([len(s["payouts"]) for s in surv])
        surv_cash = np.array([sum(p[1] for p in s["payouts"]) + s["final_excess"] for s in surv])
        print(f"\nSURVIVED runs ({n_surv}):")
        print(f"  Payouts in year:    median {int(np.median(surv_payouts))}  mean {surv_payouts.mean():.1f}")
        print(f"  Total $ extracted:  median ${np.median(surv_cash):,.0f}  mean ${surv_cash.mean():,.0f}")
        print(f"  p25 / p75:          ${np.percentile(surv_cash, 25):,.0f} / ${np.percentile(surv_cash, 75):,.0f}")

    # Expected value (unconditional, including busts as zero)
    all_cash = []
    for s in sims:
        c = sum(p[1] for p in s["payouts"]) + (s["final_excess"] if not s["busted"] else 0)
        all_cash.append(c)
    all_cash = np.array(all_cash)
    print(f"\nUNCONDITIONAL expected (busts count their extracted cash, no account fee):")
    print(f"  Mean extracted across ALL sims:  ${all_cash.mean():,.0f}")
    print(f"  Median:                          ${np.median(all_cash):,.0f}")
    print(f"  $0 outcomes:                     {(all_cash == 0).sum()} ({(all_cash == 0).mean()*100:.1f}%)")


def main():
    daily = load_daily()
    rng = np.random.default_rng(2026)
    for mnq, label in [(1.5, "MNQ = 1.5"), (4.0, "MNQ = 4.0")]:
        sims = [simulate(daily, mnq, rng) for _ in range(N_SIMS)]
        report(label, sims)


if __name__ == "__main__":
    main()

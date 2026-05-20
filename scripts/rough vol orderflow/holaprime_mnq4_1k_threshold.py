"""
Hola Prime $100k / $10k DD funded — MNQ=4, MT5 CFD slippage, $1k payout threshold.
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
TRAILING_DD     = 10_000
LOCK_PROFIT     = 5_000
PAYOUT_THRESH   = 1_000   # withdraw at +$1k
HORIZON_DAYS    = 252
N_SIMS          = 10_000

# MT5 CFD slippage (HIGHER than futures)
SLIPPAGE = {"RV": 28.0, "B2": 28.0, "OD": 70.0}
OD_MART_MULT = 1.25

MNQ             = 4


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


def sim_one(daily, mnq, rng):
    scale = mnq * 0.1
    balance = ACCT_SIZE; peak = ACCT_SIZE; locked = False
    prev_eod = ACCT_SIZE
    payouts = []
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
            if peak >= ACCT_SIZE + LOCK_PROFIT:
                cur_floor = max(ACCT_SIZE, peak - TRAILING_DD)
            else:
                cur_floor = peak - TRAILING_DD
            if eq_dip <= cur_floor or (prev_eod - eq_dip) >= DAILY_LOSS_LIM:
                busted = True; break
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur > peak: peak = cur
            if peak >= ACCT_SIZE + LOCK_PROFIT:
                cur_floor = max(ACCT_SIZE, peak - TRAILING_DD)
            else:
                cur_floor = peak - TRAILING_DD
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


def main():
    print(f"Setup: Hola Prime $100k/$10k DD funded, MNQ={MNQ}, $1k payout threshold")
    print(f"  MT5 CFD slippage applied: RV $28, B2 $28, OD $70 (NQ basis × {OD_MART_MULT} for mart)")
    print(f"  Horizon: {HORIZON_DAYS} business days, {N_SIMS} sims\n")

    daily = load_daily()
    rng = np.random.default_rng(2026)
    sims = [sim_one(daily, MNQ, rng) for _ in range(N_SIMS)]

    n = len(sims)
    n_bust = sum(1 for s in sims if s["bust_day"] >= 0)
    n_with_1 = sum(1 for s in sims if s["payouts"])
    n_with_2 = sum(1 for s in sims if len(s["payouts"]) >= 2)
    n_with_3 = sum(1 for s in sims if len(s["payouts"]) >= 3)
    n_with_5 = sum(1 for s in sims if len(s["payouts"]) >= 5)
    n_with_10 = sum(1 for s in sims if len(s["payouts"]) >= 10)

    first_payout_days = [s["payouts"][0][0] for s in sims if s["payouts"]]
    payouts_counts = [len(s["payouts"]) for s in sims]
    total_extr = [s["total_extracted"] for s in sims]
    bust_days = [s["bust_day"] for s in sims if s["bust_day"] >= 0]
    payout_sizes_all = [p[1] for s in sims for p in s["payouts"]]

    print("=== Results ===")
    print(f"\nBust outcome:")
    print(f"  P(bust in 1 yr):           {n_bust/n*100:.1f}%")
    print(f"  P(survive full year):      {(n-n_bust)/n*100:.1f}%")
    if bust_days:
        bd = np.array(bust_days)
        print(f"  Median days-to-bust:       {int(np.median(bd))}  mean: {bd.mean():.1f}")

    print(f"\nPayout reliability:")
    print(f"  P(>=1 payout):    {n_with_1/n*100:.1f}%")
    print(f"  P(>=2 payouts):   {n_with_2/n*100:.1f}%")
    print(f"  P(>=3 payouts):   {n_with_3/n*100:.1f}%")
    print(f"  P(>=5 payouts):   {n_with_5/n*100:.1f}%")
    print(f"  P(>=10 payouts):  {n_with_10/n*100:.1f}%")
    print(f"  P(bust w/o any payout): {(n-n_with_1)/n*100:.1f}%")

    if first_payout_days:
        fp = np.array(first_payout_days)
        print(f"\nDays to FIRST payout (sims that achieved one):")
        print(f"  median: {int(np.median(fp))} business days")
        print(f"  mean:   {fp.mean():.1f}")
        print(f"  p25 / p75: {int(np.percentile(fp, 25))} / {int(np.percentile(fp, 75))}")
        print(f"  p10 / p90: {int(np.percentile(fp, 10))} / {int(np.percentile(fp, 90))}")

    pc = np.array(payouts_counts)
    print(f"\nTotal payouts per account (full year):")
    print(f"  median: {int(np.median(pc))}  mean: {pc.mean():.2f}  p75: {int(np.percentile(pc, 75))}")

    ps = np.array(payout_sizes_all)
    if len(ps):
        print(f"\nIndividual payout SIZE distribution (each withdrawal):")
        print(f"  median: ${np.median(ps):,.0f}  mean: ${ps.mean():,.0f}")
        print(f"  p25 / p75: ${np.percentile(ps, 25):,.0f} / ${np.percentile(ps, 75):,.0f}")
        print(f"  max single payout: ${ps.max():,.0f}")

    te = np.array(total_extr)
    print(f"\nTotal cash extracted per account (sum of all payouts):")
    print(f"  median:    ${np.median(te):,.0f}")
    print(f"  mean:      ${te.mean():,.0f}")
    print(f"  p25 / p75: ${np.percentile(te, 25):,.0f} / ${np.percentile(te, 75):,.0f}")

    # Net of account cost
    print(f"\n=== Net of $250 account cost ===")
    net = te - 250
    print(f"  median net: ${np.median(net):,.0f}")
    print(f"  mean net:   ${net.mean():,.0f}")
    print(f"  P(net positive): {(net > 0).mean()*100:.1f}%")
    print(f"  P(lose money entirely): {(te == 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()

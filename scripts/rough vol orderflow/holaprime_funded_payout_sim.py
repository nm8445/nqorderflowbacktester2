"""
Hola Prime FUNDED account simulation — optimize MNQ size for payout frequency.

Setup: $100k account, $10k max DD (trailing-to-initial), $5k daily loss limit.
Payout = withdraw (balance - $100k) any time balance > $100k.
After payout: 2 business days downtime, account resets to $100k.
Bust = lose the account; assume no replacement here (worst case).
Horizon: 252 business days (1 year).

Compare three payout schedules:
  on_demand_2k:   withdraw whenever profit >= $2,000 (sensible threshold)
  bi_weekly:      withdraw every 10 business days if profit > 0
  monthly:        withdraw every 21 business days if profit > 0

For each (MNQ, schedule), report:
  - Bust rate over the year
  - Expected total $ withdrawn
  - Expected $/calendar-day (counting all 252 days, including downtime/busted)
  - Median number of payouts
  - Average payout size
  - Days to first payout
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLAN_DIR = HERE.parent.parent / "live" / "combined deployment plan"

COMBINED_MAE = RESULTS_DIR / "combined_3way_trades_with_mae.csv"

ACCOUNT_INIT = 100_000
MAX_DD       = 10_000
LOCK_PROFIT  = 5_000   # DD floor locks at initial after this profit
DAILY_LOSS   = 5_000
HORIZON_DAYS = 252
DOWNTIME     = 2

MNQ_SIZES = [0.5, 1, 1.5, 2, 2.5, 3, 4]
SCHEDULES = ["on_demand_2k", "bi_weekly", "monthly"]
N_SIMS = 3000


def load_daily():
    df = pd.read_csv(COMBINED_MAE)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    df = df.sort_values(["date", "entry_ts"])
    out = []
    for d, g in df.groupby("date", sort=True):
        items = list(zip(g["pnl_$"].astype(float), g["mae_$"].astype(float)))
        out.append(items)
    return out


def simulate_day(day_data, scale, balance, peak, prev_eod, dd_floor):
    """Run one trading day. Returns (busted, balance_after, peak_after).
       dd_floor is the current DD floor (caller computes)."""
    day_pnl = 0.0
    for pnl_nq, mae_nq in day_data:
        mae_d = mae_nq * scale
        pnl_d = pnl_nq * scale
        # MAE dip check
        eq_dip = balance + day_pnl + mae_d
        if eq_dip > peak: peak = eq_dip
        # Recompute floor (trailing-to-initial)
        if peak >= ACCOUNT_INIT + LOCK_PROFIT:
            cur_floor = max(ACCOUNT_INIT, peak - MAX_DD)
        else:
            cur_floor = peak - MAX_DD
        if eq_dip <= cur_floor:
            return (True, balance + day_pnl + mae_d, peak)  # busted on MAE dip
        if (prev_eod - eq_dip) >= DAILY_LOSS:
            return (True, balance + day_pnl + mae_d, peak)
        # Realized PnL
        day_pnl += pnl_d
        cur = balance + day_pnl
        if cur > peak: peak = cur
        if peak >= ACCOUNT_INIT + LOCK_PROFIT:
            cur_floor = max(ACCOUNT_INIT, peak - MAX_DD)
        else:
            cur_floor = peak - MAX_DD
        if cur <= cur_floor:
            return (True, cur, peak)
        if (prev_eod - cur) >= DAILY_LOSS:
            return (True, cur, peak)
    return (False, balance + day_pnl, peak)


def should_payout(schedule, day_in_cycle, balance):
    if balance <= ACCOUNT_INIT:
        return False
    if schedule == "on_demand_2k":
        return balance >= ACCOUNT_INIT + 2_000
    if schedule == "bi_weekly":
        return day_in_cycle >= 10  # every 10 business days
    if schedule == "monthly":
        return day_in_cycle >= 21
    return False


def run_sim(daily_lists, mnq, schedule, rng, n_sims=N_SIMS):
    scale = mnq * 0.1
    n_data = len(daily_lists)
    sims = []
    for s in range(n_sims):
        balance = ACCOUNT_INIT; peak = ACCOUNT_INIT; prev_eod = ACCOUNT_INIT
        downtime_left = 0
        payouts = []
        busted_day = -1
        cycle_start_day = 0
        active_days = 0
        for day in range(HORIZON_DAYS):
            if downtime_left > 0:
                downtime_left -= 1
                continue
            day_data = daily_lists[rng.integers(0, n_data)]
            day_in_cycle = day - cycle_start_day
            busted, balance, peak = simulate_day(day_data, scale, balance, peak, prev_eod,
                                                  None)
            active_days += 1
            if busted:
                busted_day = day
                break
            prev_eod = balance
            # Check payout
            if should_payout(schedule, day_in_cycle + 1, balance):
                payouts.append(balance - ACCOUNT_INIT)
                balance = ACCOUNT_INIT; peak = ACCOUNT_INIT; prev_eod = ACCOUNT_INIT
                downtime_left = DOWNTIME
                cycle_start_day = day + 1 + DOWNTIME
        # If end-of-horizon and balance > 100k, withdraw too
        final_unrealized = balance - ACCOUNT_INIT if busted_day < 0 else 0
        sims.append(dict(
            busted=busted_day >= 0,
            busted_day=busted_day,
            n_payouts=len(payouts),
            total_payouts=sum(payouts),
            final_excess=final_unrealized,
            total_extracted=sum(payouts) + final_unrealized,
            first_payout_day=(np.argmax([p > 0 for p in payouts]) if payouts else -1),
            payout_sizes=payouts,
        ))
    return sims


def summarize(sims, mnq, schedule):
    n = len(sims)
    n_bust = sum(1 for s in sims if s["busted"])
    bust_rate = n_bust / n
    n_payouts = np.array([s["n_payouts"] for s in sims])
    total_extracted = np.array([s["total_extracted"] for s in sims])
    return dict(
        mnq=mnq, schedule=schedule,
        bust_rate=bust_rate,
        median_payouts=int(np.median(n_payouts)),
        mean_payouts=float(np.mean(n_payouts)),
        median_total_extracted=float(np.median(total_extracted)),
        mean_total_extracted=float(np.mean(total_extracted)),
        p25_extracted=float(np.percentile(total_extracted, 25)),
        p75_extracted=float(np.percentile(total_extracted, 75)),
        avg_payout=float(np.mean([np.mean(s["payout_sizes"]) for s in sims if s["payout_sizes"]])) if any(s["payout_sizes"] for s in sims) else 0.0,
        per_day_mean=float(np.mean(total_extracted)) / HORIZON_DAYS,
    )


def main():
    print("Loading data...")
    daily = load_daily()
    print(f"  {len(daily)} historical days\n")

    rng = np.random.default_rng(2026)
    rows = []
    print(f"{'MNQ':>4} {'schedule':<14}  {'bust%':>6}  {'med pyt#':>8} {'mean pyt#':>9}  "
          f"{'med $extr':>11} {'mean $extr':>11}  {'avg pyt $':>10}  {'$/cal-day':>10}")
    for mnq in MNQ_SIZES:
        for sched in SCHEDULES:
            sims = run_sim(daily, mnq, sched, rng)
            r = summarize(sims, mnq, sched)
            rows.append(r)
            print(f"{mnq:>4} {sched:<14}  {r['bust_rate']*100:>5.1f}% "
                  f"{r['median_payouts']:>9} {r['mean_payouts']:>9.1f}  "
                  f"{r['median_total_extracted']:>+11,.0f} {r['mean_total_extracted']:>+11,.0f}  "
                  f"{r['avg_payout']:>+10,.0f}  {r['per_day_mean']:>+10,.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "holaprime_funded_payout_sim.csv", index=False)

    # Reports
    print("\n\n=== Expected $/calendar-day ranked (1-year horizon, $100k account / $10k DD) ===")
    df_sorted = df.sort_values("per_day_mean", ascending=False)
    print(df_sorted[["mnq", "schedule", "bust_rate", "mean_payouts",
                     "mean_total_extracted", "avg_payout", "per_day_mean"]].head(15).to_string(index=False))

    # Best by schedule
    print("\n=== Best MNQ size per schedule (low-bust + high-extraction) ===")
    for sched in SCHEDULES:
        sub = df[df["schedule"] == sched]
        # rank: maximize per_day_mean subject to bust_rate constraint
        for bust_cap in [0.05, 0.10, 0.20]:
            safe = sub[sub["bust_rate"] <= bust_cap].sort_values("per_day_mean", ascending=False)
            if len(safe) == 0:
                continue
            top = safe.iloc[0]
            print(f"  {sched} | bust<={int(bust_cap*100)}%:  MNQ={top['mnq']}  "
                   f"bust={top['bust_rate']*100:.1f}%  mean_extract=${top['mean_total_extracted']:+,.0f}/yr  "
                   f"= ${top['per_day_mean']:+.0f}/cal-day  mean_payouts={top['mean_payouts']:.1f}")

    out_txt = PLAN_DIR / "holaprime_funded_payout_sim.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Hola Prime $100k / $10k DD funded simulation\n{'='*80}\n\n")
        f.write(f"Setup: 1-year horizon, 2-day downtime per payout, no consistency rule\n\n")
        f.write(df.to_string(index=False))
    print(f"\nSaved -> {out_txt}")


if __name__ == "__main__":
    main()

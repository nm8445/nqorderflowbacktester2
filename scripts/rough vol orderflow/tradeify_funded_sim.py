"""
Tradeify Select 50k funded phase simulation.

Account specs after passing challenge:
  - Starting balance: $50,000
  - $2,000 trailing max DD, locks at $50,100 floor once balance hits $52,100
  - $1,000 daily loss limit
  - NO consistency rule
  - NO monthly fee
  - Daily payout available (Select Flex)
  - We assume payout when balance >= $51,000 (accumulated +$1k profit)
    Then balance resets to $50,000, DD floor resets.

Returns: distribution of total $ extracted per account before bust, days alive.
Uses futures-realistic slippage (combined log already at NQ basis $20/pt).
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
PAYOUT_THRESH   = 1_000   # withdraw when account is +$1k

HORIZON_DAYS    = 252  # 1 year
N_SIMS          = 5000

# Futures slippage
SLIPPAGE = {"RV": 8.0, "B2": 8.0, "OD": 10.0}

MNQ_GRID = [0.5, 1, 1.5, 2, 3, 4]


def load_daily():
    df = pd.read_csv(COMBINED_MAE)
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
    """Simulate one funded account. Returns dict with total extracted, days alive."""
    scale = mnq * 0.1
    balance = ACCT_SIZE
    peak = ACCT_SIZE
    locked = False
    prev_eod = ACCT_SIZE
    total_extracted = 0.0
    n_payouts = 0
    days_alive = 0
    bust_day = -1
    bust_reason = ""

    for day in range(horizon):
        idx = rng.integers(0, len(daily))
        day_pnl = 0.0
        busted = False
        for pnl_nq, mae_nq in daily[idx]:
            mae_d = mae_nq * scale
            pnl_d = pnl_nq * scale
            # MAE dip check
            eq_dip = balance + day_pnl + mae_d
            if eq_dip > peak: peak = eq_dip
            if locked:
                cur_floor = LOCK_FLOOR
            else:
                cur_floor = peak - TRAILING_DD
            if eq_dip <= cur_floor:
                busted = True; bust_reason = "dd_unreal"; break
            if (prev_eod - eq_dip) >= DAILY_LOSS_LIM:
                busted = True; bust_reason = "daily_unreal"; break
            # Realized PnL
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur > peak: peak = cur
            if not locked and cur >= LOCK_BALANCE:
                locked = True
            if locked:
                cur_floor = LOCK_FLOOR
            else:
                cur_floor = peak - TRAILING_DD
            if cur <= cur_floor:
                busted = True; bust_reason = "dd_real"; break
            if (prev_eod - cur) >= DAILY_LOSS_LIM:
                busted = True; bust_reason = "daily_real"; break
        if busted:
            bust_day = day
            days_alive = day
            break
        days_alive = day + 1
        balance += day_pnl
        prev_eod = balance
        # Payout check at end of day
        if balance >= ACCT_SIZE + PAYOUT_THRESH:
            payout = balance - ACCT_SIZE
            total_extracted += payout
            n_payouts += 1
            balance = ACCT_SIZE
            peak = ACCT_SIZE  # peak resets after payout
            locked = False
            prev_eod = ACCT_SIZE

    return dict(
        total_extracted=total_extracted,
        n_payouts=n_payouts,
        days_alive=days_alive,
        busted=bust_day >= 0,
        bust_day=bust_day,
        bust_reason=bust_reason,
    )


def main():
    print("Loading daily trade list (with futures slippage)...")
    daily = load_daily()
    print(f"  {len(daily)} historical days\n")

    rng = np.random.default_rng(2026)
    print(f"Tradeify Select 50k funded sim — {N_SIMS} sims, {HORIZON_DAYS}-day horizon\n")
    print(f"{'MNQ':>4}  {'bust%':>6}  {'median_$':>10}  {'mean_$':>10}  "
          f"{'p25_$':>9}  {'p75_$':>9}  {'med_payouts':>11}  {'med_days_alive':>15}")
    rows = []
    for mnq in MNQ_GRID:
        sims = [sim_funded(daily, mnq, rng) for _ in range(N_SIMS)]
        ex = np.array([s["total_extracted"] for s in sims])
        n_bust = sum(1 for s in sims if s["busted"])
        bust_pct = n_bust / N_SIMS * 100
        n_payouts = np.array([s["n_payouts"] for s in sims])
        days_alive = np.array([s["days_alive"] for s in sims])
        print(f"{mnq:>4}  {bust_pct:>5.1f}%  {np.median(ex):>+10,.0f}  {ex.mean():>+10,.0f}  "
              f"{np.percentile(ex, 25):>+9,.0f}  {np.percentile(ex, 75):>+9,.0f}  "
              f"{int(np.median(n_payouts)):>11}  {int(np.median(days_alive)):>15}")
        rows.append(dict(mnq=mnq, bust_rate=bust_pct/100, median_extracted=float(np.median(ex)),
                          mean_extracted=float(ex.mean()),
                          p25=float(np.percentile(ex, 25)),
                          p75=float(np.percentile(ex, 75)),
                          mean_payouts=float(n_payouts.mean()),
                          median_days_alive=int(np.median(days_alive))))

    # Bust reasons
    print("\n--- Bust reasons distribution (where applicable) ---")
    for mnq in [2, 3, 4]:
        sims = [sim_funded(daily, mnq, rng) for _ in range(2000)]
        reasons = [s["bust_reason"] for s in sims if s["busted"]]
        if reasons:
            from collections import Counter
            c = Counter(reasons)
            total = len(reasons)
            print(f"MNQ={mnq}: {len(reasons)} busts out of 2000 ({len(reasons)/20:.1f}%)")
            for reason, cnt in c.most_common():
                print(f"  {reason}: {cnt} ({cnt/total*100:.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tradeify_funded_sim.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR / 'tradeify_funded_sim.csv'}")


if __name__ == "__main__":
    main()

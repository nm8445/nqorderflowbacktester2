"""
Rough Vol ONLY on Tradeify Select 50k 2-step challenge.

Setup:
  - 1 account, rough vol trades only (no OD, no B2)
  - Tradeify rules: $3k target, $1k daily loss, $2k trailing DD, 40% consistency, min 3 days
  - Sweep MNQ size: maps to per-trade dollar risk
  - 4 independent challenge attempts (each samples fresh days)
  - Compute P(at least K of 4 pass)

Rough Vol stats (locked v3):
  - WR ~56% (not exactly 50%)
  - 1:1 RR (SL=2x ATR, TP=2x ATR)
  - Median win/loss ~67/65 NQ pts per contract = ~$135/$130 at MNQ=1
  - Per-trade dollar risk = 2 * ATR * $2/pt * MNQ_contracts

At MNQ=1: typical risk ~$130
At MNQ=2: ~$260
At MNQ=3: ~$390
At MNQ=4: ~$520 (closest to user's $500 target)
At MNQ=5: ~$650
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from math import comb

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
COMBINED_MAE = RESULTS_DIR / "combined_3way_trades_with_mae.csv"

ACCT_SIZE       = 50_000
PROFIT_TARGET   = 3_000
DAILY_LOSS_LIM  = 1_000
TRAILING_DD     = 2_000
LOCK_BALANCE    = 52_100
LOCK_FLOOR      = 50_100
CONSISTENCY_PCT = 0.40
MIN_DAYS        = 3
HORIZON_DAYS    = 60   # phase 1 max horizon

# Futures slippage for rough vol
RV_SLIP = 8.0  # NQ basis $ per trade

N_SIMS = 5000


def load_rv_daily():
    df = pd.read_csv(COMBINED_MAE)
    df = df[df["strat"] == "RV"].copy()
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    df["pnl_after_slip"] = df["pnl_$"] - RV_SLIP
    df["mae_after_slip"] = df["mae_$"] - RV_SLIP
    df = df.sort_values(["date", "entry_ts"])
    out = []
    for d, g in df.groupby("date", sort=True):
        out.append(list(zip(g["pnl_after_slip"].astype(float),
                            g["mae_after_slip"].astype(float))))
    return out


def sim_phase(daily, mnq, rng, target=PROFIT_TARGET, dd=TRAILING_DD, daily_limit=DAILY_LOSS_LIM):
    """Simulate one challenge phase. Returns (passed, days_used, fail_reason)."""
    scale = mnq * 0.1
    balance = ACCT_SIZE
    peak = ACCT_SIZE
    locked = False
    prev_eod = ACCT_SIZE
    daily_profits = []
    trading_days = 0
    target_bal = ACCT_SIZE + target

    for day in range(HORIZON_DAYS):
        idx = rng.integers(0, len(daily))
        if not daily[idx]:
            continue
        day_pnl = 0.0
        busted = False
        for pnl_nq, mae_nq in daily[idx]:
            mae_d = mae_nq * scale; pnl_d = pnl_nq * scale
            eq_dip = balance + day_pnl + mae_d
            if eq_dip > peak: peak = eq_dip
            cur_floor = LOCK_FLOOR if locked else (peak - dd)
            if eq_dip <= cur_floor:
                return (False, day + 1, "dd")
            if (prev_eod - eq_dip) >= daily_limit:
                return (False, day + 1, "daily")
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur > peak: peak = cur
            if not locked and cur >= LOCK_BALANCE: locked = True
            cur_floor = LOCK_FLOOR if locked else (peak - dd)
            if cur <= cur_floor:
                return (False, day + 1, "dd")
            if (prev_eod - cur) >= daily_limit:
                return (False, day + 1, "daily")
        balance += day_pnl
        prev_eod = balance
        trading_days += 1
        daily_profits.append(day_pnl)

        # Check pass condition
        total_profit = balance - ACCT_SIZE
        if total_profit >= target and trading_days >= MIN_DAYS:
            pos_profits = [p for p in daily_profits if p > 0]
            if pos_profits:
                max_day = max(pos_profits)
                if max_day <= CONSISTENCY_PCT * total_profit:
                    return (True, day + 1, "passed")
    return (False, HORIZON_DAYS, "horizon")


def estimate_per_trade_risk(rv_daily, mnq):
    """Rough estimate of $ per-trade SL (loss) at this MNQ size."""
    # Use median loss from rough vol data
    all_losses = []
    for day in rv_daily:
        for pnl, mae in day:
            if pnl < 0:
                all_losses.append(pnl)
    if all_losses:
        med = np.median(np.abs(all_losses))
        return med * mnq * 0.1
    return 0


def main():
    print("Loading rough vol trades...")
    daily = load_rv_daily()
    print(f"  {len(daily)} historical RV trading days\n")

    rng = np.random.default_rng(2026)

    mnq_sizes = [1, 2, 3, 4, 5, 6, 8]
    print(f"{'MNQ':>4}  {'~$/trade':>9}  {'P(pass)':>8}  {'med_days':>9}  "
          f"{'P(>=1of4)':>10}  {'P(>=2of4)':>10}  {'P(>=3of4)':>10}  {'P(all4)':>8}")
    rows = []
    for mnq in mnq_sizes:
        per_trade_risk = estimate_per_trade_risk(daily, mnq)
        results = [sim_phase(daily, mnq, rng) for _ in range(N_SIMS)]
        passes = [d for ok, d, _ in results if ok]
        p_pass = len(passes) / N_SIMS
        # Binomial: P(>=k of 4 pass)
        p4 = [0.0] * 5
        for k in range(5):
            p4[k] = comb(4, k) * p_pass**k * (1-p_pass)**(4-k)
        p_at_least_1 = sum(p4[1:])
        p_at_least_2 = sum(p4[2:])
        p_at_least_3 = sum(p4[3:])
        p_all_4 = p4[4]
        med = int(np.median(passes)) if passes else -1
        print(f"{mnq:>4}  ${per_trade_risk:>7,.0f}  {p_pass*100:>7.1f}%  {med:>9}  "
              f"{p_at_least_1*100:>9.1f}%  {p_at_least_2*100:>9.1f}%  "
              f"{p_at_least_3*100:>9.1f}%  {p_all_4*100:>7.1f}%")
        rows.append(dict(mnq=mnq, per_trade_risk=per_trade_risk,
                          p_single_pass=p_pass, median_days=med,
                          p_1_of_4=p_at_least_1, p_2_of_4=p_at_least_2,
                          p_3_of_4=p_at_least_3, p_all_4=p_all_4))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS_DIR / "tradeify_rv_only_challenge.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR / 'tradeify_rv_only_challenge.csv'}")


if __name__ == "__main__":
    main()

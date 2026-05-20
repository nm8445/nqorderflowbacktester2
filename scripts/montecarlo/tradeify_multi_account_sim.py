"""
Tradeify Select 50k multi-account challenge sim with trade alternation.

Account specs (Tradeify Select 50k — CME MNQ FUTURES, not CFD):
  - Cost: $99 ONE-TIME (current sale) or $159 regular ONE-TIME
  - Profit target: $3,000 (6%)
  - Daily loss limit: $1,000 (2%)
  - Max drawdown: $2,000 trailing (locks at $52,100 to $50,100 floor)
  - 40% consistency rule (challenge phase only): no single day > 40% of total profits
  - Minimum 3 trading days
  - No consistency rule in funded mode
  - NO fees in funded phase

FUTURES SLIPPAGE (much lower than MT5 CFD):
  CME MNQ futures: tick = $0.50, typical round-trip 1-2 ticks
  - RV (RTH, 38% limit fills): blended ~$8/trade NQ basis
  - B2 (RTH, 39% limit fills): blended ~$8/trade NQ basis
  - OD (overnight bar-close fills): ~$10/trade NQ basis × qty
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
COMBINED_MAE = RESULTS_DIR / "combined_3way_trades_with_mae.csv"

# Account rules
ACCT_SIZE       = 50_000
PROFIT_TARGET   = 3_000
DAILY_LOSS_LIM  = 1_000
TRAILING_DD     = 2_000
LOCK_BALANCE    = 52_100  # locks DD at $50,100 once EOD >= this
LOCK_FLOOR      = 50_100
CONSISTENCY_PCT = 0.40
MIN_DAYS        = 3

COST_SALE       = 99    # current sale, ONE-TIME
COST_REG        = 159   # regular price, ONE-TIME
HORIZON_DAYS    = 90    # evaluation horizon
N_SIMS          = 2000

# Slippage per trade (NQ basis $) — FUTURES values
SLIPPAGE = {"RV": 8.0, "B2": 8.0, "OD": 10.0}

# Grid
ACCOUNTS_GRID = [1, 2, 3, 5, 7, 10]
MNQ_GRID = [1, 2, 3, 4, 5]


def load_daily():
    df = pd.read_csv(COMBINED_MAE)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    df["slip_$"] = df["strat"].map(SLIPPAGE).fillna(0.0)
    od_mask = df["strat"] == "OD"
    df.loc[od_mask, "slip_$"] = df.loc[od_mask, "slip_$"] * 1.25
    df["pnl_after_slip"] = df["pnl_$"] - df["slip_$"]
    df = df.sort_values(["date", "entry_ts"])
    out = []
    for d, g in df.groupby("date", sort=True):
        out.append(list(g["pnl_after_slip"].astype(float)))
    return out


def sim_one_account(daily_seq, mnq):
    """Run one account through the daily sequence. Returns dict of outcomes."""
    scale = mnq * 0.1
    balance = ACCT_SIZE
    peak = ACCT_SIZE
    locked = False
    floor = ACCT_SIZE - TRAILING_DD
    daily_profits = []  # all closed-day net PnLs (for consistency check)
    trading_days = 0
    busted = False
    bust_day = -1
    bust_reason = ""
    passed = False
    pass_day = -1

    for day_i, day_pnls in enumerate(daily_seq):
        if not day_pnls:
            continue
        day_pnl = sum(p * scale for p in day_pnls)
        trading_days += 1
        # Intraday DD check (approximate with end-of-day balance and assume linear movement)
        # Approximation: if day_pnl < -$1000 → daily loss bust
        # If running balance dips below floor at end of day → DD bust
        new_balance = balance + day_pnl

        # Daily loss check
        if day_pnl <= -DAILY_LOSS_LIM:
            busted = True; bust_day = day_i; bust_reason = "daily_loss"
            break
        # DD check
        if locked:
            cur_floor = LOCK_FLOOR
        else:
            cur_floor = peak - TRAILING_DD
        if new_balance <= cur_floor:
            busted = True; bust_day = day_i; bust_reason = "dd"
            break

        balance = new_balance
        daily_profits.append(day_pnl)
        if balance > peak: peak = balance
        if not locked and balance >= LOCK_BALANCE:
            locked = True

        # Check pass condition: target hit + min 3 days + consistency
        total_profit = balance - ACCT_SIZE
        if total_profit >= PROFIT_TARGET and trading_days >= MIN_DAYS:
            # Consistency check: no single day > 40% of total profit
            pos_days = [p for p in daily_profits if p > 0]
            if pos_days:
                max_day = max(pos_days)
                if max_day <= CONSISTENCY_PCT * total_profit:
                    passed = True
                    pass_day = day_i
                    break
                # else: keep trading to dilute
    return dict(passed=passed, pass_day=pass_day,
                busted=busted, bust_day=bust_day, bust_reason=bust_reason,
                trading_days=trading_days,
                final_balance=balance,
                final_profit=balance - ACCT_SIZE)


def build_alternated_sequences(daily_full, num_accounts, rng):
    """Distribute each daily trade list to N accounts round-robin."""
    sequences = [[] for _ in range(num_accounts)]
    for day_trades in daily_full:
        per_acct = [[] for _ in range(num_accounts)]
        # Shuffle assignment order each day
        for i, t in enumerate(day_trades):
            acct = (i + rng.integers(0, num_accounts)) % num_accounts
            per_acct[acct].append(t)
        for k in range(num_accounts):
            sequences[k].append(per_acct[k])
    return sequences


def run_combo(daily, mnq, n_accounts, n_sims, rng):
    """For given (mnq, n_accounts), run n_sims of HORIZON_DAYS each."""
    results = []
    n_data = len(daily)
    for sim in range(n_sims):
        # Build a sampled daily sequence of HORIZON_DAYS
        sampled = [daily[rng.integers(0, n_data)] for _ in range(HORIZON_DAYS)]
        # Alternate trades across N accounts
        sequences = build_alternated_sequences(sampled, n_accounts, rng)
        # Simulate each account
        acct_outcomes = [sim_one_account(seq, mnq) for seq in sequences]
        n_passed = sum(1 for o in acct_outcomes if o["passed"])
        n_busted = sum(1 for o in acct_outcomes if o["busted"])
        # Days to first pass (min pass_day across accounts that passed, or HORIZON if none)
        pass_days = [o["pass_day"] for o in acct_outcomes if o["passed"]]
        first_pass = min(pass_days) if pass_days else HORIZON_DAYS
        results.append(dict(n_passed=n_passed, n_busted=n_busted,
                             first_pass_day=first_pass))
    return results


def main():
    print("Loading...")
    daily = load_daily()
    print(f"  {len(daily)} historical days; horizon: {HORIZON_DAYS}d; sims: {N_SIMS}\n")
    rng = np.random.default_rng(2026)

    rows = []
    print(f"{'N':>3} {'MNQ':>3}  {'P(>=1)':>6}  {'P(>=2)':>6}  {'P(>=3)':>6}  "
          f"{'mean#':>6}  {'mean_first':>10}  {'cost_deal':>9}  {'cost_reg':>8}")
    for n_acc in ACCOUNTS_GRID:
        for mnq in MNQ_GRID:
            results = run_combo(daily, mnq, n_acc, N_SIMS, rng)
            n_passed_arr = np.array([r["n_passed"] for r in results])
            first_pass_arr = np.array([r["first_pass_day"] for r in results])
            p1 = float((n_passed_arr >= 1).mean())
            p2 = float((n_passed_arr >= 2).mean())
            p3 = float((n_passed_arr >= 3).mean())
            mean_pass = float(n_passed_arr.mean())
            mean_first = float(first_pass_arr.mean())
            # ONE-TIME cost: $99 sale or $159 regular per account
            cost_deal = COST_SALE * n_acc
            cost_reg = COST_REG * n_acc
            print(f"{n_acc:>3} {mnq:>3}  {p1*100:>5.1f}%  {p2*100:>5.1f}%  {p3*100:>5.1f}%  "
                   f"{mean_pass:>6.2f}  {mean_first:>10.1f}  {cost_deal:>9,.0f}  {cost_reg:>8,.0f}")
            rows.append(dict(n_accounts=n_acc, mnq=mnq,
                              p_pass_1=p1, p_pass_2=p2, p_pass_3=p3,
                              mean_n_passed=mean_pass,
                              mean_first_pass_day=mean_first,
                              cost_horizon_deal=cost_deal,
                              cost_horizon_reg=cost_reg))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tradeify_multi_account_sim.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR / 'tradeify_multi_account_sim.csv'}")

    # Top combos by expected #passes per dollar spent
    df["pass_per_$_deal"] = df["mean_n_passed"] / df["cost_horizon_deal"]
    df["pass_per_$_reg"] = df["mean_n_passed"] / df["cost_horizon_reg"]
    print("\n=== Top 10 combos by passes-per-dollar (deal price) ===")
    top = df.sort_values("pass_per_$_deal", ascending=False).head(10)
    print(top[["n_accounts", "mnq", "p_pass_1", "p_pass_2", "mean_n_passed",
                "mean_first_pass_day", "cost_horizon_deal", "pass_per_$_deal"]].to_string(index=False))
    print("\n=== Top combos with >=70% P(2+) AND <= 5 accounts (user's preference) ===")
    user_filter = df[(df["p_pass_2"] >= 0.70) & (df["n_accounts"] <= 5)].sort_values("p_pass_2", ascending=False)
    if len(user_filter) > 0:
        print(user_filter.head(10).to_string(index=False))
    else:
        print("None meet criteria; relaxing to P(2+)>=0.50:")
        user_filter = df[(df["p_pass_2"] >= 0.50) & (df["n_accounts"] <= 5)].sort_values("p_pass_2", ascending=False)
        print(user_filter.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

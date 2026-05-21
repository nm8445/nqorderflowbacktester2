"""FundedNext 200K vs 300K (merged) CFD account — Monte Carlo on 4-strategy combined.

Strategy stack: RV + B2 + OD + Fabio ORB (combined_4way_trades.csv)

Account rules:
  200K standalone:
    - Start $200K
    - PER-TRADE MAE > $3K  -> account blown
    - DAILY realized loss > $10K  -> account blown
    - TOTAL drawdown from peak > $20K -> account blown
    - No consistency rule. Withdraw any profit above $200K at EOD daily.
    - Trader keeps 90%; FundedNext takes 10%.

  300K merged (200K + 100K):
    - Start $300K
    - Per-trade $6K | Daily $15K | Total $30K (user-stated)
    - Same payout / split rules

  Refresh-on-bust: $99 eval fee, fresh account each bust.

Sweep: MNQ 1 .. 10 (sizing multiplier — backtest is 1 NQ basis = 10 MNQ).
Horizon: 252 trading days. 5000 sims per cell.

Outputs:
  - For each (account, mnq): bust%, mean payouts/yr, mean trader $/yr (after split & fees),
    days to first payout (median, p25, p75), bust-reason breakdown.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repo root (file is in scripts/montecarlo/)
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
B2_TRADES  = ROOT / "scripts" / "overnight range strat" / "tradelogs" / "robust_configs" / "locked_v2_k08_lock045_mart_fc_filtered_trades.csv"
FAB_TRADES = Path("D:/trading_pythonbacktest_data/fabio orb/trades_final_modeA.csv")

NQ_PT = 20.0
TRADER_SPLIT = 0.90
EVAL_FEE = 99.0
HORIZON_DAYS = 252
N_SIMS = 5000

ACCOUNTS = {
    "200K": dict(start=200_000, per_trade=3_000,  daily=10_000, total=20_000),
    "300K": dict(start=300_000, per_trade=6_000,  daily=15_000, total=30_000),
}


def load_daily_pnl_and_max_trade_mae() -> tuple[np.ndarray, np.ndarray]:
    """For each trading day at 1 NQ basis:
      - total realized PnL ($)
      - WORST single-trade MAE that day ($) — used for per-trade-cap check
    """
    df = pd.read_csv(TRADES_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  utc=True).dt.tz_convert("America/New_York")
    df["date"]     = df["exit_ts"].dt.date

    # Per-strategy MAE — actual where available
    b2 = pd.read_csv(B2_TRADES)
    b2["entry_ts"] = pd.to_datetime(b2["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    b2["mae_$_1nq"] = (-b2["scaled_pnl"]).clip(lower=0) * NQ_PT   # lower bound

    fb = pd.read_csv(FAB_TRADES)
    fb = fb[fb["mode"] == "A"].copy()
    fb["entry_ts"] = pd.to_datetime(fb["entry_time"], utc=True).dt.tz_convert("America/New_York")
    fb["mae_$_1nq"] = fb["mae_pts"] * NQ_PT

    b2_key = b2.set_index("entry_ts")["mae_$_1nq"].to_dict()
    fb_key = fb.set_index("entry_ts")["mae_$_1nq"].to_dict()

    def _mae(r):
        if r["strat"] == "B2": return b2_key.get(r["entry_ts"], max(0, -r["pnl_$"]))
        if r["strat"] == "FB": return fb_key.get(r["entry_ts"], max(0, -r["pnl_$"]))
        return max(0, -r["pnl_$"])
    df["mae_$_1nq"] = df.apply(_mae, axis=1)

    daily = df.groupby("date").agg(pnl=("pnl_$", "sum"), worst_mae=("mae_$_1nq", "max"))
    return daily["pnl"].values.astype(float), daily["worst_mae"].values.astype(float)


def simulate_one_account(pnl_pool, mae_pool, mnq, rules, rng):
    mult = mnq / 10.0
    start = rules["start"]
    balance = start
    peak    = start
    busted  = False
    bust_reason = None
    days_lived = 0
    n_payouts = 0
    total_withdrawn_gross = 0.0
    days_to_first_payout = None

    n_pool = len(pnl_pool)
    for d in range(HORIZON_DAYS):
        idx = rng.integers(0, n_pool)
        day_pnl = pnl_pool[idx] * mult
        day_mae = mae_pool[idx] * mult
        days_lived = d + 1

        # 1. Per-trade MAE kill
        if day_mae > rules["per_trade"]:
            busted = True; bust_reason = "per_trade_mae"
            break
        # 2. Daily loss kill
        if day_pnl < -rules["daily"]:
            busted = True; bust_reason = "daily_cap"
            break

        balance += day_pnl
        peak = max(peak, balance)

        # 3. Total drawdown kill
        if peak - balance > rules["total"]:
            busted = True; bust_reason = "total_dd"
            break

        # 4. Payout (no consistency, any profit at EOD)
        if balance > start:
            profit = balance - start
            n_payouts += 1
            total_withdrawn_gross += profit
            balance = start   # reset to fresh start
            if days_to_first_payout is None:
                days_to_first_payout = d + 1

    return {
        "busted": busted,
        "bust_reason": bust_reason,
        "days_lived": days_lived,
        "n_payouts": n_payouts,
        "withdrawn_gross": total_withdrawn_gross,
        "trader_received": total_withdrawn_gross * TRADER_SPLIT,
        "days_to_first_payout": days_to_first_payout,
    }


def run_grid(label, rules, pnl_pool, mae_pool):
    print(f"\n========================================================================")
    print(f"  {label} account  start=${rules['start']:,}  per-trade=${rules['per_trade']:,}  "
          f"daily=${rules['daily']:,}  total=${rules['total']:,}")
    print(f"  Profit split 90% trader, $99 eval fee per blown account")
    print(f"========================================================================")
    rows = []
    for mnq in range(1, 11):
        rng = np.random.default_rng(seed=2026 + mnq)
        sims = [simulate_one_account(pnl_pool, mae_pool, mnq, rules, rng) for _ in range(N_SIMS)]

        bust_rate    = np.mean([s["busted"]    for s in sims])
        any_payout   = np.mean([s["n_payouts"] >= 1 for s in sims])
        mean_lifespan = np.mean([s["days_lived"] for s in sims])

        # Annual rate: account-lifespans averaged → accounts/yr → multiply per-account stats
        accts_per_year = HORIZON_DAYS / mean_lifespan if mean_lifespan > 0 else 0
        per_acct_payouts = np.mean([s["n_payouts"] for s in sims])
        per_acct_trader  = np.mean([s["trader_received"] for s in sims])

        annual_payouts = per_acct_payouts * accts_per_year
        annual_trader  = per_acct_trader  * accts_per_year
        annual_fees    = (accts_per_year * EVAL_FEE) if bust_rate > 0.01 else 0
        annual_net     = annual_trader - annual_fees

        # First-payout days
        d_pay = [s["days_to_first_payout"] for s in sims if s["days_to_first_payout"] is not None]
        d_med = int(np.median(d_pay)) if d_pay else None
        d_p25 = int(np.percentile(d_pay, 25)) if d_pay else None
        d_p75 = int(np.percentile(d_pay, 75)) if d_pay else None

        # Bust reason breakdown
        reasons = [s["bust_reason"] for s in sims if s["busted"]]
        n_pt = sum(1 for r in reasons if r == "per_trade_mae")
        n_dd = sum(1 for r in reasons if r == "total_dd")
        n_dl = sum(1 for r in reasons if r == "daily_cap")

        rows.append({
            "mnq": mnq,
            "bust%": bust_rate * 100,
            "by_PT%": n_pt / N_SIMS * 100,
            "by_DD%": n_dd / N_SIMS * 100,
            "by_DAY%": n_dl / N_SIMS * 100,
            "payout%": any_payout * 100,
            "d_pay_med": d_med, "d_pay_p25": d_p25, "d_pay_p75": d_p75,
            "avg_life_d": mean_lifespan,
            "accts/yr": accts_per_year,
            "payouts/yr": annual_payouts,
            "trader_$/yr": annual_trader,
            "fees_$/yr": annual_fees,
            "NET_$/yr": annual_net,
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
    return df


def main():
    print("Loading combined 4-strategy trades + per-trade MAE...")
    pnl_pool, mae_pool = load_daily_pnl_and_max_trade_mae()
    print(f"  {len(pnl_pool)} trading days   "
          f"daily PnL: mean=${pnl_pool.mean():.0f} std=${pnl_pool.std():.0f}   "
          f"worst trade MAE (1 NQ): max=${mae_pool.max():.0f}  median=${np.median(mae_pool):.0f}")

    all_dfs = []
    for label, rules in ACCOUNTS.items():
        df = run_grid(label, rules, pnl_pool, mae_pool)
        df["account"] = label
        all_dfs.append(df)

    summary = pd.concat(all_dfs).reset_index(drop=True)
    out = ROOT / "live" / "combined deployment plan" / "fundednext_200k_300k_4strat_mc.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

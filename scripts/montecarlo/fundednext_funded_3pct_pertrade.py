"""FundedNext FUNDED MC with 3% per-trade peak-MAE cap.

Updates vs `fundednext_200k_300k_4strat_mc.py`:
  - Per-trade cap = 3% of start (matches FN's real rule)
    OLD: $3K on 200K (1.5%), $6K on 300K (2%)
    NEW: $3K on 100K, $6K on 200K, $9K on 300K
  - Adds 100K account
  - 5% daily cap and 10% static DD (matches FN funded rules)

Rules per account size:
  - Per-trade peak MAE > 3% start -> BUST
  - Daily realized loss > 5% start -> BUST
  - Total DD from peak > 10% start -> BUST
  - Any profit at EOD -> withdraw, reset balance to start
  - 90% trader split, $99 eval fee per blown account
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
B2_TRADES  = ROOT / "scripts" / "overnight range strat" / "tradelogs" / "robust_configs" / "locked_v2_k08_lock045_mart_fc_filtered_trades.csv"
FAB_TRADES = Path("D:/trading_pythonbacktest_data/fabio orb/trades_final_modeA.csv")

NQ_PT = 20.0
TRADER_SPLIT = 0.90
EVAL_FEE = 99.0
HORIZON_DAYS = 252
N_SIMS = 5000

PER_TRADE_PCT = 0.03   # 3% on funded
DAILY_PCT     = 0.05   # 5%
TOTAL_PCT     = 0.10   # 10% static

ACCOUNTS = {
    "100K": dict(start=100_000),
    "200K": dict(start=200_000),
    "300K": dict(start=300_000),
}


def load_daily_pnl_and_mae() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(TRADES_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  utc=True).dt.tz_convert("America/New_York")
    df["date"]     = df["exit_ts"].dt.date

    b2 = pd.read_csv(B2_TRADES)
    b2["entry_ts"] = pd.to_datetime(b2["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    b2["mae_$_1nq"] = (-b2["scaled_pnl"]).clip(lower=0) * NQ_PT

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


def simulate_one_account(pnl_pool, mae_pool, mnq, start, rng):
    mult = mnq / 10.0
    per_trade_cap = start * PER_TRADE_PCT
    daily_cap     = start * DAILY_PCT
    total_cap     = start * TOTAL_PCT

    balance = start
    peak = start
    busted = False
    bust_reason = None
    days_lived = 0
    n_payouts = 0
    total_withdrawn = 0.0
    days_to_first_payout = None
    n_pool = len(pnl_pool)

    for d in range(HORIZON_DAYS):
        idx = rng.integers(0, n_pool)
        day_pnl = pnl_pool[idx] * mult
        day_mae = mae_pool[idx] * mult
        days_lived = d + 1

        if day_mae > per_trade_cap:
            busted = True; bust_reason = "per_trade_mae"; break
        if day_pnl < -daily_cap:
            busted = True; bust_reason = "daily_cap"; break

        balance += day_pnl
        peak = max(peak, balance)

        if peak - balance > total_cap:
            busted = True; bust_reason = "total_dd"; break

        if balance > start:
            profit = balance - start
            n_payouts += 1
            total_withdrawn += profit
            balance = start
            if days_to_first_payout is None:
                days_to_first_payout = d + 1

    return {
        "busted": busted,
        "bust_reason": bust_reason,
        "days_lived": days_lived,
        "n_payouts": n_payouts,
        "withdrawn_gross": total_withdrawn,
        "trader_received": total_withdrawn * TRADER_SPLIT,
        "days_to_first_payout": days_to_first_payout,
    }


def run_grid(label, rules, pnl_pool, mae_pool):
    start = rules["start"]
    print(f"\n{'='*92}")
    print(f"  {label} FUNDED  start=${start:,}")
    print(f"  Per-trade ${start*PER_TRADE_PCT:,.0f} (3%) | Daily ${start*DAILY_PCT:,.0f} (5%) | "
          f"Total ${start*TOTAL_PCT:,.0f} (10% static)")
    print(f"  90% trader split | $99 eval refresh per bust")
    print(f"{'='*92}")
    rows = []
    for mnq in range(1, 11):
        rng = np.random.default_rng(seed=2026 + mnq + len(label))
        sims = [simulate_one_account(pnl_pool, mae_pool, mnq, start, rng) for _ in range(N_SIMS)]

        bust_rate = np.mean([s["busted"] for s in sims])
        any_payout = np.mean([s["n_payouts"] >= 1 for s in sims])
        mean_lifespan = np.mean([s["days_lived"] for s in sims])

        accts_per_year = HORIZON_DAYS / mean_lifespan if mean_lifespan > 0 else 0
        per_acct_payouts = np.mean([s["n_payouts"] for s in sims])
        per_acct_trader  = np.mean([s["trader_received"] for s in sims])

        annual_payouts = per_acct_payouts * accts_per_year
        annual_trader  = per_acct_trader  * accts_per_year
        annual_fees    = (accts_per_year * EVAL_FEE) if bust_rate > 0.01 else 0
        annual_net     = annual_trader - annual_fees

        d_pay = [s["days_to_first_payout"] for s in sims if s["days_to_first_payout"] is not None]
        d_med = int(np.median(d_pay)) if d_pay else None
        d_p25 = int(np.percentile(d_pay, 25)) if d_pay else None
        d_p75 = int(np.percentile(d_pay, 75)) if d_pay else None

        reasons = [s["bust_reason"] for s in sims if s["busted"]]
        n_pt = sum(1 for r in reasons if r == "per_trade_mae")
        n_dd = sum(1 for r in reasons if r == "total_dd")
        n_dl = sum(1 for r in reasons if r == "daily_cap")

        rows.append({
            "mnq": mnq,
            "lots": round(mnq * 0.2, 2),
            "bust%":   bust_rate * 100,
            "by_PT%":  n_pt / N_SIMS * 100,
            "by_DD%":  n_dd / N_SIMS * 100,
            "by_DAY%": n_dl / N_SIMS * 100,
            "payout%": any_payout * 100,
            "d_pay_med": d_med, "d_pay_p25": d_p25, "d_pay_p75": d_p75,
            "avg_life_d": round(mean_lifespan, 1),
            "accts/yr": round(accts_per_year, 2),
            "payouts/yr": round(annual_payouts, 1),
            "trader_$/yr": round(annual_trader, 0),
            "NET_$/yr": round(annual_net, 0),
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
    return df


def main():
    print("Loading combined 4-strategy trades + per-trade MAE...")
    pnl_pool, mae_pool = load_daily_pnl_and_mae()
    print(f"  {len(pnl_pool)} trading days   "
          f"daily PnL: mean=${pnl_pool.mean():.0f} std=${pnl_pool.std():.0f}   "
          f"worst trade MAE (1 NQ): max=${mae_pool.max():.0f} p99=${np.percentile(mae_pool, 99):.0f}")

    all_dfs = []
    for label, rules in ACCOUNTS.items():
        df = run_grid(label, rules, pnl_pool, mae_pool)
        df["account"] = label
        all_dfs.append(df)

    summary = pd.concat(all_dfs).reset_index(drop=True)
    out = ROOT / "live" / "combined deployment plan" / "fundednext_funded_3pct_pertrade.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

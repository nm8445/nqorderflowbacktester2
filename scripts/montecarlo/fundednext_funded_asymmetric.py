"""FundedNext FUNDED MC with ASYMMETRIC per-strategy sizing.

User chose deployment: OD gets full 3% slot (runs solo overnight), B2+RV+Fabio
share the 3% cumulative cap at 1% each (can be concurrent in day session).

Sizing per account:
  100K (1% = $1K, 3% = $3K):
    OD: 2.5 MNQ | B2: 0.83 MNQ | RV: 2.5 MNQ | FB: 3.33 MNQ
  200K (1% = $2K, 3% = $6K):
    OD: 5.0 MNQ | B2: 1.67 MNQ | RV: 5.0 MNQ | FB: 6.67 MNQ
  300K (1% = $3K, 3% = $9K):
    OD: 7.5 MNQ | B2: 2.5 MNQ | RV: 7.5 MNQ | FB: 10.0 MNQ

For each account: simulate daily PnL pool built from per-strategy contributions.
Per-trade MAE check uses each trade's strat-specific multiplier.

Compared to flat-MNQ MC (`fundednext_funded_3pct_pertrade.csv`), this should
show LOWER EV because B2 is downsized (highest per-trade EV strategy) and OD
contribution is unchanged (already at 3% slot).
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

PER_TRADE_PCT = 0.03
DAILY_PCT     = 0.05
TOTAL_PCT     = 0.10

# Strategy MNQ size per account (asymmetric per-strat budget)
SIZING = {
    "100K": {"start": 100_000, "OD": 2.50, "B2": 0.83, "RV": 2.50, "FB": 3.33},
    "200K": {"start": 200_000, "OD": 5.00, "B2": 1.67, "RV": 5.00, "FB": 6.67},
    "300K": {"start": 300_000, "OD": 7.50, "B2": 2.50, "RV": 7.50, "FB": 10.00},
}


def load_trade_level_data() -> pd.DataFrame:
    """Return one row per trade with strat label, entry date, pnl_at_1nq, mae_at_1nq."""
    df = pd.read_csv(TRADES_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  utc=True).dt.tz_convert("America/New_York")
    # Realized day: use exit_ts (when PnL is locked in for daily-cap check)
    df["date"]     = df["exit_ts"].dt.date
    df["pnl_1nq"]  = df["pnl_$"].astype(float)

    # Per-strategy MAE — real where available, pnl fallback otherwise
    b2 = pd.read_csv(B2_TRADES)
    b2["entry_ts"] = pd.to_datetime(b2["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    b2["mae_1nq"]  = (-b2["scaled_pnl"]).clip(lower=0) * NQ_PT

    fb = pd.read_csv(FAB_TRADES)
    fb = fb[fb["mode"] == "A"].copy()
    fb["entry_ts"] = pd.to_datetime(fb["entry_time"], utc=True).dt.tz_convert("America/New_York")
    fb["mae_1nq"]  = fb["mae_pts"] * NQ_PT

    b2_key = b2.set_index("entry_ts")["mae_1nq"].to_dict()
    fb_key = fb.set_index("entry_ts")["mae_1nq"].to_dict()

    def _mae(r):
        if r["strat"] == "B2": return b2_key.get(r["entry_ts"], max(0, -r["pnl_$"]))
        if r["strat"] == "FB": return fb_key.get(r["entry_ts"], max(0, -r["pnl_$"]))
        return max(0, -r["pnl_$"])
    df["mae_1nq"] = df.apply(_mae, axis=1)
    return df[["date", "strat", "pnl_1nq", "mae_1nq"]]


def build_daily_pool(trades: pd.DataFrame, sizing: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply asymmetric per-strategy multipliers, then aggregate to daily.

    Returns (daily_pnl_$, daily_worst_mae_$) — both already account-sized.
    """
    t = trades.copy()
    # Multiplier = MNQ / 10 (since 1 NQ = 10 MNQ; pnl_1nq is at 1 NQ basis)
    t["mult"] = t["strat"].map({k: v / 10.0 for k, v in sizing.items() if k != "start"})
    t = t.dropna(subset=["mult"])  # drop any unknown strats
    t["pnl"] = t["pnl_1nq"] * t["mult"]
    t["mae"] = t["mae_1nq"] * t["mult"]
    daily = t.groupby("date").agg(pnl=("pnl", "sum"), worst_mae=("mae", "max"))
    return daily["pnl"].values.astype(float), daily["worst_mae"].values.astype(float)


def simulate_one_account(pnl_pool, mae_pool, start, rng):
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
        day_pnl = pnl_pool[idx]
        day_mae = mae_pool[idx]
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


def run_account(label, sizing, trades):
    start = sizing["start"]
    pnl_pool, mae_pool = build_daily_pool(trades, sizing)
    print(f"\n{'='*96}")
    print(f"  {label} FUNDED  start=${start:,}  asymmetric sizing")
    print(f"  OD={sizing['OD']:.2f}  B2={sizing['B2']:.2f}  RV={sizing['RV']:.2f}  FB={sizing['FB']:.2f}  MNQ")
    print(f"  Per-trade ${start*PER_TRADE_PCT:,.0f} (3%) | Daily ${start*DAILY_PCT:,.0f} (5%) | "
          f"Total ${start*TOTAL_PCT:,.0f} (10%)")
    print(f"  Daily pool: mean=${pnl_pool.mean():.0f} std=${pnl_pool.std():.0f}  "
          f"worst-trade MAE max=${mae_pool.max():.0f} p99=${np.percentile(mae_pool, 99):.0f}")
    print(f"{'='*96}")

    rng = np.random.default_rng(seed=2026 + len(label))
    sims = [simulate_one_account(pnl_pool, mae_pool, start, rng) for _ in range(N_SIMS)]

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

    row = {
        "account": label,
        "bust%":     round(bust_rate * 100, 2),
        "by_PT%":    round(n_pt / N_SIMS * 100, 2),
        "by_DD%":    round(n_dd / N_SIMS * 100, 2),
        "by_DAY%":   round(n_dl / N_SIMS * 100, 2),
        "payout%":   round(any_payout * 100, 2),
        "d_pay_med": d_med, "d_pay_p25": d_p25, "d_pay_p75": d_p75,
        "avg_life_d":  round(mean_lifespan, 1),
        "accts/yr":    round(accts_per_year, 2),
        "payouts/yr":  round(annual_payouts, 1),
        "trader_$/yr": round(annual_trader, 0),
        "NET_$/yr":    round(annual_net, 0),
    }
    print(pd.DataFrame([row]).to_string(index=False))
    return row


def main():
    print("Loading trade-level data with per-strategy MAE...")
    trades = load_trade_level_data()
    print(f"  {len(trades)} trades total | strats: {dict(trades['strat'].value_counts())}")

    rows = []
    for label, sizing in SIZING.items():
        row = run_account(label, sizing, trades)
        rows.append(row)

    summary = pd.DataFrame(rows)
    out = ROOT / "live" / "combined deployment plan" / "fundednext_funded_asymmetric.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"\n{'='*96}")
    print("COMPARISON vs flat-MNQ MC (free concurrency assumption)")
    print(f"{'='*96}")
    print(summary[["account", "bust%", "NET_$/yr", "payouts/yr", "d_pay_med"]].to_string(index=False))


if __name__ == "__main__":
    main()

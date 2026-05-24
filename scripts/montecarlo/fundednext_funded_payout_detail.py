"""FundedNext FUNDED MC — asymmetric sizing with detailed payout breakdown.

Same model as fundednext_funded_asymmetric.py but tracks payout sizes too:
  - Avg payout size ($)
  - Median payout size
  - Avg days between payouts
  - Monthly payout expectancy
  - Distribution of payout amounts (p25, p50, p75, p95)
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
# FN Stellar 2-Step split is "up to 95%". Default likely 80-90%, 95% with add-ons.
# Modeling 90% as baseline; multiply outputs by 95/90=1.056 for max split scenario.
TRADER_SPLIT = 0.90
EVAL_FEE = 99.0   # refresh fee — not verified on FN site, common ballpark
HORIZON_DAYS = 252
N_SIMS = 5000

PER_TRADE_PCT = 0.03
DAILY_PCT     = 0.05
TOTAL_PCT     = 0.10

SIZING = {
    "100K": {"start": 100_000, "OD": 2.50, "B2": 0.83, "RV": 2.50, "FB": 3.33},
    "200K": {"start": 200_000, "OD": 5.00, "B2": 1.67, "RV": 5.00, "FB": 6.67},
    # 300K only reached via FN scaling from 200K, not purchased directly
    "300K_scaled": {"start": 300_000, "OD": 7.50, "B2": 2.50, "RV": 7.50, "FB": 10.00},
}


def load_trade_level_data() -> pd.DataFrame:
    df = pd.read_csv(TRADES_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  utc=True).dt.tz_convert("America/New_York")
    df["date"]     = df["exit_ts"].dt.date
    df["pnl_1nq"]  = df["pnl_$"].astype(float)

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
    t = trades.copy()
    t["mult"] = t["strat"].map({k: v / 10.0 for k, v in sizing.items() if k in {"OD","B2","RV","FB"}})
    t = t.dropna(subset=["mult"])
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
    payout_amounts = []        # list of trader-side payout $ amounts
    payout_days = []           # day-index of each payout
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
            trader_take = profit * TRADER_SPLIT
            payout_amounts.append(trader_take)
            payout_days.append(d + 1)
            balance = start

    return {
        "busted": busted,
        "bust_reason": bust_reason,
        "days_lived": days_lived,
        "payout_amounts": payout_amounts,
        "payout_days": payout_days,
    }


def run_account(label, sizing, trades):
    start = sizing["start"]
    pnl_pool, mae_pool = build_daily_pool(trades, sizing)

    rng = np.random.default_rng(seed=2026 + len(label))
    sims = [simulate_one_account(pnl_pool, mae_pool, start, rng) for _ in range(N_SIMS)]

    bust_rate = np.mean([s["busted"] for s in sims])
    mean_lifespan = np.mean([s["days_lived"] for s in sims])

    # Flatten all payouts across all sims
    all_payouts = [p for s in sims for p in s["payout_amounts"]]
    n_payouts_per_sim = [len(s["payout_amounts"]) for s in sims]

    # Payout sizes
    payout_mean = np.mean(all_payouts) if all_payouts else 0
    payout_med  = np.median(all_payouts) if all_payouts else 0
    payout_p25  = np.percentile(all_payouts, 25) if all_payouts else 0
    payout_p75  = np.percentile(all_payouts, 75) if all_payouts else 0
    payout_p95  = np.percentile(all_payouts, 95) if all_payouts else 0

    # Days between payouts within each sim
    gaps = []
    for s in sims:
        days = s["payout_days"]
        if len(days) > 1:
            gaps.extend(np.diff(days).tolist())
    gap_mean = np.mean(gaps) if gaps else None
    gap_med  = np.median(gaps) if gaps else None

    accts_per_year = HORIZON_DAYS / mean_lifespan if mean_lifespan > 0 else 0
    payouts_per_year = np.mean(n_payouts_per_sim) * accts_per_year
    annual_trader = payout_mean * payouts_per_year
    annual_fees = (accts_per_year * EVAL_FEE) if bust_rate > 0.01 else 0
    annual_net = annual_trader - annual_fees
    monthly_trader = annual_trader / 12.0

    print(f"\n{'='*96}")
    print(f"  {label} FUNDED  start=${start:,}  asymmetric sizing")
    print(f"  OD={sizing['OD']:.2f}  B2={sizing['B2']:.2f}  RV={sizing['RV']:.2f}  FB={sizing['FB']:.2f}  MNQ")
    print(f"  Per-trade ${start*PER_TRADE_PCT:,.0f} (3%) | Daily ${start*DAILY_PCT:,.0f} (5%) | Total ${start*TOTAL_PCT:,.0f} (10%)")
    print(f"  Daily pool: mean=${pnl_pool.mean():.0f} std=${pnl_pool.std():.0f}  worst-trade MAE max=${mae_pool.max():.0f} ({mae_pool.max()/start*100:.2f}%)")
    print(f"{'='*96}")
    print(f"  Bust rate:               {bust_rate*100:6.2f}%   (all from total-DD; demote% = 0)")
    print(f"  Avg account lifespan:    {mean_lifespan:6.1f} days   ({accts_per_year:.2f} accts/yr)")
    print(f"  Payouts/yr:              {payouts_per_year:6.1f}")
    print(f"  ----- Payout sizes (trader take, after 90% split) -----")
    print(f"    avg:    ${payout_mean:>8,.0f}    median: ${payout_med:>8,.0f}")
    print(f"    p25:    ${payout_p25:>8,.0f}    p75:    ${payout_p75:>8,.0f}    p95: ${payout_p95:>8,.0f}")
    print(f"  ----- Days between payouts -----")
    print(f"    avg:    {gap_mean:.2f} days     median: {gap_med:.1f} days")
    print(f"  ----- Annual income -----")
    print(f"    Trader $/yr:  ${annual_trader:>10,.0f}")
    print(f"    Eval fees:    ${annual_fees:>10,.0f}")
    print(f"    NET $/yr:     ${annual_net:>10,.0f}")
    print(f"    Monthly avg:  ${monthly_trader:>10,.0f}")

    return {
        "account": label,
        "bust%": round(bust_rate * 100, 2),
        "lifespan_d": round(mean_lifespan, 1),
        "payouts/yr": round(payouts_per_year, 1),
        "avg_payout_$": round(payout_mean, 0),
        "med_payout_$": round(payout_med, 0),
        "p25_payout_$": round(payout_p25, 0),
        "p75_payout_$": round(payout_p75, 0),
        "p95_payout_$": round(payout_p95, 0),
        "avg_gap_days": round(gap_mean, 2) if gap_mean else None,
        "med_gap_days": round(gap_med, 1) if gap_med else None,
        "trader_$/yr": round(annual_trader, 0),
        "fees_$/yr": round(annual_fees, 0),
        "NET_$/yr": round(annual_net, 0),
        "NET_$/mo": round(annual_net / 12.0, 0),
    }


def main():
    print("Loading trade-level data with per-strategy MAE...")
    trades = load_trade_level_data()
    print(f"  {len(trades)} trades total | strats: {dict(trades['strat'].value_counts())}")

    rows = []
    for label, sizing in SIZING.items():
        row = run_account(label, sizing, trades)
        rows.append(row)

    summary = pd.DataFrame(rows)
    out = ROOT / "live" / "combined deployment plan" / "fundednext_funded_payout_detail.csv"
    summary.to_csv(out, index=False)
    print(f"\n{'='*96}")
    print("PAYOUT SUMMARY")
    print(f"{'='*96}")
    print(summary[["account", "avg_payout_$", "med_payout_$", "p95_payout_$",
                   "avg_gap_days", "payouts/yr", "NET_$/mo", "NET_$/yr"]].to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

"""FundedNext 2-Step Challenge MC with ASYMMETRIC sizing and 3% per-trade cap.

User confirmed (2026-05-23) that 3% per-trade rule applies to CHALLENGES too,
not just funded. Apply same asymmetric sizing as funded:
  - OD gets full 3% slot (runs solo overnight, no overlap)
  - B2 + RV + Fabio share 3% cap at 1% each (day session, can overlap)

Per-strategy MNQ per account (worst-trade peak MAE stays under 3% by construction):
  100K: OD=2.50  B2=0.83  RV=2.50  FB=3.33
  200K: OD=5.00  B2=1.67  RV=5.00  FB=6.67
  300K: OD=7.50  B2=2.50  RV=7.50  FB=10.00

Rules:
  - Phase 1: hit +8% target
  - Phase 2: hit +5% target (balance resets to start)
  - Daily realized loss > 5% start -> FAIL
  - Total drawdown (static) > 10% -> FAIL
  - Per-trade peak MAE > 3% start -> FAIL
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
HORIZON_DAYS = 252
N_SIMS = 5000

PHASE1_TARGET = 0.08
PHASE2_TARGET = 0.05
DAILY_CAP_PCT = 0.05
TOTAL_CAP_PCT = 0.10
PER_TRADE_CAP_PCT = 0.03    # 3% (same as funded now)

SIZING = {
    "100K": {"start": 100_000, "eval_fee": 550,   "OD": 2.50, "B2": 0.83, "RV": 2.50, "FB": 3.33},
    "200K": {"start": 200_000, "eval_fee": 1_100, "OD": 5.00, "B2": 1.67, "RV": 5.00, "FB": 6.67},
    # 300K is NOT purchasable as a standalone challenge — only reached via FN scaling from 200K.
    # Kept here for "expected income once scaled" reference; fee N/A.
    "300K_scaled": {"start": 300_000, "eval_fee": 0, "OD": 7.50, "B2": 2.50, "RV": 7.50, "FB": 10.00},
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


def simulate_phase(pnl_pool, mae_pool, start_bal, target_dollars,
                   daily_cap, total_cap, per_trade_cap, rng,
                   max_days=HORIZON_DAYS) -> tuple[str, int, float]:
    balance = start_bal
    floor = start_bal - total_cap
    n_pool = len(pnl_pool)
    for d in range(max_days):
        idx = rng.integers(0, n_pool)
        day_pnl = pnl_pool[idx]
        day_mae = mae_pool[idx]
        if day_mae > per_trade_cap:
            return "fail_pertrade", d + 1, balance - day_mae
        if day_pnl < -daily_cap:
            return "fail_daily", d + 1, balance + day_pnl
        balance += day_pnl
        if balance < floor:
            return "fail_total", d + 1, balance
        if balance >= start_bal + target_dollars:
            return "pass", d + 1, balance
    return "fail_timeout", max_days, balance


def simulate_full_challenge(pnl_pool, mae_pool, start_bal, rng) -> dict:
    daily_cap = start_bal * DAILY_CAP_PCT
    total_cap = start_bal * TOTAL_CAP_PCT
    per_trade_cap = start_bal * PER_TRADE_CAP_PCT
    p1_target = start_bal * PHASE1_TARGET
    p2_target = start_bal * PHASE2_TARGET

    out1, d1, _ = simulate_phase(pnl_pool, mae_pool, start_bal, p1_target,
                                  daily_cap, total_cap, per_trade_cap, rng)
    if out1 != "pass":
        return {"p1_pass": False, "p2_pass": False, "p1_days": d1, "p2_days": 0,
                "total_days": d1, "fail_reason": out1}

    out2, d2, _ = simulate_phase(pnl_pool, mae_pool, start_bal, p2_target,
                                  daily_cap, total_cap, per_trade_cap, rng)
    return {"p1_pass": True, "p2_pass": out2 == "pass", "p1_days": d1, "p2_days": d2,
            "total_days": d1 + d2, "fail_reason": out2 if out2 != "pass" else None}


def run_account(label, sizing, trades):
    start = sizing["start"]
    pnl_pool, mae_pool = build_daily_pool(trades, sizing)
    print(f"\n{'='*96}")
    print(f"  {label} 2-Step CHALLENGE  start=${start:,}  eval=${sizing['eval_fee']:,}")
    print(f"  Asymmetric: OD={sizing['OD']:.2f} B2={sizing['B2']:.2f} RV={sizing['RV']:.2f} FB={sizing['FB']:.2f}  MNQ")
    print(f"  P1 +8% (${start*PHASE1_TARGET:,.0f}) | P2 +5% (${start*PHASE2_TARGET:,.0f})")
    print(f"  Per-trade cap 3% (${start*PER_TRADE_CAP_PCT:,.0f}) | Daily 5% | Max 10%")
    print(f"  Daily pool: mean=${pnl_pool.mean():.0f} std=${pnl_pool.std():.0f}  "
          f"worst-trade MAE max=${mae_pool.max():.0f} ({mae_pool.max()/start*100:.2f}% of acct)")
    print(f"{'='*96}")

    rng = np.random.default_rng(seed=2026 + len(label))
    sims = [simulate_full_challenge(pnl_pool, mae_pool, start, rng) for _ in range(N_SIMS)]

    p1_rate = np.mean([s["p1_pass"] for s in sims])
    overall = np.mean([s["p2_pass"] for s in sims])
    p1_days_pass = [s["p1_days"] for s in sims if s["p1_pass"]]
    p2_days_pass = [s["p2_days"] for s in sims if s["p2_pass"]]
    total_pass   = [s["total_days"] for s in sims if s["p2_pass"]]

    fails = [s["fail_reason"] for s in sims if not s["p2_pass"] and s["fail_reason"]]
    n_pt = sum(1 for r in fails if r == "fail_pertrade")
    n_day = sum(1 for r in fails if r == "fail_daily")
    n_dd = sum(1 for r in fails if r == "fail_total")
    n_to = sum(1 for r in fails if r == "fail_timeout")

    avg_attempts = 1.0 / overall if overall > 0 else float("inf")
    cost_per_funded = sizing["eval_fee"] * avg_attempts if avg_attempts != float("inf") else None

    row = {
        "account":     label,
        "p1_pass%":    round(p1_rate * 100, 2),
        "overall%":    round(overall * 100, 2),
        "demote%":     round(n_pt / N_SIMS * 100, 2),
        "by_DAY%":     round(n_day / N_SIMS * 100, 2),
        "by_DD%":      round(n_dd / N_SIMS * 100, 2),
        "by_timeout%": round(n_to / N_SIMS * 100, 2),
        "p1_days_med": int(np.median(p1_days_pass)) if p1_days_pass else None,
        "p1_days_p25": int(np.percentile(p1_days_pass, 25)) if p1_days_pass else None,
        "p1_days_p75": int(np.percentile(p1_days_pass, 75)) if p1_days_pass else None,
        "p2_days_med": int(np.median(p2_days_pass)) if p2_days_pass else None,
        "total_med":   int(np.median(total_pass)) if total_pass else None,
        "total_p25":   int(np.percentile(total_pass, 25)) if total_pass else None,
        "total_p75":   int(np.percentile(total_pass, 75)) if total_pass else None,
        "$/funded":    round(cost_per_funded, 0) if cost_per_funded else None,
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
    out = ROOT / "live" / "combined deployment plan" / "fundednext_challenge_asymmetric.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"\n{'='*96}")
    print("CHALLENGE SUMMARY")
    print(f"{'='*96}")
    print(summary[["account", "overall%", "total_med", "total_p25", "total_p75", "$/funded", "demote%"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()

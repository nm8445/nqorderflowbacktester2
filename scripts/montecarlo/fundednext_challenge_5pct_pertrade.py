"""FundedNext 2-Step Challenge MC with 5% per-trade peak-MAE cap.

Updates vs `fundednext_2step_challenge_mc.py`:
  - Adds 5% per-trade MAE cap (peak open drawdown, not realized loss)
  - Adds 300K challenge size in addition to 100K and 200K
  - Loads per-day worst-MAE pool same as funded MC

Rules:
  - Phase 1: hit +8% profit target
  - Phase 2: hit +5% profit target (balance resets to start)
  - Daily realized loss > 5% start -> FAIL
  - Total drawdown (static) > 10% -> FAIL
  - Per-trade peak MAE > 5% start -> FAIL  (NEW)
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
PER_TRADE_CAP_PCT = 0.05   # 5% on challenge

ACCOUNTS = {
    "100K": dict(start=100_000, eval_fee=500),
    "200K": dict(start=200_000, eval_fee=1_100),
    "300K": dict(start=300_000, eval_fee=1_650),  # approx, verify FN pricing
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


def simulate_phase(pnl_pool, mae_pool, mnq, start_bal, target_dollars,
                   daily_cap, total_cap, per_trade_cap, rng,
                   max_days=HORIZON_DAYS) -> tuple[str, int, float]:
    """Returns (outcome, days_used, end_balance).
    outcome in {"pass", "fail_daily", "fail_total", "fail_pertrade", "fail_timeout"}.
    """
    mult = mnq / 10.0
    balance = start_bal
    floor = start_bal - total_cap
    n_pool = len(pnl_pool)
    for d in range(max_days):
        idx = rng.integers(0, n_pool)
        day_pnl = pnl_pool[idx] * mult
        day_mae = mae_pool[idx] * mult
        # Per-trade kill (checked first — instant fail on the worst trade of the day)
        if day_mae > per_trade_cap:
            return "fail_pertrade", d + 1, balance - day_mae
        # Daily realized loss kill
        if day_pnl < -daily_cap:
            return "fail_daily", d + 1, balance + day_pnl
        balance += day_pnl
        # Static total DD kill
        if balance < floor:
            return "fail_total", d + 1, balance
        # Profit target
        if balance >= start_bal + target_dollars:
            return "pass", d + 1, balance
    return "fail_timeout", max_days, balance


def simulate_full_challenge(pnl_pool, mae_pool, mnq, start_bal, rng) -> dict:
    daily_cap = start_bal * DAILY_CAP_PCT
    total_cap = start_bal * TOTAL_CAP_PCT
    per_trade_cap = start_bal * PER_TRADE_CAP_PCT
    p1_target = start_bal * PHASE1_TARGET
    p2_target = start_bal * PHASE2_TARGET

    out1, d1, bal1 = simulate_phase(pnl_pool, mae_pool, mnq, start_bal, p1_target,
                                     daily_cap, total_cap, per_trade_cap, rng)
    if out1 != "pass":
        return {"p1_pass": False, "p2_pass": False, "p1_days": d1, "p2_days": 0,
                "total_days": d1, "fail_reason": out1}

    out2, d2, bal2 = simulate_phase(pnl_pool, mae_pool, mnq, start_bal, p2_target,
                                     daily_cap, total_cap, per_trade_cap, rng)
    return {"p1_pass": True, "p2_pass": out2 == "pass", "p1_days": d1, "p2_days": d2,
            "total_days": d1 + d2, "fail_reason": out2 if out2 != "pass" else None}


def run_grid(label, rules, pnl_pool, mae_pool):
    print(f"\n{'='*88}")
    print(f"  {label} 2-Step Challenge  start=${rules['start']:,}  eval=${rules['eval_fee']:,}")
    print(f"  P1 +8% | P2 +5% | Daily 5% | Max 10% | PER-TRADE 5% (NEW peak-MAE cap)")
    print(f"{'='*88}")
    rows = []
    for mnq in range(1, 11):
        rng = np.random.default_rng(seed=2026 + mnq + len(label))
        sims = [simulate_full_challenge(pnl_pool, mae_pool, mnq, rules["start"], rng)
                for _ in range(N_SIMS)]

        p1_rate = np.mean([s["p1_pass"] for s in sims])
        p2_rate = np.mean([s["p2_pass"] for s in sims])
        p1_days_pass = [s["p1_days"] for s in sims if s["p1_pass"]]
        p2_days_pass = [s["p2_days"] for s in sims if s["p2_pass"]]
        total_pass   = [s["total_days"] for s in sims if s["p2_pass"]]

        # Fail-reason breakdown
        fails = [s["fail_reason"] for s in sims if not s["p2_pass"] and s["fail_reason"]]
        n_pt   = sum(1 for r in fails if r == "fail_pertrade")
        n_day  = sum(1 for r in fails if r == "fail_daily")
        n_dd   = sum(1 for r in fails if r == "fail_total")
        n_to   = sum(1 for r in fails if r == "fail_timeout")

        avg_attempts = 1.0 / p2_rate if p2_rate > 0 else float("inf")
        cost_per_funded = rules["eval_fee"] * avg_attempts

        rows.append({
            "mnq": mnq,
            "lots": round(mnq * 0.2, 2),  # NDX100 equivalent
            "p1_pass%":  p1_rate * 100,
            "overall%":  p2_rate * 100,
            "by_PT%":    n_pt / N_SIMS * 100,
            "by_DAY%":   n_day / N_SIMS * 100,
            "by_DD%":    n_dd / N_SIMS * 100,
            "by_TO%":    n_to / N_SIMS * 100,
            "p1_med_d":  int(np.median(p1_days_pass)) if p1_days_pass else None,
            "p2_med_d":  int(np.median(p2_days_pass)) if p2_days_pass else None,
            "total_med": int(np.median(total_pass))   if total_pass   else None,
            "total_p25": int(np.percentile(total_pass, 25)) if total_pass else None,
            "total_p75": int(np.percentile(total_pass, 75)) if total_pass else None,
            "$/funded":  round(cost_per_funded, 0) if cost_per_funded != float("inf") else None,
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
    return df


def main():
    print("Loading combined 4-strategy daily PnL + per-trade MAE...")
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
    out = ROOT / "live" / "combined deployment plan" / "fundednext_challenge_5pct_pertrade.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

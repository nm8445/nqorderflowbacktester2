"""The 5%ers High Stakes 100K — Monte Carlo with 4-strat asymmetric sizing.

Rules (per user spec + 5%ers help center):
  - Phase 1: +8% target, must hit before bust; need 3 profitable days minimum
  - Phase 2: +5% target (balance resets to start)
  - Daily loss cap: 5% (UNREALIZED equity counts — peak intraday MAE check)
  - Max loss: 10% static from initial balance
  - No per-trade rule
  - Funded: 80% trader split, daily withdrawals (any positive day > start)
  - Refresh fee: $95 (refundable on first payout — treated as gross cost here)

Sizing: asymmetric 4-strat, sweep scale factor.
  Base ratios (100K) — same shape as FN asymmetric:
    OD = 2.50 MNQ, B2 = 0.83 MNQ, RV = 2.50 MNQ, FB = 3.33 MNQ
  Scale factor multiplies all 4 proportionally.

Daily MAE check (conservative concurrent model):
  day_peak_unrealized = max(
      OD_MAE_$ at OD_mnq,                                  # overnight, solo
      B2_MAE_$ + RV_MAE_$ + FB_MAE_$ at their MNQs         # day session worst-case concurrent
  )
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
B2_TRADES  = ROOT / "scripts" / "overnight range strat" / "tradelogs" / "robust_configs" / "locked_v2_k08_lock045_mart_fc_filtered_trades.csv"
FAB_TRADES = Path("D:/trading_pythonbacktest_data/fabio orb/trades_final_modeA.csv")

NQ_PT = 20.0
ACCOUNT = 100_000
TRADER_SPLIT = 0.80
REFRESH_FEE = 95.0

PHASE1_TARGET_PCT = 0.08
PHASE2_TARGET_PCT = 0.05
MIN_PROFITABLE_DAYS_P1 = 3
DAILY_CAP_PCT = 0.05     # 5% — unrealized counts
MAX_DD_PCT = 0.10        # 10% static

HORIZON_DAYS = 252       # 1 year
N_SIMS = 5000

BASE_SIZING_100K = {"OD": 2.50, "B2": 0.83, "RV": 2.50, "FB": 3.33}  # MNQ
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

N_WORKERS = 6

_DAILY = None   # worker global


# ============================== data prep ==============================

def load_per_strategy_per_day() -> dict:
    """Returns dict: date -> {strat: (pnl_$_1mnq, mae_$_1mnq)}"""
    df = pd.read_csv(TRADES_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  utc=True).dt.tz_convert("America/New_York")
    df["date"]     = df["exit_ts"].dt.date

    # B2 + Fabio real MAEs
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

    # Convert to per-MNQ basis (1 MNQ = 0.1 NQ)
    df["pnl_$_1mnq"] = df["pnl_$"] * 0.1
    df["mae_$_1mnq"] = df["mae_$_1nq"] * 0.1

    # Aggregate per (date, strat)
    agg = df.groupby(["date", "strat"]).agg(
        pnl=("pnl_$_1mnq", "sum"),
        mae=("mae_$_1mnq", "max"),
    ).reset_index()

    # Build per-day dict
    daily = {}
    for d, g in agg.groupby("date"):
        per_strat = {s: (0.0, 0.0) for s in ["OD", "B2", "RV", "FB"]}
        for _, r in g.iterrows():
            per_strat[r["strat"]] = (float(r["pnl"]), float(r["mae"]))
        daily[d] = per_strat
    return daily


def daily_arrays(daily: dict) -> dict:
    """Pack daily dict into numpy arrays for fast sampling."""
    dates = sorted(daily.keys())
    n = len(dates)
    out = {
        "OD_pnl": np.zeros(n), "OD_mae": np.zeros(n),
        "B2_pnl": np.zeros(n), "B2_mae": np.zeros(n),
        "RV_pnl": np.zeros(n), "RV_mae": np.zeros(n),
        "FB_pnl": np.zeros(n), "FB_mae": np.zeros(n),
    }
    for i, d in enumerate(dates):
        per_strat = daily[d]
        for s in ["OD", "B2", "RV", "FB"]:
            out[f"{s}_pnl"][i] = per_strat[s][0]
            out[f"{s}_mae"][i] = per_strat[s][1]
    return out


# ============================== sim core ==============================

def _init_worker(daily_arr):
    global _DAILY
    _DAILY = daily_arr


def _day_stats(idx: int, mnq: dict) -> tuple[float, float, bool]:
    """Return (day_total_pnl, day_peak_unrealized_loss_$, has_profit_day)."""
    od_pnl = _DAILY["OD_pnl"][idx] * mnq["OD"]
    od_mae = _DAILY["OD_mae"][idx] * mnq["OD"]
    b2_pnl = _DAILY["B2_pnl"][idx] * mnq["B2"]
    b2_mae = _DAILY["B2_mae"][idx] * mnq["B2"]
    rv_pnl = _DAILY["RV_pnl"][idx] * mnq["RV"]
    rv_mae = _DAILY["RV_mae"][idx] * mnq["RV"]
    fb_pnl = _DAILY["FB_pnl"][idx] * mnq["FB"]
    fb_mae = _DAILY["FB_mae"][idx] * mnq["FB"]

    day_total = od_pnl + b2_pnl + rv_pnl + fb_pnl

    # Conservative: peak unrealized = max(OD alone, sum of day strats)
    day_session_concurrent_mae = b2_mae + rv_mae + fb_mae
    day_peak_unrealized = max(od_mae, day_session_concurrent_mae)

    return day_total, day_peak_unrealized, day_total > 0


def _run_challenge_sim(rng: np.random.Generator, mnq: dict) -> dict:
    """Simulate one full 2-step challenge attempt. Return outcome dict."""
    n_pool = len(_DAILY["OD_pnl"])
    daily_cap = ACCOUNT * DAILY_CAP_PCT
    max_dd_cap = ACCOUNT * MAX_DD_PCT
    floor = ACCOUNT - max_dd_cap

    # === Phase 1 ===
    bal = ACCOUNT
    profitable_days = 0
    days_p1 = 0
    p1_passed = False
    p1_failed_reason = None
    for d in range(HORIZON_DAYS):
        idx = rng.integers(0, n_pool)
        day_total, day_peak_unr, is_profit = _day_stats(idx, mnq)
        days_p1 = d + 1

        # Daily cap (unrealized)
        if day_peak_unr > daily_cap:
            p1_failed_reason = "daily_unrealized"; break
        bal += day_total
        # Max DD
        if bal < floor:
            p1_failed_reason = "max_dd"; break
        if is_profit:
            profitable_days += 1
        # Target check (only after min profitable days)
        if bal >= ACCOUNT * (1 + PHASE1_TARGET_PCT) and profitable_days >= MIN_PROFITABLE_DAYS_P1:
            p1_passed = True; break

    if not p1_passed:
        return {"p1_pass": False, "p2_pass": False, "p1_days": days_p1, "p2_days": 0,
                "total_days": days_p1, "fail_reason": p1_failed_reason or "p1_timeout"}

    # === Phase 2 ===
    bal = ACCOUNT
    days_p2 = 0
    p2_passed = False
    p2_failed_reason = None
    for d in range(HORIZON_DAYS):
        idx = rng.integers(0, n_pool)
        day_total, day_peak_unr, _ = _day_stats(idx, mnq)
        days_p2 = d + 1

        if day_peak_unr > daily_cap:
            p2_failed_reason = "daily_unrealized"; break
        bal += day_total
        if bal < floor:
            p2_failed_reason = "max_dd"; break
        if bal >= ACCOUNT * (1 + PHASE2_TARGET_PCT):
            p2_passed = True; break

    return {"p1_pass": True, "p2_pass": p2_passed, "p1_days": days_p1, "p2_days": days_p2,
            "total_days": days_p1 + days_p2,
            "fail_reason": p2_failed_reason if not p2_passed else None}


def _run_funded_sim(rng: np.random.Generator, mnq: dict) -> dict:
    """Simulate one funded account year. Daily withdrawal on any profit > start."""
    n_pool = len(_DAILY["OD_pnl"])
    daily_cap = ACCOUNT * DAILY_CAP_PCT
    max_dd_cap = ACCOUNT * MAX_DD_PCT
    floor = ACCOUNT - max_dd_cap

    bal = ACCOUNT
    peak = ACCOUNT
    busted = False
    bust_reason = None
    days_lived = 0
    n_payouts = 0
    total_withdrawn_trader = 0.0
    days_to_first_payout = None

    for d in range(HORIZON_DAYS):
        idx = rng.integers(0, n_pool)
        day_total, day_peak_unr, _ = _day_stats(idx, mnq)
        days_lived = d + 1

        if day_peak_unr > daily_cap:
            busted = True; bust_reason = "daily_unrealized"; break
        bal += day_total
        peak = max(peak, bal)
        # Max DD: 5%ers uses static initial-balance-based 10%, NOT trailing
        if bal < floor:
            busted = True; bust_reason = "max_dd"; break

        if bal > ACCOUNT:
            profit = bal - ACCOUNT
            n_payouts += 1
            total_withdrawn_trader += profit * TRADER_SPLIT
            bal = ACCOUNT
            if days_to_first_payout is None:
                days_to_first_payout = d + 1

    return {"busted": busted, "bust_reason": bust_reason, "days_lived": days_lived,
            "n_payouts": n_payouts, "trader_received": total_withdrawn_trader,
            "days_to_first_payout": days_to_first_payout}


def _worker_run(args):
    scale, run_mode = args
    mnq = {s: round(BASE_SIZING_100K[s] * scale, 3) for s in ["OD", "B2", "RV", "FB"]}
    seed = 2026 + int(scale * 1000) + (777 if run_mode == "funded" else 0)
    rng = np.random.default_rng(seed)
    if run_mode == "challenge":
        sims = [_run_challenge_sim(rng, mnq) for _ in range(N_SIMS)]
        p1_pass = np.mean([s["p1_pass"] for s in sims])
        p2_pass = np.mean([s["p2_pass"] for s in sims])
        total_pass = [s["total_days"] for s in sims if s["p2_pass"]]
        fails = [s["fail_reason"] for s in sims if s["fail_reason"]]
        n_daily = sum(1 for r in fails if r == "daily_unrealized")
        n_dd = sum(1 for r in fails if r == "max_dd")
        n_to = sum(1 for r in fails if r and "timeout" in r)
        return {
            "mode": "challenge", "scale": scale,
            "OD_mnq": mnq["OD"], "B2_mnq": mnq["B2"], "RV_mnq": mnq["RV"], "FB_mnq": mnq["FB"],
            "p1_pass%": round(p1_pass * 100, 2),
            "overall%": round(p2_pass * 100, 2),
            "by_daily%": round(n_daily / N_SIMS * 100, 2),
            "by_dd%":    round(n_dd / N_SIMS * 100, 2),
            "by_timeout%": round(n_to / N_SIMS * 100, 2),
            "median_days": int(np.median(total_pass)) if total_pass else None,
            "p25_days":    int(np.percentile(total_pass, 25)) if total_pass else None,
            "p75_days":    int(np.percentile(total_pass, 75)) if total_pass else None,
        }
    else:  # funded
        sims = [_run_funded_sim(rng, mnq) for _ in range(N_SIMS)]
        bust_rate = np.mean([s["busted"] for s in sims])
        mean_life = np.mean([s["days_lived"] for s in sims])
        accts_per_yr = HORIZON_DAYS / mean_life if mean_life > 0 else 0
        per_acct_payouts = np.mean([s["n_payouts"] for s in sims])
        per_acct_trader  = np.mean([s["trader_received"] for s in sims])
        annual_payouts = per_acct_payouts * accts_per_yr
        annual_trader  = per_acct_trader * accts_per_yr
        annual_fees = accts_per_yr * REFRESH_FEE if bust_rate > 0.01 else 0
        annual_net = annual_trader - annual_fees
        all_payouts = [p for s in sims for p in [s["trader_received"] / max(1, s["n_payouts"])] if s["n_payouts"] > 0]
        fails = [s["bust_reason"] for s in sims if s["busted"] and s["bust_reason"]]
        n_daily = sum(1 for r in fails if r == "daily_unrealized")
        n_dd = sum(1 for r in fails if r == "max_dd")
        return {
            "mode": "funded", "scale": scale,
            "OD_mnq": mnq["OD"], "B2_mnq": mnq["B2"], "RV_mnq": mnq["RV"], "FB_mnq": mnq["FB"],
            "bust%": round(bust_rate * 100, 2),
            "by_daily%": round(n_daily / N_SIMS * 100, 2),
            "by_dd%":    round(n_dd / N_SIMS * 100, 2),
            "avg_life_d": round(mean_life, 1),
            "payouts/yr": round(annual_payouts, 1),
            "trader_$/yr": round(annual_trader, 0),
            "NET_$/yr": round(annual_net, 0),
            "NET_$/mo": round(annual_net / 12.0, 0),
            "avg_payout_$": round(np.mean(all_payouts), 0) if all_payouts else 0,
        }


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading trade data...")
    daily = load_per_strategy_per_day()
    print(f"[{time.strftime('%H:%M:%S')}]   {len(daily)} trading days")
    daily_arr = daily_arrays(daily)
    print(f"[{time.strftime('%H:%M:%S')}]   Per-day arrays ready ({time.time()-t0:.1f}s)")

    configs = [(s, "challenge") for s in SCALES] + [(s, "funded") for s in SCALES]
    print(f"\n[{time.strftime('%H:%M:%S')}] Dispatching {len(configs)} configs to {N_WORKERS} workers...")
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(daily_arr,)) as ex:
        results = list(ex.map(_worker_run, configs))
    print(f"[{time.strftime('%H:%M:%S')}] Done in {time.time()-t1:.1f}s")

    challenge_df = pd.DataFrame([r for r in results if r["mode"] == "challenge"]).sort_values("scale")
    funded_df    = pd.DataFrame([r for r in results if r["mode"] == "funded"]).sort_values("scale")

    out_dir = ROOT / "live" / "combined deployment plan"
    challenge_df.to_csv(out_dir / "the5ers_100k_challenge.csv", index=False)
    funded_df.to_csv(out_dir / "the5ers_100k_funded.csv", index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print(f"\n=== 5%ers 100K HIGH STAKES — CHALLENGE (2-step) ===")
    print(f"   P1 +8% target ({MIN_PROFITABLE_DAYS_P1}+ profitable days needed) | P2 +5% target")
    print(f"   Daily 5% UNREALIZED cap (peak intraday) | Max 10% static DD")
    print(challenge_df.to_string(index=False))

    print(f"\n=== 5%ers 100K HIGH STAKES — FUNDED (80% split, daily payouts) ===")
    print(f"   Daily 5% UNREALIZED cap | Max 10% static DD | $95 refresh fee")
    print(funded_df.to_string(index=False))

    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    # Best challenge: pass rate >= 80% AND median days <= 65 (3 months)
    qual = challenge_df[(challenge_df["overall%"] >= 80) & (challenge_df["median_days"].notna()) &
                        (challenge_df["median_days"] <= 65)]
    if len(qual) > 0:
        # Best = highest pass within 3 mo (lowest bust)
        best_c = qual.sort_values(["overall%"], ascending=False).iloc[0]
        print(f"  CHALLENGE (best <3mo pass, >=80% rate):")
        print(f"    scale={best_c['scale']}  ->  OD={best_c['OD_mnq']} B2={best_c['B2_mnq']} "
              f"RV={best_c['RV_mnq']} FB={best_c['FB_mnq']} MNQ")
        print(f"    Pass {best_c['overall%']}%  median {best_c['median_days']} days "
              f"(p25 {best_c['p25_days']}, p75 {best_c['p75_days']})")
    else:
        print(f"  No challenge config hits >=80% pass within 65 days. Trade off speed vs safety.")

    # Best funded: highest NET $ with bust% < 50
    qual_f = funded_df[funded_df["bust%"] < 50].sort_values("NET_$/yr", ascending=False)
    if len(qual_f) > 0:
        best_f = qual_f.iloc[0]
        print(f"  FUNDED (best NET $/yr with bust < 50%):")
        print(f"    scale={best_f['scale']}  ->  OD={best_f['OD_mnq']} B2={best_f['B2_mnq']} "
              f"RV={best_f['RV_mnq']} FB={best_f['FB_mnq']} MNQ")
        print(f"    NET ${best_f['NET_$/yr']:,.0f}/yr (${best_f['NET_$/mo']:,.0f}/mo)  "
              f"bust {best_f['bust%']}%  payouts/yr {best_f['payouts/yr']}  "
              f"avg payout ${best_f['avg_payout_$']:,.0f}")


if __name__ == "__main__":
    main()

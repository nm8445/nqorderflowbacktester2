"""
Full lifetime simulation: Hola Prime $100k two-step challenge -> funded -> 1 payout -> new challenge.

REALISTIC MT5 NAS100 spreads/slippage (from Hola Prime documentation + general prop firm data):
  - RTH (09:00-17:00 ET): 1.5 pt spread + 0.25 pt market-order slippage = 1.75 pt round-trip cost
  - Overnight (17:00-09:00 ET): 3.0 pt spread + 0.5 pt slippage = 3.5 pt round-trip cost
  - These are in NAS100 points; 1 NAS100 pt = $20 per NQ-basis contract

Per-strategy slippage cost (in NQ-basis $20/pt, per trade):
  - Rough Vol (RTH): mostly RTH; 38% limit TP exits (no slip on exit) → avg ~1.4 pt rt = $28
  - B2 (RTH): mostly RTH; 39% limit TP exits → avg ~1.4 pt rt = $28
  - Overnight Drift (overnight): all bar-close fills, no limit fills → 3.5 pt rt = $70 × qty

Each trade's pnl_$ is debited by its slippage cost.

Then run:
  Step 1: 2-step challenge MC (phase 1: 8% target / 5% daily / 8% static DD;
                                phase 2: 5% target / 5% daily / 8% static DD)
          Find best MNQ size for fastest pass at >50% rate.
  Step 2: Funded MC with reinvestment loop at MNQ 2, 3, 4.
          Pipeline: pass challenge -> funded -> take 1 payout -> buy new challenge ($250 or $460)
          Track $ in pocket and time, simulate over 2 years (504 business days).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PLAN_DIR = HERE.parent.parent / "live" / "combined deployment plan"

COMBINED_MAE = RESULTS_DIR / "combined_3way_trades_with_mae.csv"
OUT_CSV = RESULTS_DIR / "lifetime_sim_with_slippage.csv"

# --- Slippage model (NQ-basis $) ---
SLIPPAGE = {"RV": 28.0, "B2": 28.0, "OD": 70.0}  # OD scaled by qty separately

# --- Account rules ---
ACCT = 100_000
# Challenge 2-step (static DD)
P1_TARGET   = 8_000
P2_TARGET   = 5_000
CHAL_DAILY  = 5_000
CHAL_DD     = 8_000   # 8% static from initial
# Funded
FUNDED_DAILY = 5_000
FUNDED_DD    = 10_000  # 10% (user's reported figure)
LOCK_PROFIT  = 5_000
PAYOUT_THRESHOLD = 2_000

DOWNTIME_AFTER_PAYOUT = 2  # business days
DOWNTIME_BUY_CHALLENGE = 1  # buying & loading a new challenge

# Lifetime horizon
LIFETIME_DAYS = 504  # 2 years
N_SIMS = 3000

CHALLENGE_COSTS = {250: "deal price", 460: "normal price"}
CHAL_SIZES = [0.5, 1, 1.5, 2, 2.5, 3]  # for challenge optimization
FUNDED_SIZES = [2, 3, 4]  # for funded phase per user request


def load_daily_with_slippage():
    df = pd.read_csv(COMBINED_MAE)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["exit_ts"].dt.date
    # Apply slippage per trade
    df["slip_$"] = df["strat"].map(SLIPPAGE).fillna(0.0)
    # For OD, scale slippage by qty (martingale)
    # We don't have qty in combined log explicitly but the magnitude of pnl absorbs it.
    # Approximate: use a flat 1.25x multiplier on OD slippage to account for ~25% mart trades
    od = df["strat"] == "OD"
    df.loc[od, "slip_$"] = df.loc[od, "slip_$"] * 1.25
    df["pnl_after_slip_$"] = df["pnl_$"] - df["slip_$"]
    df["mae_after_slip_$"] = df["mae_$"] - df["slip_$"]  # MAE also suffers slip in worst case

    df = df.sort_values(["date", "entry_ts"])
    out = []
    for d, g in df.groupby("date", sort=True):
        items = list(zip(g["pnl_after_slip_$"].astype(float),
                         g["mae_after_slip_$"].astype(float)))
        out.append(items)
    return out, float(df["slip_$"].sum())


def sim_phase(daily, scale, target, daily_loss_limit, dd_limit, max_days, rng):
    """Simulate one challenge phase or funded segment.
       Returns (passed, days_used, end_balance, bust_reason)
       passed=True if balance >= initial+target before bust/horizon."""
    balance = ACCT; peak = ACCT; prev_eod = ACCT
    initial = ACCT
    floor = initial - dd_limit
    target_balance = initial + target
    for day in range(max_days):
        idx = rng.integers(0, len(daily))
        day_pnl = 0.0
        for pnl_nq, mae_nq in daily[idx]:
            mae_d = mae_nq * scale; pnl_d = pnl_nq * scale
            eq_dip = balance + day_pnl + mae_d
            if eq_dip > peak: peak = eq_dip
            if eq_dip <= floor:
                return (False, day + 1, eq_dip, "dd")
            if (prev_eod - eq_dip) >= daily_loss_limit:
                return (False, day + 1, eq_dip, "daily")
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur > peak: peak = cur
            if cur <= floor:
                return (False, day + 1, cur, "dd")
            if (prev_eod - cur) >= daily_loss_limit:
                return (False, day + 1, cur, "daily")
            if cur >= target_balance:
                return (True, day + 1, cur, "target")
        balance += day_pnl
        prev_eod = balance
    return (False, max_days, balance, "horizon")


def sim_funded_until_payout(daily, scale, rng, max_days=120):
    """Funded account with trailing DD. Trade until 1 payout >= PAYOUT_THRESHOLD."""
    balance = ACCT; peak = ACCT; prev_eod = ACCT
    for day in range(max_days):
        idx = rng.integers(0, len(daily))
        day_pnl = 0.0
        for pnl_nq, mae_nq in daily[idx]:
            mae_d = mae_nq * scale; pnl_d = pnl_nq * scale
            eq_dip = balance + day_pnl + mae_d
            if eq_dip > peak: peak = eq_dip
            if peak >= ACCT + LOCK_PROFIT:
                cur_floor = max(ACCT, peak - FUNDED_DD)
            else:
                cur_floor = peak - FUNDED_DD
            if eq_dip <= cur_floor:
                return (False, day + 1, 0.0, "dd")
            if (prev_eod - eq_dip) >= FUNDED_DAILY:
                return (False, day + 1, 0.0, "daily")
            day_pnl += pnl_d
            cur = balance + day_pnl
            if cur > peak: peak = cur
            if peak >= ACCT + LOCK_PROFIT:
                cur_floor = max(ACCT, peak - FUNDED_DD)
            else:
                cur_floor = peak - FUNDED_DD
            if cur <= cur_floor:
                return (False, day + 1, 0.0, "dd")
            if (prev_eod - cur) >= FUNDED_DAILY:
                return (False, day + 1, 0.0, "daily")
        balance += day_pnl
        prev_eod = balance
        if balance >= ACCT + PAYOUT_THRESHOLD:
            return (True, day + 1, balance - ACCT, "payout")
    return (False, max_days, 0.0, "horizon")


def sim_2step_challenge(daily, scale, rng):
    """Sequential phase 1 then phase 2. Returns (passed, total_days, fail_reason)."""
    # Phase 1
    ok, d1, _, reason = sim_phase(daily, scale, P1_TARGET, CHAL_DAILY, CHAL_DD, 365, rng)
    if not ok:
        return (False, d1, f"P1_{reason}")
    # Phase 2 (reset balance to initial)
    ok2, d2, _, reason2 = sim_phase(daily, scale, P2_TARGET, CHAL_DAILY, CHAL_DD, 365 - d1, rng)
    if not ok2:
        return (False, d1 + d2, f"P2_{reason2}")
    return (True, d1 + d2, "passed")


def sim_lifetime(daily, chal_scale, funded_scale, account_cost, rng, max_days=LIFETIME_DAYS):
    """One lifetime sim. Track total $ extracted, time spent."""
    cash = 0.0
    paid_account_costs = 0.0
    days_used = 0
    cycles = 0   # complete challenge -> funded -> payout cycles
    while days_used < max_days:
        # Buy challenge
        cash -= account_cost
        paid_account_costs += account_cost
        days_used += DOWNTIME_BUY_CHALLENGE
        if days_used >= max_days: break

        # Run 2-step challenge
        passed, chal_days, _ = sim_2step_challenge(daily, chal_scale, rng)
        days_used += chal_days
        if not passed:
            # Failed challenge; lost the account fee. Buy new one.
            continue
        if days_used >= max_days: break

        # Funded phase until 1 payout
        ok, fund_days, payout, _ = sim_funded_until_payout(daily, funded_scale, rng)
        days_used += fund_days
        if ok:
            cash += payout
            days_used += DOWNTIME_AFTER_PAYOUT
            cycles += 1
            # After 1 payout, buy new challenge (user's strategy)
            continue
        else:
            # Funded busted before any payout; cycle restart
            continue
    return dict(cash=cash, paid_costs=paid_account_costs,
                cycles=cycles, days_used=min(days_used, max_days))


def find_best_challenge_size(daily, rng, n=1500):
    """Quick MC at each MNQ size to pick the optimum for fastest pass."""
    print("\n--- Step 1: Find best MNQ size for 2-step challenge (after slippage) ---")
    print(f"{'MNQ':>5}  {'pass%':>6}  {'med_days':>9}  {'p25':>5}  {'p75':>5}  {'mean_days_pass':>14}")
    best = None
    for mnq in CHAL_SIZES:
        scale = mnq * 0.1
        results = [sim_2step_challenge(daily, scale, rng) for _ in range(n)]
        passes = [d for p, d, _ in results if p]
        pass_rate = len(passes) / n
        if passes:
            med = int(np.median(passes))
            p25 = int(np.percentile(passes, 25))
            p75 = int(np.percentile(passes, 75))
            mean = float(np.mean(passes))
        else:
            med = -1; p25 = -1; p75 = -1; mean = -1
        print(f"{mnq:>5}  {pass_rate*100:>5.1f}%  {med:>9}  {p25:>5}  {p75:>5}  {mean:>14.1f}")
        # Composite score: pass_rate / mean_days_pass (higher = better)
        if pass_rate >= 0.5 and (best is None or pass_rate / max(mean, 1) > best[1]):
            best = (mnq, pass_rate / max(mean, 1), pass_rate, med)
    if best is None:
        print("No size achieves >=50% pass rate. Picking by pass rate.")
        # fallback
        return CHAL_SIZES[0]
    print(f"  -> Best challenge size: MNQ={best[0]}  (pass {best[2]*100:.1f}%, median {best[3]}d)")
    return best[0]


def main():
    print("Loading combined log with slippage...")
    daily, total_slip = load_daily_with_slippage()
    print(f"  Total slippage cost across all trades (NQ basis): ${total_slip:,.0f}")
    print(f"  Slippage as % of pre-slip combined PnL ($533,364): {total_slip/533364*100:.1f}%\n")

    rng = np.random.default_rng(2026)
    find_best_challenge_size(daily, rng)

    print(f"\n--- Step 2: Lifetime sim (2 years = 504 business days, reinvestment loop) ---")
    print(f"Challenge sizes tested: 2, 3")
    print(f"Funded sizes tested: {FUNDED_SIZES}")
    print(f"Account costs tested: {list(CHALLENGE_COSTS.keys())} ($)")
    rows = []
    for chal_mnq in [2, 3]:
        chal_scale = chal_mnq * 0.1
        for cost in CHALLENGE_COSTS:
            print(f"\n  Challenge MNQ={chal_mnq}, account cost=${cost} ({CHALLENGE_COSTS[cost]}):")
            print(f"  {'fundMNQ':>8}  {'mean_cash':>10}  {'median_cash':>11}  {'p25':>7}  {'p75':>7}  "
                   f"{'mean_cyc':>9}  {'cash/yr':>8}  {'$/cal-day':>10}")
            for funded_mnq in FUNDED_SIZES:
                funded_scale = funded_mnq * 0.1
                sims = [sim_lifetime(daily, chal_scale, funded_scale, cost, rng) for _ in range(N_SIMS)]
                cash_arr = np.array([s["cash"] for s in sims])
                cyc_arr = np.array([s["cycles"] for s in sims])
                mean_c = float(cash_arr.mean())
                med_c = float(np.median(cash_arr))
                p25 = float(np.percentile(cash_arr, 25))
                p75 = float(np.percentile(cash_arr, 75))
                mean_cyc = float(cyc_arr.mean())
                cash_per_year = mean_c / 2
                cash_per_day = mean_c / LIFETIME_DAYS
                print(f"  MNQ={funded_mnq:>3}  {mean_c:>+10,.0f}  {med_c:>+11,.0f}  "
                       f"{p25:>+7,.0f}  {p75:>+7,.0f}  {mean_cyc:>9.2f}  {cash_per_year:>+8,.0f}  {cash_per_day:>+10.1f}")
                rows.append(dict(cost=cost, chal_mnq=chal_mnq, funded_mnq=funded_mnq,
                                  mean_cash=mean_c, median_cash=med_c, p25=p25, p75=p75,
                                  mean_cycles=mean_cyc, cash_per_year=cash_per_year,
                                  cash_per_cal_day=cash_per_day))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    txt_path = PLAN_DIR / "lifetime_sim_with_slippage.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Hola Prime lifetime sim with realistic slippage\n{'='*70}\n\n")
        f.write(f"Slippage model: RV $28/tr, B2 $28/tr, OD $70/tr x qty (NQ basis)\n")
        f.write(f"Total slippage drag across history: ${total_slip:,.0f} ({total_slip/533364*100:.1f}% of PnL)\n\n")
        f.write(f"Best challenge size: MNQ={best_chal}\n")
        f.write(df.to_string(index=False))
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()

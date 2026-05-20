"""
CFD (FTMO / FundingPips 2-step) vs Futures (Lucid hedge guarantee $500) — head-to-head
=========================================================================================
Compares:
  A) Futures route: pay $500 once, guaranteed funded Lucid 50K, then trade.
  B) CFD route: pay challenge fee, attempt 2-step until pass, then funded 100K.

Both rate-limited by the same combined 3-way strategy daily PnL series.

Rules used:
  FTMO 100K 2-step:        P1 +10% target, P2 +5% target, 5% DLL, 10% static MaxLoss, 4 min days each.
  FundingPips 100K 2-step: P1 +8% target,  P2 +5% target, 5% DLL, 10% static MaxLoss, 3 min days each.
  Both fees reimbursed on first payout (we credit fee back on first funded payout).

  Funded $100K static $10K DD; bi-weekly 80% split, monthly 100% split assumed for FP.
  Futures funded modeled separately via Lucid 50K results (already computed) — we just import means.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades.csv"

N_SIMS = 5_000
HORIZON_CHAL_DAYS = 252      # cap challenge attempt at 1 year (no time limit in practice)
HORIZON_FUNDED_DAYS = 252

# Challenge configs
CHALLENGES = {
    "FTMO_2step":        {"p1_target": 10_000, "p2_target": 5_000, "min_days": 4, "fee": 540.0},
    "FundingPips_2step": {"p1_target":  8_000, "p2_target": 5_000, "min_days": 3, "fee": 549.0},
}

CHAL_START   = 100_000.0
CHAL_FLOOR   = 90_000.0      # 10% max loss
CHAL_DLL     = 5_000.0       # 5% daily

# Funded config (bi-weekly 80% by default)
FUND_START   = 100_000.0
FUND_FLOOR   = 90_000.0
FUND_DLL     = 5_000.0
FUND_CYCLE_TD = 10
FUND_SPLIT    = 0.80


def load_daily():
    """Returns (daily_pnl, daily_trade_counts) aligned arrays at 1 NQ per strat."""
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    grp = df.groupby("date")
    pnl = grp["pnl_$"].sum().sort_index()
    counts = grp.size().reindex(pnl.index)
    return pnl.values.astype(float), counts.values.astype(float)


# CFD round-trip cost per MNQ (spread + slippage on NAS100, blended OD/RTH)
# FTMO/FP indices are tighter than average: ~1pt RTH spread, ~2.5pt OD spread; +0.5-1pt slip.
# Blended: 51% RTH @ $4 + 49% OD @ $7 ≈ $5.5/MNQ — round to $6.
CFD_COST_PER_TRADE_PER_MNQ = 6.0
# Futures round-trip cost per MNQ (commissions + tick slippage)
FUT_COST_PER_TRADE_PER_MNQ = 2.0


def run_phase(daily_pnl, daily_trades, mnq, target, min_days, rng,
              cost_per_trade=CFD_COST_PER_TRADE_PER_MNQ,
              max_days=HORIZON_CHAL_DAYS):
    """Run a single challenge phase. Returns (passed, days_taken)."""
    mult = mnq / 10.0
    balance = CHAL_START
    n_pool = len(daily_pnl)
    for d in range(max_days):
        idx = rng.integers(0, n_pool)
        pnl = daily_pnl[idx] * mult - daily_trades[idx] * cost_per_trade * mnq
        if pnl <= -CHAL_DLL:
            return False, d + 1
        balance += pnl
        if balance < CHAL_FLOOR:
            return False, d + 1
        if balance - CHAL_START >= target and (d + 1) >= min_days:
            return True, d + 1
    return False, max_days


def run_challenge_2step(daily_pnl, daily_trades, mnq, cfg, rng,
                        cost_per_trade=CFD_COST_PER_TRADE_PER_MNQ):
    """Attempt phase 1 then phase 2. Returns (passed_both, total_days, p1_passed)."""
    p1_pass, p1_days = run_phase(daily_pnl, daily_trades, mnq, cfg["p1_target"], cfg["min_days"], rng, cost_per_trade)
    if not p1_pass:
        return False, p1_days, False
    p2_pass, p2_days = run_phase(daily_pnl, daily_trades, mnq, cfg["p2_target"], cfg["min_days"], rng, cost_per_trade)
    return p2_pass, p1_days + p2_days, True


def run_funded(daily_pnl, daily_trades, mnq, rng,
               split=FUND_SPLIT, cycle_td=FUND_CYCLE_TD, min_wd=50.0,
               cost_per_trade=CFD_COST_PER_TRADE_PER_MNQ):
    """Funded 100K bi-weekly. Returns dict."""
    mult = mnq / 10.0
    balance = FUND_START
    days_since_cycle = 0
    payouts = 0
    cash = 0.0
    days_to_1st = None
    busted = False
    bust_day = None
    n_pool = len(daily_pnl)
    for d in range(HORIZON_FUNDED_DAYS):
        idx = rng.integers(0, n_pool)
        pnl = daily_pnl[idx] * mult - daily_trades[idx] * cost_per_trade * mnq
        if pnl <= -FUND_DLL:
            busted = True; bust_day = d; break
        balance += pnl
        if balance < FUND_FLOOR:
            busted = True; bust_day = d; break
        days_since_cycle += 1
        if days_since_cycle >= cycle_td:
            profit = balance - FUND_START
            if profit >= min_wd:
                trader = profit * split
                balance -= profit
                payouts += 1
                cash += trader
                if days_to_1st is None:
                    days_to_1st = d
            days_since_cycle = 0
    return dict(busted=busted, bust_day=bust_day, payouts=payouts,
                cash=cash, days_to_1st=days_to_1st)


def simulate_lifetime_cfd(daily_pnl, daily_trades, mnq_chal, mnq_fund, cfg, rng):
    """One attempt: challenge until pass-or-give-up, then funded. We model 1 attempt only."""
    passed, chal_days, p1_passed = run_challenge_2step(daily_pnl, daily_trades, mnq_chal, cfg, rng)
    fee = cfg["fee"]
    if not passed:
        return dict(passed=False, chal_days=chal_days, p1_passed=p1_passed,
                    funded_days=0, payouts=0, cash=0.0, net=-fee, fee=fee,
                    days_to_1st_payout=None, total_days=chal_days)
    fund = run_funded(daily_pnl, daily_trades, mnq_fund, rng)
    # Fee reimbursed on first payout (typical practice for both firms)
    reimbursed = fee if fund["payouts"] >= 1 else 0.0
    net = fund["cash"] + reimbursed - fee
    total_days = chal_days + (HORIZON_FUNDED_DAYS if not fund["busted"] else fund["bust_day"] + 1)
    return dict(passed=True, chal_days=chal_days, p1_passed=p1_passed,
                funded_days=(HORIZON_FUNDED_DAYS if not fund["busted"] else fund["bust_day"] + 1),
                busted_funded=fund["busted"],
                payouts=fund["payouts"], cash=fund["cash"], net=net, fee=fee,
                days_to_1st_payout=fund["days_to_1st"],
                total_days=total_days)


def main():
    daily_pnl, daily_trades = load_daily()
    print(f"Loaded {len(daily_pnl)} historical days. mean=${daily_pnl.mean():.0f} std=${daily_pnl.std():.0f} "
          f"min=${daily_pnl.min():.0f} max=${daily_pnl.max():.0f}")
    print(f"Avg trades/day: {daily_trades.mean():.2f}")
    print(f"CFD cost: ${CFD_COST_PER_TRADE_PER_MNQ}/RT/MNQ (avg daily drag at 1 MNQ: "
          f"${daily_trades.mean() * CFD_COST_PER_TRADE_PER_MNQ:.2f})\n")

    # --- 1) Sweep challenge pass rate vs MNQ for each firm ---
    chal_rows = []
    for firm, cfg in CHALLENGES.items():
        for mnq in [1, 2, 3, 4, 5, 6, 7, 8]:
            rng = np.random.default_rng(seed=hash(firm) % 9973 + mnq)
            results = [run_challenge_2step(daily_pnl, daily_trades, mnq, cfg, rng) for _ in range(N_SIMS)]
            passes = [r[0] for r in results]
            days = [r[1] for r in results]
            p1_only = [r[2] for r in results]
            pass_rate = np.mean(passes)
            p1_rate = np.mean(p1_only)
            passed_days = [d for r, d in zip(results, days) if r[0]]
            chal_rows.append({
                "firm": firm,
                "mnq": mnq,
                "p1_pass": p1_rate,
                "both_pass": pass_rate,
                "median_days_to_pass": int(np.median(passed_days)) if passed_days else None,
                "p25_days": int(np.percentile(passed_days, 25)) if passed_days else None,
                "p75_days": int(np.percentile(passed_days, 75)) if passed_days else None,
                "median_days_all": int(np.median(days)),
            })

    chal_df = pd.DataFrame(chal_rows)
    pd.set_option("display.width", 220)
    print("=== 2-STEP CHALLENGE PASS-RATE SWEEP ($100K, 5% DLL, 10% MaxLoss) ===")
    for firm in CHALLENGES:
        sub = chal_df[chal_df.firm == firm].drop(columns=["firm"])
        print(f"\n{firm}:")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    # --- 2) Lifetime EV: pay fee, challenge (1 attempt), then funded ---
    # User wants ~70% pass. Find best (chal_mnq, fund_mnq) combos.
    # We'll evaluate: chal_mnq in {3,4,5,6}, fund_mnq in {2,3,4}.
    print("\n=== LIFETIME EV (1 attempt, pay fee, challenge -> funded; fee reimbursed on 1st funded payout) ===")
    life_rows = []
    for firm, cfg in CHALLENGES.items():
        for mnq_chal in [3, 4, 5, 6]:
            for mnq_fund in [2, 3, 4]:
                rng = np.random.default_rng(seed=hash(firm + str(mnq_chal) + str(mnq_fund)) % 99991)
                sims = [simulate_lifetime_cfd(daily_pnl, daily_trades, mnq_chal, mnq_fund, cfg, rng) for _ in range(N_SIMS)]
                passed = [s["passed"] for s in sims]
                pass_rate = np.mean(passed)
                # Conditional on pass:
                paid = [s for s in sims if s["passed"]]
                chal_days_pass = [s["chal_days"] for s in paid] if paid else [0]
                fund_payouts = [s["payouts"] for s in paid] if paid else [0]
                fund_cash = [s["cash"] for s in paid] if paid else [0.0]
                first_payout_days = [s["days_to_1st_payout"] for s in paid if s["days_to_1st_payout"] is not None]
                # Overall EV (across ALL sims, including failed challenges)
                net_all = [s["net"] for s in sims]
                cash_all = [s["cash"] for s in sims]

                life_rows.append({
                    "firm": firm,
                    "chal_mnq": mnq_chal,
                    "fund_mnq": mnq_fund,
                    "pass_rate": pass_rate,
                    "median_chal_days_if_pass": int(np.median(chal_days_pass)) if paid else None,
                    "median_days_to_1st_pmt": (int(np.median(first_payout_days)) + int(np.median(chal_days_pass))) if first_payout_days and paid else None,
                    "median_payouts_if_pass": int(np.median(fund_payouts)),
                    "median_cash_if_pass": np.median(fund_cash),
                    "mean_cash_if_pass": np.mean(fund_cash),
                    "ev_net_overall": np.mean(net_all),  # includes failed attempts
                    "ev_cash_overall": np.mean(cash_all),
                })

    life_df = pd.DataFrame(life_rows)
    for firm in CHALLENGES:
        sub = life_df[life_df.firm == firm].drop(columns=["firm"])
        print(f"\n--- {firm} ---")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))

    out = ROOT / "live" / "combined deployment plan" / "cfd_vs_futures_lifetime.csv"
    life_df.to_csv(out, index=False)
    chal_df.to_csv(out.with_name("cfd_challenge_sweep.csv"), index=False)
    print(f"\nSaved: {out}")
    print(f"Saved: {out.with_name('cfd_challenge_sweep.csv')}")


if __name__ == "__main__":
    main()

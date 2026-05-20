"""
FundingPips $50K 2-step challenge + funded lifetime EV
=========================================================
Rules (verified from FP docs / reviews):
  - Cost: $176
  - P1 target: 8% = $4,000 | P2 target: 5% = $2,500
  - Daily loss: 5% = $2,500 (closed + unrealized; we use realized daily PnL)
  - Max loss: 10% = $5,000 static, floor at $45,000
  - Min trading days: 3 per phase
  - On funded: same 10% static DD, 5% DLL carry over
  - Payout: 80% bi-weekly default (60/80/90/100% by cycle)
  - Fee refunded after 4 payouts

Compares to FP 100K and Lucid 50K (futures) head-to-head with same combined log.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades.csv"

CFD_COST = 6.0   # $/RT/MNQ blended

# FP 50K constants
START = 50_000.0
FLOOR = 45_000.0
DLL = 2_500.0
P1_TGT = 4_000.0
P2_TGT = 2_500.0
MIN_DAYS = 3
FEE = 176.0
CYCLE_TD = 10        # bi-weekly
SPLIT = 0.80
MIN_WD = 1_000.0     # 2% of 50K
HORIZON_CHAL = 252
HORIZON_FUND = 252
N_SIMS = 5_000


def load_daily():
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    grp = df.groupby("date")
    pnl = grp["pnl_$"].sum().sort_index()
    counts = grp.size().reindex(pnl.index)
    return pnl.values.astype(float), counts.values.astype(float)


def run_phase(daily_pnl, daily_tr, mnq, target, rng):
    mult = mnq / 10.0
    bal = START
    n_pool = len(daily_pnl)
    for d in range(HORIZON_CHAL):
        idx = rng.integers(0, n_pool)
        pnl = daily_pnl[idx] * mult - daily_tr[idx] * CFD_COST * mnq
        if pnl <= -DLL: return False, d + 1
        bal += pnl
        if bal < FLOOR: return False, d + 1
        if bal - START >= target and (d + 1) >= MIN_DAYS:
            return True, d + 1
    return False, HORIZON_CHAL


def run_2step(daily_pnl, daily_tr, mnq, rng):
    p1, d1 = run_phase(daily_pnl, daily_tr, mnq, P1_TGT, rng)
    if not p1: return False, d1
    p2, d2 = run_phase(daily_pnl, daily_tr, mnq, P2_TGT, rng)
    return p2, d1 + d2


def run_funded(daily_pnl, daily_tr, mnq, rng):
    mult = mnq / 10.0
    bal = START
    days_since = 0
    pmts = 0
    cash = 0.0
    days_to_1st = None
    busted = False
    bust_d = HORIZON_FUND
    n_pool = len(daily_pnl)
    for d in range(HORIZON_FUND):
        idx = rng.integers(0, n_pool)
        pnl = daily_pnl[idx] * mult - daily_tr[idx] * CFD_COST * mnq
        if pnl <= -DLL:
            busted = True; bust_d = d; break
        bal += pnl
        if bal < FLOOR:
            busted = True; bust_d = d; break
        days_since += 1
        if days_since >= CYCLE_TD:
            profit = bal - START
            if profit >= MIN_WD:
                bal -= profit
                pmts += 1
                cash += profit * SPLIT
                if days_to_1st is None:
                    days_to_1st = d
            days_since = 0
    return dict(busted=busted, bust_d=bust_d, payouts=pmts, cash=cash,
                days_to_1st=days_to_1st)


def lifetime(daily_pnl, daily_tr, mnq_chal, mnq_fund, rng):
    passed, cd = run_2step(daily_pnl, daily_tr, mnq_chal, rng)
    if not passed:
        return dict(passed=False, chal_days=cd, payouts=0, cash=0.0,
                    net=-FEE, days_to_1st=None)
    f = run_funded(daily_pnl, daily_tr, mnq_fund, rng)
    # fee refunded after 4 payouts (FP 50K policy)
    refund = FEE if f["payouts"] >= 4 else 0
    net = f["cash"] + refund - FEE
    return dict(passed=True, chal_days=cd, payouts=f["payouts"], cash=f["cash"],
                net=net, days_to_1st=f["days_to_1st"], busted_funded=f["busted"])


def main():
    daily_pnl, daily_tr = load_daily()
    print(f"Loaded {len(daily_pnl)} days. avg trades/day={daily_tr.mean():.2f}, "
          f"CFD cost ${CFD_COST}/RT/MNQ (drag $/day at 1 MNQ: ${daily_tr.mean()*CFD_COST:.2f})\n")

    # 2-step pass rate sweep
    print("=== FP 50K 2-STEP CHALLENGE SWEEP (P1 +$4K, P2 +$2.5K, $2.5K DLL, $5K MaxLoss) ===")
    chal_rows = []
    for mnq in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed=51000 + mnq)
        results = [run_2step(daily_pnl, daily_tr, mnq, rng) for _ in range(N_SIMS)]
        passes = [r[0] for r in results]
        days = [r[1] for r in results]
        passed_days = [d for r, d in zip(results, days) if r[0]]
        chal_rows.append({
            "mnq": mnq,
            "pass_rate": np.mean(passes),
            "median_days_if_pass": int(np.median(passed_days)) if passed_days else None,
            "p25_days": int(np.percentile(passed_days, 25)) if passed_days else None,
            "p75_days": int(np.percentile(passed_days, 75)) if passed_days else None,
        })
    cdf = pd.DataFrame(chal_rows)
    print(cdf.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    # Funded only (chal_mnq=fund_mnq)
    print("\n=== FP 50K FUNDED-ONLY (assume already passed) ===")
    fund_rows = []
    for mnq in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed=52000 + mnq)
        sims = [run_funded(daily_pnl, daily_tr, mnq, rng) for _ in range(N_SIMS)]
        bust = np.mean([s["busted"] for s in sims])
        pmts = [s["payouts"] for s in sims]
        cash = [s["cash"] for s in sims]
        t1 = [s["days_to_1st"] for s in sims if s["days_to_1st"] is not None]
        any_pmt = np.mean([s["payouts"] >= 1 for s in sims])
        avg_per = np.mean([s["cash"]/s["payouts"] for s in sims if s["payouts"] >= 1]) if any_pmt > 0 else 0
        fund_rows.append({
            "mnq": mnq, "bust_rate": bust, "any_pmt": any_pmt,
            "median_pmts": int(np.median(pmts)),
            "median_cash_$": np.median(cash),
            "mean_cash_$": np.mean(cash),
            "avg_per_pmt_$": avg_per,
            "median_d_to_1st": int(np.median(t1)) if t1 else None,
        })
    fdf = pd.DataFrame(fund_rows)
    print(fdf.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    # Lifetime EV
    print("\n=== FP 50K LIFETIME EV (challenge + funded; fee refunded after 4 payouts) ===")
    life_rows = []
    for mnq_c in [2, 3, 4]:
        for mnq_f in [1, 2, 3]:
            rng = np.random.default_rng(seed=53000 + mnq_c*10 + mnq_f)
            sims = [lifetime(daily_pnl, daily_tr, mnq_c, mnq_f, rng) for _ in range(N_SIMS)]
            paid = [s for s in sims if s["passed"]]
            pr = len(paid) / N_SIMS
            chal_days = [s["chal_days"] for s in paid] if paid else [0]
            pmts = [s["payouts"] for s in paid] if paid else [0]
            cash = [s["cash"] for s in paid] if paid else [0]
            d1 = [s["days_to_1st"] for s in paid if s["days_to_1st"] is not None]
            ev_net = np.mean([s["net"] for s in sims])
            ev_cash = np.mean([s["cash"] for s in sims])
            life_rows.append({
                "chal_mnq": mnq_c, "fund_mnq": mnq_f,
                "pass_rate": pr,
                "median_chal_days_if_pass": int(np.median(chal_days)) if paid else None,
                "median_d_to_1st_pmt": (int(np.median(d1)) + int(np.median(chal_days))) if d1 and paid else None,
                "median_pmts_if_pass": int(np.median(pmts)),
                "median_cash_if_pass": np.median(cash),
                "mean_cash_if_pass": np.mean(cash),
                "ev_net_all": ev_net,
                "ev_cash_all": ev_cash,
            })
    ldf = pd.DataFrame(life_rows)
    print(ldf.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))

    out = ROOT / "live" / "combined deployment plan" / "fp_50k_2step_lifetime.csv"
    ldf.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

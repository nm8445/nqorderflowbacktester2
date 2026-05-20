"""
MAE-aware prop firm Monte Carlo — accounts for floating intraday losses.
=========================================================================
Uses combined_trades_with_mae.csv which has both realized pnl_$ and
worst-floating mae_$ per trade (at 1 NQ basis).

Bust logic per trade:
  - Per-trade rule: |mae_$ * (mnq/10)| > trade_limit  -> instant bust
  - Intraday DLL (FTMO/FP): daily_realized + this_mae > -DLL -> bust
  - Intraday MaxLoss (all firms): balance + this_mae < floor -> bust
  - Otherwise apply realized pnl, continue

CFD costs ($/RT/MNQ) deducted from realized pnl. Float MAE is gross of cost.

Bootstrap: sample WHOLE DAYS (preserving trade ordering within day) so we
don't break the temporal sequence of realized+floating within a Prague day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"

N_SIMS = 5_000
HORIZON = 252

# Firm configs
CONFIGS = {
    # name:                  start    floor   dll    trade_lim  min_wd   cycle  split  cost  label
    "Lucid_50K_Flex": dict(start=50_000,  floor_init=48_000,  lock_after=53_000, lock_floor=50_000,
                            dll=None, trade_lim=None, min_wd=500,  cycle=None, split=0.90, cost=2.0,
                            payout_cap=2000, max_payouts=6,
                            label="Lucid Flex 50K ($2K EOD trail, no DLL, no trade rule)"),
    "FP_50K_2pct":     dict(start=50_000,  floor_init=45_000,  lock_after=None, lock_floor=None,
                            dll=2_500, trade_lim=1_000, min_wd=1_000, cycle=10, split=0.80, cost=6.0,
                            payout_cap=None, max_payouts=None,
                            label="FundingPips 50K ($5K static, $2.5K DLL, 2% trade=$1K)"),
    "FP_100K_2pct":    dict(start=100_000, floor_init=90_000,  lock_after=None, lock_floor=None,
                            dll=5_000, trade_lim=2_000, min_wd=2_000, cycle=10, split=0.80, cost=6.0,
                            payout_cap=None, max_payouts=None,
                            label="FundingPips 100K ($10K static, $5K DLL, 2% trade=$2K)"),
    "FTMO_100K_1pct":  dict(start=100_000, floor_init=90_000,  lock_after=None, lock_floor=None,
                            dll=5_000, trade_lim=1_000, min_wd=50,   cycle=10, split=0.80, cost=6.0,
                            payout_cap=None, max_payouts=None,
                            label="FTMO 100K ($10K static, $5K DLL, 1% trade=$1K)"),
}


def load_packs():
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"])
    packs = []
    for date, grp in df.groupby("date", sort=True):
        trades = [(r["strat"], r["pnl_$"], r["mae_$"], r["entry_ts"])
                  for _, r in grp.iterrows()]
        packs.append((date, trades))
    return packs


def simulate(packs, mnq, cfg, mode_use_mae, rng):
    """Run a single 252-day sim. Returns dict.

    mode_use_mae=True: intraday bust check (floating MAE).
    mode_use_mae=False: realized-PnL-only (legacy mode for comparison).
    """
    scale = mnq / 10.0
    cost = cfg["cost"]
    balance = cfg["start"]
    floor = cfg["floor_init"]
    locked = (cfg["lock_after"] is None)  # lucid locks when balance reaches lock_after; others have static floor
    hwm_eod = balance
    busted = False
    bust_reason = None
    payouts = 0
    cash = 0.0
    days_to_1st = None
    days_since_cycle = 0
    cycle_qual_days = 0
    cycle_profit = 0.0
    stagger_first = False    # for Lucid
    n_packs = len(packs)

    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        _, trades = packs[idx]
        daily_realized = 0.0

        for strat, pnl, mae, _ts in trades:
            pnl_scaled = pnl * scale - cost * mnq    # net realized
            mae_scaled = mae * scale - cost * mnq    # floating low w/ cost already paid

            # 1. Per-trade rule (uses MAE — worst floating during trade)
            if mode_use_mae and cfg["trade_lim"] is not None:
                if abs(mae_scaled) > cfg["trade_lim"]:
                    busted = True; bust_reason = "trade_rule"; break

            # 2. Intraday DLL: daily_realized + this_trade_floating > -DLL
            if cfg["dll"] is not None:
                check_val = daily_realized + (mae_scaled if mode_use_mae else pnl_scaled)
                if check_val <= -cfg["dll"]:
                    busted = True; bust_reason = "DLL"; break

            # 3. Intraday MaxLoss: balance + this_trade_floating < floor
            check_eq = balance + daily_realized + (mae_scaled if mode_use_mae else pnl_scaled)
            if check_eq < floor:
                busted = True; bust_reason = "MaxLoss"; break

            # Apply realized
            daily_realized += pnl_scaled

        if busted:
            break

        # End of day
        balance += daily_realized

        # Lucid trailing DD update at EOD
        if cfg["lock_after"] is not None and not locked:
            if balance > hwm_eod:
                hwm_eod = balance
            new_floor = max(cfg["floor_init"], hwm_eod - (cfg["start"] - cfg["floor_init"]))
            floor = new_floor
            if hwm_eod >= cfg["lock_after"]:
                locked = True
                floor = cfg["lock_floor"]

        days_since_cycle += 1
        cycle_profit += daily_realized
        if cfg["dll"] is None:  # Lucid uses qualifying day count
            if daily_realized >= 150:
                cycle_qual_days += 1

        # ---- PAYOUT LOGIC ----
        eligible = False
        gross_payout = 0.0

        if cfg["cycle"] is not None:
            # CFD style: bi-weekly cycle
            if days_since_cycle >= cfg["cycle"]:
                cycle_p = balance - cfg["start"]
                if cycle_p >= cfg["min_wd"]:
                    gross_payout = cycle_p
                    eligible = True
                days_since_cycle = 0
        else:
            # Lucid Flex style: 5 qualifying days + cycle profit + stagger logic
            if cfg["max_payouts"] and payouts < cfg["max_payouts"] and cycle_qual_days >= 5 and cycle_profit > 0:
                # stagger: $1500 first, $1000 subsequent
                if not stagger_first:
                    if cycle_profit >= 3000:
                        gross_payout = 1500
                        eligible = True
                        stagger_first = True
                else:
                    if cycle_profit >= 2000:
                        gross_payout = 1000
                        eligible = True

        if eligible:
            if cfg["payout_cap"]:
                gross_payout = min(gross_payout, cfg["payout_cap"])
            trader_cash = gross_payout * cfg["split"]
            balance -= gross_payout
            if cfg["lock_after"] is not None and not locked:
                hwm_eod = max(cfg["start"], hwm_eod - gross_payout)
                floor = max(cfg["floor_init"], hwm_eod - (cfg["start"] - cfg["floor_init"]))
            payouts += 1
            cash += trader_cash
            if days_to_1st is None:
                days_to_1st = d
            cycle_qual_days = 0
            cycle_profit = 0.0
            if cfg["max_payouts"] and payouts >= cfg["max_payouts"]:
                break

    return dict(busted=busted, bust_reason=bust_reason, payouts=payouts,
                cash=cash, days_to_1st=days_to_1st)


def main():
    packs = load_packs()
    print(f"Loaded {len(packs)} historical trading days, "
          f"{sum(len(p[1]) for p in packs)} trades.\n")

    for firm_key, cfg in CONFIGS.items():
        print(f"\n=== {cfg['label']} ===")
        rows = []
        for mnq in [1, 2, 3, 4, 5]:
            # Realized-only mode (for comparison with prior sims)
            rng_r = np.random.default_rng(seed=hash(firm_key) % 9973 + mnq + 100)
            sims_r = [simulate(packs, mnq, cfg, False, rng_r) for _ in range(N_SIMS)]

            # MAE-aware mode (correct, includes floating)
            rng_m = np.random.default_rng(seed=hash(firm_key) % 9973 + mnq + 200)
            sims_m = [simulate(packs, mnq, cfg, True, rng_m) for _ in range(N_SIMS)]

            br_r = np.mean([s["busted"] for s in sims_r])
            br_m = np.mean([s["busted"] for s in sims_m])
            reasons = {}
            for s in sims_m:
                if s["busted"]:
                    reasons[s["bust_reason"]] = reasons.get(s["bust_reason"], 0) + 1
            top_reason = max(reasons.items(), key=lambda x: x[1]) if reasons else ("-", 0)
            any_p = np.mean([s["payouts"] >= 1 for s in sims_m])
            pmts = [s["payouts"] for s in sims_m]
            cash = [s["cash"] for s in sims_m]
            t1 = [s["days_to_1st"] for s in sims_m if s["days_to_1st"] is not None]
            rows.append({
                "mnq": mnq,
                "bust_realized": br_r,
                "bust_MAE": br_m,
                "delta_pp": (br_m - br_r) * 100,
                "top_bust_reason": top_reason[0],
                "any_payout_MAE": any_p,
                "median_pmts_MAE": int(np.median(pmts)),
                "median_cash_$": np.median(cash),
                "mean_cash_$": np.mean(cash),
                "median_days_to_1st": int(np.median(t1)) if t1 else None,
            })
        df = pd.DataFrame(rows)
        pd.set_option("display.width", 220)
        print(df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    out = ROOT / "live" / "combined deployment plan" / "mae_aware_propfirm_sim.csv"
    print(f"\n(MAE-aware results saved separately per firm above)")


if __name__ == "__main__":
    main()

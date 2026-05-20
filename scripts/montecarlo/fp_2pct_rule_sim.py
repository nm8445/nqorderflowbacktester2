"""
FundingPips funded simulation with the 2% MAX RISK PER TRADE rule.
====================================================================
Rule: on $50K and above funded Master, any single trade losing more than
2% of starting balance closes the account. Trades within 10 min of closing
a same-direction loser are combined for the check (martingale-killer).

Rule applies ONLY on funded (Master) account, NOT during challenge.

Models BOTH FP 50K and FP 100K funded with trade-level bootstrap.
Each simulated day, we sample a real historical date and apply every
trade from that day in sequence.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades.csv"

CFD_COST = 6.0
HORIZON = 252
N_SIMS = 5_000

FUNDED_CONFIGS = {
    "FP_50K_2pct":   {"start":  50_000.0, "floor":  45_000.0, "dll": 2_500.0,
                      "trade_limit": 1_000.0, "min_wd": 1_000.0, "cycle_td": 10, "split": 0.80,
                      "label": "FP 50K funded (2% trade rule = $1,000/trade)"},
    "FP_100K_2pct":  {"start": 100_000.0, "floor":  90_000.0, "dll": 5_000.0,
                      "trade_limit": 2_000.0, "min_wd": 2_000.0, "cycle_td": 10, "split": 0.80,
                      "label": "FP 100K funded (2% trade rule = $2,000/trade)"},
    "FTMO_100K_1pct":{"start": 100_000.0, "floor":  90_000.0, "dll": 5_000.0,
                      "trade_limit": 1_000.0, "min_wd":   50.0, "cycle_td": 10, "split": 0.80,
                      "label": "FTMO 100K funded (1% trade rule = $1,000/trade)"},
}


def load_trade_packs():
    """Returns list of (date, list_of_(strat,pnl,entry_ts)) tuples, sorted chronologically."""
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"])
    packs = []
    for date, group in df.groupby("date", sort=True):
        trades = list(zip(group["strat"].tolist(),
                          group["pnl_$"].tolist(),
                          group["entry_ts"].tolist()))
        packs.append((date, trades))
    return packs


def combine_martingale_trades(trades, mnq):
    """Combine same-direction trades within 10 min into single risk events for the 2% check.
    Heuristic: same strat + entry_ts within 10 min of a prior trade's entry = combined."""
    combined = []
    current_group = None
    for strat, pnl, entry_ts in trades:
        pnl_scaled = pnl * (mnq / 10.0)
        if current_group is None:
            current_group = [strat, pnl_scaled, entry_ts]
        else:
            cur_strat, cur_pnl, cur_ts = current_group
            if (strat == cur_strat
                    and (entry_ts - cur_ts).total_seconds() < 600
                    and cur_pnl < 0):
                # combine
                current_group[1] += pnl_scaled
                current_group[2] = entry_ts
            else:
                combined.append(current_group[1])
                current_group = [strat, pnl_scaled, entry_ts]
    if current_group is not None:
        combined.append(current_group[1])
    return combined


def simulate_funded(packs, mnq, cfg, rng, use_2pct=True):
    n_days = len(packs)
    bal = cfg["start"]
    days_since = 0
    pmts = 0
    cash = 0.0
    days_to_1st = None
    busted = False
    bust_reason = None
    n_trades_per_day = np.mean([len(p[1]) for p in packs])

    for d in range(HORIZON):
        idx = rng.integers(0, n_days)
        _, trades = packs[idx]
        # check each combined trade against 2% rule
        if use_2pct:
            combined_pnls = combine_martingale_trades(trades, mnq)
            for cp in combined_pnls:
                if cp <= -cfg["trade_limit"]:
                    busted = True
                    bust_reason = "2pct_trade"
                    break
            if busted:
                break

        # daily aggregate PnL (gross strategy pnl - CFD costs)
        gross_day = sum(p[1] for p in trades) * (mnq / 10.0)
        cost_day = len(trades) * CFD_COST * mnq
        day_pnl = gross_day - cost_day

        if day_pnl <= -cfg["dll"]:
            busted = True
            bust_reason = "DLL"
            break
        bal += day_pnl
        if bal < cfg["floor"]:
            busted = True
            bust_reason = "MaxLoss"
            break
        days_since += 1
        if days_since >= cfg["cycle_td"]:
            profit = bal - cfg["start"]
            if profit >= cfg["min_wd"]:
                bal -= profit
                pmts += 1
                cash += profit * cfg["split"]
                if days_to_1st is None:
                    days_to_1st = d
            days_since = 0
    return dict(busted=busted, bust_reason=bust_reason, payouts=pmts, cash=cash,
                days_to_1st=days_to_1st)


def main():
    packs = load_trade_packs()
    print(f"Loaded {len(packs)} trading days, {sum(len(p[1]) for p in packs)} total trades")
    print(f"Avg trades/day: {np.mean([len(p[1]) for p in packs]):.2f}\n")

    for cfg_key, cfg in FUNDED_CONFIGS.items():
        print(f"\n=== {cfg['label']} ===")
        rows = []
        for mnq in [1, 2, 3, 4, 5]:
            # WITHOUT the 2% rule (just MaxLoss + DLL) — for reference
            rng_off = np.random.default_rng(seed=hash(cfg_key) % 9973 + mnq + 1000)
            sims_off = [simulate_funded(packs, mnq, cfg, rng_off, use_2pct=False) for _ in range(N_SIMS)]
            bust_off = np.mean([s["busted"] for s in sims_off])
            pmts_off = [s["payouts"] for s in sims_off]
            cash_off = [s["cash"] for s in sims_off]

            # WITH the 2% rule
            rng_on = np.random.default_rng(seed=hash(cfg_key) % 9973 + mnq + 2000)
            sims_on = [simulate_funded(packs, mnq, cfg, rng_on, use_2pct=True) for _ in range(N_SIMS)]
            bust_on = np.mean([s["busted"] for s in sims_on])
            two_pct_share = sum(1 for s in sims_on if s["bust_reason"] == "2pct_trade") / N_SIMS
            pmts_on = [s["payouts"] for s in sims_on]
            cash_on = [s["cash"] for s in sims_on]
            t1 = [s["days_to_1st"] for s in sims_on if s["days_to_1st"] is not None]

            rows.append({
                "mnq": mnq,
                "bust_NO_2pct": bust_off,
                "bust_WITH_2pct": bust_on,
                "2pct_specific_busts": two_pct_share,
                "any_pmt_2pct": np.mean([s["payouts"] >= 1 for s in sims_on]),
                "median_pmts_2pct": int(np.median(pmts_on)),
                "median_cash_$_2pct": np.median(cash_on),
                "mean_cash_$_2pct": np.mean(cash_on),
                "median_d_to_1st": int(np.median(t1)) if t1 else None,
            })
        df = pd.DataFrame(rows)
        pd.set_option("display.width", 220)
        print(df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    out = ROOT / "live" / "combined deployment plan" / "fp_2pct_rule_funded.csv"
    print(f"\nNote: 'combine martingale' applied — same-strat trades within 10 min combined for 2% check.")


if __name__ == "__main__":
    main()

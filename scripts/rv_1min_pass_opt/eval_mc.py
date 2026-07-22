"""Decorrelated futures-eval Monte Carlo for the RV 1-min strategy.

Faithful to the live farm `futures_50` plan (live/farm/eval_passer.py):
  start $50k, $2k trailing DD, floor = min($50,100, peak_EOD_balance - $2,000)  (locks at $50,100
    once banked >= +$2,100 EOD), FLOATING-blowable: blow when cash + trade's worst float <= floor.
  target $3,000; CONSISTENCY 0.50 -> true_target = max($3,000, biggest_day / 0.5); PASS when
    total >= true_target.  day_cap $1,500 banked/day; stop_new_at $1,400 -> no new trades that day.

Decorrelation (copies=1): each account takes an independent subsample of the signal stream. The
low-variance design point = ~1 trade per account per active day, so we draw `trades_per_day` iid
trades/day from the strategy's trade pool. (The daily cap keeps biggest_day ~ one $1,500 win, so
true_target stays ~$3,000 -- raising trades_per_day is a variance/throughput sensitivity.)

Milk phase (total>=$2,800 -> 1-MNQ 4-strat nudge) is omitted: with the daily cap, true_target stays
$3,000 and ±$1,500 trades cross it cleanly, so the nudge is near-deterministic.
"""
from __future__ import annotations

import numpy as np

START = 50_000.0
DD = 2_000.0
LOCK_FLOOR = 50_100.0      # floor locks here once peak_EOD_balance >= 52,100
TARGET = 3_000.0
DAY_CAP = 1_500.0
STOP_NEW_AT = 1_400.0
CONSISTENCY = 0.50


def simulate(pool_pnl: np.ndarray, pool_mae: np.ndarray, rng: np.random.Generator,
             trades_per_day: int = 1, cap_days: int = 504) -> tuple[str, int]:
    """One eval account. pool_pnl/pool_mae: per-trade realized $ and worst-float $ (<=0).
    Returns (outcome, days) where outcome in {pass, bust, timeout}."""
    n = pool_pnl.size
    cash = START
    peak_bal = START
    peak_day = 0.0           # biggest finalized day's profit -> consistency target
    for d in range(cap_days):
        day_profit = 0.0
        for _ in range(trades_per_day):
            # stop taking NEW trades once today's banked profit hit the cap
            if day_profit >= STOP_NEW_AT - 1e-9:
                break
            j = rng.integers(0, n)
            pnl, mae = pool_pnl[j], pool_mae[j]
            floor = min(LOCK_FLOOR, peak_bal - DD)
            # floating blow: the open trade's worst excursion breaches the floor intratrade
            if cash + mae <= floor + 1e-9:
                return "bust", d + 1
            cash += pnl
            day_profit += pnl
            # realized blow (e.g. a force-close loss landing exactly on the floor)
            if cash <= floor + 1e-9:
                return "bust", d + 1
        # ---- end of day ----
        if day_profit > peak_day:
            peak_day = day_profit
        if cash > peak_bal:
            peak_bal = cash
        total = cash - START
        true_target = max(TARGET, peak_day / CONSISTENCY)
        if total >= true_target:
            return "pass", d + 1
    return "timeout", cap_days


def run_mc(pool_pnl: np.ndarray, pool_mae: np.ndarray, n_sims: int = 20_000,
           trades_per_day: int = 1, cap_days: int = 504, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    outs = np.empty(n_sims, dtype=object)
    days = np.empty(n_sims, dtype=int)
    for i in range(n_sims):
        outs[i], days[i] = simulate(pool_pnl, pool_mae, rng, trades_per_day, cap_days)
    pa = outs == "pass"; bu = outs == "bust"; to = outs == "timeout"
    pdays = days[pa]
    return {
        "n": n_sims,
        "pass": float(pa.mean()), "bust": float(bu.mean()), "timeout": float(to.mean()),
        "median_days": float(np.median(pdays)) if pa.any() else float("nan"),
        "p90_days": float(np.percentile(pdays, 90)) if pa.any() else float("nan"),
        "win_rate": float((pool_pnl > 0).mean()),
        "ev_per_trade": float(pool_pnl.mean()),
        "n_trades_pool": int(pool_pnl.size),
    }

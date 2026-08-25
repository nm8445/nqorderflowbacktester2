"""Consistency-plan comparison: what a 40%-consistency firm costs vs a 50% firm, in the REAL
lane-rotation farm (10 decorrelated lanes x 3 firms = 30 accounts, user's 2026-08-17 spec).

THE RULE -> THE RISK
  A firm's consistency rule caps the biggest day at `c` x total. With the farm's 1:1 bracket and its
  cap force-close, the max bankable day IS the per-trade risk, so:  risk = c x target.
      50% x $3,000 = $1,500  (Topstep, Lucid)      40% x $3,000 = $1,200  (Tradeify challenge)
  That's the ONLY structural difference: same $50k, same $2k trailing-then-lock DD, same $3k target.
  A 50% account passes on 2 clean wins; a 40% account needs 3 (the third force-closed at the ~$610
  of remaining room, at full $1,200 risk -- user's choice, neg-RR trades hit more often).

WHAT'S SIMULATED (all from real data, no assumed win rates)
  * Signal stream = the REAL chronological farm-config 1:1 R sequence (combined_eval_R.csv, 3,504
    trades 2020-12 -> 2026-06, WR 54.0%, incl. 884 partial force-closes). Sequential mode walks the
    real order from a random start, so clustering/regime survives; iid mode resamples R for contrast.
  * Lane rotation: a lane takes every Nth signal (N = n_lanes). All 3 accounts in a lane get the SAME
    signal, so the three plans are compared on IDENTICAL signal sequences (paired).
  * Account machine mirrors live/farm/eval_passer.py: trailing floor min($100, peak-$2,000);
    tier_risk ($750 while remaining DD < $2,000, else the plan's day_cap); cap force-close at
    min(day_room, pass_room); DONE at stop_new_at; MILKING at $2,800 (1 MNQ, every signal);
    true_target = max($3,000, biggest_day / c).

KNOWN UNMODELLED (both plans equally, so the DELTA is the trustworthy number)
  * Intra-trade floating drawdown (see reference_mae_exit_bar_bug) -> blow rates are optimistic.
  * The 40-micro contract cap: a very tight stop would be sized down, risking < the target $.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_R_CSV = HERE / "combined_eval_R.csv"
MILK_CSV = HERE.parents[1] / "scripts" / "futurespropmc" / "results" / "combined_4way_with_mae_1min.csv"

START_DD = 2_000.0        # $50k eval max loss (trailing, then locks)
LOCK_AT = 100.0           # floor locks at start_bal + $100
TARGET = 3_000.0
PASS_BUFFER = 10.0        # force-close the final trade just OVER the line
MILK_AT = 2_800.0         # leave the $-bracket rotation, milk 1 MNQ to the (consistency-adj) target
TIGHT_RISK = 750.0        # DD-aware tight tier
RESUME_DD = 2_000.0       # remaining DD at/above which full risk resumes
TRADING_DAYS_YR = 251

# Funded-side inputs, MEASURED by farm_income_mc.py (Structure A, 120k lifecycles, this session).
# Tradeify's FUNDED stage has the same rules as Lucid funded (user), so both plans share these.
PER_FUNDED = 1_272.0      # E[$ the user pockets] per passed eval, after the 90% split
FUNDED_P80_DAYS = 22.0    # p80 trading days to extract a funded account
EVAL_FEE = 100.0

PLANS = {
    "futures_50": dict(label="Topstep / Lucid (50%)", day_cap=1500.0, stop_new_at=1400.0, consistency=0.50),
    "tradeify_40": dict(label="Tradeify (40%)", day_cap=1200.0, stop_new_at=1100.0, consistency=0.40),
}


def load_signals() -> tuple[np.ndarray, np.ndarray]:
    """(R per signal, business-day ordinal per signal), sorted into real chronological order.

    combined_eval_R.csv is written STRAT-BY-STRAT, not chronologically -- it must be sorted or a
    'sequential' walk just marches through OD, then RV, then B2, then FB. And OD enters 19:00 ET
    (255 Sunday rows, 2 Friday rows), so its trade settles in the NEXT session: roll OD dates by one
    business day to put every signal on its FIRM day, same convention as farm_income_mc.
    """
    df = pd.read_csv(EVAL_R_CSV)
    dt = pd.to_datetime(df["date"])
    firm = dt.where(df["strat"] != "OD", dt + pd.tseries.offsets.BDay(1))
    df = df.assign(_firm=firm).sort_values(["_firm", "strat"], kind="stable").reset_index(drop=True)
    # business-day ordinal (weekends collapse; holidays count as 1 day -- ~9/yr, immaterial here)
    ords = np.busday_count(np.datetime64("2020-12-01"),
                           df["_firm"].to_numpy("datetime64[D]")).astype(np.int64)
    return df["R"].to_numpy(float), ords


def load_milk_per_trade() -> np.ndarray:
    """Per-trade 1-MNQ P&L of the 4-way combined with DYNAMIC exits (pnl_1c is 1 NQ -> /10)."""
    return pd.read_csv(MILK_CSV)["pnl_1c"].to_numpy(float) / 10.0


def run_eval(R: np.ndarray, day: np.ndarray, milk: np.ndarray, start: int, n_lanes: int,
             plan: dict, rng: np.random.Generator, max_signals: int = 6000) -> tuple[bool, int, int]:
    """One eval account on one lane. Returns (passed, trading_days_elapsed, trades_taken)."""
    cap, stop_new, c = plan["day_cap"], plan["stop_new_at"], plan["consistency"]
    n = R.size
    profit = peak = 0.0
    day_profit = peak_day = 0.0
    cur_day = day[start % n]
    day0 = cur_day
    days_elapsed = trades = 0
    milking = False

    for k in range(max_signals):
        i = (start + k) % n
        d = day[i]
        if d != cur_day:                              # firm rolled the day
            delta = int(d - cur_day)
            days_elapsed += delta if delta > 0 else 1  # delta<0 = wrapped past the end -> count 1 day
            peak_day = max(peak_day, day_profit)
            day_profit = 0.0
            cur_day = d

        true_target = max(TARGET, max(peak_day, day_profit) / c)
        floor = min(LOCK_AT, peak - START_DD)

        if milking:
            dp = milk[rng.integers(milk.size)]        # 1 MNQ, every signal (milkers don't rotate)
        else:
            if k % n_lanes != 0:                      # not this lane's turn
                continue
            if day_profit >= stop_new - 1e-9:         # DONE for the day
                continue
            risk = TIGHT_RISK if (profit - floor) < RESUME_DD else cap
            dp = R[i] * risk
            if dp > 0:                                # cap force-close: day room / pass room
                dp = min(dp, max(0.0, min(cap - day_profit, true_target + PASS_BUFFER - profit)))
            trades += 1

        profit += dp
        day_profit += dp
        peak = max(peak, profit)
        peak_day = max(peak_day, day_profit)

        if profit <= min(LOCK_AT, peak - START_DD) + 1e-9:
            return False, days_elapsed, trades
        if profit >= max(TARGET, peak_day / c) - 1e-9:
            return True, days_elapsed, trades
        if not milking and profit >= MILK_AT:
            milking = True
    return False, days_elapsed, trades


def simulate(n_iter: int, n_lanes: int, mode: str, seed: int) -> dict:
    R, day = load_signals()
    milk = load_milk_per_trade()
    rng = np.random.default_rng(seed)
    keys = list(PLANS)
    out = {k: dict(passed=[], days=[], trades=[]) for k in keys}
    all_pass = np.zeros(n_iter, dtype=bool)
    all_blow = np.zeros(n_iter, dtype=bool)

    for it in range(n_iter):
        if mode == "iid":                             # resample R, keep the real day cadence
            idx = rng.integers(R.size, size=R.size)
            Rs, ds = R[idx], day
            start = 0
        else:
            Rs, ds = R, day
            start = int(rng.integers(R.size))
        res = {}
        for k in keys:
            p, dd, tr = run_eval(Rs, ds, milk, start, n_lanes, PLANS[k], rng)
            out[k]["passed"].append(p); out[k]["days"].append(dd); out[k]["trades"].append(tr)
            res[k] = p
        # lane = 1 Topstep + 1 Lucid (both futures_50, identical signals -> identical outcome) + 1 Tradeify
        all_pass[it] = res["futures_50"] and res["tradeify_40"]
        all_blow[it] = (not res["futures_50"]) and (not res["tradeify_40"])

    stats = {}
    for k in keys:
        p = np.array(out[k]["passed"]); d = np.array(out[k]["days"]); t = np.array(out[k]["trades"])
        stats[k] = dict(
            pass_rate=p.mean(),
            se=p.std(ddof=1) / np.sqrt(p.size),
            days_med=float(np.median(d[p])) if p.any() else float("nan"),
            days_p80=float(np.percentile(d[p], 80)) if p.any() else float("nan"),
            trades_med=float(np.median(t[p])) if p.any() else float("nan"),
            days_blow=float(np.median(d[~p])) if (~p).any() else float("nan"),
        )
    stats["_lane"] = dict(all3_pass=all_pass.mean(), all3_blow=all_blow.mean())
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--lanes", type=int, default=10)
    ap.add_argument("--mode", choices=["sequential", "iid", "both"], default="both")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    R, day = load_signals()
    n_days = int(day.max() - day.min() + 1)
    print(f"signal stream: {R.size} trades over {n_days} trading days = {R.size / n_days:.2f}/day  "
          f"(WR {(R > 0).mean() * 100:.1f}%, mean R {R.mean():+.3f})")
    print(f"lane rotation: {a.lanes} lanes -> a lane's turn every {a.lanes / (R.size / n_days):.1f} "
          f"trading days\n")

    for mode in (["sequential", "iid"] if a.mode == "both" else [a.mode]):
        s = simulate(a.iters, a.lanes, mode, a.seed)
        print(f"=== {mode} ({a.iters:,} evals/plan, {a.lanes} lanes) ===")
        for k, p in PLANS.items():
            st = s[k]
            print(f"  {p['label']:<24} risk ${p['day_cap']:,.0f}  "
                  f"pass {st['pass_rate'] * 100:5.1f}% (+-{st['se'] * 100:.1f})  "
                  f"trades(med) {st['trades_med']:.0f}  "
                  f"days-to-pass med {st['days_med']:.0f} / p80 {st['days_p80']:.0f}  "
                  f"days-to-blow med {st['days_blow']:.0f}")
        d = s["tradeify_40"]["pass_rate"] - s["futures_50"]["pass_rate"]
        rel = d / s["futures_50"]["pass_rate"] * 100
        print(f"  delta (40% - 50%): {d * 100:+.1f} pp  ({rel:+.1f}% relative)")
        print(f"  lane co-movement: all-3 pass {s['_lane']['all3_pass'] * 100:.1f}%  "
              f"all-3 blow {s['_lane']['all3_blow'] * 100:.1f}%")

        # --- $/yr per ACCOUNT SLOT: a slot runs evals serially (blow -> rebuy; pass -> funded, then
        #     rebuy). Throughput, not pass rate, is where a tighter consistency rule actually bites.
        econ = {}
        for k, p in PLANS.items():
            st = s[k]
            pr = st["pass_rate"]
            cycle = pr * (st["days_med"] + FUNDED_P80_DAYS) + (1 - pr) * st["days_blow"]
            per_attempt = pr * PER_FUNDED - EVAL_FEE
            econ[k] = dict(cycle=cycle, per_yr=TRADING_DAYS_YR / cycle,
                           dollars=TRADING_DAYS_YR / cycle * per_attempt)
            print(f"  {p['label']:<24} cycle {cycle:4.1f} td -> {econ[k]['per_yr']:4.1f} evals/yr/slot "
                  f"x ${per_attempt:,.0f}/eval = ${econ[k]['dollars']:,.0f}/yr per slot")
        lane_mix = 2 * econ["futures_50"]["dollars"] + econ["tradeify_40"]["dollars"]
        lane_all50 = 3 * econ["futures_50"]["dollars"]
        print(f"  LANE (Topstep+Lucid+Tradeify) ${lane_mix:,.0f}/yr vs all-50% ${lane_all50:,.0f}/yr "
              f"({(lane_mix / lane_all50 - 1) * 100:+.0f}%)")
        print(f"  10 LANES / 30 accounts: ${lane_mix * 10:,.0f}/yr vs ${lane_all50 * 10:,.0f}/yr "
              f"-> the 40% rule costs ${(lane_all50 - lane_mix) * 10:,.0f}/yr\n")


if __name__ == "__main__":
    main()

"""Phase 4 — full-cycle portfolio Monte Carlo.

CHALLENGE (per the user's trading discipline):
  30 accounts, $50k, $2k trailing EOD drawdown, +$3k target, 50% consistency
  (best single day <= 50% of total profit). Each day a historical signal-pack is
  sampled; signals fan out to PAIRS of accounts (signal -> next 2 not-done
  accounts, cyclic). Per account: take 1 trade -> WIN stops the day; LOSS -> take
  the next signal as recovery; 2nd LOSS -> ~-$2k day -> bust. Max 2 trades/day.
  Trades use the Phase-1 fixed-ATR brackets sized to ~$1k risk.

FUNDED (milking, per passed account):
  Trade the EXISTING combined strat at 1 MNQ (combined_trades_with_mae.csv x0.1),
  same $2k trailing EOD floor. Count winning days (>= $150). At 5 winning days ->
  withdraw 50% of profit; balance -= payout; trailing FLOOR STAYS (HWM-based), so
  payouts shrink the cushion. Loop until bust or 504 trading days.

OUTPUT: per candidate config -> single-acct P(pass), E[#passed of 30],
E[$ withdrawn], net = withdrawn - $3000 account spend, distribution + cycle multiple.

Run:  python -m scripts.propfirm_milking.mc_portfolio
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.propfirm_milking.common import (
    RESULTS_DIR, CONFIGS_DIR, FUNDED_MAE_CSV, eval_bracket, COST_RT_PER_MNQ,
)
from scripts.propfirm_milking.entries import build_entry_packs

# ============ Locked Phase-1 challenge brackets (atr_len, sl_mult, tp_mult) ============
CHALLENGE_BRACKETS = {
    "RV":  (10, 1.50, 2.00),   # RR 1.33, WR 48.9%
    "FB":  (28, 2.25, 3.00),   # RR 1.33, WR 48.4%
    "MEC": (10, 2.00, 2.75),   # RR 1.375, WR 48.2%
    "B2":  (20, 1.50, 2.75),   # RR 1.83, WR 42.6%
}
# Candidate combined configs to compare
CANDIDATES = {
    "RV+FB+MEC":     ["RV", "FB", "MEC"],
    "RV+FB+MEC+B2":  ["RV", "FB", "MEC", "B2"],
}

# ============ Sim params ============
N_SIMS = 4000
N_ACCTS = 30
PAIR = 2                 # signal mirrored to this many accounts
MAX_TRADES_DAY = 2
START_BAL = 50_000.0
PROFIT_TGT = 3_000.0
DD = 2_000.0
INIT_FLOOR = START_BAL - DD
CONSISTENCY = 0.50
CHALLENGE_MAX_DAYS = 504
FUNDED_MAX_DAYS = 504
WIN_DAY_MIN = 150.0
PAYOUT_FRAC = 0.50
ACCOUNT_COST = 100.0     # $/challenge account
MAX_WORKERS = 6

# Worker globals
_CHAL_DAYS = None        # list[ list[(win:bool, pnl:float)] ] ordered by time within day
_FUNDED_DAILY = None     # np.ndarray of funded daily-net $ at 1 MNQ


# ============ Build day-packs ============
def build_challenge_day_packs(strats: list[str]) -> list[list[tuple]]:
    """Per-day ordered list of (win, pnl_$) signals across the chosen strats."""
    frames = []
    for s in strats:
        atr_len, sl, tp = CHALLENGE_BRACKETS[s]
        packs = build_entry_packs(s, verbose=False)
        tr = eval_bracket(packs, atr_len, sl, tp)
        tr = tr[["date", "fill_time", "win", "pnl_$"]].copy()
        frames.append(tr)
    allt = pd.concat(frames, ignore_index=True).sort_values(["date", "fill_time"])
    days = []
    for _, grp in allt.groupby("date", sort=True):
        days.append(list(zip(grp["win"].astype(bool).tolist(), grp["pnl_$"].tolist())))
    return days


def build_funded_daily() -> np.ndarray:
    """Funded daily NET $ at 1 MNQ from the existing combined (x0.1), minus 1-MNQ costs."""
    df = pd.read_csv(FUNDED_MAE_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["pnl_1mnq"] = df["pnl_$"] * 0.1          # log pnl is at 1 NQ = 10 MNQ
    g = df.groupby("date").agg(pnl=("pnl_1mnq", "sum"), n=("pnl_1mnq", "size"))
    daily = (g["pnl"] - g["n"] * COST_RT_PER_MNQ).values   # 1 MNQ cost per trade
    return daily.astype(float)


# ============ Challenge sim (30 accounts, coupled by shared daily pack) ============
def run_challenge(rng) -> tuple[int, list]:
    bal = np.full(N_ACCTS, START_BAL)
    hwm = np.full(N_ACCTS, START_BAL)
    floor = np.full(N_ACCTS, INIT_FLOOR)
    maxday = np.zeros(N_ACCTS)          # best positive day per account
    alive = np.ones(N_ACCTS, dtype=bool)
    passed = np.zeros(N_ACCTS, dtype=bool)
    pass_days = []
    nday = len(_CHAL_DAYS)

    for d in range(CHALLENGE_MAX_DAYS):
        active = [a for a in range(N_ACCTS) if alive[a] and not passed[a]]
        if not active:
            break
        pack = _CHAL_DAYS[rng.integers(0, nday)]
        nsig = len(pack)
        day_pnl = dict.fromkeys(active, 0.0)
        trades = dict.fromkeys(active, 0)
        done = dict.fromkeys(active, False)
        order = active
        pi = 0
        si = 0
        n_order = len(order)
        while si < nsig:
            win, pnl = pack[si]
            assigned = 0
            tries = 0
            while assigned < PAIR and tries < n_order:
                a = order[pi % n_order]
                pi += 1
                tries += 1
                if done[a]:
                    continue
                day_pnl[a] += pnl
                trades[a] += 1
                if win or trades[a] >= MAX_TRADES_DAY:
                    done[a] = True
                assigned += 1
            si += 1
            if assigned == 0:           # everyone done
                break

        for a in active:
            dp = day_pnl[a]
            bal[a] += dp
            if dp > maxday[a]:
                maxday[a] = dp
            if bal[a] < floor[a]:
                alive[a] = False
                continue
            tot = bal[a] - START_BAL
            if tot >= PROFIT_TGT and maxday[a] <= CONSISTENCY * tot:
                passed[a] = True
                pass_days.append(d + 1)
                continue
            if bal[a] > hwm[a]:
                hwm[a] = bal[a]
                floor[a] = max(INIT_FLOOR, hwm[a] - DD)

    return int(passed.sum()), pass_days


# ============ Funded sim (one passed account) ============
def _play_day_single(pack, rng) -> float:
    """One funded day under the eval discipline (single account): take a signal;
    WIN stops the day; LOSS -> one recovery signal; (2nd loss => ~-$2k natural bust)."""
    nsig = len(pack)
    if nsig == 0:
        return 0.0
    i = int(rng.integers(0, nsig))
    win0, pnl0 = pack[i]
    if win0:
        return pnl0
    dp = pnl0
    if i + 1 < nsig:
        _, pnl1 = pack[i + 1]
        dp += pnl1
    return dp


def run_funded(rng, reach_scale: float = 1.0) -> tuple[float, int, int]:
    """Two-phase funded milking (user's actual approach):
      Phase 1: reach +$3k via the EVAL discipline -- else bust -> $0.
               `reach_scale` scales per-trade risk vs the ~$1k baseline (e.g. 0.5 = ~$500
               risk) by scaling each eval-day pnl linearly (win/loss both scale with size).
      Phase 2: 1 MNQ combined for winning days; at 5 winning days withdraw 50% of
               profit (floor stays -> cushion shrinks). Loop until bust / day budget.
    Winning days count across both phases (firm rule). Returns (withdrawn, n_payouts, reached).
    """
    bal = START_BAL
    hwm = START_BAL
    floor = INIT_FLOOR
    win_days = 0
    nday = len(_CHAL_DAYS)
    nfund = len(_FUNDED_DAILY)

    # --- Phase 1: reach +$3k via eval discipline ---
    reached = False
    used = 0
    for d in range(FUNDED_MAX_DAYS):
        used += 1
        dp = _play_day_single(_CHAL_DAYS[rng.integers(0, nday)], rng) * reach_scale
        bal += dp
        if bal < floor:
            return 0.0, 0, 0
        if dp >= WIN_DAY_MIN:
            win_days += 1
        if bal > hwm:
            hwm = bal
            floor = max(INIT_FLOOR, hwm - DD)
        if bal - START_BAL >= PROFIT_TGT:
            reached = True
            break
    if not reached:
        return 0.0, 0, 0

    # --- Phase 2: 1 MNQ maintenance milking until bust ---
    withdrawn = 0.0
    n_payouts = 0
    for d in range(used, FUNDED_MAX_DAYS):
        dp = _FUNDED_DAILY[rng.integers(0, nfund)]
        bal += dp
        if bal < floor:
            break
        if dp >= WIN_DAY_MIN:
            win_days += 1
        if bal > hwm:
            hwm = bal
            floor = max(INIT_FLOOR, hwm - DD)
        if win_days >= 5:
            profit = bal - START_BAL
            if profit > 0:
                pay = PAYOUT_FRAC * profit
                withdrawn += pay
                bal -= pay              # floor stays -> cushion shrinks
                n_payouts += 1
            win_days = 0
    return withdrawn, n_payouts, 1


# ============ One full cycle ============
def run_cycle(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_passed, pass_days = run_challenge(rng)
    withdrawn = 0.0
    n_reached = 0
    n_payouts = 0
    for _ in range(n_passed):
        w, p, r = run_funded(rng)
        withdrawn += w
        n_reached += r
        n_payouts += p
    cost = N_ACCTS * ACCOUNT_COST
    return {
        "n_passed": n_passed,
        "n_funded_reached_3k": n_reached,
        "n_payouts": n_payouts,
        "withdrawn": withdrawn,
        "net": withdrawn - cost,
        "med_pass_day": int(np.median(pass_days)) if pass_days else -1,
    }


def _init_worker(chal_days, funded_daily):
    global _CHAL_DAYS, _FUNDED_DAILY
    _CHAL_DAYS = chal_days
    _FUNDED_DAILY = funded_daily


def single_acct_pass_rate(n_cycles_passed, n_sims) -> float:
    return None  # computed from aggregate below


def run_candidate(name: str, strats: list[str], funded_daily: np.ndarray, n_sims=N_SIMS) -> dict:
    t0 = time.time()
    print(f"\n=== {name} ===")
    chal_days = build_challenge_day_packs(strats)
    n_signals = sum(len(d) for d in chal_days)
    avg_sig = n_signals / len(chal_days)
    print(f"  {len(chal_days)} trading days | {n_signals} signals | {avg_sig:.1f} sig/day "
          f"| {n_sims} sims x {N_ACCTS} accts on {MAX_WORKERS} workers")
    seeds = list(range(n_sims))
    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                             initializer=_init_worker, initargs=(chal_days, funded_daily)) as ex:
        rows = list(ex.map(run_cycle, seeds, chunksize=40))
    df = pd.DataFrame(rows)
    p_single = df["n_passed"].mean() / N_ACCTS
    tot_passed = df["n_passed"].sum()
    reach_rate = df["n_funded_reached_3k"].sum() / tot_passed if tot_passed else 0.0
    withdrawn_per_funded = df["withdrawn"].sum() / tot_passed if tot_passed else 0.0
    payouts_per_funded = df["n_payouts"].sum() / tot_passed if tot_passed else 0.0
    res = {
        "config": name,
        "P(pass) single-acct": round(p_single, 4),
        "E[#passed of 30]": round(df["n_passed"].mean(), 2),
        "funded reach-$3k rate": round(reach_rate, 3),
        "$/funded acct": round(withdrawn_per_funded, 0),
        "payouts/funded acct": round(payouts_per_funded, 2),
        "E[$ withdrawn/cycle]": round(df["withdrawn"].mean(), 0),
        "E[net $/cycle]": round(df["net"].mean(), 0),
        "median net": round(df["net"].median(), 0),
        "P(net>0)": round((df["net"] > 0).mean(), 3),
        "cycle_multiple": round(df["withdrawn"].mean() / (N_ACCTS * ACCOUNT_COST), 2),
        "med_pass_day": int(df.loc[df["med_pass_day"] > 0, "med_pass_day"].median())
                         if (df["med_pass_day"] > 0).any() else -1,
    }
    print(f"  done in {time.time()-t0:.1f}s")
    for k, v in res.items():
        print(f"    {k:24} {v}")
    return res


def main():
    pd.set_option("display.width", 240)
    print("Building funded daily-net distribution (existing combined @1 MNQ)...")
    funded_daily = build_funded_daily()
    print(f"  {len(funded_daily)} funded days | mean ${funded_daily.mean():.1f}/day "
          f"| win-day frac (>=$150) {np.mean(funded_daily>=WIN_DAY_MIN):.2f}")

    results = []
    for name, strats in CANDIDATES.items():
        results.append(run_candidate(name, strats, funded_daily))

    out = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "portfolio_report.csv", index=False)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIGS_DIR / "challenge_config.json", "w") as f:
        json.dump({"brackets": CHALLENGE_BRACKETS, "candidates": CANDIDATES}, f, indent=2)
    print("\n=== PORTFOLIO REPORT ===")
    print(out.to_string(index=False))
    print(f"\nSaved -> {RESULTS_DIR / 'portfolio_report.csv'}")


if __name__ == "__main__":
    main()

"""ONE slot continuously running over 1 year with eval refresh.
   1 MNQ no marti, stagger A ($1500 first @ $3K, $1000 subseq @ $2K).
   When account busts, buy new eval, gambler's ruin pass, restart."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
OD_RAW_CSV = ROOT / "live" / "overnight drift" / "trades.csv"

N_SIMS = 10_000
DAYS_PER_YEAR = 365
TRADING_DAYS_YEAR = 252
EVAL_COST = 100.0
EVAL_PASS_RATE = 0.30   # gambler's ruin estimate
EVAL_DAYS = 3           # avg days per attempt


def load_packs():
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"]).reset_index(drop=True)
    od_raw = pd.read_csv(OD_RAW_CSV)
    od_raw["entry_time"] = pd.to_datetime(od_raw["entry_time"], utc=True, format="mixed")
    qty_map = dict(zip(od_raw["entry_time"], od_raw["qty"]))
    df["qty"] = 1
    for i in df.index[df["strat"] == "OD"]:
        df.at[i, "qty"] = qty_map.get(df.at[i, "entry_ts"], 1)
    scale = np.where((df["strat"] == "OD") & (df["qty"] == 2), 0.5, 1.0)
    df["pnl_$"] = df["pnl_$"] * scale
    df["mae_$"] = df["mae_$"] * scale
    return [[(r["pnl_$"], r["mae_$"]) for _, r in grp.iterrows()]
            for _, grp in df.groupby("date", sort=True)]


def run_one_account(packs, mnq, rng, start_day, max_td_remaining, max_payouts=6):
    """Run one funded account from start_day onward. Returns (payouts, cash, end_day, busted)."""
    bal = 50_000.0
    floor = 48_000.0
    hwm = bal
    locked = False
    qual = 0
    cycle = 0.0
    stagger_first = False
    payouts = 0
    cash = 0.0
    n_packs = len(packs)
    for offset in range(max_td_remaining):
        d = start_day + offset
        if d >= TRADING_DAYS_YEAR:
            return payouts, cash, d, False
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        realized = 0.0
        for pnl, mae in trades:
            ps = pnl * (mnq/10) - 2.0 * mnq
            ms = mae * (mnq/10) - 2.0 * mnq
            if bal + realized + ms < floor:
                return payouts, cash, d, True
            realized += ps
        bal += realized
        if not locked:
            if bal > hwm: hwm = bal
            floor = max(48_000, hwm - 2000)
            if hwm >= 53_000:
                locked = True; floor = 50_000
        if realized >= 150: qual += 1
        cycle += realized
        if qual >= 5 and cycle > 0:
            gross = 0
            if not stagger_first:
                if cycle >= 3000:
                    gross = 1500; stagger_first = True
            else:
                if cycle >= 2000:
                    gross = 1000
            if gross > 0:
                bal -= gross
                if not locked:
                    hwm = max(50_000, hwm - gross)
                    floor = max(48_000, hwm - 2000)
                payouts += 1
                cash += gross * 0.9
                qual = 0; cycle = 0.0
                if payouts >= max_payouts:
                    return payouts, cash, d, False  # graduated
    return payouts, cash, start_day + max_td_remaining - 1, False


def sim_one_slot_year(packs, mnq, rng):
    """One slot running continuously for 1 trading year with eval refresh."""
    cur_day = 0  # current trading day index
    total_payouts = 0
    total_cash = 0.0
    total_eval_cost = 0.0
    total_evals = 0
    n_account_attempts = 0
    n_grads = 0
    n_busts = 0
    while cur_day < TRADING_DAYS_YEAR:
        # Eval phase
        passed = False
        while not passed and cur_day < TRADING_DAYS_YEAR:
            total_eval_cost += EVAL_COST
            total_evals += 1
            cur_day += EVAL_DAYS
            if rng.random() < EVAL_PASS_RATE:
                passed = True
        if cur_day >= TRADING_DAYS_YEAR:
            break
        # Funded phase
        remaining = TRADING_DAYS_YEAR - cur_day
        payouts, cash, end_day, busted = run_one_account(packs, mnq, rng, cur_day, remaining)
        total_payouts += payouts
        total_cash += cash
        n_account_attempts += 1
        if busted: n_busts += 1
        if payouts >= 6: n_grads += 1
        cur_day = end_day + 1
    return dict(payouts=total_payouts, cash=total_cash, eval_cost=total_eval_cost,
                evals=total_evals, attempts=n_account_attempts, busts=n_busts,
                grads=n_grads)


def report(label, results):
    payouts = np.array([r["payouts"] for r in results])
    cash = np.array([r["cash"] for r in results])
    evals = np.array([r["evals"] for r in results])
    eval_cost = np.array([r["eval_cost"] for r in results])
    attempts = np.array([r["attempts"] for r in results])
    busts = np.array([r["busts"] for r in results])
    grads = np.array([r["grads"] for r in results])
    net = cash - eval_cost
    print(f"\n=== {label} ===")
    print(f"  Account attempts per year: mean {attempts.mean():.1f}  ({busts.mean():.1f} busts, {grads.mean():.1f} grads)")
    print(f"  Total payouts per year: mean {payouts.mean():.1f}  median {int(np.median(payouts))}")
    print(f"  Total evals bought: mean {evals.mean():.1f}  eval cost ${eval_cost.mean():.0f}")
    print(f"  Annual gross cash: mean ${cash.mean():,.0f}  median ${np.median(cash):,.0f}")
    print(f"  Annual NET (gross - evals): mean ${net.mean():,.0f}  median ${np.median(net):,.0f}")
    print(f"    p25: ${np.percentile(net, 25):,.0f}   p75: ${np.percentile(net, 75):,.0f}")
    print(f"  Monthly NET: mean ${net.mean()/12:,.0f}")


def main():
    packs = load_packs()
    print("ONE slot, continuous eval refresh, 1 year. Stagger A ($1500 @ $3K, $1000 @ $2K).")
    print(f"Eval cost ${EVAL_COST} per attempt, {EVAL_PASS_RATE:.0%} pass rate, {EVAL_DAYS}d per attempt.")
    print(f"{N_SIMS} sims.")

    for mnq in [1, 2, 3]:
        rng = np.random.default_rng(seed=5000 + mnq)
        results = [sim_one_slot_year(packs, mnq, rng) for _ in range(N_SIMS)]
        report(f"{mnq} MNQ, no marti, stagger A, with refresh", results)


if __name__ == "__main__":
    main()

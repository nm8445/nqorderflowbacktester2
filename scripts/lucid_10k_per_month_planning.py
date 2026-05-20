"""
Lucid Flex 50K — slot throughput planning for $10K/month NET goal.

Assumes:
  - RV+B2 only strategy on funded account, 1 MNQ stagger A (best Lucid config)
  - Each eval costs $100 to buy
  - Gambler's-ruin eval style: ~50% pass per attempt (user supplied)
  - Each eval attempt takes ~5 calendar days (quick pass-or-fail)
  - Each slot runs continuously: eval until pass, then funded until bust/grad/cap,
    then back to eval, repeating across a calendar year.

Each slot's annual NET cash = funded_cash_extracted - eval_costs_paid.
We find the number of slots needed for $120K/yr ($10K/mo) NET.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"

N_SIMS = 10_000
DAYS_PER_YEAR = 365   # calendar days
TRADING_DAYS_PER_YEAR = 252
FUTURES_COST = 2.0
EVAL_COST = 100.0
EVAL_PASS_RATE = 0.50       # user-supplied estimate (gambler's ruin)
EVAL_DAYS_PER_ATTEMPT = 5   # calendar days

# Lucid Flex 50K
LUCID = dict(
    start=50_000, floor_init=48_000,
    lock_after=53_000, lock_floor=50_000,
    payout_cap=2000, max_payouts=6, split=0.90,
)


def load_packs(strats):
    df = pd.read_csv(TRADES_CSV)
    df = df[df["strat"].isin(strats)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"])
    packs = []
    for date, grp in df.groupby("date", sort=True):
        trades = [(r["pnl_$"], r["mae_$"]) for _, r in grp.iterrows()]
        packs.append(trades)
    return packs


def load_packs_rvb2():
    return load_packs(["RV", "B2"])


def load_packs_full():
    return load_packs(["RV", "B2", "OD"])


def simulate_funded_run(packs, rng, mnq=1, cost=FUTURES_COST, max_td=TRADING_DAYS_PER_YEAR):
    """Simulate a funded account. Returns (cash, trading_days_alive)."""
    scale = mnq / 10.0
    cfg = LUCID
    balance = cfg["start"]
    floor = cfg["floor_init"]
    hwm = balance
    locked = False
    payouts = 0
    cash = 0.0
    cycle_qual_days = 0
    cycle_profit = 0.0
    stagger_first_done = False
    n_packs = len(packs)
    days_alive = 0

    for d in range(max_td):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        daily_realized = 0.0
        busted = False
        for pnl, mae in trades:
            pnl_scaled = pnl * scale - cost * mnq
            mae_scaled = mae * scale - cost * mnq
            if balance + daily_realized + mae_scaled < floor:
                busted = True
                break
            daily_realized += pnl_scaled
        if busted:
            return cash, d + 1
        days_alive = d + 1
        balance += daily_realized
        if not locked:
            if balance > hwm: hwm = balance
            floor = max(cfg["floor_init"], hwm - (cfg["start"] - cfg["floor_init"]))
            if hwm >= cfg["lock_after"]:
                locked = True; floor = cfg["lock_floor"]
        if daily_realized >= 150:
            cycle_qual_days += 1
        cycle_profit += daily_realized
        if payouts < cfg["max_payouts"] and cycle_qual_days >= 5 and cycle_profit > 0:
            gross = 0.0
            if not stagger_first_done:
                if cycle_profit >= 3000:
                    gross = 1500; stagger_first_done = True
            else:
                if cycle_profit >= 2000:
                    gross = 1000
            if gross >= 500:
                gross = min(gross, cfg["payout_cap"])
                trader = gross * cfg["split"]
                balance -= gross
                if not locked:
                    hwm = max(cfg["start"], hwm - gross)
                    floor = max(cfg["floor_init"], hwm - (cfg["start"] - cfg["floor_init"]))
                payouts += 1
                cash += trader
                cycle_qual_days = 0; cycle_profit = 0.0
                if payouts >= cfg["max_payouts"]:
                    break

    return cash, days_alive


def simulate_slot_year(packs, rng, pass_rate=EVAL_PASS_RATE):
    return simulate_slot_year_size(packs, rng, mnq=1, pass_rate=pass_rate)


def simulate_slot_year_size(packs, rng, mnq=1, pass_rate=EVAL_PASS_RATE):
    """Simulate one slot over a 365-day calendar year at given MNQ size."""
    total_cash = 0.0
    total_eval_cost = 0.0
    total_evals_bought = 0
    n_funded_starts = 0
    cal_days_elapsed = 0

    while cal_days_elapsed < DAYS_PER_YEAR:
        while True:
            if cal_days_elapsed >= DAYS_PER_YEAR:
                break
            total_eval_cost += EVAL_COST
            total_evals_bought += 1
            cal_days_elapsed += EVAL_DAYS_PER_ATTEMPT
            if rng.random() < pass_rate:
                break
        if cal_days_elapsed >= DAYS_PER_YEAR:
            break
        cal_remaining = DAYS_PER_YEAR - cal_days_elapsed
        max_td = int(cal_remaining * 5 / 7)
        cash, td_alive = simulate_funded_run(packs, rng, mnq=mnq, max_td=max_td)
        cal_funded_elapsed = int(td_alive * 7 / 5)
        cal_days_elapsed += cal_funded_elapsed
        total_cash += cash
        n_funded_starts += 1

    net = total_cash - total_eval_cost
    return {
        "gross_cash": total_cash,
        "eval_cost": total_eval_cost,
        "net_cash": net,
        "evals_bought": total_evals_bought,
        "funded_starts": n_funded_starts,
    }


def run_set(label, packs, mnq_list=(1, 2, 3), pass_rate=0.80):
    print(f"\n=== {label} — eval pass rate {int(pass_rate*100)}% ===")
    rows = []
    for mnq in mnq_list:
        rng = np.random.default_rng(seed=mnq * 1000 + hash(label) % 999)
        sims = [simulate_slot_year_size(packs, rng, mnq=mnq, pass_rate=pass_rate) for _ in range(N_SIMS)]
        mean_net = np.mean([s["net_cash"] for s in sims])
        mean_gross = np.mean([s["gross_cash"] for s in sims])
        mean_eval = np.mean([s["eval_cost"] for s in sims])
        mean_evals = np.mean([s["evals_bought"] for s in sims])
        mean_starts = np.mean([s["funded_starts"] for s in sims])
        p25 = np.percentile([s["net_cash"] for s in sims], 25)
        p75 = np.percentile([s["net_cash"] for s in sims], 75)
        rows.append({
            "mnq": mnq,
            "starts/yr": mean_starts,
            "evals/yr": mean_evals,
            "eval_$": mean_eval,
            "gross_$": mean_gross,
            "NET_$": mean_net,
            "p25_NET": p25,
            "p75_NET": p75,
            "slots_for_10K/mo": 120_000 / mean_net if mean_net > 0 else None,
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.0f}"))
    return df


def main():
    rvb2 = load_packs_rvb2()
    full = load_packs_full()
    print(f"RV+B2 only: {len(rvb2)} days / {sum(len(p) for p in rvb2)} trades")
    print(f"Full stack: {len(full)} days / {sum(len(p) for p in full)} trades")

    df_rvb2 = run_set("RV+B2 only (no OD)", rvb2)
    df_full = run_set("FULL STACK (OD + RV + B2)", full)

    print("\n=== HEAD-TO-HEAD per MNQ (slots needed for $10K/mo NET) ===")
    print(f"  MNQ |  RV+B2 slots  |  Full stack slots  |  Delta")
    for m in [1, 2, 3]:
        r = df_rvb2[df_rvb2['mnq'] == m].iloc[0]
        f = df_full[df_full['mnq'] == m].iloc[0]
        print(f"   {m}   |     {r['slots_for_10K/mo']:5.1f}    |       {f['slots_for_10K/mo']:5.1f}        |  {f['slots_for_10K/mo'] - r['slots_for_10K/mo']:+5.1f}")
    return  # short-circuit old code

    # legacy code below (kept dead)
    print("=== ANNUAL NET PER SLOT — by funded sizing ===")
    rows = []
    for mnq in [1, 2, 3]:
        rng = np.random.default_rng(seed=mnq * 1000)
        sims = [simulate_slot_year_size(packs, rng, mnq=mnq, pass_rate=0.80) for _ in range(N_SIMS)]
        mean_net = np.mean([s["net_cash"] for s in sims])
        mean_gross = np.mean([s["gross_cash"] for s in sims])
        mean_eval = np.mean([s["eval_cost"] for s in sims])
        mean_evals = np.mean([s["evals_bought"] for s in sims])
        mean_starts = np.mean([s["funded_starts"] for s in sims])
        p25 = np.percentile([s["net_cash"] for s in sims], 25)
        p75 = np.percentile([s["net_cash"] for s in sims], 75)
        rows.append({
            "mnq_funded": mnq,
            "mean_funded_starts": mean_starts,
            "mean_evals_bought": mean_evals,
            "mean_eval_cost_$": mean_eval,
            "mean_gross_cash_$": mean_gross,
            "mean_NET_cash_$": mean_net,
            "p25_NET": p25,
            "p75_NET": p75,
            "slots_for_120K_NET": 120_000 / mean_net if mean_net > 0 else None,
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.1f}"))

    print("\n=== AT BASELINE: 1 MNQ ===")
    base = next(r for r in rows if r["mnq_funded"] == 1)
    slots = base["slots_for_120K_NET"]
    print(f"  Mean NET cash per slot per year: ${base['mean_NET_cash_$']:,.0f}")
    print(f"  Slots needed for $120K/yr ($10K/mo): {slots:.1f} -> round up to {int(np.ceil(slots))}")
    print(f"  Total evals bought per year: {base['mean_evals_bought'] * np.ceil(slots):.0f}")
    print(f"  Total eval cost per year: ${base['mean_eval_cost_$'] * np.ceil(slots):,.0f}")
    print(f"  Total funded starts per year (replacements): {base['mean_funded_starts'] * np.ceil(slots):.0f}")
    print(f"  Approx evals per month: {base['mean_evals_bought'] * np.ceil(slots) / 12:.1f}")
    print(f"  Approx eval spend per month: ${base['mean_eval_cost_$'] * np.ceil(slots) / 12:,.0f}")


if __name__ == "__main__":
    main()

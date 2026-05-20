"""Opposite mix sim: OD=2 MNQ, B2=1 MNQ, RV=1 MNQ.
   Single account stagger A + 15 copy-traded with eval refresh."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
OD_RAW_CSV = ROOT / "live" / "overnight drift" / "trades.csv"

N_SIMS = 10_000
HORIZON = 252
FUTURES_COST = 2.0
EVAL_COST = 100.0
EVAL_PASS_RATE = 0.30
EVAL_DAYS = 3


def load_packs_mixed(mnq_per_strat):
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
    marti_scale = np.where((df["strat"] == "OD") & (df["qty"] == 2), 0.5, 1.0)
    df["pnl_$"] = df["pnl_$"] * marti_scale
    df["mae_$"] = df["mae_$"] * marti_scale
    mnq_arr = df["strat"].map(mnq_per_strat).to_numpy()
    scale = mnq_arr / 10.0
    df["pnl_scaled"] = df["pnl_$"] * scale - FUTURES_COST * mnq_arr
    df["mae_scaled"] = df["mae_$"] * scale - FUTURES_COST * mnq_arr
    return [[(r["pnl_scaled"], r["mae_scaled"]) for _, r in grp.iterrows()]
            for _, grp in df.groupby("date", sort=True)]


def sim_stagger_a_single(packs, rng, start_day=0, max_payouts=6):
    """Returns dict including end_day for refresh chaining."""
    bal = 50_000.0
    floor = 48_000.0
    hwm = bal
    locked = False
    qual = 0
    cycle = 0.0
    stagger_first = False
    payouts = 0
    cash = 0.0
    days_at_payouts = []
    bust_day = None
    end_day = HORIZON
    n_packs = len(packs)
    for offset in range(HORIZON - start_day):
        d = start_day + offset
        if d >= HORIZON: break
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        realized = 0.0
        for ps, ms in trades:
            if bal + realized + ms < floor:
                return dict(payouts=payouts, cash=cash, days_at_payouts=days_at_payouts,
                            busted=True, bust_day=d, end_day=d)
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
                if cycle >= 3000: gross = 1500; stagger_first = True
            else:
                if cycle >= 2000: gross = 1000
            if gross > 0:
                bal -= gross
                if not locked:
                    hwm = max(50_000, hwm - gross)
                    floor = max(48_000, hwm - 2000)
                payouts += 1
                cash += gross * 0.9
                days_at_payouts.append(d + 1)
                qual = 0; cycle = 0.0
                if payouts >= max_payouts:
                    return dict(payouts=payouts, cash=cash, days_at_payouts=days_at_payouts,
                                busted=False, bust_day=None, end_day=d)
    return dict(payouts=payouts, cash=cash, days_at_payouts=days_at_payouts,
                busted=False, bust_day=None, end_day=HORIZON)


def sim_15_copy_trade_with_refresh(packs, rng):
    """15 accounts copy-traded perfectly = behaves as 1 account, multiplied by 15.
       With eval refresh after each portfolio bust."""
    cur_day = 0
    total_payouts = 0
    total_cash = 0.0
    total_eval_cost = 0.0
    total_evals = 0
    n_attempts = 0
    n_busts = 0
    n_grads = 0
    while cur_day < HORIZON:
        # Eval phase: pass 15 accounts (sync — but in gambler's ruin each independent)
        # Pass rate 30%, avg 3.3 evals per account-pass, 3 days each
        # In real life: pass 1 account, then duplicate signal to 15. So eval cost is for ALL 15.
        for _ in range(15):
            passed = False
            while not passed and cur_day < HORIZON:
                total_eval_cost += EVAL_COST
                total_evals += 1
                if rng.random() < EVAL_PASS_RATE:
                    passed = True
        # Assume eval phase takes ~10 calendar days for all 15 in parallel (not sequential)
        cur_day += int(10 * 5 / 7)  # ~7 trading days
        if cur_day >= HORIZON: break
        # Funded phase: run as single account, then scale cash by 15
        result = sim_stagger_a_single(packs, rng, start_day=cur_day)
        n_attempts += 1
        # Scale single-account result by 15 (copy-trade)
        total_payouts += result["payouts"] * 15
        total_cash += result["cash"] * 15
        if result["busted"]: n_busts += 1
        if result["payouts"] >= 6: n_grads += 1
        cur_day = result["end_day"] + 1
    return dict(payouts=total_payouts, cash=total_cash, eval_cost=total_eval_cost,
                evals=total_evals, attempts=n_attempts, busts=n_busts, grads=n_grads)


def report_single(label, results):
    p = np.array([r["payouts"] for r in results])
    c = np.array([r["cash"] for r in results])
    b = np.array([r["busted"] for r in results])
    p_any = (p >= 1).mean()
    p_2 = (p >= 2).mean()
    p_grad = (p >= 6).mean()
    first = [r["days_at_payouts"][0] for r in results if r["days_at_payouts"]]
    busts = [r["bust_day"] for r in results if r["busted"]]
    print(f"\n=== {label} (single account, stagger A) ===")
    print(f"  P(any payout): {p_any:.0%}   P(2+): {p_2:.0%}   P(graduate): {p_grad:.0%}")
    print(f"  Mean payouts/yr: {p.mean():.2f}   Mean cash/yr: ${c.mean():,.0f}")
    print(f"  Bust rate: {b.mean():.0%}")
    if busts:
        print(f"  Median days to bust: {int(np.median(busts))}d")
    if first:
        print(f"  Days to 1st payout (when achieved): median {int(np.median(first))}d, mean {np.mean(first):.0f}d")


def report_portfolio(label, results):
    p = np.array([r["payouts"] for r in results])
    c = np.array([r["cash"] for r in results])
    ec = np.array([r["eval_cost"] for r in results])
    ev = np.array([r["evals"] for r in results])
    att = np.array([r["attempts"] for r in results])
    busts = np.array([r["busts"] for r in results])
    grads = np.array([r["grads"] for r in results])
    net = c - ec
    print(f"\n=== {label} (15 copy-traded, eval refresh, full year) ===")
    print(f"  Attempts per year: mean {att.mean():.1f}   busts {busts.mean():.1f}   grads {grads.mean():.1f}")
    print(f"  Total payouts (across 15): mean {p.mean():.0f}")
    print(f"  Eval count: mean {ev.mean():.0f}   cost ${ec.mean():,.0f}")
    print(f"  Annual gross: mean ${c.mean():,.0f}")
    print(f"  Annual NET: mean ${net.mean():,.0f}   median ${np.median(net):,.0f}")
    print(f"    p25: ${np.percentile(net,25):,.0f}   p75: ${np.percentile(net,75):,.0f}")
    print(f"  Monthly NET: mean ${net.mean()/12:,.0f}")


def main():
    print("OPPOSITE MIX: OD=2 MNQ, B2=1 MNQ, RV=1 MNQ. OD marti OFF.")
    print(f"Lucid Flex 50K, stagger A ($1500 first @ $3K, $1000 subseq @ $2K).")
    print(f"{N_SIMS} sims.")

    packs_opp = load_packs_mixed({"OD": 2, "B2": 1, "RV": 1})

    # Single account
    rng = np.random.default_rng(seed=7001)
    res_single = [sim_stagger_a_single(packs_opp, rng) for _ in range(N_SIMS)]
    report_single("OPPOSITE MIX (OD=2, B2=1, RV=1)", res_single)

    # 15 copy-traded with refresh
    rng = np.random.default_rng(seed=7002)
    res_15 = [sim_15_copy_trade_with_refresh(packs_opp, rng) for _ in range(N_SIMS)]
    report_portfolio("OPPOSITE MIX 15 accts copy-traded + refresh", res_15)

    # Comparisons
    print("\n----- BASELINES FOR COMPARISON -----")

    packs_1mnq = load_packs_mixed({"OD": 1, "B2": 1, "RV": 1})
    rng = np.random.default_rng(seed=7003)
    res_1 = [sim_stagger_a_single(packs_1mnq, rng) for _ in range(N_SIMS)]
    report_single("ALL 1 MNQ baseline", res_1)
    rng = np.random.default_rng(seed=7004)
    res_1_15 = [sim_15_copy_trade_with_refresh(packs_1mnq, rng) for _ in range(N_SIMS)]
    report_portfolio("ALL 1 MNQ baseline (15 copy + refresh)", res_1_15)


if __name__ == "__main__":
    main()

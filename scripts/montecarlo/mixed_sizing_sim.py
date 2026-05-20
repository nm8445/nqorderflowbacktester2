"""Mixed per-strategy MNQ sizing — OD 1 MNQ, B2 2 MNQ, RV 2 MNQ. Marti OFF.
   Single account + 15 copy-traded portfolio. Stagger A payouts."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
OD_RAW_CSV = ROOT / "live" / "overnight drift" / "trades.csv"

N_SIMS = 10_000
HORIZON = 252
FUTURES_COST = 2.0


def load_packs_mixed(mnq_per_strat):
    """Returns day packs where each trade is (pnl_scaled, mae_scaled) using per-strat mnq."""
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"]).reset_index(drop=True)
    # Disable OD marti: halve OD trades that had qty=2
    od_raw = pd.read_csv(OD_RAW_CSV)
    od_raw["entry_time"] = pd.to_datetime(od_raw["entry_time"], utc=True, format="mixed")
    qty_map = dict(zip(od_raw["entry_time"], od_raw["qty"]))
    df["qty"] = 1
    for i in df.index[df["strat"] == "OD"]:
        df.at[i, "qty"] = qty_map.get(df.at[i, "entry_ts"], 1)
    marti_scale = np.where((df["strat"] == "OD") & (df["qty"] == 2), 0.5, 1.0)
    df["pnl_$"] = df["pnl_$"] * marti_scale
    df["mae_$"] = df["mae_$"] * marti_scale
    # Apply per-strategy MNQ scaling
    mnq_arr = df["strat"].map(mnq_per_strat).to_numpy()
    scale = mnq_arr / 10.0
    df["pnl_scaled"] = df["pnl_$"] * scale - FUTURES_COST * mnq_arr
    df["mae_scaled"] = df["mae_$"] * scale - FUTURES_COST * mnq_arr
    packs = []
    for date, grp in df.groupby("date", sort=True):
        trades = [(r["pnl_scaled"], r["mae_scaled"]) for _, r in grp.iterrows()]
        packs.append(trades)
    return packs


def sim_stagger_a(packs, rng, max_payouts=6):
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
    n_packs = len(packs)
    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        realized = 0.0
        for ps, ms in trades:
            if bal + realized + ms < floor:
                return dict(payouts=payouts, cash=cash, days_at_payouts=days_at_payouts,
                            busted=True, bust_day=d)
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
                if payouts >= max_payouts: break
    return dict(payouts=payouts, cash=cash, days_at_payouts=days_at_payouts,
                busted=False, bust_day=None)


def report(label, results):
    p = np.array([r["payouts"] for r in results])
    c = np.array([r["cash"] for r in results])
    b = np.array([r["busted"] for r in results])
    p_any = (p >= 1).mean()
    p_2 = (p >= 2).mean()
    p_3 = (p >= 3).mean()
    p_4 = (p >= 4).mean()
    p_5 = (p >= 5).mean()
    p_6 = (p >= 6).mean()
    first = [r["days_at_payouts"][0] for r in results if r["days_at_payouts"]]
    between = []
    for r in results:
        d = r["days_at_payouts"]
        for i in range(1, len(d)):
            between.append(d[i] - d[i-1])
    busts = [r["bust_day"] for r in results if r["busted"]]
    print(f"\n=== {label} ===")
    print(f"  P(any payout): {p_any:.0%}   P(2+): {p_2:.0%}   P(3+): {p_3:.0%}")
    print(f"  P(4+): {p_4:.0%}   P(5+): {p_5:.0%}   P(graduate at 6): {p_6:.0%}")
    print(f"  Mean payouts/yr: {p.mean():.2f}   Mean cash/yr: ${c.mean():,.0f}")
    print(f"  Median cash: ${np.median(c):,.0f}   p25: ${np.percentile(c,25):,.0f}   p75: ${np.percentile(c,75):,.0f}")
    print(f"  Bust rate (any time): {b.mean():.0%}")
    if busts: print(f"  Median days to bust (when busted): {int(np.median(busts))}d")
    if first:
        print(f"  Days to 1st payout (when achieved): median {int(np.median(first))}d   mean {np.mean(first):.0f}d")
        print(f"    p25: {int(np.percentile(first,25))}d   p75: {int(np.percentile(first,75))}d")
    if between:
        print(f"  Days BETWEEN payouts: median {int(np.median(between))}d   mean {np.mean(between):.0f}d")


def main():
    print("MIXED SIZING — OD 1 MNQ, B2 2 MNQ, RV 2 MNQ (OD marti OFF). Stagger A.")
    print(f"Lucid Flex 50K. {N_SIMS} sims.")

    # Mixed: OD=1, B2=2, RV=2
    packs_mixed = load_packs_mixed({"OD": 1, "B2": 2, "RV": 2})
    rng = np.random.default_rng(seed=6001)
    res_mixed = [sim_stagger_a(packs_mixed, rng) for _ in range(N_SIMS)]
    report("MIXED: OD=1, B2=2, RV=2 MNQ", res_mixed)

    # For comparison
    packs_all1 = load_packs_mixed({"OD": 1, "B2": 1, "RV": 1})
    rng = np.random.default_rng(seed=6002)
    res_all1 = [sim_stagger_a(packs_all1, rng) for _ in range(N_SIMS)]
    report("ALL 1 MNQ baseline", res_all1)

    packs_all2 = load_packs_mixed({"OD": 2, "B2": 2, "RV": 2})
    rng = np.random.default_rng(seed=6003)
    res_all2 = [sim_stagger_a(packs_all2, rng) for _ in range(N_SIMS)]
    report("ALL 2 MNQ comparison", res_all2)


if __name__ == "__main__":
    main()

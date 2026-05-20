"""Single account 2 MNQ no-marti stagger A — P(payout before bust) + cycle stats."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
OD_RAW_CSV = ROOT / "live" / "overnight drift" / "trades.csv"

N_SIMS = 10_000
HORIZON = 252


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


def sim_stagger_a(packs, mnq, rng, max_payouts=6):
    """$1,500 at first $3K, $1,000 at subsequent $2K. Returns dict."""
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
        for pnl, mae in trades:
            ps = pnl * (mnq/10) - 2.0 * mnq
            ms = mae * (mnq/10) - 2.0 * mnq
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
                if cycle >= 3000:
                    gross = 1500
                    stagger_first = True
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
                days_at_payouts.append(d + 1)
                qual = 0
                cycle = 0.0
                if payouts >= max_payouts:
                    break
    return dict(payouts=payouts, cash=cash, days_at_payouts=days_at_payouts,
                busted=False, bust_day=None)


def main():
    packs = load_packs()
    print("Single account 2 MNQ no marti, STAGGER A ($1,500 first @ $3K, $1,000 subseq @ $2K)")
    print(f"Lucid Flex 50K rules. {N_SIMS} sims.\n")

    rng = np.random.default_rng(seed=2025)
    results = [sim_stagger_a(packs, 2, rng) for _ in range(N_SIMS)]

    payouts = np.array([r["payouts"] for r in results])
    cash = np.array([r["cash"] for r in results])
    busts = np.array([r["busted"] for r in results])

    p_any = np.mean(payouts >= 1)
    p_2 = np.mean(payouts >= 2)
    p_3 = np.mean(payouts >= 3)
    p_4 = np.mean(payouts >= 4)
    p_5 = np.mean(payouts >= 5)
    p_6 = np.mean(payouts >= 6)

    first_days = [r["days_at_payouts"][0] for r in results if r["days_at_payouts"]]
    between_days = []
    for r in results:
        d = r["days_at_payouts"]
        for i in range(1, len(d)):
            between_days.append(d[i] - d[i-1])
    bust_days = [r["bust_day"] for r in results if r["busted"]]

    print("=== Payout distribution ===")
    print(f"  P(0 payouts, busted before any): {1-p_any:.0%}")
    print(f"  P(at least 1 payout): {p_any:.0%}")
    print(f"  P(at least 2): {p_2:.0%}")
    print(f"  P(at least 3): {p_3:.0%}")
    print(f"  P(at least 4): {p_4:.0%}")
    print(f"  P(at least 5): {p_5:.0%}")
    print(f"  P(all 6, graduation): {p_6:.0%}")
    print(f"\n  Mean payouts per year: {payouts.mean():.2f}")
    print(f"  Median payouts: {int(np.median(payouts))}")

    print(f"\n=== Cash per account ===")
    print(f"  Mean annual cash: ${cash.mean():,.0f}")
    print(f"  Median: ${np.median(cash):,.0f}")
    print(f"  p25: ${np.percentile(cash,25):,.0f}   p75: ${np.percentile(cash,75):,.0f}")

    print(f"\n=== Timing ===")
    print(f"  Bust rate: {busts.mean():.0%}")
    if bust_days:
        print(f"  Median days to bust (when busted): {int(np.median(bust_days))}d")
    if first_days:
        print(f"  Days to 1st payout (when achieved): median {int(np.median(first_days))}d, "
              f"mean {np.mean(first_days):.0f}d")
        print(f"    p25: {int(np.percentile(first_days,25))}d   p75: {int(np.percentile(first_days,75))}d")
    if between_days:
        print(f"  Days BETWEEN payouts: median {int(np.median(between_days))}d, "
              f"mean {np.mean(between_days):.0f}d")


if __name__ == "__main__":
    main()

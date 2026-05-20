"""FP 100K funded with $1,900 hard SL per trade (avoid 2% rule).
   Tests 1-4 MNQ. Single account funded behavior."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
OD_RAW_CSV = ROOT / "live" / "overnight drift" / "trades.csv"

N_SIMS = 10_000
HORIZON = 252
CFD_COST = 6.0   # FP uses NAS100 CFD, $6/RT/MNQ blended
HARD_SL_AT_TRADER = 1_900.0  # $1,900 max loss per trade

# FP 100K funded
START = 100_000.0
FLOOR = 90_000.0     # $10K static DD
DLL = 5_000.0        # $5K daily loss limit
TRADE_RULE = 2_000.0 # 2% per-trade rule (won't trigger with our $1,900 cap)
CYCLE_TD = 10        # bi-weekly = ~10 trading days
SPLIT = 0.80         # bi-weekly 80%
MIN_WD = 2_000.0     # 2% of balance


def load_packs_with_hard_sl(mnq_trading, disable_od_marti=True):
    """Apply per-trade $1,900 hard SL at user's MNQ trading size."""
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"]).reset_index(drop=True)

    if disable_od_marti:
        od_raw = pd.read_csv(OD_RAW_CSV)
        od_raw["entry_time"] = pd.to_datetime(od_raw["entry_time"], utc=True, format="mixed")
        qty_map = dict(zip(od_raw["entry_time"], od_raw["qty"]))
        df["qty"] = 1
        for i in df.index[df["strat"] == "OD"]:
            df.at[i, "qty"] = qty_map.get(df.at[i, "entry_ts"], 1)
        scale = np.where((df["strat"] == "OD") & (df["qty"] == 2), 0.5, 1.0)
        df["pnl_$"] = df["pnl_$"] * scale
        df["mae_$"] = df["mae_$"] * scale

    # Hard SL: at 1 NQ basis, threshold = HARD_SL / (mnq/10) = HARD_SL * 10 / mnq
    sl_at_1nq = HARD_SL_AT_TRADER * (10.0 / mnq_trading)
    breached = df["mae_$"] < -sl_at_1nq
    df.loc[breached, "pnl_$"] = -sl_at_1nq
    df.loc[breached, "mae_$"] = -sl_at_1nq

    n_breached = breached.sum()

    return ([[(r["pnl_$"], r["mae_$"]) for _, r in grp.iterrows()]
             for _, grp in df.groupby("date", sort=True)], n_breached, len(df))


def sim_fp100k(packs, mnq, rng):
    """Single FP 100K funded account, bi-weekly 80% payout.
       Returns dict with payouts, cash, days_to_1st, busted, bust_reason."""
    bal = START
    busted = False
    bust_reason = None
    bust_day = None
    days_since_cycle = 0
    payouts = 0
    cash = 0.0
    days_to_1st = None
    n_packs = len(packs)

    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        daily_realized = 0.0
        for pnl, mae in trades:
            ps = pnl * (mnq/10) - CFD_COST * mnq
            ms = mae * (mnq/10) - CFD_COST * mnq
            # 2% rule check (with hard SL, this should never trigger)
            if abs(ms) > TRADE_RULE:
                busted = True
                bust_reason = "trade_rule"
                bust_day = d
                break
            # Intraday DLL check
            if daily_realized + ms <= -DLL:
                busted = True
                bust_reason = "DLL"
                bust_day = d
                break
            # Intraday MaxLoss check
            if bal + daily_realized + ms < FLOOR:
                busted = True
                bust_reason = "MaxLoss"
                bust_day = d
                break
            daily_realized += ps
        if busted: break
        bal += daily_realized
        days_since_cycle += 1
        if days_since_cycle >= CYCLE_TD:
            cycle_profit = bal - START
            if cycle_profit >= MIN_WD:
                gross = cycle_profit  # withdraw all profit
                trader = gross * SPLIT
                bal -= gross
                payouts += 1
                cash += trader
                if days_to_1st is None:
                    days_to_1st = d
            days_since_cycle = 0

    return dict(busted=busted, bust_reason=bust_reason, bust_day=bust_day,
                payouts=payouts, cash=cash, days_to_1st=days_to_1st, final_bal=bal)


def main():
    print("FP 100K FUNDED with $1,900 hard SL per trade (avoid 2% rule breach)")
    print(f"Bi-weekly 80% payout cycle, $5K DLL, $10K static DD")
    print(f"CFD cost ${CFD_COST}/RT/MNQ, {N_SIMS} sims, OD marti OFF\n")

    rows = []
    for mnq in [1, 2, 3, 4]:
        packs, n_breached, n_total = load_packs_with_hard_sl(mnq)
        print(f"\n--- {mnq} MNQ (hard SL clips {n_breached}/{n_total} = {100*n_breached/n_total:.1f}% of trades) ---")
        rng = np.random.default_rng(seed=8000 + mnq)
        sims = [sim_fp100k(packs, mnq, rng) for _ in range(N_SIMS)]

        bust_rate = np.mean([s["busted"] for s in sims])
        bust_reasons = {}
        for s in sims:
            if s["busted"]:
                bust_reasons[s["bust_reason"]] = bust_reasons.get(s["bust_reason"], 0) + 1
        any_pmt = np.mean([s["payouts"] >= 1 for s in sims])
        pmts = np.array([s["payouts"] for s in sims])
        cash = np.array([s["cash"] for s in sims])
        t1 = [s["days_to_1st"] for s in sims if s["days_to_1st"] is not None]

        print(f"  P(any payout): {any_pmt:.0%}   Bust rate: {bust_rate:.0%}")
        print(f"  Bust reasons: " + ", ".join(f"{k}={v}" for k,v in bust_reasons.items()))
        print(f"  Mean payouts/yr: {pmts.mean():.1f}   median {int(np.median(pmts))}")
        print(f"  Mean annual cash: ${cash.mean():,.0f}   median ${np.median(cash):,.0f}")
        print(f"    p25 ${np.percentile(cash,25):,.0f}   p75 ${np.percentile(cash,75):,.0f}")
        if t1:
            print(f"  Days to 1st payout: median {int(np.median(t1))}d   mean {np.mean(t1):.0f}d")
        rows.append({"mnq": mnq, "P_payout": any_pmt, "bust": bust_rate,
                     "mean_payouts": pmts.mean(), "mean_cash": cash.mean(),
                     "days_to_1st_median": int(np.median(t1)) if t1 else None})

    print("\n=== SUMMARY ===")
    print(f"{'MNQ':>4} {'P(payout)':>10} {'Bust':>8} {'Payouts/yr':>12} {'Mean cash':>12} {'Days to 1st':>13}")
    for r in rows:
        d1 = f"{r['days_to_1st_median']}d" if r['days_to_1st_median'] else "n/a"
        print(f"{r['mnq']:>4} {r['P_payout']*100:>9.0f}% {r['bust']*100:>7.0f}% "
              f"{r['mean_payouts']:>12.1f} ${r['mean_cash']:>10,.0f} {d1:>13}")


if __name__ == "__main__":
    main()

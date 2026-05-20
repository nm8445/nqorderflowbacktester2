"""
Lucid Flex 50K with OD martingale disabled on prop.
=====================================================
OD's worst MAE comes from 2c martingale recovery trades. If we keep OD entry/exit
logic but force all OD trades to 1c, the tail halves while keeping trade cadence
(so the 5-profitable-day payout gate fills faster than RV+B2-only).

Compares 4 scenarios at 1 MNQ on Lucid Flex 50K:
  A) Full stack (OD martingale ON)             — baseline
  B) Full stack with OD at 1c always (no marti) — middle path
  C) RV + B2 only (drop OD entirely)           — safer but slower
  D) Full stack but OD skipped when prior OD was a loss — alt middle path
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
OD_RAW_CSV = ROOT / "live" / "overnight drift" / "trades.csv"

N_SIMS = 5_000
HORIZON = 252
FUTURES_COST = 2.0

LUCID = dict(
    start=50_000, floor_init=48_000,
    lock_after=53_000, lock_floor=50_000,
    payout_cap=2000, max_payouts=6, split=0.90,
)


def load_packs_modified(mode: str):
    """mode in {'full', 'no_marti', 'rvb2_only', 'skip_after_loss'}"""
    df = pd.read_csv(TRADES_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"]).reset_index(drop=True)

    if mode == "rvb2_only":
        df = df[df["strat"].isin(["RV", "B2"])].copy()

    elif mode == "no_marti":
        # OD trades with qty=2 need their pnl/mae halved to 1c equivalent.
        # We pull qty from the raw OD log.
        od_raw = pd.read_csv(OD_RAW_CSV)
        od_raw["entry_time"] = pd.to_datetime(od_raw["entry_time"], utc=True, format="mixed")
        qty_map = dict(zip(od_raw["entry_time"], od_raw["qty"]))
        df["qty"] = 1
        for i in df.index[df["strat"] == "OD"]:
            ts = df.at[i, "entry_ts"]
            q = qty_map.get(ts, 1)
            df.at[i, "qty"] = q
        # halve pnl_$ and mae_$ on OD 2c trades
        scale_factor = np.where((df["strat"] == "OD") & (df["qty"] == 2), 0.5, 1.0)
        df["pnl_$"] = df["pnl_$"] * scale_factor
        df["mae_$"] = df["mae_$"] * scale_factor

    elif mode == "skip_after_loss":
        # Drop OD trades that follow an OD loss (the recovery trades)
        od_raw = pd.read_csv(OD_RAW_CSV)
        od_raw["entry_time"] = pd.to_datetime(od_raw["entry_time"], utc=True, format="mixed")
        od_raw = od_raw.sort_values("entry_time").reset_index(drop=True)
        # An OD trade is a "recovery" iff prev OD trade was a loss
        skip_ts = set()
        for i in range(1, len(od_raw)):
            if od_raw.at[i-1, "pnl_dollars"] < 0:
                skip_ts.add(od_raw.at[i, "entry_time"])
        # Also scale recovery trades back to 1c (in case some weren't 2c but mark as skip anyway)
        df = df[~((df["strat"] == "OD") & (df["entry_ts"].isin(skip_ts)))].copy()

    packs = []
    for date, grp in df.groupby("date", sort=True):
        trades = [(r["strat"], r["pnl_$"], r["mae_$"]) for _, r in grp.iterrows()]
        packs.append((date, trades))
    return packs, len(df)


def simulate(packs, mnq, rng, cost=FUTURES_COST):
    scale = mnq / 10.0
    cfg = LUCID
    balance = cfg["start"]
    floor = cfg["floor_init"]
    hwm = balance
    locked = False
    busted = False
    bust_reason = None
    payouts = 0
    cash = 0.0
    days_to_1st = None
    cycle_qual_days = 0
    cycle_profit = 0.0
    stagger_first_done = False
    n_packs = len(packs)

    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        _, trades = packs[idx]
        daily_realized = 0.0
        for strat, pnl, mae in trades:
            pnl_scaled = pnl * scale - cost * mnq
            mae_scaled = mae * scale - cost * mnq
            check = balance + daily_realized + mae_scaled
            if check < floor:
                busted = True
                bust_reason = "MaxLoss_intraday"
                break
            daily_realized += pnl_scaled
        if busted:
            break
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
                if days_to_1st is None: days_to_1st = d
                cycle_qual_days = 0; cycle_profit = 0.0
                if payouts >= cfg["max_payouts"]:
                    break
    return dict(busted=busted, payouts=payouts, cash=cash, days_to_1st=days_to_1st)


def run(mode_name: str, mode_key: str, mnq: int = 1):
    packs, n_trades = load_packs_modified(mode_key)
    rng = np.random.default_rng(seed=hash(mode_name + str(mnq)) % 99991)
    sims = [simulate(packs, mnq, rng) for _ in range(N_SIMS)]
    br = np.mean([s["busted"] for s in sims])
    any_p = np.mean([s["payouts"] >= 1 for s in sims])
    pmts = [s["payouts"] for s in sims]
    cash = [s["cash"] for s in sims]
    t1 = [s["days_to_1st"] for s in sims if s["days_to_1st"] is not None]
    return {
        "mode": mode_name,
        "n_packs": len(packs),
        "n_trades": n_trades,
        "bust_rate": br,
        "any_payout": any_p,
        "median_pmts": int(np.median(pmts)),
        "median_cash_$": np.median(cash),
        "mean_cash_$": np.mean(cash),
        "median_d_to_1st": int(np.median(t1)) if t1 else None,
    }


def check_mae_distribution():
    """Show how max MAE changes under each modification."""
    df = pd.read_csv(TRADES_CSV)
    od_raw = pd.read_csv(OD_RAW_CSV)
    od_raw["entry_time"] = pd.to_datetime(od_raw["entry_time"], utc=True, format="mixed")
    qty_map = dict(zip(od_raw["entry_time"], od_raw["qty"]))
    od_raw_sorted = od_raw.sort_values("entry_time").reset_index(drop=True)
    skip_ts = set()
    for i in range(1, len(od_raw_sorted)):
        if od_raw_sorted.at[i-1, "pnl_dollars"] < 0:
            skip_ts.add(od_raw_sorted.at[i, "entry_time"])

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    # Full stack
    full_worst = df["mae_$"].min()
    # No marti
    df2 = df.copy()
    df2["qty"] = 1
    for i in df2.index[df2["strat"] == "OD"]:
        df2.at[i, "qty"] = qty_map.get(df2.at[i, "entry_ts"], 1)
    scale = np.where((df2["strat"] == "OD") & (df2["qty"] == 2), 0.5, 1.0)
    no_marti_mae = df2["mae_$"] * scale
    no_marti_worst = no_marti_mae.min()
    # Skip after loss
    skip_df = df[~((df["strat"] == "OD") & (df["entry_ts"].isin(skip_ts)))]
    skip_worst = skip_df["mae_$"].min()
    # RV+B2 only
    rvb2 = df[df["strat"].isin(["RV", "B2"])]
    rvb2_worst = rvb2["mae_$"].min()
    print("=== Worst MAE under each modification (at 1 NQ) ===")
    print(f"  Full stack (OD marti ON):           ${full_worst:>9,.0f}")
    print(f"  OD marti OFF (always 1c):           ${no_marti_worst:>9,.0f}")
    print(f"  Skip OD after OD loss:              ${skip_worst:>9,.0f}")
    print(f"  RV + B2 only:                       ${rvb2_worst:>9,.0f}")
    print(f"  At 1 MNQ scaled (Lucid $2K floor):  full=${full_worst*0.1:>+7,.0f}, "
          f"no_marti=${no_marti_worst*0.1:>+7,.0f}, "
          f"skip=${skip_worst*0.1:>+7,.0f}, "
          f"rvb2=${rvb2_worst*0.1:>+7,.0f}")


def main():
    check_mae_distribution()
    print()

    rows = []
    for label, key in [
        ("A. Full stack (marti ON)",      "full"),
        ("B. OD marti OFF (always 1c)",   "no_marti"),
        ("C. RV+B2 only (drop OD)",       "rvb2_only"),
        ("D. Skip OD after OD loss",      "skip_after_loss"),
    ]:
        r = run(label, key, mnq=1)
        rows.append(r)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("=== Lucid 50K, 1 MNQ stagger A, MAE-aware (5,000 sims, 252-day horizon) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    print("\n=== EV summary (mean cash minus $500 hedge cost) ===")
    for r in rows:
        print(f"  {r['mode']:<32}  mean_$=${r['mean_cash_$']:>7,.0f}  "
              f"EV=${r['mean_cash_$']-500:>+7,.0f}  bust={r['bust_rate']*100:>4.1f}%")


if __name__ == "__main__":
    main()

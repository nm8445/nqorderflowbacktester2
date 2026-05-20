"""
Hard $1,500 SL cap impact analysis.

For each trade: if MAE (worst floating loss) was worse than -$1,500, the trade
would have been stopped out at -$1,500. So replace realized pnl with -$1,500.
Otherwise keep original pnl (trade exit was within the new SL boundary).

This approximates the impact of adding a hard dollar stop to each strategy
while keeping the original TP logic intact (only the SL is changed).

Compares:
  - Original strategy stats vs hard-$1500-SL stats per strategy
  - Combined-stack stats both ways
  - Prop firm sim outcomes both ways
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"

HARD_SL = 1_500.0   # dollar cap at 1 NQ basis
N_SIMS = 5_000
HORIZON = 252
FUTURES_COST = 2.0


def apply_hard_sl(df, sl_cap_at_trading_size, mnq_trading):
    """Return modified df where 1-NQ-basis pnl is replaced if a $sl_cap loss at
    the trader's actual position size (mnq_trading) would have been triggered.

    At 1 MNQ trading, hard SL of $1,500 = scaled equivalent of $15,000 at 1 NQ basis.
    At 1 NQ trading (= 10 MNQ), hard SL of $1,500 = $1,500 at 1 NQ basis.
    Conversion: sl_cap_at_1nq = sl_cap_at_trading * (10 / mnq_trading)
    """
    out = df.copy()
    sl_cap_at_1nq = sl_cap_at_trading_size * (10.0 / mnq_trading)
    breached = out["mae_$"] < -sl_cap_at_1nq
    out.loc[breached, "pnl_$"] = -sl_cap_at_1nq
    out.loc[breached, "mae_$"] = -sl_cap_at_1nq
    return out, breached.sum(), sl_cap_at_1nq


def stats(s):
    pos = s[s > 0]
    neg = s[s < 0]
    cum = s.cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    return {
        "trades": len(s),
        "win_pct": (s > 0).mean() * 100,
        "gross_$": s.sum(),
        "avg_$": s.mean(),
        "best_$": s.max(),
        "worst_$": s.min(),
        "avg_win_$": pos.mean() if len(pos) else 0,
        "avg_loss_$": neg.mean() if len(neg) else 0,
        "PF": pos.sum() / abs(neg.sum()) if len(neg) else float("inf"),
        "max_DD_$": dd,
        "G/DD": abs(s.sum() / dd) if dd < 0 else float("inf"),
    }


def simulate_lucid_50k(packs, mnq, rng):
    """MAE-aware Lucid 50K Flex sim with stagger. Returns (busted, cash, days_to_1st_payout, n_payouts)."""
    scale = mnq / 10.0
    balance = 50_000.0
    floor = 48_000.0
    hwm = balance
    locked = False
    qual_days = 0
    cycle_profit = 0.0
    stagger_first_done = False
    payouts = 0
    cash = 0.0
    days_to_1st = None
    n_packs = len(packs)
    busted = False
    for d in range(HORIZON):
        idx = rng.integers(0, n_packs)
        trades = packs[idx]
        daily_realized = 0.0
        for pnl, mae in trades:
            pnl_scaled = pnl * scale - FUTURES_COST * mnq
            mae_scaled = mae * scale - FUTURES_COST * mnq
            if balance + daily_realized + mae_scaled < floor:
                busted = True
                break
            daily_realized += pnl_scaled
        if busted:
            break
        balance += daily_realized
        if not locked:
            if balance > hwm: hwm = balance
            floor = max(48_000.0, hwm - 2000)
            if hwm >= 53_000.0:
                locked = True
                floor = 50_000.0
        if daily_realized >= 150:
            qual_days += 1
        cycle_profit += daily_realized
        if payouts < 6 and qual_days >= 5 and cycle_profit > 0:
            gross = 0.0
            if not stagger_first_done:
                if cycle_profit >= 3000:
                    gross = 1500
                    stagger_first_done = True
            else:
                if cycle_profit >= 2000:
                    gross = 1000
            if gross > 0:
                trader = gross * 0.9
                balance -= gross
                if not locked:
                    hwm = max(50_000.0, hwm - gross)
                    floor = max(48_000.0, hwm - 2000)
                payouts += 1
                cash += trader
                if days_to_1st is None:
                    days_to_1st = d
                qual_days = 0
                cycle_profit = 0.0
                if payouts >= 6:
                    break
    return busted, cash, days_to_1st, payouts


def build_packs(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")
    df = df.sort_values(["date", "entry_ts"])
    return [
        [(r["pnl_$"], r["mae_$"]) for _, r in grp.iterrows()]
        for _, grp in df.groupby("date", sort=True)
    ]


def main():
    df = pd.read_csv(TRADES_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed")

    print("=" * 95)
    print(f"HARD $1,500 SL CAP (per-trade max dollar loss at the trader's actual size)")
    print(f"For each MNQ size, the hard SL acts as an absolute floor on per-trade loss.")
    print("=" * 95)

    # For each MNQ trading size, simulate with the appropriate hard SL
    for mnq_trading in [1, 2, 3]:
        sl_cap_at_1nq = HARD_SL * (10.0 / mnq_trading)
        print(f"\n========== Trader sizing = {mnq_trading} MNQ. ${HARD_SL:,.0f} hard SL = "
              f"${sl_cap_at_1nq:,.0f} at 1 NQ basis ==========")
        df_hard, n_breached, _ = apply_hard_sl(df, HARD_SL, mnq_trading)

        # Per-strategy comparison
        for strat in ["RV", "B2", "OD"]:
            orig = df[df["strat"] == strat]["pnl_$"]
            hard = df_hard[df_hard["strat"] == strat]["pnl_$"]
            strat_breached = ((df["strat"] == strat) & (df["mae_$"] < -sl_cap_at_1nq)).sum()
            n_trades = len(orig)
            so = stats(orig)
            sh = stats(hard)
            print(f"\n  {strat} ({n_trades} trades, {strat_breached} breach hard SL = "
                  f"{100*strat_breached/n_trades:.1f}%):")
            print(f"    Gross $:  ${so['gross_$']:>10,.0f} -> ${sh['gross_$']:>10,.0f} "
                  f"(delta ${sh['gross_$']-so['gross_$']:+,.0f})")
            print(f"    Max DD :  ${so['max_DD_$']:>10,.0f} -> ${sh['max_DD_$']:>10,.0f}")
            print(f"    Worst  :  ${so['worst_$']:>10,.0f} -> ${sh['worst_$']:>10,.0f}")
            print(f"    PF     :  {so['PF']:>10.2f} -> {sh['PF']:>10.2f}")

        # Combined
        so = stats(df["pnl_$"])
        sh = stats(df_hard["pnl_$"])
        total_breached = (df["mae_$"] < -sl_cap_at_1nq).sum()
        print(f"\n  COMBINED ({total_breached} of {len(df)} = {100*total_breached/len(df):.1f}% breach):")
        print(f"    Gross $:  ${so['gross_$']:>10,.0f} -> ${sh['gross_$']:>10,.0f} "
              f"(delta ${sh['gross_$']-so['gross_$']:+,.0f}, {100*(sh['gross_$']-so['gross_$'])/so['gross_$']:+.1f}%)")
        print(f"    Max DD :  ${so['max_DD_$']:>10,.0f} -> ${sh['max_DD_$']:>10,.0f}")
        print(f"    Worst  :  ${so['worst_$']:>10,.0f} -> ${sh['worst_$']:>10,.0f}")
        print(f"    PF     :  {so['PF']:>10.2f} -> {sh['PF']:>10.2f}")

        # Run Lucid 50K Flex prop firm sim with both versions AT THIS MNQ size
        packs_orig = build_packs(df)
        packs_hard = build_packs(df_hard)
        rng = np.random.default_rng(seed=mnq_trading * 100)
        sims_orig = [simulate_lucid_50k(packs_orig, mnq_trading, rng) for _ in range(N_SIMS)]
        rng = np.random.default_rng(seed=mnq_trading * 100 + 50)
        sims_hard = [simulate_lucid_50k(packs_hard, mnq_trading, rng) for _ in range(N_SIMS)]
        print(f"\n  LUCID 50K SIM @ {mnq_trading} MNQ:")
        for label, sims in [("ORIGINAL    ", sims_orig), ("+HARD $1500 SL", sims_hard)]:
            bust = np.mean([s[0] for s in sims])
            cash = np.array([s[1] for s in sims])
            t1 = [s[2] for s in sims if s[2] is not None]
            pmts = np.array([s[3] for s in sims])
            any_p = np.mean(pmts >= 1)
            print(f"    {label}  bust={bust*100:>5.1f}%  any_pmt={any_p*100:>5.1f}%  "
                  f"mean_cash=${cash.mean():>7,.0f}  median_cash=${np.median(cash):>7,.0f}  "
                  f"median_d_to_1st={int(np.median(t1)) if t1 else 'n/a':>4}d  "
                  f"median_pmts={int(np.median(pmts))}")


if __name__ == "__main__":
    main()

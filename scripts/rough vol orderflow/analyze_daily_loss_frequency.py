"""
For the 3-strategy combined portfolio, compute the bar-by-bar account equity
(including open-position MAE) and analyze how often daily loss would breach
the $5k Hola Prime limit.

Outputs:
  1. Distribution of worst intraday excursion (vs prior EOD) per historical day
  2. % of days that would have triggered a $5k daily-loss bust at each MNQ size
  3. Per-strategy attribution: which strat was responsible for the worst dip
  4. Per-trade analysis of OD trades that contributed >$5k single-trade MAE
     -> these are the cutoff candidates
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

NQ_PT = 20.0
TZ = "America/New_York"

TIMEBARS_DIRS = [
    Path("D:/trading_pythonbacktest_data/timebars_5min_5yr"),
    Path("D:/trading_pythonbacktest_data/timebars_5min"),
]

COMBINED_MAE = "C:/trading/nqorderflowbacktester/scripts/rough vol orderflow/results/combined_3way_trades_with_mae.csv"
OUT_DIR = Path("C:/trading/nqorderflowbacktester/live/combined deployment plan")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_20min_bars():
    files_by_date = {}
    for d in TIMEBARS_DIRS:
        for f in sorted(d.glob("timebars_5min_202*.pkl")):
            files_by_date[f.stem] = f
    frames = []
    for stem in sorted(files_by_date.keys()):
        with open(files_by_date[stem], "rb") as fh:
            bars = pickle.load(fh)
        if not bars: continue
        rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
                 "low": b["low"], "close": b["close"]} for b in bars]
        df5 = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df5["group"] = df5.index.floor("20min")
        agg = df5.groupby("group").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
        )
        agg.index += pd.Timedelta(minutes=20)
        frames.append(agg)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.DatetimeIndex([
        (t.tz_convert(TZ) if hasattr(t, "tz_convert") and getattr(t, "tzinfo", None) is not None
         else pd.Timestamp(t).tz_localize("UTC").tz_convert(TZ))
        for t in df.index
    ])
    return df


def main():
    print("Building 20-min bars...")
    bars = build_20min_bars()
    print(f"  bars: {len(bars):,}")

    print("Loading MAE-augmented combined log...")
    df = pd.read_csv(COMBINED_MAE)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert(TZ)
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert(TZ)
    df = df.sort_values("entry_ts").reset_index(drop=True)

    # ----------------------------------------------------------
    # Simple daily-loss test using MAE per trade:
    # For each historical day, compute worst intraday equity excursion
    # at NQ basis (1 NQ contract).
    # We use a conservative model: per trade's MAE is "spent" during the trade.
    # On any day where a trade is open, the daily equity can dip by MAE.
    # ----------------------------------------------------------
    # Build a calendar of (date, list of (strat, mae_$, pnl_$)) where MAE counts
    # against the day the trade was open. Since trades can span midnight (OD: 19:00 -> 08:00),
    # we attribute MAE to the SECOND day (where the bottom usually occurs intraday).
    # Realized PnL is attributed to the exit day.
    df["mae_date"] = df["exit_ts"].dt.date  # use exit day for MAE event (most OD trades MAE overnight)
    df["pnl_date"] = df["exit_ts"].dt.date

    # Daily aggregate: list of trades exiting that day with mae and pnl
    print("Building daily MAE timeline...")
    daily_rows = []
    for d, g in df.groupby("pnl_date"):
        # Worst intraday excursion approximation: sum of MAEs (cumulative open exposure)
        # Plus realized PnL trajectory.
        # We approximate the worst point as: cumsum of MAEs across trades (sorted by entry_ts)
        # minus closed PnLs already realized.
        g = g.sort_values("entry_ts")
        # The "open position at any time" pattern is hard to reconstruct in 1 line.
        # Use sum of MAEs as the proxy for worst simultaneous unrealized loss:
        # if 2 trades active same day, both could be at MAE simultaneously in worst case.
        total_mae = float(g["mae_$"].sum())
        total_pnl = float(g["pnl_$"].sum())
        worst_excursion_nq = total_mae  # most pessimistic: all MAEs sum
        # least pessimistic: max single MAE (only one strat at MAE at a time)
        worst_single_nq = float(g["mae_$"].min())  # min because MAE is negative
        # PnL by strat
        strat_pnl = g.groupby("strat")["pnl_$"].sum().to_dict()
        strat_mae = g.groupby("strat")["mae_$"].sum().to_dict()
        daily_rows.append(dict(
            date=d, n_trades=len(g),
            total_mae_nq=total_mae,
            total_pnl_nq=total_pnl,
            worst_single_mae_nq=worst_single_nq,
            od_mae=strat_mae.get("OD", 0), rv_mae=strat_mae.get("RV", 0), b2_mae=strat_mae.get("B2", 0),
            od_pnl=strat_pnl.get("OD", 0), rv_pnl=strat_pnl.get("RV", 0), b2_pnl=strat_pnl.get("B2", 0),
        ))
    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    n_days = len(daily)
    print(f"  {n_days} trading days in combined log")

    # ----------------------------------------------------------
    # Q1: At each MNQ size, % of days that would have busted $5k daily loss
    # Use two excursion models:
    #   pessimistic: sum-of-MAEs (all open trades at worst simultaneously)
    #   realistic:   max-single-MAE (only the worst trade at its bottom)
    # ----------------------------------------------------------
    DAILY_LIMIT_MNQ = 5_000
    print("\n=== Q1: Daily-loss bust frequency by MNQ size ===\n")
    print("Conservative (sum-of-MAEs):       all open trades simultaneously at worst")
    print("Realistic (max-single-MAE):       only the single worst-MAE trade at its bottom")
    print()
    print(f"{'MNQ':>4} {'NAS lots':>9}  {'pessim_bust%':>13} {'realistic_bust%':>16}  "
          f"{'pessim_days':>12} {'real_days':>11}")
    for mnq in [0.5, 1, 2, 3, 4, 5, 6, 8, 10]:
        scale = mnq * 0.1
        pessim_excursion = daily["total_mae_nq"] * scale  # negative dollars
        real_excursion = daily["worst_single_mae_nq"] * scale
        pessim_bust = (pessim_excursion <= -DAILY_LIMIT_MNQ).sum()
        real_bust = (real_excursion <= -DAILY_LIMIT_MNQ).sum()
        print(f"{mnq:>4} {mnq*2:>9.1f}  {100*pessim_bust/n_days:>12.2f}% {100*real_bust/n_days:>15.2f}%  "
              f"{pessim_bust:>12} {real_bust:>11}")

    # ----------------------------------------------------------
    # Q2: Per-strategy attribution — which strat caused the daily-loss days?
    # ----------------------------------------------------------
    print("\n=== Q2: Per-strategy worst-MAE attribution ===\n")
    # Filter to "would-bust" days at MNQ=2 (a common safe size)
    for mnq_test in [2, 3, 4]:
        scale = mnq_test * 0.1
        bust_mask_real = (daily["worst_single_mae_nq"] * scale) <= -DAILY_LIMIT_MNQ
        bust_days = daily[bust_mask_real]
        if len(bust_days) == 0:
            print(f"MNQ={mnq_test}: no daily-loss-bust days (realistic model)")
            continue
        # Find which strat's MAE was the largest contributor on those days
        od_worst = (bust_days["od_mae"] <= bust_days["rv_mae"]) & (bust_days["od_mae"] <= bust_days["b2_mae"])
        rv_worst = (bust_days["rv_mae"] <= bust_days["od_mae"]) & (bust_days["rv_mae"] <= bust_days["b2_mae"])
        b2_worst = (bust_days["b2_mae"] <= bust_days["od_mae"]) & (bust_days["b2_mae"] <= bust_days["rv_mae"])
        print(f"MNQ={mnq_test}: {len(bust_days)} bust days (realistic model)")
        print(f"  Caused by OD trade MAE:  {od_worst.sum()} days ({100*od_worst.mean():.1f}%)")
        print(f"  Caused by RV trade MAE:  {rv_worst.sum()} days ({100*rv_worst.mean():.1f}%)")
        print(f"  Caused by B2 trade MAE:  {b2_worst.sum()} days ({100*b2_worst.mean():.1f}%)")

    # ----------------------------------------------------------
    # Q3: OD single-trade MAE distribution + cutoff impact
    # ----------------------------------------------------------
    print("\n=== Q3: OD per-trade MAE distribution ===\n")
    od_only = df[df["strat"] == "OD"].copy()
    print(f"OD trades total: {len(od_only)}")
    print("Per-trade MAE percentiles (NQ basis $):")
    for pct in [50, 75, 90, 95, 99, 99.5, 99.9]:
        v = np.percentile(od_only["mae_$"], 100 - pct)  # MAE is negative, so we want low percentile
        print(f"  P{pct:>5.1f} (worst {100-pct:>4.1f}%):  ${v:>+9,.0f}")
    print(f"  WORST single trade:        ${od_only['mae_$'].min():+,.0f}")

    # ----------------------------------------------------------
    # Q4: Cutoff impact simulation
    # If we cap OD single-trade MAE at -$X (i.e., force-close if unrealized loss
    # exceeds X), how does that change the picture?
    # Important: capping MAE doesn't help if the realized exit was also bad.
    # Approximation: if MAE < -$X, cap MAE at -$X AND cap the trade's realized PnL
    # at min(actual_pnl, -$X) since closing at MAE would lock that loss.
    # ----------------------------------------------------------
    print("\n=== Q4: OD hard-cutoff (force-close if unrealized loss > $X) ===\n")
    # Cutoffs in NQ basis (i.e., per 1 NQ contract):
    CUTOFFS = [3000, 5000, 7500, 10000, 12500, 15000, 20000, 25000]
    print(f"{'Cutoff_NQ$':>10}  {'OD_PnL_NQ':>10}  {'OD_delta':>10}  {'Combined_PnL':>13}  {'MNQ=2 daily-bust%':>18}  {'MNQ=4 daily-bust%':>18}")
    base_od_pnl = od_only["pnl_$"].sum()
    base_combined_pnl = df["pnl_$"].sum()
    for cutoff in CUTOFFS:
        # Simulate: clip MAE to -cutoff, clip PnL to max(actual_pnl, -cutoff) — actually
        # closing at MAE means we realize -cutoff (if MAE was deeper). So:
        new_pnl = np.where(od_only["mae_$"] < -cutoff,
                            -cutoff,  # forced exit at MAE = -cutoff
                            od_only["pnl_$"])
        new_mae = np.where(od_only["mae_$"] < -cutoff, -cutoff, od_only["mae_$"])
        n_capped = int((od_only["mae_$"] < -cutoff).sum())
        new_od_pnl = float(new_pnl.sum())
        delta_od = new_od_pnl - base_od_pnl
        new_combined_pnl = base_combined_pnl - base_od_pnl + new_od_pnl
        # Recompute daily-loss bust at MNQ=2 and 4 with new MAE
        df_copy = df.copy()
        od_idx = df_copy["strat"] == "OD"
        df_copy.loc[od_idx, "mae_$"] = new_mae
        df_copy.loc[od_idx, "pnl_$"] = new_pnl
        # Rebuild daily worst_single_mae
        d2 = df_copy.groupby(df_copy["exit_ts"].dt.date).agg(
            worst_single_mae_nq=("mae_$", "min"))
        for mnq_test, label in [(2, "MNQ=2"), (4, "MNQ=4")]:
            scale = mnq_test * 0.1
            bust = (d2["worst_single_mae_nq"] * scale <= -DAILY_LIMIT_MNQ).sum()
            d2[f"bust_{mnq_test}"] = bust / len(d2)
        bust2_pct = 100 * (d2["worst_single_mae_nq"] * 0.2 <= -DAILY_LIMIT_MNQ).sum() / len(d2)
        bust4_pct = 100 * (d2["worst_single_mae_nq"] * 0.4 <= -DAILY_LIMIT_MNQ).sum() / len(d2)
        print(f"{cutoff:>10,}  {new_od_pnl:>+10,.0f}  {delta_od:>+10,.0f}  "
              f"{new_combined_pnl:>+13,.0f}  {bust2_pct:>17.2f}% {bust4_pct:>17.2f}%  "
              f"({n_capped}/{len(od_only)} trades capped)")
    print(f"\nNO CUTOFF baseline: OD_PnL ${base_od_pnl:+,.0f}  Combined ${base_combined_pnl:+,.0f}")

    daily.to_csv(OUT_DIR / "daily_loss_frequency_analysis.csv", index=False)
    print(f"\nSaved daily timeline -> {OUT_DIR / 'daily_loss_frequency_analysis.csv'}")


if __name__ == "__main__":
    main()

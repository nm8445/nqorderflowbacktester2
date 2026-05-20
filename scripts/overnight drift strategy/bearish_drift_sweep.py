"""
Sweep bearish_drift parameter for OD yellow mode vs pure_ratchet baseline.

Tests downward yellow drift (in points/bar) when bar is bearish (close < open).
The user's question: does giving the stop more room on bearish candles improve OD stats?

Locked-config baseline (from live config overnight drift.md):
  yellow_atr_mult=1.30, green_atr_mult=1.00, green_base=82.5, green_decay=1.50,
  red_drift=0.45, use_be=False, use_martingale=True, base=1, loss=2, mode=pure_ratchet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import (  # noqa: E402
    StrategyParams,
    build_full_20min_series,
    run_backtest,
    trades_to_df,
)

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    pnl = df["pnl_dollars"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    cum = pnl.cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    return {
        "trades": len(df),
        "win_pct": (pnl > 0).mean() * 100,
        "gross_$": pnl.sum(),
        "best_$": pnl.max(),
        "worst_$": pnl.min(),
        "avg_win_$": wins.mean() if len(wins) else 0,
        "avg_loss_$": losses.mean() if len(losses) else 0,
        "pf": (wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
        "max_dd_$": dd,
        "g_over_dd": abs(pnl.sum() / dd) if dd < 0 else np.inf,
    }


def yearly_pf(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    d = df.copy()
    d["year"] = d["entry_time"].dt.year
    out = {}
    for yr, grp in d.groupby("year"):
        wins = grp[grp["pnl_dollars"] > 0]["pnl_dollars"].sum()
        losses = abs(grp[grp["pnl_dollars"] < 0]["pnl_dollars"].sum())
        pf = wins / losses if losses > 0 else float("inf")
        out[int(yr)] = (pf, grp["pnl_dollars"].sum())
    return out


def make_params(mode: str, drift: float = 0.0) -> StrategyParams:
    return StrategyParams(
        yellow_atr_len=14, yellow_atr_mult=1.30,
        yellow_mode=mode, bearish_drift_pts=drift,
        green_atr_len=14, green_atr_mult=1.00,
        green_base=82.5, green_decay=1.50,
        red_intercept=0.0, red_drift=0.45,
        use_be=False, use_martingale=True, base_qty=1, loss_qty=2,
    )


def main():
    print("Building 20-min bars from data sources...")
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(f"Loaded {len(bars):,} 20-min bars from {bars.index[0]} to {bars.index[-1]}\n")

    rows = []
    # Baseline: pure_ratchet
    p = make_params("pure_ratchet")
    trades = run_backtest(bars, p)
    df = trades_to_df(trades)
    s = stats(df)
    rows.append({"mode": "pure_ratchet", "drift_pts": 0.0, **s})
    yr_baseline = yearly_pf(df)

    # bearish_drift sweep
    for d in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
        p = make_params("bearish_drift", drift=d)
        trades = run_backtest(bars, p)
        df = trades_to_df(trades)
        s = stats(df)
        rows.append({"mode": "bearish_drift", "drift_pts": d, **s})

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("=== BEARISH_DRIFT SWEEP — full history ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))

    # Year-by-year for top 2 candidates + baseline
    print("\n=== YEARLY PF — baseline vs top drift candidates ===")
    top_drifts = out[out["mode"] == "bearish_drift"].nlargest(3, "g_over_dd")
    print(f"\nBaseline pure_ratchet yearly PF / $:")
    for yr, (pf, g) in sorted(yr_baseline.items()):
        print(f"  {yr}: PF={pf:.2f}  Gross=${g:>9,.0f}")

    for _, row in top_drifts.iterrows():
        drift = row["drift_pts"]
        p = make_params("bearish_drift", drift=drift)
        trades = run_backtest(bars, p)
        df = trades_to_df(trades)
        yr = yearly_pf(df)
        print(f"\nbearish_drift={drift} yearly PF / $:")
        for yrk, (pf, g) in sorted(yr.items()):
            base_pf, base_g = yr_baseline.get(yrk, (0, 0))
            delta = g - base_g
            print(f"  {yrk}: PF={pf:.2f}  Gross=${g:>9,.0f}  (delta vs baseline: ${delta:>+9,.0f})")

    out.to_csv(Path(__file__).parent / "bearish_drift_sweep_results.csv", index=False)
    print(f"\nSaved: {Path(__file__).parent / 'bearish_drift_sweep_results.csv'}")


if __name__ == "__main__":
    main()

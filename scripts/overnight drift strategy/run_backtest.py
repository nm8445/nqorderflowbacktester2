"""
Run the Overnight Drift strategy across the full 5-year history and report
yearly stats + gamma-regime split.

Data sources:
- 1-min bars: D:/trading_pythonbacktest_data/markettick_1min_bars.parquet
              (covers 2020-12-01 -> 2025-12-01)
- 5-min daily pickles (Databento): D:/trading_pythonbacktest_data/timebars_5min/
              (covers 2025-03-13 -> 2026-04-17, used for the post-parquet tail)
- Daily gamma regime: D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet
              column qqq_gamma_sign  (1 = positive gamma, -1 = negative)
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
    yearly_summary,
)

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"
GAMMA_PATH = "D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet"

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def gamma_regime_split(trades: pd.DataFrame, gamma_path: str) -> pd.DataFrame:
    """Tag each trade with the most recent gamma sign at entry, then summarise.

    Uses an asof-backward merge so Sunday and holiday entries pick up the
    previous business-day settle (capped at 5 days stale).
    """
    gamma = pd.read_parquet(gamma_path)
    gamma["date"] = pd.to_datetime(gamma["date"])
    gamma = (
        gamma[["date", "qqq_gamma_sign"]]
        .dropna(subset=["qqq_gamma_sign"])
        .sort_values("date")
    )

    trades = trades.copy()
    trades["session_date"] = (
        trades["entry_time"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    )
    trades = trades.sort_values("session_date").reset_index(drop=True)
    trades = pd.merge_asof(
        trades, gamma, left_on="session_date", right_on="date", direction="backward"
    )
    days_stale = (trades["session_date"] - trades["date"]).dt.days
    trades.loc[days_stale > 5, "qqq_gamma_sign"] = np.nan
    trades = trades.drop(columns=["date"])

    def stats(g: pd.DataFrame) -> pd.Series:
        wins = (g["pnl_dollars"] > 0).sum()
        losses = (g["pnl_dollars"] < 0).sum()
        gross_w = g.loc[g["pnl_dollars"] > 0, "pnl_dollars"].sum()
        gross_l = abs(g.loc[g["pnl_dollars"] < 0, "pnl_dollars"].sum())
        return pd.Series(
            {
                "trades": len(g),
                "wins": wins,
                "losses": losses,
                "win_rate_%": (wins / len(g) * 100) if len(g) else np.nan,
                "gross_pnl_$": g["pnl_dollars"].sum(),
                "avg_pnl_$": g["pnl_dollars"].mean(),
                "median_pnl_$": g["pnl_dollars"].median(),
                "avg_pts": g["pnl_points"].mean(),
                "profit_factor": gross_w / gross_l if gross_l > 0 else np.inf,
            }
        )

    by_regime = trades.groupby("qqq_gamma_sign", dropna=False).apply(
        stats, include_groups=False
    )
    return trades, by_regime


def main() -> None:
    print("Building 20-min bars from 1-min parquet + 5-min daily pickles...", flush=True)
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(
        f"  bars: {len(bars):,}  range: {bars.index.min()}  ->  {bars.index.max()}",
        flush=True,
    )

    params = StrategyParams()
    print("Running backtest with default params...", flush=True)
    trades = run_backtest(bars, params)
    print(f"  trades: {len(trades)}", flush=True)

    df = trades_to_df(trades)
    if df.empty:
        print("No trades.")
        return

    df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.tz_convert("America/New_York")
    df["exit_time"] = pd.to_datetime(df["exit_time"]).dt.tz_convert("America/New_York")

    yearly = yearly_summary(df)
    print("\n=== Yearly summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(yearly.round(2).to_string())

    overall_pnl = df["pnl_dollars"].sum()
    overall_pf = (
        df.loc[df["pnl_dollars"] > 0, "pnl_dollars"].sum()
        / abs(df.loc[df["pnl_dollars"] < 0, "pnl_dollars"].sum())
        if (df["pnl_dollars"] < 0).any()
        else float("inf")
    )
    print(
        f"\nTotal trades: {len(df)}  |  total $: {overall_pnl:,.2f}  |  "
        f"win-rate: {(df['pnl_dollars'] > 0).mean() * 100:.1f}%  |  PF: {overall_pf:.2f}"
    )
    print("Exit reasons:")
    print(df["reason"].value_counts().to_string())

    print("\n=== Gamma-regime split (longs only) ===")
    trades_with_regime, by_regime = gamma_regime_split(df, GAMMA_PATH)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(by_regime.round(2).to_string())

    df.to_csv(OUT_DIR / "trades.csv", index=False)
    yearly.to_csv(OUT_DIR / "yearly_summary.csv")
    by_regime.to_csv(OUT_DIR / "gamma_regime_summary.csv")
    trades_with_regime.to_csv(OUT_DIR / "trades_with_gamma.csv", index=False)
    print(f"\nWrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()

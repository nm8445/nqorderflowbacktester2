"""
ADX regime analysis for VWAP Reaction Strategy.

Computes ADX over 5-min bars at each trade entry and breaks down
performance by ADX range to find where the strategy works vs doesn't.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import run_backtest

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
TIMEBAR_DIR = DATA_DIR / "timebars_5min"

ADX_PERIOD = 14


def load_5min_bars_for_date(date_str: str) -> pd.DataFrame | None:
    fmt = date_str.replace("-", "_")
    path = TIMEBAR_DIR / f"timebars_5min_{fmt}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        bars = pickle.load(f)
    rows = []
    for b in bars:
        rows.append({
            "timestamp": b["open_time"],
            "open": b["open"],
            "high": b["high"],
            "low": b["low"],
            "close": b["close"],
        })
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """Compute ADX from OHLC DataFrame. Returns Series aligned to df index."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move = high - high.shift()
    down_move = low.shift() - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smoothed with Wilder's method (EMA with alpha=1/period)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False).mean() / atr

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return adx


def tag_trades_with_adx(trades_df: pd.DataFrame) -> pd.DataFrame:
    trades = trades_df.copy()
    trades["adx"] = np.nan

    bars_cache = {}

    for idx, trade in trades.iterrows():
        date_str = trade["date"]

        if date_str not in bars_cache:
            df_5m = load_5min_bars_for_date(date_str)
            if df_5m is not None and len(df_5m) > ADX_PERIOD * 2:
                bars_cache[date_str] = (df_5m, compute_adx(df_5m, ADX_PERIOD))
            else:
                bars_cache[date_str] = (None, None)

        df_5m, adx_series = bars_cache[date_str]
        if adx_series is None:
            continue

        entry_time = trade["entry_time"]
        if not hasattr(entry_time, "tzinfo") or entry_time.tzinfo is None:
            entry_time = pd.Timestamp(entry_time, tz=ET)

        prior = adx_series[adx_series.index <= entry_time]
        if len(prior) == 0:
            continue

        trades.at[idx, "adx"] = prior.iloc[-1]

    return trades


def print_adx_breakdown(trades: pd.DataFrame):
    valid = trades.dropna(subset=["adx"])

    print()
    print("=" * 95)
    print(f"ADX REGIME ANALYSIS — {ADX_PERIOD}-period ADX on 5-min bars at entry")
    print("=" * 95)
    print()

    # Fixed buckets
    buckets = [
        ("0-10", 0, 10),
        ("10-15", 10, 15),
        ("15-20", 15, 20),
        ("20-25", 20, 25),
        ("25-30", 25, 30),
        ("30-40", 30, 40),
        ("40-50", 40, 50),
        ("50+", 50, 999),
    ]

    print(f"{'ADX Range':<10s} | {'Trades':>6s} | {'WR':>6s} | {'PF':>6s} | "
          f"{'Avg P&L':>9s} | {'Total P&L':>10s} | {'Avg ADX':>7s}")
    print("-" * 75)

    for label, lo, hi in buckets:
        subset = valid[(valid["adx"] >= lo) & (valid["adx"] < hi)]
        n = len(subset)
        if n == 0:
            print(f"{label:<10s} | {0:6d} |       |       |           |            |")
            continue
        w = subset[subset["pnl_dollars"] > 0]
        l = subset[subset["pnl_dollars"] <= 0]
        wr = len(w) / n * 100
        gp = w["pnl_dollars"].sum() if len(w) else 0
        gl = abs(l["pnl_dollars"].sum()) if len(l) else 1
        pf = gp / gl if gl > 0 else 999
        avg_pnl = subset["pnl_dollars"].mean()
        total_pnl = subset["pnl_dollars"].sum()
        avg_adx = subset["adx"].mean()
        print(f"{label:<10s} | {n:6d} | {wr:5.1f}% | {pf:6.2f} | "
              f"${avg_pnl:>+8,.0f} | ${total_pnl:>+9,.0f} | {avg_adx:7.1f}")

    print("-" * 75)
    n = len(valid)
    total_pnl = valid["pnl_dollars"].sum()
    print(f"{'TOTAL':<10s} | {n:6d} |       |       |           | ${total_pnl:>+9,.0f} |")
    print()

    # Low vs High ADX comparison
    low_adx = valid[valid["adx"] < 20]
    mid_adx = valid[(valid["adx"] >= 20) & (valid["adx"] < 35)]
    high_adx = valid[valid["adx"] >= 35]

    print("LOW vs MID vs HIGH ADX:")
    for label, subset in [("Low (<20)", low_adx), ("Mid (20-35)", mid_adx), ("High (35+)", high_adx)]:
        n = len(subset)
        if n == 0:
            print(f"  {label:<12s}: 0 trades")
            continue
        w = subset[subset["pnl_dollars"] > 0]
        l = subset[subset["pnl_dollars"] <= 0]
        wr = len(w) / n * 100
        gp = w["pnl_dollars"].sum() if len(w) else 0
        gl = abs(l["pnl_dollars"].sum()) if len(l) else 1
        pf = gp / gl if gl > 0 else 999
        exp = subset["pnl_dollars"].mean()
        total = subset["pnl_dollars"].sum()
        print(f"  {label:<12s}: {n:3d} trades | WR {wr:5.1f}% | PF {pf:.2f} | "
              f"Exp ${exp:+,.0f}/trade | Total ${total:+,.0f}")

    # Flag meaningful differences
    if len(low_adx) > 10 and len(high_adx) > 10:
        low_exp = low_adx["pnl_dollars"].mean()
        high_exp = high_adx["pnl_dollars"].mean()
        diff = high_exp - low_exp
        if abs(diff) > 50:
            better = "HIGH ADX" if diff > 0 else "LOW ADX"
            print(f"\n  *** {better} performs better: exp diff ${diff:+,.0f}/trade ***")
    print()

    # ADX x Direction
    print("ADX x DIRECTION:")
    print(f"{'ADX Range':<10s} | {'Dir':>5s} | {'Trades':>6s} | {'WR':>6s} | {'Total P&L':>10s}")
    print("-" * 50)
    for label, lo, hi in [("Low <20", 0, 20), ("Mid 20-35", 20, 35), ("High 35+", 35, 999)]:
        for d in ["long", "short"]:
            subset = valid[(valid["adx"] >= lo) & (valid["adx"] < hi) & (valid["direction"] == d)]
            if len(subset) == 0:
                continue
            n = len(subset)
            wr = len(subset[subset["pnl_dollars"] > 0]) / n * 100
            total = subset["pnl_dollars"].sum()
            print(f"{label:<10s} | {d:>5s} | {n:6d} | {wr:5.1f}% | ${total:>+9,.0f}")
    print()

    # Distribution stats
    print("ADX DISTRIBUTION AT ENTRY:")
    print(f"  Mean:   {valid['adx'].mean():.1f}")
    print(f"  Median: {valid['adx'].median():.1f}")
    print(f"  Min:    {valid['adx'].min():.1f}")
    print(f"  Max:    {valid['adx'].max():.1f}")
    print(f"  Std:    {valid['adx'].std():.1f}")
    print()


def main():
    print("Running VWAP reaction backtest...\n")
    trades_df = run_backtest()

    print("\nComputing ADX at each trade entry...")
    tagged = tag_trades_with_adx(trades_df)

    print_adx_breakdown(tagged)

    # Save
    out = DATA_DIR / "vwap_reaction_adx_trades.csv"
    save_df = tagged.copy()
    save_df["entry_time"] = save_df["entry_time"].astype(str)
    save_df["exit_time"] = save_df["exit_time"].astype(str)
    save_df.to_csv(out, index=False)
    print(f"Per-trade ADX labels saved to {out}")


if __name__ == "__main__":
    main()

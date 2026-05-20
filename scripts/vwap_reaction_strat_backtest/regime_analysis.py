"""
Regime analysis for VWAP Reaction Strategy.

Tags each historical trade with a market regime label based on rolling
linear regression of 5-min closing prices leading up to the entry.

Pure analysis layer — does not modify the backtest or strategy logic.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import run_backtest

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
TIMEBAR_DIR = DATA_DIR / "timebars_5min"

# Regime classification parameters
LOOKBACK = 14  # number of 5-min bars before entry
R2_TRENDING = 0.6   # R² above this = trending
R2_MODERATE = 0.3   # R² between moderate and trending
SLOPE_THRESH = 0.5  # normalized slope threshold (bps/bar)


def load_5min_bars_for_date(date_str: str) -> pd.DataFrame | None:
    """Load 5-min bars for a date, return DataFrame with ET index."""
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


def linreg_regime(closes: np.ndarray) -> dict:
    """
    Compute linear regression slope, R², and regime label.

    Parameters
    ----------
    closes : array of closing prices (length = LOOKBACK)

    Returns
    -------
    dict with slope, r2, norm_slope, regime
    """
    n = len(closes)
    if n < 3:
        return {"slope": 0, "r2": 0, "norm_slope": 0, "regime": "insufficient_data"}

    x = np.arange(n, dtype=float)
    y = closes.astype(float)

    # Linear regression: y = mx + b
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()
    ss_xx = ((x - x_mean) ** 2).sum()
    ss_yy = ((y - y_mean) ** 2).sum()

    if ss_xx == 0 or ss_yy == 0:
        return {"slope": 0, "r2": 0, "norm_slope": 0, "regime": "choppy"}

    slope = ss_xy / ss_xx
    r2 = (ss_xy ** 2) / (ss_xx * ss_yy)

    # Normalized slope: basis points per bar
    mean_price = y_mean if y_mean != 0 else 1
    norm_slope = slope / mean_price * 10000

    # Classify
    if r2 > R2_TRENDING and norm_slope > SLOPE_THRESH:
        regime = "trending_up"
    elif r2 > R2_TRENDING and norm_slope < -SLOPE_THRESH:
        regime = "trending_down"
    elif r2 < R2_MODERATE or abs(norm_slope) <= SLOPE_THRESH:
        regime = "choppy"
    else:
        regime = "moderate"

    return {
        "slope": round(slope, 4),
        "r2": round(r2, 4),
        "norm_slope": round(norm_slope, 4),
        "regime": regime,
    }


def tag_trades_with_regime(trades_df: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """
    Add regime columns to trades DataFrame.

    For each trade, loads 5-min bars for that date, finds the lookback window
    ending at (or just before) the entry time, runs linear regression.
    """
    trades = trades_df.copy()
    trades["regime"] = "unknown"
    trades["r2"] = np.nan
    trades["norm_slope"] = np.nan
    trades["slope"] = np.nan

    # Cache 5-min bars by date
    bars_cache = {}

    for idx, trade in trades.iterrows():
        date_str = trade["date"]

        if date_str not in bars_cache:
            bars_cache[date_str] = load_5min_bars_for_date(date_str)

        bars_5m = bars_cache[date_str]
        if bars_5m is None or bars_5m.empty:
            continue

        # Find bars at or before entry time
        entry_time = trade["entry_time"]
        if not hasattr(entry_time, "tzinfo") or entry_time.tzinfo is None:
            entry_time = pd.Timestamp(entry_time, tz=ET)

        prior = bars_5m[bars_5m.index <= entry_time]
        if len(prior) < lookback:
            continue

        window = prior.iloc[-lookback:]
        result = linreg_regime(window["close"].values)

        trades.at[idx, "regime"] = result["regime"]
        trades.at[idx, "r2"] = result["r2"]
        trades.at[idx, "norm_slope"] = result["norm_slope"]
        trades.at[idx, "slope"] = result["slope"]

    return trades


def print_regime_breakdown(trades: pd.DataFrame):
    """Print performance breakdown by regime."""
    print()
    print("=" * 90)
    print(f"REGIME ANALYSIS — {LOOKBACK}-bar lookback on 5-min closes")
    print(f"Trending: R² > {R2_TRENDING} & |slope| > {SLOPE_THRESH} bps/bar")
    print(f"Moderate: R² {R2_MODERATE}-{R2_TRENDING}")
    print(f"Choppy:   R² < {R2_MODERATE} or |slope| <= {SLOPE_THRESH}")
    print("=" * 90)
    print()

    regimes = ["trending_up", "trending_down", "choppy", "moderate"]
    present = [r for r in regimes if r in trades["regime"].values]

    print(f"{'Regime':<16s} | {'Trades':>6s} | {'WR':>6s} | {'PF':>6s} | "
          f"{'Avg P&L':>9s} | {'Total P&L':>10s} | {'Avg R²':>6s} | {'Avg Slope':>10s}")
    print("-" * 90)

    for regime in present:
        subset = trades[trades["regime"] == regime]
        n = len(subset)
        w = subset[subset["pnl_dollars"] > 0]
        l = subset[subset["pnl_dollars"] <= 0]
        wr = len(w) / n * 100 if n > 0 else 0
        gp = w["pnl_dollars"].sum() if len(w) else 0
        gl = abs(l["pnl_dollars"].sum()) if len(l) else 1
        pf = gp / gl if gl > 0 else 999
        avg_pnl = subset["pnl_dollars"].mean()
        total_pnl = subset["pnl_dollars"].sum()
        avg_r2 = subset["r2"].mean()
        avg_slope = subset["norm_slope"].mean()

        print(f"{regime:<16s} | {n:6d} | {wr:5.1f}% | {pf:6.2f} | "
              f"${avg_pnl:>+8,.0f} | ${total_pnl:>+9,.0f} | {avg_r2:6.3f} | {avg_slope:>+9.2f}")

    print("-" * 90)

    # Totals
    tagged = trades[trades["regime"] != "unknown"]
    n = len(tagged)
    total_pnl = tagged["pnl_dollars"].sum()
    print(f"{'TOTAL':<16s} | {n:6d} |       |       |           | ${total_pnl:>+9,.0f} |")
    print()

    # Trending vs choppy comparison
    trending = trades[trades["regime"].isin(["trending_up", "trending_down"])]
    choppy = trades[trades["regime"] == "choppy"]

    if len(trending) > 0 and len(choppy) > 0:
        t_wr = len(trending[trending["pnl_dollars"] > 0]) / len(trending) * 100
        c_wr = len(choppy[choppy["pnl_dollars"] > 0]) / len(choppy) * 100
        t_exp = trending["pnl_dollars"].mean()
        c_exp = choppy["pnl_dollars"].mean()
        t_gp = trending[trending["pnl_dollars"] > 0]["pnl_dollars"].sum()
        t_gl = abs(trending[trending["pnl_dollars"] <= 0]["pnl_dollars"].sum()) or 1
        c_gp = choppy[choppy["pnl_dollars"] > 0]["pnl_dollars"].sum()
        c_gl = abs(choppy[choppy["pnl_dollars"] <= 0]["pnl_dollars"].sum()) or 1

        print("TRENDING vs CHOPPY COMPARISON:")
        print(f"  Trending:  {len(trending):3d} trades | WR {t_wr:5.1f}% | "
              f"PF {t_gp/t_gl:.2f} | Exp ${t_exp:+,.0f}/trade | Total ${trending['pnl_dollars'].sum():+,.0f}")
        print(f"  Choppy:    {len(choppy):3d} trades | WR {c_wr:5.1f}% | "
              f"PF {c_gp/c_gl:.2f} | Exp ${c_exp:+,.0f}/trade | Total ${choppy['pnl_dollars'].sum():+,.0f}")

        wr_diff = t_wr - c_wr
        exp_diff = t_exp - c_exp
        print()
        if abs(wr_diff) > 5 or abs(exp_diff) > 50:
            better = "TRENDING" if exp_diff > 0 else "CHOPPY"
            print(f"  *** {better} regimes perform better: "
                  f"WR diff {wr_diff:+.1f}%, Exp diff ${exp_diff:+,.0f}/trade ***")
        else:
            print("  No significant difference between trending and choppy regimes.")
    print()

    # Direction within regime
    print("REGIME x DIRECTION:")
    print(f"{'Regime':<16s} | {'Dir':>5s} | {'Trades':>6s} | {'WR':>6s} | {'Total P&L':>10s}")
    print("-" * 60)
    for regime in present:
        for d in ["long", "short"]:
            subset = trades[(trades["regime"] == regime) & (trades["direction"] == d)]
            if len(subset) == 0:
                continue
            n = len(subset)
            wr = len(subset[subset["pnl_dollars"] > 0]) / n * 100
            total = subset["pnl_dollars"].sum()
            print(f"{regime:<16s} | {d:>5s} | {n:6d} | {wr:5.1f}% | ${total:>+9,.0f}")
    print()


def main():
    print("Running VWAP reaction backtest...\n")
    trades_df = run_backtest()

    print("\nTagging trades with regime labels...")
    tagged = tag_trades_with_regime(trades_df, lookback=LOOKBACK)

    print_regime_breakdown(tagged)

    # Save tagged trades
    out = DATA_DIR / "vwap_reaction_regime_trades.csv"
    save_df = tagged.copy()
    save_df["entry_time"] = save_df["entry_time"].astype(str)
    save_df["exit_time"] = save_df["exit_time"].astype(str)
    save_df.to_csv(out, index=False)
    print(f"Per-trade regime labels saved to {out}")


if __name__ == "__main__":
    main()

"""
Deep dive: time-of-day within regimes + entry type split across regimes.

1. Choppy regime performance by time of day
2. Entry type split: retracement hold vs rejection after VWAP break
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
LOOKBACK = 14
R2_TRENDING = 0.6
R2_MODERATE = 0.3
SLOPE_THRESH = 0.5


def load_5min_bars_for_date(date_str: str) -> pd.DataFrame | None:
    fmt = date_str.replace("-", "_")
    path = TIMEBAR_DIR / f"timebars_5min_{fmt}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        bars = pickle.load(f)
    rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
             "low": b["low"], "close": b["close"]} for b in bars]
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1/period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def linreg_regime(closes: np.ndarray) -> str:
    n = len(closes)
    if n < 3:
        return "unknown"
    x = np.arange(n, dtype=float)
    y = closes.astype(float)
    x_mean, y_mean = x.mean(), y.mean()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()
    ss_xx = ((x - x_mean) ** 2).sum()
    ss_yy = ((y - y_mean) ** 2).sum()
    if ss_xx == 0 or ss_yy == 0:
        return "choppy"
    slope = ss_xy / ss_xx
    r2 = (ss_xy ** 2) / (ss_xx * ss_yy)
    norm_slope = slope / y_mean * 10000 if y_mean != 0 else 0
    if r2 > R2_TRENDING and norm_slope > SLOPE_THRESH:
        return "trending_up"
    elif r2 > R2_TRENDING and norm_slope < -SLOPE_THRESH:
        return "trending_down"
    elif r2 < R2_MODERATE or abs(norm_slope) <= SLOPE_THRESH:
        return "choppy"
    return "moderate"


def classify_entry_type(trade) -> str:
    """
    Type 1 — Retracement hold: price pulled back to VWAP from the trend side
             and entry is on the trend side of VWAP.
             Long with entry >= VWAP, Short with entry <= VWAP.
             Clean continuation — VWAP held as S/R.

    Type 2 — Rejection after break: price pierced through VWAP zone and
             entry is on the wrong side of VWAP.
             Long with entry < VWAP, Short with entry > VWAP.
             More aggressive — fading after VWAP was breached.
    """
    vwap = trade["vwap"]
    entry = trade["entry_price"]
    direction = trade["direction"]

    if direction == "long":
        return "retracement" if entry >= vwap else "rejection"
    else:  # short
        return "retracement" if entry <= vwap else "rejection"


def tag_all(trades_df: pd.DataFrame) -> pd.DataFrame:
    trades = trades_df.copy()
    trades["regime"] = "unknown"
    trades["adx"] = np.nan
    trades["entry_type"] = ""
    trades["hour"] = pd.to_datetime(trades["entry_time"]).dt.hour
    trades["minute"] = pd.to_datetime(trades["entry_time"]).dt.minute

    bars_cache = {}

    for idx, trade in trades.iterrows():
        date_str = trade["date"]

        if date_str not in bars_cache:
            df_5m = load_5min_bars_for_date(date_str)
            adx_s = compute_adx(df_5m, ADX_PERIOD) if df_5m is not None and len(df_5m) > ADX_PERIOD * 2 else None
            bars_cache[date_str] = (df_5m, adx_s)

        df_5m, adx_s = bars_cache[date_str]

        entry_time = trade["entry_time"]
        if not hasattr(entry_time, "tzinfo") or entry_time.tzinfo is None:
            entry_time = pd.Timestamp(entry_time, tz=ET)

        # Regime from linreg
        if df_5m is not None:
            prior = df_5m[df_5m.index <= entry_time]
            if len(prior) >= LOOKBACK:
                trades.at[idx, "regime"] = linreg_regime(prior.iloc[-LOOKBACK:]["close"].values)

        # ADX
        if adx_s is not None:
            prior_adx = adx_s[adx_s.index <= entry_time]
            if len(prior_adx) > 0:
                trades.at[idx, "adx"] = prior_adx.iloc[-1]

        # Entry type
        trades.at[idx, "entry_type"] = classify_entry_type(trade)

    return trades


def fmt_row(label, subset, total_n):
    n = len(subset)
    if n == 0:
        return f"{label:<20s} | {0:6d} |       |       |           |            |"
    w = subset[subset["pnl_dollars"] > 0]
    l = subset[subset["pnl_dollars"] <= 0]
    wr = len(w) / n * 100
    gp = w["pnl_dollars"].sum() if len(w) else 0
    gl = abs(l["pnl_dollars"].sum()) if len(l) else 1
    pf = gp / gl if gl > 0 else 999
    exp = subset["pnl_dollars"].mean()
    total = subset["pnl_dollars"].sum()
    pct = n / total_n * 100 if total_n > 0 else 0
    return (f"{label:<20s} | {n:6d} ({pct:4.1f}%) | {wr:5.1f}% | {pf:6.2f} | "
            f"${exp:>+8,.0f} | ${total:>+9,.0f}")


def analysis_1_time_within_regime(trades: pd.DataFrame):
    """Time-of-day breakdown within each regime."""
    print()
    print("=" * 95)
    print("1. TIME OF DAY WITHIN REGIME")
    print("=" * 95)

    for regime in ["choppy", "moderate", "trending_up", "trending_down"]:
        subset = trades[trades["regime"] == regime]
        if len(subset) < 5:
            continue
        print(f"\n--- {regime.upper()} ({len(subset)} trades) ---")
        print(f"{'Time Window':<20s} | {'Trades':>14s} | {'WR':>6s} | {'PF':>6s} | "
              f"{'Exp':>9s} | {'Total P&L':>10s}")
        print("-" * 85)

        windows = [
            ("7pm-9:30am (pre)", 19, 0, 9, 30),
            ("9:30-10:00", 9, 30, 10, 0),
            ("10:00-10:30", 10, 0, 10, 30),
            ("10:30-11:00", 10, 30, 11, 0),
            ("11:00-12:00", 11, 0, 12, 0),
            ("12:00-1:00pm", 12, 0, 13, 0),
            ("1:00-2:00pm", 13, 0, 14, 0),
            ("2:00-3:00pm", 14, 0, 15, 0),
            ("3:00-4:00pm", 15, 0, 16, 0),
        ]

        for label, sh, sm, eh, em in windows:
            start_min = sh * 60 + sm
            end_min = eh * 60 + em
            entry_min = subset["hour"] * 60 + subset["minute"]

            if sh >= 19:  # overnight
                mask = (entry_min >= start_min) | (entry_min < end_min)
            else:
                mask = (entry_min >= start_min) & (entry_min < end_min)

            window_trades = subset[mask]
            print(fmt_row(label, window_trades, len(subset)))

    # Choppy early vs late
    choppy = trades[trades["regime"] == "choppy"]
    if len(choppy) > 0:
        entry_min = choppy["hour"] * 60 + choppy["minute"]
        early = choppy[(entry_min >= 570) & (entry_min < 630)]  # 9:30-10:30
        mid = choppy[(entry_min >= 630) & (entry_min < 780)]    # 10:30-1pm
        late = choppy[(entry_min >= 780) & (entry_min < 960)]   # 1pm-4pm

        print(f"\n--- CHOPPY: EARLY vs MID vs LATE SESSION ---")
        print(f"{'Window':<20s} | {'Trades':>14s} | {'WR':>6s} | {'PF':>6s} | "
              f"{'Exp':>9s} | {'Total P&L':>10s}")
        print("-" * 85)
        print(fmt_row("Early (9:30-10:30)", early, len(choppy)))
        print(fmt_row("Mid (10:30-1pm)", mid, len(choppy)))
        print(fmt_row("Late (1pm-4pm)", late, len(choppy)))

        if len(early) > 5 and len(late) > 5:
            e_exp = early["pnl_dollars"].mean()
            l_exp = late["pnl_dollars"].mean()
            diff = l_exp - e_exp
            if abs(diff) > 50:
                better = "LATE" if diff > 0 else "EARLY"
                print(f"\n  *** {better} choppy trades have better expectancy: "
                      f"diff ${diff:+,.0f}/trade ***")


def analysis_2_entry_types(trades: pd.DataFrame):
    """Entry type breakdown across regimes."""
    print()
    print("=" * 95)
    print("2. ENTRY TYPE ACROSS REGIMES")
    print("   Retracement = entry on trend side of VWAP (VWAP held)")
    print("   Rejection   = entry on wrong side of VWAP (price pierced through)")
    print("=" * 95)

    # Overall entry type split
    print(f"\n--- OVERALL ---")
    print(f"{'Entry Type':<20s} | {'Trades':>14s} | {'WR':>6s} | {'PF':>6s} | "
          f"{'Exp':>9s} | {'Total P&L':>10s}")
    print("-" * 85)
    for etype in ["retracement", "rejection"]:
        subset = trades[trades["entry_type"] == etype]
        print(fmt_row(etype, subset, len(trades)))

    # Entry type within each regime
    for regime in ["choppy", "moderate", "trending_up", "trending_down"]:
        regime_trades = trades[trades["regime"] == regime]
        if len(regime_trades) < 5:
            continue

        print(f"\n--- {regime.upper()} ({len(regime_trades)} trades) ---")
        print(f"{'Entry Type':<20s} | {'Trades':>14s} | {'WR':>6s} | {'PF':>6s} | "
              f"{'Exp':>9s} | {'Total P&L':>10s}")
        print("-" * 85)
        for etype in ["retracement", "rejection"]:
            subset = regime_trades[regime_trades["entry_type"] == etype]
            print(fmt_row(etype, subset, len(regime_trades)))

    # Entry type within ADX bands
    valid_adx = trades.dropna(subset=["adx"])
    print(f"\n--- ENTRY TYPE x ADX ---")
    print(f"{'Bucket':<20s} | {'Trades':>14s} | {'WR':>6s} | {'PF':>6s} | "
          f"{'Exp':>9s} | {'Total P&L':>10s}")
    print("-" * 85)
    for adx_label, lo, hi in [("Low ADX <20", 0, 20), ("Mid ADX 20-30", 20, 30), ("High ADX 30+", 30, 999)]:
        adx_band = valid_adx[(valid_adx["adx"] >= lo) & (valid_adx["adx"] < hi)]
        for etype in ["retracement", "rejection"]:
            subset = adx_band[adx_band["entry_type"] == etype]
            label = f"{adx_label} {etype[:5]}"
            print(fmt_row(label, subset, len(valid_adx)))

    # Key question: are rejection entries in trending regimes the losers?
    trending = trades[trades["regime"].isin(["trending_up", "trending_down"])]
    if len(trending) > 5:
        print(f"\n--- TRENDING REGIME: RETRACEMENT vs REJECTION ---")
        print(f"{'Entry Type':<20s} | {'Trades':>14s} | {'WR':>6s} | {'PF':>6s} | "
              f"{'Exp':>9s} | {'Total P&L':>10s}")
        print("-" * 85)
        for etype in ["retracement", "rejection"]:
            subset = trending[trending["entry_type"] == etype]
            print(fmt_row(etype, subset, len(trending)))

        ret = trending[trending["entry_type"] == "retracement"]
        rej = trending[trending["entry_type"] == "rejection"]
        if len(ret) > 3 and len(rej) > 3:
            r_exp = ret["pnl_dollars"].mean()
            j_exp = rej["pnl_dollars"].mean()
            diff = r_exp - j_exp
            if abs(diff) > 50:
                better = "RETRACEMENT" if diff > 0 else "REJECTION"
                print(f"\n  *** In trending regimes, {better} entries perform better: "
                      f"diff ${diff:+,.0f}/trade ***")
            else:
                print(f"\n  No significant difference in trending regimes.")

    # Suggested filter
    print()
    print("=" * 95)
    print("SUGGESTED FILTER TEST")
    print("=" * 95)

    # What if we only took rejection entries in choppy/moderate?
    choppy_mod = trades[trades["regime"].isin(["choppy", "moderate"])]
    rej_in_trend = trades[(trades["regime"].isin(["trending_up", "trending_down"])) &
                          (trades["entry_type"] == "rejection")]
    filtered_out = rej_in_trend
    kept = trades[~trades.index.isin(filtered_out.index)]

    if len(filtered_out) > 0:
        print(f"\nFilter: Block REJECTION entries in TRENDING regimes")
        print(f"  Trades removed: {len(filtered_out)}")
        print(f"  P&L removed:    ${filtered_out['pnl_dollars'].sum():+,.0f}")
        print(f"  Remaining:      {len(kept)} trades")

        w = kept[kept["pnl_dollars"] > 0]
        l = kept[kept["pnl_dollars"] <= 0]
        wr = len(w) / len(kept) * 100
        gp = w["pnl_dollars"].sum() if len(w) else 0
        gl = abs(l["pnl_dollars"].sum()) if len(l) else 1
        pf = gp / gl if gl > 0 else 999
        total = kept["pnl_dollars"].sum()
        print(f"  New WR:         {wr:.1f}%")
        print(f"  New PF:         {pf:.2f}")
        print(f"  New Total P&L:  ${total:+,.0f}")
        print(f"  Original P&L:   ${trades['pnl_dollars'].sum():+,.0f}")
        diff = total - trades["pnl_dollars"].sum()
        print(f"  Impact:         ${diff:+,.0f} ({'better' if diff > 0 else 'worse'})")


def main():
    print("Running VWAP reaction backtest...\n")
    trades_df = run_backtest()

    print("\nTagging trades with regime, ADX, and entry type...")
    tagged = tag_all(trades_df)

    analysis_1_time_within_regime(tagged)
    analysis_2_entry_types(tagged)

    # Save
    out = DATA_DIR / "vwap_reaction_deep_dive_trades.csv"
    save_df = tagged.copy()
    save_df["entry_time"] = save_df["entry_time"].astype(str)
    save_df["exit_time"] = save_df["exit_time"].astype(str)
    save_df.to_csv(out, index=False)
    print(f"\nPer-trade data saved to {out}")


if __name__ == "__main__":
    main()

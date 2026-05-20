"""
Test ADX with 24-bar lookback (2hr on 5-min bars) vs default 14-bar.
Also show time-filter-only results for SL=1.0 TP=2.0.
"""

import pickle
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import run_backtest

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
TIMEBAR_DIR = DATA_DIR / "timebars_5min"


def load_5min_bars(date_str):
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


def compute_adx(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    up = high - high.shift()
    dn = low.shift() - low
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def build_adx_lookup(period):
    from run_backtest import VWAP_REACTION_CACHE_DIR
    dates = sorted([f.stem for f in VWAP_REACTION_CACHE_DIR.glob("*.pkl")])
    all_rows = []
    for date_str in dates:
        df_5m = load_5min_bars(date_str)
        if df_5m is None or len(df_5m) <= period * 2:
            continue
        adx_s = compute_adx(df_5m, period)
        tmp = pd.DataFrame({"adx": adx_s}, index=df_5m.index)
        all_rows.append(tmp)
    return pd.concat(all_rows).sort_index()


def tag_adx(trades, lookup):
    trades = trades.copy()
    trades["entry_dt"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(ET)
    trades = trades.sort_values("entry_dt").reset_index(drop=True)
    lk = lookup.reset_index().rename(columns={"timestamp": "entry_dt"})
    merged = pd.merge_asof(trades[["entry_dt"]], lk, on="entry_dt", direction="backward")
    trades["adx"] = merged["adx"].values
    return trades


def time_filter(df):
    h = df["entry_dt"].dt.hour
    m = df["entry_dt"].dt.minute
    em = h * 60 + m
    return df[(em >= 570) & (em < 960) & ~((em >= 720) & (em < 780))].copy()


def calc(df):
    n = len(df)
    if n < 5:
        return None
    w = df[df["pnl_dollars"] > 0]
    l = df[df["pnl_dollars"] <= 0]
    gp = w["pnl_dollars"].sum() if len(w) else 0
    gl = abs(l["pnl_dollars"].sum()) if len(l) else 1
    cum = df["pnl_dollars"].cumsum()
    dd = float((cum - cum.cummax()).min())
    return {
        "trades": n, "wr": len(w)/n*100,
        "pf": gp/gl if gl > 0 else 999,
        "exp": df["pnl_dollars"].mean(),
        "total": df["pnl_dollars"].sum(), "dd": dd,
    }


def print_stats(label, s):
    if s is None:
        print(f"  {label}: insufficient trades")
        return
    print(f"  {label}: {s['trades']} trades | WR {s['wr']:.1f}% | PF {s['pf']:.2f} | "
          f"Exp ${s['exp']:+,.0f} | Total ${s['total']:+,.0f} | DD ${s['dd']:+,.0f} | "
          f"Ret/DD {abs(s['total']/s['dd']):.2f}" if s['dd'] != 0 else
          f"  {label}: {s['trades']} trades | WR {s['wr']:.1f}% | PF {s['pf']:.2f} | "
          f"Exp ${s['exp']:+,.0f} | Total ${s['total']:+,.0f} | DD ${s['dd']:+,.0f}")


def main():
    print("Loading backtest trades (SL=1.0 TP=2.0)...")
    with redirect_stdout(io.StringIO()):
        trades = run_backtest(sl_mult=1.0, tp_mult=2.0)

    print(f"Total unfiltered trades: {len(trades)}\n")

    # Build ADX lookups for both periods
    print("Building ADX-14 lookup...")
    adx14_lookup = build_adx_lookup(14)
    print("Building ADX-24 lookup...")
    adx24_lookup = build_adx_lookup(24)

    # Tag with both
    tagged14 = tag_adx(trades, adx14_lookup)
    tagged24 = tag_adx(trades, adx24_lookup)
    tagged24 = tagged24.rename(columns={"adx": "adx24"})
    tagged14["adx24"] = tagged24["adx24"].values

    # Time filter
    tf = time_filter(tagged14)
    tf = tf.sort_values("entry_dt").reset_index(drop=True)

    n = len(tf)
    sp = n // 2

    # ── 1) Time filter only ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("TIME FILTER ONLY (9:30-4pm, no lunch) — SL=1.0 TP=2.0")
    print("=" * 100)
    print_stats("IS  (50%)", calc(tf.iloc[:sp]))
    print_stats("OOS (50%)", calc(tf.iloc[sp:]))
    print_stats("FULL     ", calc(tf))

    # ── 2) ADX-14 (original) ranges ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("ADX-14 (original 70min lookback) — BUCKETED")
    print("=" * 100)
    for lo, hi, label in [(0, 15, "0-15"), (15, 20, "15-20"), (20, 25, "20-25"),
                           (25, 30, "25-30"), (20, 30, "20-30 *"), (30, 50, "30-50"), (50, 999, "50+")]:
        sub = tf[(tf["adx"] >= lo) & (tf["adx"] < hi)].reset_index(drop=True)
        ssp = len(sub) // 2
        s_full = calc(sub)
        s_oos = calc(sub.iloc[ssp:]) if ssp >= 5 else None
        if s_full:
            oos_str = f"OOS PF {s_oos['pf']:.2f}" if s_oos else "OOS: <5"
            print(f"  ADX {label:<7s}: {s_full['trades']:3d} trades | PF {s_full['pf']:.2f} | "
                  f"Exp ${s_full['exp']:+,.0f} | Total ${s_full['total']:+,.0f} | DD ${s_full['dd']:+,.0f} | {oos_str}")

    # ── 3) ADX-24 (2hr lookback) ranges ──────────────────────────────────────
    print("\n" + "=" * 100)
    print("ADX-24 (2hr / 24-bar lookback) — BUCKETED")
    print("=" * 100)
    for lo, hi, label in [(0, 15, "0-15"), (15, 20, "15-20"), (20, 25, "20-25"),
                           (25, 30, "25-30"), (20, 30, "20-30 *"), (30, 50, "30-50"), (50, 999, "50+")]:
        sub = tf[(tf["adx24"] >= lo) & (tf["adx24"] < hi)].reset_index(drop=True)
        ssp = len(sub) // 2
        s_full = calc(sub)
        s_oos = calc(sub.iloc[ssp:]) if ssp >= 5 else None
        if s_full:
            oos_str = f"OOS PF {s_oos['pf']:.2f}" if s_oos else "OOS: <5"
            print(f"  ADX {label:<7s}: {s_full['trades']:3d} trades | PF {s_full['pf']:.2f} | "
                  f"Exp ${s_full['exp']:+,.0f} | Total ${s_full['total']:+,.0f} | DD ${s_full['dd']:+,.0f} | {oos_str}")

    # ── 4) Side by side: ADX-14 vs ADX-24, both 20-30, IS/OOS ────────────────
    print("\n" + "=" * 100)
    print("HEAD-TO-HEAD: ADX 20-30, 14-bar vs 24-bar — 50/50 IS/OOS")
    print("=" * 100)

    for label, col in [("ADX-14 (70min)", "adx"), ("ADX-24 (2hr)", "adx24")]:
        sub = tf[(tf[col] >= 20) & (tf[col] < 30)].reset_index(drop=True)
        ssp = len(sub) // 2
        print(f"\n  {label}:")
        print_stats("    IS ", calc(sub.iloc[:ssp]))
        print_stats("    OOS", calc(sub.iloc[ssp:]))
        print_stats("    ALL", calc(sub))

    # ── 5) ADX distribution comparison ────────────────────────────────────────
    print("\n" + "=" * 100)
    print("ADX DISTRIBUTION AT ENTRY")
    print("=" * 100)
    for label, col in [("ADX-14", "adx"), ("ADX-24", "adx24")]:
        v = tf[col].dropna()
        print(f"  {label}: mean={v.mean():.1f} med={v.median():.1f} std={v.std():.1f} "
              f"min={v.min():.1f} max={v.max():.1f}")
        # What pct falls in 20-30?
        in_range = ((v >= 20) & (v < 30)).sum()
        print(f"         {in_range}/{len(v)} ({in_range/len(v)*100:.0f}%) in 20-30 range")
    print()


if __name__ == "__main__":
    main()

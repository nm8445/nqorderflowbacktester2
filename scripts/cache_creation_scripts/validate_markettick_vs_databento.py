"""
Validate MarketTick-built caches against Databento-built caches.

Compares signal_cache_5yr vs signal_cache and timebars_5min_5yr vs timebars_5min
for the overlap period (Mar-Apr 2025) to verify that:
  1. Range bar counts are similar
  2. Buy/sell volume and delta direction match
  3. Absorption signals fire at similar times

Usage:
    python -u scripts/cache_creation_scripts/validate_markettick_vs_databento.py
"""
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

DATA_DIR = Path("D:/trading_pythonbacktest_data")
DATABENTO_SIGNAL = DATA_DIR / "signal_cache"
MARKETTICK_SIGNAL = DATA_DIR / "signal_cache_5yr"
DATABENTO_5MIN = DATA_DIR / "timebars_5min"
MARKETTICK_5MIN = DATA_DIR / "timebars_5min_5yr"

ET = "America/New_York"


def compare_signal_caches(dates):
    """Compare range bars between Databento and MarketTick signal caches."""
    print("=" * 90)
    print("SIGNAL CACHE COMPARISON: Range bars (Databento vs MarketTick)")
    print("=" * 90)

    results = []
    for d in dates:
        db_file = DATABENTO_SIGNAL / f"{d}.pkl"
        mt_file = MARKETTICK_SIGNAL / f"{d}.pkl"

        if not db_file.exists() or not mt_file.exists():
            continue

        with open(db_file, 'rb') as f:
            db_data = pickle.load(f)
        with open(mt_file, 'rb') as f:
            mt_data = pickle.load(f)

        db_bars = [b for b in db_data["bars"] if b.closed]
        mt_bars = [b for b in mt_data["bars"] if b.closed]

        # Total volume comparison
        db_total_buy = sum(b.buy_vol for b in db_bars)
        db_total_sell = sum(b.sell_vol for b in db_bars)
        mt_total_buy = sum(b.buy_vol for b in mt_bars)
        mt_total_sell = sum(b.sell_vol for b in mt_bars)

        db_total_delta = db_total_buy - db_total_sell
        mt_total_delta = mt_total_buy - mt_total_sell

        # Check if delta direction matches
        delta_sign_match = (db_total_delta > 0) == (mt_total_delta > 0) if db_total_delta != 0 else True

        # Absorption signal comparison
        db_signals = db_data.get("all_signals", [])
        mt_signals = mt_data.get("all_signals", [])

        results.append({
            "date": d,
            "db_bars": len(db_bars),
            "mt_bars": len(mt_bars),
            "bar_diff": len(mt_bars) - len(db_bars),
            "db_buy": db_total_buy,
            "mt_buy": mt_total_buy,
            "db_sell": db_total_sell,
            "mt_sell": mt_total_sell,
            "db_delta": db_total_delta,
            "mt_delta": mt_total_delta,
            "delta_sign": delta_sign_match,
            "db_signals": len(db_signals),
            "mt_signals": len(mt_signals),
        })

    if not results:
        print("No overlap dates found!")
        return

    # Print per-day table
    print(f"\n{'Date':>12} | {'DB Bars':>7} {'MT Bars':>7} {'Diff':>5} | "
          f"{'DB Buy':>8} {'MT Buy':>8} {'Buy%':>5} | "
          f"{'DB Sell':>8} {'MT Sell':>8} {'Sell%':>5} | "
          f"{'DB Delta':>9} {'MT Delta':>9} {'Sign':>4} | "
          f"{'DB Sig':>6} {'MT Sig':>6}")
    print("-" * 150)

    for r in results:
        buy_ratio = r["mt_buy"] / r["db_buy"] * 100 if r["db_buy"] > 0 else 0
        sell_ratio = r["mt_sell"] / r["db_sell"] * 100 if r["db_sell"] > 0 else 0
        sign = "OK" if r["delta_sign"] else "FLIP"
        print(f"{r['date']:>12} | {r['db_bars']:>7,} {r['mt_bars']:>7,} {r['bar_diff']:>+5} | "
              f"{r['db_buy']:>8,} {r['mt_buy']:>8,} {buy_ratio:>4.0f}% | "
              f"{r['db_sell']:>8,} {r['mt_sell']:>8,} {sell_ratio:>4.0f}% | "
              f"{r['db_delta']:>+9,} {r['mt_delta']:>+9,} {sign:>4} | "
              f"{r['db_signals']:>6} {r['mt_signals']:>6}")

    # Summary
    bar_diffs = [abs(r["bar_diff"]) for r in results]
    buy_ratios = [r["mt_buy"] / r["db_buy"] for r in results if r["db_buy"] > 0]
    sell_ratios = [r["mt_sell"] / r["db_sell"] for r in results if r["db_sell"] > 0]
    delta_matches = sum(1 for r in results if r["delta_sign"])

    print(f"\nSummary ({len(results)} days):")
    print(f"  Bar count diff: mean={np.mean(bar_diffs):.1f}, max={max(bar_diffs)}")
    print(f"  MT/DB buy vol ratio:  mean={np.mean(buy_ratios):.3f}, min={min(buy_ratios):.3f}, max={max(buy_ratios):.3f}")
    print(f"  MT/DB sell vol ratio: mean={np.mean(sell_ratios):.3f}, min={min(sell_ratios):.3f}, max={max(sell_ratios):.3f}")
    print(f"  Delta sign match: {delta_matches}/{len(results)} ({100*delta_matches/len(results):.0f}%)")

    # Check if buy/sell mapping is inverted by looking at correlation
    db_deltas = [r["db_delta"] for r in results]
    mt_deltas = [r["mt_delta"] for r in results]
    corr = np.corrcoef(db_deltas, mt_deltas)[0, 1]
    print(f"  Delta correlation: {corr:.3f}")
    if corr < -0.5:
        print("  WARNING: Negative correlation — buy/sell sides may be INVERTED!")
    elif corr > 0.5:
        print("  GOOD: Positive correlation — buy/sell mapping looks correct")


def compare_5min_bars(dates):
    """Compare 5-min time bars."""
    print(f"\n{'='*90}")
    print("5-MIN BARS COMPARISON: Databento vs MarketTick")
    print("=" * 90)

    results = []
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        db_file = DATABENTO_5MIN / f"timebars_5min_{dt.strftime('%Y_%m_%d')}.pkl"
        mt_file = MARKETTICK_5MIN / f"timebars_5min_{dt.strftime('%Y_%m_%d')}.pkl"

        if not db_file.exists() or not mt_file.exists():
            continue

        with open(db_file, 'rb') as f:
            db_bars = pickle.load(f)
        with open(mt_file, 'rb') as f:
            mt_bars = pickle.load(f)

        db_total_vol = sum(b.get("volume", b.get("buy_vol", 0) + b.get("sell_vol", 0)) for b in db_bars)
        mt_total_vol = sum(b.get("volume", b.get("buy_vol", 0) + b.get("sell_vol", 0)) for b in mt_bars)

        # Check if buy_vol/sell_vol exist in Databento bars
        db_buy = sum(b.get("buy_vol", 0) for b in db_bars)
        db_sell = sum(b.get("sell_vol", 0) for b in db_bars)
        mt_buy = sum(b.get("buy_vol", 0) for b in mt_bars)
        mt_sell = sum(b.get("sell_vol", 0) for b in mt_bars)

        # OHLC comparison — match bars by open_time and compare close prices
        db_by_time = {b["open_time"]: b for b in db_bars}
        mt_by_time = {b["open_time"]: b for b in mt_bars}
        common_times = set(db_by_time.keys()) & set(mt_by_time.keys())
        close_diffs = []
        for t in common_times:
            diff = abs(db_by_time[t]["close"] - mt_by_time[t]["close"])
            close_diffs.append(diff)

        results.append({
            "date": d,
            "db_count": len(db_bars),
            "mt_count": len(mt_bars),
            "db_vol": db_total_vol,
            "mt_vol": mt_total_vol,
            "vol_ratio": mt_total_vol / db_total_vol if db_total_vol > 0 else 0,
            "db_buy": db_buy,
            "mt_buy": mt_buy,
            "db_sell": db_sell,
            "mt_sell": mt_sell,
            "common_bars": len(common_times),
            "close_diff_mean": np.mean(close_diffs) if close_diffs else 0,
            "close_diff_max": max(close_diffs) if close_diffs else 0,
        })

    if not results:
        print("No overlap dates found!")
        return

    print(f"\n{'Date':>12} | {'DB#':>4} {'MT#':>4} | {'DB Vol':>9} {'MT Vol':>9} {'Ratio':>5} | "
          f"{'DB Buy':>8} {'MT Buy':>8} | {'DB Sell':>8} {'MT Sell':>8} | "
          f"{'Match':>5} {'ClsDiff':>7}")
    print("-" * 120)

    for r in results:
        print(f"{r['date']:>12} | {r['db_count']:>4} {r['mt_count']:>4} | "
              f"{r['db_vol']:>9,} {r['mt_vol']:>9,} {r['vol_ratio']:>4.2f}x | "
              f"{r['db_buy']:>8,} {r['mt_buy']:>8,} | "
              f"{r['db_sell']:>8,} {r['mt_sell']:>8,} | "
              f"{r['common_bars']:>5} {r['close_diff_mean']:>7.2f}")

    vol_ratios = [r["vol_ratio"] for r in results]
    print(f"\nSummary ({len(results)} days):")
    print(f"  Volume ratio (MT/DB): mean={np.mean(vol_ratios):.3f}")
    print(f"  Close price diff: mean={np.mean([r['close_diff_mean'] for r in results]):.3f} pts")


def main():
    # Find overlap dates (dates that exist in both Databento and MarketTick caches)
    db_dates = {f.stem for f in DATABENTO_SIGNAL.glob("*.pkl")}
    mt_dates = {f.stem for f in MARKETTICK_SIGNAL.glob("*.pkl")}
    overlap = sorted(db_dates & mt_dates)

    print(f"Databento signal_cache: {len(db_dates)} dates")
    print(f"MarketTick signal_cache_5yr: {len(mt_dates)} dates")
    print(f"Overlap: {len(overlap)} dates")
    if overlap:
        print(f"Range: {overlap[0]} to {overlap[-1]}")
    print()

    if not overlap:
        print("No overlap found! Run the MarketTick signal cache builder first.")
        return

    # Use first 10 overlap days for validation
    sample = overlap[:10]
    print(f"Validating on {len(sample)} sample days: {sample[0]} to {sample[-1]}\n")

    compare_signal_caches(sample)
    compare_5min_bars(sample)


if __name__ == "__main__":
    main()

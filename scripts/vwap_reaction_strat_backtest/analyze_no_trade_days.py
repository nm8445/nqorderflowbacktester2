"""
Analyze no-trade days: trending (price consistently on one side of VWAP) vs choppy.

Uses 14-bar rolling window on 5-min bars to measure VWAP position consistency.
No-trade days = days with 0 raw signals in vwap_reaction_cache (no absorption at VWAP zone
with matching bias/approach). ADX filtering is separate (backtest-time).
"""
import sys
import pickle
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

VWAP_CACHE = Path("D:/trading_pythonbacktest_data/vwap_cache")
TIMEBAR_CACHE = Path("D:/trading_pythonbacktest_data/timebars_5min")
SIGNAL_CACHE = Path("D:/trading_pythonbacktest_data/vwap_reaction_cache")

ROLLING_WINDOW = 14  # 14 x 5-min = 70 min
VWAP_ZONE = 3.0  # pts
TREND_THRESHOLD = 0.85  # 85%+ on one side = trending


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_day(date_str):
    """Analyze a single day's VWAP position consistency."""
    vwap_path = VWAP_CACHE / f"{date_str}.pkl"
    tb_path = TIMEBAR_CACHE / f"timebars_5min_{date_str.replace('-', '_')}.pkl"

    if not vwap_path.exists() or not tb_path.exists():
        return None

    vwap_df = load_pickle(vwap_path)
    time_bars = load_pickle(tb_path)

    if len(time_bars) < ROLLING_WINDOW:
        return None

    # Build series of 5-min close vs VWAP
    closes = []
    vwap_vals = []

    for bar in time_bars:
        close_time = bar["close_time"]
        close_price = bar["close"]

        # Look up VWAP at this time
        if hasattr(vwap_df.index, 'tz') and vwap_df.index.tz is not None:
            ct = close_time if close_time.tzinfo else close_time.tz_localize("UTC")
        else:
            ct = close_time.tz_localize(None) if hasattr(close_time, 'tz_localize') and close_time.tzinfo else close_time

        mask = vwap_df.index <= ct
        if not mask.any():
            continue

        vwap_val = vwap_df.loc[mask, "vwap"].iloc[-1]
        closes.append(close_price)
        vwap_vals.append(vwap_val)

    if len(closes) < ROLLING_WINDOW:
        return None

    closes = np.array(closes)
    vwap_vals = np.array(vwap_vals)
    diffs = closes - vwap_vals

    # Position: above=1, below=-1
    positions = np.where(diffs > 0, 1, -1)

    # Rolling consistency: max(% above, % below) over 14-bar windows
    consistencies = []
    for i in range(ROLLING_WINDOW - 1, len(positions)):
        window = positions[i - ROLLING_WINDOW + 1:i + 1]
        pct_above = np.mean(window == 1)
        pct_below = np.mean(window == -1)
        consistencies.append(max(pct_above, pct_below))

    avg_consistency = np.mean(consistencies)
    max_consistency = np.max(consistencies)

    # Time near VWAP zone
    near_vwap = np.abs(diffs) <= VWAP_ZONE
    pct_near_vwap = np.mean(near_vwap) * 100

    # Median distance from VWAP
    median_dist = np.median(np.abs(diffs))

    # Dominant side
    pct_above_all = np.mean(positions == 1) * 100

    return {
        "avg_consistency": avg_consistency,
        "max_consistency": max_consistency,
        "pct_near_vwap": pct_near_vwap,
        "median_dist": median_dist,
        "pct_above": pct_above_all,
        "n_bars": len(closes),
    }


def main():
    signal_files = sorted(SIGNAL_CACHE.glob("*.pkl"))
    if not signal_files:
        print("No signal cache files found")
        return

    trade_days = []
    no_trade_days = []

    for sf in signal_files:
        date_str = sf.stem
        data = load_pickle(sf)
        n_signals = data["num_signals"]

        result = analyze_day(date_str)
        if result is None:
            continue

        result["date"] = date_str
        result["n_signals"] = n_signals

        if n_signals > 0:
            trade_days.append(result)
        else:
            no_trade_days.append(result)

    print(f"Total days analyzed: {len(trade_days) + len(no_trade_days)}")
    print(f"  Days with signals: {len(trade_days)}")
    print(f"  Days with 0 signals (no-trade): {len(no_trade_days)}")
    print()

    # Classify no-trade days
    trending = [d for d in no_trade_days if d["avg_consistency"] >= TREND_THRESHOLD]
    choppy = [d for d in no_trade_days if d["avg_consistency"] < TREND_THRESHOLD]

    print("=" * 60)
    print("NO-TRADE DAY CLASSIFICATION")
    print("=" * 60)
    print(f"  Trending (avg consistency >= {TREND_THRESHOLD:.0%}): {len(trending)} ({100*len(trending)/max(1,len(no_trade_days)):.0f}%)")
    print(f"  Choppy   (avg consistency <  {TREND_THRESHOLD:.0%}): {len(choppy)} ({100*len(choppy)/max(1,len(no_trade_days)):.0f}%)")
    print()

    # Stats comparison
    def show_stats(label, days):
        if not days:
            print(f"  {label}: no days")
            return
        avg_con = np.mean([d["avg_consistency"] for d in days])
        avg_near = np.mean([d["pct_near_vwap"] for d in days])
        avg_dist = np.mean([d["median_dist"] for d in days])
        print(f"  {label} ({len(days)} days):")
        print(f"    Avg rolling consistency: {avg_con:.1%}")
        print(f"    Avg % time near VWAP (+-{VWAP_ZONE}pts): {avg_near:.1f}%")
        print(f"    Avg median distance from VWAP: {avg_dist:.1f} pts")

    show_stats("Trade days", trade_days)
    show_stats("No-trade (TRENDING)", trending)
    show_stats("No-trade (CHOPPY)", choppy)
    print()

    # Distribution of consistency values
    print("=" * 60)
    print("CONSISTENCY DISTRIBUTION")
    print("=" * 60)
    print(f"  {'Bucket':<12} {'Trade':>8} {'No-Trade':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10}")
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        t_count = sum(1 for d in trade_days if lo <= d["avg_consistency"] < hi)
        nt_count = sum(1 for d in no_trade_days if lo <= d["avg_consistency"] < hi)
        label = f"{lo:.0%}-{hi:.0%}"
        t_bar = "#" * t_count
        nt_bar = "X" * nt_count
        print(f"  {label:<12} {t_count:>4}  {t_bar}")
        print(f"  {'':12} {nt_count:>4}  {nt_bar}")
    print()

    # Individual no-trade days sorted by consistency
    print("=" * 60)
    print("NO-TRADE DAYS DETAIL (sorted by consistency desc)")
    print("=" * 60)
    no_trade_sorted = sorted(no_trade_days, key=lambda d: d["avg_consistency"], reverse=True)
    print(f"  {'Date':<12} {'AvgCon':>7} {'NearVWAP':>9} {'MedDist':>8} {'%Above':>7} {'Type':<10}")
    print(f"  {'-'*12} {'-'*7} {'-'*9} {'-'*8} {'-'*7} {'-'*10}")
    for d in no_trade_sorted:
        dtype = "TRENDING" if d["avg_consistency"] >= TREND_THRESHOLD else "CHOPPY"
        print(f"  {d['date']:<12} {d['avg_consistency']:>6.1%} {d['pct_near_vwap']:>8.1f}% {d['median_dist']:>7.1f} {d['pct_above']:>6.1f}% {dtype:<10}")

    print()
    print("=" * 60)
    print("TRADE vs NO-TRADE COMPARISON")
    print("=" * 60)
    if trade_days and no_trade_days:
        t_con = np.mean([d["avg_consistency"] for d in trade_days])
        t_near = np.mean([d["pct_near_vwap"] for d in trade_days])
        t_dist = np.mean([d["median_dist"] for d in trade_days])
        nt_con = np.mean([d["avg_consistency"] for d in no_trade_days])
        nt_near = np.mean([d["pct_near_vwap"] for d in no_trade_days])
        nt_dist = np.mean([d["median_dist"] for d in no_trade_days])

        print(f"  {'Metric':<35} {'Trade':>10} {'No-Trade':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*10}")
        print(f"  {'Avg rolling consistency':<35} {t_con:>9.1%} {nt_con:>9.1%}")
        print(f"  {'Avg % near VWAP zone':<35} {t_near:>9.1f}% {nt_near:>9.1f}%")
        print(f"  {'Avg median dist from VWAP':<35} {t_dist:>8.1f}p {nt_dist:>8.1f}p")

    # Summary insight
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if no_trade_days:
        pct_trend = 100 * len(trending) / len(no_trade_days)
        pct_chop = 100 * len(choppy) / len(no_trade_days)
        print(f"  {len(no_trade_days)} no-trade days out of {len(trade_days)+len(no_trade_days)} total ({100*len(no_trade_days)/(len(trade_days)+len(no_trade_days)):.0f}%)")
        print(f"  {len(trending)} ({pct_trend:.0f}%) were TRENDING - price stayed on one side of VWAP")
        print(f"    -> Strategy correctly avoided these (no pullback to VWAP zone)")
        print(f"  {len(choppy)} ({pct_chop:.0f}%) were CHOPPY - price crossed VWAP but no absorption signal")
        print(f"    -> Either no absorption at zone, or bias/approach didn't match")


if __name__ == "__main__":
    main()

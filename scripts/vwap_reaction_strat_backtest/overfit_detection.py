"""
Overfit Detection Suite — Trending VWAP Band Strategy (STD1, LB=14, ADX 30+)

Runs 5 tests:
  1. Parameter stability heatmap (SL/TP neighbors)
  2. Walk-forward rolling windows (3-month IS -> 3-month OOS)
  3. Monte Carlo shuffle (10,000 iterations)
  4. Trade-level bootstrap (10,000 resamples)
  5. Signal-level permutation (10,000 random direction assignments)

Usage:
    python -u scripts/vwap_reaction_strat_backtest/overfit_detection.py
"""

import pickle
import sys
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import build_adx_lookup, get_adx_at_time
from trending_vwap_atr_grid import (
    load_pickle, load_5min_bars,
    precompute_day_state, detect_signals, simulate, calc_stats,
    SIGNAL_CACHE_DIR, VWAP_CACHE_DIR, TIMEBARS_5MIN_DIR, POINT_VALUE,
)

ET = "America/New_York"
START_DATE = "2025-03-13"
END_DATE = "2026-04-08"
LOOKBACK = 14
STD_BAND = 1
TARGET_SL = 1.00
TARGET_TP = 1.00
N_ITERATIONS = 10000
ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"


def load_all_data():
    """Load all day data and precompute signals."""
    print("Building ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done ({len(adx_lookup)} rows)")

    print("Pre-loading all day data...", end=" ", flush=True)
    start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    day_data = {}
    for vf in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        ds = vf.stem
        fd = datetime.strptime(ds, "%Y-%m-%d").date()
        if fd < start or fd > end:
            continue
        sf = SIGNAL_CACHE_DIR / f"{ds}.pkl"
        tf = TIMEBARS_5MIN_DIR / f"timebars_5min_{ds.replace('-','_')}.pkl"
        if not sf.exists() or not tf.exists():
            continue
        bars = load_pickle(sf)["bars"]
        bars_5min = load_5min_bars(ds)
        vwap_df = load_pickle(vf)
        if bars and bars_5min is not None:
            day_data[ds] = (bars, bars_5min, vwap_df)
    print(f"done ({len(day_data)} days)")

    # Precompute signals with ADX filter
    print("Computing signals...", end=" ", flush=True)
    signals_by_day = {}
    for ds, (bars, b5, vdf) in day_data.items():
        state = precompute_day_state(b5, vdf, LOOKBACK, STD_BAND)
        sigs = detect_signals(bars, state)
        filtered = []
        for s in sigs:
            s["date"] = ds
            ct = s["confirm_time"]
            ct_et = ct.tz_convert("America/New_York") if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert("America/New_York")
            adx = get_adx_at_time(ct_et, adx_lookup)
            if not np.isnan(adx) and adx >= 30.0:
                filtered.append(s)
        if filtered:
            signals_by_day[ds] = filtered
    total_sigs = sum(len(v) for v in signals_by_day.values())
    print(f"done ({total_sigs} signals on {len(signals_by_day)} days)")
    sys.stdout.flush()

    return day_data, signals_by_day


def get_all_trades(signals_by_day, day_data, sl, tp):
    """Run simulation across all days, return list of trades."""
    trades = []
    for ds in sorted(signals_by_day.keys()):
        bars = day_data[ds][0]
        trades.extend(simulate(signals_by_day[ds], bars, sl, tp))
    return trades


# ============================================================
# TEST 1: Parameter Stability
# ============================================================
def test_parameter_stability(signals_by_day, day_data):
    print("\n" + "=" * 80)
    print("  TEST 1: PARAMETER STABILITY HEATMAP")
    print("  Do neighboring SL/TP configs also work? (plateau = good, spike = overfit)")
    print("=" * 80)
    sys.stdout.flush()

    mults = [round(0.50 + i * 0.25, 2) for i in range(7)]  # 0.50 to 2.00

    # Print header
    print(f"\n  PF Heatmap (rows=SL, cols=TP):")
    print(f"  {'SL\\TP':>6}", end="")
    for tp in mults:
        print(f" {tp:>6.2f}", end="")
    print()
    print(f"  {'':>6}", end="")
    for _ in mults:
        print(f" {'------':>6}", end="")
    print()

    pf_grid = {}
    for sl in mults:
        print(f"  {sl:>6.2f}", end="")
        for tp in mults:
            trades = get_all_trades(signals_by_day, day_data, sl, tp)
            stats = calc_stats(trades)
            pf = stats["pf"] if stats else 0
            pf_grid[(sl, tp)] = pf
            marker = " ***" if sl == TARGET_SL and tp == TARGET_TP else ""
            if pf >= 1.5:
                print(f" {pf:>5.2f}*", end="")
            elif pf >= 1.0:
                print(f" {pf:>5.2f} ", end="")
            else:
                print(f" {pf:>5.2f}.", end="")
        print()

    print(f"\n  Legend: * = PF >= 1.5 (strong)  [space] = PF >= 1.0  . = PF < 1.0 (losing)")
    print(f"  *** marks target config (SL={TARGET_SL}, TP={TARGET_TP})")

    # Count profitable neighbors
    target_pf = pf_grid.get((TARGET_SL, TARGET_TP), 0)
    neighbors = []
    for dsl in [-0.25, 0, 0.25]:
        for dtp in [-0.25, 0, 0.25]:
            if dsl == 0 and dtp == 0:
                continue
            key = (round(TARGET_SL + dsl, 2), round(TARGET_TP + dtp, 2))
            if key in pf_grid:
                neighbors.append((key, pf_grid[key]))

    profitable_neighbors = sum(1 for _, pf in neighbors if pf >= 1.0)
    print(f"\n  Target PF: {target_pf:.2f}")
    print(f"  Neighbors profitable (PF >= 1.0): {profitable_neighbors}/{len(neighbors)}")
    for (sl, tp), pf in neighbors:
        status = "OK" if pf >= 1.0 else "LOSING"
        print(f"    SL={sl:.2f} TP={tp:.2f} -> PF {pf:.2f} [{status}]")

    # Expectancy heatmap
    print(f"\n  Expectancy Heatmap (rows=SL, cols=TP, in $ per trade):")
    print(f"  {'SL\\TP':>6}", end="")
    for tp in mults:
        print(f" {tp:>7.2f}", end="")
    print()
    for sl in mults:
        print(f"  {sl:>6.2f}", end="")
        for tp in mults:
            trades = get_all_trades(signals_by_day, day_data, sl, tp)
            stats = calc_stats(trades)
            exp = stats["exp"] * POINT_VALUE if stats else 0
            print(f" {exp:>+7.0f}", end="")
        print()

    # DD heatmap
    print(f"\n  Max DD Heatmap (rows=SL, cols=TP, in $):")
    print(f"  {'SL\\TP':>6}", end="")
    for tp in mults:
        print(f" {tp:>7.2f}", end="")
    print()
    for sl in mults:
        print(f"  {sl:>6.2f}", end="")
        for tp in mults:
            trades = get_all_trades(signals_by_day, day_data, sl, tp)
            stats = calc_stats(trades)
            dd = stats["dd"] * POINT_VALUE if stats else 0
            print(f" {dd:>+7.0f}", end="")
        print()

    sys.stdout.flush()
    return pf_grid


# ============================================================
# TEST 2: Walk-Forward Rolling Windows
# ============================================================
def test_walk_forward(signals_by_day, day_data):
    print("\n" + "=" * 80)
    print("  TEST 2: WALK-FORWARD ANALYSIS")
    print("  3-month IS -> 3-month OOS, rolling forward by 3 months")
    print("=" * 80)
    sys.stdout.flush()

    all_dates = sorted(signals_by_day.keys())
    if not all_dates:
        print("  No signals found!")
        return

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    # Build 3-month windows
    windows = []
    is_start = start_dt
    while True:
        is_end = is_start + timedelta(days=90)
        oos_start = is_end + timedelta(days=1)
        oos_end = oos_start + timedelta(days=90)
        if oos_end > end_dt:
            oos_end = end_dt
        if oos_start >= end_dt:
            break
        windows.append((is_start, is_end, oos_start, oos_end))
        is_start = oos_start  # slide forward

    print(f"\n  {len(windows)} walk-forward windows:")
    print(f"  {'Window':>8} {'IS Period':>25} {'OOS Period':>25} | {'IS Tr':>5} {'IS WR':>6} {'IS PF':>6} {'IS P&L':>9} | {'OOS Tr':>6} {'OOS WR':>7} {'OOS PF':>7} {'OOS P&L':>9}")
    print(f"  {'--------':>8} {'-------------------------':>25} {'-------------------------':>25} | {'-----':>5} {'------':>6} {'------':>6} {'---------':>9} | {'------':>6} {'-------':>7} {'-------':>7} {'---------':>9}")

    oos_pfs = []
    oos_totals = []
    for idx, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        is_trades = []
        oos_trades = []
        for ds, sigs in signals_by_day.items():
            fd = datetime.strptime(ds, "%Y-%m-%d").date()
            bars = day_data[ds][0]
            if is_s <= fd <= is_e:
                is_trades.extend(simulate(sigs, bars, TARGET_SL, TARGET_TP))
            elif oos_s <= fd <= oos_e:
                oos_trades.extend(simulate(sigs, bars, TARGET_SL, TARGET_TP))

        is_stats = calc_stats(is_trades)
        oos_stats = calc_stats(oos_trades)

        is_tr = is_stats["trades"] if is_stats else 0
        is_wr = is_stats["wr"] if is_stats else 0
        is_pf = is_stats["pf"] if is_stats else 0
        is_pnl = is_stats["total"] * POINT_VALUE if is_stats else 0

        oos_tr = oos_stats["trades"] if oos_stats else 0
        oos_wr = oos_stats["wr"] if oos_stats else 0
        oos_pf = oos_stats["pf"] if oos_stats else 0
        oos_pnl = oos_stats["total"] * POINT_VALUE if oos_stats else 0

        if oos_stats:
            oos_pfs.append(oos_pf)
            oos_totals.append(oos_pnl)

        print(f"  {idx+1:>8} {str(is_s)+' -> '+str(is_e):>25} {str(oos_s)+' -> '+str(oos_e):>25} | "
              f"{is_tr:>5} {is_wr:>5.1f}% {is_pf:>6.2f} ${is_pnl:>+8,.0f} | "
              f"{oos_tr:>6} {oos_wr:>6.1f}% {oos_pf:>7.2f} ${oos_pnl:>+8,.0f}")

    if oos_pfs:
        profitable_windows = sum(1 for p in oos_pfs if p >= 1.0)
        print(f"\n  OOS Summary:")
        print(f"    Windows with PF >= 1.0: {profitable_windows}/{len(oos_pfs)} ({100*profitable_windows/len(oos_pfs):.0f}%)")
        print(f"    Avg OOS PF: {np.mean(oos_pfs):.2f}")
        print(f"    Total OOS P&L: ${sum(oos_totals):+,.0f}")
        print(f"    Verdict: {'PASS — edge persists across periods' if profitable_windows >= len(oos_pfs)*0.6 else 'CONCERN — inconsistent OOS performance'}")

    sys.stdout.flush()


# ============================================================
# TEST 3: Monte Carlo Shuffle
# ============================================================
def test_monte_carlo_shuffle(trades):
    print("\n" + "=" * 80)
    print("  TEST 3: MONTE CARLO SHUFFLE (10,000 iterations)")
    print("  Randomly shuffle trade P&Ls to test if sequence matters")
    print("=" * 80)
    sys.stdout.flush()

    pnls = np.array([t["pnl"] for t in trades])
    real_total = pnls.sum() * POINT_VALUE
    real_cum = np.cumsum(pnls)
    real_dd = (real_cum - np.maximum.accumulate(real_cum)).min() * POINT_VALUE
    real_pf_val = calc_stats(trades)["pf"]

    shuffle_totals = []
    shuffle_dds = []
    shuffle_pfs = []

    rng = np.random.default_rng(42)
    for _ in range(N_ITERATIONS):
        shuffled = rng.permutation(pnls)
        cum = np.cumsum(shuffled)
        dd = (cum - np.maximum.accumulate(cum)).min()

        total = shuffled.sum()
        wins = shuffled[shuffled > 0]
        losses = shuffled[shuffled < 0]
        gp = wins.sum() if len(wins) > 0 else 0
        gl = abs(losses.sum()) if len(losses) > 0 else 1
        pf = gp / gl if gl > 0 else float("inf")

        shuffle_totals.append(total * POINT_VALUE)
        shuffle_dds.append(dd * POINT_VALUE)
        shuffle_pfs.append(pf)

    shuffle_totals = np.array(shuffle_totals)
    shuffle_dds = np.array(shuffle_dds)
    shuffle_pfs = np.array(shuffle_pfs)

    # Total P&L is same for all shuffles (same trades), so percentile is meaningless for total
    # What matters: DD percentile (is our DD better than random ordering?)
    dd_percentile = np.mean(real_dd > shuffle_dds) * 100  # % of shuffles with WORSE DD
    pf_percentile = np.mean(real_pf_val > shuffle_pfs) * 100

    print(f"\n  Real Results:")
    print(f"    Total P&L: ${real_total:+,.0f}")
    print(f"    Max DD:    ${real_dd:+,.0f}")
    print(f"    PF:        {real_pf_val:.2f}")

    print(f"\n  Shuffle Distribution (10,000 random orderings):")
    print(f"    DD — Real: ${real_dd:+,.0f} | Shuffle median: ${np.median(shuffle_dds):+,.0f} | "
          f"5th: ${np.percentile(shuffle_dds, 5):+,.0f} | 95th: ${np.percentile(shuffle_dds, 95):+,.0f}")
    print(f"    Real DD is better than {dd_percentile:.1f}% of shuffles")

    print(f"\n  Note: Since total P&L is identical for all shuffles (same trades, just reordered),")
    print(f"  this test evaluates whether the SEQUENCE of your trades creates unusually good or bad DD.")
    print(f"  Verdict: {'PASS — DD is favorable vs random ordering' if dd_percentile >= 50 else 'NOTE — DD is worse than typical random ordering (unlucky sequencing)'}")

    sys.stdout.flush()


# ============================================================
# TEST 4: Trade-Level Bootstrap
# ============================================================
def test_bootstrap(trades):
    print("\n" + "=" * 80)
    print("  TEST 4: TRADE-LEVEL BOOTSTRAP (10,000 resamples)")
    print("  Resample trades with replacement -> confidence intervals on PF, WR, DD")
    print("=" * 80)
    sys.stdout.flush()

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)

    boot_pfs = []
    boot_wrs = []
    boot_exps = []
    boot_totals = []
    boot_dds = []

    rng = np.random.default_rng(42)
    for _ in range(N_ITERATIONS):
        sample = rng.choice(pnls, size=n, replace=True)

        wins = sample[sample > 0]
        losses = sample[sample < 0]
        wr = len(wins) / n * 100
        gp = wins.sum() if len(wins) > 0 else 0
        gl = abs(losses.sum()) if len(losses) > 0 else 1
        pf = gp / gl if gl > 0 else float("inf")
        exp = sample.mean()
        total = sample.sum()
        cum = np.cumsum(sample)
        dd = (cum - np.maximum.accumulate(cum)).min()

        boot_pfs.append(pf)
        boot_wrs.append(wr)
        boot_exps.append(exp * POINT_VALUE)
        boot_totals.append(total * POINT_VALUE)
        boot_dds.append(dd * POINT_VALUE)

    boot_pfs = np.array(boot_pfs)
    boot_wrs = np.array(boot_wrs)
    boot_exps = np.array(boot_exps)
    boot_totals = np.array(boot_totals)
    boot_dds = np.array(boot_dds)

    real_stats = calc_stats(trades)

    print(f"\n  Real: {n} trades | WR {real_stats['wr']:.1f}% | PF {real_stats['pf']:.2f} | "
          f"Exp ${real_stats['exp']*POINT_VALUE:+.0f} | Total ${real_stats['total']*POINT_VALUE:+,.0f}")

    print(f"\n  Bootstrap 95% Confidence Intervals (10,000 resamples):")
    print(f"    {'Metric':<15} {'2.5th':>10} {'Median':>10} {'97.5th':>10} {'Real':>10}")
    print(f"    {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"    {'Win Rate':.<15} {np.percentile(boot_wrs, 2.5):>9.1f}% {np.median(boot_wrs):>9.1f}% {np.percentile(boot_wrs, 97.5):>9.1f}% {real_stats['wr']:>9.1f}%")
    print(f"    {'Profit Factor':.<15} {np.percentile(boot_pfs, 2.5):>10.2f} {np.median(boot_pfs):>10.2f} {np.percentile(boot_pfs, 97.5):>10.2f} {real_stats['pf']:>10.2f}")
    print(f"    {'Expectancy':.<15} ${np.percentile(boot_exps, 2.5):>+9.0f} ${np.median(boot_exps):>+9.0f} ${np.percentile(boot_exps, 97.5):>+9.0f} ${real_stats['exp']*POINT_VALUE:>+9.0f}")
    print(f"    {'Total P&L':.<15} ${np.percentile(boot_totals, 2.5):>+9,.0f} ${np.median(boot_totals):>+9,.0f} ${np.percentile(boot_totals, 97.5):>+9,.0f} ${real_stats['total']*POINT_VALUE:>+9,.0f}")
    print(f"    {'Max DD':.<15} ${np.percentile(boot_dds, 2.5):>+9,.0f} ${np.median(boot_dds):>+9,.0f} ${np.percentile(boot_dds, 97.5):>+9,.0f} ${(real_stats['dd']*POINT_VALUE):>+9,.0f}")

    # Key question: how often is PF < 1.0 (losing)?
    pct_losing = np.mean(boot_pfs < 1.0) * 100
    print(f"\n  Probability of being a losing strategy (PF < 1.0): {pct_losing:.1f}%")
    print(f"  Probability of PF >= 1.5: {np.mean(boot_pfs >= 1.5)*100:.1f}%")

    if pct_losing < 5:
        print(f"  Verdict: PASS — less than 5% chance of being a losing strategy")
    elif pct_losing < 15:
        print(f"  Verdict: MARGINAL — {pct_losing:.0f}% chance of being a loser, use caution")
    else:
        print(f"  Verdict: CONCERN — {pct_losing:.0f}% chance this is actually a losing strategy")

    sys.stdout.flush()


# ============================================================
# TEST 5: Signal Direction Permutation
# ============================================================
def test_direction_permutation(signals_by_day, day_data):
    print("\n" + "=" * 80)
    print("  TEST 5: SIGNAL DIRECTION PERMUTATION (10,000 iterations)")
    print("  Randomly assign long/short to each signal. If random directions")
    print("  produce similar results, directional logic adds no value.")
    print("=" * 80)
    sys.stdout.flush()

    # Get real trades
    real_trades = get_all_trades(signals_by_day, day_data, TARGET_SL, TARGET_TP)
    real_stats = calc_stats(real_trades)
    real_total = real_stats["total"] * POINT_VALUE
    real_pf = real_stats["pf"]

    # Collect all signals with their bars for re-simulation
    all_signals = []
    for ds in sorted(signals_by_day.keys()):
        for s in signals_by_day[ds]:
            all_signals.append((ds, s))

    rng = np.random.default_rng(42)
    perm_totals = []
    perm_pfs = []

    for iteration in range(N_ITERATIONS):
        # Randomly flip directions
        perm_signals_by_day = {}
        for ds, s in all_signals:
            s_copy = dict(s)
            s_copy["direction"] = rng.choice(["long", "short"])
            perm_signals_by_day.setdefault(ds, []).append(s_copy)

        trades = []
        for ds in sorted(perm_signals_by_day.keys()):
            bars = day_data[ds][0]
            trades.extend(simulate(perm_signals_by_day[ds], bars, TARGET_SL, TARGET_TP))

        stats = calc_stats(trades)
        if stats:
            perm_totals.append(stats["total"] * POINT_VALUE)
            perm_pfs.append(stats["pf"])

        if (iteration + 1) % 2000 == 0:
            print(f"    {iteration+1}/{N_ITERATIONS} iterations complete...", flush=True)

    perm_totals = np.array(perm_totals)
    perm_pfs = np.array(perm_pfs)

    # What percentile is the real result vs random directions?
    total_pctile = np.mean(real_total > perm_totals) * 100
    pf_pctile = np.mean(real_pf > perm_pfs) * 100

    print(f"\n  Real Strategy: Total ${real_total:+,.0f} | PF {real_pf:.2f}")
    print(f"\n  Random Direction Distribution (10,000 permutations):")
    print(f"    Total P&L — Median: ${np.median(perm_totals):+,.0f} | "
          f"5th: ${np.percentile(perm_totals, 5):+,.0f} | 95th: ${np.percentile(perm_totals, 95):+,.0f}")
    print(f"    PF — Median: {np.median(perm_pfs):.2f} | "
          f"5th: {np.percentile(perm_pfs, 5):.2f} | 95th: {np.percentile(perm_pfs, 95):.2f}")
    print(f"\n  Real total P&L is better than {total_pctile:.1f}% of random directions")
    print(f"  Real PF is better than {pf_pctile:.1f}% of random directions")

    if total_pctile >= 95:
        print(f"  Verdict: STRONG PASS — directional logic is highly significant (p < 0.05)")
    elif total_pctile >= 90:
        print(f"  Verdict: PASS — directional logic adds real value (p < 0.10)")
    elif total_pctile >= 75:
        print(f"  Verdict: MARGINAL — some directional edge but not statistically strong")
    else:
        print(f"  Verdict: FAIL — random directions do just as well, directional logic may not matter")

    sys.stdout.flush()


def main():
    print("=" * 80)
    print("  OVERFIT DETECTION SUITE")
    print(f"  Config: STD{STD_BAND} | LB={LOOKBACK} | SL={TARGET_SL}x | TP={TARGET_TP}x | ADX >= 30")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print("=" * 80)
    sys.stdout.flush()

    day_data, signals_by_day = load_all_data()

    # Get baseline trades
    all_trades = get_all_trades(signals_by_day, day_data, TARGET_SL, TARGET_TP)
    stats = calc_stats(all_trades)
    print(f"\n  Baseline: {stats['trades']} trades | WR {stats['wr']:.1f}% | PF {stats['pf']:.2f} | "
          f"Total ${stats['total']*POINT_VALUE:+,.0f} | DD ${stats['dd']*POINT_VALUE:+,.0f}")

    # Run all 5 tests
    test_parameter_stability(signals_by_day, day_data)
    test_walk_forward(signals_by_day, day_data)
    test_monte_carlo_shuffle(all_trades)
    test_bootstrap(all_trades)
    test_direction_permutation(signals_by_day, day_data)

    print("\n" + "=" * 80)
    print("  ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Trending VWAP Band — Low Drawdown Config Search

Finds trending band configs (STD1, ADX 30+) with lowest drawdown
for prop firm compatibility. Shows DD in both points and dollars.

Usage:
    python -u scripts/vwap_reaction_strat_backtest/trending_low_dd_search.py
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import build_adx_lookup, get_adx_at_time
from trending_vwap_atr_grid import (
    load_pickle, load_5min_bars,
    precompute_day_state, detect_signals, simulate, calc_stats,
    SIGNAL_CACHE_DIR, VWAP_CACHE_DIR, TIMEBARS_5MIN_DIR,
    START_DATE, END_DATE, IS_END, OOS_START, POINT_VALUE,
)

STD_BAND = 1


def main():
    print("=" * 100)
    print("  TRENDING VWAP BAND — LOW DRAWDOWN CONFIG SEARCH")
    print("  STD1 | ADX >= 30 | 50/50 IS/OOS split")
    print(f"  IS: {START_DATE} to {IS_END} | OOS: {OOS_START} to {END_DATE}")
    print("=" * 100)
    sys.stdout.flush()

    print("Building ADX lookup...", end=" ", flush=True)
    adx_lookup = build_adx_lookup()
    print(f"done ({len(adx_lookup)} rows)")

    # Pre-load all day data
    print("Pre-loading all day data...", end=" ", flush=True)
    start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end = datetime.strptime(END_DATE, "%Y-%m-%d").date()
    is_end = datetime.strptime(IS_END, "%Y-%m-%d").date()

    day_data = {}
    for vwap_file in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        ds = vwap_file.stem
        fd = datetime.strptime(ds, "%Y-%m-%d").date()
        if fd < start or fd > end:
            continue
        sf = SIGNAL_CACHE_DIR / f"{ds}.pkl"
        tf = TIMEBARS_5MIN_DIR / f"timebars_5min_{ds.replace('-','_')}.pkl"
        if not sf.exists() or not tf.exists():
            continue
        vwap_df = load_pickle(vwap_file)
        bars = load_pickle(sf)["bars"]
        bars_5min = load_5min_bars(ds)
        if bars and bars_5min is not None:
            day_data[ds] = (bars, bars_5min, vwap_df)
    print(f"done ({len(day_data)} days)")
    sys.stdout.flush()

    mults = [round(0.50 + i * 0.25, 2) for i in range(7)]  # 0.50 to 2.00
    lookbacks = [10, 14, 20]

    all_results = []

    for lookback in lookbacks:
        print(f"  Computing signals for LB={lookback}...", end=" ", flush=True)

        is_signals_by_day = {}
        oos_signals_by_day = {}

        for ds, (bars, bars_5min, vwap_df) in day_data.items():
            fd = datetime.strptime(ds, "%Y-%m-%d").date()
            day_state = precompute_day_state(bars_5min, vwap_df, lookback, STD_BAND)
            sigs = detect_signals(bars, day_state)

            filtered = []
            for s in sigs:
                s["date"] = ds
                ct = s["confirm_time"]
                ct_et = ct.tz_convert("America/New_York") if hasattr(ct, 'tz_convert') else pd.Timestamp(ct, tz='UTC').tz_convert("America/New_York")
                adx_val = get_adx_at_time(ct_et, adx_lookup)
                if not np.isnan(adx_val) and adx_val >= 30.0:
                    filtered.append(s)

            if filtered:
                if fd <= is_end:
                    is_signals_by_day[ds] = filtered
                else:
                    oos_signals_by_day[ds] = filtered

        is_total = sum(len(v) for v in is_signals_by_day.values())
        oos_total = sum(len(v) for v in oos_signals_by_day.values())
        print(f"IS={is_total} sigs, OOS={oos_total} sigs")
        sys.stdout.flush()

        for sl_mult in mults:
            for tp_mult in mults:
                # IS
                is_trades = []
                for ds in sorted(is_signals_by_day.keys()):
                    bars = day_data[ds][0]
                    is_trades.extend(simulate(is_signals_by_day[ds], bars, sl_mult, tp_mult))
                is_stats = calc_stats(is_trades)

                # OOS
                oos_trades = []
                for ds in sorted(oos_signals_by_day.keys()):
                    bars = day_data[ds][0]
                    oos_trades.extend(simulate(oos_signals_by_day[ds], bars, sl_mult, tp_mult))
                oos_stats = calc_stats(oos_trades)

                # Combined (full period)
                all_trades = is_trades + oos_trades
                combined_stats = calc_stats(all_trades)

                if is_stats and oos_stats and combined_stats:
                    all_results.append({
                        "lb": lookback, "sl": sl_mult, "tp": tp_mult,
                        "rr": tp_mult / sl_mult,
                        "is": is_stats, "oos": oos_stats, "combined": combined_stats,
                    })

    # Filter: both IS and OOS must be profitable
    profitable = [r for r in all_results if r["is"]["exp"] > 0 and r["oos"]["exp"] > 0]

    # Sort by smallest combined max DD (least negative = best)
    profitable.sort(key=lambda r: r["combined"]["dd"], reverse=True)

    print(f"\n{'='*120}")
    print(f"  TOP 30 LOW-DD CONFIGS — STD1 (both IS+OOS profitable, sorted by lowest combined DD)")
    print(f"{'='*120}")
    hdr = (f"  {'LB':>3} {'SL':>5} {'TP':>5} {'RR':>4} | "
           f"{'Trades':>6} {'WR':>6} {'PF':>5} {'Total$':>9} {'MaxDD$':>9} | "
           f"{'IS Tr':>5} {'IS WR':>6} {'IS DD$':>9} | "
           f"{'OOS Tr':>6} {'OOS WR':>7} {'OOS DD$':>9}")
    print(hdr)
    print(f"  {'---':>3} {'-----':>5} {'-----':>5} {'----':>4} | "
          f"{'------':>6} {'------':>6} {'-----':>5} {'---------':>9} {'---------':>9} | "
          f"{'-----':>5} {'------':>6} {'---------':>9} | "
          f"{'------':>6} {'-------':>7} {'---------':>9}")

    for r in profitable[:30]:
        c = r["combined"]
        i = r["is"]
        o = r["oos"]
        print(f"  {r['lb']:>3} {r['sl']:>5.2f} {r['tp']:>5.2f} {r['rr']:>4.1f} | "
              f"{c['trades']:>6} {c['wr']:>5.1f}% {c['pf']:>5.2f} ${c['total']*POINT_VALUE:>+8,.0f} ${c['dd']*POINT_VALUE:>+8,.0f} | "
              f"{i['trades']:>5} {i['wr']:>5.1f}% ${i['dd']*POINT_VALUE:>+8,.0f} | "
              f"{o['trades']:>6} {o['wr']:>5.1f}% ${o['dd']*POINT_VALUE:>+8,.0f}")

    # Also show positive-RR only
    pos_rr = [r for r in profitable if r["rr"] >= 1.0]
    pos_rr.sort(key=lambda r: r["combined"]["dd"], reverse=True)

    print(f"\n{'='*120}")
    print(f"  TOP 30 LOW-DD CONFIGS — STD1 (positive RR only, sorted by lowest combined DD)")
    print(f"{'='*120}")
    print(hdr)
    print(f"  {'---':>3} {'-----':>5} {'-----':>5} {'----':>4} | "
          f"{'------':>6} {'------':>6} {'-----':>5} {'---------':>9} {'---------':>9} | "
          f"{'-----':>5} {'------':>6} {'---------':>9} | "
          f"{'------':>6} {'-------':>7} {'---------':>9}")

    for r in pos_rr[:30]:
        c = r["combined"]
        i = r["is"]
        o = r["oos"]
        print(f"  {r['lb']:>3} {r['sl']:>5.2f} {r['tp']:>5.2f} {r['rr']:>4.1f} | "
              f"{c['trades']:>6} {c['wr']:>5.1f}% {c['pf']:>5.2f} ${c['total']*POINT_VALUE:>+8,.0f} ${c['dd']*POINT_VALUE:>+8,.0f} | "
              f"{i['trades']:>5} {i['wr']:>5.1f}% ${i['dd']*POINT_VALUE:>+8,.0f} | "
              f"{o['trades']:>6} {o['wr']:>5.1f}% ${o['dd']*POINT_VALUE:>+8,.0f}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()

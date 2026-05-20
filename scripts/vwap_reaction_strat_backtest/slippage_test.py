"""
VWAP Reaction Strategy — Slippage Sensitivity Test.

For all positive R:R robust combos, deduct round-trip slippage (0–10 ticks)
and show at what level each combo breaks down.

Slippage cost per trade = ticks * 0.25 (entry) + ticks * 0.25 (exit) = ticks * 0.50 pts
"""

import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0
TICK_SIZE = 0.25

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"

FULL_START = "2025-03-13"
FULL_END   = "2026-04-08"


def _load_all_days():
    start = datetime.strptime(FULL_START, "%Y-%m-%d").date()
    end = datetime.strptime(FULL_END, "%Y-%m-%d").date()
    days = []
    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        with open(cache_file, 'rb') as f:
            vwap_data = pickle.load(f)
        signals = vwap_data["signals"]
        if not signals:
            continue
        signal_cache_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_cache_file.exists():
            continue
        with open(signal_cache_file, 'rb') as f:
            signal_data = pickle.load(f)
        days.append((date_str, signals, signal_data["bars"]))
    return days


def _run_collect_pnl(days, sl_mult, tp_mult):
    """Run backtest, return list of per-trade P&L in points."""
    pnl_list = []
    for date_str, signals, bars in days:
        last_exit_time = None
        for signal in signals:
            entry_price = signal["entry_price"]
            direction = signal["direction"]
            atr = signal["atr"]
            confirm_bar_idx = signal["bar_index"] + 1

            if atr is None or atr <= 0:
                continue

            confirm_time = signal["confirm_time"]
            if hasattr(confirm_time, 'tz_convert'):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz='UTC').tz_convert(ET)
            if "16:00" <= confirm_et.strftime("%H:%M") < "19:10":
                continue
            if last_exit_time is not None and confirm_time <= last_exit_time:
                continue

            if direction == "long":
                sl_price = entry_price - atr * sl_mult
                tp_price = entry_price + atr * tp_mult
                if tp_price <= entry_price or sl_price >= entry_price:
                    continue
            else:
                sl_price = entry_price + atr * sl_mult
                tp_price = entry_price - atr * tp_mult
                if tp_price >= entry_price or sl_price <= entry_price:
                    continue

            exit_price = None
            exit_time = None

            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                bar_ct = bar.close_time
                if hasattr(bar_ct, 'tz_convert'):
                    bar_et = bar_ct.tz_convert(ET)
                else:
                    bar_et = pd.Timestamp(bar_ct, tz='UTC').tz_convert(ET)
                if bar_et.strftime("%H:%M") >= FORCE_CLOSE:
                    exit_price = bar.close
                    exit_time = bar.close_time
                    break
                if direction == "short":
                    if bar.high >= sl_price:
                        exit_price = sl_price
                        exit_time = bar.close_time
                        break
                    if bar.low <= tp_price:
                        exit_price = tp_price
                        exit_time = bar.close_time
                        break
                else:
                    if bar.low <= sl_price:
                        exit_price = sl_price
                        exit_time = bar.close_time
                        break
                    if bar.high >= tp_price:
                        exit_price = tp_price
                        exit_time = bar.close_time
                        break

            if exit_price is None:
                last_bar = bars[-1]
                exit_price = last_bar.close
                exit_time = last_bar.close_time

            if direction == "short":
                pnl_list.append(entry_price - exit_price)
            else:
                pnl_list.append(exit_price - entry_price)
            last_exit_time = exit_time

    return pnl_list


def _stats_at_slippage(pnl_list, slip_ticks):
    """Apply round-trip slippage and compute stats."""
    slip_cost = slip_ticks * TICK_SIZE * 2  # entry + exit
    arr = np.array(pnl_list) - slip_cost
    if len(arr) == 0:
        return 0, 0, 0, 0
    winners = arr[arr > 0]
    losers = arr[arr < 0]
    total = len(arr)
    wr = len(winners) / total * 100
    exp = arr.mean()
    gp = winners.sum() if len(winners) > 0 else 0
    gl = abs(losers.sum()) if len(losers) > 0 else 1
    pf = gp / gl if gl > 0 else 999
    total_pnl = arr.sum()
    return total, wr, pf, exp, total_pnl


def main():
    # Positive R:R robust combos from optimization
    sl_vals = np.arange(0.5, 3.25, 0.25)
    tp_vals = np.arange(0.5, 3.25, 0.25)

    print("Loading cache data...")
    days = _load_all_days()
    print(f"  {len(days)} trading days loaded\n")

    # First pass: find all positive R:R combos that are robust (PF > 1 at 0 slippage, min trades)
    print("Finding robust positive R:R combos...")
    combos = []
    for sl in sl_vals:
        for tp in tp_vals:
            if tp <= sl:
                continue
            pnl = _run_collect_pnl(days, sl, tp)
            if len(pnl) < 30:
                continue
            arr = np.array(pnl)
            gp = arr[arr > 0].sum() if len(arr[arr > 0]) > 0 else 0
            gl = abs(arr[arr < 0].sum()) if len(arr[arr < 0]) > 0 else 1
            pf = gp / gl if gl > 0 else 999
            if pf > 1.0 and arr.sum() > 0:
                combos.append((round(sl, 2), round(tp, 2), pnl))

    print(f"  {len(combos)} robust combos found\n")

    # Slippage levels
    slip_ticks = list(range(0, 11))

    # Header
    slip_headers = "".join([f" {t:>4d}t" for t in slip_ticks])
    print(f"{'SL':>5s} {'TP':>5s} {'R:R':>5s} {'Trd':>4s} |  PF at slippage (ticks, round-trip)             | {'BE':>4s}")
    print(f"{'':>5s} {'':>5s} {'':>5s} {'':>4s} |{slip_headers}  | {'tick':>4s}")
    print("-" * 100)

    results = []
    for sl, tp, pnl in combos:
        rr = tp / sl
        n = len(pnl)
        pf_values = []
        be_tick = ">10"
        for st in slip_ticks:
            _, _, pf, exp, _ = _stats_at_slippage(pnl, st)
            pf_values.append(pf)
            if pf < 1.0 and be_tick == ">10":
                be_tick = str(st)

        pf_str = "".join([f" {pf:5.2f}" for pf in pf_values])
        results.append((sl, tp, rr, n, pf_values, be_tick))

    # Sort by breakeven tick (higher = more robust), then by 0-slip PF
    def sort_key(r):
        be = 11 if r[5] == ">10" else int(r[5])
        return (-be, -r[4][0])

    results.sort(key=sort_key)

    for sl, tp, rr, n, pf_values, be_tick in results:
        pf_str = "".join([f" {pf:5.2f}" for pf in pf_values])
        print(f"{sl:5.2f} {tp:5.2f} {rr:5.2f} {n:4d} |{pf_str}  | {be_tick:>4s}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SLIPPAGE RESILIENCE SUMMARY")
    print(f"{'=' * 80}")
    survive_5 = sum(1 for r in results if (r[5] == ">10" or int(r[5]) > 5))
    survive_3 = sum(1 for r in results if (r[5] == ">10" or int(r[5]) > 3))
    print(f"  Combos surviving 3+ ticks slippage:  {survive_3}/{len(results)}")
    print(f"  Combos surviving 5+ ticks slippage:  {survive_5}/{len(results)}")
    print(f"  1 tick slippage = {TICK_SIZE * 2:.2f} pts round-trip (${TICK_SIZE * 2 * POINT_VALUE:.0f})")
    print(f"  5 tick slippage = {5 * TICK_SIZE * 2:.2f} pts round-trip (${5 * TICK_SIZE * 2 * POINT_VALUE:.0f})")
    print(f"  10 tick slippage = {10 * TICK_SIZE * 2:.2f} pts round-trip (${10 * TICK_SIZE * 2 * POINT_VALUE:.0f})")


if __name__ == "__main__":
    main()

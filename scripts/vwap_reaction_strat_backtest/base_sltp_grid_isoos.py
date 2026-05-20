"""
Base strategy SL/TP ATR-multiple grid sweep — IS/OOS 50/50 split.
No ADX filtering. Slippage: entry 0t, SL 2t, TP 0t.

Grid: SL 0.5 to 5.0, TP 0.5 to 5.0, increments of 0.5.

Usage:
    python scripts/vwap_reaction_strat_backtest/base_sltp_grid_isoos.py
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import ET, DATA_DIR, LUNCH_START, LUNCH_END

VWAP_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"

POINT_VALUE = 20.0
TICK_SIZE = 0.25

ENTRY_SLIP_TICKS = 0
SL_SLIP_TICKS = 2
TP_SLIP_TICKS = 0

FORCE_CLOSE = "16:58"

START_DATE = "2025-03-13"
END_DATE = "2026-04-17"


def to_et(ct):
    return ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)


def simulate(bars, sigs, sl_mult, tp_mult):
    sigs = sorted(sigs, key=lambda x: x["confirm_time"])
    trades, last_exit = [], None
    for sig in sigs:
        if last_exit is not None and sig["confirm_time"] <= last_exit:
            continue
        d = sig["direction"]
        ep = sig["entry_price"]
        atr = sig["atr"]
        slip = ENTRY_SLIP_TICKS * TICK_SIZE
        ep_s = ep + slip if d == "long" else ep - slip

        if d == "long":
            sl = ep - atr * sl_mult
            tp = ep + atr * tp_mult
        else:
            sl = ep + atr * sl_mult
            tp = ep - atr * tp_mult

        cbi = sig["bar_index"] + 1
        ex_price = ex_time = ex_reason = None
        for j in range(cbi + 1, len(bars)):
            b = bars[j]
            if not b.closed:
                continue
            bet = to_et(b.close_time)
            if bet.strftime("%H:%M") >= FORCE_CLOSE:
                ex_price, ex_time, ex_reason = b.close, b.close_time, "eod"
                break
            if d == "long":
                if b.low <= sl:
                    ex_price = sl - SL_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "sl"
                    break
                if b.high >= tp:
                    ex_price = tp - TP_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "tp"
                    break
            else:
                if b.high >= sl:
                    ex_price = sl + SL_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "sl"
                    break
                if b.low <= tp:
                    ex_price = tp + TP_SLIP_TICKS * TICK_SIZE
                    ex_time, ex_reason = b.close_time, "tp"
                    break
        if ex_price is None:
            ex_price, ex_time, ex_reason = bars[-1].close, bars[-1].close_time, "eod"
        pnl_pts = (ex_price - ep_s) if d == "long" else (ep_s - ex_price)
        date_str = str(to_et(sig["confirm_time"]).date())
        trades.append({"date": date_str, "pnl": pnl_pts * POINT_VALUE, "reason": ex_reason})
        last_exit = ex_time
    return trades


def stats(trades):
    if not trades:
        return dict(n=0, pnl=0, wr=0, pf=0, dd=0, avg=0)
    df = pd.DataFrame(trades)
    n = len(df)
    pnl = df["pnl"].sum()
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    wr = len(wins) / n * 100
    gp = wins["pnl"].sum() if len(wins) else 0
    gl = abs(losses["pnl"].sum()) if len(losses) else 1
    pf = gp / gl if gl > 0 else float("inf")
    cum = df["pnl"].cumsum()
    dd = float((cum - cum.cummax()).min())
    avg = pnl / n
    return dict(n=n, pnl=pnl, wr=wr, pf=pf, dd=dd, avg=avg)


def main():
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    # Collect dates, 50/50 split
    all_dates = []
    for cache_file in sorted(VWAP_CACHE_DIR.glob("*.pkl")):
        ds = cache_file.stem
        try:
            fd = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd < start_d or fd > end_d:
            continue
        if (SIGNAL_CACHE_DIR / f"{ds}.pkl").exists():
            all_dates.append(fd)

    split = len(all_dates) // 2
    is_dates = set(all_dates[:split])
    is_end = all_dates[split - 1]
    oos_start = all_dates[split]

    print(f"Total days: {len(all_dates)} | IS: {all_dates[0]}->{is_end} ({split}d) | OOS: {oos_start}->{all_dates[-1]} ({len(all_dates)-split}d)")

    # Pre-load all day data
    print("Loading day caches...")
    day_data = {}
    for fd in all_dates:
        ds = fd.strftime("%Y-%m-%d")
        with open(SIGNAL_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            bars = pickle.load(f).get("bars") or []
        if not bars:
            continue
        with open(VWAP_CACHE_DIR / f"{ds}.pkl", "rb") as f:
            base_raw = pickle.load(f).get("signals") or []

        # Pre-filter signals (time filter only, no ADX)
        filtered = []
        for s in base_raw:
            ct_et = to_et(s["confirm_time"])
            hm = ct_et.strftime("%H:%M")
            if "16:00" <= hm < "19:10" or LUNCH_START <= hm < LUNCH_END:
                continue
            atr = s["atr"]
            if atr is None or atr <= 0:
                continue
            filtered.append(s)

        day_data[fd] = (bars, filtered)

    print(f"Loaded {len(day_data)} days\n")

    # Grid sweep
    sl_values = [x / 2.0 for x in range(1, 11)]  # 0.5 to 5.0
    tp_values = [x / 2.0 for x in range(1, 11)]  # 0.5 to 5.0

    print("=" * 140)
    print("  BASE STRATEGY SL/TP GRID (no ADX filter)")
    print("  entry_slip=0t, sl_slip=2t, tp_slip=0t")
    print("=" * 140)

    # Collect all results for summary
    results = []

    print(f"\n  {'SL':>4} {'TP':>4} | {'IS n':>5} {'IS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}"
          f"  |  {'OOS n':>5} {'OOS PnL':>10} {'PF':>5} {'DD':>9} {'WR%':>5} {'Avg':>7}")
    print("  " + "-" * 130)

    for sl_m in sl_values:
        for tp_m in tp_values:
            is_tr, oos_tr = [], []
            for fd, (bars, sigs) in day_data.items():
                day_trades = simulate(bars, sigs, sl_m, tp_m)
                (is_tr if fd in is_dates else oos_tr).extend(day_trades)

            sis = stats(is_tr)
            soos = stats(oos_tr)
            results.append((sl_m, tp_m, sis, soos))
            print(f"  {sl_m:>4.1f} {tp_m:>4.1f} | {sis['n']:>5} ${sis['pnl']:>+9,.0f} {sis['pf']:>5.2f} ${sis['dd']:>+7,.0f} {sis['wr']:>5.1f} ${sis['avg']:>+6,.0f}"
                  f"  |  {soos['n']:>5} ${soos['pnl']:>+9,.0f} {soos['pf']:>5.2f} ${soos['dd']:>+7,.0f} {soos['wr']:>5.1f} ${soos['avg']:>+6,.0f}")

    # Top 10 by OOS PF (min 30 OOS trades)
    print(f"\n{'='*80}")
    print(f"  TOP 10 by OOS Profit Factor (min 30 OOS trades)")
    print(f"{'='*80}")
    viable = [(sl, tp, si, so) for sl, tp, si, so in results if so['n'] >= 30]
    top_pf = sorted(viable, key=lambda x: x[3]['pf'], reverse=True)[:10]
    print(f"  {'SL':>4} {'TP':>4} | {'IS PF':>5} {'IS PnL':>9} {'IS DD':>8} | {'OOS PF':>6} {'OOS PnL':>9} {'OOS DD':>8} {'OOS n':>5} {'WR%':>5}")
    print("  " + "-" * 78)
    for sl, tp, si, so in top_pf:
        print(f"  {sl:>4.1f} {tp:>4.1f} | {si['pf']:>5.2f} ${si['pnl']:>+8,.0f} ${si['dd']:>+7,.0f} | {so['pf']:>6.2f} ${so['pnl']:>+8,.0f} ${so['dd']:>+7,.0f} {so['n']:>5} {so['wr']:>5.1f}")

    # Top 10 by OOS PnL (min 30 OOS trades)
    print(f"\n{'='*80}")
    print(f"  TOP 10 by OOS Total PnL (min 30 OOS trades)")
    print(f"{'='*80}")
    top_pnl = sorted(viable, key=lambda x: x[3]['pnl'], reverse=True)[:10]
    print(f"  {'SL':>4} {'TP':>4} | {'IS PF':>5} {'IS PnL':>9} {'IS DD':>8} | {'OOS PF':>6} {'OOS PnL':>9} {'OOS DD':>8} {'OOS n':>5} {'WR%':>5}")
    print("  " + "-" * 78)
    for sl, tp, si, so in top_pnl:
        print(f"  {sl:>4.1f} {tp:>4.1f} | {si['pf']:>5.2f} ${si['pnl']:>+8,.0f} ${si['dd']:>+7,.0f} | {so['pf']:>6.2f} ${so['pnl']:>+8,.0f} ${so['dd']:>+7,.0f} {so['n']:>5} {so['wr']:>5.1f}")

    # Heatmap: OOS PF
    print(f"\n{'='*80}")
    print(f"  OOS PROFIT FACTOR HEATMAP")
    print(f"{'='*80}")
    print(f"  SL\\TP ", end="")
    for tp_m in tp_values:
        print(f" {tp_m:>5.1f}", end="")
    print()
    print("  " + "-" * (7 + 6 * len(tp_values)))
    for sl_m in sl_values:
        print(f"  {sl_m:>4.1f} |", end="")
        for tp_m in tp_values:
            match = [r for r in results if r[0] == sl_m and r[1] == tp_m]
            if match:
                pf = match[0][3]['pf']
                print(f" {pf:>5.2f}", end="")
            else:
                print(f"   N/A", end="")
        print()


if __name__ == "__main__":
    main()

"""
Base strategy time-of-day analysis — IS/OOS 50/50 split.
No ADX filtering. SL=0.5, TP=1.5 (best OOS PF from grid).
Shows performance by hourly bucket to find when the edge exists.

Usage:
    python scripts/vwap_reaction_strat_backtest/base_time_analysis.py
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

# Test multiple SL/TP combos to see if time patterns are consistent
CONFIGS = [
    (0.5, 0.5, "0.5/0.5"),
    (0.5, 1.0, "0.5/1.0"),
    (0.5, 1.5, "0.5/1.5"),
    (1.0, 1.0, "1.0/1.0"),
    (1.0, 1.5, "1.0/1.5"),
]


def to_et(ct):
    return ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)


def simulate_with_times(bars, sigs, sl_mult, tp_mult):
    """Returns trades with entry hour bucket."""
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
        ct_et = to_et(sig["confirm_time"])
        hour = ct_et.hour
        date_str = str(ct_et.date())
        trades.append({
            "date": date_str,
            "pnl": pnl_pts * POINT_VALUE,
            "hour": hour,
            "direction": d,
            "reason": ex_reason,
        })
        last_exit = ex_time
    return trades


def stats_row(trades):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    n = len(df)
    pnl = df["pnl"].sum()
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    wr = len(wins) / n * 100
    gp = wins["pnl"].sum() if len(wins) else 0
    gl = abs(losses["pnl"].sum()) if len(losses) else 1
    pf = gp / gl if gl > 0 else float("inf")
    avg = pnl / n
    return dict(n=n, pnl=pnl, wr=wr, pf=pf, avg=avg)


def main():
    start_d = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_d = datetime.strptime(END_DATE, "%Y-%m-%d").date()

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

    print(f"Total days: {len(all_dates)} | IS: {all_dates[0]}->{all_dates[split-1]} ({split}d) | OOS: {all_dates[split]}->{all_dates[-1]} ({len(all_dates)-split}d)")

    # Pre-load
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
        # Time filter only (no ADX, no lunch — we want to see lunch performance too)
        filtered = []
        for s in base_raw:
            ct_et = to_et(s["confirm_time"])
            hm = ct_et.strftime("%H:%M")
            if "16:00" <= hm < "19:10":
                continue
            atr = s["atr"]
            if atr is None or atr <= 0:
                continue
            filtered.append(s)
        day_data[fd] = (bars, filtered)

    print(f"Loaded {len(day_data)} days\n")

    # Session hours in ET order: 19,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    session_hours = list(range(19, 24)) + list(range(0, 16))

    # Hour labels for readability
    hour_labels = {
        19: "7pm", 20: "8pm", 21: "9pm", 22: "10pm", 23: "11pm",
        0: "12am", 1: "1am", 2: "2am", 3: "3am", 4: "4am", 5: "5am",
        6: "6am", 7: "7am", 8: "8am", 9: "9am", 10: "10am", 11: "11am",
        12: "12pm", 13: "1pm", 14: "2pm", 15: "3pm",
    }

    for sl_m, tp_m, label in CONFIGS:
        print("=" * 120)
        print(f"  TIME-OF-DAY ANALYSIS: BASE SL={sl_m} TP={tp_m}")
        print("=" * 120)
        print(f"  {'Hour':>6} | {'IS n':>5} {'IS PnL':>9} {'PF':>5} {'WR%':>5} {'Avg':>7}"
              f"  |  {'OOS n':>5} {'OOS PnL':>9} {'PF':>5} {'WR%':>5} {'Avg':>7}"
              f"  |  {'ALL n':>5} {'ALL PnL':>9} {'PF':>5} {'WR%':>5}")
        print("  " + "-" * 115)

        # Gather all trades
        is_all, oos_all = [], []
        for fd, (bars, sigs) in day_data.items():
            day_trades = simulate_with_times(bars, sigs, sl_m, tp_m)
            (is_all if fd in is_dates else oos_all).extend(day_trades)

        for h in session_hours:
            is_h = [t for t in is_all if t["hour"] == h]
            oos_h = [t for t in oos_all if t["hour"] == h]
            all_h = is_h + oos_h

            si = stats_row(is_h) or dict(n=0, pnl=0, wr=0, pf=0, avg=0)
            so = stats_row(oos_h) or dict(n=0, pnl=0, wr=0, pf=0, avg=0)
            sa = stats_row(all_h) or dict(n=0, pnl=0, wr=0, pf=0, avg=0)

            hl = hour_labels.get(h, f"{h}:00")
            print(f"  {hl:>6} | {si['n']:>5} ${si['pnl']:>+8,.0f} {si['pf']:>5.2f} {si['wr']:>5.1f} ${si['avg']:>+6,.0f}"
                  f"  |  {so['n']:>5} ${so['pnl']:>+8,.0f} {so['pf']:>5.2f} {so['wr']:>5.1f} ${so['avg']:>+6,.0f}"
                  f"  |  {sa['n']:>5} ${sa['pnl']:>+8,.0f} {sa['pf']:>5.2f} {sa['wr']:>5.1f}")

        # Totals
        si_t = stats_row(is_all) or dict(n=0, pnl=0, wr=0, pf=0, avg=0)
        so_t = stats_row(oos_all) or dict(n=0, pnl=0, wr=0, pf=0, avg=0)
        sa_t = stats_row(is_all + oos_all) or dict(n=0, pnl=0, wr=0, pf=0, avg=0)
        print("  " + "-" * 115)
        print(f"  {'TOTAL':>6} | {si_t['n']:>5} ${si_t['pnl']:>+8,.0f} {si_t['pf']:>5.2f} {si_t['wr']:>5.1f} ${si_t['avg']:>+6,.0f}"
              f"  |  {so_t['n']:>5} ${so_t['pnl']:>+8,.0f} {so_t['pf']:>5.2f} {so_t['wr']:>5.1f} ${so_t['avg']:>+6,.0f}"
              f"  |  {sa_t['n']:>5} ${sa_t['pnl']:>+8,.0f} {sa_t['pf']:>5.2f} {sa_t['wr']:>5.1f}")
        print()

    # Session bucket analysis (overnight vs RTH vs London vs Asia)
    print("=" * 120)
    print("  SESSION BUCKET ANALYSIS (SL=0.5, TP=1.5)")
    print("=" * 120)

    buckets = {
        "Asia (7pm-2am)":     [19, 20, 21, 22, 23, 0, 1],
        "London (2am-5am)":   [2, 3, 4],
        "Pre-mkt (5am-9am)":  [5, 6, 7, 8],
        "RTH-AM (9am-12pm)":  [9, 10, 11],
        "Lunch (12pm-1pm)":   [12],
        "RTH-PM (1pm-4pm)":   [13, 14, 15],
    }

    sl_m, tp_m = 0.5, 1.5
    is_all, oos_all = [], []
    for fd, (bars, sigs) in day_data.items():
        day_trades = simulate_with_times(bars, sigs, sl_m, tp_m)
        (is_all if fd in is_dates else oos_all).extend(day_trades)

    print(f"  {'Session':>20} | {'IS n':>5} {'IS PnL':>9} {'PF':>5} {'WR%':>5}"
          f"  |  {'OOS n':>5} {'OOS PnL':>9} {'PF':>5} {'WR%':>5}"
          f"  |  {'ALL n':>5} {'ALL PnL':>9} {'PF':>5} {'WR%':>5}")
    print("  " + "-" * 110)

    for bname, hours in buckets.items():
        is_h = [t for t in is_all if t["hour"] in hours]
        oos_h = [t for t in oos_all if t["hour"] in hours]
        all_h = is_h + oos_h
        si = stats_row(is_h) or dict(n=0, pnl=0, wr=0, pf=0, avg=0)
        so = stats_row(oos_h) or dict(n=0, pnl=0, wr=0, pf=0, avg=0)
        sa = stats_row(all_h) or dict(n=0, pnl=0, wr=0, pf=0, avg=0)
        print(f"  {bname:>20} | {si['n']:>5} ${si['pnl']:>+8,.0f} {si['pf']:>5.2f} {si['wr']:>5.1f}"
              f"  |  {so['n']:>5} ${so['pnl']:>+8,.0f} {so['pf']:>5.2f} {so['wr']:>5.1f}"
              f"  |  {sa['n']:>5} ${sa['pnl']:>+8,.0f} {sa['pf']:>5.2f} {sa['wr']:>5.1f}")

    # Long vs Short breakdown by session
    print()
    print("=" * 120)
    print("  DIRECTION x SESSION (SL=0.5, TP=1.5, full dataset)")
    print("=" * 120)
    all_trades = is_all + oos_all
    print(f"  {'Session':>20} {'Dir':>6} | {'n':>5} {'PnL':>9} {'PF':>5} {'WR%':>5} {'Avg':>7}")
    print("  " + "-" * 70)
    for bname, hours in buckets.items():
        for d in ["long", "short"]:
            sub = [t for t in all_trades if t["hour"] in hours and t["direction"] == d]
            s = stats_row(sub) or dict(n=0, pnl=0, wr=0, pf=0, avg=0)
            print(f"  {bname:>20} {d:>6} | {s['n']:>5} ${s['pnl']:>+8,.0f} {s['pf']:>5.2f} {s['wr']:>5.1f} ${s['avg']:>+6,.0f}")


if __name__ == "__main__":
    main()

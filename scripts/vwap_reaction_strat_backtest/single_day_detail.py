"""
Run combined strategy backtest on a single session and print full trade detail
(signal time, entry time, direction, SL/TP, exit time, exit reason, PnL).

Usage:
    python scripts/vwap_reaction_strat_backtest/single_day_detail.py 2026-04-14
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time, passes_adx_filter,
    LUNCH_START, LUNCH_END,
)
from trending_vwap_atr_grid import (
    load_5min_bars as load_5min_bars_trending,
    precompute_day_state, detect_signals as detect_trending_signals,
)

VWAP_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
VWAP_PRICE_CACHE_DIR = DATA_DIR / "vwap_cache"
TIMEBARS_DIR = DATA_DIR / "timebars_5min"

POINT_VALUE = 20.0
ENTRY_CUTOFF = "16:00"
FORCE_CLOSE = "16:58"

BASE_SL = 0.50
BASE_TP = 1.90

TREND_SL = 1.00
TREND_TP = 1.00
TREND_STD = 1
TREND_LB = 14
TREND_ADX_MIN = 25.0


def fmt_et(ts):
    if ts is None:
        return "None"
    if hasattr(ts, "tz_convert"):
        return ts.tz_convert(ET).strftime("%H:%M:%S")
    return pd.Timestamp(ts, tz="UTC").tz_convert(ET).strftime("%H:%M:%S")


def run(date_str: str):
    signal_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
    vwap_file = VWAP_CACHE_DIR / f"{date_str}.pkl"
    vwap_price_file = VWAP_PRICE_CACHE_DIR / f"{date_str}.pkl"
    tb_file = TIMEBARS_DIR / f"timebars_5min_{date_str.replace('-','_')}.pkl"

    for f in (signal_file, vwap_file, vwap_price_file, tb_file):
        if not f.exists():
            print(f"MISSING CACHE: {f}")
            return

    signal_data = pickle.load(open(signal_file, "rb"))
    bars = signal_data["bars"]
    vwap_data = pickle.load(open(vwap_file, "rb"))
    base_signals = vwap_data["signals"]

    vwap_df = pickle.load(open(vwap_price_file, "rb"))
    bars_5min = load_5min_bars_trending(date_str)

    adx_lookup = build_adx_lookup()

    day_state = precompute_day_state(bars_5min, vwap_df, TREND_LB, TREND_STD)
    trend_signals = detect_trending_signals(bars, day_state)

    print(f"\n{'='*100}")
    print(f"  SINGLE-DAY DETAIL: {date_str}")
    print(f"  {len(bars)} range bars | {len(base_signals)} base signals | {len(trend_signals)} trend signals raw")
    print(f"{'='*100}\n")

    all_signals = []

    for s in base_signals:
        ct = s["confirm_time"]
        ct_et = ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)
        hm = ct_et.strftime("%H:%M")
        reject = None
        if "16:00" <= hm < "19:10": reject = f"after {ENTRY_CUTOFF}"
        elif LUNCH_START <= hm < LUNCH_END: reject = "lunch"
        atr = s["atr"]
        if reject is None and (atr is None or atr <= 0): reject = "bad atr"
        adx_val = get_adx_at_time(ct_et, adx_lookup)
        if reject is None and not passes_adx_filter(adx_val): reject = f"adx {adx_val:.1f} out of 15-30"

        ep = s["entry_price"]; d = s["direction"]
        if d == "long":
            sl, tp = ep - atr * BASE_SL, ep + atr * BASE_TP
        else:
            sl, tp = ep + atr * BASE_SL, ep - atr * BASE_TP

        all_signals.append({
            "source": "base", "bar_index": s["bar_index"], "confirm_time": ct, "confirm_time_et": ct_et,
            "entry_price": ep, "direction": d, "sl": sl, "tp": tp, "atr": atr, "adx": adx_val,
            "signal_bar_close_et": bars[s["bar_index"]].close_time, "reject": reject,
        })

    for s in trend_signals:
        ct = s["confirm_time"]
        ct_et = ct.tz_convert(ET) if hasattr(ct, "tz_convert") else pd.Timestamp(ct, tz="UTC").tz_convert(ET)
        hm = ct_et.strftime("%H:%M")
        reject = None
        if "16:00" <= hm < "19:10": reject = f"after {ENTRY_CUTOFF}"
        elif LUNCH_START <= hm < LUNCH_END: reject = "lunch"
        atr = s["atr"]
        if reject is None and (atr is None or atr <= 0): reject = "bad atr"
        adx_val = get_adx_at_time(ct_et, adx_lookup)
        if reject is None and (np.isnan(adx_val) or adx_val < TREND_ADX_MIN): reject = f"adx {adx_val:.1f} < {TREND_ADX_MIN:.0f}"

        ep = s["entry_price"]; d = s["direction"]
        if d == "long":
            sl, tp = ep - atr * TREND_SL, ep + atr * TREND_TP
        else:
            sl, tp = ep + atr * TREND_SL, ep - atr * TREND_TP

        all_signals.append({
            "source": "trend", "bar_index": s["bar_index"], "confirm_time": ct, "confirm_time_et": ct_et,
            "entry_price": ep, "direction": d, "sl": sl, "tp": tp, "atr": atr, "adx": adx_val,
            "signal_bar_close_et": bars[s["bar_index"]].close_time, "reject": reject,
        })

    all_signals.sort(key=lambda x: x["confirm_time"])

    # Log every signal (taken or rejected)
    print(f"--- ALL SIGNALS (chronological) ---")
    for i, s in enumerate(all_signals, 1):
        tag = "REJECTED: " + s["reject"] if s["reject"] else "CANDIDATE"
        print(f"[{i:2d}] {fmt_et(s['confirm_time_et'])} {s['source']:>5} {s['direction']:>5} "
              f"@ {s['entry_price']:.2f} ATR={s['atr']:.2f} ADX={s['adx']:.1f} | {tag}")
    print()

    # Simulate (one at a time, skip rejected)
    trades = []
    last_exit_time = None
    trade_no = 0

    for sig in all_signals:
        if sig["reject"] is not None:
            continue
        if last_exit_time is not None and sig["confirm_time"] <= last_exit_time:
            print(f"  [skipped] {fmt_et(sig['confirm_time_et'])} {sig['source']} {sig['direction']} "
                  f"@ {sig['entry_price']:.2f} — position still open")
            continue

        trade_no += 1
        confirm_bar_idx = sig["bar_index"] + 1
        d = sig["direction"]
        ep = sig["entry_price"]; sl = sig["sl"]; tp = sig["tp"]

        exit_price = exit_time = exit_reason = None

        for j in range(confirm_bar_idx + 1, len(bars)):
            bar = bars[j]
            if not bar.closed:
                continue
            bar_ct = bar.close_time
            bar_et = bar_ct.tz_convert(ET) if hasattr(bar_ct, "tz_convert") else pd.Timestamp(bar_ct, tz="UTC").tz_convert(ET)
            if bar_et.strftime("%H:%M") >= FORCE_CLOSE:
                exit_price, exit_time, exit_reason = bar.close, bar.close_time, "session_close"; break
            if d == "long":
                if bar.low <= sl:
                    exit_price, exit_time, exit_reason = sl, bar.close_time, "stop_loss"; break
                if bar.high >= tp:
                    exit_price, exit_time, exit_reason = tp, bar.close_time, "target"; break
            else:
                if bar.high >= sl:
                    exit_price, exit_time, exit_reason = sl, bar.close_time, "stop_loss"; break
                if bar.low <= tp:
                    exit_price, exit_time, exit_reason = tp, bar.close_time, "target"; break

        if exit_price is None:
            exit_price, exit_time, exit_reason = bars[-1].close, bars[-1].close_time, "eod"

        pnl_pts = (exit_price - ep) if d == "long" else (ep - exit_price)
        pnl_dollars = pnl_pts * POINT_VALUE
        last_exit_time = exit_time

        trades.append({
            "no": trade_no, "source": sig["source"], "direction": d,
            "signal_bar_close": sig["signal_bar_close_et"], "confirm_time": sig["confirm_time_et"],
            "entry_price": ep, "sl": sl, "tp": tp, "atr": sig["atr"], "adx": sig["adx"],
            "exit_time": exit_time, "exit_price": exit_price, "exit_reason": exit_reason,
            "pnl_pts": pnl_pts, "pnl": pnl_dollars,
        })

    print(f"--- TAKEN TRADES ({len(trades)}) ---\n")
    for t in trades:
        print(f"Trade #{t['no']} [{t['source']}] {t['direction'].upper()}")
        print(f"  Signal bar close: {fmt_et(t['signal_bar_close'])}")
        print(f"  Entry (confirm) : {fmt_et(t['confirm_time'])}  @ {t['entry_price']:.2f}")
        print(f"  SL / TP         : {t['sl']:.2f}  /  {t['tp']:.2f}   (ATR {t['atr']:.2f}, ADX {t['adx']:.1f})")
        print(f"  Exit            : {fmt_et(t['exit_time'])}  @ {t['exit_price']:.2f}   [{t['exit_reason']}]")
        print(f"  PnL             : {t['pnl_pts']:+.2f} pts = ${t['pnl']:+.2f}")
        print()

    if trades:
        total = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        print(f"--- SUMMARY ---")
        print(f"  Trades: {len(trades)} | Winners: {wins} | Losers: {len(trades)-wins}")
        print(f"  Total PnL: ${total:+.2f}")
    else:
        print("No trades taken.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python single_day_detail.py YYYY-MM-DD")
        sys.exit(1)
    run(sys.argv[1])

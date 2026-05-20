"""
Session breakdown comparison for two configs with ADX 15-30 filter.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from adx_filter_common import (
    ET, DATA_DIR, build_adx_lookup, get_adx_at_time, passes_adx_filter,
)

VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0
ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"


def get_session(hour, minute):
    t = hour * 60 + minute
    if t >= 19 * 60 or t < 2 * 60:
        return "Asia"
    elif t < 9 * 60 + 30:
        return "London"
    elif t < 16 * 60:
        return "New York"
    else:
        return "Evening"


def run_backtest(days, sl_mult, tp_mult):
    trades = []
    for date_str, signals, bars in days:
        last_exit_time = None
        for signal in signals:
            if not signal["_adx_pass"]:
                continue
            ep = signal["entry_price"]; d = signal["direction"]; atr = signal["atr"]
            cbi = signal["bar_index"] + 1
            if atr is None or atr <= 0:
                continue
            cet = signal["_confirm_et"]
            ct = signal["confirm_time"]
            if "16:00" <= cet.strftime("%H:%M") < "19:10":
                continue
            if last_exit_time is not None and ct <= last_exit_time:
                continue

            sl = ep - atr * sl_mult if d == "long" else ep + atr * sl_mult
            tp = ep + atr * tp_mult if d == "long" else ep - atr * tp_mult
            if d == "long" and (tp <= ep or sl >= ep): continue
            if d == "short" and (tp >= ep or sl <= ep): continue

            exit_price = None; exit_time = None
            for j in range(cbi + 1, len(bars)):
                bar = bars[j]
                if not bar.closed: continue
                bct = bar.close_time
                if hasattr(bct, 'tz_convert'): bet = bct.tz_convert(ET)
                else: bet = pd.Timestamp(bct, tz='UTC').tz_convert(ET)
                if bet.strftime("%H:%M") >= FORCE_CLOSE:
                    exit_price = bar.close; exit_time = bct; break
                if d == "short":
                    if bar.high >= sl: exit_price = sl; exit_time = bct; break
                    if bar.low <= tp: exit_price = tp; exit_time = bct; break
                else:
                    if bar.low <= sl: exit_price = sl; exit_time = bct; break
                    if bar.high >= tp: exit_price = tp; exit_time = bct; break
            if exit_price is None:
                exit_price = bars[-1].close; exit_time = bars[-1].close_time

            pnl = (ep - exit_price) if d == "short" else (exit_price - ep)
            last_exit_time = exit_time

            trades.append({
                "session": get_session(cet.hour, cet.minute),
                "pnl": pnl,
            })
    return trades


def session_stats(trades, label):
    if not trades:
        print(f"  {label:>12s}:    0 trades")
        return
    pnl = np.array([t["pnl"] for t in trades])
    n = len(pnl)
    w = pnl[pnl > 0]
    l = pnl[pnl < 0]
    wr = len(w) / n * 100
    gp = w.sum() if len(w) else 0
    gl = abs(l.sum()) if len(l) else 1
    pf = gp / gl if gl > 0 else 999
    exp = pnl.mean()
    total = pnl.sum()

    # Max DD
    cum = np.cumsum(pnl) * POINT_VALUE
    hwm = np.maximum.accumulate(cum)
    dd = hwm - cum
    max_dd = dd.max()

    print(f"  {label:>12s}: {n:4d} trades | WR {wr:5.1f}% | PF {pf:5.2f} | "
          f"Exp ${exp * POINT_VALUE:+7.0f} | Total ${total * POINT_VALUE:+10,.0f} | MaxDD ${max_dd:8,.0f}")


def load_days(adx_lookup):
    start = datetime.strptime("2025-03-13", "%Y-%m-%d").date()
    end = datetime.strptime("2026-04-08", "%Y-%m-%d").date()
    days = []
    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        with open(cache_file, 'rb') as f:
            vd = pickle.load(f)
        signals = vd["signals"]
        if not signals: continue
        scf = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not scf.exists(): continue
        with open(scf, 'rb') as f:
            sd = pickle.load(f)
        tagged = []
        for signal in signals:
            ct = signal["confirm_time"]
            if hasattr(ct, 'tz_convert'): cet = ct.tz_convert(ET)
            else: cet = pd.Timestamp(ct, tz='UTC').tz_convert(ET)
            adx_val = get_adx_at_time(cet, adx_lookup)
            signal["_adx_pass"] = passes_adx_filter(adx_val)
            signal["_confirm_et"] = cet
            tagged.append(signal)
        days.append((date_str, tagged, sd["bars"]))
    return days


if __name__ == "__main__":
    print("Building ADX lookup...")
    adx_lookup = build_adx_lookup()
    days = load_days(adx_lookup)
    print(f"  {len(days)} days\n")

    configs = [
        (0.50, 1.90, "SL 0.50 / TP 1.90"),
        (0.50, 2.10, "SL 0.50 / TP 2.10"),
    ]

    sessions = ["Asia", "London", "New York", "Evening"]

    for sl, tp, name in configs:
        trades = run_backtest(days, sl, tp)
        print(f"{'='*100}")
        print(f"  {name} (ADX 15-30) — {len(trades)} total trades")
        print(f"{'='*100}")

        session_stats(trades, "ALL")
        print()
        for sess in sessions:
            st = [t for t in trades if t["session"] == sess]
            session_stats(st, sess)
        print()

"""
VWAP Reaction Strategy — Session & Hourly Performance Breakdown.

Sessions (ET):
  Asia:    7:00 PM - 2:00 AM
  London:  2:00 AM - 9:30 AM
  New York: 9:30 AM - 4:00 PM
  Evening: 4:00 PM - 5:00 PM (force close zone)

Also breaks down by hour within each session.
"""

import pickle
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0

SL_MULT = 1.0
TP_MULT = 2.0
ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"

FULL_START = "2025-03-13"
FULL_END   = "2026-04-08"


def get_session(hour, minute):
    """Return session name based on ET hour."""
    t = hour * 60 + minute
    if t >= 19 * 60 or t < 2 * 60:      # 7pm - 2am
        return "Asia"
    elif t < 9 * 60 + 30:                # 2am - 9:30am
        return "London"
    elif t < 16 * 60:                     # 9:30am - 4pm
        return "New York"
    else:                                  # 4pm - 5pm
        return "Evening"


def load_and_run():
    start = datetime.strptime(FULL_START, "%Y-%m-%d").date()
    end = datetime.strptime(FULL_END, "%Y-%m-%d").date()

    trades = []

    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        with open(cache_file, 'rb') as f:
            vd = pickle.load(f)
        signals = vd["signals"]
        if not signals:
            continue
        scf = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not scf.exists():
            continue
        with open(scf, 'rb') as f:
            sd = pickle.load(f)
        bars = sd["bars"]

        last_exit_time = None

        for signal in signals:
            ep = signal["entry_price"]
            d = signal["direction"]
            atr = signal["atr"]
            cbi = signal["bar_index"] + 1

            if atr is None or atr <= 0:
                continue

            ct = signal["confirm_time"]
            if hasattr(ct, 'tz_convert'):
                cet = ct.tz_convert(ET)
            else:
                cet = pd.Timestamp(ct, tz='UTC').tz_convert(ET)
            if "16:00" <= cet.strftime("%H:%M") < "19:10":
                continue
            if last_exit_time is not None and ct <= last_exit_time:
                continue

            sl = ep - atr * SL_MULT if d == "long" else ep + atr * SL_MULT
            tp = ep + atr * TP_MULT if d == "long" else ep - atr * TP_MULT
            if d == "long" and (tp <= ep or sl >= ep): continue
            if d == "short" and (tp >= ep or sl <= ep): continue

            exit_price = None
            exit_time = None
            exit_reason = None

            for j in range(cbi + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                bct = bar.close_time
                if hasattr(bct, 'tz_convert'):
                    bet = bct.tz_convert(ET)
                else:
                    bet = pd.Timestamp(bct, tz='UTC').tz_convert(ET)
                if bet.strftime("%H:%M") >= FORCE_CLOSE:
                    exit_price = bar.close
                    exit_time = bar.close_time
                    exit_reason = "session_close"
                    break
                if d == "short":
                    if bar.high >= sl:
                        exit_price = sl; exit_time = bct; exit_reason = "stop"; break
                    if bar.low <= tp:
                        exit_price = tp; exit_time = bct; exit_reason = "target"; break
                else:
                    if bar.low <= sl:
                        exit_price = sl; exit_time = bct; exit_reason = "stop"; break
                    if bar.high >= tp:
                        exit_price = tp; exit_time = bct; exit_reason = "target"; break

            if exit_price is None:
                exit_price = bars[-1].close
                exit_time = bars[-1].close_time
                exit_reason = "eod"

            pnl = (ep - exit_price) if d == "short" else (exit_price - ep)
            last_exit_time = exit_time

            entry_hour = cet.hour
            entry_min = cet.minute
            session = get_session(entry_hour, entry_min)

            trades.append({
                "date": date_str,
                "entry_time_et": cet,
                "hour": entry_hour,
                "minute": entry_min,
                "session": session,
                "direction": d,
                "pnl_pts": pnl,
                "pnl_dollars": pnl * POINT_VALUE,
                "exit_reason": exit_reason,
            })

    return trades


def print_stats(label, pnl_arr):
    if len(pnl_arr) == 0:
        print(f"  {label:>20s}:   0 trades")
        return
    n = len(pnl_arr)
    w = pnl_arr[pnl_arr > 0]
    l = pnl_arr[pnl_arr < 0]
    wr = len(w) / n * 100
    gp = w.sum() if len(w) else 0
    gl = abs(l.sum()) if len(l) else 1
    pf = gp / gl if gl > 0 else 999
    exp = pnl_arr.mean()
    total = pnl_arr.sum()
    print(f"  {label:>20s}: {n:4d} trades | WR {wr:5.1f}% | PF {pf:5.2f} | "
          f"Exp {exp:+6.2f} pts | Total {total:+8.1f} pts (${total * POINT_VALUE:+10,.0f})")


def main():
    print("Loading data and running backtest...", flush=True)
    trades = load_and_run()
    print(f"  {len(trades)} total trades\n", flush=True)

    df = pd.DataFrame(trades)
    all_pnl = np.array(df["pnl_pts"])

    # Overall
    print("=" * 110)
    print("OVERALL")
    print("=" * 110)
    print_stats("All Trades", all_pnl)

    # By session
    print(f"\n{'=' * 110}")
    print("BY SESSION")
    print("=" * 110)
    session_order = ["Asia", "London", "New York", "Evening"]
    for sess in session_order:
        mask = df["session"] == sess
        if mask.sum() > 0:
            print_stats(sess, np.array(df.loc[mask, "pnl_pts"]))

    # By direction within session
    print(f"\n{'=' * 110}")
    print("BY SESSION + DIRECTION")
    print("=" * 110)
    for sess in session_order:
        for dirn in ["long", "short"]:
            mask = (df["session"] == sess) & (df["direction"] == dirn)
            if mask.sum() > 0:
                print_stats(f"{sess} {dirn}", np.array(df.loc[mask, "pnl_pts"]))

    # By hour
    print(f"\n{'=' * 110}")
    print("BY HOUR (ET)")
    print("=" * 110)

    # Build hour labels in session order
    hour_order = list(range(19, 24)) + list(range(0, 17))
    for h in hour_order:
        mask = df["hour"] == h
        if mask.sum() > 0:
            label = f"{h:02d}:00-{h:02d}:59"
            sess = get_session(h, 0)
            print_stats(f"{label} ({sess[:3]})", np.array(df.loc[mask, "pnl_pts"]))

    # By 2-hour blocks during NY session
    print(f"\n{'=' * 110}")
    print("NEW YORK SESSION — 2-HOUR BLOCKS")
    print("=" * 110)
    ny_blocks = [
        ("9:30-11:30 (Open)", 9, 30, 11, 30),
        ("11:30-13:30 (Midday)", 11, 30, 13, 30),
        ("13:30-16:00 (Close)", 13, 30, 16, 0),
    ]
    for label, sh, sm, eh, em in ny_blocks:
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        mask = df.apply(lambda r: start_min <= r["hour"] * 60 + r["minute"] < end_min
                        and r["session"] == "New York", axis=1)
        if mask.sum() > 0:
            print_stats(label, np.array(df.loc[mask, "pnl_pts"]))

    # Evening specifically
    print(f"\n{'=' * 110}")
    print("EVENING (4:00 PM - 4:50 PM ET) — Should you avoid?")
    print("=" * 110)
    mask = df["session"] == "Evening"
    if mask.sum() > 0:
        print_stats("Evening", np.array(df.loc[mask, "pnl_pts"]))
        # Show what overall stats look like without evening
        mask_no_eve = df["session"] != "Evening"
        print_stats("Without Evening", np.array(df.loc[mask_no_eve, "pnl_pts"]))
    else:
        print("  No evening trades (entry cutoff prevents them)")

    # Asia overnight — is it choppy?
    print(f"\n{'=' * 110}")
    print("ASIA (7PM-2AM) — Is overnight choppy?")
    print("=" * 110)
    mask = df["session"] == "Asia"
    if mask.sum() > 0:
        print_stats("Asia", np.array(df.loc[mask, "pnl_pts"]))
        mask_no_asia = df["session"] != "Asia"
        print_stats("Without Asia", np.array(df.loc[mask_no_asia, "pnl_pts"]))

    # Win rate by hour heatmap-style
    print(f"\n{'=' * 110}")
    print("HOURLY WIN RATE & TRADE COUNT HEATMAP")
    print("=" * 110)
    print(f"  {'Hour':>12s} {'Sess':>4s} {'#':>4s} {'WR%':>6s} {'PF':>6s} {'Exp':>8s} {'Total$':>10s}  Bar")
    print(f"  {'-'*12} {'-'*4} {'-'*4} {'-'*6} {'-'*6} {'-'*8} {'-'*10}  {'-'*30}")

    for h in hour_order:
        mask = df["hour"] == h
        if mask.sum() == 0:
            continue
        arr = np.array(df.loc[mask, "pnl_pts"])
        n = len(arr)
        w = arr[arr > 0]
        l = arr[arr < 0]
        wr = len(w) / n * 100
        gp = w.sum() if len(w) else 0
        gl = abs(l.sum()) if len(l) else 1
        pf = gp / gl if gl > 0 else 999
        exp = arr.mean()
        total_d = arr.sum() * POINT_VALUE
        sess = get_session(h, 0)[:3]

        # Visual bar
        bar_len = int(abs(total_d) / 500)
        if total_d > 0:
            bar = "+" * min(bar_len, 30)
        else:
            bar = "-" * min(bar_len, 30)

        print(f"  {h:02d}:00-{h:02d}:59 {sess:>4s} {n:4d} {wr:5.1f}% {pf:5.2f} {exp:+7.2f} ${total_d:+9,.0f}  {bar}")


if __name__ == "__main__":
    main()

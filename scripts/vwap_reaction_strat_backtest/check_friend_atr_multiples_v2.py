"""
Reverse-engineer friend's SL/TP ATR multiples from full trade log (~170 trades).
Match entry price to 15-min bar close, get ATR(14), compute exit multiple.
"""
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
ATR_LEN = 14
LOG_FILE = Path("data/friend_trade_log_full.txt")


def load_5min(ds):
    f = TIMEBARS_DIR / f"timebars_5min_{ds.replace('-','_')}.pkl"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        bars = pickle.load(fh)
    if not bars:
        return None
    rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
             "low": b["low"], "close": b["close"]} for b in bars]
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df.index = df.index + pd.Timedelta(minutes=5)
    return df


def agg_15min(df5):
    df = df5.copy()
    df["group"] = df.index.floor("15min")
    agg = df.groupby("group").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"))
    agg.index = agg.index + pd.Timedelta(minutes=15)
    return agg


def parse_trades(path):
    trades = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        # Split on 4+ spaces (fields separated by multiple spaces)
        parts = re.split(r"    +", line)
        if len(parts) < 9:
            continue
        symbol = parts[0]
        action = parts[1]
        qty = int(parts[2])
        entry = float(parts[3].replace("$", "").replace(",", ""))
        exit_p = float(parts[4].replace("$", "").replace(",", ""))
        pnl_str = parts[5].replace("$", "").replace(",", "").replace("+", "")
        pnl = float(pnl_str)
        duration = parts[6]
        entry_time = parts[7]
        exit_time = parts[8]
        trades.append((action, qty, entry, exit_p, pnl, duration, entry_time, exit_time))
    return trades


def main():
    # Build continuous 15-min bars
    print("Building continuous 15-min bars from 5-min data...")
    all_files = sorted(TIMEBARS_DIR.glob("timebars_5min_202*.pkl"))
    frames = []
    for f in all_files:
        ds = f.stem.replace("timebars_5min_", "").replace("_", "-")
        df5 = load_5min(ds)
        if df5 is not None:
            df15 = agg_15min(df5)
            frames.append(df15)

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    print(f"Total 15-min bars: {len(df)}, range: {df.index[0]} to {df.index[-1]}")

    # Compute ATR(14)
    closes = df["close"]
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - closes.shift()).abs(),
        (df["low"] - closes.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()

    # Normalize index to ET
    df.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df.index
    ])

    # Parse trades
    trades = parse_trades(LOG_FILE)
    print(f"Loaded {len(trades)} trades from log")

    # For each trade, find the 15-min bar whose close matches entry price
    print()
    print("=" * 130)
    print("  ATR MULTIPLE ANALYSIS -- Match entry to 15-min bar close, compute exit distance / ATR")
    print("=" * 130)

    hdr = (f"{'#':>3} {'Side':<5} {'Q':>1} {'Entry':>10} {'Exit':>10} {'PnL':>8} "
           f"{'BarTime':<20} {'BarCl':>10} {'ATR14':>8} {'Dist':>9} {'Mult':>7} {'Type':<4}")
    print(hdr)
    print("-" * len(hdr))

    tp_mults = []
    sl_mults = []
    no_match = 0
    no_atr = 0
    session_exit = 0  # trades that look like session close exits

    for idx, (side, qty, entry, exit_p, pnl, duration, entry_time_str, exit_time_str) in enumerate(trades, 1):
        # Parse entry time -- format like "Apr 17, 1:30 PM" (year is 2025 or 2026)
        # Determine year from price range and date
        month_str = entry_time_str.split(",")[0].split(" ")[0]
        day_str = entry_time_str.split(",")[0].split(" ")[1] if len(entry_time_str.split(",")[0].split(" ")) > 1 else ""

        # Oct-Dec entries are 2025, Jan-Apr entries are 2026
        if month_str in ("Oct", "Nov", "Dec"):
            year = 2025
        else:
            year = 2026

        try:
            dt = pd.Timestamp(f"{entry_time_str}, {year}", tz=ET)
        except Exception:
            print(f"{idx:>3} {side:<5} {qty:>1} {entry:>10.2f} {exit_p:>10.2f} {pnl:>+8.0f} PARSE ERROR: {entry_time_str}")
            continue

        # Find the 15-min bar whose close matches entry price exactly
        # Look within a 2-hour window around the entry time
        window_start = dt - pd.Timedelta(hours=1)
        window_end = dt + pd.Timedelta(minutes=15)
        mask = (df.index >= window_start) & (df.index <= window_end)
        candidates = df[mask]

        if candidates.empty:
            print(f"{idx:>3} {side:<5} {qty:>1} {entry:>10.2f} {exit_p:>10.2f} {pnl:>+8.0f} NO BARS IN WINDOW ({dt})")
            no_match += 1
            continue

        # Find bar with closest close to entry price
        close_diffs = (candidates["close"] - entry).abs()
        best_idx = close_diffs.idxmin()
        best_diff = close_diffs[best_idx]

        if best_diff > 2.0:  # more than 2 pts off = no match
            # Try wider window
            window_start2 = dt - pd.Timedelta(hours=2)
            mask2 = (df.index >= window_start2) & (df.index <= window_end)
            candidates2 = df[mask2]
            if not candidates2.empty:
                close_diffs2 = (candidates2["close"] - entry).abs()
                best_idx2 = close_diffs2.idxmin()
                best_diff2 = close_diffs2[best_idx2]
                if best_diff2 <= 2.0:
                    best_idx = best_idx2
                    best_diff = best_diff2
                    candidates = candidates2

        row = df.loc[best_idx]
        bc = row["close"]
        atr = row["atr"]

        if pd.isna(atr) or atr <= 0:
            print(f"{idx:>3} {side:<5} {qty:>1} {entry:>10.2f} {exit_p:>10.2f} {pnl:>+8.0f} {str(best_idx)[:19]:<20} {bc:>10.2f} {'NaN':>8} -- no ATR --")
            no_atr += 1
            continue

        # Exit distance from entry price (positive = in favor)
        if side == "BUY":
            dist = exit_p - entry
        else:
            dist = entry - exit_p

        mult = dist / atr
        trade_type = "TP" if pnl > 0 else "SL" if pnl < 0 else "BE"

        match_flag = "" if best_diff < 0.5 else f" d={best_diff:.1f}"

        # Check if exit looks like session close (exit at :00 PM times, 3:45-4:45 PM)
        try:
            exit_dt = pd.Timestamp(f"{exit_time_str}, {year}", tz=ET)
            exit_hour = exit_dt.hour
            exit_min = exit_dt.minute
            is_session_exit = (exit_hour >= 15 and exit_min >= 30) or exit_hour >= 16
        except Exception:
            is_session_exit = False

        sess_flag = " [SESSION]" if is_session_exit else ""

        if pnl > 0:
            tp_mults.append(mult)
        elif pnl < 0:
            sl_mults.append(mult)

        if is_session_exit:
            session_exit += 1

        print(f"{idx:>3} {side:<5} {qty:>1} {entry:>10.2f} {exit_p:>10.2f} {pnl:>+8.0f} "
              f"{str(best_idx)[:19]:<20} {bc:>10.2f} {atr:>8.2f} {dist:>+9.2f} {mult:>+7.2f} {trade_type:<4}{match_flag}{sess_flag}")

    # Summary
    print()
    print("=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"  Total trades: {len(trades)}")
    print(f"  No bar match: {no_match}")
    print(f"  No ATR: {no_atr}")
    print(f"  Session-close exits: {session_exit}")

    if tp_mults:
        tp_arr = np.array(tp_mults)
        print(f"\n  WINNERS (TP candidates, n={len(tp_arr)}):")
        print(f"    Mean:   {tp_arr.mean():+.3f} x ATR")
        print(f"    Median: {np.median(tp_arr):+.3f} x ATR")
        print(f"    Std:    {tp_arr.std():.3f}")
        print(f"    Min:    {tp_arr.min():+.3f}")
        print(f"    Max:    {tp_arr.max():+.3f}")
        print(f"    P25:    {np.percentile(tp_arr, 25):+.3f}")
        print(f"    P75:    {np.percentile(tp_arr, 75):+.3f}")

        print(f"\n    Distribution:")
        for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0),
                       (1.0, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 2.5),
                       (2.5, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 20.0)]:
            cnt = ((tp_arr >= lo) & (tp_arr < hi)).sum()
            if cnt > 0:
                bar = "#" * cnt
                print(f"      [{lo:>5.2f}, {hi:>5.2f}): {cnt:>3}  {bar}")
        # Also show negative TP multiples (weird cases)
        neg_tp = (tp_arr < 0).sum()
        if neg_tp > 0:
            print(f"      [< 0         ): {neg_tp:>3}  (exit worse than entry but still profit via bar-close diff)")

    if sl_mults:
        sl_arr = np.array(sl_mults)
        print(f"\n  LOSERS (SL candidates, n={len(sl_arr)}):")
        print(f"    Mean:   {sl_arr.mean():+.3f} x ATR")
        print(f"    Median: {np.median(sl_arr):+.3f} x ATR")
        print(f"    Std:    {sl_arr.std():.3f}")
        print(f"    Min:    {sl_arr.min():+.3f}")
        print(f"    Max:    {sl_arr.max():+.3f}")
        print(f"    P25:    {np.percentile(sl_arr, 25):+.3f}")
        print(f"    P75:    {np.percentile(sl_arr, 75):+.3f}")

        print(f"\n    Distribution (negative = against):")
        for lo, hi in [(-20, -6), (-6, -4), (-4, -3), (-3, -2.5), (-2.5, -2.0),
                       (-2.0, -1.75), (-1.75, -1.5), (-1.5, -1.25), (-1.25, -1.0),
                       (-1.0, -0.75), (-0.75, -0.5), (-0.5, -0.25), (-0.25, 0)]:
            cnt = ((sl_arr >= lo) & (sl_arr < hi)).sum()
            if cnt > 0:
                bar = "#" * cnt
                print(f"      [{lo:>6.2f}, {hi:>6.2f}): {cnt:>3}  {bar}")
        # positive SL multiples (exit in favor but still lost money somehow)
        pos_sl = (sl_arr > 0).sum()
        if pos_sl > 0:
            print(f"      [> 0           ): {pos_sl:>3}  (exit in favor direction but lost money -- check these)")

    # Cluster analysis -- look for round ATR multiples
    if sl_mults:
        sl_abs = np.abs(np.array(sl_mults))
        print(f"\n  SL CLUSTER ANALYSIS (absolute multiples):")
        print(f"    Looking for the most common ATR multiple for stops...")
        # Round to nearest 0.25
        rounded = np.round(sl_abs * 4) / 4
        from collections import Counter
        counts = Counter(rounded)
        print(f"    Rounded to 0.25:")
        for val, cnt in sorted(counts.items()):
            bar = "#" * cnt
            print(f"      {val:>5.2f}x: {cnt:>3}  {bar}")

    if tp_mults:
        tp_pos = np.array([m for m in tp_mults if m > 0])
        if len(tp_pos) > 0:
            print(f"\n  TP CLUSTER ANALYSIS (positive multiples only, n={len(tp_pos)}):")
            rounded = np.round(tp_pos * 4) / 4
            from collections import Counter
            counts = Counter(rounded)
            print(f"    Rounded to 0.25:")
            for val, cnt in sorted(counts.items()):
                bar = "#" * cnt
                print(f"      {val:>5.2f}x: {cnt:>3}  {bar}")


if __name__ == "__main__":
    main()

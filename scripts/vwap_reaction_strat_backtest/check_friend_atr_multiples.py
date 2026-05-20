"""
Reverse-engineer friend's SL/TP ATR multiples from trade log.
Build 15-min bars from 5-min data, compute ATR(14) at each entry,
then calculate exit distance / ATR to find implied multiples.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
ATR_LEN = 14


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


def main():
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

    # Friend's trades
    user_trades = [
        ("BUY", 2, 26771.75, 26827.50, 2230, "Apr 17, 2026 1:30 PM"),
        ("BUY", 1, 26827.25, 26736.75, -1810, "Apr 17, 2026 11:00 AM"),
        ("BUY", 1, 26692.25, 26802.25, 2200, "Apr 17, 2026 9:30 AM"),
        ("BUY", 2, 26608.25, 26780.00, 6870, "Apr 17, 2026 8:45 AM"),
        ("BUY", 1, 26517.75, 26441.75, -1520, "Apr 16, 2026 12:31 PM"),
        ("BUY", 1, 26196.75, 26241.50, 895, "Apr 15, 2026 12:00 PM"),
        ("SELL", 1, 25989.00, 26196.75, -4155, "Apr 15, 2026 9:45 AM"),
        ("BUY", 2, 25990.00, 26057.50, 2700, "Apr 15, 2026 9:00 AM"),
        ("SELL", 2, 25966.00, 25990.00, -960, "Apr 15, 2026 8:15 AM"),
        ("SELL", 1, 25980.50, 26038.00, -1150, "Apr 15, 2026 7:30 AM"),
        ("BUY", 1, 25972.50, 26000.75, 565, "Apr 14, 2026 3:45 PM"),
        ("BUY", 1, 25926.00, 25957.50, 630, "Apr 14, 2026 12:45 PM"),
        ("BUY", 1, 25679.00, 25692.50, 270, "Apr 14, 2026 8:45 AM"),
        ("BUY", 1, 25380.50, 25510.50, 2600, "Apr 13, 2026 12:45 PM"),
        ("BUY", 2, 25182.00, 25369.00, 7480, "Apr 13, 2026 9:45 AM"),
        ("BUY", 1, 25320.50, 25284.25, -725, "Apr 10, 2026 10:00 AM"),
        ("BUY", 1, 25312.50, 25354.25, 835, "Apr 10, 2026 8:45 AM"),
        ("BUY", 1, 25181.50, 25249.00, 1350, "Apr 9, 2026 12:45 PM"),
        ("BUY", 2, 25033.25, 25035.00, 70, "Apr 9, 2026 10:00 AM"),
        ("BUY", 1, 24387.00, 24194.25, -3855, "Apr 7, 2026 6:00 AM"),
        ("BUY", 2, 24315.50, 24309.50, -240, "Apr 6, 2026 7:01 AM"),
        ("BUY", 1, 24338.00, 24271.25, -1335, "Apr 6, 2026 6:01 AM"),
        ("BUY", 1, 24138.25, 24217.75, 1590, "Apr 2, 2026 1:46 PM"),
        ("SELL", 1, 23758.75, 23714.00, 895, "Apr 2, 2026 7:15 AM"),
        ("SELL", 2, 23788.50, 23771.25, 690, "Apr 2, 2026 6:15 AM"),
        ("SELL", 1, 24154.00, 24201.00, -940, "Apr 1, 2026 2:30 PM"),
        ("BUY", 1, 24249.50, 24154.00, -1910, "Apr 1, 2026 11:30 AM"),
        ("BUY", 1, 23775.75, 23909.50, 2675, "Mar 31, 2026 1:00 PM"),
        ("BUY", 1, 23492.00, 23642.00, 3000, "Mar 31, 2026 11:30 AM"),
        ("BUY", 1, 23524.50, 23498.25, -525, "Mar 31, 2026 10:15 AM"),
        ("SELL", 2, 23393.25, 23048.25, 13800, "Mar 30, 2026 9:45 AM"),
        ("BUY", 1, 23466.50, 23431.75, -695, "Mar 30, 2026 7:45 AM"),
        ("BUY", 1, 23386.75, 23472.00, 1705, "Mar 30, 2026 6:45 AM"),
        ("BUY", 1, 24450.50, 24373.50, -1540, "Mar 25, 2026 9:15 AM"),
        ("SELL", 1, 24432.00, 24450.50, -370, "Mar 25, 2026 8:30 AM"),
        ("BUY", 1, 24443.50, 24437.75, -115, "Mar 25, 2026 6:30 AM"),
        ("SELL", 1, 24244.50, 24190.50, 1080, "Mar 24, 2026 11:00 AM"),
        ("SELL", 1, 24373.50, 24285.75, 1755, "Mar 24, 2026 7:45 AM"),
        ("BUY", 1, 24390.50, 24373.50, -340, "Mar 24, 2026 6:30 AM"),
        ("BUY", 1, 24369.00, 24398.75, 595, "Mar 23, 2026 1:00 PM"),
        ("SELL", 1, 24317.50, 24369.00, -1030, "Mar 23, 2026 12:30 PM"),
        ("BUY", 1, 24516.00, 24317.50, -3970, "Mar 23, 2026 7:30 AM"),
        ("SELL", 1, 24306.25, 23987.50, 6375, "Mar 20, 2026 12:30 PM"),
        ("SELL", 1, 24813.00, 24651.25, 3235, "Mar 18, 2026 2:45 PM"),
        ("SELL", 1, 24874.25, 24795.00, 1585, "Mar 18, 2026 2:15 PM"),
        ("BUY", 2, 25025.25, 25010.00, -610, "Mar 17, 2026 10:15 AM"),
        ("BUY", 1, 24704.75, 24670.25, -690, "Mar 16, 2026 3:00 PM"),
        ("SELL", 1, 24595.50, 24477.00, 2370, "Mar 13, 2026 11:30 AM"),
        ("BUY", 1, 24726.00, 24595.50, -2610, "Mar 13, 2026 10:45 AM"),
        ("BUY", 1, 24667.75, 24697.50, 595, "Mar 13, 2026 10:00 AM"),
    ]

    print()
    print("=" * 120)
    print("  ATR MULTIPLE ANALYSIS — Friend's Trade Exits")
    print("=" * 120)

    hdr = (f"{'#':>2} {'Side':<5} {'Entry':>10} {'Exit':>10} {'PnL':>8} "
           f"{'BarCl':>10} {'ATR14':>8} {'ExitDist':>9} {'ATRmult':>8} {'Type':<6}")
    print(hdr)
    print("-" * len(hdr))

    sl_multiples = []
    tp_multiples = []
    all_multiples = []

    for idx, (side, qty, entry, exit_p, pnl, time_str) in enumerate(user_trades, 1):
        dt = pd.Timestamp(time_str, tz=ET)

        mask = abs((df.index - dt).total_seconds()) <= 900
        if not mask.any():
            print(f"{idx:>2} {side:<5} {entry:>10.2f} {exit_p:>10.2f} "
                  f"{pnl:>+8} -- NO MATCHING BAR --")
            continue

        nearest_idx = df.index[mask][abs((df.index[mask] - dt).total_seconds()).argmin()]
        row = df.loc[nearest_idx]

        bc = row["close"]
        atr = row["atr"]

        if pd.isna(atr) or atr <= 0:
            print(f"{idx:>2} {side:<5} {entry:>10.2f} {exit_p:>10.2f} "
                  f"{pnl:>+8} {bc:>10.2f} {'NaN':>8} -- ATR unavailable --")
            continue

        # Exit distance: positive = in favor, negative = against
        if side == "BUY":
            exit_dist = exit_p - bc  # positive if exit > bar close (profit)
        else:
            exit_dist = bc - exit_p  # positive if exit < bar close (profit for short)

        atr_mult = exit_dist / atr
        trade_type = "TP" if pnl > 0 else "SL"

        if pnl > 0:
            tp_multiples.append(atr_mult)
        elif pnl < 0:
            sl_multiples.append(atr_mult)
        all_multiples.append((idx, side, entry, exit_p, pnl, bc, atr, exit_dist, atr_mult, trade_type))

        print(f"{idx:>2} {side:<5} {entry:>10.2f} {exit_p:>10.2f} "
              f"{pnl:>+8} {bc:>10.2f} {atr:>8.2f} {exit_dist:>+9.2f} {atr_mult:>+8.2f} {trade_type:<6}")

    # Summary statistics
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    if tp_multiples:
        tp_arr = np.array(tp_multiples)
        print(f"\n  TP exits (winners, n={len(tp_arr)}):")
        print(f"    Mean:   {tp_arr.mean():+.2f} x ATR")
        print(f"    Median: {np.median(tp_arr):+.2f} x ATR")
        print(f"    Min:    {tp_arr.min():+.2f} x ATR")
        print(f"    Max:    {tp_arr.max():+.2f} x ATR")
        print(f"    Std:    {tp_arr.std():.2f}")

        # Distribution buckets
        print(f"\n    Distribution of TP multiples:")
        for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
                       (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 10.0)]:
            cnt = ((tp_arr >= lo) & (tp_arr < hi)).sum()
            if cnt > 0:
                print(f"      [{lo:.1f}, {hi:.1f}): {cnt}")

    if sl_multiples:
        sl_arr = np.array(sl_multiples)
        print(f"\n  SL exits (losers, n={len(sl_arr)}):")
        print(f"    Mean:   {sl_arr.mean():+.2f} x ATR")
        print(f"    Median: {np.median(sl_arr):+.2f} x ATR")
        print(f"    Min:    {sl_arr.min():+.2f} x ATR")
        print(f"    Max:    {sl_arr.max():+.2f} x ATR")
        print(f"    Std:    {sl_arr.std():.2f}")

        # Distribution buckets (SL multiples are negative)
        print(f"\n    Distribution of SL multiples:")
        for lo, hi in [(-10, -4), (-4, -3), (-3, -2.5), (-2.5, -2.0),
                       (-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0)]:
            cnt = ((sl_arr >= lo) & (sl_arr < hi)).sum()
            if cnt > 0:
                print(f"      [{lo:.1f}, {hi:.1f}): {cnt}")

    # Also show using entry price instead of bar close
    print()
    print("=" * 80)
    print("  ALT: Using ENTRY PRICE instead of bar close")
    print("=" * 80)

    tp2 = []
    sl2 = []
    for idx, side, entry, exit_p, pnl, bc, atr, _, _, trade_type in all_multiples:
        if side == "BUY":
            dist = exit_p - entry
        else:
            dist = entry - exit_p
        mult = dist / atr
        if pnl > 0:
            tp2.append(mult)
        else:
            sl2.append(mult)

    if tp2:
        tp2_arr = np.array(tp2)
        print(f"\n  TP exits (from entry, n={len(tp2_arr)}):")
        print(f"    Mean:   {tp2_arr.mean():+.2f} x ATR")
        print(f"    Median: {np.median(tp2_arr):+.2f} x ATR")
        print(f"    Min:    {tp2_arr.min():+.2f} x ATR")
        print(f"    Max:    {tp2_arr.max():+.2f} x ATR")

    if sl2:
        sl2_arr = np.array(sl2)
        print(f"\n  SL exits (from entry, n={len(sl2_arr)}):")
        print(f"    Mean:   {sl2_arr.mean():+.2f} x ATR")
        print(f"    Median: {np.median(sl2_arr):+.2f} x ATR")
        print(f"    Min:    {sl2_arr.min():+.2f} x ATR")
        print(f"    Max:    {sl2_arr.max():+.2f} x ATR")


if __name__ == "__main__":
    main()

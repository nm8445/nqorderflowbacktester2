"""
For each of friend's trades, show the z_vol reading at entry.
Build 15-min bars from 5-min data, compute rough vol signals, report z_vol.
"""
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

# Rough vol params (same as friend's)
H = 0.10; KERNEL_LEN = 100; ETA = 1.0; V0 = 0.0001
NORM_LEN = 200; Z_LOOKBACK = 100; HIGH_Z = 1.0; LOW_Z = -1.0
EMA_LEN = 50; ATR_LEN = 14
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
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"    +", line)
        if len(parts) < 9:
            continue
        action = parts[1]
        qty = int(parts[2])
        entry = float(parts[3].replace("$", "").replace(",", ""))
        exit_p = float(parts[4].replace("$", "").replace(",", ""))
        pnl_str = parts[5].replace("$", "").replace(",", "").replace("+", "")
        pnl = float(pnl_str)
        entry_time = parts[7]
        trades.append((action, qty, entry, exit_p, pnl, entry_time))
    return trades


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

    # Compute rough vol z-score
    closes = df["close"]
    n = len(closes)
    ret = np.log(closes / closes.shift(1))
    mean_ret = ret.rolling(NORM_LEN).mean()
    std_ret = ret.rolling(NORM_LEN).std()
    shock = ((ret - mean_ret) / std_ret.replace(0, np.nan)).fillna(0)

    ks = np.arange(1, KERNEL_LEN + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = ks ** (H - 0.5) - (ks - 1) ** (H - 0.5)
    kernel[0] = 1.0
    xH = np.convolve(shock.values, kernel, mode="full")[:n]
    v_model = np.clip(V0 * np.exp(ETA * xH), 0, V0 * 1e4)

    v_series = pd.Series(v_model, index=df.index)
    v_mean = v_series.rolling(Z_LOOKBACK).mean()
    v_std = v_series.rolling(Z_LOOKBACK).std()
    df["z_vol"] = ((v_series - v_mean) / v_std.replace(0, np.nan)).values
    df["ema"] = closes.ewm(span=EMA_LEN, adjust=False).mean()

    # Normalize to ET
    df.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df.index
    ])

    # Parse trades
    trades = parse_trades(LOG_FILE)
    print(f"Loaded {len(trades)} trades")

    print()
    hdr = (f"{'#':>3} {'Side':<5} {'Q':>1} {'Entry':>10} {'PnL':>8} "
           f"{'z_vol':>7} {'Close':>10} {'EMA50':>10} {'EMAdir':<7} {'Signal':<8} {'EntryTime':<22}")
    print(hdr)
    print("-" * len(hdr))

    zvols = []
    zvols_long = []
    zvols_short = []
    matched = 0
    no_match = 0

    for idx, (side, qty, entry, exit_p, pnl, entry_time_str) in enumerate(trades, 1):
        month_str = entry_time_str.split(",")[0].split(" ")[0]
        if month_str in ("Oct", "Nov", "Dec"):
            year = 2025
        else:
            year = 2026

        try:
            dt = pd.Timestamp(f"{entry_time_str}, {year}", tz=ET)
        except Exception:
            print(f"{idx:>3} {side:<5} {qty:>1} {entry:>10.2f} {pnl:>+8.0f} PARSE ERROR")
            continue

        # Find bar whose close matches entry price (within 1-hour window before entry)
        window_start = dt - pd.Timedelta(hours=1)
        window_end = dt + pd.Timedelta(minutes=15)
        mask = (df.index >= window_start) & (df.index <= window_end)
        candidates = df[mask]

        if candidates.empty:
            print(f"{idx:>3} {side:<5} {qty:>1} {entry:>10.2f} {pnl:>+8.0f} {'--':>7} {'--':>10} {'--':>10} {'--':<7} {'--':<8} {entry_time_str}")
            no_match += 1
            continue

        # Match on close price
        close_diffs = (candidates["close"] - entry).abs()
        best_idx = close_diffs.idxmin()
        best_diff = close_diffs[best_idx]

        row = df.loc[best_idx]
        zv = row["z_vol"]
        bc = row["close"]
        ema = row["ema"]

        if pd.isna(zv):
            print(f"{idx:>3} {side:<5} {qty:>1} {entry:>10.2f} {pnl:>+8.0f} {'NaN':>7} {bc:>10.2f} {ema:>10.2f} {'--':<7} {'--':<8} {entry_time_str} d={best_diff:.1f}")
            continue

        matched += 1
        zvols.append(zv)
        if side == "BUY":
            zvols_long.append(zv)
        else:
            zvols_short.append(zv)

        # What signal would fire?
        ema_dir = "above" if bc > ema else "below"
        if zv > HIGH_Z and bc > ema:
            sig = "LONG"
        elif zv > HIGH_Z and bc < ema:
            sig = "SHORT"
        elif zv > HIGH_Z:
            sig = "HIGH-Z"
        else:
            sig = f"z<{HIGH_Z}"

        # Does trade direction match signal?
        match_flag = ""
        if side == "BUY" and sig == "LONG":
            match_flag = " OK"
        elif side == "SELL" and sig == "SHORT":
            match_flag = " OK"
        elif side == "BUY" and sig == "SHORT":
            match_flag = " WRONG"
        elif side == "SELL" and sig == "LONG":
            match_flag = " WRONG"

        d_flag = f" d={best_diff:.1f}" if best_diff > 2.0 else ""

        print(f"{idx:>3} {side:<5} {qty:>1} {entry:>10.2f} {pnl:>+8.0f} "
              f"{zv:>+7.2f} {bc:>10.2f} {ema:>10.2f} {ema_dir:<7} {sig:<8} {entry_time_str}{match_flag}{d_flag}")

    # Summary
    print()
    print("=" * 90)
    print("  Z_VOL SUMMARY")
    print("=" * 90)
    print(f"  Matched trades: {matched}/{len(trades)} (no match: {no_match})")

    if zvols:
        arr = np.array(zvols)
        print(f"\n  ALL TRADES (n={len(arr)}):")
        print(f"    Mean:   {arr.mean():+.3f}")
        print(f"    Median: {np.median(arr):+.3f}")
        print(f"    Std:    {arr.std():.3f}")
        print(f"    Min:    {arr.min():+.3f}")
        print(f"    Max:    {arr.max():+.3f}")
        print(f"    P10:    {np.percentile(arr, 10):+.3f}")
        print(f"    P25:    {np.percentile(arr, 25):+.3f}")
        print(f"    P75:    {np.percentile(arr, 75):+.3f}")
        print(f"    P90:    {np.percentile(arr, 90):+.3f}")

        print(f"\n    Distribution:")
        for lo, hi in [(-5, -2), (-2, -1), (-1, -0.5), (-0.5, 0),
                       (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0),
                       (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 20.0)]:
            cnt = ((arr >= lo) & (arr < hi)).sum()
            if cnt > 0:
                bar = "#" * cnt
                print(f"      [{lo:>5.1f}, {hi:>5.1f}): {cnt:>3}  {bar}")

        above_1 = (arr >= 1.0).sum()
        above_05 = (arr >= 0.5).sum()
        above_0 = (arr >= 0.0).sum()
        print(f"\n    z_vol >= 1.0 (HIGH_Z): {above_1}/{len(arr)} ({100*above_1/len(arr):.1f}%)")
        print(f"    z_vol >= 0.5:          {above_05}/{len(arr)} ({100*above_05/len(arr):.1f}%)")
        print(f"    z_vol >= 0.0:          {above_0}/{len(arr)} ({100*above_0/len(arr):.1f}%)")

    if zvols_long:
        arr_l = np.array(zvols_long)
        print(f"\n  LONG TRADES (n={len(arr_l)}):")
        print(f"    Mean:   {arr_l.mean():+.3f}")
        print(f"    Median: {np.median(arr_l):+.3f}")
        print(f"    z_vol >= 1.0: {(arr_l >= 1.0).sum()}/{len(arr_l)}")

    if zvols_short:
        arr_s = np.array(zvols_short)
        print(f"\n  SHORT TRADES (n={len(arr_s)}):")
        print(f"    Mean:   {arr_s.mean():+.3f}")
        print(f"    Median: {np.median(arr_s):+.3f}")
        print(f"    z_vol >= 1.0: {(arr_s >= 1.0).sum()}/{len(arr_s)}")


if __name__ == "__main__":
    main()

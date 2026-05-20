"""
Detailed trade log for April 2026 — Rough Vol strategy.
NORM=350, ZLOOK=150, HIGH_Z=1.0, LOW_Z=-1.0, marti streak=1.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
H=0.10; KERNEL_LEN=100; ETA=1.0; V0=0.0001
NORM_LEN=300; Z_LOOKBACK=200; HIGH_Z=1.0; LOW_Z=1.0; EMA_LEN=50; ATR_LEN=14
ATR_SL=2.0; ATR_TP=3.0; ET="America/New_York"; POINT_VALUE=20.0
SS="09:30"; SE="16:00"; MT=5
MART_STREAK=1; MART_MULT=1.5; MART_MAX_DOUBLES=4

TARGET_MONTH = 0
TARGET_YEAR = 0


def main():
    # Build bars
    frames = []
    for f in sorted(TIMEBARS_DIR.glob("timebars_5min_202*.pkl")):
        with open(f, "rb") as fh:
            bars = pickle.load(fh)
        if not bars: continue
        rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
                 "low": b["low"], "close": b["close"]} for b in bars]
        df5 = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df5["group"] = df5.index.floor("15min")
        agg = df5.groupby("group").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"))
        agg.index += pd.Timedelta(minutes=15)
        frames.append(agg)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    closes = df["close"]; n = len(closes)

    # Signals
    ret = np.log(closes / closes.shift(1)); ret.iloc[0] = 0.0
    mean_ret = ret.rolling(NORM_LEN).mean()
    std_ret = ret.rolling(NORM_LEN).std(ddof=0)
    shock = np.nan_to_num(np.where(std_ret > 0, (ret - mean_ret) / std_ret, 0.0), nan=0.0)
    ks = np.arange(1, KERNEL_LEN + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = ks ** (H - 0.5) - (ks - 1) ** (H - 0.5)
    kernel[0] = 1.0
    xH = np.convolve(shock, kernel, mode="full")[:n]
    v_model = np.minimum(V0 * np.exp(ETA * xH), V0 * 1e4)
    v_s = pd.Series(v_model, index=df.index)
    v_m = v_s.rolling(Z_LOOKBACK).mean()
    v_d = v_s.rolling(Z_LOOKBACK).std(ddof=0)
    df["z_vol"] = np.nan_to_num(np.where(v_d > 0, (v_s - v_m) / v_d, 0.0), nan=0.0)
    df["ema"] = closes.ewm(span=EMA_LEN, adjust=False).mean()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - closes.shift()).abs(),
                    (df["low"] - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n); atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i+1].mean()
        else: atr_vals[i] = (atr_vals[i-1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    df.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df.index
    ])

    # Backtest — full history but only log April
    trades = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
    qty = 1; loss_streak = 0
    entry_time = None; entry_z = 0; entry_type = ""

    for i in range(len(df)):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE

        # Session close
        if pos != 0 and not ins and hm >= SE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE * qty
            if td.month == TARGET_MONTH and td.year == TARGET_YEAR:
                trades.append({
                    "date": td, "action": "BUY" if d == "long" else "SELL",
                    "qty": qty, "entry": ep, "exit": row["close"],
                    "pnl": pnl, "reason": "SESSION",
                    "entry_time": entry_time, "exit_time": idx,
                    "entry_type": entry_type, "z": entry_z,
                })
            if pnl > 0: loss_streak = 0
            else: loss_streak += 1
            pos = 0

        if not ins: continue
        if td != cd: cd = td; dt = 0; loss_streak = 0

        if loss_streak >= MART_STREAK:
            steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
            qty = max(1, round(MART_MULT ** steps))
        else: qty = 1

        atr_v = row["atr"]
        if atr_v <= 0: continue
        z = row["z_vol"]; cl = row["close"]; ema = row["ema"]

        # Manage position
        if pos != 0:
            xp = None; xr = None
            if d == "long":
                if row["low"] <= sl: xp, xr = sl, "SL"
                elif row["high"] >= tp: xp, xr = tp, "TP"
            else:
                if row["high"] >= sl: xp, xr = sl, "SL"
                elif row["low"] <= tp: xp, xr = tp, "TP"
            if xp:
                pnl = ((xp - ep) if d == "long" else (ep - xp)) * POINT_VALUE * qty
                if td.month == TARGET_MONTH and td.year == TARGET_YEAR:
                    trades.append({
                        "date": td, "action": "BUY" if d == "long" else "SELL",
                        "qty": qty, "entry": ep, "exit": xp,
                        "pnl": pnl, "reason": xr,
                        "entry_time": entry_time, "exit_time": idx,
                        "entry_type": entry_type, "z": entry_z,
                    })
                if pnl > 0: loss_streak = 0
                else: loss_streak += 1
                if loss_streak >= MART_STREAK:
                    steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                    qty = max(1, round(MART_MULT ** steps))
                else: qty = 1
                pos = 0; continue

        # Entry
        if pos == 0 and dt < MT:
            entered = False
            if z > HIGH_Z and cl > ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v
                entered = True; entry_type = "TREND"
            elif z > HIGH_Z and cl < ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v
                entered = True; entry_type = "TREND"
            if not entered and z < -LOW_Z:
                if cl < ema:
                    pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v
                    entered = True; entry_type = "MEAN-REV"
                elif cl > ema:
                    pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v
                    entered = True; entry_type = "MEAN-REV"
            if entered:
                dt += 1
                entry_time = idx
                entry_z = z

    # Print
    print(f"{'Symbol':<6} {'Action':<6} {'Qty':>4} {'Entry':>12} {'Exit':>12} {'P&L':>12} {'Duration':<20} {'Entry Time':<22} {'Exit Time':<22} {'Type':<10} {'Exit Rsn':<8} {'z_vol':>6}")
    print("-" * 170)

    current_date = None
    day_pnl = 0.0
    month_pnl = 0.0

    for t in trades:
        if t["date"] != current_date:
            if current_date is not None:
                print(f"{'':>42} Day total: ${day_pnl:>+10,.2f}")
                print()
            current_date = t["date"]
            day_pnl = 0.0
            print(f"--- {t['date'].strftime('%A, %B %d, %Y')} ---")

        dur = t["exit_time"] - t["entry_time"]
        hours = int(dur.total_seconds() // 3600)
        mins = int((dur.total_seconds() % 3600) // 60)
        if hours > 0:
            dur_str = f"{hours}h {mins}m"
        else:
            dur_str = f"{mins}m"

        entry_str = t["entry_time"].strftime("%b %d, %I:%M %p")
        exit_str = t["exit_time"].strftime("%b %d, %I:%M %p")

        print(f"{'NQ':<6} {t['action']:<6} {t['qty']:>4} ${t['entry']:>10,.2f} ${t['exit']:>10,.2f} ${t['pnl']:>+10,.2f} {dur_str:<20} {entry_str:<22} {exit_str:<22} {t['entry_type']:<10} {t['reason']:<8} {t['z']:>+6.2f}")

        day_pnl += t["pnl"]
        month_pnl += t["pnl"]

    if current_date is not None:
        print(f"{'':>42} Day total: ${day_pnl:>+10,.2f}")

    print()
    print("=" * 80)
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    pf = sum(wins) / abs(sum(losses)) if losses else 99
    print(f"April {TARGET_YEAR} Summary: {len(trades)} trades | PF={pf:.2f} | "
          f"PnL=${month_pnl:+,.2f} | WR={100*len(wins)/len(trades):.1f}%" if trades else "No trades")


if __name__ == "__main__":
    main()

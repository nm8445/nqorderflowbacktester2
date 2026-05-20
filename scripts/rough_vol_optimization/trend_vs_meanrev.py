"""
Trend vs Mean-Revert breakdown — NORM=300, ZLOOK=200, streak=1.
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


def main():
    frames = []
    for f in sorted(TIMEBARS_DIR.glob("timebars_5min_202*.pkl")):
        with open(f, "rb") as fh:
            bars = pickle.load(fh)
        if not bars: continue
        rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
                 "low": b["low"], "close": b["close"]} for b in bars]
        df5 = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df5["group"] = df5.index.floor("15min")
        agg = df5.groupby("group").agg(open=("open","first"), high=("high","max"),
                                        low=("low","min"), close=("close","last"))
        agg.index += pd.Timedelta(minutes=15)
        frames.append(agg)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    closes=df["close"]; highs=df["high"]; lows=df["low"]; n=len(closes)

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
    v_m = v_s.rolling(Z_LOOKBACK).mean(); v_d = v_s.rolling(Z_LOOKBACK).std(ddof=0)
    df["z_vol"] = np.nan_to_num(np.where(v_d > 0, (v_s - v_m) / v_d, 0.0), nan=0.0)
    df["ema"] = closes.ewm(span=EMA_LEN, adjust=False).mean()
    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(), (lows - closes.shift()).abs()], axis=1).max(axis=1)
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

    trades = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
    qty = 1; loss_streak = 0; entry_type = ""

    for i in range(len(df)):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE
        if pos != 0 and not ins and hm >= SE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE * qty
            trades.append({"pnl": pnl, "type": entry_type, "dir": d, "qty": qty})
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
        if pos != 0:
            xp = None
            if d == "long":
                if row["low"] <= sl: xp = sl
                elif row["high"] >= tp: xp = tp
            else:
                if row["high"] >= sl: xp = sl
                elif row["low"] <= tp: xp = tp
            if xp:
                pnl = ((xp - ep) if d == "long" else (ep - xp)) * POINT_VALUE * qty
                trades.append({"pnl": pnl, "type": entry_type, "dir": d, "qty": qty})
                if pnl > 0: loss_streak = 0
                else: loss_streak += 1
                if loss_streak >= MART_STREAK:
                    steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                    qty = max(1, round(MART_MULT ** steps))
                else: qty = 1
                pos = 0; continue
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
            if entered: dt += 1

    # Breakdown
    trend = [t for t in trades if t["type"] == "TREND"]
    mr = [t for t in trades if t["type"] == "MEAN-REV"]

    def stats(label, tlist):
        if not tlist:
            print(f"{label}: No trades")
            return
        pnls = np.array([t["pnl"] for t in tlist])
        w = pnls[pnls > 0]; l = pnls[pnls < 0]
        pf = w.sum() / abs(l.sum()) if len(l) else 99
        wr = 100 * len(w) / len(pnls)
        cum = pnls.cumsum(); dd = (cum - np.maximum.accumulate(cum)).min()
        longs = [t["pnl"] for t in tlist if t["dir"] == "long"]
        shorts = [t["pnl"] for t in tlist if t["dir"] == "short"]
        print(f"{label}")
        print(f"  Trades: {len(pnls)}  PF: {pf:.2f}  WR: {wr:.1f}%  PnL: ${pnls.sum():+,.0f}  DD: ${dd:,.0f}")
        print(f"  Long: {len(longs)}  Short: {len(shorts)}")
        print(f"  Avg Win: ${w.mean():+,.0f}  Avg Loss: ${l.mean():+,.0f}" if len(w) and len(l) else "")
        print()

    print("=" * 60)
    stats("ALL TRADES", trades)
    stats("TREND (z > 1.0)", trend)
    stats("MEAN-REVERT (z < -1.0)", mr)

    pct_trend = 100 * len(trend) / len(trades)
    pct_mr = 100 * len(mr) / len(trades)
    print(f"Ratio: {len(trend)} trend ({pct_trend:.0f}%) vs {len(mr)} mean-revert ({pct_mr:.0f}%)")


if __name__ == "__main__":
    main()

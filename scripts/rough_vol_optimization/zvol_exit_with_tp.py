"""
z_vol exit + SL + TP strategy — NO Hurst.
Norm=201, ZLook=23, HIGH_Z=1.1, EMA=40
SL=2.0x ATR, TP=1.2x ATR, also exit when z_vol <= -1.0
Whichever hits first: SL, TP, z_vol exit, or force close 15:27.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

H = 0.40; KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
NORM_LEN = 201; Z_LOOKBACK = 23; HIGH_Z = 1.1; LOW_Z = -1.0; EMA_LEN = 40; ATR_LEN = 14
ATR_SL = 2.0; ATR_TP = 1.2; ET = "America/New_York"; POINT_VALUE = 20.0
SS = "09:30"; SE = "15:30"; FORCE_CLOSE = "15:27"; MT = 5
MART_STREAK = 1; MART_MULT = 2.0; MART_MAX_DOUBLES = 4


def build():
    df_mt = pd.read_parquet(MARKETTICK_PARQUET)
    frames = []
    for f in sorted(TIMEBARS_DIR.glob("timebars_5min_202*.pkl")):
        with open(f, "rb") as fh:
            bars = pickle.load(fh)
        if not bars: continue
        rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
                 "low": b["low"], "close": b["close"]} for b in bars]
        df5 = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df5["group"] = df5.index.floor("15min")
        agg = df5.groupby("group").agg(open=("open", "first"), high=("high", "max"),
                                        low=("low", "min"), close=("close", "last"))
        agg.index += pd.Timedelta(minutes=15)
        frames.append(agg)
    df_pkl = pd.concat(frames).sort_index()
    df_pkl = df_pkl[~df_pkl.index.duplicated(keep="first")]
    cutoff = df_mt.index[-1]
    df_pkl_new = df_pkl[df_pkl.index > cutoff]
    df = pd.concat([df_mt[["open", "high", "low", "close"]], df_pkl_new[["open", "high", "low", "close"]]]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df.index
    ])
    return df


def main():
    print("Loading...", flush=True)
    df = build()
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)

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

    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(),
                    (lows - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n); atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i+1].mean()
        else: atr_vals[i] = (atr_vals[i-1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    # IS/OOS
    trading_days = sorted(set(idx.date() for idx in df.index))
    split_idx = int(len(trading_days) * 0.60)
    is_cutoff = str(trading_days[split_idx])

    for use_mart, mlabel in [(False, "FLAT"), (True, "MARTINGALE (2x)")]:
        print(f"\n{'='*60}\n{mlabel}\n{'='*60}", flush=True)
        trades = []
        pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
        qty = 1; loss_streak = 0

        for i in range(n):
            row = df.iloc[i]; idx = df.index[i]; td = idx.date()
            hm = idx.strftime("%H:%M"); ins = SS <= hm < SE

            if pos != 0 and hm >= FORCE_CLOSE:
                pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE * qty
                trades.append({"pnl": round(pnl, 2), "date": str(td), "exit": "CLOSE"})
                if pnl > 0: loss_streak = 0
                else: loss_streak += 1
                pos = 0; continue

            if not ins: continue
            if td != cd: cd = td; dt = 0; loss_streak = 0

            if use_mart and loss_streak >= MART_STREAK:
                steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                qty = max(1, round(MART_MULT ** steps))
            else:
                qty = 1

            atr_v = row["atr"]
            if atr_v <= 0: continue
            z = row["z_vol"]; cl = row["close"]; ema = row["ema"]

            if pos != 0:
                # Check SL and TP first (price-based exits take priority)
                xp = None; xr = None
                if d == "long":
                    if row["low"] <= sl: xp = sl; xr = "SL"
                    elif row["high"] >= tp: xp = tp; xr = "TP"
                else:
                    if row["high"] >= sl: xp = sl; xr = "SL"
                    elif row["low"] <= tp: xp = tp; xr = "TP"

                if xp:
                    pnl = ((xp - ep) if d == "long" else (ep - xp)) * POINT_VALUE * qty
                    trades.append({"pnl": round(pnl, 2), "date": str(td), "exit": xr})
                    if pnl > 0: loss_streak = 0
                    else: loss_streak += 1
                    if use_mart and loss_streak >= MART_STREAK:
                        steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                        qty = max(1, round(MART_MULT ** steps))
                    else: qty = 1
                    pos = 0; continue

                # z_vol exit (vol contraction)
                if z <= LOW_Z:
                    pnl = ((cl - ep) if d == "long" else (ep - cl)) * POINT_VALUE * qty
                    trades.append({"pnl": round(pnl, 2), "date": str(td), "exit": "Z_EXIT"})
                    if pnl > 0: loss_streak = 0
                    else: loss_streak += 1
                    if use_mart and loss_streak >= MART_STREAK:
                        steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                        qty = max(1, round(MART_MULT ** steps))
                    else: qty = 1
                    pos = 0; continue

            # Entry
            if pos == 0 and dt < MT:
                if z > HIGH_Z and cl > ema:
                    pos = 1; d = "long"; ep = cl
                    sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
                elif z > HIGH_Z and cl < ema:
                    pos = -1; d = "short"; ep = cl
                    sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1

        # Results
        if not trades:
            print("No trades"); continue

        pnls = np.array([t["pnl"] for t in trades])
        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) else 0
        wr = 100 * len(wins) / len(pnls)
        cum = pnls.cumsum(); dd = (cum - np.maximum.accumulate(cum)).min()
        is_t = [t for t in trades if t["date"] < is_cutoff]
        oos_t = [t for t in trades if t["date"] >= is_cutoff]
        is_p = np.array([t["pnl"] for t in is_t]) if is_t else np.array([0.0])
        oos_p = np.array([t["pnl"] for t in oos_t]) if oos_t else np.array([0.0])
        is_pf = is_p[is_p > 0].sum() / abs(is_p[is_p < 0].sum()) if (is_p < 0).any() else 0
        oos_pf = oos_p[oos_p > 0].sum() / abs(oos_p[oos_p < 0].sum()) if (oos_p < 0).any() else 0

        print(f"\nSL={ATR_SL}x TP={ATR_TP}x | Exit also on z<={LOW_Z} | NO HURST")
        print(f"Trades: {len(pnls)}  WR: {wr:.1f}%  PF: {pf:.2f}")
        print(f"IS PF: {is_pf:.2f}  OOS PF: {oos_pf:.2f}")
        print(f"PnL: ${pnls.sum():+,.0f}  MaxDD: ${dd:+,.0f}  Avg: ${pnls.mean():+,.0f}")
        print(f"Avg Win: ${wins.mean():+,.0f}  Avg Loss: ${losses.mean():+,.0f}")

        # Exit breakdown
        exits = defaultdict(list)
        for t in trades:
            exits[t["exit"]].append(t["pnl"])
        print(f"\nExit breakdown:")
        for reason in sorted(exits.keys()):
            ep_arr = np.array(exits[reason])
            epf = ep_arr[ep_arr > 0].sum() / abs(ep_arr[ep_arr < 0].sum()) if (ep_arr < 0).any() else 0
            ewr = 100 * sum(1 for p in ep_arr if p > 0) / len(ep_arr)
            print(f"  {reason:>7}: {len(ep_arr):>5} trades  WR: {ewr:.1f}%  PF: {epf:.2f}  PnL: ${ep_arr.sum():+,.0f}  Avg: ${ep_arr.mean():+,.0f}")

        # Yearly
        print(f"\n{'Year':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>11}")
        print("-" * 50)
        yearly = defaultdict(list)
        for t in trades:
            yearly[t["date"][:4]].append(t)
        for yr in sorted(yearly.keys()):
            yr_t = yearly[yr]
            yr_p = np.array([t["pnl"] for t in yr_t])
            yr_pf = yr_p[yr_p > 0].sum() / abs(yr_p[yr_p < 0].sum()) if (yr_p < 0).any() else 0
            yr_wr = 100 * sum(1 for p in yr_p if p > 0) / len(yr_p)
            print(f"{yr:>6} | {len(yr_t):>6} | {yr_wr:>5.1f} | {yr_pf:>5.2f} | ${yr_p.sum():>+10,.0f}")


if __name__ == "__main__":
    main()

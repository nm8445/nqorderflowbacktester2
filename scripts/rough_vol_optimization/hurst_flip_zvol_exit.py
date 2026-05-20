"""
Hurst flip with alternate params:
  Norm=201, ZLook=23, HIGH_Z=1.1, EMA=40, SL=2.0x ATR
  NO TP — exit when z_vol <= -1.0 (vol contraction exit)
  Keep Hurst flip: Normal H<0.56, Flip H>=0.60
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

H = 0.40; KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
NORM_LEN = 201; Z_LOOKBACK = 23; HIGH_Z = 1.1; LOW_Z = -1.0; EMA_LEN = 40; ATR_LEN = 14
ATR_SL = 2.0; ET = "America/New_York"; POINT_VALUE = 20.0
SS = "09:30"; SE = "15:30"; FORCE_CLOSE = "15:27"; MT = 5

HURST_WINDOW = 30; H_NORMAL_MAX = 0.56; H_FLIP_MIN = 0.60


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


def hurst_rs(x, min_chunk=8):
    n = len(x)
    if n < 50: return np.nan
    chunk_sizes = []
    s = min_chunk
    while s <= n // 2:
        chunk_sizes.append(s)
        s = int(s * 1.5)
        if s == chunk_sizes[-1]: s += 1
    log_ns = []; log_rs = []
    for cs in chunk_sizes:
        n_chunks = n // cs
        if n_chunks < 1: continue
        rs_vals = []
        for i in range(n_chunks):
            chunk = x[i*cs:(i+1)*cs]
            m = chunk.mean()
            y = np.cumsum(chunk - m)
            r = y.max() - y.min()
            sd = chunk.std(ddof=1)
            if sd > 0: rs_vals.append(r / sd)
        if rs_vals:
            log_ns.append(np.log(cs))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_ns) < 3: return np.nan
    return np.polyfit(log_ns, log_rs, 1)[0]


def main():
    print("Loading...", flush=True)
    df = build()
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)

    # z_vol with new params
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

    # ATR
    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(),
                    (lows - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n); atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i+1].mean()
        else: atr_vals[i] = (atr_vals[i-1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    # 30-day Hurst
    print("Computing 30-day rolling Hurst...", flush=True)
    session_mask = (df.index.strftime("%H:%M") >= SS) & (df.index.strftime("%H:%M") < SE)
    session_df = df[session_mask]
    log_ret = np.log(session_df["close"] / session_df["close"].shift(1)).dropna()
    dates = sorted(set(idx.date() for idx in log_ret.index))
    ret_by_date = {}
    for d in dates:
        mask = log_ret.index.date == d
        ret_by_date[d] = log_ret[mask].values
    t0 = time.time()
    daily_hurst = {}
    for i, d in enumerate(dates):
        start = max(0, i - HURST_WINDOW + 1)
        window_dates = dates[start:i+1]
        all_rets = np.concatenate([ret_by_date[wd] for wd in window_dates])
        if len(all_rets) >= 50:
            daily_hurst[d] = hurst_rs(all_rets)
        else:
            daily_hurst[d] = np.nan
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)

    # IS/OOS
    trading_days = sorted(set(idx.date() for idx in df.index))
    split_idx = int(len(trading_days) * 0.60)
    is_cutoff = str(trading_days[split_idx])

    # Backtest: entry on z>HIGH_Z, exit on z<=LOW_Z or SL, force close 15:27
    # No TP. Exit signal = z_vol drops to -1.0 or below.
    print("Running backtest...", flush=True)
    trades = []
    pos = 0; ep = 0; sl = 0; d = ""; cd = None; dt = 0; src = ""

    for i in range(n):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE

        # Force close
        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE
            trades.append({"pnl": round(pnl, 2), "date": str(td), "src": src, "exit": "CLOSE"})
            pos = 0; continue

        if not ins: continue
        if td != cd: cd = td; dt = 0

        atr_v = row["atr"]
        if atr_v <= 0: continue
        z = row["z_vol"]; cl = row["close"]; ema = row["ema"]

        # Exit logic (while in position)
        if pos != 0:
            # Check SL
            hit_sl = False
            if d == "long" and row["low"] <= sl:
                pnl = (sl - ep) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td), "src": src, "exit": "SL"})
                pos = 0; hit_sl = True
            elif d == "short" and row["high"] >= sl:
                pnl = (ep - sl) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td), "src": src, "exit": "SL"})
                pos = 0; hit_sl = True

            if hit_sl:
                continue

            # Check z_vol exit (vol contraction)
            if z <= LOW_Z:
                pnl = ((cl - ep) if d == "long" else (ep - cl)) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td), "src": src, "exit": "Z_EXIT"})
                pos = 0; continue

        # Entry
        if pos == 0 and dt < MT:
            cur_h = daily_hurst.get(td, np.nan)
            if np.isnan(cur_h): continue
            flip = False
            if cur_h < H_NORMAL_MAX: pass
            elif cur_h >= H_FLIP_MIN: flip = True
            else: continue

            # Only enter on z > HIGH_Z (no low-z entry since low-z is now exit)
            signal = None
            if z > HIGH_Z and cl > ema:
                signal = "long"
            elif z > HIGH_Z and cl < ema:
                signal = "short"

            if signal is None: continue

            if flip:
                signal = "short" if signal == "long" else "long"
                src = "flip"
            else:
                src = "normal"

            if signal == "long":
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; dt += 1
            else:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; dt += 1

    # Results
    print(f"\nTotal trades: {len(trades)}")
    if not trades:
        return

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

    print(f"\nConfig: Norm={NORM_LEN} ZLook={Z_LOOKBACK} HIGH_Z={HIGH_Z} EMA={EMA_LEN} SL={ATR_SL}x | Exit: z<={LOW_Z} or SL")
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
        print(f"  {reason:>7}: {len(ep_arr):>5} trades  PF: {epf:.2f}  PnL: ${ep_arr.sum():+,.0f}  Avg: ${ep_arr.mean():+,.0f}")

    # Yearly
    print(f"\n{'Year':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>11} | {'Normal':>6} | {'Flip':>6}")
    print("-" * 65)
    yearly = defaultdict(list)
    for t in trades:
        yearly[t["date"][:4]].append(t)
    for yr in sorted(yearly.keys()):
        yr_t = yearly[yr]
        yr_p = np.array([t["pnl"] for t in yr_t])
        yr_pf = yr_p[yr_p > 0].sum() / abs(yr_p[yr_p < 0].sum()) if (yr_p < 0).any() else 0
        yr_wr = 100 * sum(1 for p in yr_p if p > 0) / len(yr_p)
        norm_n = sum(1 for t in yr_t if t["src"] == "normal")
        flip_n = sum(1 for t in yr_t if t["src"] == "flip")
        print(f"{yr:>6} | {len(yr_t):>6} | {yr_wr:>5.1f} | {yr_pf:>5.2f} | ${yr_p.sum():>+10,.0f} | {norm_n:>6} | {flip_n:>6}")

    # Normal vs flip
    norm_t = [t for t in trades if t["src"] == "normal"]
    flip_t = [t for t in trades if t["src"] == "flip"]
    if norm_t:
        np_n = np.array([t["pnl"] for t in norm_t])
        npf = np_n[np_n > 0].sum() / abs(np_n[np_n < 0].sum()) if (np_n < 0).any() else 0
        print(f"\nNormal: {len(norm_t)} trades  WR: {100*sum(1 for p in np_n if p>0)/len(np_n):.1f}%  PF: {npf:.2f}  PnL: ${np_n.sum():+,.0f}")
    if flip_t:
        fp_n = np.array([t["pnl"] for t in flip_t])
        fpf = fp_n[fp_n > 0].sum() / abs(fp_n[fp_n < 0].sum()) if (fp_n < 0).any() else 0
        print(f"Flip:   {len(flip_t)} trades  WR: {100*sum(1 for p in fp_n if p>0)/len(fp_n):.1f}%  PF: {fpf:.2f}  PnL: ${fp_n.sum():+,.0f}")


if __name__ == "__main__":
    main()

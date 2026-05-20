"""
Per-trade H-bucket analysis using 30-day rolling Hurst.
Tags each trade with its entry-day H value, bins into ranges,
shows wins/losses/WR/PF/PnL per bucket.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

H = 0.40; KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
NORM_LEN = 250; Z_LOOKBACK = 100; HIGH_Z = 1.5; LOW_Z = -1.0; EMA_LEN = 50; ATR_LEN = 14
ATR_SL = 1.5; ATR_TP = 2.5; ET = "America/New_York"; POINT_VALUE = 20.0
SS = "09:30"; SE = "15:30"; FORCE_CLOSE = "15:27"; MT = 5

HURST_WINDOW = 30


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
    if n < 50:
        return np.nan
    chunk_sizes = []
    s = min_chunk
    while s <= n // 2:
        chunk_sizes.append(s)
        s = int(s * 1.5)
        if s == chunk_sizes[-1]:
            s += 1
    log_ns = []; log_rs = []
    for cs in chunk_sizes:
        n_chunks = n // cs
        if n_chunks < 1:
            continue
        rs_vals = []
        for i in range(n_chunks):
            chunk = x[i*cs:(i+1)*cs]
            m = chunk.mean()
            y = np.cumsum(chunk - m)
            r = y.max() - y.min()
            sd = chunk.std(ddof=1)
            if sd > 0:
                rs_vals.append(r / sd)
        if rs_vals:
            log_ns.append(np.log(cs))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_ns) < 3:
        return np.nan
    return np.polyfit(log_ns, log_rs, 1)[0]


def main():
    print("Loading bars...", flush=True)
    df = build()
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)

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
    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(),
                    (lows - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n); atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN:
            atr_vals[i] = tr.iloc[:i+1].mean()
        else:
            atr_vals[i] = (atr_vals[i-1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    # Compute 30-day rolling Hurst
    print(f"Computing {HURST_WINDOW}-day rolling Hurst...", flush=True)
    session_mask = (df.index.strftime("%H:%M") >= SS) & (df.index.strftime("%H:%M") < SE)
    session_df = df[session_mask]
    log_ret = np.log(session_df["close"] / session_df["close"].shift(1)).dropna()

    dates = sorted(set(idx.date() for idx in log_ret.index))
    ret_by_date = {}
    for d in dates:
        mask = log_ret.index.date == d
        ret_by_date[d] = log_ret[mask].values

    t0 = time.time()
    daily_h = {}
    for i, d in enumerate(dates):
        start = max(0, i - HURST_WINDOW + 1)
        window_dates = dates[start:i+1]
        all_rets = np.concatenate([ret_by_date[wd] for wd in window_dates])
        if len(all_rets) >= 50:
            daily_h[d] = hurst_rs(all_rets)
        else:
            daily_h[d] = np.nan
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)

    # Run backtest, tag each trade with its entry-day H
    print("Running backtest...", flush=True)
    trades = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0

    for i in range(n):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE
        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE
            trades.append({"pnl": round(pnl, 2), "date": str(td), "h": entry_h})
            pos = 0; continue
        if not ins: continue
        if td != cd: cd = td; dt = 0
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
                pnl = ((xp - ep) if d == "long" else (ep - xp)) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td), "h": entry_h})
                pos = 0; continue
        if pos == 0 and dt < MT:
            entry_h = daily_h.get(td, np.nan)
            if z > HIGH_Z and cl > ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
            elif z > HIGH_Z and cl < ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1
            elif z < LOW_Z and cl < ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
            elif z < LOW_Z and cl > ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1

    print(f"Total trades: {len(trades)}")

    # Filter trades with valid H
    valid = [t for t in trades if not np.isnan(t["h"])]
    print(f"Trades with valid H: {len(valid)}")

    # Bin by H range
    bins = [(0.40, 0.46), (0.46, 0.48), (0.48, 0.50), (0.50, 0.52),
            (0.52, 0.54), (0.54, 0.56), (0.56, 0.58), (0.58, 0.60),
            (0.60, 0.62), (0.62, 0.65), (0.65, 0.75)]

    print(f"\n{'H Range':>12} | {'Trades':>6} | {'Wins':>5} | {'Losses':>6} | {'WR%':>5} | {'Win PnL':>11} | {'Loss PnL':>11} | {'Net PnL':>11} | {'PF':>5} | {'Avg Trade':>9}")
    print("-" * 115)

    all_win_h = []
    all_loss_h = []

    for lo, hi in bins:
        bucket = [t for t in valid if lo <= t["h"] < hi]
        if not bucket:
            print(f"  {lo:.2f}-{hi:.2f} |      0 |")
            continue
        pnls = np.array([t["pnl"] for t in bucket])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        wr = 100 * len(wins) / len(pnls)
        pf = wins.sum() / abs(losses.sum()) if len(losses) else 99.0
        print(f"  {lo:.2f}-{hi:.2f} | {len(pnls):>6} | {len(wins):>5} | {len(losses):>6} | {wr:>5.1f} | "
              f"${wins.sum():>+10,.0f} | ${losses.sum():>+10,.0f} | ${pnls.sum():>+10,.0f} | {pf:>5.2f} | ${pnls.mean():>+8,.0f}")

        for t in bucket:
            if t["pnl"] > 0:
                all_win_h.append(t["h"])
            else:
                all_loss_h.append(t["h"])

    # Summary stats
    print(f"\nWin H mean:  {np.mean(all_win_h):.4f}  std: {np.std(all_win_h):.4f}  median: {np.median(all_win_h):.4f}")
    print(f"Loss H mean: {np.mean(all_loss_h):.4f}  std: {np.std(all_loss_h):.4f}  median: {np.median(all_loss_h):.4f}")
    print(f"Difference:  {np.mean(all_win_h) - np.mean(all_loss_h):.4f}")

    # Cumulative: show H < threshold results
    print(f"\n{'Threshold':>12} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'Net PnL':>11} | {'Avg Trade':>9}")
    print("-" * 65)
    for thr in [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60]:
        sub = [t for t in valid if t["h"] < thr]
        if not sub: continue
        pnls = np.array([t["pnl"] for t in sub])
        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        wr = 100 * len(wins) / len(pnls)
        pf = wins.sum() / abs(losses.sum()) if len(losses) else 0
        print(f"    H < {thr:.2f} | {len(pnls):>6} | {wr:>5.1f} | {pf:>5.2f} | ${pnls.sum():>+10,.0f} | ${pnls.mean():>+8,.0f}")

    # Also show H >= threshold
    print()
    for thr in [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60]:
        sub = [t for t in valid if t["h"] >= thr]
        if not sub: continue
        pnls = np.array([t["pnl"] for t in sub])
        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        wr = 100 * len(wins) / len(pnls)
        pf = wins.sum() / abs(losses.sum()) if len(losses) else 0
        print(f"   H >= {thr:.2f} | {len(pnls):>6} | {wr:>5.1f} | {pf:>5.2f} | ${pnls.sum():>+10,.0f} | ${pnls.mean():>+8,.0f}")


if __name__ == "__main__":
    main()

"""
Experiment: Flip entry direction when 30-day rolling H > threshold.
Hypothesis: in strongly trending regimes (H>0.60), the z_vol spike arrives
AFTER the move is done, so reversing the signal catches the reversion.

Tests:
1. Baseline (no filter)
2. Normal entries H < 0.56 only
3. Normal H < 0.56 + FLIPPED entries H >= 0.60
4. FLIPPED entries H >= 0.60 only
5. FLIPPED entries H >= 0.58 only
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time
from collections import defaultdict

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


def run_backtest(df, daily_h, mode="normal", h_normal_max=0.56, h_flip_min=0.60):
    """
    mode:
      "baseline" - all trades, no filter
      "normal_only" - only trade when H < h_normal_max
      "flip_only" - only trade when H >= h_flip_min, but FLIP direction
      "combined" - normal when H < h_normal_max, flip when H >= h_flip_min, skip middle
    """
    n = len(df)
    trades = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0

    for i in range(n):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE
        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE
            trades.append({"pnl": round(pnl, 2), "date": str(td)})
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
                trades.append({"pnl": round(pnl, 2), "date": str(td)})
                pos = 0; continue
        if pos == 0 and dt < MT:
            cur_h = daily_h.get(td, np.nan)
            if np.isnan(cur_h):
                continue

            # Determine if we should trade and whether to flip
            flip = False
            if mode == "baseline":
                pass  # always trade normal
            elif mode == "normal_only":
                if cur_h >= h_normal_max:
                    continue
            elif mode == "flip_only":
                if cur_h < h_flip_min:
                    continue
                flip = True
            elif mode == "combined":
                if cur_h < h_normal_max:
                    pass  # normal
                elif cur_h >= h_flip_min:
                    flip = True
                else:
                    continue  # skip middle zone

            # Generate signal
            signal = None
            if z > HIGH_Z and cl > ema:
                signal = "long"
            elif z > HIGH_Z and cl < ema:
                signal = "short"
            elif z < LOW_Z and cl < ema:
                signal = "long"
            elif z < LOW_Z and cl > ema:
                signal = "short"

            if signal is None:
                continue

            # Flip if needed
            if flip:
                signal = "short" if signal == "long" else "long"

            if signal == "long":
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
            else:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1

    return trades


def calc_stats(trades, is_cutoff):
    if not trades:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    cum = pnls.cumsum(); dd = (cum - np.maximum.accumulate(cum)).min()
    pf = wins.sum() / abs(losses.sum()) if len(losses) else 0
    wr = 100 * len(wins) / len(pnls)
    is_t = [t for t in trades if t["date"] < is_cutoff]
    oos_t = [t for t in trades if t["date"] >= is_cutoff]
    is_p = np.array([t["pnl"] for t in is_t]) if is_t else np.array([0.0])
    oos_p = np.array([t["pnl"] for t in oos_t]) if oos_t else np.array([0.0])
    is_pf = is_p[is_p > 0].sum() / abs(is_p[is_p < 0].sum()) if (is_p < 0).any() else 0
    oos_pf = oos_p[oos_p > 0].sum() / abs(oos_p[oos_p < 0].sum()) if (oos_p < 0).any() else 0
    return {"n": len(pnls), "pf": pf, "wr": wr, "pnl": pnls.sum(), "dd": dd,
            "is_pf": is_pf, "oos_pf": oos_pf, "avg": pnls.mean()}


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

    # IS/OOS
    trading_days = sorted(set(idx.date() for idx in df.index))
    split_idx = int(len(trading_days) * 0.60)
    is_cutoff = str(trading_days[split_idx])

    # Run experiments
    configs = [
        ("Baseline (no filter)", "baseline", 99, 0),
        ("Normal H < 0.56 only", "normal_only", 0.56, 0),
        ("FLIP H >= 0.60 only", "flip_only", 0, 0.60),
        ("FLIP H >= 0.58 only", "flip_only", 0, 0.58),
        ("FLIP H >= 0.56 only", "flip_only", 0, 0.56),
        ("Normal<0.56 + Flip>=0.60", "combined", 0.56, 0.60),
        ("Normal<0.56 + Flip>=0.58", "combined", 0.56, 0.58),
        ("Normal<0.55 + Flip>=0.60", "combined", 0.55, 0.60),
    ]

    print(f"\n{'Config':>28} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'IS PF':>6} | {'OOS PF':>6} | {'Total PnL':>11} | {'MaxDD':>10} | {'Avg':>8}")
    print("-" * 115)

    for label, mode, h_max, h_flip in configs:
        trades = run_backtest(df, daily_h, mode=mode, h_normal_max=h_max, h_flip_min=h_flip)
        s = calc_stats(trades, is_cutoff)
        if s:
            print(f"{label:>28} | {s['n']:>6} | {s['wr']:>5.1f} | {s['pf']:>5.2f} | "
                  f"{s['is_pf']:>6.2f} | {s['oos_pf']:>6.2f} | ${s['pnl']:>+10,.0f} | "
                  f"${s['dd']:>+9,.0f} | ${s['avg']:>+7,.0f}")
        else:
            print(f"{label:>28} | no trades")

    # Detailed yearly for the flip-only H>=0.60
    print("\n\nYearly breakdown: FLIP H >= 0.60 only")
    print(f"{'Year':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>11}")
    print("-" * 50)
    trades_flip = run_backtest(df, daily_h, mode="flip_only", h_normal_max=0, h_flip_min=0.60)
    yearly = defaultdict(list)
    for t in trades_flip:
        yearly[t["date"][:4]].append(t)
    for yr in sorted(yearly.keys()):
        yr_p = np.array([t["pnl"] for t in yearly[yr]])
        yr_pf = yr_p[yr_p > 0].sum() / abs(yr_p[yr_p < 0].sum()) if (yr_p < 0).any() else 0
        yr_wr = 100 * sum(1 for p in yr_p if p > 0) / len(yr_p)
        print(f"{yr:>6} | {len(yr_p):>6} | {yr_wr:>5.1f} | {yr_pf:>5.2f} | ${yr_p.sum():>+10,.0f}")

    # Detailed yearly for combined
    print("\n\nYearly breakdown: Normal<0.56 + Flip>=0.60")
    print(f"{'Year':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>11}")
    print("-" * 50)
    trades_comb = run_backtest(df, daily_h, mode="combined", h_normal_max=0.56, h_flip_min=0.60)
    yearly2 = defaultdict(list)
    for t in trades_comb:
        yearly2[t["date"][:4]].append(t)
    for yr in sorted(yearly2.keys()):
        yr_p = np.array([t["pnl"] for t in yearly2[yr]])
        yr_pf = yr_p[yr_p > 0].sum() / abs(yr_p[yr_p < 0].sum()) if (yr_p < 0).any() else 0
        yr_wr = 100 * sum(1 for p in yr_p if p > 0) / len(yr_p)
        print(f"{yr:>6} | {len(yr_p):>6} | {yr_wr:>5.1f} | {yr_pf:>5.2f} | ${yr_p.sum():>+10,.0f}")


if __name__ == "__main__":
    main()

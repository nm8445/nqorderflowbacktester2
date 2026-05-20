"""
5-year backtest — sweep H values using NQGRAY base config.
Tests H = 0.10, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
NORM_LEN = 300; Z_LOOKBACK = 100; HIGH_Z = 1.5; EMA_LEN = 40; ATR_LEN = 14
ATR_SL = 2.0; ATR_TP = 1.2; ET = "America/New_York"; POINT_VALUE = 20.0
SS = "06:45"; SE = "15:45"; MT = 5
MART_STREAK = 1; MART_MULT = 1.5; MART_MAX_DOUBLES = 4


def build_combined_bars():
    print("Loading data...", flush=True)
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
    print(f"Combined: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})", flush=True)
    return df


def compute_and_backtest(df, H):
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
    v_m = v_s.rolling(Z_LOOKBACK).mean(); v_d = v_s.rolling(Z_LOOKBACK).std(ddof=0)
    z_vol = np.nan_to_num(np.where(v_d > 0, (v_s - v_m) / v_d, 0.0), nan=0.0)
    ema = closes.ewm(span=EMA_LEN, adjust=False).mean().values
    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(), (lows - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n); atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i + 1].mean()
        else: atr_vals[i] = (atr_vals[i - 1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN

    # Backtest
    closes_v = closes.values; highs_v = highs.values; lows_v = lows.values
    idx_list = df.index
    trades = []; trade_years = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
    qty = 1; loss_streak = 0

    for i in range(n):
        idx = idx_list[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE
        cl = closes_v[i]; hi = highs_v[i]; lo = lows_v[i]
        if pos != 0 and not ins and hm >= SE:
            pnl = ((cl - ep) if d == "long" else (ep - cl)) * POINT_VALUE * qty
            trades.append(pnl); trade_years.append(td.year)
            if pnl > 0: loss_streak = 0
            else: loss_streak += 1
            pos = 0
        if not ins: continue
        if td != cd: cd = td; dt = 0; loss_streak = 0
        if loss_streak >= MART_STREAK:
            steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
            qty = max(1, round(MART_MULT ** steps))
        else: qty = 1
        atr_v = atr_vals[i]
        if atr_v <= 0: continue
        z = z_vol[i]; e = ema[i]
        if pos != 0:
            xp = None
            if d == "long":
                if lo <= sl: xp = sl
                elif hi >= tp: xp = tp
            else:
                if hi >= sl: xp = sl
                elif lo <= tp: xp = tp
            if xp:
                pnl = ((xp - ep) if d == "long" else (ep - xp)) * POINT_VALUE * qty
                trades.append(pnl); trade_years.append(td.year)
                if pnl > 0: loss_streak = 0
                else: loss_streak += 1
                if loss_streak >= MART_STREAK:
                    steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                    qty = max(1, round(MART_MULT ** steps))
                else: qty = 1
                pos = 0; continue
        if pos == 0 and dt < MT:
            if z > HIGH_Z and cl > e:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
            elif z > HIGH_Z and cl < e:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1

    return np.array(trades), np.array(trade_years)


def main():
    df = build_combined_bars()

    h_values = [0.10, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]

    print(f"\nBase config: KERNEL={KERNEL_LEN}, NORM={NORM_LEN}, ZLOOK={Z_LOOKBACK}, HIGH_Z={HIGH_Z}")
    print(f"EMA={EMA_LEN}, SL={ATR_SL}, TP={ATR_TP}, Session {SS}-{SE} ET, streak={MART_STREAK}")
    print()

    # Header
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    print(f"{'H':>5} {'Trades':>7} {'PF':>6} {'WR':>6} {'PnL':>10} {'DD':>10}  |", end="")
    for y in years:
        print(f" {y} PF", end="")
    print()
    print("-" * 105)

    for H in h_values:
        pnls, tyears = compute_and_backtest(df, H)
        if len(pnls) == 0:
            print(f"{H:>5.2f}   No trades")
            continue
        w = pnls[pnls > 0]; l = pnls[pnls < 0]
        pf = w.sum() / abs(l.sum()) if len(l) else 99
        wr = 100 * len(w) / len(pnls)
        cum = pnls.cumsum(); dd = (cum - np.maximum.accumulate(cum)).min()

        line = f"{H:>5.2f} {len(pnls):>7} {pf:>6.2f} {wr:>5.1f}% ${pnls.sum():>+9,.0f} ${dd:>9,.0f}  |"

        for y in years:
            mask = tyears == y
            if mask.sum() == 0:
                line += "      "
                continue
            yp = pnls[mask]
            yw = yp[yp > 0]; yl = yp[yp < 0]
            ypf = yw.sum() / abs(yl.sum()) if len(yl) else 99
            line += f" {ypf:>4.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()

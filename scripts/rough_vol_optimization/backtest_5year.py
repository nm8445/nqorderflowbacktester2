"""
5-year backtest — combine MarketTick parquet (Dec 2020 - Nov 2025) with pkl data (Dec 2025 - Apr 2026).
Locked config: NORM=300, ZLOOK=200, HIGH_Z=1.0, SL=2, TP=2, trend-only, streak=1.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

H = 0.40; KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
NORM_LEN = 300; Z_LOOKBACK = 100; HIGH_Z = 1.5; EMA_LEN = 40; ATR_LEN = 14
ATR_SL = 2.0; ATR_TP = 1.2; ET = "America/New_York"; POINT_VALUE = 20.0
SS = "06:45"; SE = "15:45"; MT = 5
MART_STREAK = 1; MART_MULT = 1.5; MART_MAX_DOUBLES = 4


def build_combined_bars():
    # Part 1: MarketTick parquet (Dec 2020 - Nov 2025)
    print("Loading MarketTick parquet...", flush=True)
    df_mt = pd.read_parquet(MARKETTICK_PARQUET)
    print(f"  MarketTick bars: {len(df_mt):,}  ({df_mt.index[0]} to {df_mt.index[-1]})", flush=True)

    # Part 2: pkl data (Dec 2025 - Apr 2026)
    print("Loading pkl data...", flush=True)
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
    print(f"  Pkl bars: {len(df_pkl):,}  ({df_pkl.index[0]} to {df_pkl.index[-1]})", flush=True)

    # Combine: use MarketTick up to its end, then pkl after that
    cutoff = df_mt.index[-1]
    df_pkl_new = df_pkl[df_pkl.index > cutoff]
    print(f"  Pkl bars after cutoff: {len(df_pkl_new):,}", flush=True)

    # MarketTick has tick_count column, drop it to align
    mt_cols = ["open", "high", "low", "close", "volume"]
    pkl_cols = ["open", "high", "low", "close"]
    # Align columns
    if "volume" in df_mt.columns:
        df_mt_use = df_mt[["open", "high", "low", "close"]].copy()
    else:
        df_mt_use = df_mt[["open", "high", "low", "close"]].copy()

    df = pd.concat([df_mt_use, df_pkl_new[["open", "high", "low", "close"]]]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    print(f"  Combined bars: {len(df):,}  ({df.index[0]} to {df.index[-1]})", flush=True)
    return df


def main():
    df = build_combined_bars()

    # Convert to ET
    df.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df.index
    ])

    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)
    print(f"\nComputing signals on {n:,} bars...", flush=True)

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
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i + 1].mean()
        else: atr_vals[i] = (atr_vals[i - 1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    print("Running backtest...", flush=True)

    trades = []
    trade_dates = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
    qty = 1; loss_streak = 0

    for i in range(len(df)):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE
        if pos != 0 and not ins and hm >= SE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE * qty
            trades.append(pnl); trade_dates.append(td)
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
                trades.append(pnl); trade_dates.append(td)
                if pnl > 0: loss_streak = 0
                else: loss_streak += 1
                if loss_streak >= MART_STREAK:
                    steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                    qty = max(1, round(MART_MULT ** steps))
                else: qty = 1
                pos = 0; continue
        if pos == 0 and dt < MT:
            if z > HIGH_Z and cl > ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
            elif z > HIGH_Z and cl < ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1

    pnls = np.array(trades)
    w = pnls[pnls > 0]; l = pnls[pnls < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else 99
    wr = 100 * len(w) / len(pnls)
    cum = pnls.cumsum(); dd = (cum - np.maximum.accumulate(cum)).min()

    print(f"\n{'='*80}")
    print(f"5-YEAR BACKTEST RESULTS")
    print(f"Config: NORM={NORM_LEN}, ZLOOK={Z_LOOKBACK}, HIGH_Z={HIGH_Z}, SL={ATR_SL}, TP={ATR_TP}")
    print(f"Trend-only, Martingale streak={MART_STREAK}, daily reset")
    print(f"{'='*80}")
    print(f"Trades: {len(pnls):,}  PF: {pf:.2f}  WR: {wr:.1f}%")
    print(f"PnL: ${pnls.sum():+,.0f}  DD: ${dd:,.0f}")
    print(f"Avg Win: ${w.mean():+,.0f}  Avg Loss: ${l.mean():+,.0f}")
    print(f"Win/Loss ratio: {w.mean()/abs(l.mean()):.2f}")

    # Yearly breakdown
    print(f"\n{'Year':<6} {'Trades':>7} {'PF':>6} {'WR':>6} {'PnL':>12} {'DD':>10}")
    print("-" * 50)
    dates_arr = np.array(trade_dates)
    for year in sorted(set(d.year for d in trade_dates)):
        mask = np.array([d.year == year for d in trade_dates])
        yp = pnls[mask]
        yw = yp[yp > 0]; yl = yp[yp < 0]
        ypf = yw.sum() / abs(yl.sum()) if len(yl) else 99
        ywr = 100 * len(yw) / len(yp) if len(yp) else 0
        ycum = yp.cumsum(); ydd = (ycum - np.maximum.accumulate(ycum)).min()
        print(f"{year:<6} {len(yp):>7} {ypf:>6.2f} {ywr:>5.1f}% ${yp.sum():>+10,.0f} ${ydd:>9,.0f}")


if __name__ == "__main__":
    main()

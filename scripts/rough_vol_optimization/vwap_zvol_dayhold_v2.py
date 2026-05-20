"""
VWAP + z_vol day-hold strategy — v2: VWAP cross model.

Entry model A (same as v1): z_vol > 1.5 AND close > VWAP (long) or close < VWAP (short)
Entry model B (VWAP cross):  z_vol > 1.5 AND close crosses above VWAP this bar (prev close <= VWAP)
                             z_vol > 1.5 AND close crosses below VWAP this bar (prev close >= VWAP)

Stop:  ATR multiple below VWAP (long) or above VWAP (short)
Exit:  Force close at 15:45 ET (no TP), max 1 trade/day
Entry window: 09:30 - 15:30 ET (RTH only, avoids late entries)
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
VWAP_CACHE = Path("D:/trading_pythonbacktest_data/vwap_cache_5yr")

H = 0.40; KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
NORM_LEN = 250; Z_LOOKBACK = 100; HIGH_Z = 1.5; ATR_LEN = 14
ET = "America/New_York"; POINT_VALUE = 20.0

FORCE_CLOSE = "15:45"
SESSION_START = "09:30"
SESSION_END = "15:30"
MAX_TRADES_DAY = 1


def build_combined_bars():
    print("Loading 15-min bars...", flush=True)
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


def load_vwap_for_date(date_obj):
    """Load VWAP cache, return dict of {UTC 15-min bucket -> vwap}."""
    cache_file = VWAP_CACHE / f"{date_obj}.pkl"
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as f:
        df = pickle.load(f)
    df["group"] = df.index.floor("15min")
    vwap_15 = df.groupby("group")["vwap"].last()
    return vwap_15.to_dict()


def get_vwap(vwap_dict, idx):
    """Get VWAP value for a bar timestamp (ET) from UTC-keyed dict."""
    if vwap_dict is None:
        return None
    bar_utc = idx.tz_convert("UTC").floor("15min")
    val = vwap_dict.get(bar_utc)
    if val is None:
        prev = bar_utc - pd.Timedelta(minutes=15)
        val = vwap_dict.get(prev)
    return val


def run_backtest(df, vwap_by_date, atr_sl, cross_mode=False):
    """
    cross_mode=False: enter when z>1.5 AND close above/below VWAP
    cross_mode=True:  enter when z>1.5 AND close CROSSES above/below VWAP (prev bar was on other side)
    """
    trades = []
    pos = 0; ep = 0; sl = 0; d = ""; cd = None; dt = 0
    prev_close = None; prev_vwap = None
    n = len(df)

    for i in range(n):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M")

        # Force close
        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE
            trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": d,
                           "time": idx.strftime("%Y-%m-%d %H:%M"), "reason": "CLOSE"})
            pos = 0

        # Get current VWAP
        vwap_dict = vwap_by_date.get(td)
        cur_vwap = get_vwap(vwap_dict, idx)

        ins = SESSION_START <= hm < SESSION_END

        if td != cd:
            cd = td; dt = 0; prev_close = None; prev_vwap = None

        if not ins:
            prev_close = row["close"]
            prev_vwap = cur_vwap
            continue

        atr_v = row["atr"]
        if atr_v <= 0:
            prev_close = row["close"]
            prev_vwap = cur_vwap
            continue

        # Check stop
        if pos != 0:
            if d == "long" and row["low"] <= sl:
                pnl = (sl - ep) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": d,
                               "time": idx.strftime("%Y-%m-%d %H:%M"), "reason": "SL"})
                pos = 0
            elif d == "short" and row["high"] >= sl:
                pnl = (ep - sl) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": d,
                               "time": idx.strftime("%Y-%m-%d %H:%M"), "reason": "SL"})
                pos = 0

        # Entry
        if pos == 0 and dt < MAX_TRADES_DAY and hm < FORCE_CLOSE:
            z = row["z_vol"]; cl = row["close"]
            if z > HIGH_Z and cur_vwap is not None:
                if cross_mode:
                    # Need previous bar context
                    if prev_close is not None and prev_vwap is not None:
                        was_below = prev_close <= prev_vwap
                        was_above = prev_close >= prev_vwap
                        now_above = cl > cur_vwap
                        now_below = cl < cur_vwap

                        if was_below and now_above:
                            pos = 1; d = "long"; ep = cl
                            sl = cur_vwap - atr_sl * atr_v; dt += 1
                        elif was_above and now_below:
                            pos = -1; d = "short"; ep = cl
                            sl = cur_vwap + atr_sl * atr_v; dt += 1
                else:
                    # Simple: close above/below VWAP
                    if cl > cur_vwap:
                        pos = 1; d = "long"; ep = cl
                        sl = cur_vwap - atr_sl * atr_v; dt += 1
                    elif cl < cur_vwap:
                        pos = -1; d = "short"; ep = cl
                        sl = cur_vwap + atr_sl * atr_v; dt += 1

        prev_close = row["close"]
        prev_vwap = cur_vwap

    return trades


def print_stats(trades, label, is_cutoff):
    if not trades:
        print(f"  {label}: no trades")
        return

    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) else 0
    wr = 100 * len(wins) / len(pnls)
    cum = pnls.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()

    is_t = [t for t in trades if t["date"] < is_cutoff]
    oos_t = [t for t in trades if t["date"] >= is_cutoff]
    is_pnls = np.array([t["pnl"] for t in is_t]) if is_t else np.array([0.0])
    oos_pnls = np.array([t["pnl"] for t in oos_t]) if oos_t else np.array([0.0])
    is_pf = is_pnls[is_pnls > 0].sum() / abs(is_pnls[is_pnls < 0].sum()) if (is_pnls < 0).any() else 0
    oos_pf = oos_pnls[oos_pnls > 0].sum() / abs(oos_pnls[oos_pnls < 0].sum()) if (oos_pnls < 0).any() else 0

    longs = sum(1 for t in trades if t["dir"] == "long")
    shorts = sum(1 for t in trades if t["dir"] == "short")

    return {"n": len(trades), "pf": pf, "wr": wr, "pnl": pnls.sum(), "dd": dd,
            "is_pf": is_pf, "oos_pf": oos_pf, "longs": longs, "shorts": shorts}


def main():
    df = build_combined_bars()
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)

    print("Computing signals...", flush=True)
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

    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(),
                    (lows - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n); atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i+1].mean()
        else: atr_vals[i] = (atr_vals[i-1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    print("Loading VWAP cache...", flush=True)
    all_dates = sorted(set(idx.date() for idx in df.index))
    vwap_by_date = {}
    for d in all_dates:
        v = load_vwap_for_date(d)
        if v is not None:
            vwap_by_date[d] = v
    print(f"  Loaded {len(vwap_by_date)}/{len(all_dates)} dates", flush=True)

    trading_days = sorted(set(idx.date() for idx in df.index))
    split_idx = int(len(trading_days) * 0.60)
    is_cutoff = str(trading_days[split_idx])

    # Model A: close above/below VWAP (RTH, close 15:45)
    print("\n" + "=" * 130)
    print("MODEL A: z>1.5 + close above/below VWAP (09:30-15:30, close 15:45, 1 trade/day)")
    print("=" * 130)
    print(f"{'SL_ATR':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'IS PF':>6} | {'OOS PF':>6} | {'Avg PF':>6} | {'Total PnL':>11} | {'MaxDD':>10} | {'Longs':>6} | {'Shorts':>6}")
    print("-" * 110)

    for atr_sl in [1.0, 1.5, 2.0, 2.5, 3.0]:
        trades = run_backtest(df, vwap_by_date, atr_sl, cross_mode=False)
        s = print_stats(trades, f"SL={atr_sl}", is_cutoff)
        if s:
            print(f"{atr_sl:>6.1f} | {s['n']:>6} | {s['wr']:>5.1f} | {s['pf']:>5.2f} | {s['is_pf']:>6.2f} | {s['oos_pf']:>6.2f} | {(s['is_pf']+s['oos_pf'])/2:>6.2f} | ${s['pnl']:>+10,.0f} | ${s['dd']:>+9,.0f} | {s['longs']:>6} | {s['shorts']:>6}")

    # Model B: VWAP cross + z_vol
    print("\n" + "=" * 130)
    print("MODEL B: z>1.5 + close CROSSES above/below VWAP (prev bar other side) (09:30-15:30, close 15:45, 1 trade/day)")
    print("=" * 130)
    print(f"{'SL_ATR':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'IS PF':>6} | {'OOS PF':>6} | {'Avg PF':>6} | {'Total PnL':>11} | {'MaxDD':>10} | {'Longs':>6} | {'Shorts':>6}")
    print("-" * 110)

    for atr_sl in [1.0, 1.5, 2.0, 2.5, 3.0]:
        trades = run_backtest(df, vwap_by_date, atr_sl, cross_mode=True)
        s = print_stats(trades, f"SL={atr_sl}", is_cutoff)
        if s:
            print(f"{atr_sl:>6.1f} | {s['n']:>6} | {s['wr']:>5.1f} | {s['pf']:>5.2f} | {s['is_pf']:>6.2f} | {s['oos_pf']:>6.2f} | {(s['is_pf']+s['oos_pf'])/2:>6.2f} | ${s['pnl']:>+10,.0f} | ${s['dd']:>+9,.0f} | {s['longs']:>6} | {s['shorts']:>6}")

    # Best model A yearly
    print("\n\nYearly breakdown — Model A, SL=1.5x:")
    trades = run_backtest(df, vwap_by_date, 1.5, cross_mode=False)
    yearly = defaultdict(list)
    for t in trades:
        yearly[t["date"][:4]].append(t)
    print(f"{'Year':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>11}")
    print("-" * 50)
    for yr in sorted(yearly.keys()):
        yr_t = yearly[yr]
        yr_p = np.array([t["pnl"] for t in yr_t])
        yr_pf = yr_p[yr_p>0].sum() / abs(yr_p[yr_p<0].sum()) if (yr_p<0).any() else 0
        print(f"{yr:>6} | {len(yr_t):>6} | {100*sum(1 for p in yr_p if p>0)/len(yr_p):>5.1f} | {yr_pf:>5.2f} | ${yr_p.sum():>+10,.0f}")

    # Best model B yearly
    print("\nYearly breakdown — Model B (cross), SL=1.5x:")
    trades = run_backtest(df, vwap_by_date, 1.5, cross_mode=True)
    yearly = defaultdict(list)
    for t in trades:
        yearly[t["date"][:4]].append(t)
    print(f"{'Year':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>11}")
    print("-" * 50)
    for yr in sorted(yearly.keys()):
        yr_t = yearly[yr]
        yr_p = np.array([t["pnl"] for t in yr_t])
        yr_pf = yr_p[yr_p>0].sum() / abs(yr_p[yr_p<0].sum()) if (yr_p<0).any() else 0
        print(f"{yr:>6} | {len(yr_t):>6} | {100*sum(1 for p in yr_p if p>0)/len(yr_p):>5.1f} | {yr_pf:>5.2f} | ${yr_p.sum():>+10,.0f}")


if __name__ == "__main__":
    main()

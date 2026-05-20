"""
Test rolling Hurst exponent as a regime filter for rough vol strategy.
Compute rolling H daily (R/S on trailing N days of 15-min returns),
suppress signals when H > threshold.

Grid: window_days x H_threshold
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

H_PARAM = 0.40; KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
NORM_LEN = 250; Z_LOOKBACK = 100; HIGH_Z = 1.5; LOW_Z = -1.0; EMA_LEN = 50; ATR_LEN = 14
ATR_SL = 1.5; ATR_TP = 2.5; ET = "America/New_York"; POINT_VALUE = 20.0
SS = "09:30"; SE = "15:30"; FORCE_CLOSE = "15:27"; MT = 5


def build_combined_bars():
    df_mt = pd.read_parquet(MARKETTICK_PARQUET)
    frames = []
    for f in sorted(TIMEBARS_DIR.glob("timebars_5min_202*.pkl")):
        with open(f, "rb") as fh:
            bars = pickle.load(fh)
        if not bars:
            continue
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
    """R/S Hurst on a 1D array of returns."""
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
    log_ns = []
    log_rs = []
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
    coeffs = np.polyfit(log_ns, log_rs, 1)
    return coeffs[0]


def compute_daily_hurst(df, window_days):
    """Compute H(R/S) once per trading day using trailing window_days of session 15-min returns."""
    session_mask = (df.index.strftime("%H:%M") >= SS) & (df.index.strftime("%H:%M") < SE)
    session_df = df[session_mask].copy()
    log_ret = np.log(session_df["close"] / session_df["close"].shift(1)).dropna()

    # Group returns by date
    dates = sorted(set(idx.date() for idx in log_ret.index))
    ret_by_date = {}
    for d in dates:
        mask = log_ret.index.date == d
        ret_by_date[d] = log_ret[mask].values

    # Compute rolling H per day
    daily_h = {}
    for i, d in enumerate(dates):
        # Gather trailing window_days of returns
        start = max(0, i - window_days + 1)
        window_dates = dates[start:i+1]
        all_rets = np.concatenate([ret_by_date[wd] for wd in window_dates])
        if len(all_rets) >= 100:
            daily_h[d] = hurst_rs(all_rets)
        else:
            daily_h[d] = np.nan

    return daily_h


def run_backtest(df, daily_h, h_max, use_martingale=False):
    """Run backtest, suppressing entries when daily H > h_max."""
    trades = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
    qty = 1; loss_streak = 0; suppressed = 0
    for i in range(len(df)):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE
        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE * qty
            trades.append({"pnl": round(pnl, 2), "date": str(td)})
            if pnl > 0: loss_streak = 0
            else: loss_streak += 1
            pos = 0; continue
        if not ins:
            continue
        if td != cd:
            cd = td; dt = 0; loss_streak = 0
        if use_martingale and loss_streak >= 1:
            steps = min(loss_streak, 4)
            qty = max(1, round(3.0 ** steps))
        else:
            qty = 1
        atr_v = row["atr"]
        if atr_v <= 0:
            continue
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
                trades.append({"pnl": round(pnl, 2), "date": str(td)})
                if pnl > 0: loss_streak = 0
                else: loss_streak += 1
                if use_martingale and loss_streak >= 1:
                    steps = min(loss_streak, 4)
                    qty = max(1, round(3.0 ** steps))
                else:
                    qty = 1
                pos = 0; continue
        if pos == 0 and dt < MT:
            # Regime filter: check daily H
            cur_h = daily_h.get(td, np.nan)
            if not np.isnan(cur_h) and cur_h > h_max:
                suppressed += 1
                continue  # Skip entry — regime too persistent

            if z > HIGH_Z and cl > ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
            elif z > HIGH_Z and cl < ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1
            elif z < LOW_Z and cl < ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
            elif z < LOW_Z and cl > ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1
    return trades, suppressed


def main():
    print("Loading bars...", flush=True)
    df = build_combined_bars()
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)

    # Compute signals
    ret = np.log(closes / closes.shift(1)); ret.iloc[0] = 0.0
    mean_ret = ret.rolling(NORM_LEN).mean()
    std_ret = ret.rolling(NORM_LEN).std(ddof=0)
    shock = np.nan_to_num(np.where(std_ret > 0, (ret - mean_ret) / std_ret, 0.0), nan=0.0)
    ks = np.arange(1, KERNEL_LEN + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = ks ** (H_PARAM - 0.5) - (ks - 1) ** (H_PARAM - 0.5)
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

    # IS/OOS split
    trading_days = sorted(set(idx.date() for idx in df.index))
    split_idx = int(len(trading_days) * 0.60)
    is_cutoff = str(trading_days[split_idx])

    # Pre-compute daily Hurst for each window size
    WINDOWS = [20, 40, 60, 90, 120]
    H_THRESHOLDS = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 99.0]  # 99.0 = no filter (baseline)

    print(f"\nPre-computing daily Hurst for {len(WINDOWS)} window sizes...", flush=True)
    t0 = time.time()
    hurst_cache = {}
    for w in WINDOWS:
        print(f"  Window={w} days...", end=" ", flush=True)
        tw = time.time()
        hurst_cache[w] = compute_daily_hurst(df, w)
        print(f"{time.time()-tw:.0f}s", flush=True)
    print(f"Hurst computation: {time.time()-t0:.0f}s total\n", flush=True)

    # Grid search
    print("=" * 130)
    print(f"{'Window':>7} | {'H_max':>6} | {'Trades':>6} | {'Suppr':>6} | {'IS PnL':>10} | {'IS PF':>6} | {'IS WR':>6} | {'OOS PnL':>10} | {'OOS PF':>6} | {'OOS WR':>6} | {'Total PnL':>10} | {'Avg PF':>6} | {'MaxDD':>10}")
    print("-" * 130)

    results = []
    for w in WINDOWS:
        for h_max in H_THRESHOLDS:
            trades, suppressed = run_backtest(df, hurst_cache[w], h_max, use_martingale=False)
            if not trades:
                continue

            pnls = np.array([t["pnl"] for t in trades])
            cum = pnls.cumsum()
            dd = (cum - np.maximum.accumulate(cum)).min()

            is_t = [t for t in trades if t["date"] < is_cutoff]
            oos_t = [t for t in trades if t["date"] >= is_cutoff]
            is_pnls = np.array([t["pnl"] for t in is_t]) if is_t else np.array([0.0])
            oos_pnls = np.array([t["pnl"] for t in oos_t]) if oos_t else np.array([0.0])

            is_pf = is_pnls[is_pnls > 0].sum() / abs(is_pnls[is_pnls < 0].sum()) if (is_pnls < 0).any() else 0
            oos_pf = oos_pnls[oos_pnls > 0].sum() / abs(oos_pnls[oos_pnls < 0].sum()) if (oos_pnls < 0).any() else 0
            is_wr = 100 * (is_pnls > 0).sum() / len(is_pnls) if len(is_pnls) else 0
            oos_wr = 100 * (oos_pnls > 0).sum() / len(oos_pnls) if len(oos_pnls) else 0
            avg_pf = (is_pf + oos_pf) / 2

            label_hmax = "none" if h_max > 90 else f"{h_max:.2f}"

            print(f"{w:>7} | {label_hmax:>6} | {len(trades):>6} | {suppressed:>6} | ${is_pnls.sum():>+9,.0f} | {is_pf:>6.2f} | {is_wr:>5.1f}% | ${oos_pnls.sum():>+9,.0f} | {oos_pf:>6.2f} | {oos_wr:>5.1f}% | ${pnls.sum():>+9,.0f} | {avg_pf:>6.2f} | ${dd:>+9,.0f}")

            results.append({
                "window": w, "h_max": h_max, "trades": len(trades), "suppressed": suppressed,
                "is_pnl": is_pnls.sum(), "is_pf": is_pf, "oos_pnl": oos_pnls.sum(), "oos_pf": oos_pf,
                "total_pnl": pnls.sum(), "avg_pf": avg_pf, "max_dd": dd,
            })

        print("-" * 130)

    print("=" * 130)

    # Best by avg PF (excluding baseline)
    filtered = [r for r in results if r["h_max"] < 90 and r["trades"] >= 500]
    if filtered:
        best = max(filtered, key=lambda r: r["avg_pf"])
        print(f"\nBest avg PF (>=500 trades): Window={best['window']}, H_max={best['h_max']:.2f}")
        print(f"  Trades: {best['trades']}  Suppressed: {best['suppressed']}")
        print(f"  IS PF: {best['is_pf']:.2f}  OOS PF: {best['oos_pf']:.2f}  Avg: {best['avg_pf']:.2f}")
        print(f"  Total PnL: ${best['total_pnl']:+,.0f}  MaxDD: ${best['max_dd']:,.0f}")

    # Best by OOS PF
    if filtered:
        best_oos = max(filtered, key=lambda r: r["oos_pf"])
        print(f"\nBest OOS PF (>=500 trades): Window={best_oos['window']}, H_max={best_oos['h_max']:.2f}")
        print(f"  Trades: {best_oos['trades']}  Suppressed: {best_oos['suppressed']}")
        print(f"  IS PF: {best_oos['is_pf']:.2f}  OOS PF: {best_oos['oos_pf']:.2f}  Avg: {best_oos['avg_pf']:.2f}")
        print(f"  Total PnL: ${best_oos['total_pnl']:+,.0f}  MaxDD: ${best_oos['max_dd']:,.0f}")


if __name__ == "__main__":
    main()

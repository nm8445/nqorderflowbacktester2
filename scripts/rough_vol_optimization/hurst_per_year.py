"""
Compute realized Hurst exponent per year vs strategy PnL per year.
Uses R/S (rescaled range) and variance-ratio estimators on 15-min log returns.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

H=0.40; KERNEL_LEN=80; ETA=1.0; V0=0.0001
NORM_LEN=250; Z_LOOKBACK=100; HIGH_Z=1.5; LOW_Z=-1.0; EMA_LEN=50; ATR_LEN=14
ATR_SL=1.5; ATR_TP=2.5; ET="America/New_York"; POINT_VALUE=20.0
SS="09:30"; SE="15:30"; FORCE_CLOSE="15:27"; MT=5
MART_STREAK=1; MART_MULT=3.0; MART_MAX_DOUBLES=4


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


def hurst_rs(series, min_chunk=8):
    """Rescaled range (R/S) Hurst estimator."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
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


def hurst_var_ratio(series, lags=None):
    """Variance ratio Hurst estimator."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 100:
        return np.nan
    if lags is None:
        lags = [2, 4, 8, 16, 32, 64]
        lags = [lag for lag in lags if lag < n // 4]
    if len(lags) < 3:
        return np.nan
    log_lags = []
    log_vars = []
    for lag in lags:
        diffs = x[lag:] - x[:-lag]
        v = np.var(diffs)
        if v > 0:
            log_lags.append(np.log(lag))
            log_vars.append(np.log(v))
    if len(log_lags) < 3:
        return np.nan
    coeffs = np.polyfit(log_lags, log_vars, 1)
    return coeffs[0] / 2.0


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

    # Backtest — flat sizing (qty=1 always) for clean signal quality
    trades = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
    for i in range(len(df)):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE
        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE
            trades.append({"pnl": round(pnl, 2), "date": str(td)})
            pos = 0; continue
        if not ins:
            continue
        if td != cd:
            cd = td; dt = 0
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
                pnl = ((xp - ep) if d == "long" else (ep - xp)) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td)})
                pos = 0; continue
        if pos == 0 and dt < MT:
            if z > HIGH_Z and cl > ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
            elif z > HIGH_Z and cl < ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1
            elif z < LOW_Z and cl < ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr_v; tp = cl + ATR_TP * atr_v; dt += 1
            elif z < LOW_Z and cl > ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr_v; tp = cl - ATR_TP * atr_v; dt += 1

    # Yearly PnL
    yearly_pnl = {}
    yearly_trades = {}
    yearly_wins = {}
    for t in trades:
        yr = t["date"][:4]
        yearly_pnl[yr] = yearly_pnl.get(yr, 0) + t["pnl"]
        yearly_trades[yr] = yearly_trades.get(yr, 0) + 1
        if t["pnl"] > 0:
            yearly_wins[yr] = yearly_wins.get(yr, 0) + 1

    # Hurst per year on session 15-min log returns
    session_df = df[(df.index.strftime("%H:%M") >= SS) & (df.index.strftime("%H:%M") < SE)]
    log_ret = np.log(session_df["close"] / session_df["close"].shift(1)).dropna()

    years = sorted(set(idx.year for idx in log_ret.index))

    print()
    print("=" * 110)
    print(f"{'Year':>6} | {'H(R/S)':>7} | {'H(VarR)':>7} | {'Trades':>6} | {'WR%':>5} | {'PnL(flat)':>11} | {'PF':>5} | {'z>1.5 bars':>10} | {'Avg z_vol':>9}")
    print("-" * 110)

    results = []
    for yr in years:
        yr_str = str(yr)
        mask_ret = log_ret.index.year == yr
        rets_yr = log_ret[mask_ret].values

        h_rs = hurst_rs(rets_yr)
        h_vr = hurst_var_ratio(rets_yr)

        pnl = yearly_pnl.get(yr_str, 0)
        n_tr = yearly_trades.get(yr_str, 0)
        n_wins = yearly_wins.get(yr_str, 0)
        wr = 100 * n_wins / n_tr if n_tr else 0

        # PF
        yr_trades = [t for t in trades if t["date"][:4] == yr_str]
        yr_pnls = np.array([t["pnl"] for t in yr_trades])
        w = yr_pnls[yr_pnls > 0].sum()
        l = abs(yr_pnls[yr_pnls < 0].sum())
        pf = w / l if l > 0 else 0

        yr_mask = session_df.index.year == yr
        high_z_bars = (session_df.loc[yr_mask, "z_vol"] > HIGH_Z).sum()
        avg_z = session_df.loc[yr_mask, "z_vol"].mean()

        print(f"{yr:>6} | {h_rs:>7.3f} | {h_vr:>7.3f} | {n_tr:>6} | {wr:>5.1f} | ${pnl:>+10,.0f} | {pf:>5.2f} | {high_z_bars:>10} | {avg_z:>9.3f}")
        results.append((yr, h_rs, h_vr, pnl, pf, high_z_bars, avg_z))

    print("=" * 110)

    # Hurst on log-prices (trending vs mean-reverting regime)
    print()
    print("Hurst on LOG-PRICES (>0.5 = trending, <0.5 = mean-reverting, 0.5 = random walk):")
    print("=" * 75)
    print(f"{'Year':>6} | {'H(R/S)':>7} | {'H(VarR)':>7} | {'Regime':>20} | {'PnL':>11}")
    print("-" * 75)
    for yr in years:
        yr_str = str(yr)
        yr_mask = session_df.index.year == yr
        prices = session_df.loc[yr_mask, "close"].values
        h_rs = hurst_rs(np.diff(np.log(prices)))
        h_vr = hurst_var_ratio(np.log(prices))
        pnl = yearly_pnl.get(yr_str, 0)
        if h_vr < 0.45:
            interp = "MEAN-REVERTING"
        elif h_vr > 0.55:
            interp = "TRENDING"
        else:
            interp = "random walk"
        print(f"{yr:>6} | {h_rs:>7.3f} | {h_vr:>7.3f} | {interp:>20} | ${pnl:>+10,.0f}")
    print("=" * 75)

    # Correlation
    print()
    print("Correlation analysis (flat PnL vs Hurst):")
    yr_list = sorted(yearly_pnl.keys())
    flat_pnls = [yearly_pnl[y] for y in yr_list]

    h_vr_list = []
    h_rs_list = []
    for yr_str in yr_list:
        yr = int(yr_str)
        mask = log_ret.index.year == yr
        h_rs_list.append(hurst_rs(log_ret[mask].values))
        h_vr_list.append(hurst_var_ratio(log_ret[mask].values))

    if len(yr_list) > 2:
        corr_vr = np.corrcoef(h_vr_list, flat_pnls)[0, 1]
        corr_rs = np.corrcoef(h_rs_list, flat_pnls)[0, 1]
        print(f"  corr(H_VarRatio, PnL) = {corr_vr:+.3f}")
        print(f"  corr(H_RS,       PnL) = {corr_rs:+.3f}")

    # Also: avg z_vol vs PnL
    avg_z_list = []
    for yr_str in yr_list:
        yr = int(yr_str)
        yr_mask = session_df.index.year == yr
        avg_z_list.append(session_df.loc[yr_mask, "z_vol"].mean())
    if len(yr_list) > 2:
        corr_z = np.corrcoef(avg_z_list, flat_pnls)[0, 1]
        print(f"  corr(avg_z_vol,  PnL) = {corr_z:+.3f}")

    # High-z count vs PnL
    hz_list = []
    for yr_str in yr_list:
        yr = int(yr_str)
        yr_mask = session_df.index.year == yr
        hz_list.append(int((session_df.loc[yr_mask, "z_vol"] > HIGH_Z).sum()))
    if len(yr_list) > 2:
        corr_hz = np.corrcoef(hz_list, flat_pnls)[0, 1]
        print(f"  corr(z>1.5_count,PnL) = {corr_hz:+.3f}")


if __name__ == "__main__":
    main()

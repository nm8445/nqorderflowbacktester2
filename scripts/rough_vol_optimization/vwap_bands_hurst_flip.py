"""
VWAP std dev bands + z_vol + Hurst flip strategy.

Entry logic:
  - z_vol > 1.5 AND close >= VWAP +1 std -> BUY (normal H<0.56) or SELL (flip H>=0.60)
  - z_vol > 1.5 AND close <= VWAP -1 std -> SELL (normal H<0.56) or BUY (flip H>=0.60)
  - z_vol < -1.0 AND close <= VWAP -1 std -> BUY (normal) or SELL (flip) [mean revert]
  - z_vol < -1.0 AND close >= VWAP +1 std -> SELL (normal) or BUY (flip) [mean revert]

Session VWAP accumulates from 6pm ET, stops at market close.
Entry: 9:01 AM - 4:00 PM ET. 1 entry at a time. Force close 4:50 PM.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
VWAP_CACHE = Path("D:/trading_pythonbacktest_data/vwap_cache_5yr")

H = 0.40; KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
NORM_LEN = 250; Z_LOOKBACK = 100; HIGH_Z = 1.5; LOW_Z = -1.0; ATR_LEN = 14
ATR_SL = 1.5; ATR_TP = 2.5; ET = "America/New_York"; POINT_VALUE = 20.0
SS = "09:01"; SE = "16:00"; FORCE_CLOSE = "16:50"; MT = 1

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


def load_vwap_for_date(date_obj):
    """Load VWAP cache, return dict of {UTC 15-min bucket -> {vwap, std1_upper, std1_lower}}."""
    cache_file = VWAP_CACHE / f"{date_obj}.pkl"
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as f:
        df = pickle.load(f)
    # Resample to 15-min (last value per bucket)
    df["group"] = df.index.floor("15min")
    agg = df.groupby("group").agg(
        vwap=("vwap", "last"),
        std1_upper=("std1_upper", "last"),
        std1_lower=("std1_lower", "last"),
    )
    return agg.to_dict("index")


def get_vwap_bands(vwap_dict, idx):
    """Get VWAP and std bands for a bar timestamp (ET-localized)."""
    if vwap_dict is None:
        return None, None, None
    bar_utc = idx.tz_convert("UTC").floor("15min")
    row = vwap_dict.get(bar_utc)
    if row is None:
        prev = bar_utc - pd.Timedelta(minutes=15)
        row = vwap_dict.get(prev)
    if row is None:
        return None, None, None
    return row["vwap"], row["std1_upper"], row["std1_lower"]


def main():
    print("Loading bars...", flush=True)
    df = build()
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)

    # z_vol signal
    print("Computing z_vol...", flush=True)
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

    # ATR
    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(),
                    (lows - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n); atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i+1].mean()
        else: atr_vals[i] = (atr_vals[i-1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    # Load VWAP cache
    print("Loading VWAP cache...", flush=True)
    all_dates = sorted(set(idx.date() for idx in df.index))
    vwap_by_date = {}
    for d in all_dates:
        v = load_vwap_for_date(d)
        if v is not None:
            vwap_by_date[d] = v
    print(f"  {len(vwap_by_date)} dates loaded", flush=True)

    # Compute 30-day rolling Hurst
    print(f"Computing {HURST_WINDOW}-day rolling Hurst...", flush=True)
    # Use RTH bars for Hurst calc
    hurst_ss = "09:30"; hurst_se = "15:30"
    session_mask = (df.index.strftime("%H:%M") >= hurst_ss) & (df.index.strftime("%H:%M") < hurst_se)
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

    # Grid: ATR SL/TP multiples
    configs = [
        (1.5, 2.5, "SL 1.5 / TP 2.5"),
        (1.5, 3.0, "SL 1.5 / TP 3.0"),
        (2.0, 3.0, "SL 2.0 / TP 3.0"),
        (2.0, 4.0, "SL 2.0 / TP 4.0"),
        (2.5, 4.0, "SL 2.5 / TP 4.0"),
        (1.0, 2.0, "SL 1.0 / TP 2.0"),
        (1.5, 2.0, "SL 1.5 / TP 2.0"),
    ]

    print(f"\n{'Config':>18} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'IS PF':>6} | {'OOS PF':>6} | {'Total PnL':>11} | {'MaxDD':>10} | {'Avg':>8}")
    print("-" * 105)

    best_avg_pf = 0
    best_trades = None
    best_label = ""

    for atr_sl, atr_tp, label in configs:
        trades = []
        pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0

        for i in range(n):
            row = df.iloc[i]; idx = df.index[i]; td = idx.date()
            hm = idx.strftime("%H:%M")

            # Force close at 16:50
            if pos != 0 and hm >= FORCE_CLOSE:
                pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": d, "src": src})
                pos = 0; continue

            if hm >= FORCE_CLOSE:
                continue

            # Entry window check
            ins = SS <= hm <= SE

            if td != cd:
                cd = td; dt = 0

            atr_v = row["atr"]
            if atr_v <= 0: continue

            # Check stops
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
                    trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": d, "src": src})
                    pos = 0; continue

            if not ins:
                continue

            if pos == 0 and dt < MT:
                cur_h = daily_hurst.get(td, np.nan)
                if np.isnan(cur_h):
                    continue

                # Determine flip
                flip = False
                if cur_h < H_NORMAL_MAX:
                    pass
                elif cur_h >= H_FLIP_MIN:
                    flip = True
                else:
                    continue  # skip middle zone

                z = row["z_vol"]; cl = row["close"]

                # Get VWAP bands
                vwap_dict = vwap_by_date.get(td)
                vwap_val, upper1, lower1 = get_vwap_bands(vwap_dict, idx)
                if vwap_val is None:
                    continue

                # Entry logic using VWAP std bands
                signal = None
                if z > HIGH_Z and cl >= upper1:
                    signal = "long"  # momentum above upper band
                elif z > HIGH_Z and cl <= lower1:
                    signal = "short"  # momentum below lower band
                elif z < LOW_Z and cl <= lower1:
                    signal = "long"  # mean revert from lower band
                elif z < LOW_Z and cl >= upper1:
                    signal = "short"  # mean revert from upper band

                if signal is None:
                    continue

                # Flip if H >= 0.60
                if flip:
                    signal = "short" if signal == "long" else "long"
                    src = "flip"
                else:
                    src = "normal"

                if signal == "long":
                    pos = 1; d = "long"; ep = cl
                    sl = cl - atr_sl * atr_v; tp = cl + atr_tp * atr_v; dt += 1
                else:
                    pos = -1; d = "short"; ep = cl
                    sl = cl + atr_sl * atr_v; tp = cl - atr_tp * atr_v; dt += 1

        # Stats
        if not trades:
            print(f"{label:>18} | no trades")
            continue

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
        avg_pf = (is_pf + oos_pf) / 2

        print(f"{label:>18} | {len(pnls):>6} | {wr:>5.1f} | {pf:>5.2f} | {is_pf:>6.2f} | {oos_pf:>6.2f} | ${pnls.sum():>+10,.0f} | ${dd:>+9,.0f} | ${pnls.mean():>+7,.0f}")

        if avg_pf > best_avg_pf and len(trades) >= 30:
            best_avg_pf = avg_pf
            best_trades = trades
            best_label = label

    # Yearly for best
    if best_trades:
        print(f"\nBest: {best_label}")
        print(f"\n{'Year':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>11} | {'Normal':>6} | {'Flip':>6}")
        print("-" * 65)
        yearly = defaultdict(list)
        for t in best_trades:
            yearly[t["date"][:4]].append(t)
        for yr in sorted(yearly.keys()):
            yr_t = yearly[yr]
            yr_p = np.array([t["pnl"] for t in yr_t])
            yr_pf = yr_p[yr_p > 0].sum() / abs(yr_p[yr_p < 0].sum()) if (yr_p < 0).any() else 0
            yr_wr = 100 * sum(1 for p in yr_p if p > 0) / len(yr_p)
            norm_n = sum(1 for t in yr_t if t["src"] == "normal")
            flip_n = sum(1 for t in yr_t if t["src"] == "flip")
            print(f"{yr:>6} | {len(yr_t):>6} | {yr_wr:>5.1f} | {yr_pf:>5.2f} | ${yr_p.sum():>+10,.0f} | {norm_n:>6} | {flip_n:>6}")

        # Normal vs flip breakdown
        norm_trades = [t for t in best_trades if t["src"] == "normal"]
        flip_trades = [t for t in best_trades if t["src"] == "flip"]
        print(f"\nNormal trades: {len(norm_trades)}")
        if norm_trades:
            np_n = np.array([t["pnl"] for t in norm_trades])
            print(f"  WR: {100*sum(1 for p in np_n if p>0)/len(np_n):.1f}%  PF: {np_n[np_n>0].sum()/abs(np_n[np_n<0].sum()) if (np_n<0).any() else 0:.2f}  PnL: ${np_n.sum():+,.0f}")
        print(f"Flip trades: {len(flip_trades)}")
        if flip_trades:
            fp_n = np.array([t["pnl"] for t in flip_trades])
            print(f"  WR: {100*sum(1 for p in fp_n if p>0)/len(fp_n):.1f}%  PF: {fp_n[fp_n>0].sum()/abs(fp_n[fp_n<0].sum()) if (fp_n<0).any() else 0:.2f}  PnL: ${fp_n.sum():+,.0f}")


if __name__ == "__main__":
    main()

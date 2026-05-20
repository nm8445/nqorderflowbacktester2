"""
Phase 1: Optimize NORM_LEN x Z_LOOKBACK with 60/40 IS/OOS split.
Base: H=0.40, KERNEL=80, HIGH_Z=1.5, EMA=40, SL=2, TP=1.2, session 6:45-15:45 ET.
Requirement: 250+ trades per year.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

H = 0.40; KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
HIGH_Z = 1.5; EMA_LEN = 40; ATR_LEN = 14
ATR_SL = 2.0; ATR_TP = 1.2; ET = "America/New_York"; POINT_VALUE = 20.0
SS = "06:45"; SE = "15:45"; MT = 5
MART_STREAK = 1; MART_MULT = 1.5; MART_MAX_DOUBLES = 4

# Grid
NORM_LENS = [100, 150, 200, 250, 300, 350, 400, 500, 600, 800]
Z_LOOKBACKS = [50, 75, 100, 150, 200, 250, 300, 400, 500]


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


def precompute_shared(df):
    """Precompute things that don't depend on NORM_LEN or Z_LOOKBACK."""
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)

    log_ret = np.log(closes / closes.shift(1)).values
    log_ret[0] = 0.0

    ema = closes.ewm(span=EMA_LEN, adjust=False).mean().values

    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(), (lows - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n); atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i + 1].mean()
        else: atr_vals[i] = (atr_vals[i - 1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN

    # Kernel (doesn't depend on NORM/ZLOOK)
    ks = np.arange(1, KERNEL_LEN + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = ks ** (H - 0.5) - (ks - 1) ** (H - 0.5)
    kernel[0] = 1.0

    return log_ret, ema, atr_vals, kernel, n


def compute_signal(log_ret, kernel, n, norm_len, z_lookback):
    """Compute z_vol for given NORM_LEN and Z_LOOKBACK."""
    # Rolling mean and std of returns
    ret_s = pd.Series(log_ret)
    mean_ret = ret_s.rolling(norm_len).mean().values
    std_ret = ret_s.rolling(norm_len).std(ddof=0).values

    shock = np.where(std_ret > 0, (log_ret - mean_ret) / std_ret, 0.0)
    shock = np.nan_to_num(shock, nan=0.0)

    xH = np.convolve(shock, kernel, mode="full")[:n]
    v_model = np.minimum(V0 * np.exp(ETA * xH), V0 * 1e4)

    v_s = pd.Series(v_model)
    v_m = v_s.rolling(z_lookback).mean().values
    v_d = v_s.rolling(z_lookback).std(ddof=0).values

    z_vol = np.where(v_d > 0, (v_model - v_m) / v_d, 0.0)
    z_vol = np.nan_to_num(z_vol, nan=0.0)

    return z_vol


def backtest(closes_v, highs_v, lows_v, z_vol, ema, atr_vals, idx_list, n, date_mask_is, date_mask_oos):
    """Run backtest, return IS and OOS results separately."""
    results = {}

    for label, mask in [("IS", date_mask_is), ("OOS", date_mask_oos)]:
        trades = []; trade_years = []
        pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
        qty = 1; loss_streak = 0

        for i in range(n):
            idx = idx_list[i]; td = idx.date()
            hm = idx.strftime("%H:%M"); ins = SS <= hm < SE
            cl = closes_v[i]; hi = highs_v[i]; lo = lows_v[i]

            if pos != 0 and not ins and hm >= SE:
                pnl = ((cl - ep) if d == "long" else (ep - cl)) * POINT_VALUE * qty
                if mask[i]:
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
                    if mask[i]:
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

        pnls = np.array(trades) if trades else np.array([0.0])
        years_arr = np.array(trade_years) if trade_years else np.array([0])
        w = pnls[pnls > 0]; l = pnls[pnls < 0]
        pf = w.sum() / abs(l.sum()) if len(l) else 99
        wr = 100 * len(w) / len(pnls) if len(pnls) else 0
        cum = pnls.cumsum()
        dd = (cum - np.maximum.accumulate(cum)).min() if len(cum) else 0

        # Trades per year
        unique_years = sorted(set(trade_years)) if trade_years else []
        n_years = len(unique_years) if unique_years else 1
        tpy = len(trades) / n_years if n_years else 0

        # Per-year PFs
        year_pfs = {}
        for y in unique_years:
            ym = years_arr == y
            yp = pnls[ym]
            yw = yp[yp > 0]; yl = yp[yp < 0]
            year_pfs[y] = yw.sum() / abs(yl.sum()) if len(yl) else 99

        # Count years with PF < 1.0
        losing_years = sum(1 for p in year_pfs.values() if p < 1.0)

        results[label] = {
            "trades": len(trades), "pf": pf, "wr": wr,
            "pnl": pnls.sum(), "dd": dd, "tpy": tpy,
            "year_pfs": year_pfs, "losing_years": losing_years,
            "n_years": n_years,
        }

    return results


def main():
    df = build_combined_bars()
    closes_v = df["close"].values; highs_v = df["high"].values; lows_v = df["low"].values
    idx_list = df.index; n = len(df)

    # Precompute shared
    log_ret, ema, atr_vals, kernel, n = precompute_shared(df)

    # 60/40 IS/OOS split by trading days
    trading_days = sorted(set(idx.date() for idx in idx_list))
    split_idx = int(len(trading_days) * 0.60)
    is_days = set(trading_days[:split_idx])
    oos_days = set(trading_days[split_idx:])
    print(f"IS: {len(is_days)} days ({trading_days[0]} to {trading_days[split_idx-1]})")
    print(f"OOS: {len(oos_days)} days ({trading_days[split_idx]} to {trading_days[-1]})")

    date_mask_is = np.array([idx.date() in is_days for idx in idx_list])
    date_mask_oos = np.array([idx.date() in oos_days for idx in idx_list])

    # Run grid
    print(f"\nGrid: NORM_LEN {NORM_LENS} x Z_LOOKBACK {Z_LOOKBACKS}")
    print(f"Total combos: {len(NORM_LENS) * len(Z_LOOKBACKS)}")
    print(f"Min trades/year: 250\n")

    all_results = []
    total = len(NORM_LENS) * len(Z_LOOKBACKS)
    done = 0
    t0 = time.time()

    for norm_len in NORM_LENS:
        for z_lookback in Z_LOOKBACKS:
            z_vol = compute_signal(log_ret, kernel, n, norm_len, z_lookback)
            res = backtest(closes_v, highs_v, lows_v, z_vol, ema, atr_vals, idx_list, n, date_mask_is, date_mask_oos)

            is_r = res["IS"]; oos_r = res["OOS"]
            all_results.append({
                "norm": norm_len, "zlook": z_lookback,
                "is": is_r, "oos": oos_r,
            })

            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted {total} combos in {elapsed:.0f}s\n")

    # Filter: IS trades/year >= 250
    valid = []
    for r in all_results:
        is_tpy = r["is"]["tpy"]
        oos_tpy = r["oos"]["tpy"]
        if is_tpy >= 250:
            valid.append(r)

    print(f"Configs with IS 250+ trades/year: {len(valid)} / {len(all_results)}\n")

    # Sort by IS PF
    valid.sort(key=lambda x: x["is"]["pf"], reverse=True)

    # Print top 20
    print(f"{'NORM':>5} {'ZLOOK':>6} | {'IS Trd':>7} {'IS PF':>6} {'IS WR':>6} {'IS PnL':>10} {'IS DD':>10} {'IS LY':>5} | "
          f"{'OOS Trd':>7} {'OOS PF':>7} {'OOS WR':>7} {'OOS PnL':>10} {'OOS DD':>10} {'OOS LY':>6}")
    print("-" * 130)

    for r in valid[:30]:
        is_r = r["is"]; oos_r = r["oos"]
        print(f"{r['norm']:>5} {r['zlook']:>6} | "
              f"{is_r['trades']:>7} {is_r['pf']:>6.2f} {is_r['wr']:>5.1f}% ${is_r['pnl']:>+9,.0f} ${is_r['dd']:>9,.0f} {is_r['losing_years']:>5} | "
              f"{oos_r['trades']:>7} {oos_r['pf']:>7.2f} {oos_r['wr']:>6.1f}% ${oos_r['pnl']:>+9,.0f} ${oos_r['dd']:>9,.0f} {oos_r['losing_years']:>6}")

    # Also show sorted by OOS PF (top 15)
    print(f"\n--- Top 15 by OOS PF (from configs with IS 250+/yr) ---")
    valid_oos = sorted(valid, key=lambda x: x["oos"]["pf"], reverse=True)
    print(f"{'NORM':>5} {'ZLOOK':>6} | {'IS Trd':>7} {'IS PF':>6} {'IS WR':>6} {'IS PnL':>10} {'IS DD':>10} | "
          f"{'OOS Trd':>7} {'OOS PF':>7} {'OOS WR':>7} {'OOS PnL':>10} {'OOS DD':>10}")
    print("-" * 115)
    for r in valid_oos[:15]:
        is_r = r["is"]; oos_r = r["oos"]
        print(f"{r['norm']:>5} {r['zlook']:>6} | "
              f"{is_r['trades']:>7} {is_r['pf']:>6.2f} {is_r['wr']:>5.1f}% ${is_r['pnl']:>+9,.0f} ${is_r['dd']:>9,.0f} | "
              f"{oos_r['trades']:>7} {oos_r['pf']:>7.2f} {oos_r['wr']:>6.1f}% ${oos_r['pnl']:>+9,.0f} ${oos_r['dd']:>9,.0f}")

    # Best balanced: average of IS and OOS PF
    print(f"\n--- Top 15 by balanced score (avg IS+OOS PF) ---")
    valid_bal = sorted(valid, key=lambda x: (x["is"]["pf"] + x["oos"]["pf"]) / 2, reverse=True)
    print(f"{'NORM':>5} {'ZLOOK':>6} | {'IS PF':>6} {'OOS PF':>7} {'AVG':>5} | {'IS Trd':>7} {'OOS Trd':>7} {'Total PnL':>10} {'Worst DD':>10}")
    print("-" * 90)
    for r in valid_bal[:15]:
        is_r = r["is"]; oos_r = r["oos"]
        avg_pf = (is_r["pf"] + oos_r["pf"]) / 2
        total_pnl = is_r["pnl"] + oos_r["pnl"]
        worst_dd = min(is_r["dd"], oos_r["dd"])
        print(f"{r['norm']:>5} {r['zlook']:>6} | "
              f"{is_r['pf']:>6.2f} {oos_r['pf']:>7.2f} {avg_pf:>5.2f} | "
              f"{is_r['trades']:>7} {oos_r['trades']:>7} ${total_pnl:>+9,.0f} ${worst_dd:>9,.0f}")


if __name__ == "__main__":
    main()

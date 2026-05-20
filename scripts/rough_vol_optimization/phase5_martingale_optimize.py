"""
Phase 5: Optimize martingale with 60/40 IS/OOS split.
Force close 3 min before session end (15:27 for 15:30 session).
Locked: H=0.40, KERNEL=80, NORM=250, ZLOOK=100, HIGH_Z=1.5, EMA=50,
        SL=1.5, TP=2.5, ATR_LEN=14, session 06:45-15:30 ET, MT=5.

Tests: session start (06:45 vs 09:00 vs 09:30), streak, mult, max_doubles, daily reset vs carry.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

H = 0.40; KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
NORM_LEN = 250; Z_LOOKBACK = 100; HIGH_Z = 1.5; EMA_LEN = 50; ATR_LEN = 14
ATR_SL = 1.5; ATR_TP = 2.5
ET = "America/New_York"; POINT_VALUE = 20.0
MT = 5

# Grid
SESSION_CONFIGS = [("06:45", "15:30", "15:27"), ("09:00", "15:30", "15:27"), ("09:30", "15:30", "15:27")]
STREAKS = [1, 2, 3]
MULTS = [1.5, 2.0, 3.0]
MAX_DOUBLES_LIST = [1, 2, 3, 4]
RESET_DAILY = [True, False]
# Also test no martingale
TEST_NO_MART = True


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


def precompute(df):
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)
    log_ret = np.log(closes / closes.shift(1)).values; log_ret[0] = 0.0
    ret_s = pd.Series(log_ret)
    mean_ret = ret_s.rolling(NORM_LEN).mean().values
    std_ret = ret_s.rolling(NORM_LEN).std(ddof=0).values
    shock = np.nan_to_num(np.where(std_ret > 0, (log_ret - mean_ret) / std_ret, 0.0), nan=0.0)
    ks = np.arange(1, KERNEL_LEN + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = ks ** (H - 0.5) - (ks - 1) ** (H - 0.5)
    kernel[0] = 1.0
    xH = np.convolve(shock, kernel, mode="full")[:n]
    v_model = np.minimum(V0 * np.exp(ETA * xH), V0 * 1e4)
    v_s = pd.Series(v_model)
    v_m = v_s.rolling(Z_LOOKBACK).mean().values
    v_d = v_s.rolling(Z_LOOKBACK).std(ddof=0).values
    z_vol = np.nan_to_num(np.where(v_d > 0, (v_model - v_m) / v_d, 0.0), nan=0.0)
    ema = closes.ewm(span=EMA_LEN, adjust=False).mean().values
    tr = np.maximum(highs.values - lows.values,
                    np.maximum(np.abs(highs.values - np.roll(closes.values, 1)),
                               np.abs(lows.values - np.roll(closes.values, 1))))
    tr[0] = highs.values[0] - lows.values[0]
    atr_vals = np.zeros(n); atr_vals[0] = tr[0]
    for i in range(1, n):
        if i < ATR_LEN: atr_vals[i] = tr[:i + 1].mean()
        else: atr_vals[i] = (atr_vals[i - 1] * (ATR_LEN - 1) + tr[i]) / ATR_LEN
    return z_vol, ema, atr_vals, n


def backtest(closes_v, highs_v, lows_v, z_vol, ema, atr_vals, idx_list, hm_list, n,
             ss, se, force_close_time, mart_streak, mart_mult, max_doubles, reset_daily,
             use_mart, date_mask_is, date_mask_oos):
    results = {}
    for label, mask in [("IS", date_mask_is), ("OOS", date_mask_oos)]:
        trades = []; trade_years = []
        pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
        qty = 1; loss_streak = 0

        for i in range(n):
            idx = idx_list[i]; td = idx.date()
            hm = hm_list[i]; ins = ss <= hm < se
            cl = closes_v[i]; hi = highs_v[i]; lo = lows_v[i]

            # Force close 3 min before session end
            if pos != 0 and hm >= force_close_time:
                pnl = ((cl - ep) if d == "long" else (ep - cl)) * POINT_VALUE * qty
                if mask[i]: trades.append(pnl); trade_years.append(td.year)
                if pnl > 0: loss_streak = 0
                else: loss_streak += 1
                pos = 0
                continue

            if not ins: continue
            if td != cd:
                cd = td; dt = 0
                if reset_daily: loss_streak = 0
            if use_mart and loss_streak >= mart_streak:
                steps = min(loss_streak - mart_streak + 1, max_doubles)
                qty = max(1, round(mart_mult ** steps))
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
                    if mask[i]: trades.append(pnl); trade_years.append(td.year)
                    if pnl > 0: loss_streak = 0
                    else: loss_streak += 1
                    if use_mart and loss_streak >= mart_streak:
                        steps = min(loss_streak - mart_streak + 1, max_doubles)
                        qty = max(1, round(mart_mult ** steps))
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
        unique_years = sorted(set(trade_years)) if trade_years else []
        n_years = len(unique_years) if unique_years else 1
        tpy = len(trades) / n_years if n_years else 0
        year_pfs = {}
        for y in unique_years:
            ym = years_arr == y; yp = pnls[ym]; yw = yp[yp > 0]; yl = yp[yp < 0]
            year_pfs[y] = yw.sum() / abs(yl.sum()) if len(yl) else 99
        losing_years = sum(1 for p in year_pfs.values() if p < 1.0)
        results[label] = {"trades": len(trades), "pf": pf, "wr": wr, "pnl": pnls.sum(),
                          "dd": dd, "tpy": tpy, "losing_years": losing_years, "year_pfs": year_pfs}
    return results


def main():
    df = build_combined_bars()
    closes_v = df["close"].values; highs_v = df["high"].values; lows_v = df["low"].values
    idx_list = df.index; n = len(df)
    hm_list = np.array([idx.strftime("%H:%M") for idx in idx_list])

    z_vol, ema, atr_vals, n = precompute(df)

    # 60/40 split
    trading_days = sorted(set(idx.date() for idx in idx_list))
    split_idx = int(len(trading_days) * 0.60)
    is_days = set(trading_days[:split_idx])
    oos_days = set(trading_days[split_idx:])
    print(f"IS: {len(is_days)} days ({trading_days[0]} to {trading_days[split_idx-1]})")
    print(f"OOS: {len(oos_days)} days ({trading_days[split_idx]} to {trading_days[-1]})")
    date_mask_is = np.array([idx.date() in is_days for idx in idx_list])
    date_mask_oos = np.array([idx.date() in oos_days for idx in idx_list])

    all_results = []
    done = 0; t0 = time.time()

    # First: no martingale baseline for each session
    for ss, se, fc in SESSION_CONFIGS:
        res = backtest(closes_v, highs_v, lows_v, z_vol, ema, atr_vals, idx_list, hm_list, n,
                       ss, se, fc, 1, 1.5, 4, True, False, date_mask_is, date_mask_oos)
        all_results.append({"ss": ss, "se": se, "streak": "-", "mult": "-", "max_d": "-",
                           "reset": "-", "is": res["IS"], "oos": res["OOS"], "label": "NO MART"})
        done += 1

    # Martingale grid
    total_mart = len(SESSION_CONFIGS) * len(STREAKS) * len(MULTS) * len(MAX_DOUBLES_LIST) * len(RESET_DAILY)
    total = total_mart + len(SESSION_CONFIGS)
    print(f"\nTotal combos: {total} ({len(SESSION_CONFIGS)} no-mart + {total_mart} mart)")
    print(f"Force close at 15:27 (3 min before 15:30)\n")

    for ss, se, fc in SESSION_CONFIGS:
        for streak in STREAKS:
            for mult in MULTS:
                for max_d in MAX_DOUBLES_LIST:
                    for reset in RESET_DAILY:
                        res = backtest(closes_v, highs_v, lows_v, z_vol, ema, atr_vals, idx_list, hm_list, n,
                                       ss, se, fc, streak, mult, max_d, reset, True, date_mask_is, date_mask_oos)
                        reset_str = "D" if reset else "C"
                        all_results.append({"ss": ss, "se": se, "streak": streak, "mult": mult,
                                           "max_d": max_d, "reset": reset_str,
                                           "is": res["IS"], "oos": res["OOS"], "label": "MART"})
                        done += 1
                        if done % 50 == 0:
                            elapsed = time.time() - t0
                            eta = elapsed / done * (total - done)
                            print(f"  [{done}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted {done} combos in {elapsed:.0f}s\n")

    # Show no-mart baselines first
    print("=" * 120)
    print("NO MARTINGALE BASELINES (force close at 15:27)")
    print("=" * 120)
    print(f"{'Session':>14} | {'IS Trd':>7} {'IS PF':>6} {'IS WR':>6} {'IS PnL':>10} {'IS DD':>10} | "
          f"{'OOS Trd':>7} {'OOS PF':>7} {'OOS WR':>7} {'OOS PnL':>10} {'OOS DD':>10}")
    print("-" * 110)
    for r in all_results:
        if r["label"] != "NO MART": continue
        is_r = r["is"]; oos_r = r["oos"]
        print(f"{r['ss']+'-'+r['se']:>14} | "
              f"{is_r['trades']:>7} {is_r['pf']:>6.2f} {is_r['wr']:>5.1f}% ${is_r['pnl']:>+9,.0f} ${is_r['dd']:>9,.0f} | "
              f"{oos_r['trades']:>7} {oos_r['pf']:>7.2f} {oos_r['wr']:>6.1f}% ${oos_r['pnl']:>+9,.0f} ${oos_r['dd']:>9,.0f}")

    # Filter mart results
    mart_results = [r for r in all_results if r["label"] == "MART"]

    # Top 30 by balanced PF
    print(f"\n{'='*130}")
    print("TOP 30 MARTINGALE CONFIGS by balanced score (avg IS+OOS PF)")
    print(f"{'='*130}")
    mart_bal = sorted(mart_results, key=lambda x: (x["is"]["pf"] + x["oos"]["pf"]) / 2, reverse=True)
    print(f"{'Session':>14} {'Str':>3} {'Mult':>4} {'MxD':>3} {'Rst':>3} | {'IS PF':>6} {'OOS PF':>7} {'AVG':>5} | "
          f"{'IS Trd':>7} {'OOS Trd':>7} {'Total PnL':>10} {'Worst DD':>10} {'IS WR':>6} {'OOS WR':>7}")
    print("-" * 120)
    for r in mart_bal[:30]:
        is_r = r["is"]; oos_r = r["oos"]
        avg_pf = (is_r["pf"] + oos_r["pf"]) / 2
        total_pnl = is_r["pnl"] + oos_r["pnl"]
        worst_dd = min(is_r["dd"], oos_r["dd"])
        print(f"{r['ss']+'-'+r['se']:>14} {r['streak']:>3} {r['mult']:>4} {r['max_d']:>3} {r['reset']:>3} | "
              f"{is_r['pf']:>6.2f} {oos_r['pf']:>7.2f} {avg_pf:>5.2f} | "
              f"{is_r['trades']:>7} {oos_r['trades']:>7} ${total_pnl:>+9,.0f} ${worst_dd:>9,.0f} {is_r['wr']:>5.1f}% {oos_r['wr']:>6.1f}%")

    # Best by OOS PF
    print(f"\n--- Top 15 by OOS PF ---")
    mart_oos = sorted(mart_results, key=lambda x: x["oos"]["pf"], reverse=True)
    print(f"{'Session':>14} {'Str':>3} {'Mult':>4} {'MxD':>3} {'Rst':>3} | {'IS PF':>6} {'OOS PF':>7} | "
          f"{'OOS Trd':>7} {'OOS PnL':>10} {'OOS DD':>10} {'OOS WR':>7}")
    print("-" * 100)
    for r in mart_oos[:15]:
        is_r = r["is"]; oos_r = r["oos"]
        print(f"{r['ss']+'-'+r['se']:>14} {r['streak']:>3} {r['mult']:>4} {r['max_d']:>3} {r['reset']:>3} | "
              f"{is_r['pf']:>6.2f} {oos_r['pf']:>7.2f} | "
              f"{oos_r['trades']:>7} ${oos_r['pnl']:>+9,.0f} ${oos_r['dd']:>9,.0f} {oos_r['wr']:>6.1f}%")

    # Best by lowest DD
    print(f"\n--- Top 15 by tightest OOS DD (with OOS PF > 1.0) ---")
    mart_dd = [r for r in mart_results if r["oos"]["pf"] > 1.0]
    mart_dd.sort(key=lambda x: x["oos"]["dd"], reverse=True)  # least negative first
    print(f"{'Session':>14} {'Str':>3} {'Mult':>4} {'MxD':>3} {'Rst':>3} | {'IS PF':>6} {'OOS PF':>7} | "
          f"{'OOS Trd':>7} {'OOS PnL':>10} {'OOS DD':>10} {'Total PnL':>10}")
    print("-" * 105)
    for r in mart_dd[:15]:
        is_r = r["is"]; oos_r = r["oos"]
        total_pnl = is_r["pnl"] + oos_r["pnl"]
        print(f"{r['ss']+'-'+r['se']:>14} {r['streak']:>3} {r['mult']:>4} {r['max_d']:>3} {r['reset']:>3} | "
              f"{is_r['pf']:>6.2f} {oos_r['pf']:>7.2f} | "
              f"{oos_r['trades']:>7} ${oos_r['pnl']:>+9,.0f} ${oos_r['dd']:>9,.0f} ${total_pnl:>+9,.0f}")


if __name__ == "__main__":
    main()

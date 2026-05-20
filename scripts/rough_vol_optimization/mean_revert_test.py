"""
Test mean-revert (low z) addition to optimized trend config.
Locked: H=0.40, KERNEL=80, NORM=250, ZLOOK=100, HIGH_Z=1.5, EMA=50,
        SL=1.5, TP=2.5, ATR=14, Session 09:30-15:30 (close 15:27), MT=5,
        Marti streak=1, mult=3.0, max_doubles=4, daily reset.

Compares: trend-only baseline vs trend+MR at various LOW_Z thresholds.
Also tests MR with different SL/TP since MR trades may need different risk.
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
ET = "America/New_York"; POINT_VALUE = 20.0
SS = "09:30"; SE = "15:30"; FC = "15:27"; MT = 5
MART_STREAK = 1; MART_MULT = 3.0; MART_MAX_DOUBLES = 4

# Grid: LOW_Z thresholds to test, plus separate MR SL/TP options
LOW_ZS = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
# MR risk: (sl_mult, tp_mult) — try same as trend, and tighter options
MR_RISKS = [
    (1.5, 2.5),   # same as trend
    (1.0, 1.5),   # tighter
    (1.0, 2.0),
    (2.0, 1.5),   # wider SL, tighter TP
    (2.0, 3.0),   # wider both
]


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
             low_z, mr_sl, mr_tp, date_mask_is, date_mask_oos, include_mr=True):
    """
    Backtest with trend entries (z > HIGH_Z) and optionally mean-revert entries (z < low_z).
    MR entries use counter-trend direction and separate SL/TP.
    """
    trend_sl = 1.5; trend_tp = 2.5  # locked trend risk

    results = {}
    for label, mask in [("IS", date_mask_is), ("OOS", date_mask_oos)]:
        trades = []; trade_years = []; mr_trades = []; trend_trades_list = []
        pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
        qty = 1; loss_streak = 0; trade_type = ""

        for i in range(n):
            idx = idx_list[i]; td = idx.date()
            hm = hm_list[i]
            cl = closes_v[i]; hi = highs_v[i]; lo = lows_v[i]

            # Force close at 15:27
            if pos != 0 and hm >= FC:
                pnl = ((cl - ep) if d == "long" else (ep - cl)) * POINT_VALUE * qty
                if mask[i]:
                    trades.append(pnl); trade_years.append(td.year)
                    if trade_type == "MR": mr_trades.append(pnl)
                    else: trend_trades_list.append(pnl)
                if pnl > 0: loss_streak = 0
                else: loss_streak += 1
                pos = 0
                continue

            ins = SS <= hm < SE
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
                        if trade_type == "MR": mr_trades.append(pnl)
                        else: trend_trades_list.append(pnl)
                    if pnl > 0: loss_streak = 0
                    else: loss_streak += 1
                    if loss_streak >= MART_STREAK:
                        steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                        qty = max(1, round(MART_MULT ** steps))
                    else: qty = 1
                    pos = 0; continue

            if pos == 0 and dt < MT:
                # Trend entry: z > HIGH_Z, trend direction
                if z > HIGH_Z and cl > e:
                    pos = 1; d = "long"; ep = cl
                    sl = cl - trend_sl * atr_v; tp = cl + trend_tp * atr_v
                    dt += 1; trade_type = "T"
                elif z > HIGH_Z and cl < e:
                    pos = -1; d = "short"; ep = cl
                    sl = cl + trend_sl * atr_v; tp = cl - trend_tp * atr_v
                    dt += 1; trade_type = "T"
                # Mean-revert entry: z < low_z, counter-trend direction
                elif include_mr and z < low_z and cl < e:
                    pos = 1; d = "long"; ep = cl
                    sl = cl - mr_sl * atr_v; tp = cl + mr_tp * atr_v
                    dt += 1; trade_type = "MR"
                elif include_mr and z < low_z and cl > e:
                    pos = -1; d = "short"; ep = cl
                    sl = cl + mr_sl * atr_v; tp = cl - mr_tp * atr_v
                    dt += 1; trade_type = "MR"

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

        mr_pnls = np.array(mr_trades) if mr_trades else np.array([0.0])
        mr_w = mr_pnls[mr_pnls > 0]; mr_l = mr_pnls[mr_pnls < 0]
        mr_pf = mr_w.sum() / abs(mr_l.sum()) if len(mr_l) else 99
        mr_wr = 100 * len(mr_w) / len(mr_pnls) if len(mr_pnls) else 0

        results[label] = {
            "trades": len(trades), "pf": pf, "wr": wr, "pnl": pnls.sum(), "dd": dd, "tpy": tpy,
            "mr_trades": len(mr_trades), "mr_pf": mr_pf, "mr_wr": mr_wr, "mr_pnl": sum(mr_trades),
            "trend_trades": len(trend_trades_list), "trend_pnl": sum(trend_trades_list),
        }
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
    print(f"IS: {len(is_days)} days | OOS: {len(oos_days)} days")
    date_mask_is = np.array([idx.date() in is_days for idx in idx_list])
    date_mask_oos = np.array([idx.date() in oos_days for idx in idx_list])

    # Baseline: trend-only
    print("\n=== BASELINE: Trend-only ===")
    base = backtest(closes_v, highs_v, lows_v, z_vol, ema, atr_vals, idx_list, hm_list, n,
                    0, 0, 0, date_mask_is, date_mask_oos, include_mr=False)
    is_b = base["IS"]; oos_b = base["OOS"]
    print(f"  IS:  {is_b['trades']:>5} trades  PF {is_b['pf']:.2f}  WR {is_b['wr']:.1f}%  PnL ${is_b['pnl']:>+10,.0f}  DD ${is_b['dd']:>9,.0f}")
    print(f"  OOS: {oos_b['trades']:>5} trades  PF {oos_b['pf']:.2f}  WR {oos_b['wr']:.1f}%  PnL ${oos_b['pnl']:>+10,.0f}  DD ${oos_b['dd']:>9,.0f}")
    print(f"  AVG PF: {(is_b['pf'] + oos_b['pf'])/2:.2f}  Total: ${is_b['pnl']+oos_b['pnl']:>+10,.0f}")

    # MR-only test first: how does MR perform by itself?
    print("\n=== MR-ONLY (no trend) — how does mean-revert do alone? ===")
    print(f"{'LOW_Z':>6} {'MR_SL':>5} {'MR_TP':>5} | {'IS Trd':>6} {'IS PF':>6} {'IS WR':>6} {'IS PnL':>10} | {'OOS Trd':>7} {'OOS PF':>7} {'OOS WR':>7} {'OOS PnL':>10}")
    print("-" * 100)
    for low_z in LOW_ZS:
        for mr_sl, mr_tp in MR_RISKS:
            # Hack: set HIGH_Z very high so no trend trades fire, only MR
            res = backtest(closes_v, highs_v, lows_v, z_vol, ema, atr_vals, idx_list, hm_list, n,
                           low_z, mr_sl, mr_tp, date_mask_is, date_mask_oos, include_mr=True)
            # But we need a clean MR-only test. Let me just look at the MR portion from the combined run
            pass

    # Actually, let's just run combined and look at MR contribution
    print("\n=== TREND + MEAN-REVERT COMBINED ===")
    print(f"{'LOW_Z':>6} {'MR_SL':>5} {'MR_TP':>5} | {'IS Tot':>6} {'IS PF':>6} {'IS PnL':>10} {'IS DD':>10} | "
          f"{'OOS Tot':>7} {'OOS PF':>7} {'OOS PnL':>10} {'OOS DD':>10} | {'IS MR':>5} {'IS MR$':>10} {'OOS MR':>6} {'OOS MR$':>10} {'AVG PF':>6}")
    print("-" * 160)

    all_results = []
    t0 = time.time()
    total = len(LOW_ZS) * len(MR_RISKS)
    done = 0

    for low_z in LOW_ZS:
        for mr_sl, mr_tp in MR_RISKS:
            res = backtest(closes_v, highs_v, lows_v, z_vol, ema, atr_vals, idx_list, hm_list, n,
                           low_z, mr_sl, mr_tp, date_mask_is, date_mask_oos, include_mr=True)
            is_r = res["IS"]; oos_r = res["OOS"]
            avg_pf = (is_r["pf"] + oos_r["pf"]) / 2
            all_results.append({
                "low_z": low_z, "mr_sl": mr_sl, "mr_tp": mr_tp,
                "is": is_r, "oos": oos_r, "avg_pf": avg_pf
            })
            print(f"{low_z:>6.1f} {mr_sl:>5.1f} {mr_tp:>5.1f} | "
                  f"{is_r['trades']:>6} {is_r['pf']:>6.2f} ${is_r['pnl']:>+9,.0f} ${is_r['dd']:>9,.0f} | "
                  f"{oos_r['trades']:>7} {oos_r['pf']:>7.2f} ${oos_r['pnl']:>+9,.0f} ${oos_r['dd']:>9,.0f} | "
                  f"{is_r['mr_trades']:>5} ${is_r['mr_pnl']:>+9,.0f} {oos_r['mr_trades']:>6} ${oos_r['mr_pnl']:>+9,.0f} {avg_pf:>6.2f}")
            done += 1

    elapsed = time.time() - t0
    print(f"\nCompleted {done} combos in {elapsed:.0f}s")

    # Top 10 by balanced PF that beat baseline
    base_avg = (is_b["pf"] + oos_b["pf"]) / 2
    better = [r for r in all_results if r["avg_pf"] > base_avg]
    better.sort(key=lambda x: x["avg_pf"], reverse=True)

    print(f"\n=== CONFIGS THAT BEAT BASELINE (avg PF {base_avg:.2f}) ===")
    if not better:
        print("  None! Mean-revert hurts in all configurations tested.")
    else:
        print(f"{'LOW_Z':>6} {'MR_SL':>5} {'MR_TP':>5} | {'AVG PF':>6} | {'IS PF':>6} {'OOS PF':>7} | {'Total PnL':>10} {'Worst DD':>10} | {'MR Trades':>9} {'MR PnL':>10}")
        print("-" * 110)
        for r in better[:15]:
            is_r = r["is"]; oos_r = r["oos"]
            total_pnl = is_r["pnl"] + oos_r["pnl"]
            worst_dd = min(is_r["dd"], oos_r["dd"])
            mr_total = is_r["mr_trades"] + oos_r["mr_trades"]
            mr_pnl = is_r["mr_pnl"] + oos_r["mr_pnl"]
            print(f"{r['low_z']:>6.1f} {r['mr_sl']:>5.1f} {r['mr_tp']:>5.1f} | {r['avg_pf']:>6.2f} | "
                  f"{is_r['pf']:>6.2f} {oos_r['pf']:>7.2f} | ${total_pnl:>+9,.0f} ${worst_dd:>9,.0f} | "
                  f"{mr_total:>9} ${mr_pnl:>+9,.0f}")


if __name__ == "__main__":
    main()

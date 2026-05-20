"""
VWAP + z_vol day-hold strategy backtest.

Entry: z_vol > 1.5 AND close above VWAP (long) or below VWAP (short)
Stop:  ATR multiple below VWAP (long) or above VWAP (short)
Exit:  Force close at 16:30 ET (no TP)
Entry window: 00:00 - 16:30 ET (overnight + RTH)

Uses 15-min bars + VWAP cache (1-min, resampled to 15-min via last value).
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
VWAP_CACHE = Path("D:/trading_pythonbacktest_data/vwap_cache_5yr")

# Rough vol params (same kernel as optimized)
H = 0.40; KERNEL_LEN = 80; ETA = 1.0; V0 = 0.0001
NORM_LEN = 250; Z_LOOKBACK = 100; HIGH_Z = 1.5; EMA_LEN = 50; ATR_LEN = 14
ET = "America/New_York"; POINT_VALUE = 20.0

# Strategy params to grid
ATR_SL_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
FORCE_CLOSE = "16:30"
SESSION_START = "00:00"  # entries from midnight
MAX_TRADES_DAY = 1       # one trade per day since we hold all day

OUT_DIR = Path("C:/trading/nqorderflowbacktester/results/html")


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
    """Load VWAP cache for a date, return dict of {UTC 15-min bucket -> vwap value}."""
    cache_file = VWAP_CACHE / f"{date_obj}.pkl"
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as f:
        df = pickle.load(f)
    # Keep in UTC, floor to 15-min buckets
    df["group"] = df.index.floor("15min")
    vwap_15 = df.groupby("group")["vwap"].last()
    # Return as dict keyed by UTC timestamp
    return vwap_15.to_dict()


def main():
    df = build_combined_bars()
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)

    # Compute z_vol
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

    # ATR (RMA)
    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(),
                    (lows - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n); atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i+1].mean()
        else: atr_vals[i] = (atr_vals[i-1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    # Preload all VWAP data
    print("Loading VWAP cache...", flush=True)
    all_dates = sorted(set(idx.date() for idx in df.index))
    vwap_by_date = {}
    loaded = 0
    for d in all_dates:
        v = load_vwap_for_date(d)
        if v is not None:
            vwap_by_date[d] = v
            loaded += 1
    print(f"  Loaded VWAP for {loaded}/{len(all_dates)} dates", flush=True)

    # IS/OOS split
    trading_days = sorted(set(idx.date() for idx in df.index))
    split_idx = int(len(trading_days) * 0.60)
    is_cutoff = str(trading_days[split_idx])

    # Grid search over ATR SL multiples
    print(f"\n{'SL_ATR':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'IS PF':>6} | {'OOS PF':>6} | {'Avg PF':>6} | {'Total PnL':>11} | {'MaxDD':>10} | {'Avg PnL':>8} | {'Longs':>6} | {'Shorts':>6}")
    print("-" * 120)

    best_avg_pf = 0
    best_config = None
    best_trades = None

    for atr_sl in ATR_SL_VALUES:
        trades = []
        pos = 0; ep = 0; sl = 0; d = ""; cd = None; dt = 0; vwap_at_entry = 0

        for i in range(n):
            row = df.iloc[i]; idx = df.index[i]; td = idx.date()
            hm = idx.strftime("%H:%M")

            # Force close at 16:30
            if pos != 0 and hm >= FORCE_CLOSE:
                pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": d,
                               "time": idx.strftime("%Y-%m-%d %H:%M"), "reason": "CLOSE"})
                pos = 0
                continue

            if hm >= FORCE_CLOSE:
                continue

            # New day reset
            if td != cd:
                cd = td; dt = 0

            atr_v = row["atr"]
            if atr_v <= 0: continue

            # Check stop while in position
            if pos != 0:
                if d == "long" and row["low"] <= sl:
                    pnl = (sl - ep) * POINT_VALUE
                    trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": d,
                                   "time": idx.strftime("%Y-%m-%d %H:%M"), "reason": "SL"})
                    pos = 0; continue
                elif d == "short" and row["high"] >= sl:
                    pnl = (ep - sl) * POINT_VALUE
                    trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": d,
                                   "time": idx.strftime("%Y-%m-%d %H:%M"), "reason": "SL"})
                    pos = 0; continue

            # Entry logic
            if pos == 0 and dt < MAX_TRADES_DAY:
                z = row["z_vol"]; cl = row["close"]
                if z <= HIGH_Z:
                    continue

                # Get VWAP for this bar
                vwap_dict = vwap_by_date.get(td)
                if vwap_dict is None:
                    continue
                # Convert ET bar time to UTC for VWAP lookup
                bar_utc = idx.tz_convert("UTC").floor("15min")
                vwap_val = vwap_dict.get(bar_utc)
                if vwap_val is None:
                    # Try previous 15-min bucket
                    prev = bar_utc - pd.Timedelta(minutes=15)
                    vwap_val = vwap_dict.get(prev)
                if vwap_val is None:
                    continue

                if cl > vwap_val:
                    # Long: close above VWAP
                    pos = 1; d = "long"; ep = cl
                    sl = vwap_val - atr_sl * atr_v  # stop below VWAP
                    vwap_at_entry = vwap_val; dt += 1
                elif cl < vwap_val:
                    # Short: close below VWAP
                    pos = -1; d = "short"; ep = cl
                    sl = vwap_val + atr_sl * atr_v  # stop above VWAP
                    vwap_at_entry = vwap_val; dt += 1

        # Stats
        if not trades:
            print(f"{atr_sl:>6.1f} | {'no trades':>50}")
            continue

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
        avg_pf = (is_pf + oos_pf) / 2

        longs = sum(1 for t in trades if t["dir"] == "long")
        shorts = sum(1 for t in trades if t["dir"] == "short")

        print(f"{atr_sl:>6.1f} | {len(trades):>6} | {wr:>5.1f} | {pf:>5.2f} | {is_pf:>6.2f} | {oos_pf:>6.2f} | {avg_pf:>6.2f} | ${pnls.sum():>+10,.0f} | ${dd:>+9,.0f} | ${pnls.mean():>+7,.0f} | {longs:>6} | {shorts:>6}")

        if avg_pf > best_avg_pf and len(trades) >= 50:
            best_avg_pf = avg_pf
            best_config = atr_sl
            best_trades = trades

    print()

    # Show yearly breakdown for best config
    if best_trades:
        print(f"Best config: SL = {best_config}x ATR below/above VWAP")
        print(f"\nYearly breakdown (SL={best_config}x):")
        print(f"{'Year':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>11} | {'Longs':>6} | {'Shorts':>6}")
        print("-" * 65)

        yearly = defaultdict(list)
        for t in best_trades:
            yearly[t["date"][:4]].append(t)

        for yr in sorted(yearly.keys()):
            yr_trades = yearly[yr]
            yr_pnls = np.array([t["pnl"] for t in yr_trades])
            yr_wins = yr_pnls[yr_pnls > 0]
            yr_losses = yr_pnls[yr_pnls < 0]
            yr_pf = yr_wins.sum() / abs(yr_losses.sum()) if len(yr_losses) else 0
            yr_wr = 100 * len(yr_wins) / len(yr_pnls)
            yr_longs = sum(1 for t in yr_trades if t["dir"] == "long")
            yr_shorts = sum(1 for t in yr_trades if t["dir"] == "short")
            print(f"{yr:>6} | {len(yr_trades):>6} | {yr_wr:>5.1f} | {yr_pf:>5.2f} | ${yr_pnls.sum():>+10,.0f} | {yr_longs:>6} | {yr_shorts:>6}")

        # Exit reason breakdown
        print(f"\nExit reasons:")
        reasons = defaultdict(list)
        for t in best_trades:
            reasons[t["reason"]].append(t["pnl"])
        for r in sorted(reasons.keys()):
            pnls_r = np.array(reasons[r])
            print(f"  {r}: {len(pnls_r)} trades, PnL ${pnls_r.sum():+,.0f}, Avg ${pnls_r.mean():+,.0f}")

        # Entry time distribution
        print(f"\nEntry time distribution (hour ET):")
        hour_counts = defaultdict(int)
        for t in best_trades:
            hr = int(t["time"].split(" ")[1].split(":")[0])
            hour_counts[hr] += 1
        for hr in sorted(hour_counts.keys()):
            print(f"  {hr:02d}:00 - {hr:02d}:59  {hour_counts[hr]:>4} entries")


if __name__ == "__main__":
    main()

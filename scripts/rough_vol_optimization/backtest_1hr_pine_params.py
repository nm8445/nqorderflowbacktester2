"""
5-year backtest on 1HR bars using Pine Script parameters.

Resamples 15-min MarketTick parquet → 1hr, then appends Databento 5-min pkl → 1hr.
Config from TradingView Pine Script rough vol strat.

Usage:
    python -u scripts/rough_vol_optimization/backtest_1hr_pine_params.py
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ── Data paths ──────────────────────────────────────────────────────────────
MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

# ── Pine Script parameters (1hr bars) ──────────────────────────────────────
H = 0.40
KERNEL_LEN = 80
ETA = 1.0
V0 = 0.0001
NORM_LEN = 500
Z_LOOKBACK = 150
HIGH_Z = 1.1
LOW_Z = -1.0       # exit when z_vol drops below this
EMA_LEN = 100
ATR_LEN = 14
ATR_TP = 4.0

ET = "America/New_York"
# Pine session 0545-1445 CT = 0645-1545 ET
SS = "06:45"
SE = "15:45"
POINT_VALUE = 20.0
MT = 5              # max trades per day (pyramiding=0 but allow re-entry after exit)

# No martingale for this test (Pine script has pyramiding=0, no sizing logic)
USE_MARTINGALE = False


def build_1hr_bars():
    """Resample 15-min parquet + 5-min pkl into 1hr OHLC bars."""
    # Part 1: MarketTick 15-min → 1hr
    print("Loading MarketTick 15-min parquet...", flush=True)
    df_mt = pd.read_parquet(MARKETTICK_PARQUET)
    print(f"  15-min bars: {len(df_mt):,}  ({df_mt.index[0]} to {df_mt.index[-1]})")

    # Resample 15min → 1hr
    ohlc_mt = df_mt[["open", "high", "low", "close"]].resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna(subset=["open"])
    print(f"  1hr bars from MarketTick: {len(ohlc_mt):,}")

    # Part 2: Databento 5-min pkl → 1hr (for dates after MarketTick ends)
    print("Loading Databento 5-min pkl...", flush=True)
    frames = []
    for f in sorted(TIMEBARS_DIR.glob("timebars_5min_202*.pkl")):
        with open(f, "rb") as fh:
            bars = pickle.load(fh)
        if not bars:
            continue
        rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
                 "low": b["low"], "close": b["close"]} for b in bars]
        df5 = pd.DataFrame(rows).set_index("timestamp").sort_index()
        frames.append(df5)

    if frames:
        df_pkl = pd.concat(frames).sort_index()
        df_pkl = df_pkl[~df_pkl.index.duplicated(keep="first")]
        # Normalize to UTC (some pkl bars may have mixed tz)
        df_pkl.index = pd.DatetimeIndex([
            t.tz_convert("UTC") if hasattr(t, "tz_convert") and t.tzinfo
            else pd.Timestamp(t).tz_localize("UTC")
            for t in df_pkl.index
        ])
        print(f"  5-min bars from Databento: {len(df_pkl):,}  ({df_pkl.index[0]} to {df_pkl.index[-1]})")

        # Resample 5min → 1hr
        ohlc_pkl = df_pkl.resample("1h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last"
        }).dropna(subset=["open"])
        print(f"  1hr bars from Databento: {len(ohlc_pkl):,}")

        # Only keep Databento bars after MarketTick ends
        cutoff = ohlc_mt.index[-1]
        ohlc_pkl_new = ohlc_pkl[ohlc_pkl.index > cutoff]
        print(f"  1hr bars after cutoff: {len(ohlc_pkl_new):,}")

        df = pd.concat([ohlc_mt, ohlc_pkl_new]).sort_index()
    else:
        df = ohlc_mt

    df = df[~df.index.duplicated(keep="first")]
    print(f"  Combined 1hr bars: {len(df):,}  ({df.index[0]} to {df.index[-1]})")
    return df


def run_backtest(df, atr_sl):
    """Run backtest with given ATR SL multiplier. Returns results dict."""
    trades = []
    trade_dates = []
    exit_types = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0

    for i in range(len(df)):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M")
        ins = SS <= hm < SE

        if pos != 0 and not ins and hm >= SE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE
            trades.append(pnl); trade_dates.append(td)
            exit_types.append("session")
            pos = 0

        if not ins:
            continue

        if td != cd:
            cd = td; dt = 0

        atr_v = row["atr"]
        if atr_v <= 0:
            continue

        z = row["z_vol"]; cl = row["close"]; ema = row["ema"]

        if pos != 0:
            xp = None; xt = None

            if z < LOW_Z:
                xp = cl; xt = "low_z"
            elif d == "long":
                if row["low"] <= sl:
                    xp = sl; xt = "sl"
                elif row["high"] >= tp:
                    xp = tp; xt = "tp"
            else:
                if row["high"] >= sl:
                    xp = sl; xt = "sl"
                elif row["low"] <= tp:
                    xp = tp; xt = "tp"

            if xp is not None:
                pnl = ((xp - ep) if d == "long" else (ep - xp)) * POINT_VALUE
                trades.append(pnl); trade_dates.append(td)
                exit_types.append(xt)
                pos = 0
                continue

        if pos == 0 and dt < MT:
            if z > HIGH_Z and cl > ema:
                pos = 1; d = "long"; ep = cl
                sl = cl - atr_sl * atr_v; tp = cl + ATR_TP * atr_v
                dt += 1
            elif z > HIGH_Z and cl < ema:
                pos = -1; d = "short"; ep = cl
                sl = cl + atr_sl * atr_v; tp = cl - ATR_TP * atr_v
                dt += 1

    pnls = np.array(trades)
    return pnls, trade_dates, exit_types


def print_results(pnls, trade_dates, exit_types, atr_sl):
    w = pnls[pnls > 0]; l = pnls[pnls < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else 99
    wr = 100 * len(w) / len(pnls) if len(pnls) else 0
    cum = pnls.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min() if len(cum) else 0

    print(f"\n{'='*80}")
    print(f"ROUGH VOL — 1HR BARS — SL={atr_sl}x ATR  TP={ATR_TP}x ATR")
    print(f"{'='*80}")
    print(f"H={H}  kernel={KERNEL_LEN}  norm={NORM_LEN}  z_look={Z_LOOKBACK}  high_z={HIGH_Z}  low_z={LOW_Z}")
    print(f"ema={EMA_LEN}  session={SS}-{SE} ET")
    print(f"{'='*80}")
    print(f"Trades: {len(pnls):,}  PF: {pf:.2f}  WR: {wr:.1f}%")
    print(f"PnL: ${pnls.sum():+,.0f}  Max DD: ${dd:,.0f}")
    if len(w):
        print(f"Avg Win: ${w.mean():+,.0f}  Avg Loss: ${l.mean():+,.0f}")
        print(f"Win/Loss ratio: {w.mean()/abs(l.mean()):.2f}")

    # Exit breakdown
    print(f"\nExit Breakdown:")
    print(f"{'Type':<10} {'Count':>6} {'%':>6} {'Avg PnL':>10} {'Win%':>6}")
    print("-" * 42)
    et_arr = np.array(exit_types)
    for etype in ["sl", "tp", "low_z", "session"]:
        mask = et_arr == etype
        if mask.sum() == 0:
            continue
        ep_sub = pnls[mask]
        ew = ep_sub[ep_sub > 0]
        print(f"{etype:<10} {mask.sum():>6} {100*mask.sum()/len(pnls):>5.1f}% ${ep_sub.mean():>+9,.0f} {100*len(ew)/len(ep_sub):>5.1f}%")

    # Yearly breakdown
    print(f"\n{'Year':<6} {'Trades':>7} {'PF':>6} {'WR':>6} {'PnL':>12} {'DD':>10}")
    print("-" * 55)
    for year in sorted(set(d.year for d in trade_dates)):
        mask = np.array([d.year == year for d in trade_dates])
        yp = pnls[mask]
        yw = yp[yp > 0]; yl = yp[yp < 0]
        ypf = yw.sum() / abs(yl.sum()) if len(yl) else 99
        ywr = 100 * len(yw) / len(yp) if len(yp) else 0
        ycum = yp.cumsum()
        ydd = (ycum - np.maximum.accumulate(ycum)).min() if len(ycum) else 0
        print(f"{year:<6} {len(yp):>7} {ypf:>6.2f} {ywr:>5.1f}% ${yp.sum():>+10,.0f} ${ydd:>9,.0f}")


def main():
    df = build_1hr_bars()

    # Convert to ET
    df.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df.index
    ])

    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)
    print(f"\nComputing signals on {n:,} 1hr bars...", flush=True)

    # ── Rough vol model ─────────────────────────────────────────────────────
    ret = np.log(closes / closes.shift(1))
    ret.iloc[0] = 0.0
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

    tr = pd.concat([
        highs - lows,
        (highs - closes.shift()).abs(),
        (lows - closes.shift()).abs()
    ], axis=1).max(axis=1)
    atr_vals = np.zeros(n)
    atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN:
            atr_vals[i] = tr.iloc[:i + 1].mean()
        else:
            atr_vals[i] = (atr_vals[i - 1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    # ── Sweep SL multipliers ────────────────────────────────────────────────
    for atr_sl in [3.0, 3.5, 4.0]:
        pnls, trade_dates, exit_types = run_backtest(df, atr_sl)
        print_results(pnls, trade_dates, exit_types, atr_sl)

    # ── Monthly breakdown ───────────────────────────────────────────────────
    print(f"\n{'Month':<8} {'Trades':>7} {'PF':>6} {'WR':>6} {'PnL':>12}")
    print("-" * 45)
    for ym in sorted(set(f"{d.year}-{d.month:02d}" for d in trade_dates)):
        y, m = int(ym[:4]), int(ym[5:])
        mask = np.array([d.year == y and d.month == m for d in trade_dates])
        mp = pnls[mask]
        mw = mp[mp > 0]; ml = mp[mp < 0]
        mpf = mw.sum() / abs(ml.sum()) if len(ml) else 99
        mwr = 100 * len(mw) / len(mp) if len(mp) else 0
        print(f"{ym:<8} {len(mp):>7} {mpf:>6.2f} {mwr:>5.1f}% ${mp.sum():>+10,.0f}")


if __name__ == "__main__":
    main()

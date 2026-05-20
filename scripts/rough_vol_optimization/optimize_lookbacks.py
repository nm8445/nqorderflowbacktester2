"""
Optimize NORM_LEN and Z_LOOKBACK for the rough vol strategy.
Grid: NORM_LEN 200-500 step 50, Z_LOOKBACK 100-300 step 25.
IS/OOS 50/50 split by trading days.
Pine-matched calcs: ddof=0, RMA ATR, z_vol=0 when v_std=0.
HIGH_Z=1.0, LOW_Z=-1.0 (mean-revert entries when z < -1.0).
Session 9:30-16:00, max 5 trades/day, marti streak=2 mult=1.5x reset=session.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
H = 0.10; KERNEL_LEN = 100; ETA = 1.0; V0 = 0.0001
HIGH_Z = 1.0; LOW_Z = 1.0  # stored positive, compared as z < -LOW_Z
EMA_LEN = 50; ATR_LEN = 14; ATR_SL = 2.0; ATR_TP = 3.0
ET = "America/New_York"; POINT_VALUE = 20.0
MART_STREAK = 2; MART_MULT = 1.5; MART_MAX_DOUBLES = 4
SESSION_START = "09:30"; SESSION_END = "16:00"
MAX_TRADES = 5


def build_bars():
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
        agg = df5.groupby("group").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"))
        agg.index = agg.index + pd.Timedelta(minutes=15)
        frames.append(agg)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Convert to ET once
    df.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df.index
    ])

    # Precompute EMA (doesn't change with norm_len/z_lookback)
    df["ema"] = df["close"].ewm(span=EMA_LEN, adjust=False).mean()

    # Precompute RMA ATR
    closes = df["close"]; highs = df["high"]; lows = df["low"]
    n = len(closes)
    tr = pd.concat([highs - lows, (highs - closes.shift()).abs(),
                    (lows - closes.shift()).abs()], axis=1).max(axis=1)
    atr_vals = np.zeros(n)
    atr_vals[0] = tr.iloc[0]
    for i in range(1, n):
        if i < ATR_LEN:
            atr_vals[i] = tr.iloc[:i+1].mean()
        else:
            atr_vals[i] = (atr_vals[i-1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    return df


def compute_zvol(df, norm_len, z_lookback):
    closes = df["close"]
    n = len(closes)
    ret = np.log(closes / closes.shift(1))
    ret.iloc[0] = 0.0
    mean_ret = ret.rolling(norm_len).mean()
    std_ret = ret.rolling(norm_len).std(ddof=0)
    shock = np.nan_to_num(np.where(std_ret > 0, (ret - mean_ret) / std_ret, 0.0), nan=0.0)

    ks = np.arange(1, KERNEL_LEN + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = ks ** (H - 0.5) - (ks - 1) ** (H - 0.5)
    kernel[0] = 1.0
    xH = np.convolve(shock, kernel, mode="full")[:n]
    v_model = np.minimum(V0 * np.exp(ETA * xH), V0 * 1e4)
    v_s = pd.Series(v_model, index=df.index)
    v_m = v_s.rolling(z_lookback).mean()
    v_d = v_s.rolling(z_lookback).std(ddof=0)
    return np.nan_to_num(np.where(v_d > 0, (v_s - v_m) / v_d, 0.0), nan=0.0)


def backtest(df, z_vol, date_set):
    """Run backtest on bars whose date is in date_set."""
    trades = []
    position = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    direction = ""
    qty = 1
    loss_streak = 0
    current_date = None
    daily_trades = 0

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    emas = df["ema"].values
    atrs = df["atr"].values
    indices = df.index

    for i in range(len(df)):
        td = indices[i].date()
        if td not in date_set:
            continue
        hm = indices[i].strftime("%H:%M")
        in_session = SESSION_START <= hm < SESSION_END

        # Session end
        if position != 0 and not in_session and hm >= SESSION_END:
            if direction == "long":
                pnl = (closes[i] - entry_price) * POINT_VALUE * qty
            else:
                pnl = (entry_price - closes[i]) * POINT_VALUE * qty
            trades.append(pnl)
            if pnl > 0:
                loss_streak = 0
            else:
                loss_streak += 1
            position = 0

        if not in_session:
            continue

        if td != current_date:
            current_date = td
            daily_trades = 0
            loss_streak = 0

        if loss_streak >= MART_STREAK:
            steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
            qty = max(1, round(MART_MULT ** steps))
        else:
            qty = 1

        atr = atrs[i]
        if atr <= 0:
            continue
        z = z_vol[i]
        cl = closes[i]
        ema = emas[i]

        # Manage position
        if position != 0:
            ep = None
            if direction == "long":
                if lows[i] <= sl_price:
                    ep = sl_price
                elif highs[i] >= tp_price:
                    ep = tp_price
            else:
                if highs[i] >= sl_price:
                    ep = sl_price
                elif lows[i] <= tp_price:
                    ep = tp_price
            if ep is not None:
                if direction == "long":
                    pnl = (ep - entry_price) * POINT_VALUE * qty
                else:
                    pnl = (entry_price - ep) * POINT_VALUE * qty
                trades.append(pnl)
                if pnl > 0:
                    loss_streak = 0
                else:
                    loss_streak += 1
                if loss_streak >= MART_STREAK:
                    steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                    qty = max(1, round(MART_MULT ** steps))
                else:
                    qty = 1
                position = 0
                continue

        # Entry
        if position == 0 and daily_trades < MAX_TRADES:
            entered = False
            # High-vol trend
            if z > HIGH_Z and cl > ema:
                position = 1; direction = "long"; entry_price = cl
                sl_price = cl - ATR_SL * atr; tp_price = cl + ATR_TP * atr
                entered = True
            elif z > HIGH_Z and cl < ema:
                position = -1; direction = "short"; entry_price = cl
                sl_price = cl + ATR_SL * atr; tp_price = cl - ATR_TP * atr
                entered = True
            # Mean-revert in low vol
            if not entered and z < -LOW_Z:
                if cl < ema:
                    position = 1; direction = "long"; entry_price = cl
                    sl_price = cl - ATR_SL * atr; tp_price = cl + ATR_TP * atr
                    entered = True
                elif cl > ema:
                    position = -1; direction = "short"; entry_price = cl
                    sl_price = cl + ATR_SL * atr; tp_price = cl - ATR_TP * atr
                    entered = True
            if entered:
                daily_trades += 1

    return np.array(trades) if trades else np.array([0.0])


def metrics(pnls):
    if len(pnls) == 0 or (len(pnls) == 1 and pnls[0] == 0):
        return 0, 0, 0, 0
    w = pnls[pnls > 0]
    l = pnls[pnls < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else 99
    wr = 100 * len(w) / len(pnls)
    cum = pnls.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()
    return len(pnls), pf, pnls.sum(), dd


def main():
    print("Building 15-min bars...")
    df = build_bars()
    n = len(df)
    print(f"Total bars: {n}")

    # Get all unique trading days (within session)
    session_mask = df.index.strftime("%H:%M").to_series(index=df.index)
    session_bars = df[(session_mask >= SESSION_START).values & (session_mask < SESSION_END).values]
    all_dates = sorted(session_bars.index.date)
    unique_dates = sorted(set(all_dates))
    split = len(unique_dates) // 2
    is_dates = set(unique_dates[:split])
    oos_dates = set(unique_dates[split:])
    print(f"Trading days: {len(unique_dates)}")
    print(f"IS:  {min(is_dates)} to {max(is_dates)} ({len(is_dates)} days)")
    print(f"OOS: {min(oos_dates)} to {max(oos_dates)} ({len(oos_dates)} days)")
    print()

    # Grid
    norm_lens = list(range(200, 501, 50))    # 200, 250, 300, 350, 400, 450, 500
    z_lookbacks = list(range(100, 301, 25))  # 100, 125, 150, 175, 200, 225, 250, 275, 300

    print(f"Grid: NORM_LEN {norm_lens}")
    print(f"Grid: Z_LOOKBACK {z_lookbacks}")
    print(f"Total combos: {len(norm_lens) * len(z_lookbacks)}")
    print()

    # Header
    print(f"{'NORM':>5} {'ZLOOK':>6} | {'IS_n':>5} {'IS_PF':>6} {'IS_PnL':>10} {'IS_DD':>10} | {'OOS_n':>5} {'OOS_PF':>6} {'OOS_PnL':>10} {'OOS_DD':>10} | {'ALL_PF':>6}")
    print("-" * 100)

    results = []

    for norm_len in norm_lens:
        for z_look in z_lookbacks:
            z_vol = compute_zvol(df, norm_len, z_look)

            is_pnls = backtest(df, z_vol, is_dates)
            oos_pnls = backtest(df, z_vol, oos_dates)
            all_pnls = np.concatenate([is_pnls, oos_pnls])

            is_n, is_pf, is_total, is_dd = metrics(is_pnls)
            oos_n, oos_pf, oos_total, oos_dd = metrics(oos_pnls)
            _, all_pf, _, _ = metrics(all_pnls)

            results.append({
                "norm_len": norm_len, "z_lookback": z_look,
                "is_n": is_n, "is_pf": is_pf, "is_pnl": is_total, "is_dd": is_dd,
                "oos_n": oos_n, "oos_pf": oos_pf, "oos_pnl": oos_total, "oos_dd": oos_dd,
                "all_pf": all_pf,
            })

            print(f"{norm_len:>5} {z_look:>6} | {is_n:>5} {is_pf:>6.2f} ${is_total:>+9,.0f} ${is_dd:>+9,.0f} | {oos_n:>5} {oos_pf:>6.2f} ${oos_total:>+9,.0f} ${oos_dd:>+9,.0f} | {all_pf:>6.2f}")

    # Sort by OOS PF
    print()
    print("=" * 100)
    print("TOP 10 BY OOS PF (minimum 30 OOS trades)")
    print("=" * 100)
    filtered = [r for r in results if r["oos_n"] >= 30]
    top_oos = sorted(filtered, key=lambda x: x["oos_pf"], reverse=True)[:10]
    print(f"{'NORM':>5} {'ZLOOK':>6} | {'IS_PF':>6} {'IS_PnL':>10} | {'OOS_PF':>6} {'OOS_PnL':>10} {'OOS_DD':>10} | {'ALL_PF':>6}")
    print("-" * 80)
    for r in top_oos:
        print(f"{r['norm_len']:>5} {r['z_lookback']:>6} | {r['is_pf']:>6.2f} ${r['is_pnl']:>+9,.0f} | {r['oos_pf']:>6.2f} ${r['oos_pnl']:>+9,.0f} ${r['oos_dd']:>+9,.0f} | {r['all_pf']:>6.2f}")

    # Sort by combined IS+OOS stability (smallest PF gap)
    print()
    print("=" * 100)
    print("TOP 10 MOST STABLE (smallest |IS_PF - OOS_PF|, both > 1.0)")
    print("=" * 100)
    stable = [r for r in results if r["is_pf"] > 1.0 and r["oos_pf"] > 1.0 and r["oos_n"] >= 30]
    stable.sort(key=lambda x: abs(x["is_pf"] - x["oos_pf"]))
    print(f"{'NORM':>5} {'ZLOOK':>6} | {'IS_PF':>6} {'IS_PnL':>10} | {'OOS_PF':>6} {'OOS_PnL':>10} | {'Gap':>6} {'ALL_PF':>6}")
    print("-" * 80)
    for r in stable[:10]:
        gap = abs(r["is_pf"] - r["oos_pf"])
        print(f"{r['norm_len']:>5} {r['z_lookback']:>6} | {r['is_pf']:>6.2f} ${r['is_pnl']:>+9,.0f} | {r['oos_pf']:>6.2f} ${r['oos_pnl']:>+9,.0f} | {gap:>6.3f} {r['all_pf']:>6.2f}")


if __name__ == "__main__":
    main()

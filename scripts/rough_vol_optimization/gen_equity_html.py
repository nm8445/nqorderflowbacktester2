"""
Generate equity curve HTML for optimized rough vol strategy.
Config: H=0.40, KERNEL=80, NORM=250, ZLOOK=100, HIGH_Z=1.5, EMA=50,
        SL=1.5x, TP=2.5x ATR(14), Session 09:30-15:30 (force close 15:27),
        Martingale streak=1, mult=3.0, max_doubles=4, daily reset.
Toggles: Martingale on/off, Hurst regime filter on/off.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import json
import time

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")

H=0.40; KERNEL_LEN=80; ETA=1.0; V0=0.0001
NORM_LEN=250; Z_LOOKBACK=100; HIGH_Z=1.5; LOW_Z=-1.0; EMA_LEN=50; ATR_LEN=14
ATR_SL=1.5; ATR_TP=2.5; ET="America/New_York"; POINT_VALUE=20.0
SS="09:30"; SE="15:30"; FORCE_CLOSE="15:27"; MT=5
MART_STREAK=1; MART_MULT=3.0; MART_MAX_DOUBLES=4

# Hurst regime filter params (best from grid search)
HURST_WINDOW=40; HURST_MAX=0.55

OUT_DIR = Path("C:/trading/nqorderflowbacktester/results/html")


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


def hurst_rs(x, min_chunk=8):
    """R/S Hurst estimator on 1D array."""
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
    """Compute H(R/S) once per trading day using trailing window of session 15-min returns."""
    session_mask = (df.index.strftime("%H:%M") >= SS) & (df.index.strftime("%H:%M") < SE)
    session_df = df[session_mask]
    log_ret = np.log(session_df["close"] / session_df["close"].shift(1)).dropna()
    dates = sorted(set(idx.date() for idx in log_ret.index))
    ret_by_date = {}
    for d in dates:
        mask = log_ret.index.date == d
        ret_by_date[d] = log_ret[mask].values
    daily_h = {}
    for i, d in enumerate(dates):
        start = max(0, i - window_days + 1)
        window_dates = dates[start:i+1]
        all_rets = np.concatenate([ret_by_date[wd] for wd in window_dates])
        if len(all_rets) >= 100:
            daily_h[d] = hurst_rs(all_rets)
        else:
            daily_h[d] = np.nan
    return daily_h


def run_backtest(df, use_martingale=True, daily_hurst=None, hurst_max=99.0):
    """Run backtest. If daily_hurst provided and hurst_max < 99, suppress entries when H > hurst_max."""
    trades = []
    pos = 0; ep = 0; sl = 0; tp = 0; d = ""; cd = None; dt = 0
    qty = 1; loss_streak = 0; ttype = ""
    for i in range(len(df)):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M"); ins = SS <= hm < SE
        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((row["close"] - ep) if d == "long" else (ep - row["close"])) * POINT_VALUE * qty
            trades.append({"pnl": round(pnl, 2), "reason": "CLOSE", "dir": d,
                           "date": str(td), "time": idx.strftime("%Y-%m-%d %H:%M"), "qty": qty, "src": ttype})
            if pnl > 0: loss_streak = 0
            else: loss_streak += 1
            pos = 0
            continue
        if not ins: continue
        if td != cd: cd = td; dt = 0; loss_streak = 0
        if use_martingale and loss_streak >= MART_STREAK:
            steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
            qty = max(1, round(MART_MULT ** steps))
        else: qty = 1
        atr = row["atr"]
        if atr <= 0: continue
        z = row["z_vol"]; cl = row["close"]; ema = row["ema"]
        if pos != 0:
            xp = None; xr = None
            if d == "long":
                if row["low"] <= sl: xp, xr = sl, "SL"
                elif row["high"] >= tp: xp, xr = tp, "TP"
            else:
                if row["high"] >= sl: xp, xr = sl, "SL"
                elif row["low"] <= tp: xp, xr = tp, "TP"
            if xp:
                pnl = ((xp - ep) if d == "long" else (ep - xp)) * POINT_VALUE * qty
                trades.append({"pnl": round(pnl, 2), "reason": xr, "dir": d,
                               "date": str(td), "time": idx.strftime("%Y-%m-%d %H:%M"), "qty": qty, "src": ttype})
                if pnl > 0: loss_streak = 0
                else: loss_streak += 1
                if use_martingale and loss_streak >= MART_STREAK:
                    steps = min(loss_streak - MART_STREAK + 1, MART_MAX_DOUBLES)
                    qty = max(1, round(MART_MULT ** steps))
                else: qty = 1
                pos = 0; continue
        if pos == 0 and dt < MT:
            # Hurst regime filter
            if daily_hurst is not None and hurst_max < 90:
                cur_h = daily_hurst.get(td, np.nan)
                if not np.isnan(cur_h) and cur_h > hurst_max:
                    continue

            if z > HIGH_Z and cl > ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr; tp = cl + ATR_TP * atr; dt += 1; ttype = "trend"
            elif z > HIGH_Z and cl < ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr; tp = cl - ATR_TP * atr; dt += 1; ttype = "trend"
            elif z < LOW_Z and cl < ema:
                pos = 1; d = "long"; ep = cl; sl = cl - ATR_SL * atr; tp = cl + ATR_TP * atr; dt += 1; ttype = "mr"
            elif z < LOW_Z and cl > ema:
                pos = -1; d = "short"; ep = cl; sl = cl + ATR_SL * atr; tp = cl - ATR_TP * atr; dt += 1; ttype = "mr"
    return trades


def compute_stats(trades, is_cutoff):
    """Compute all stats from a trade list."""
    if not trades:
        # Return empty stats
        return _empty_stats()

    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    total_pnl = pnls.sum()
    pf = wins.sum() / abs(losses.sum()) if len(losses) else 0
    wr = 100 * len(wins) / len(pnls) if len(pnls) else 0
    cum = pnls.cumsum()
    dd_arr = cum - np.maximum.accumulate(cum)
    max_dd = dd_arr.min()
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    risk_adj = abs(total_pnl / max_dd) if max_dd != 0 else 0

    daily_pnl = {}
    for t in trades:
        daily_pnl[t["date"]] = daily_pnl.get(t["date"], 0) + t["pnl"]
    daily_vals = list(daily_pnl.values())
    sharpe = (np.mean(daily_vals) / np.std(daily_vals)) * np.sqrt(252) if len(daily_vals) > 1 and np.std(daily_vals) > 0 else 0

    long_trades = [t for t in trades if t["dir"] == "long"]
    short_trades = [t for t in trades if t["dir"] == "short"]
    long_pnl = sum(t["pnl"] for t in long_trades)
    short_pnl = sum(t["pnl"] for t in short_trades)
    long_wr = 100 * sum(1 for t in long_trades if t["pnl"] > 0) / len(long_trades) if long_trades else 0
    short_wr = 100 * sum(1 for t in short_trades if t["pnl"] > 0) / len(short_trades) if short_trades else 0

    trend_tr = [t for t in trades if t["src"] == "trend"]
    mr_tr = [t for t in trades if t["src"] == "mr"]
    trend_pnl_total = sum(t["pnl"] for t in trend_tr)
    mr_pnl_total = sum(t["pnl"] for t in mr_tr)
    trend_wr = 100 * sum(1 for t in trend_tr if t["pnl"] > 0) / len(trend_tr) if trend_tr else 0
    mr_wr = 100 * sum(1 for t in mr_tr if t["pnl"] > 0) / len(mr_tr) if mr_tr else 0

    reasons = Counter(t["reason"] for t in trades)
    tp_pnl = sum(t["pnl"] for t in trades if t["reason"] == "TP")
    sl_pnl = sum(t["pnl"] for t in trades if t["reason"] == "SL")
    close_pnl = sum(t["pnl"] for t in trades if t["reason"] == "CLOSE")

    is_trades = [t for t in trades if t["date"] < is_cutoff]
    oos_trades = [t for t in trades if t["date"] >= is_cutoff]
    is_pnls = np.array([t["pnl"] for t in is_trades]) if is_trades else np.array([0.0])
    oos_pnls = np.array([t["pnl"] for t in oos_trades]) if oos_trades else np.array([0.0])
    is_pf = is_pnls[is_pnls > 0].sum() / abs(is_pnls[is_pnls < 0].sum()) if len(is_pnls[is_pnls < 0]) else 0
    oos_pf = oos_pnls[oos_pnls > 0].sum() / abs(oos_pnls[oos_pnls < 0].sum()) if len(oos_pnls[oos_pnls < 0]) else 0
    is_wr = 100 * (is_pnls > 0).sum() / len(is_pnls) if len(is_pnls) else 0
    oos_wr = 100 * (oos_pnls > 0).sum() / len(oos_pnls) if len(oos_pnls) else 0
    is_date_min = min(t["date"] for t in is_trades) if is_trades else "N/A"
    is_date_max = max(t["date"] for t in is_trades) if is_trades else "N/A"
    oos_date_min = min(t["date"] for t in oos_trades) if oos_trades else "N/A"
    oos_date_max = max(t["date"] for t in oos_trades) if oos_trades else "N/A"

    monthly = {}
    for t in trades:
        m = t["date"][:7]
        monthly[m] = monthly.get(m, 0) + t["pnl"]
    m_labels = sorted(monthly.keys())
    m_values = [round(monthly[m]) for m in m_labels]

    trend_cum = []; mr_cum = []
    t_run = 0; m_run = 0
    for t in trades:
        if t["src"] == "trend": t_run += t["pnl"]
        else: m_run += t["pnl"]
        trend_cum.append(round(t_run, 2))
        mr_cum.append(round(m_run, 2))

    return {
        "pnls": pnls, "total_pnl": total_pnl, "pf": pf, "wr": wr,
        "cum": cum, "dd_arr": dd_arr, "max_dd": max_dd,
        "avg_win": avg_win, "avg_loss": avg_loss, "risk_adj": risk_adj, "sharpe": sharpe,
        "long_trades": len(long_trades), "short_trades": len(short_trades),
        "long_pnl": long_pnl, "short_pnl": short_pnl, "long_wr": long_wr, "short_wr": short_wr,
        "trend_tr": len(trend_tr), "mr_tr": len(mr_tr),
        "trend_pnl": trend_pnl_total, "mr_pnl": mr_pnl_total, "trend_wr": trend_wr, "mr_wr": mr_wr,
        "reasons": dict(reasons), "tp_pnl": tp_pnl, "sl_pnl": sl_pnl, "close_pnl": close_pnl,
        "is_trades": len(is_trades), "oos_trades": len(oos_trades),
        "is_pnl": float(is_pnls.sum()), "oos_pnl": float(oos_pnls.sum()),
        "is_pf": is_pf, "oos_pf": oos_pf, "is_wr": is_wr, "oos_wr": oos_wr,
        "is_date_min": is_date_min, "is_date_max": is_date_max,
        "oos_date_min": oos_date_min, "oos_date_max": oos_date_max,
        "m_labels": m_labels, "m_values": m_values,
        "trend_cum": trend_cum, "mr_cum": mr_cum,
        "dates": [t["time"] for t in trades],
        "sp_idx": len(is_trades),
    }


def stats_to_js(s, prefix):
    """Convert stats dict to JS variable declarations."""
    lines = []
    lines.append(f"var {prefix}_dates = {json.dumps(s['dates'])};")
    lines.append(f"var {prefix}_equity = {json.dumps([round(x, 2) for x in s['cum'].tolist()])};")
    lines.append(f"var {prefix}_dd = {json.dumps([round(x, 2) for x in s['dd_arr'].tolist()])};")
    lines.append(f"var {prefix}_trendCum = {json.dumps(s['trend_cum'])};")
    lines.append(f"var {prefix}_mrCum = {json.dumps(s['mr_cum'])};")
    lines.append(f"var {prefix}_mL = {json.dumps(s['m_labels'])};")
    lines.append(f"var {prefix}_mV = {json.dumps(s['m_values'])};")
    lines.append(f"var {prefix}_sp = {s['sp_idx']};")

    card_data = {
        "total_pnl": round(s["total_pnl"]), "n_trades": len(s["pnls"]),
        "wr": round(s["wr"], 1), "pf": round(s["pf"], 2),
        "max_dd": round(abs(s["max_dd"])), "sharpe": round(s["sharpe"], 2),
        "avg_win": round(s["avg_win"]), "avg_loss": round(s["avg_loss"]),
        "avg_trade": round(float(s["pnls"].mean())) if len(s["pnls"]) else 0,
        "risk_adj": round(s["risk_adj"], 2),
        "long_n": s["long_trades"], "short_n": s["short_trades"],
        "long_wr": round(s["long_wr"], 1), "short_wr": round(s["short_wr"], 1),
        "long_pnl": round(s["long_pnl"]), "short_pnl": round(s["short_pnl"]),
        "trend_n": s["trend_tr"], "mr_n": s["mr_tr"],
        "trend_pnl": round(s["trend_pnl"]), "mr_pnl": round(s["mr_pnl"]),
        "trend_wr": round(s["trend_wr"], 1), "mr_wr": round(s["mr_wr"], 1),
        "tp_n": s["reasons"].get("TP", 0), "sl_n": s["reasons"].get("SL", 0), "close_n": s["reasons"].get("CLOSE", 0),
        "tp_pnl": round(s["tp_pnl"]), "sl_pnl": round(s["sl_pnl"]), "close_pnl": round(s["close_pnl"]),
        "is_n": s["is_trades"], "oos_n": s["oos_trades"],
        "is_pnl": round(s["is_pnl"]), "oos_pnl": round(s["oos_pnl"]),
        "is_pf": round(s["is_pf"], 2), "oos_pf": round(s["oos_pf"], 2),
        "is_wr": round(s["is_wr"], 1), "oos_wr": round(s["oos_wr"], 1),
        "is_date_min": s["is_date_min"], "is_date_max": s["is_date_max"],
        "oos_date_min": s["oos_date_min"], "oos_date_max": s["oos_date_max"],
    }
    lines.append(f"var {prefix}_stats = {json.dumps(card_data)};")
    return "\n".join(lines)


def main():
    df = build_combined_bars()
    closes = df["close"]; highs = df["high"]; lows = df["low"]; n = len(closes)

    # Signals
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
        if i < ATR_LEN: atr_vals[i] = tr.iloc[:i+1].mean()
        else: atr_vals[i] = (atr_vals[i-1] * (ATR_LEN - 1) + tr.iloc[i]) / ATR_LEN
    df["atr"] = atr_vals

    # 60/40 IS/OOS split
    trading_days = sorted(set(idx.date() for idx in df.index))
    split_idx = int(len(trading_days) * 0.60)
    is_cutoff = str(trading_days[split_idx])

    # Compute daily Hurst
    print(f"Computing daily Hurst (window={HURST_WINDOW})...", flush=True)
    t0 = time.time()
    daily_hurst = compute_daily_hurst(df, HURST_WINDOW)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)

    # Run 4 backtest combos: mart x hurst
    combos = [
        ("mo",  True,  None,         99.0,       "Mart ON"),
        ("mf",  False, None,         99.0,       "Mart OFF"),
        ("moh", True,  daily_hurst,  HURST_MAX,  f"Mart ON + Hurst<{HURST_MAX}"),
        ("mfh", False, daily_hurst,  HURST_MAX,  f"Mart OFF + Hurst<{HURST_MAX}"),
    ]

    all_stats = {}
    for prefix, mart, dh, hmax, label in combos:
        print(f"Running: {label}...", flush=True)
        trades = run_backtest(df, use_martingale=mart, daily_hurst=dh, hurst_max=hmax)
        print(f"  -> {len(trades)} trades", flush=True)
        all_stats[prefix] = compute_stats(trades, is_cutoff)

    # Build JS data for all 4
    js_blocks = []
    for prefix, _, _, _, _ in combos:
        js_blocks.append(stats_to_js(all_stats[prefix], prefix))
    js_data = "\n".join(js_blocks)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Rough Vol Strategy -- Equity Curve (Optimized)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; background: #1e1e1e; color: #e0e0e0; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #4ec9b0; text-align: center; }}
        .subtitle {{ text-align: center; color: #a0a0a0; margin-bottom: 20px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #2d2d2d; padding: 15px; border-radius: 8px; border-left: 4px solid #4ec9b0; }}
        .stat-label {{ color: #999; font-size: 12px; text-transform: uppercase; }}
        .stat-value {{ color: #fff; font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .positive {{ color: #4ec9b0; }}
        .negative {{ color: #f48771; }}
        .highlight {{ background: #2d4a3e; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #4ec9b0; }}
        .toggle-row {{ text-align: center; margin: 10px 0; }}
        .toggle-btn {{ display: inline-block; padding: 10px 24px; border: 2px solid #4ec9b0; border-radius: 6px;
                       cursor: pointer; font-size: 14px; font-weight: bold; transition: all 0.2s; margin: 0 5px; }}
        .toggle-btn.active {{ background: #4ec9b0; color: #1e1e1e; }}
        .toggle-btn:not(.active) {{ background: transparent; color: #4ec9b0; }}
        .toggle-btn:hover:not(.active) {{ background: rgba(78,201,176,0.15); }}
        .toggle-btn.hurst {{ border-color: #c586c0; }}
        .toggle-btn.hurst.active {{ background: #c586c0; color: #1e1e1e; }}
        .toggle-btn.hurst:not(.active) {{ color: #c586c0; }}
        .toggle-btn.hurst:hover:not(.active) {{ background: rgba(197,134,192,0.15); }}
    </style>
</head>
<body>
<div class="container">
    <h1>Rough Vol Strategy (Optimized)</h1>
    <div class="subtitle">H={H} | NORM={NORM_LEN} | ZLOOK={Z_LOOKBACK} | HIGH_Z={HIGH_Z} / LOW_Z={LOW_Z} | EMA={EMA_LEN} | SL {ATR_SL}x / TP {ATR_TP}x ATR | {SS}-{SE} ET (close {FORCE_CLOSE})</div>

    <div class="toggle-row">
        <span class="toggle-btn active" id="btn-mart-on" onclick="setMart(true)">Martingale ON</span>
        <span class="toggle-btn" id="btn-mart-off" onclick="setMart(false)">Martingale OFF</span>
        <span style="display:inline-block;width:30px;"></span>
        <span class="toggle-btn hurst" id="btn-hurst-off" onclick="setHurst(false)">No Hurst Filter</span>
        <span class="toggle-btn hurst active" id="btn-hurst-on" onclick="setHurst(true)">Hurst Filter (H&lt;{HURST_MAX})</span>
    </div>
    <div style="color:#666;font-size:12px;margin-top:4px;text-align:center;" id="mode-label">Martingale s={MART_STREAK} m={MART_MULT}x d={MART_MAX_DOUBLES} daily | Hurst regime: {HURST_WINDOW}-day R/S, suppress when H &gt; {HURST_MAX}</div>

    <div class="stats" id="top-stats"></div>
    <div class="stats" id="avg-stats"></div>
    <div class="highlight" id="strat-breakdown"></div>
    <div class="highlight" id="dir-breakdown"></div>
    <div class="highlight" id="exit-breakdown"></div>
    <div class="highlight" id="is-oos"></div>
    <div id="equity-chart"></div>
    <div id="dd-chart"></div>
    <div id="monthly-chart"></div>

    <div style="margin-top:30px;padding:20px;background:#2d2d2d;border-radius:8px;">
        <h3 style="color:#4ec9b0;">Strategy Configuration (Phase 1-5 Optimized)</h3>
        <ul style="line-height:1.8;">
            <li><strong>Signal:</strong> Rough Vol z-score (H={H}, Kernel={KERNEL_LEN}, ETA={ETA}, V0={V0})</li>
            <li><strong>Lookbacks:</strong> NORM_LEN={NORM_LEN}, Z_LOOKBACK={Z_LOOKBACK}</li>
            <li><strong>Entry:</strong> Trend (z &gt; {HIGH_Z}, with-trend) + Mean-Revert (z &lt; {LOW_Z}, counter-trend) | EMA({EMA_LEN}) filter</li>
            <li><strong>Risk:</strong> SL {ATR_SL}x / TP {ATR_TP}x ATR({ATR_LEN}) RMA</li>
            <li><strong>Session:</strong> {SS} - {SE} ET | Force close {FORCE_CLOSE} | Max {MT} trades/day</li>
            <li><strong>Martingale:</strong> streak={MART_STREAK}, mult={MART_MULT}x, max_doubles={MART_MAX_DOUBLES}, daily reset</li>
            <li><strong>Hurst filter:</strong> {HURST_WINDOW}-day rolling R/S Hurst, suppress entries when H &gt; {HURST_MAX}</li>
            <li><strong>Calcs:</strong> Population std (ddof=0), RMA ATR (Pine-matched)</li>
        </ul>
    </div>
</div>

<script>
{js_data}

var useMart = true;
var useHurst = true;

function getPrefix() {{
    if (useMart && !useHurst) return 'mo';
    if (!useMart && !useHurst) return 'mf';
    if (useMart && useHurst) return 'moh';
    return 'mfh';
}}

function setMart(on) {{
    useMart = on;
    document.getElementById('btn-mart-on').className = 'toggle-btn' + (on?' active':'');
    document.getElementById('btn-mart-off').className = 'toggle-btn' + (!on?' active':'');
    updateLabel();
    render();
}}

function setHurst(on) {{
    useHurst = on;
    document.getElementById('btn-hurst-on').className = 'toggle-btn hurst' + (on?' active':'');
    document.getElementById('btn-hurst-off').className = 'toggle-btn hurst' + (!on?' active':'');
    updateLabel();
    render();
}}

function updateLabel() {{
    var parts = [];
    if (useMart) parts.push('Martingale s={MART_STREAK} m={MART_MULT}x d={MART_MAX_DOUBLES} daily');
    else parts.push('Flat sizing (qty=1)');
    if (useHurst) parts.push('Hurst regime: {HURST_WINDOW}-day R/S, suppress when H > {HURST_MAX}');
    else parts.push('No regime filter');
    document.getElementById('mode-label').textContent = parts.join(' | ');
}}

function fmt(v) {{ return '$' + Math.abs(v).toLocaleString('en-US', {{maximumFractionDigits:0}}); }}
function fmtS(v) {{ return (v>=0?'':'-') + '$' + Math.abs(v).toLocaleString('en-US', {{maximumFractionDigits:0}}); }}
function cls(v) {{ return v >= 0 ? 'positive' : 'negative'; }}
function pct(n, total) {{ return total > 0 ? (100*n/total).toFixed(1) : '0.0'; }}

function D(name) {{ var p = getPrefix(); return window[p + '_' + name]; }}

function render() {{
    var s = D('stats');
    var dates = D('dates');
    var equity = D('equity');
    var dd = D('dd');
    var tC = D('trendCum');
    var mC = D('mrCum');
    var mL = D('mL');
    var mV = D('mV');
    var sp = D('sp');

    document.getElementById('top-stats').innerHTML =
        '<div class="stat-card"><div class="stat-label">Total P&L</div><div class="stat-value '+cls(s.total_pnl)+'">'+fmtS(s.total_pnl)+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">Total Trades</div><div class="stat-value">'+s.n_trades+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value '+cls(s.wr-50)+'">'+s.wr.toFixed(1)+'%</div></div>'+
        '<div class="stat-card"><div class="stat-label">Profit Factor</div><div class="stat-value '+cls(s.pf-1)+'">'+s.pf.toFixed(2)+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">Max Drawdown</div><div class="stat-value negative">'+fmt(s.max_dd)+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">Sharpe Ratio</div><div class="stat-value '+cls(s.sharpe)+'">'+s.sharpe.toFixed(2)+'</div></div>';

    document.getElementById('avg-stats').innerHTML =
        '<div class="stat-card"><div class="stat-label">Avg Winner</div><div class="stat-value positive">'+fmt(s.avg_win)+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">Avg Loser</div><div class="stat-value negative">'+fmtS(s.avg_loss)+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">Avg P&L/Trade</div><div class="stat-value '+cls(s.avg_trade)+'">'+fmtS(s.avg_trade)+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">Risk-Adjusted Return</div><div class="stat-value '+cls(s.risk_adj)+'">'+s.risk_adj.toFixed(2)+'</div></div>';

    document.getElementById('strat-breakdown').innerHTML =
        '<h3 style="margin-top:0;color:#4ec9b0;">Strategy Breakdown</h3>'+
        '<div class="stats" style="margin-top:15px;">'+
        '<div class="stat-card" style="border-left-color:#4ec9b0;"><div class="stat-label">Trend (z &gt; {HIGH_Z})</div><div class="stat-value '+cls(s.trend_pnl)+'">'+fmtS(s.trend_pnl)+'</div><div style="color:#999;font-size:12px;margin-top:5px;">'+s.trend_n+' trades | WR: '+s.trend_wr.toFixed(1)+'%</div></div>'+
        '<div class="stat-card" style="border-left-color:#dcdcaa;"><div class="stat-label">Mean-Revert (z &lt; {LOW_Z})</div><div class="stat-value '+cls(s.mr_pnl)+'">'+fmtS(s.mr_pnl)+'</div><div style="color:#999;font-size:12px;margin-top:5px;">'+s.mr_n+' trades | WR: '+s.mr_wr.toFixed(1)+'%</div></div>'+
        '</div>';

    document.getElementById('dir-breakdown').innerHTML =
        '<h3 style="margin-top:0;color:#4ec9b0;">Direction Breakdown</h3>'+
        '<div class="stats" style="margin-top:15px;">'+
        '<div class="stat-card"><div class="stat-label">Longs</div><div class="stat-value">'+s.long_n+'</div><div style="color:#999;font-size:12px;margin-top:5px;">WR: '+s.long_wr.toFixed(1)+'% | P&L: '+fmtS(s.long_pnl)+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">Shorts</div><div class="stat-value">'+s.short_n+'</div><div style="color:#999;font-size:12px;margin-top:5px;">WR: '+s.short_wr.toFixed(1)+'% | P&L: '+fmtS(s.short_pnl)+'</div></div>'+
        '</div>';

    document.getElementById('exit-breakdown').innerHTML =
        '<h3 style="margin-top:0;color:#4ec9b0;">Exit Breakdown</h3>'+
        '<div class="stats" style="margin-top:15px;">'+
        '<div class="stat-card"><div class="stat-label">Target Hits</div><div class="stat-value positive">'+s.tp_n+' ('+pct(s.tp_n,s.n_trades)+'%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: '+fmtS(s.tp_pnl)+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">Stop Losses</div><div class="stat-value negative">'+s.sl_n+' ('+pct(s.sl_n,s.n_trades)+'%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: '+fmtS(s.sl_pnl)+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">Force Close (15:27)</div><div class="stat-value">'+s.close_n+' ('+pct(s.close_n,s.n_trades)+'%)</div><div style="color:#999;font-size:12px;margin-top:5px;">P&L: '+fmtS(s.close_pnl)+'</div></div>'+
        '</div>';

    document.getElementById('is-oos').innerHTML =
        '<h3 style="margin-top:0;color:#4ec9b0;">In-Sample / Out-of-Sample (60/40 by trading days)</h3>'+
        '<div class="stats" style="margin-top:15px;">'+
        '<div class="stat-card"><div class="stat-label">IS ('+s.is_date_min+' to '+s.is_date_max+')</div><div class="stat-value '+cls(s.is_pnl)+'">'+fmtS(s.is_pnl)+'</div><div style="color:#999;font-size:12px;margin-top:5px;">'+s.is_n+' trades | WR '+s.is_wr.toFixed(1)+'% | PF '+s.is_pf.toFixed(2)+'</div></div>'+
        '<div class="stat-card"><div class="stat-label">OOS ('+s.oos_date_min+' to '+s.oos_date_max+')</div><div class="stat-value '+cls(s.oos_pnl)+'">'+fmtS(s.oos_pnl)+'</div><div style="color:#999;font-size:12px;margin-top:5px;">'+s.oos_n+' trades | WR '+s.oos_wr.toFixed(1)+'% | PF '+s.oos_pf.toFixed(2)+'</div></div>'+
        '</div>';

    var titleSuffix = (useMart?'Mart ON':'Mart OFF') + (useHurst?' + Hurst Filter':'');
    Plotly.newPlot('equity-chart', [
        {{ x: dates, y: equity, type:'scatter', mode:'lines', name:'Combined', line:{{color:'#4ec9b0',width:3}}, fill:'tozeroy', fillcolor:'rgba(78,201,176,0.1)' }},
        {{ x: dates, y: tC, type:'scatter', mode:'lines', name:'Trend (z>{HIGH_Z})', line:{{color:'#569cd6',width:2,dash:'dot'}} }},
        {{ x: dates, y: mC, type:'scatter', mode:'lines', name:'Mean-Revert (z<{LOW_Z})', line:{{color:'#dcdcaa',width:2,dash:'dot'}} }}
    ], {{
        title:{{text:'Cumulative P&L -- '+titleSuffix,font:{{color:'#e0e0e0',size:16}}}},
        xaxis:{{gridcolor:'#333',color:'#999'}}, yaxis:{{title:'P&L ($)',gridcolor:'#333',color:'#999',tickformat:'$,.0f'}},
        plot_bgcolor:'#1e1e1e', paper_bgcolor:'#1e1e1e', font:{{color:'#e0e0e0'}}, hovermode:'x unified', height:600,
        shapes:[{{type:'line',x0:dates[sp],x1:dates[sp],y0:0,y1:1,yref:'paper',line:{{color:'#f48771',width:2,dash:'dash'}}}}],
        annotations:[{{x:dates[sp],y:1.05,yref:'paper',text:'IS | OOS',showarrow:false,font:{{color:'#f48771',size:12}}}}],
        legend:{{x:0.02,y:0.98,bgcolor:'rgba(45,45,45,0.8)',bordercolor:'#4ec9b0',borderwidth:1}}
    }}, {{responsive:true}});

    Plotly.newPlot('dd-chart', [{{x:dates,y:dd,type:'scatter',mode:'lines',name:'Drawdown',line:{{color:'#f48771',width:2}},fill:'tozeroy',fillcolor:'rgba(244,135,113,0.15)'}}],
        {{title:{{text:'Drawdown',font:{{color:'#e0e0e0',size:16}}}},xaxis:{{gridcolor:'#333',color:'#999'}},yaxis:{{title:'DD ($)',gridcolor:'#333',color:'#999',tickformat:'$,.0f'}},plot_bgcolor:'#1e1e1e',paper_bgcolor:'#1e1e1e',font:{{color:'#e0e0e0'}},hovermode:'x unified',height:350}},{{responsive:true}});

    var mc = mV.map(v=>v>=0?'#4ec9b0':'#f48771');
    Plotly.newPlot('monthly-chart', [{{x:mL,y:mV,type:'bar',marker:{{color:mc}},text:mV.map(v=>'$'+v.toLocaleString('en-US',{{maximumFractionDigits:0}})),textposition:'outside',textfont:{{color:'#e0e0e0',size:11}}}}],
        {{title:{{text:'Monthly P&L',font:{{color:'#e0e0e0',size:16}}}},xaxis:{{gridcolor:'#333',color:'#999'}},yaxis:{{title:'P&L ($)',gridcolor:'#333',color:'#999',tickformat:'$,.0f'}},plot_bgcolor:'#1e1e1e',paper_bgcolor:'#1e1e1e',font:{{color:'#e0e0e0'}},height:400}},{{responsive:true}});
}}

render();
</script>
</body>
</html>"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "roughvol_equity.html"
    with open(out_path, "w") as f:
        f.write(html)

    print(f"\nSaved to {out_path}")
    print(f"{'Mode':<30} {'Trades':>6} {'PF':>6} {'PnL':>12} {'MaxDD':>12}")
    print("-" * 70)
    for prefix, _, _, _, label in combos:
        s = all_stats[prefix]
        print(f"{label:<30} {len(s['pnls']):>6} {s['pf']:>6.2f} ${s['total_pnl']:>+11,.0f} ${s['max_dd']:>+11,.0f}")


if __name__ == "__main__":
    main()

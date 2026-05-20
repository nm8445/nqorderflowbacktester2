"""Band-break entry with VWAP ± ATR-cushion trailing stop, 15-min candles.

Stop placement:
  - LONG:  stop = VWAP - (ATR_14 * multiplier)
  - SHORT: stop = VWAP + (ATR_14 * multiplier)

ATR_14: classic 14-day ATR on daily bars. Computed once per day from prior 14
trading days.

Sweep multipliers: 0.5, 1.0, 1.5, 2.0
Re-entry: allowed (same rules as 5-min/15-min VWAP-stop scripts).
TP: 17:00 ET close.
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

NQ_1MIN     = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
BAND_PARQ   = Path(__file__).parent / "overnight_band_per_day.parquet"
OUT_DIR     = Path(__file__).parent
TRADES_OUT  = OUT_DIR / "vwap_atr_trades_15min.parquet"

BAR_FREQ = "15min"
MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]


def load_bars() -> tuple[dict, pd.DataFrame]:
    """Returns (bars_by_date dict, daily_atr14 series indexed by date)."""
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    df = df.sort_index()

    # Daily OHLC for ATR (using full session, not just RTH, to be safe)
    daily = df.resample("D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    daily["prev_close"] = daily["close"].shift(1)
    daily["tr"] = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - daily["prev_close"]).abs(),
        (daily["low"] - daily["prev_close"]).abs(),
    ], axis=1).max(axis=1)
    daily["atr14"] = daily["tr"].rolling(14).mean().shift(1)  # shift to use prior 14 days
    atr_by_date = daily["atr14"]
    atr_by_date.index = atr_by_date.index.date

    # 15-min RTH bars
    rth = (df.index.time >= dt.time(9, 30)) & (df.index.time <= dt.time(17, 0))
    rth_df = df[rth]
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    bars = rth_df.resample(BAR_FREQ, origin="epoch").agg(agg).dropna(subset=["close"])
    bars = bars[(bars.index.time >= dt.time(9, 30)) & (bars.index.time <= dt.time(17, 0))]
    bars["typical"] = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    bars["pv"] = bars["typical"] * bars["volume"]
    return {d: g for d, g in bars.groupby(bars.index.date)}, atr_by_date


def simulate_day(bars: pd.DataFrame, upper: float, lower: float,
                 atr: float, mult: float) -> list[dict]:
    trades = []
    pos = 0
    entry_t = None; entry_px = None; entry_idx = None
    cum_pv = 0.0; cum_vol = 0.0
    bars_list = list(bars.iterrows())

    for i, (t, b) in enumerate(bars_list):
        cum_pv += b["pv"]; cum_vol += b["volume"]
        vwap = cum_pv / cum_vol if cum_vol > 0 else b["close"]
        close = b["close"]
        is_last = (i == len(bars_list) - 1) or t.time() == dt.time(17, 0)

        # Stops with ATR cushion
        long_stop = vwap - mult * atr
        short_stop = vwap + mult * atr

        # 1) Stop check (skip same-bar)
        if pos != 0 and i > entry_idx:
            stopped = (pos > 0 and close < long_stop) or (pos < 0 and close > short_stop)
            if stopped:
                trades.append({
                    "entry_time": entry_t, "exit_time": t,
                    "direction": "long" if pos > 0 else "short",
                    "entry_price": entry_px, "exit_price": close,
                    "exit_reason": "atr_stop",
                })
                pos = 0; entry_t = None; entry_px = None

        # 2) New entry from flat
        if pos == 0 and not is_last:
            if close > upper:
                pos = 1; entry_t = t; entry_px = close; entry_idx = i
            elif close < lower:
                pos = -1; entry_t = t; entry_px = close; entry_idx = i

        # 3) Force close on last bar
        if is_last and pos != 0:
            trades.append({
                "entry_time": entry_t, "exit_time": t,
                "direction": "long" if pos > 0 else "short",
                "entry_price": entry_px, "exit_price": close,
                "exit_reason": "eod_close",
            })
            pos = 0

    for tr in trades:
        if tr["direction"] == "long":
            tr["pnl_pts"] = tr["exit_price"] - tr["entry_price"]
        else:
            tr["pnl_pts"] = tr["entry_price"] - tr["exit_price"]
    return trades


def main():
    print(f"loading {BAR_FREQ} RTH bars + daily ATR(14)...")
    bars_by_date, atr_by_date = load_bars()
    print(f"  bar-days: {len(bars_by_date)}")
    print(f"  atr days: {len(atr_by_date.dropna())}")
    band = pd.read_parquet(BAND_PARQ)
    band["date"] = pd.to_datetime(band["date"]).dt.date
    print(f"  band-days: {len(band)}\n")

    rows = []
    t0 = time.time()
    for mult in MULTIPLIERS:
        print(f"=== multiplier = {mult} ===")
        cnt = 0
        for r in band.itertuples(index=False):
            d = r.date
            bars = bars_by_date.get(d)
            atr = atr_by_date.get(d)
            if bars is None or len(bars) < 5: continue
            if atr is None or not np.isfinite(atr): continue
            upper = float(r.upper_band_14d) if np.isfinite(r.upper_band_14d) else np.nan
            lower = float(r.lower_band_14d) if np.isfinite(r.lower_band_14d) else np.nan
            if not (np.isfinite(upper) and np.isfinite(lower)): continue
            for tr in simulate_day(bars, upper, lower, float(atr), mult):
                tr["date"] = d
                tr["multiplier"] = mult
                tr["atr14"] = float(atr)
                tr["qqq_regime_open"] = r.qqq_regime_open
                tr["ndx_regime_open"] = r.ndx_regime_open
                tr["upper_band"] = upper
                tr["lower_band"] = lower
                rows.append(tr)
                cnt += 1
        print(f"  trades for mult={mult}: {cnt}  (elapsed {time.time()-t0:.0f}s)")

    tr = pd.DataFrame(rows)
    tr.to_parquet(TRADES_OUT, compression="zstd", index=False)
    print(f"\nwrote {TRADES_OUT}  ({len(tr)} total trades)\n")

    n_years = (tr["date"].max() - tr["date"].min()).days / 365.25
    print(f"sample span: {tr['date'].min()} -> {tr['date'].max()} ({n_years:.2f} years)\n")

    def report(label, sub: pd.DataFrame):
        n = len(sub)
        if n < 2:
            print(f"  {label:<55}  n={n}"); return
        pnl = sub["pnl_pts"].dropna().values
        n = len(pnl)
        p_profit = float((pnl > 0).mean())
        m = float(pnl.mean()); total = float(pnl.sum())
        per_yr = n / n_years
        eod = (sub["exit_reason"] == "eod_close").mean()
        t, p = ss.ttest_1samp(pnl, 0)
        sig = "  ***" if p<0.001 else ("   **" if p<0.01 else ("    *" if p<0.05 else "     "))
        print(f"  {label:<55}  n={n:>5}  ({per_yr:>4.0f}/yr)  P(profit)={p_profit:.1%}  "
              f"mean={m:+6.2f} pts  total={total:+8.0f}  eod%={eod:.0%}  t={t:+5.2f}{sig}")

    print("=" * 110)
    print("MULTIPLIER COMPARISON — overall")
    print("=" * 110)
    for mult in MULTIPLIERS:
        sub = tr[tr["multiplier"] == mult]
        print(f"\n--- multiplier = {mult} ---")
        report(f"  ALL trades", sub)
        report(f"  LONG only", sub[sub["direction"]=="long"])
        report(f"  SHORT only", sub[sub["direction"]=="short"])
        for reason, sub2 in sub.groupby("exit_reason"):
            report(f"    exit = {reason}", sub2)

    print()
    print("=" * 110)
    print("MULTIPLIER × GAMMA REGIME (long, NDX-derived)")
    print("=" * 110)
    for mult in MULTIPLIERS:
        sub = tr[(tr["multiplier"] == mult) & (tr["direction"] == "long")]
        print(f"\nmult = {mult}:")
        for regime in ["pos", "neg"]:
            report(f"  long + NDX_{regime}", sub[sub["ndx_regime_open"] == regime])

    print()
    print("=" * 110)
    print("MULTIPLIER × GAMMA REGIME (short, QQQ-derived)")
    print("=" * 110)
    for mult in MULTIPLIERS:
        sub = tr[(tr["multiplier"] == mult) & (tr["direction"] == "short")]
        print(f"\nmult = {mult}:")
        for regime in ["pos", "neg"]:
            report(f"  short + QQQ_{regime}", sub[sub["qqq_regime_open"] == regime])

    # Best multiplier — entries by time
    # Pick the multiplier with highest total PnL
    best_mult = tr.groupby("multiplier")["pnl_pts"].sum().idxmax()
    print(f"\n\n" + "=" * 110)
    print(f"BEST MULTIPLIER (by total PnL) = {best_mult}")
    print("=" * 110)
    best_tr = tr[tr["multiplier"] == best_mult].copy()
    best_tr["entry_hour_bucket"] = best_tr["entry_time"].apply(
        lambda t: pd.Timestamp(t).strftime("%H:%M")
    )
    best_tr["entry_t_min"] = best_tr["entry_time"].apply(
        lambda t: pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute
    )
    print(f"\nALL DIRECTIONS — 30-min entry buckets:")
    starts = list(range(9*60+30, 17*60, 30))
    for s in starts:
        e = s + 30
        sub = best_tr[(best_tr["entry_t_min"] >= s) & (best_tr["entry_t_min"] < e)]
        sh, sm = divmod(s, 60); eh, em = divmod(e, 60)
        report(f"  {sh:02d}:{sm:02d}-{eh:02d}:{em:02d}", sub)

    for direction in ["long", "short"]:
        print(f"\n{direction.upper()} ONLY — 30-min entry buckets:")
        df_d = best_tr[best_tr["direction"] == direction]
        for s in starts:
            e = s + 30
            sub = df_d[(df_d["entry_t_min"] >= s) & (df_d["entry_t_min"] < e)]
            sh, sm = divmod(s, 60); eh, em = divmod(e, 60)
            report(f"  {sh:02d}:{sm:02d}-{eh:02d}:{em:02d}", sub)


if __name__ == "__main__":
    sys.exit(main())

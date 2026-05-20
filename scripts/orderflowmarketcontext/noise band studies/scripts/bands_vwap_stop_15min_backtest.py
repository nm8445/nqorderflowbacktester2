"""Band-break entry with VWAP trailing stop and re-entry — 15-min candles.

Same logic as bands_vwap_stop_backtest.py but on 15-min closes instead of
5-min. Bands are 14-day overnight σ. RTH session 9:30-17:00 ET.

Trade rules:
  - On each 15-min bar:
      * Update VWAP cumulatively
      * If FLAT and close > upper_band -> open LONG at close
      * If FLAT and close < lower_band -> open SHORT at close
      * If LONG and close < VWAP -> close at close (skip same-bar)
      * If SHORT and close > VWAP -> close at close (skip same-bar)
  - Last bar (17:00): force-close any open position.
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
TRADES_OUT  = OUT_DIR / "vwap_stop_trades_15min.parquet"

BAR_FREQ = "15min"


def load_bars() -> dict:
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    df = df.sort_index()
    rth = (df.index.time >= dt.time(9, 30)) & (df.index.time <= dt.time(17, 0))
    df = df[rth]
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    bars = df.resample(BAR_FREQ, origin="epoch").agg(agg).dropna(subset=["close"])
    bars = bars[(bars.index.time >= dt.time(9, 30)) & (bars.index.time <= dt.time(17, 0))]
    bars["typical"] = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    bars["pv"] = bars["typical"] * bars["volume"]
    return {d: g for d, g in bars.groupby(bars.index.date)}


def simulate_day(bars: pd.DataFrame, upper: float, lower: float) -> list[dict]:
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

        # 1) Stop check (skip same-bar)
        if pos != 0 and i > entry_idx:
            stopped = (pos > 0 and close < vwap) or (pos < 0 and close > vwap)
            if stopped:
                trades.append({
                    "entry_time": entry_t, "exit_time": t,
                    "direction": "long" if pos > 0 else "short",
                    "entry_price": entry_px, "exit_price": close,
                    "exit_reason": "vwap_stop",
                    "vwap_at_exit": vwap,
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
                "vwap_at_exit": vwap,
            })
            pos = 0

    for tr in trades:
        if tr["direction"] == "long":
            tr["pnl_pts"] = tr["exit_price"] - tr["entry_price"]
        else:
            tr["pnl_pts"] = tr["entry_price"] - tr["exit_price"]
    return trades


def main():
    print(f"loading {BAR_FREQ} RTH bars...")
    bars_by_date = load_bars()
    print(f"  bar-days: {len(bars_by_date)}")
    band = pd.read_parquet(BAND_PARQ)
    band["date"] = pd.to_datetime(band["date"]).dt.date
    print(f"  band-days: {len(band)}\n")

    rows = []
    t0 = time.time()
    for i, r in enumerate(band.itertuples(index=False), 1):
        d = r.date
        bars = bars_by_date.get(d)
        if bars is None or len(bars) < 5: continue
        upper = float(r.upper_band_14d) if np.isfinite(r.upper_band_14d) else np.nan
        lower = float(r.lower_band_14d) if np.isfinite(r.lower_band_14d) else np.nan
        if not (np.isfinite(upper) and np.isfinite(lower)): continue
        for tr in simulate_day(bars, upper, lower):
            tr["date"] = d
            tr["qqq_regime_open"] = r.qqq_regime_open
            tr["ndx_regime_open"] = r.ndx_regime_open
            tr["upper_band"] = upper
            tr["lower_band"] = lower
            rows.append(tr)
        if i % 200 == 0:
            print(f"  {i}/{len(band)}  elapsed={time.time()-t0:.0f}s  trades={len(rows)}")

    tr = pd.DataFrame(rows)
    tr.to_parquet(TRADES_OUT, compression="zstd", index=False)
    print(f"\nwrote {TRADES_OUT}  ({len(tr)} trades)\n")

    n_years = (tr["date"].max() - tr["date"].min()).days / 365.25
    print(f"sample span: {tr['date'].min()} -> {tr['date'].max()} ({n_years:.2f} years)")

    def report(label, sub: pd.DataFrame):
        n = len(sub)
        if n < 2:
            print(f"  {label:<55}  n={n}"); return
        pnl = sub["pnl_pts"].dropna().values
        n = len(pnl)
        p_profit = float((pnl > 0).mean())
        m = float(pnl.mean())
        total = float(pnl.sum())
        eod_close_rate = (sub["exit_reason"] == "eod_close").mean() if "exit_reason" in sub.columns else float("nan")
        per_yr = n / n_years
        t, p = ss.ttest_1samp(pnl, 0)
        sig = "  ***" if p<0.001 else ("   **" if p<0.01 else ("    *" if p<0.05 else "     "))
        print(f"  {label:<55}  n={n:>5}  ({per_yr:>5.0f}/yr)  P(profit)={p_profit:.1%}  "
              f"mean={m:+6.2f} pts  total={total:+8.0f}  eod%={eod_close_rate:.0%}  "
              f"t={t:+5.2f}  p={p:.4f}{sig}")

    print()
    print("=" * 110)
    print(f"OVERALL ({BAR_FREQ}) — VWAP trailing stop + re-entry, TP at 17:00")
    print("=" * 110)
    report("ALL trades (long + short, all gamma)", tr)
    report("LONG only", tr[tr["direction"]=="long"])
    report("SHORT only", tr[tr["direction"]=="short"])
    print()
    print("Exit-reason breakdown:")
    for reason, sub in tr.groupby("exit_reason"):
        report(f"  exit = {reason}", sub)
    print()

    print("=" * 110)
    print("BY GAMMA REGIME")
    print("=" * 110)
    for source, col in [("QQQ", "qqq_regime_open"), ("NDX", "ndx_regime_open")]:
        print(f"\n--- {source}-derived regime ---")
        for direction in ["long", "short"]:
            df_d = tr[tr["direction"] == direction]
            print(f"  {direction.upper()}:")
            for regime in ["pos", "neg"]:
                sub = df_d[df_d[col] == regime]
                report(f"    {direction:<5} + {source}_{regime}-gamma", sub)


if __name__ == "__main__":
    sys.exit(main())

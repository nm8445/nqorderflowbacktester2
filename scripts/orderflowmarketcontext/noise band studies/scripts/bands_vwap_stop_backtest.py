"""Band-break entry with VWAP trailing stop and re-entry allowed.

Trade rules:
  - Each day, build static 14-day overnight noise bands (upper / lower).
  - Walk RTH 5-min bars 9:30 to 17:00 ET.
  - On each bar:
      * Update VWAP cumulatively (typical_price * volume / volume from 9:30).
      * If FLAT: if 5-min close > upper_band -> open LONG at close.
                 if 5-min close < lower_band -> open SHORT at close.
      * If LONG: if 5-min close < VWAP -> CLOSE LONG at close. (skip same-bar)
                 (VWAP itself acts as a trailing stop since it moves with price.)
      * If SHORT: if 5-min close > VWAP -> CLOSE SHORT at close. (skip same-bar)
  - At 17:00 (last RTH bar): close any open position at the bar close.
  - Re-entry allowed: after stop-out, can re-enter on next band-break signal.

Per trade: entry_time, exit_time, direction, entry_price, exit_price,
           pnl_pts, exit_reason (vwap_stop / eod_close), gamma_regime.

Backtests per gamma cohort (QQQ + NDX, 14d lookback only).
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
TRADES_OUT  = OUT_DIR / "vwap_stop_trades.parquet"


def load_5min_bars() -> dict:
    """Returns {date: DataFrame of 5-min RTH bars (open/high/low/close/volume) in ET}."""
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    df = df.sort_index()
    rth = (df.index.time >= dt.time(9, 30)) & (df.index.time <= dt.time(17, 0))
    df = df[rth]
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    bars = df.resample("5min", origin="epoch").agg(agg).dropna(subset=["close"])
    bars = bars[(bars.index.time >= dt.time(9, 30)) & (bars.index.time <= dt.time(17, 0))]
    bars["typical"] = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    bars["pv"] = bars["typical"] * bars["volume"]
    return {d: g for d, g in bars.groupby(bars.index.date)}


def simulate_day(bars: pd.DataFrame, upper: float, lower: float) -> list[dict]:
    """Returns a list of trade dicts for this day."""
    trades = []
    pos = 0          # 0=flat, +1=long, -1=short
    entry_t = None; entry_px = None; entry_idx = None
    cum_pv = 0.0; cum_vol = 0.0

    for i, (t, b) in enumerate(bars.iterrows()):
        cum_pv += b["pv"]; cum_vol += b["volume"]
        vwap = cum_pv / cum_vol if cum_vol > 0 else b["close"]
        close = b["close"]

        # 1) Manage existing position (skip stop on the entry bar)
        if pos != 0 and i > entry_idx:
            stopped = False
            if pos > 0 and close < vwap: stopped = True
            elif pos < 0 and close > vwap: stopped = True
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
        if pos == 0:
            if close > upper:
                pos = 1; entry_t = t; entry_px = close; entry_idx = i
            elif close < lower:
                pos = -1; entry_t = t; entry_px = close; entry_idx = i

        # 3) Force close on the last bar of session (17:00)
        if t.time() == dt.time(17, 0) and pos != 0:
            trades.append({
                "entry_time": entry_t, "exit_time": t,
                "direction": "long" if pos > 0 else "short",
                "entry_price": entry_px, "exit_price": close,
                "exit_reason": "eod_close",
                "vwap_at_exit": vwap,
            })
            pos = 0

    # Compute PnL per trade
    for tr in trades:
        if tr["direction"] == "long":
            tr["pnl_pts"] = tr["exit_price"] - tr["entry_price"]
        else:
            tr["pnl_pts"] = tr["entry_price"] - tr["exit_price"]
    return trades


def main():
    print("loading 5-min RTH bars...")
    bars_by_date = load_5min_bars()
    print(f"  bar-days: {len(bars_by_date)}")
    band = pd.read_parquet(BAND_PARQ)
    band["date"] = pd.to_datetime(band["date"]).dt.date
    print(f"  band-days: {len(band)}")
    print()

    rows = []
    t0 = time.time()
    for i, r in enumerate(band.itertuples(index=False), 1):
        d = r.date
        bars = bars_by_date.get(d)
        if bars is None or len(bars) < 5: continue
        upper = float(r.upper_band_14d) if np.isfinite(r.upper_band_14d) else np.nan
        lower = float(r.lower_band_14d) if np.isfinite(r.lower_band_14d) else np.nan
        if not (np.isfinite(upper) and np.isfinite(lower)): continue
        day_trades = simulate_day(bars, upper, lower)
        for tr in day_trades:
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
    print(f"\nwrote {TRADES_OUT}  ({len(tr)} trades)")
    print()

    # ------------------------------ Reporting ------------------------------

    n_years = (tr["date"].max() - tr["date"].min()).days / 365.25
    print(f"sample span: {tr['date'].min()} -> {tr['date'].max()} ({n_years:.2f} years)")

    def report(label, sub: pd.DataFrame):
        n = len(sub)
        if n < 2:
            print(f"  {label:<55}  n={n}")
            return
        pnl = sub["pnl_pts"].dropna().values
        n = len(pnl)
        p_profit = float((pnl > 0).mean())
        m = float(pnl.mean())
        std = float(pnl.std())
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
    print("OVERALL — vanilla strategy with VWAP trailing stop + re-entry, TP at 17:00")
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
    print("BY GAMMA REGIME (open at 9:30)")
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

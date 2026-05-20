"""Filtered cohort stats: QQQ_neg-gamma, longs+shorts, entries 9:30-14:00,
mult=0.5 stop. Computes total trades, PF, Sharpe, PnL, max realized DD,
max intra-trade unrealized DD."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NQ_1MIN     = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
TRADES_PARQ = Path(__file__).parent / "vwap_atr_trades_15min.parquet"


def load_15min_bars():
    df = pd.read_parquet(NQ_1MIN)
    idx = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("America/New_York")
    rth = (df.index.time >= dt.time(9, 30)) & (df.index.time <= dt.time(17, 0))
    df = df[rth]
    bars = df.resample("15min", origin="epoch").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    return bars


def per_trade_max_unrealized_dd(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    """For each trade, compute the worst unrealized PnL during the trade
    using 15-min bar lows (long) or highs (short) between entry and exit."""
    out = []
    for r in trades.itertuples(index=False):
        entry_t = pd.Timestamp(r.entry_time); exit_t = pd.Timestamp(r.exit_time)
        if entry_t.tz is None: entry_t = entry_t.tz_localize("America/New_York")
        if exit_t.tz is None:  exit_t  = exit_t.tz_localize("America/New_York")
        seg = bars.loc[entry_t:exit_t]
        if seg.empty: out.append(0.0); continue
        if r.direction == "long":
            worst = float(seg["low"].min()) - r.entry_price  # negative if drawdown
        else:
            worst = r.entry_price - float(seg["high"].max())  # negative if drawdown
        out.append(worst)
    return pd.Series(out, index=trades.index)


def main():
    print("loading trades + 15-min bars...")
    tr = pd.read_parquet(TRADES_PARQ)
    bars = load_15min_bars()
    print(f"  total trades in parquet: {len(tr)}")

    # Filter: mult=0.5, QQQ_neg-gamma, entry 9:30-14:00
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    tr["exit_time"]  = pd.to_datetime(tr["exit_time"])
    tr["entry_hm"] = tr["entry_time"].dt.strftime("%H:%M")
    tr["entry_min"] = tr["entry_time"].dt.hour * 60 + tr["entry_time"].dt.minute

    f = (
        (tr["multiplier"] == 0.5) &
        (tr["qqq_regime_open"] == "neg") &
        (tr["entry_min"] >= 9*60 + 30) &
        (tr["entry_min"] <= 14*60)
    )
    sub = tr[f].copy().reset_index(drop=True)
    print(f"  filtered: {len(sub)} trades  (mult=0.5, QQQ_neg, entry 9:30-14:00)")
    print(f"  long: {(sub['direction']=='long').sum()}  short: {(sub['direction']=='short').sum()}")

    # Per-trade unrealized DD
    sub["unrealized_dd_pts"] = per_trade_max_unrealized_dd(sub, bars)
    # unrealized_dd_pts is negative if a drawdown happened during the trade
    # (bar low/high went against entry); positive only if price never moved against entry

    # Sort by entry_time for the equity curve
    sub = sub.sort_values("entry_time").reset_index(drop=True)
    sub["cum_pnl"] = sub["pnl_pts"].cumsum()
    sub["running_peak"] = sub["cum_pnl"].cummax()
    sub["realized_dd"] = sub["cum_pnl"] - sub["running_peak"]  # 0 or negative

    n = len(sub)
    pnl = sub["pnl_pts"]
    total = float(pnl.sum())

    # Profit factor
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe — daily PnL series
    sub["date"] = pd.to_datetime(sub["date"])
    daily = sub.groupby(sub["date"].dt.date)["pnl_pts"].sum()
    n_years = (sub["date"].max() - sub["date"].min()).days / 365.25
    annual_pnl = total / n_years
    daily_mean = daily.mean()
    daily_std = daily.std()
    sharpe_daily = daily_mean / daily_std if daily_std > 0 else float("nan")
    sharpe_ann = sharpe_daily * np.sqrt(252)

    # Max realized drawdown
    max_realized_dd = float(sub["realized_dd"].min())  # most negative
    # Max intra-trade unrealized DD (most negative excursion)
    max_unrealized_dd = float(sub["unrealized_dd_pts"].min())
    mean_unrealized_dd = float(sub.loc[sub["unrealized_dd_pts"]<0, "unrealized_dd_pts"].mean())

    # Per direction stats
    print()
    print("=" * 90)
    print("FILTERED COHORT STATS")
    print("  Filter: multiplier=0.5, QQQ_neg-gamma at open, entry between 9:30-14:00 ET")
    print("  Direction: both LONG and SHORT")
    print("=" * 90)
    print(f"  Sample span         : {sub['date'].min().date()} -> {sub['date'].max().date()}  ({n_years:.2f} yrs)")
    print(f"  Total trades        : {n}")
    print(f"  Trades / year       : {n / n_years:.0f}")
    print(f"  Hit rate            : {(pnl > 0).mean():.1%}")
    print(f"  Mean PnL / trade    : {pnl.mean():+.2f} pts")
    print(f"  Total PnL           : {total:+.0f} pts  (= ${total*20:+,.0f} per contract)")
    print(f"  Annualized PnL      : {annual_pnl:+.0f} pts/yr  (= ${annual_pnl*20:+,.0f}/yr)")
    print(f"  Profit Factor       : {pf:.3f}")
    print(f"  Sharpe (annualized) : {sharpe_ann:.2f}  (daily {sharpe_daily:.3f})")
    print(f"  Max realized DD     : {max_realized_dd:+.0f} pts  (= ${max_realized_dd*20:+,.0f})")
    print(f"  Max unrealized DD   : {max_unrealized_dd:+.0f} pts  (= ${max_unrealized_dd*20:+,.0f}) — worst excursion within a single trade")
    print(f"  Avg unrealized DD   : {mean_unrealized_dd:+.2f} pts  (only trades that ever drew down)")
    print(f"  Largest winner      : {pnl.max():+.2f} pts")
    print(f"  Largest loser       : {pnl.min():+.2f} pts")
    print()

    # Per direction
    print("--- by direction ---")
    for direction in ["long","short"]:
        d = sub[sub["direction"]==direction]
        if len(d) == 0: continue
        d_total = d["pnl_pts"].sum()
        d_pf = d.loc[d["pnl_pts"]>0,"pnl_pts"].sum() / max(1e-9, -d.loc[d["pnl_pts"]<0,"pnl_pts"].sum())
        print(f"  {direction:<6}  n={len(d):>4}  hit={(d['pnl_pts']>0).mean():.1%}  "
              f"mean={d['pnl_pts'].mean():+.2f}  total={d_total:+7.0f}  PF={d_pf:.2f}")

    # By exit reason
    print()
    print("--- by exit reason ---")
    for r, g in sub.groupby("exit_reason"):
        print(f"  {r:<14}  n={len(g):>4}  hit={(g['pnl_pts']>0).mean():.1%}  "
              f"mean={g['pnl_pts'].mean():+.2f}  total={g['pnl_pts'].sum():+7.0f}")


if __name__ == "__main__":
    sys.exit(main())

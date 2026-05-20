"""
Overnight Gap Mean Reversion Backtest — NQ 1-minute
===================================================

Strategy
--------
If Gap_pct > +0.30%  -> enter SHORT at close of 9:30-9:31 ET bar
If Gap_pct < -0.30%  -> enter LONG  at close of 9:30-9:31 ET bar
Gap_pct = (close of 9:30-9:31 bar on day t) / (close of 3:59-4:00 bar on day t-1) - 1
Position size: 1 contract.

Exit variants
  A1..A5 — Time exit at entry + {15, 30, 45, 60, 90} minutes
  B      — ATR SL 1.0x / TP 0.5x, hard cutoff 120 min
  C      — ATR SL 1.0x / TP 0.5x, hard cutoff 60  min

Costs (NQ: 1 point = $20, 1 tick = 0.25 pt = $5)
  Commission $0.85/side
  Exchange fees $1.40/side
  Slippage 0.125 pt (0.5 tick) adverse per side
  => $4.50 fixed + $5.00 slippage = ~$9.50 round-trip

Train/Test split: first 60% of date range = IS,  last 40% = OOS.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────
PARQUET = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT_DIR = Path(__file__).parent / "results"
ET = "America/New_York"

GAP_THRESHOLD = 0.0030   # 0.3%
TIME_EXITS_MIN = [15, 30, 45, 60, 90]
ATR_SL_MULT = 1.0
ATR_TP_MULT = 0.5
ATR_LEN = 14
B_CUTOFF_MIN = 120
C_CUTOFF_MIN = 60

POINT_VALUE = 20.0      # NQ: $ per point
TICK_SIZE = 0.25
SLIPPAGE_PTS = 0.125    # per side
COMMISSION_PER_SIDE = 0.85
FEES_PER_SIDE = 1.40
FIXED_COST_RT = 2 * (COMMISSION_PER_SIDE + FEES_PER_SIDE)   # $4.50

IS_FRACTION = 0.60
MARGIN_PER_CONTRACT = 25_000.0   # prop-firm context

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Load & prep data ─────────────────────────────────────────────────
def load_and_prep():
    print("[1] Loading 1-min parquet...", flush=True)
    df = pd.read_parquet(PARQUET)
    # Convert UTC -> ET for downstream trading-day semantics
    df.index = df.index.tz_convert(ET)
    df = df.sort_index()
    print(f"    rows={len(df):,}  range={df.index.min()} -> {df.index.max()}")

    # RTH filter: 9:30:00 <= t < 16:00:00 ET
    rth_mask = ((df.index.hour > 9) | ((df.index.hour == 9) & (df.index.minute >= 30))) \
               & (df.index.hour < 16)
    rth = df[rth_mask].copy()
    rth["et_date"] = rth.index.normalize()
    print(f"    RTH bars: {len(rth):,}")

    # Trading calendar
    cal = mcal.get_calendar("CME_Equity")
    start = str(df.index.min().date() - pd.Timedelta(days=5))
    end = str(df.index.max().date() + pd.Timedelta(days=5))
    sched = cal.schedule(start_date=start, end_date=end)
    sched["close_et"] = sched["market_close"].dt.tz_convert(ET)
    sched["open_et"]  = sched["market_open"].dt.tz_convert(ET)
    sched["is_early_close"] = sched["close_et"].dt.hour < 16
    trading_dates = pd.Index([d.date() for d in sched.index], name="date")
    print(f"    Calendar trading days in window: {len(trading_dates)}")

    # Post-holiday flag: True if PREVIOUS CALENDAR trading gap was > 1 day
    # (i.e. any non-trading day between the prior trading day and today) OR
    # the prior trading day was an early close (day-after-Thanksgiving etc.).
    post_holiday = pd.Series(False, index=trading_dates)
    prior_early = sched["is_early_close"].shift(1).fillna(False)
    prior_trading = pd.Series(trading_dates, index=trading_dates).shift(1)
    gap_days = (pd.to_datetime(trading_dates) - pd.to_datetime(prior_trading)).dt.days
    post_holiday = ((gap_days > 1) | prior_early.values).fillna(False)
    post_holiday.index = trading_dates

    # Daily OHLC from RTH bars + daily ATR
    daily = pd.DataFrame({
        "open":  rth.groupby("et_date")["open"].first(),
        "high":  rth.groupby("et_date")["high"].max(),
        "low":   rth.groupby("et_date")["low"].min(),
        "close": rth.groupby("et_date")["close"].last(),
        "bars":  rth.groupby("et_date").size(),
    })
    daily.index = daily.index.date
    daily.index.name = "date"

    # Restrict to real trading days and require enough bars to compute gap
    daily = daily.loc[daily.index.intersection(trading_dates)]
    # Drop days with <100 RTH bars (truly broken data — 2022-06-17)
    daily = daily[daily["bars"] >= 100]

    # True range using prior close
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    daily["atr"] = tr.rolling(ATR_LEN, min_periods=ATR_LEN).mean()
    daily["post_holiday"] = post_holiday.reindex(daily.index).fillna(False)
    daily["dow"] = pd.to_datetime(daily.index).day_name()

    print(f"    Usable trading days: {len(daily):,}")
    print(f"    Date span: {daily.index.min()} -> {daily.index.max()}")

    # Fast-lookup table for minute bars: {date: DataFrame of that day's RTH minutes}
    day_minutes = {d: g for d, g in rth.groupby("et_date")}
    day_minutes = {k.date(): v for k, v in day_minutes.items()}

    return df, rth, daily, day_minutes, sched


# ── Step 2: Gap table + spot-check ──────────────────────────────────────────
def build_gap_table(daily, day_minutes, verbose=True):
    print("[2] Computing gaps...", flush=True)
    # Open_930_t := close of the 9:30-9:31 bar on day t
    # Close_1600_{t-1} := close of the 15:59-16:00 bar on day t-1
    opens = {}
    closes = {}
    for d, g in day_minutes.items():
        bar_930 = g[(g.index.hour == 9) & (g.index.minute == 30)]
        bar_1559 = g[(g.index.hour == 15) & (g.index.minute == 59)]
        if not bar_930.empty:
            opens[d] = float(bar_930["close"].iloc[0])
        if not bar_1559.empty:
            closes[d] = float(bar_1559["close"].iloc[0])

    gap_rows = []
    sorted_dates = sorted(set(opens) & set(daily.index))
    prev_close_map = {}
    for i, d in enumerate(sorted_dates):
        if i == 0:
            continue
        prev_d = sorted_dates[i - 1]
        if prev_d not in closes:
            continue
        o = opens[d]
        c = closes[prev_d]
        gap_rows.append({
            "date": d,
            "prev_close": c,
            "open_930": o,
            "gap_pct": o / c - 1,
            "prev_date": prev_d,
        })
    gaps = pd.DataFrame(gap_rows).set_index("date")
    gaps = gaps.join(daily[["atr", "post_holiday", "dow", "bars"]], how="inner")

    if verbose:
        print(f"    Gap table: {len(gaps):,} days")
        print(f"    Gap_pct stats: mean={gaps['gap_pct'].mean():.4%}  "
              f"std={gaps['gap_pct'].std():.4%}  "
              f"p1={gaps['gap_pct'].quantile(0.01):.4%}  "
              f"p99={gaps['gap_pct'].quantile(0.99):.4%}")
        print(f"    |gap|>0.3% days: {(gaps['gap_pct'].abs() > GAP_THRESHOLD).sum()}")
        print()
        print("    Random spot-check (10 days):")
        sample = gaps.sample(10, random_state=42).sort_index()
        for d, r in sample.iterrows():
            print(f"      {d}  prev_close={r['prev_close']:.2f}  "
                  f"open_930={r['open_930']:.2f}  gap_pct={r['gap_pct']:+.4%}")
    return gaps


# ── Step 4: Simulate ─────────────────────────────────────────────────────────
@dataclass
class Trade:
    date: pd.Timestamp
    variant: str
    direction: str   # 'long' | 'short'
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gap_pct: float
    atr: float
    exit_reason: str
    post_holiday: bool
    dow: str
    pnl_gross: float   # $ per contract, no costs
    pnl_net: float     # $ per contract, after slippage + fixed costs


def apply_slippage(price: float, direction: str, side: str) -> float:
    """side in {'entry','exit'}.  Adverse slippage: worse fill."""
    if direction == "long":
        return price + SLIPPAGE_PTS if side == "entry" else price - SLIPPAGE_PTS
    else:  # short
        return price - SLIPPAGE_PTS if side == "entry" else price + SLIPPAGE_PTS


def simulate_day(date, day_bars, gap_row, variant, cutoff_min, use_atr, atr_val):
    """Returns a Trade or None for this date+variant."""
    gap = gap_row["gap_pct"]
    if abs(gap) <= GAP_THRESHOLD:
        return None

    direction = "short" if gap > 0 else "long"

    # Entry at 9:30-9:31 bar close; entry_time = 9:31 ET (close time of that bar)
    bar_930 = day_bars[(day_bars.index.hour == 9) & (day_bars.index.minute == 30)]
    if bar_930.empty:
        return None
    entry_bar_ts = bar_930.index[0]            # open time = 9:30
    entry_time   = entry_bar_ts + pd.Timedelta(minutes=1)  # close time = 9:31
    entry_price_raw = float(bar_930["close"].iloc[0])
    entry_price = apply_slippage(entry_price_raw, direction, "entry")

    # ATR-based SL/TP prices (in raw terms; slippage only on fills)
    if use_atr:
        if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
            return None
        if direction == "long":
            sl_px = entry_price_raw - ATR_SL_MULT * atr_val
            tp_px = entry_price_raw + ATR_TP_MULT * atr_val
        else:
            sl_px = entry_price_raw + ATR_SL_MULT * atr_val
            tp_px = entry_price_raw - ATR_TP_MULT * atr_val
    else:
        sl_px = tp_px = None

    # Walk post-entry bars up to the cutoff (cutoff is an end-of-bar time)
    post = day_bars[day_bars.index > entry_bar_ts]
    if post.empty:
        return None

    cutoff_time = entry_time + pd.Timedelta(minutes=cutoff_min - 1)
    # cutoff_min=15 means exit at entry_time+15min which is 9:46 close -> last bar open=9:45
    # We want to include that bar.

    exit_price_raw = None
    exit_time = None
    exit_reason = None

    for ts, row in post.iterrows():
        # ts is the OPEN of a minute bar; bar close is ts + 1 minute
        bar_close_time = ts + pd.Timedelta(minutes=1)
        high = row["high"]; low = row["low"]; close = row["close"]

        if use_atr:
            hit_sl = (low <= sl_px) if direction == "long" else (high >= sl_px)
            hit_tp = (high >= tp_px) if direction == "long" else (low <= tp_px)
            # Conservative: if both hit in the same bar, assume SL first
            if hit_sl and hit_tp:
                exit_price_raw = sl_px; exit_time = bar_close_time; exit_reason = "sl_conflict"; break
            if hit_sl:
                exit_price_raw = sl_px; exit_time = bar_close_time; exit_reason = "sl"; break
            if hit_tp:
                exit_price_raw = tp_px; exit_time = bar_close_time; exit_reason = "tp"; break

        # Time cutoff: if this bar's close time >= cutoff_time, exit at close
        if bar_close_time >= cutoff_time:
            exit_price_raw = close
            exit_time = bar_close_time
            exit_reason = "time"
            break

    # If we ran out of bars before the cutoff (end of RTH), exit on last bar close
    if exit_price_raw is None:
        last = post.iloc[-1]
        exit_price_raw = float(last["close"])
        exit_time = post.index[-1] + pd.Timedelta(minutes=1)
        exit_reason = "eod"

    exit_price = apply_slippage(exit_price_raw, direction, "exit")

    # PnL in $
    if direction == "long":
        pts_gross = exit_price_raw - entry_price_raw
        pts_net   = exit_price     - entry_price
    else:
        pts_gross = entry_price_raw - exit_price_raw
        pts_net   = entry_price     - exit_price

    pnl_gross = pts_gross * POINT_VALUE
    pnl_net   = pts_net   * POINT_VALUE - FIXED_COST_RT

    return Trade(
        date=date, variant=variant, direction=direction,
        entry_time=entry_time, exit_time=exit_time,
        entry_price=entry_price, exit_price=exit_price,
        gap_pct=gap, atr=atr_val if atr_val is not None else float("nan"),
        exit_reason=exit_reason,
        post_holiday=bool(gap_row["post_holiday"]),
        dow=gap_row["dow"],
        pnl_gross=pnl_gross, pnl_net=pnl_net,
    )


def run_variant(variant, cutoff_min, use_atr, gaps, day_minutes):
    trades = []
    for date, row in gaps.iterrows():
        day_bars = day_minutes.get(date)
        if day_bars is None:
            continue
        t = simulate_day(date, day_bars, row, variant, cutoff_min, use_atr, row["atr"])
        if t is not None:
            trades.append(t)
    return pd.DataFrame([t.__dict__ for t in trades])


# ── Step 7: Metrics ─────────────────────────────────────────────────────────
def metrics(trades: pd.DataFrame, daily_dates):
    if trades.empty:
        return {"n": 0}
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"])
    wins = t[t["pnl_net"] > 0]
    losses = t[t["pnl_net"] < 0]

    # Daily returns per $25K margin
    daily_pnl = t.groupby(t["date"].dt.date)["pnl_net"].sum()
    daily_ret = daily_pnl / MARGIN_PER_CONTRACT
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    # Equity curve (per-trade cumulative)
    eq = t.sort_values("date")["pnl_net"].cumsum()
    peak = eq.cummax()
    dd = eq - peak
    max_dd = dd.min()
    max_dd_pct = max_dd / MARGIN_PER_CONTRACT * 100

    # Annualized return
    span_days = (t["date"].max() - t["date"].min()).days or 1
    years = span_days / 365.25
    total_pnl = t["pnl_net"].sum()
    ann_return = (total_pnl / MARGIN_PER_CONTRACT) / years * 100

    return {
        "n":            len(t),
        "total_pnl":    total_pnl,
        "ann_return":   ann_return,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "max_dd_pct":   max_dd_pct,
        "win_rate":     len(wins) / len(t) * 100 if len(t) else 0,
        "avg_win":      wins["pnl_net"].mean() if len(wins) else 0,
        "avg_loss":     losses["pnl_net"].mean() if len(losses) else 0,
        "payoff":       abs(wins["pnl_net"].mean() / losses["pnl_net"].mean())
                         if len(losses) and losses["pnl_net"].mean() != 0 else float("nan"),
        "exp_per_trade": t["pnl_net"].mean(),
    }


def fmt_metrics(m):
    if m["n"] == 0: return "(no trades)"
    return (f"n={m['n']}  PnL=${m['total_pnl']:+,.0f}  AnnRet={m['ann_return']:+.1f}%  "
            f"Sharpe={m['sharpe']:.2f}  DD=${m['max_dd']:,.0f} ({m['max_dd_pct']:.1f}%)  "
            f"WR={m['win_rate']:.1f}%  AvgWin=${m['avg_win']:+,.0f}  AvgLoss=${m['avg_loss']:+,.0f}  "
            f"Payoff={m['payoff']:.2f}  Exp/Trade=${m['exp_per_trade']:+,.1f}")


def split_is_oos(trades, cutoff_date):
    is_ = trades[pd.to_datetime(trades["date"]).dt.date < cutoff_date].copy()
    oos = trades[pd.to_datetime(trades["date"]).dt.date >= cutoff_date].copy()
    return is_, oos


# ── Breakdown helpers ───────────────────────────────────────────────────────
def breakdown_by_direction(tr):
    rows = []
    for d in ["short", "long"]:
        sub = tr[tr["direction"] == d]
        if sub.empty: continue
        rows.append({
            "direction": d,
            "n": len(sub),
            "pnl": sub["pnl_net"].sum(),
            "win_rate": (sub["pnl_net"] > 0).mean() * 100,
            "avg_pnl": sub["pnl_net"].mean(),
        })
    return pd.DataFrame(rows)


def breakdown_by_magnitude(tr):
    bins = [0.003, 0.005, 0.008, 0.012, 1.0]
    labels = ["0.3-0.5%", "0.5-0.8%", "0.8-1.2%", "1.2%+"]
    tr = tr.copy()
    tr["absgap"] = tr["gap_pct"].abs()
    tr["bucket"] = pd.cut(tr["absgap"], bins=bins, labels=labels, right=False, include_lowest=True)
    rows = []
    for b in labels:
        sub = tr[tr["bucket"] == b]
        if sub.empty: continue
        rows.append({
            "magnitude": b, "n": len(sub),
            "pnl": sub["pnl_net"].sum(),
            "win_rate": (sub["pnl_net"] > 0).mean() * 100,
            "avg_pnl": sub["pnl_net"].mean(),
        })
    return pd.DataFrame(rows)


def breakdown_by_dow(tr):
    rows = []
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        sub = tr[tr["dow"] == d]
        if sub.empty:
            rows.append({"dow": d, "n": 0}); continue
        vals = sub["pnl_net"].values
        tstat, pval = stats.ttest_1samp(vals, 0.0)
        daily = sub.groupby(pd.to_datetime(sub["date"]).dt.date)["pnl_net"].sum() / MARGIN_PER_CONTRACT
        sh = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
        rows.append({
            "dow": d, "n": len(sub),
            "pnl": vals.sum(),
            "win_rate": (vals > 0).mean() * 100,
            "avg_pnl": vals.mean(),
            "sharpe": sh,
            "t_stat": tstat, "p_value": pval,
            "flag": "SMALL_N" if len(sub) < 30 else "",
        })
    return pd.DataFrame(rows)


def breakdown_post_holiday(tr):
    rows = []
    for flag, sub in tr.groupby("post_holiday"):
        rows.append({
            "post_holiday": bool(flag), "n": len(sub),
            "pnl": sub["pnl_net"].sum(),
            "win_rate": (sub["pnl_net"] > 0).mean() * 100,
            "avg_pnl": sub["pnl_net"].mean(),
        })
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_equity(trades_all, cutoff_date, out_path):
    tr = trades_all.sort_values("date").copy()
    tr["date"] = pd.to_datetime(tr["date"])
    tr["cum"] = tr["pnl_net"].cumsum()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(tr["date"], tr["cum"], lw=1.2)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.axvline(pd.Timestamp(cutoff_date), color="red", lw=1.2, ls="--", label="IS | OOS")
    ax.set_title("Equity curve — best variant (IS + OOS)")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative PnL ($ per contract)")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_pnl_hist(trades_all, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(trades_all["pnl_net"], bins=60, edgecolor="black", alpha=0.85)
    ax.axvline(0, color="red", lw=0.8)
    ax.set_title("Trade PnL distribution (net of costs)")
    ax.set_xlabel("PnL ($)"); ax.set_ylabel("Count"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_gap_vs_pnl(trades_all, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    long_tr = trades_all[trades_all["direction"] == "long"]
    short_tr = trades_all[trades_all["direction"] == "short"]
    ax.scatter(long_tr["gap_pct"]*100, long_tr["pnl_net"], s=12, alpha=0.55, label=f"long ({len(long_tr)})")
    ax.scatter(short_tr["gap_pct"]*100, short_tr["pnl_net"], s=12, alpha=0.55, color="orange", label=f"short ({len(short_tr)})")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Gap %"); ax.set_ylabel("Net PnL ($)")
    ax.set_title("Gap magnitude vs trade PnL")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


# ── Main driver ──────────────────────────────────────────────────────────────
def main():
    df_all, rth, daily, day_minutes, sched = load_and_prep()
    gaps = build_gap_table(daily, day_minutes)

    # Train/test split point
    all_dates = sorted(gaps.index)
    cutoff_idx = int(len(all_dates) * IS_FRACTION)
    cutoff_date = all_dates[cutoff_idx]
    print(f"\n[6] IS/OOS split at {cutoff_date}  (IS={cutoff_idx}  OOS={len(all_dates)-cutoff_idx} trading days)")

    # Run all variants
    variants = [("A15", 15, False), ("A30", 30, False), ("A45", 45, False),
                ("A60", 60, False), ("A90", 90, False),
                ("B",   B_CUTOFF_MIN, True),
                ("C",   C_CUTOFF_MIN, True)]

    print("\n[4] Simulating variants...", flush=True)
    all_trades = {}
    for name, cutoff, use_atr in variants:
        t0 = pd.Timestamp.now()
        tr = run_variant(name, cutoff, use_atr, gaps, day_minutes)
        dt = (pd.Timestamp.now() - t0).total_seconds()
        all_trades[name] = tr
        print(f"    {name}: {len(tr)} trades  ({dt:.1f}s)")

    # Save trades CSV (all variants combined)
    combined = pd.concat([tr for tr in all_trades.values()], ignore_index=True)
    combined.to_csv(OUT_DIR / "trades_all.csv", index=False)
    print(f"\n[8] Wrote {OUT_DIR/'trades_all.csv'} ({len(combined)} rows across {len(all_trades)} variants)")

    # Build summary table
    print("\n[7] Computing IS/OOS metrics...")
    summary_rows = []
    for name in all_trades:
        tr = all_trades[name]
        is_tr, oos_tr = split_is_oos(tr, cutoff_date)
        m_is = metrics(is_tr, daily.index)
        m_oos = metrics(oos_tr, daily.index)
        summary_rows.append({
            "variant": name,
            "is_n": m_is.get("n", 0),  "is_pnl": m_is.get("total_pnl", 0),
            "is_wr": m_is.get("win_rate", 0), "is_sharpe": m_is.get("sharpe", 0),
            "is_dd": m_is.get("max_dd", 0), "is_exp": m_is.get("exp_per_trade", 0),
            "oos_n": m_oos.get("n", 0), "oos_pnl": m_oos.get("total_pnl", 0),
            "oos_wr": m_oos.get("win_rate", 0), "oos_sharpe": m_oos.get("sharpe", 0),
            "oos_dd": m_oos.get("max_dd", 0), "oos_exp": m_oos.get("exp_per_trade", 0),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "variant_summary.csv", index=False)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Pick the best variant by OOS Sharpe (conservative) and by OOS PnL
    print("\n--- Per-variant detail ---")
    for name in all_trades:
        tr = all_trades[name]
        is_tr, oos_tr = split_is_oos(tr, cutoff_date)
        print(f"\n### {name}")
        print(f"  IS:  {fmt_metrics(metrics(is_tr, daily.index))}")
        print(f"  OOS: {fmt_metrics(metrics(oos_tr, daily.index))}")

    # Best variant selection on IS ONLY (non-negotiable per task)
    best_is_sharpe = max(all_trades.keys(),
                        key=lambda k: metrics(split_is_oos(all_trades[k], cutoff_date)[0],
                                              daily.index).get("sharpe", -999))
    best_is_pnl = max(all_trades.keys(),
                      key=lambda k: metrics(split_is_oos(all_trades[k], cutoff_date)[0],
                                             daily.index).get("total_pnl", -1e18))
    print(f"\nBest variant by IS Sharpe: {best_is_sharpe}")
    print(f"Best variant by IS total PnL: {best_is_pnl}")
    best_variant = best_is_sharpe
    print(f"--> Using {best_variant} for detailed breakdowns and plots")

    tr_best = all_trades[best_variant]
    is_tr, oos_tr = split_is_oos(tr_best, cutoff_date)

    # Breakdowns
    print(f"\n### {best_variant} — Direction breakdown (IS)")
    print(breakdown_by_direction(is_tr).to_string(index=False))
    print(f"\n### {best_variant} — Direction breakdown (OOS)")
    print(breakdown_by_direction(oos_tr).to_string(index=False))

    print(f"\n### {best_variant} — Magnitude bucket (IS)")
    print(breakdown_by_magnitude(is_tr).to_string(index=False))
    print(f"\n### {best_variant} — Magnitude bucket (OOS)")
    print(breakdown_by_magnitude(oos_tr).to_string(index=False))

    print(f"\n### {best_variant} — Day of week (IS)  [flag SMALL_N if n<30]")
    print(breakdown_by_dow(is_tr).to_string(index=False))
    print(f"\n### {best_variant} — Day of week (OOS)")
    print(breakdown_by_dow(oos_tr).to_string(index=False))

    print(f"\n### {best_variant} — Post-holiday vs normal (IS)")
    print(breakdown_post_holiday(is_tr).to_string(index=False))
    print(f"\n### {best_variant} — Post-holiday vs normal (OOS)")
    print(breakdown_post_holiday(oos_tr).to_string(index=False))

    # Plots — best variant only
    plot_equity(tr_best, cutoff_date, OUT_DIR / f"equity_{best_variant}.png")
    plot_pnl_hist(tr_best, OUT_DIR / f"pnl_hist_{best_variant}.png")
    plot_gap_vs_pnl(tr_best, OUT_DIR / f"gap_vs_pnl_{best_variant}.png")
    print(f"\nPlots saved to {OUT_DIR}/")

    # Write markdown summary
    write_summary(all_trades, summary_df, best_variant, cutoff_date,
                  is_tr, oos_tr, daily.index)

    print("\nDONE.")


def write_summary(all_trades, summary_df, best, cutoff_date, is_tr, oos_tr, daily_idx):
    lines = []
    lines.append(f"# Overnight Gap Mean Reversion — NQ 1-min Backtest\n")
    lines.append(f"Cost model: ${FIXED_COST_RT:.2f} commissions+fees + "
                 f"{SLIPPAGE_PTS} pt slippage per side (=${SLIPPAGE_PTS*POINT_VALUE*2:.2f} slippage RT).\n")
    lines.append(f"Gap threshold: ±{GAP_THRESHOLD*100:.1f}%.  "
                 f"IS/OOS cutoff date: {cutoff_date}.\n")
    lines.append("## Variant comparison (IS | OOS)\n")
    lines.append("| Variant | IS n | IS PnL | IS WR | IS Sharpe | IS DD | IS Exp/Trade "
                 "| OOS n | OOS PnL | OOS WR | OOS Sharpe | OOS DD | OOS Exp/Trade |\n"
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['is_n'])} | ${r['is_pnl']:+,.0f} | "
            f"{r['is_wr']:.1f}% | {r['is_sharpe']:.2f} | ${r['is_dd']:,.0f} | "
            f"${r['is_exp']:+.1f} | {int(r['oos_n'])} | ${r['oos_pnl']:+,.0f} | "
            f"{r['oos_wr']:.1f}% | {r['oos_sharpe']:.2f} | ${r['oos_dd']:,.0f} | "
            f"${r['oos_exp']:+.1f} |\n")

    lines.append(f"\n## Best variant (by IS Sharpe): **{best}**\n")
    m_is = metrics(is_tr, daily_idx); m_oos = metrics(oos_tr, daily_idx)

    def block(label, m):
        lines.append(f"### {label}\n")
        lines.append(f"- Trades: {m['n']}\n")
        lines.append(f"- Total PnL: ${m['total_pnl']:+,.0f}\n")
        lines.append(f"- Annualized return (vs ${int(MARGIN_PER_CONTRACT)} margin): {m['ann_return']:+.1f}%\n")
        lines.append(f"- Sharpe (daily, annualized): {m['sharpe']:.2f}\n")
        lines.append(f"- Max drawdown: ${m['max_dd']:,.0f} ({m['max_dd_pct']:.1f}%)\n")
        lines.append(f"- Win rate: {m['win_rate']:.1f}%\n")
        lines.append(f"- Avg win: ${m['avg_win']:+,.0f}  Avg loss: ${m['avg_loss']:+,.0f}  Payoff: {m['payoff']:.2f}\n")
        lines.append(f"- Expected value per trade: ${m['exp_per_trade']:+,.1f}\n\n")
    block("In-sample", m_is); block("Out-of-sample", m_oos)

    lines.append(f"## Direction (OOS)\n")
    lines.append(breakdown_by_direction(oos_tr).to_markdown(index=False) + "\n\n")
    lines.append(f"## Magnitude buckets (OOS)\n")
    lines.append(breakdown_by_magnitude(oos_tr).to_markdown(index=False) + "\n\n")
    lines.append(f"## Day of week (OOS) — t-test vs zero\n")
    lines.append(breakdown_by_dow(oos_tr).to_markdown(index=False) + "\n\n")
    lines.append(f"## Post-holiday (OOS)\n")
    lines.append(breakdown_post_holiday(oos_tr).to_markdown(index=False) + "\n\n")

    # Final written verdict
    verdict = []
    oos_sharpe = m_oos["sharpe"]; oos_pnl = m_oos["total_pnl"]; oos_exp = m_oos["exp_per_trade"]
    if oos_sharpe > 0.5 and oos_exp > 0:
        verdict.append("**Edge on NQ: YES (OOS Sharpe > 0.5 and OOS expectancy > 0).**")
    elif oos_sharpe > 0 and oos_exp > 0:
        verdict.append("**Edge on NQ: WEAK / INCONCLUSIVE — OOS PnL positive but Sharpe < 0.5.**")
    elif oos_pnl > 0:
        verdict.append("**Edge on NQ: MARGINAL — OOS PnL barely positive, likely noise.**")
    else:
        verdict.append("**Edge on NQ: NO — OOS PnL non-positive.**")

    is_sharpe_val = m_is["sharpe"]
    if is_sharpe_val > 0 and oos_sharpe > 0.5 * is_sharpe_val:
        verdict.append("- OOS consistent with IS (Sharpe within 50%).")
    elif is_sharpe_val > 0 and oos_sharpe > 0:
        verdict.append("- OOS degraded vs IS but same sign.")
    else:
        verdict.append("- OOS degraded significantly / flipped sign vs IS.")

    lines.append("## Verdict\n")
    lines.extend(l + "\n" for l in verdict)

    (OUT_DIR / "summary.md").write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'summary.md'}")


if __name__ == "__main__":
    main()

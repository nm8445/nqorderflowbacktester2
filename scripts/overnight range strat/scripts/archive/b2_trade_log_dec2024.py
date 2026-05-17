"""Per-trade log for a B2 config in Dec 2024.

Current schema (lookahead-fixed, symmetric wick-anchored framework):
  filters:  pinbar_X >= X, abs_delta_w{N} >= D, optional strict-shorts
  exits:    SL_dist = base + sl_M*ATR ; TP_dist = base + tp_M*ATR

Reports per trade: OHI/OLO, signal+confirmation OHLC, entry, level, TP/SL prices, hit times.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import (  # noqa: E402
    apply_filters, mode1_chained_dedupe,
)

PARQUET_DIR = Path(__file__).parent / "parquets"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
TRADELOG_DIR.mkdir(exist_ok=True)
TRADES = PARQUET_DIR / "entry_signal_trades.parquet"
MQ     = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
VOL5M  = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")

# Top config from in-sample (chained Mode 1, BAND_K=0.25, sym 1:1 R:R)
VARIANT = "B2"
X       = 1.25    # pinbar_X
N       = 5       # window_N (ticks)
D       = 70      # delta_D (min |delta| in best window)
STRICT  = True
TP_M    = 1.0     # symmetric: TP_dist = base + TP_M*ATR
SL_P    = 1.0     # symmetric: SL_dist = base + SL_P*ATR  (=> R:R = 1:1 when TP_M=SL_P)


def load_5m_bars_for_dates(dates: list[dt.date]) -> pd.DataFrame:
    """Load just the 5-min bar headers (open/close/high/low/time) for a date set,
    RTH-only (9:30-17:00 ET) to match simulate_trade's bar indexing."""
    df = pd.read_parquet(VOL5M, columns=["session_date", "bar_open_time",
                                           "open", "high", "low", "close"])
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    df = df[df["session_date"].isin(dates)]
    bot = pd.to_datetime(df["bar_open_time"], utc=True).dt.tz_convert("America/New_York")
    df["bar_open_time"] = bot
    rth_mask = ((bot.dt.time >= dt.time(9, 30)) &
                (bot.dt.time <= dt.time(17, 0)))
    df = df[rth_mask & df["open"].notna()].copy()
    df = df.drop_duplicates(subset=["session_date", "bar_open_time"]).reset_index(drop=True)
    df = df.sort_values(["session_date", "bar_open_time"])
    return df


def label_level(mq_row: pd.Series, level_price: float, tol: float = 0.5) -> str:
    """Find which named level (CR, PS, GEX_n × QQQ/NDX) matches level_price."""
    candidates = []
    for col in mq_row.index:
        if not col.endswith("_nq"):
            continue
        v = mq_row[col]
        if pd.isna(v):
            continue
        if abs(v - level_price) <= tol:
            candidates.append((col, v))
    if not candidates:
        # closest match
        best = None; best_d = 1e9
        for col in mq_row.index:
            if not col.endswith("_nq"):
                continue
            v = mq_row[col]
            if pd.isna(v):
                continue
            d = abs(v - level_price)
            if d < best_d:
                best_d = d; best = col
        if best:
            return f"{best.replace('_nq','')}_~{best_d:.1f}pt_off"
        return "unknown"
    name = candidates[0][0].replace("_nq", "")
    return name


def fmt_time(t):
    if t is None or (not hasattr(t, "strftime") and pd.isna(t)):
        return "-"
    if hasattr(t, "strftime"):
        return t.strftime("%m-%d %H:%M")
    return str(t)


def fmt_hit(t):
    return fmt_time(t) if t is not None else "-"


def main():
    print(f"loading trades parquet...")
    df = pd.read_parquet(TRADES)
    df["date"] = pd.to_datetime(df["date"])

    # Filter to Dec 2024
    dec24 = df[(df["date"] >= "2024-12-01") & (df["date"] < "2025-01-01")].copy()
    print(f"  Dec 2024 raw rows: {len(dec24)}")

    # Apply config filters with default adaptive band (BAND_K=0.25 per in-sample best)
    filtered = apply_filters(dec24, VARIANT, X, N, D, STRICT)
    # Chained Mode 1: sequential trades per day, no overlap
    deduped = mode1_chained_dedupe(filtered, TP_M, SL_P)
    print(f"  After filters (X={X} N={N} D={D} strict={STRICT}, BAND_K=0.25) + "
          f"chained Mode 1: {len(deduped)} trades")
    if deduped.empty:
        return

    # Load MenthorQ for level naming
    mq = pd.read_parquet(MQ)
    mq["date"] = pd.to_datetime(mq["date"]).dt.date
    mq = mq.set_index("date")

    # Lookahead-free label lookup: for trade on day D, look up MQ from PRIOR
    # trading day (the same one used in the backtest signal-matching).
    mq_dates_sorted = sorted(mq.index.tolist())

    def prior_mq_day(session_date):
        prev = None
        for md in mq_dates_sorted:
            if md < session_date: prev = md
            else: break
        return prev

    # Load 5-min bars for days in question
    dates = sorted(set(deduped["date"].dt.date.tolist()))
    bars = load_5m_bars_for_dates(dates)
    bars_by_day = {d: g.reset_index(drop=True) for d, g in bars.groupby("session_date")}

    # Build per-trade log
    print()
    width = 280
    print("=" * width)
    header = (f"{'Date':<10} {'OHI':>8} {'OLO':>8} | "
              f"{'Signal Time':<11} {'SigO':>9} {'SigH':>9} {'SigL':>9} {'SigC':>9} | "
              f"{'Conf Time':<11} {'ConfO':>9} {'ConfH':>9} {'ConfL':>9} {'ConfC':>9} | "
              f"{'Entry Time':<11} {'Entry':>9} {'Dir':>5} | "
              f"{'Level':<12} {'LvlPrice':>9} | "
              f"{'TP @':>9} {'SL @':>9} | "
              f"{'TP Hit':<11} {'SL Hit':<11} {'Outcome':<14} {'Result(pts)':>11}")
    print(header)
    print("-" * width)

    for _, t in deduped.sort_values("entry_time").iterrows():
        d = t["date"].date()
        sig_t = pd.Timestamp(t["signal_time"])
        ent_t = pd.Timestamp(t["entry_time"])
        # Confirmation candle = bar between signal_time and entry_time
        # For B2/B3 variants, entry is at signal_time + 2 bars (signal + confirm + entry-bar-open)
        conf_t = sig_t + pd.Timedelta(minutes=5)
        day_bars = bars_by_day.get(d)
        if day_bars is None:
            continue
        # Find confirmation candle OHLC
        conf_row = day_bars[day_bars["bar_open_time"] == conf_t]
        if not conf_row.empty:
            conf_o = float(conf_row["open"].iloc[0])
            conf_h = float(conf_row["high"].iloc[0])
            conf_l = float(conf_row["low"].iloc[0])
            conf_c = float(conf_row["close"].iloc[0])
        else:
            conf_o = conf_h = conf_l = conf_c = float("nan")

        # Level naming — use PRIOR trading day's MQ (matches what backtest used)
        prior_d = prior_mq_day(d)
        mq_row = mq.loc[prior_d] if prior_d is not None else pd.Series(dtype=float)
        level_name = label_level(mq_row, t["near_level"]) if not mq_row.empty else "no-mq-data"

        # TP and SL prices (symmetric wick-anchored framework)
        sign = 1 if t["direction"] == "LONG" else -1
        atr = t["atr_at_entry"]
        entry = t["entry_price"]
        anchor = t["sl_anchor"]
        base = sign * (entry - anchor)
        tp_price = entry + sign * (base + TP_M * atr)   # symmetric: matches SL distance when TP_M=SL_P
        sl_price = anchor - sign * SL_P * atr

        # Hit times — convert bar index to bar_open_time
        tp_idx = int(t[f"tp_{TP_M:.2f}_idx"])
        sl_idx = int(t[f"sl_{SL_P:.2f}_idx"])
        tp_hit_t = day_bars.iloc[tp_idx]["bar_open_time"] if tp_idx >= 0 and tp_idx < len(day_bars) else None
        sl_hit_t = day_bars.iloc[sl_idx]["bar_open_time"] if sl_idx >= 0 and sl_idx < len(day_bars) else None

        # Determine outcome (symmetric wick-anchored: TP_dist = base + TP_M * ATR)
        if sl_idx >= 0 and (tp_idx < 0 or sl_idx < tp_idx):
            outcome = "SL hit"
            realized = sign * (sl_price - entry)
        elif tp_idx >= 0:
            outcome = "TP hit"
            realized = base + TP_M * atr
        else:
            outcome = "held->close"
            realized = t["close_pnl_pts"]

        print(f"{d!s:<10} {t['ohi']:>8.2f} {t['olo']:>8.2f} | "
              f"{sig_t.strftime('%m-%d %H:%M'):<11} "
              f"{t['signal_open']:>9.2f} {t['signal_high']:>9.2f} {t['signal_low']:>9.2f} {t['signal_close']:>9.2f} | "
              f"{fmt_time(conf_t):<11} "
              f"{conf_o:>9.2f} {conf_h:>9.2f} {conf_l:>9.2f} {conf_c:>9.2f} | "
              f"{ent_t.strftime('%m-%d %H:%M'):<11} {entry:>9.2f} {t['direction']:>5} | "
              f"{level_name:<12} {t['near_level']:>9.2f} | "
              f"{tp_price:>9.2f} {sl_price:>9.2f} | "
              f"{fmt_time(tp_hit_t):<11} {fmt_time(sl_hit_t):<11} "
              f"{outcome:<14} {realized:>+11.2f}")

    print("-" * width)
    total_pnl = 0.0
    for _, t in deduped.iterrows():
        sign = 1 if t["direction"] == "LONG" else -1
        atr = t["atr_at_entry"]; entry = t["entry_price"]; anchor = t["sl_anchor"]
        base = sign * (entry - anchor)
        sl_price = anchor - sign * SL_P * atr
        tp_idx = int(t[f"tp_{TP_M:.2f}_idx"]); sl_idx = int(t[f"sl_{SL_P:.2f}_idx"])
        if sl_idx >= 0 and (tp_idx < 0 or sl_idx < tp_idx):
            total_pnl += sign * (sl_price - entry)
        elif tp_idx >= 0:
            total_pnl += base + TP_M * atr
        else:
            total_pnl += t["close_pnl_pts"]
    print(f"Dec 2024 total: n={len(deduped)} realized PnL = {total_pnl:+.2f} pts "
          f"(${total_pnl*20:+.0f}/contract @ $20/pt)")


if __name__ == "__main__":
    sys.exit(main())

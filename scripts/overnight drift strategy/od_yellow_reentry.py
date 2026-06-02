"""OD with yellow re-entry: re-enter long after SL Yellow exit if a subsequent
20-min candle closes BACK ABOVE the prior yellow level.

Test idea: yellow stops often hit on noise spikes during overnight drift.
If price recovers above the yellow within the session, the bullish drift
edge may resume. Capture it via re-entry.

Variants tested:
  - max_reentries: 1, 2, 3
  - reentry only same-night (cannot cross next-day boundary)
  - Martingale: each re-entry uses current marti state (independent trades)
  - Force-close still at 08:00 ET

Compare to baseline (current live OD config).
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field, replace
from datetime import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import (
    StrategyParams, Trade, rma_atr, build_full_20min_series, trades_to_df,
    NQ_POINT_VALUE,
)

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"


@dataclass
class ReentryParams(StrategyParams):
    allow_yellow_reentry: bool = True
    max_reentries: int = 2          # per session (after the original entry)


def run_backtest_with_reentry(bars: pd.DataFrame, params: ReentryParams) -> list[Trade]:
    if bars.index.tz is None:
        raise ValueError("bars index must be tz-aware (America/New_York)")

    atr_y = rma_atr(bars["high"], bars["low"], bars["close"], params.yellow_atr_len)
    atr_g = rma_atr(bars["high"], bars["low"], bars["close"], params.green_atr_len)

    entry_t = time(params.entry_hour, params.entry_minute)
    force_t = time(params.forced_hour, params.forced_minute)

    open_a = bars["open"].values
    high_a = bars["high"].values
    low_a = bars["low"].values
    close_a = bars["close"].values
    atr_y_a = atr_y.values
    atr_g_a = atr_g.values
    idx = bars.index

    trades: list[Trade] = []

    in_pos = False
    entry_price = np.nan
    entry_idx = -1
    entry_qty = 0
    yellow_val = np.nan
    prev_yellow = np.nan
    prev_close = np.nan

    # Re-entry state (per-session)
    awaiting_reentry = False
    exit_yellow_level = np.nan
    session_reentries_used = 0
    session_started = False

    marti_state = 0
    next_qty = params.base_qty

    def enter(i, qty_override=None):
        nonlocal in_pos, entry_price, entry_idx, entry_qty, yellow_val, prev_yellow, prev_close, awaiting_reentry
        c = close_a[i]; ay = atr_y_a[i]
        if params.use_martingale:
            if marti_state == 0:
                qty = params.base_qty
            elif marti_state == 1:
                qty = params.loss_qty
            else:
                qty = params.base_qty
        else:
            qty = params.base_qty
        if qty_override is not None:
            qty = qty_override
        in_pos = True
        entry_price = c
        entry_idx = i
        entry_qty = int(qty)
        yellow_val = c - params.yellow_atr_mult * ay
        prev_yellow = np.nan
        prev_close = c
        awaiting_reentry = False

    for i in range(len(bars)):
        ts = idx[i]
        local_t = ts.time()
        o = open_a[i]; h = high_a[i]; l = low_a[i]; c = close_a[i]
        ay = atr_y_a[i]; ag = atr_g_a[i]

        # Reset session state at the 19:00 entry boundary
        if local_t == entry_t and not session_started:
            session_started = True
            session_reentries_used = 0
            awaiting_reentry = False
            exit_yellow_level = np.nan

        # Reset session_started flag if past force-close (allow new session next day)
        if local_t == force_t:
            session_started = False

        # ---------- INITIAL ENTRY at 19:00 ----------
        if (not in_pos and local_t == entry_t
                and not np.isnan(ay) and not np.isnan(c)):
            enter(i)
            continue

        # ---------- RE-ENTRY logic ----------
        if (not in_pos and awaiting_reentry and params.allow_yellow_reentry
                and session_reentries_used < params.max_reentries
                and not np.isnan(ay) and not np.isnan(c)
                and local_t != force_t):
            if c > exit_yellow_level:
                enter(i)
                session_reentries_used += 1
                continue

        # ---------- IN-TRADE LOGIC ----------
        if in_pos:
            bars_in_trade = i - entry_idx
            raw_yellow = c - params.yellow_atr_mult * ay if not np.isnan(ay) else np.nan
            # Pure ratchet
            if not np.isnan(prev_yellow):
                yellow_val = max(prev_yellow, raw_yellow) if not np.isnan(raw_yellow) else prev_yellow
            else:
                yellow_val = raw_yellow

            red_val = entry_price + params.red_intercept + params.red_drift * bars_in_trade
            green_val = (red_val + params.green_base
                         - params.green_decay * bars_in_trade
                         + (params.green_atr_mult * ag if not np.isnan(ag) else 0.0))

            exited = False
            exit_price = np.nan
            exit_reason = ""

            # TP green (at close)
            if not np.isnan(green_val) and h >= green_val:
                exit_price = c
                exit_reason = "TP Green"
                exited = True

            # SL yellow on bearish close
            if (not exited and not np.isnan(yellow_val) and c <= yellow_val and c < o
                    and bars_in_trade >= params.yellow_suppress_bars):
                exit_price = c
                exit_reason = "SL Yellow"
                exited = True

            # Force close at 08:00
            if not exited and local_t == force_t:
                exit_price = c
                exit_reason = "Force Close"
                exited = True

            if exited:
                trades.append(Trade(
                    entry_time=idx[entry_idx], exit_time=ts,
                    entry_price=entry_price, exit_price=exit_price,
                    qty=entry_qty, reason=exit_reason, bars_held=bars_in_trade,
                ))
                last_was_loss = (exit_price - entry_price) < 0
                if marti_state == 0:
                    marti_state = 1 if last_was_loss else 0
                elif marti_state == 1:
                    marti_state = 2
                else:
                    marti_state = 1 if last_was_loss else 0

                # Set up potential re-entry
                if exit_reason == "SL Yellow" and params.allow_yellow_reentry:
                    awaiting_reentry = True
                    exit_yellow_level = yellow_val

                in_pos = False
                entry_price = np.nan
                entry_idx = -1
                entry_qty = 0
                yellow_val = np.nan
                prev_yellow = np.nan
                prev_close = np.nan
                continue

            prev_yellow = yellow_val
            prev_close = c

    return trades


def summarize(trades_df: pd.DataFrame, label: str):
    if trades_df.empty:
        print(f"  {label}: no trades"); return
    pnl = trades_df["pnl_dollars"].values
    n = len(pnl); w = (pnl > 0).sum()
    gross_w = pnl[pnl > 0].sum(); gross_l = -pnl[pnl < 0].sum()
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    cum = np.cumsum(pnl)
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    worst = float(min(pnl))
    print(f"  {label}: n={n} WR={w/n*100:.1f}% net=${pnl.sum():,.0f} "
          f"PF={pf:.3f} MDD=${mdd:,.0f} worst=${worst:,.0f}")


def main():
    print("Loading 20-min bars...")
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    bars = bars.tz_convert("America/New_York")
    print(f"  {len(bars):,} bars  ({bars.index.min()} to {bars.index.max()})")

    print("\nBaseline (original OD live):")
    from overnight_drift_strategy import run_backtest as run_baseline
    base_trades = run_baseline(bars, StrategyParams())
    base_df = trades_to_df(base_trades)
    summarize(base_df, "BASELINE (no reentry)")

    print("\nWith yellow re-entry:")
    for max_re in [1, 2, 3, 5]:
        p = ReentryParams(allow_yellow_reentry=True, max_reentries=max_re)
        re_trades = run_backtest_with_reentry(bars, p)
        re_df = trades_to_df(re_trades)
        summarize(re_df, f"RE-ENTRY (max={max_re})")

    # Detailed breakdown for max=2 (likely sweet spot)
    print("\nDetailed comparison (BASELINE vs RE-ENTRY max=2):")
    p = ReentryParams(allow_yellow_reentry=True, max_reentries=2)
    re_trades = run_backtest_with_reentry(bars, p)
    re_df = trades_to_df(re_trades)

    base_df["session"] = base_df["entry_time"].dt.tz_convert("America/New_York").dt.date
    re_df["session"]   = re_df["entry_time"].dt.tz_convert("America/New_York").dt.date

    base_daily = base_df.groupby("session")["pnl_dollars"].sum()
    re_daily   = re_df.groupby("session")["pnl_dollars"].sum()

    common = base_daily.index.intersection(re_daily.index)
    diff = re_daily.loc[common] - base_daily.loc[common]
    print(f"  Sessions covered: {len(common)}")
    print(f"  Total net diff (re-entry - baseline): ${diff.sum():,.0f}")
    print(f"  Avg per-session diff: ${diff.mean():.0f}")
    print(f"  Sessions where re-entry helped: {(diff > 0).sum()} ({(diff>0).mean()*100:.1f}%)")
    print(f"  Sessions where re-entry hurt:   {(diff < 0).sum()} ({(diff<0).mean()*100:.1f}%)")
    print(f"  Best improvement: ${diff.max():,.0f}    Worst hurt: ${diff.min():,.0f}")

    # Save trades
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    re_df.to_csv(out / "od_yellow_reentry_trades.csv", index=False)
    print(f"\nSaved trades to {out / 'od_yellow_reentry_trades.csv'}")


if __name__ == "__main__":
    main()

"""
VWAP Reaction Strategy — ATR SL/TP Grid Optimization.

Sweeps SL mult [0.5–3.0] x TP mult [0.5–3.0] in 0.25 increments.
Splits data into in-sample / out-of-sample, ranks by robustness.
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0

ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"

# Date split (~50/50)
FULL_START = "2025-03-13"
FULL_END   = "2026-04-08"
IS_END     = "2025-09-15"   # in-sample:  2025-03-13 to 2025-09-15
OOS_START  = "2025-09-16"   # out-of-sample: 2025-09-16 to 2026-04-08

# Grid
SL_MIN, SL_MAX, SL_STEP = 0.5, 3.0, 0.25
TP_MIN, TP_MAX, TP_STEP = 0.5, 3.0, 0.25


def _load_all_days(start_date, end_date):
    """Pre-load all cache data for the date range. Returns list of (date_str, signals, bars)."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    days = []
    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        with open(cache_file, 'rb') as f:
            vwap_data = pickle.load(f)
        signals = vwap_data["signals"]
        if not signals:
            continue
        signal_cache_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_cache_file.exists():
            continue
        with open(signal_cache_file, 'rb') as f:
            signal_data = pickle.load(f)
        bars = signal_data["bars"]
        days.append((date_str, signals, bars))
    return days


def _run_fast(days, sl_mult, tp_mult):
    """Run backtest on pre-loaded days. Returns list of trade dicts."""
    trades = []
    for date_str, signals, bars in days:
        last_exit_time = None
        for signal in signals:
            entry_price = signal["entry_price"]
            direction = signal["direction"]
            atr = signal["atr"]
            confirm_bar_idx = signal["bar_index"] + 1

            if atr is None or atr <= 0:
                continue

            confirm_time = signal["confirm_time"]
            if hasattr(confirm_time, 'tz_convert'):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz='UTC').tz_convert(ET)
            if "16:00" <= confirm_et.strftime("%H:%M") < "19:10":
                continue
            if last_exit_time is not None and confirm_time <= last_exit_time:
                continue

            if direction == "long":
                sl_price = entry_price - atr * sl_mult
                tp_price = entry_price + atr * tp_mult
                if tp_price <= entry_price or sl_price >= entry_price:
                    continue
            else:
                sl_price = entry_price + atr * sl_mult
                tp_price = entry_price - atr * tp_mult
                if tp_price >= entry_price or sl_price <= entry_price:
                    continue

            exit_price = None
            exit_time = None
            exit_reason = None

            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                bar_ct = bar.close_time
                if hasattr(bar_ct, 'tz_convert'):
                    bar_et = bar_ct.tz_convert(ET)
                else:
                    bar_et = pd.Timestamp(bar_ct, tz='UTC').tz_convert(ET)
                if bar_et.strftime("%H:%M") >= FORCE_CLOSE:
                    exit_price = bar.close
                    exit_time = bar.close_time
                    exit_reason = "session_close"
                    break
                if direction == "short":
                    if bar.high >= sl_price:
                        exit_price = sl_price
                        exit_time = bar.close_time
                        exit_reason = "stop_loss"
                        break
                    if bar.low <= tp_price:
                        exit_price = tp_price
                        exit_time = bar.close_time
                        exit_reason = "target"
                        break
                else:
                    if bar.low <= sl_price:
                        exit_price = sl_price
                        exit_time = bar.close_time
                        exit_reason = "stop_loss"
                        break
                    if bar.high >= tp_price:
                        exit_price = tp_price
                        exit_time = bar.close_time
                        exit_reason = "target"
                        break

            if exit_price is None:
                last_bar = bars[-1]
                exit_price = last_bar.close
                exit_time = last_bar.close_time
                exit_reason = "eod"

            if direction == "short":
                pnl_points = entry_price - exit_price
            else:
                pnl_points = exit_price - entry_price

            trades.append(pnl_points)
            last_exit_time = exit_time

    return trades


def _stats(pnl_list):
    """Compute stats from a list of P&L values."""
    if not pnl_list:
        return {
            "trades": 0, "win_rate": 0, "expectancy": 0, "pf": 0,
            "total_pnl": 0, "max_dd": 0, "avg_win": 0, "avg_loss": 0,
        }
    arr = np.array(pnl_list)
    total = len(arr)
    winners = arr[arr > 0]
    losers = arr[arr < 0]
    win_rate = len(winners) / total * 100
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = losers.mean() if len(losers) > 0 else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    cum = arr.cumsum()
    max_dd = float((cum - np.maximum.accumulate(cum)).min())
    total_pnl = arr.sum()
    return {
        "trades": total,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "pf": pf,
        "total_pnl": total_pnl,
        "max_dd": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def main():
    print("Loading cache data...")
    is_days = _load_all_days(FULL_START, IS_END)
    oos_days = _load_all_days(OOS_START, FULL_END)
    print(f"  In-sample:      {FULL_START} to {IS_END} ({len(is_days)} trading days)")
    print(f"  Out-of-sample:  {OOS_START} to {FULL_END} ({len(oos_days)} trading days)")

    sl_vals = np.arange(SL_MIN, SL_MAX + SL_STEP / 2, SL_STEP)
    tp_vals = np.arange(TP_MIN, TP_MAX + TP_STEP / 2, TP_STEP)
    total_combos = len(sl_vals) * len(tp_vals)
    print(f"\nRunning {total_combos} combinations ({len(sl_vals)} SL x {len(tp_vals)} TP)...\n")

    results = []
    done = 0
    for sl in sl_vals:
        for tp in tp_vals:
            is_pnl = _run_fast(is_days, sl, tp)
            oos_pnl = _run_fast(oos_days, sl, tp)
            is_s = _stats(is_pnl)
            oos_s = _stats(oos_pnl)
            results.append({
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "is_trades": is_s["trades"],
                "is_wr": is_s["win_rate"],
                "is_exp": is_s["expectancy"],
                "is_pf": is_s["pf"],
                "is_pnl": is_s["total_pnl"],
                "is_dd": is_s["max_dd"],
                "oos_trades": oos_s["trades"],
                "oos_wr": oos_s["win_rate"],
                "oos_exp": oos_s["expectancy"],
                "oos_pf": oos_s["pf"],
                "oos_pnl": oos_s["total_pnl"],
                "oos_dd": oos_s["max_dd"],
            })
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total_combos} done...")

    df = pd.DataFrame(results)

    # ── Robustness filter ──────────────────────────────────────────────────
    # Must be profitable in BOTH IS and OOS, PF > 1 in both, min 20 trades each
    robust = df[
        (df["is_pnl"] > 0) &
        (df["oos_pnl"] > 0) &
        (df["is_pf"] > 1.0) &
        (df["oos_pf"] > 1.0) &
        (df["is_trades"] >= 20) &
        (df["oos_trades"] >= 10)
    ].copy()

    # Rank by OOS expectancy (since IS is fit territory)
    robust = robust.sort_values("oos_exp", ascending=False)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("FULL GRID RESULTS (sorted by OOS expectancy)")
    print("=" * 120)

    if robust.empty:
        print("No robust combinations found! Showing top 20 by IS expectancy instead.\n")
        top = df.sort_values("is_exp", ascending=False).head(20)
    else:
        print(f"{len(robust)} robust combinations (profitable IS + OOS, PF > 1, min trades met)\n")
        top = robust.head(30)

    header = (
        f"{'SL':>5s} {'TP':>5s} | "
        f"{'IS Trd':>6s} {'IS WR%':>6s} {'IS Exp':>7s} {'IS PF':>6s} {'IS PnL':>9s} {'IS DD':>9s} | "
        f"{'OOS Trd':>7s} {'OOS WR%':>7s} {'OOS Exp':>7s} {'OOS PF':>6s} {'OOS PnL':>9s} {'OOS DD':>9s}"
    )
    print(header)
    print("-" * len(header))

    for _, r in top.iterrows():
        print(
            f"{r['sl']:5.2f} {r['tp']:5.2f} | "
            f"{r['is_trades']:6.0f} {r['is_wr']:5.1f}% {r['is_exp']:+7.2f} {r['is_pf']:6.2f} {r['is_pnl']:+9.1f} {r['is_dd']:9.1f} | "
            f"{r['oos_trades']:7.0f} {r['oos_wr']:6.1f}% {r['oos_exp']:+7.2f} {r['oos_pf']:6.2f} {r['oos_pnl']:+9.1f} {r['oos_dd']:9.1f}"
        )

    # ── Best overall pick ──────────────────────────────────────────────────
    if not robust.empty:
        best = robust.iloc[0]
        print(f"\n{'=' * 80}")
        print(f"RECOMMENDED: SL {best['sl']:.2f}x / TP {best['tp']:.2f}x ATR")
        print(f"{'=' * 80}")
        print(f"  IN-SAMPLE:      {best['is_trades']:.0f} trades | WR {best['is_wr']:.1f}% | "
              f"Exp {best['is_exp']:+.2f} pts | PF {best['is_pf']:.2f} | "
              f"PnL {best['is_pnl']:+.1f} pts (${best['is_pnl'] * POINT_VALUE:+,.0f}) | "
              f"DD {best['is_dd']:.1f} pts")
        print(f"  OUT-OF-SAMPLE:  {best['oos_trades']:.0f} trades | WR {best['oos_wr']:.1f}% | "
              f"Exp {best['oos_exp']:+.2f} pts | PF {best['oos_pf']:.2f} | "
              f"PnL {best['oos_pnl']:+.1f} pts (${best['oos_pnl'] * POINT_VALUE:+,.0f}) | "
              f"DD {best['oos_dd']:.1f} pts")

    # Save full grid
    out_path = Path("results/vwap_atr_grid.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nFull grid saved to {out_path}")


if __name__ == "__main__":
    main()

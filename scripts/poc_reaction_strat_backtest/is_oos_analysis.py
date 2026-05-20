"""IS/OOS detailed breakdown + 10-day POC comparison for SL 2.0 / TP 3.0."""

import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
POC_REACTION_CACHE_DIR = DATA_DIR / "poc_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0
MAX_MARTINGALE = 8

ENTRY_START = "09:00"
ENTRY_END = "11:00"
SL_MULT = 2.0
TP_MULT = 3.0


def compute_recent_cvd(bars, up_to_index, lookback=10):
    start_idx = max(0, up_to_index - lookback)
    cvd = 0.0
    for i in range(start_idx, min(up_to_index, len(bars))):
        bar = bars[i]
        if not bar.closed:
            continue
        for price, lv in bar.levels.items():
            cvd += lv.delta
    return cvd


def run_backtest():
    start = datetime.strptime("2025-03-13", "%Y-%m-%d").date()
    end = datetime.strptime("2026-04-08", "%Y-%m-%d").date()

    all_trades = []

    for cache_file in sorted(POC_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue

        with open(cache_file, "rb") as f:
            poc_data = pickle.load(f)

        signals = poc_data["signals"]
        if not signals:
            continue

        signal_cache_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_cache_file.exists():
            continue

        with open(signal_cache_file, "rb") as f:
            signal_data = pickle.load(f)

        bars = signal_data["bars"]

        filtered_signals = []
        for s in signals:
            if s["direction"] != "long":
                continue
            confirm_time = s["confirm_time"]
            if hasattr(confirm_time, "tz_convert"):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz="UTC").tz_convert(ET)
            hour_min = confirm_et.strftime("%H:%M")
            if ENTRY_START <= hour_min < ENTRY_END:
                filtered_signals.append(s)

        if not filtered_signals:
            continue

        in_trade = False
        for signal in filtered_signals:
            if in_trade:
                continue

            entry_price = signal["entry_price"]
            atr = signal["atr"]
            confirm_bar_idx = signal["bar_index"] + 1

            if atr is None or atr <= 0:
                continue

            cvd = compute_recent_cvd(bars, confirm_bar_idx + 1)
            if cvd >= 0:
                continue

            sl_price = entry_price - atr * SL_MULT
            tp_price = entry_price + atr * TP_MULT

            if tp_price <= entry_price or sl_price >= entry_price:
                continue

            in_trade = True
            exit_price = None
            exit_reason = None

            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                if bar.low <= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                    break
                if bar.high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "target"
                    break

            if exit_price is None:
                last_bar = bars[-1]
                exit_price = last_bar.close
                exit_reason = "eod"

            pnl_points = exit_price - entry_price
            pnl_dollars = pnl_points * POINT_VALUE

            all_trades.append({
                "date": file_date,
                "exit_reason": exit_reason,
                "pnl_points": pnl_points,
                "pnl_dollars": pnl_dollars,
            })
            in_trade = False

    return pd.DataFrame(all_trades)


def print_period(df, label):
    if df.empty:
        print(f"  {label}: 0 trades")
        return

    total = len(df)
    winners = df[df["pnl_dollars"] > 0]
    losers = df[df["pnl_dollars"] < 0]
    wr = len(winners) / total * 100

    avg_win = winners["pnl_dollars"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl_dollars"].mean() if len(losers) > 0 else 0

    gp = winners["pnl_dollars"].sum() if len(winners) > 0 else 0
    gl = abs(losers["pnl_dollars"].sum()) if len(losers) > 0 else 1
    pf = gp / gl if gl > 0 else 999

    total_pnl = df["pnl_dollars"].sum()
    exp = total_pnl / total

    cum = df["pnl_dollars"].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    tp = len(df[df["exit_reason"] == "target"])
    sl = len(df[df["exit_reason"] == "stop_loss"])
    eod = len(df[df["exit_reason"] == "eod"])

    # Martingale
    marty_cum = 0.0
    contracts = 1
    marty_vals = []
    for p in df["pnl_dollars"]:
        marty_cum += p * contracts
        marty_vals.append(marty_cum)
        contracts = 1 if p > 0 else min(contracts * 2, MAX_MARTINGALE)
    marty_total = marty_cum
    marr = np.array(marty_vals)
    marty_dd = float((marr - np.maximum.accumulate(marr)).min())

    # Monthly breakdown
    df_copy = df.copy()
    df_copy["month"] = df_copy["date"].apply(lambda d: d.strftime("%Y-%m"))
    monthly = df_copy.groupby("month").agg(
        trades=("pnl_dollars", "count"),
        pnl=("pnl_dollars", "sum"),
    )

    print(f"{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"Trades:      {total}")
    print(f"Win rate:    {wr:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"Avg win:     ${avg_win:+,.0f}")
    print(f"Avg loss:    ${avg_loss:+,.0f}")
    print(f"Exp/trade:   ${exp:+,.0f}")
    print(f"PF:          {pf:.2f}")
    print(f"Total P&L:   ${total_pnl:+,.0f}")
    print(f"Max DD:      ${max_dd:+,.0f}")
    print(f"Exits:       {tp} TP | {sl} SL | {eod} EOD")
    print(f"Martingale:  ${marty_total:+,.0f}  (DD: ${marty_dd:+,.0f})")
    print()
    print(f"  Monthly:")
    for month, row in monthly.iterrows():
        bar = "+" * max(0, int(row["pnl"] / 500)) + "-" * max(0, int(-row["pnl"] / 500))
        print(f"    {month}: {row['trades']:2.0f} trades | ${row['pnl']:+8,.0f} | {bar}")
    print()


if __name__ == "__main__":
    df = run_backtest()

    dates = sorted(df["date"].unique())
    mid = len(dates) // 2
    is_dates = set(dates[:mid])
    oos_dates = set(dates[mid:])

    is_df = df[df["date"].isin(is_dates)].reset_index(drop=True)
    oos_df = df[df["date"].isin(oos_dates)].reset_index(drop=True)

    print(f"Date split: IS {min(is_dates)} to {max(is_dates)} | OOS {min(oos_dates)} to {max(oos_dates)}")
    print()

    print_period(df, "FULL PERIOD (6-day POC)")
    print_period(is_df, "IN-SAMPLE (6-day POC)")
    print_period(oos_df, "OUT-OF-SAMPLE (6-day POC)")

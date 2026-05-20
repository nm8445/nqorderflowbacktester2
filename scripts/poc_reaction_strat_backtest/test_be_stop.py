"""Test breakeven stop rule at 70%, 80%, 90% of TP distance."""

import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
POC_REACTION_CACHE_DIR = DATA_DIR / "poc_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
TICK_SIZE = 0.25
TICK_VALUE = 5.0
POINT_VALUE = TICK_VALUE / TICK_SIZE

ENTRY_START = "09:00"
ENTRY_END = "11:00"


def run_with_be_stop(be_threshold: float):
    """Run backtest with breakeven stop triggered at be_threshold % of TP distance."""
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
            stop_loss = signal["stop_loss"]
            target = signal["target"]
            direction = signal["direction"]
            confirm_bar_idx = signal["bar_index"] + 1

            if direction == "short":
                if target >= entry_price or stop_loss <= entry_price:
                    continue
            else:
                if target <= entry_price or stop_loss >= entry_price:
                    continue

            in_trade = True
            exit_price = None
            exit_time = None
            exit_reason = None
            current_sl = stop_loss
            be_triggered = False

            # Calculate BE trigger level
            tp_dist = abs(target - entry_price)
            if direction == "short":
                be_trigger_price = entry_price - (tp_dist * be_threshold)
            else:
                be_trigger_price = entry_price + (tp_dist * be_threshold)

            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue

                if direction == "short":
                    # Check if BE trigger hit (price went low enough)
                    if not be_triggered and bar.low <= be_trigger_price:
                        be_triggered = True
                        current_sl = entry_price  # move stop to breakeven

                    # Check SL first (adverse)
                    if bar.high >= current_sl:
                        exit_price = current_sl
                        exit_time = bar.close_time
                        exit_reason = "breakeven" if be_triggered and current_sl == entry_price else "stop_loss"
                        break
                    # Check TP
                    if bar.low <= target:
                        exit_price = target
                        exit_time = bar.close_time
                        exit_reason = "target"
                        break
                else:  # long
                    # Check if BE trigger hit (price went high enough)
                    if not be_triggered and bar.high >= be_trigger_price:
                        be_triggered = True
                        current_sl = entry_price  # move stop to breakeven

                    # Check SL first (adverse)
                    if bar.low <= current_sl:
                        exit_price = current_sl
                        exit_time = bar.close_time
                        exit_reason = "breakeven" if be_triggered and current_sl == entry_price else "stop_loss"
                        break
                    # Check TP
                    if bar.high >= target:
                        exit_price = target
                        exit_time = bar.close_time
                        exit_reason = "target"
                        break

            if exit_price is None and in_trade:
                last_bar = bars[-1]
                exit_price = last_bar.close
                exit_time = last_bar.close_time
                exit_reason = "eod"

            if direction == "short":
                pnl_points = entry_price - exit_price
            else:
                pnl_points = exit_price - entry_price

            pnl_dollars = pnl_points * POINT_VALUE

            all_trades.append({
                "date": date_str,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_points": pnl_points,
                "pnl_dollars": pnl_dollars,
            })
            in_trade = False

    return pd.DataFrame(all_trades)


def print_results(df, label):
    if df.empty:
        print(f"{label}: No trades")
        return

    total = len(df)
    winners = df[df["pnl_points"] > 0]
    losers = df[df["pnl_points"] < 0]
    flat = df[df["pnl_points"] == 0]
    win_rate = len(winners) / total * 100

    avg_win = winners["pnl_points"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl_points"].mean() if len(losers) > 0 else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    gross_profit = winners["pnl_points"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_points"].sum()) if len(losers) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    cumulative = df["pnl_points"].cumsum()
    max_dd = (cumulative - cumulative.cummax()).min()

    tp = df[df["exit_reason"] == "target"]
    sl = df[df["exit_reason"] == "stop_loss"]
    be = df[df["exit_reason"] == "breakeven"]
    eod = df[df["exit_reason"] == "eod"]

    shorts = df[df["direction"] == "short"]
    longs = df[df["direction"] == "long"]

    print(f"{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"Total trades:    {total}")
    print(f"Winners:         {len(winners)} ({win_rate:.1f}%)")
    print(f"Losers:          {len(losers)} ({100 - win_rate - len(flat)/total*100:.1f}%)")
    print(f"Breakeven:       {len(flat)}")
    print()
    print(f"Avg win:         {avg_win:+.2f} pts (${avg_win * POINT_VALUE:+,.0f})")
    print(f"Avg loss:        {avg_loss:+.2f} pts (${avg_loss * POINT_VALUE:+,.0f})")
    print(f"Expectancy:      {expectancy:+.2f} pts/trade (${expectancy * POINT_VALUE:+,.0f})")
    print(f"Profit factor:   {profit_factor:.2f}")
    print()
    print(f"Total P&L:       {df['pnl_points'].sum():+.2f} pts (${df['pnl_dollars'].sum():+,.0f})")
    print(f"Max drawdown:    {max_dd:.2f} pts (${max_dd * POINT_VALUE:+,.0f})")
    print()
    print(f"--- By Exit ---")
    print(f"Target:          {len(tp)} ({len(tp)/total*100:.1f}%)")
    print(f"Stop loss:       {len(sl)} ({len(sl)/total*100:.1f}%)")
    print(f"Breakeven:       {len(be)} ({len(be)/total*100:.1f}%)")
    print(f"EOD:             {len(eod)} ({len(eod)/total*100:.1f}%)")
    print()
    print(f"--- By Direction ---")
    s_wr = len(shorts[shorts['pnl_points'] > 0]) / max(len(shorts), 1) * 100
    l_wr = len(longs[longs['pnl_points'] > 0]) / max(len(longs), 1) * 100
    print(f"Shorts:          {len(shorts)} trades | {shorts['pnl_points'].sum():+.2f} pts | WR: {s_wr:.1f}%")
    print(f"Longs:           {len(longs)} trades | {longs['pnl_points'].sum():+.2f} pts | WR: {l_wr:.1f}%")
    print()


if __name__ == "__main__":
    for threshold in [0.70, 0.80, 0.90]:
        df = run_with_be_stop(threshold)
        print_results(df, f"BE STOP AT {int(threshold*100)}% OF TP DISTANCE")

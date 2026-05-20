"""
POC Reaction Strategy Backtest.

Entry: absorption signal + confirmation at recent POC zones (last 10 days),
       at/above developing VAH (shorts) or at/below developing VAL (longs).
Exit:  target = developing POC at entry (locked), SL = signal bar high/low + ATR buffer.
Time:  9:00 AM - 11:00 AM ET only.
"""

import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
POC_REACTION_CACHE_DIR = DATA_DIR / "poc_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"

# NQ contract specs
TICK_SIZE = 0.25
TICK_VALUE = 5.0
POINT_VALUE = TICK_VALUE / TICK_SIZE  # $20 per point

# Session filter
ENTRY_START = "09:00"
ENTRY_END = "11:00"


def run_backtest(start_date: str = "2025-03-13", end_date: str = "2026-04-08"):
    """
    Run backtest using POC reaction cache + signal cache bars for trade simulation.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    all_trades = []

    for cache_file in sorted(POC_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue

        with open(cache_file, 'rb') as f:
            poc_data = pickle.load(f)

        signals = poc_data["signals"]
        if not signals:
            continue

        # Load signal cache for bar-by-bar simulation
        signal_cache_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_cache_file.exists():
            continue

        with open(signal_cache_file, 'rb') as f:
            signal_data = pickle.load(f)

        bars = signal_data["bars"]

        # Filter signals to entry window (9:00 - 11:00 AM ET)
        filtered_signals = []
        for s in signals:
            confirm_time = s["confirm_time"]
            if hasattr(confirm_time, 'tz_convert'):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz='UTC').tz_convert(ET)

            hour_min = confirm_et.strftime("%H:%M")
            if ENTRY_START <= hour_min < ENTRY_END:
                filtered_signals.append(s)

        if not filtered_signals:
            continue

        # Simulate trades: one at a time
        in_trade = False
        for signal in filtered_signals:
            if in_trade:
                continue

            entry_price = signal["entry_price"]
            stop_loss = signal["stop_loss"]
            target = signal["target"]
            direction = signal["direction"]
            confirm_bar_idx = signal["bar_index"] + 1  # confirm bar

            # Validate trade makes sense
            if direction == "short":
                if target >= entry_price or stop_loss <= entry_price:
                    continue
            else:  # long
                if target <= entry_price or stop_loss >= entry_price:
                    continue

            # Walk forward through bars after confirm bar
            in_trade = True
            exit_price = None
            exit_time = None
            exit_reason = None

            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue

                if direction == "short":
                    # Check SL first (adverse)
                    if bar.high >= stop_loss:
                        exit_price = stop_loss
                        exit_time = bar.close_time
                        exit_reason = "stop_loss"
                        break
                    # Check TP
                    if bar.low <= target:
                        exit_price = target
                        exit_time = bar.close_time
                        exit_reason = "target"
                        break
                else:  # long
                    # Check SL first (adverse)
                    if bar.low <= stop_loss:
                        exit_price = stop_loss
                        exit_time = bar.close_time
                        exit_reason = "stop_loss"
                        break
                    # Check TP
                    if bar.high >= target:
                        exit_price = target
                        exit_time = bar.close_time
                        exit_reason = "target"
                        break

            # If no exit found, close at last bar (EOD)
            if exit_price is None and in_trade:
                last_bar = bars[-1]
                exit_price = last_bar.close
                exit_time = last_bar.close_time
                exit_reason = "eod"

            # Calculate P&L
            if direction == "short":
                pnl_points = entry_price - exit_price
            else:
                pnl_points = exit_price - entry_price

            pnl_dollars = pnl_points * POINT_VALUE

            confirm_time = signal["confirm_time"]
            if hasattr(confirm_time, 'tz_convert'):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz='UTC').tz_convert(ET)

            if hasattr(exit_time, 'tz_convert'):
                exit_et = exit_time.tz_convert(ET)
            else:
                exit_et = pd.Timestamp(exit_time, tz='UTC').tz_convert(ET)

            trade = {
                "date": date_str,
                "direction": direction,
                "entry_time": confirm_et,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "exit_time": exit_et,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_points": pnl_points,
                "pnl_dollars": pnl_dollars,
                "poc_source_date": signal["poc_source_date"],
                "prev_poc": signal["prev_poc"],
                "dev_vah": signal["dev_vah"],
                "dev_val": signal["dev_val"],
                "dev_poc": signal["dev_poc"],
                "atr": signal["atr"],
            }

            all_trades.append(trade)
            in_trade = False

    # Results
    if not all_trades:
        print("No trades found.")
        return

    df = pd.DataFrame(all_trades)

    total = len(df)
    winners = df[df["pnl_points"] > 0]
    losers = df[df["pnl_points"] < 0]
    breakeven = df[df["pnl_points"] == 0]

    win_rate = len(winners) / total * 100
    avg_win = winners["pnl_points"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl_points"].mean() if len(losers) > 0 else 0
    total_pnl = df["pnl_points"].sum()
    total_dollars = df["pnl_dollars"].sum()

    # Expectancy
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # Profit factor
    gross_profit = winners["pnl_points"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_points"].sum()) if len(losers) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    cumulative = df["pnl_points"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()

    # By direction
    shorts = df[df["direction"] == "short"]
    longs = df[df["direction"] == "long"]

    # By exit reason
    sl_exits = df[df["exit_reason"] == "stop_loss"]
    tp_exits = df[df["exit_reason"] == "target"]
    eod_exits = df[df["exit_reason"] == "eod"]

    print("=" * 80)
    print("POC REACTION STRATEGY — BACKTEST RESULTS")
    print("=" * 80)
    print(f"Period:          {start_date} to {end_date}")
    print(f"Entry window:    {ENTRY_START} - {ENTRY_END} ET")
    print()
    print(f"Total trades:    {total}")
    print(f"Winners:         {len(winners)} ({win_rate:.1f}%)")
    print(f"Losers:          {len(losers)} ({100 - win_rate:.1f}%)")
    print(f"Breakeven:       {len(breakeven)}")
    print()
    print(f"Avg win:         {avg_win:+.2f} pts (${avg_win * POINT_VALUE:+,.0f})")
    print(f"Avg loss:        {avg_loss:+.2f} pts (${avg_loss * POINT_VALUE:+,.0f})")
    print(f"Expectancy:      {expectancy:+.2f} pts/trade (${expectancy * POINT_VALUE:+,.0f})")
    print(f"Profit factor:   {profit_factor:.2f}")
    print()
    print(f"Total P&L:       {total_pnl:+.2f} pts (${total_dollars:+,.0f})")
    print(f"Max drawdown:    {max_dd:.2f} pts (${max_dd * POINT_VALUE:+,.0f})")
    print()
    print(f"--- By Direction ---")
    print(f"Shorts:          {len(shorts)} trades | {shorts['pnl_points'].sum():+.2f} pts | WR: {len(shorts[shorts['pnl_points'] > 0]) / max(len(shorts), 1) * 100:.1f}%")
    print(f"Longs:           {len(longs)} trades | {longs['pnl_points'].sum():+.2f} pts | WR: {len(longs[longs['pnl_points'] > 0]) / max(len(longs), 1) * 100:.1f}%")
    print()
    print(f"--- By Exit ---")
    print(f"Target:          {len(tp_exits)} ({len(tp_exits)/total*100:.1f}%)")
    print(f"Stop loss:       {len(sl_exits)} ({len(sl_exits)/total*100:.1f}%)")
    print(f"EOD:             {len(eod_exits)} ({len(eod_exits)/total*100:.1f}%)")
    print()

    # Top 5 winners and losers
    print(f"--- Top 5 Winners ---")
    for _, t in df.nlargest(5, "pnl_points").iterrows():
        print(f"  {t['date']} {t['direction']:5s} | entry:{t['entry_price']:.2f} exit:{t['exit_price']:.2f} | {t['pnl_points']:+.2f} pts | {t['exit_reason']}")

    print(f"--- Top 5 Losers ---")
    for _, t in df.nsmallest(5, "pnl_points").iterrows():
        print(f"  {t['date']} {t['direction']:5s} | entry:{t['entry_price']:.2f} exit:{t['exit_price']:.2f} | {t['pnl_points']:+.2f} pts | {t['exit_reason']}")

    print("=" * 80)

    return df


if __name__ == "__main__":
    trades_df = run_backtest()

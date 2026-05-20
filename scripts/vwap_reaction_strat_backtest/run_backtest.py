"""
VWAP Reaction Continuation Strategy Backtest.

Entry: absorption signal + confirmation at VWAP zone (+/-3 pts),
       5-min bias confirms direction, price approached from trend side.
Exit:  SL/TP based on ATR multiples (recomputed from stored ATR + signal bar extremes).
Time:  7pm ET to 4:59pm ET (entry window filtered in cache).
"""

import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"

# NQ contract specs
TICK_SIZE = 0.25
TICK_VALUE = 5.0
POINT_VALUE = TICK_VALUE / TICK_SIZE  # $20 per point

# Default ATR multiples (override via function args)
SL_MULT = 1.0
TP_MULT = 2.0

# Prop firm session rules
ENTRY_CUTOFF = "16:00"   # no new entries after 4:00 PM ET
FORCE_CLOSE  = "16:58"   # flatten all positions at 4:58 PM ET


def run_backtest(
    start_date: str = "2025-03-13",
    end_date: str = "2026-04-08",
    sl_mult: float = SL_MULT,
    tp_mult: float = TP_MULT,
    direction_filter: str | None = None,
):
    """
    Run backtest using VWAP reaction cache + signal cache bars for trade simulation.

    Parameters
    ----------
    direction_filter : 'long', 'short', or None (both)
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    all_trades = []

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

        # Load signal cache for bar-by-bar simulation
        signal_cache_file = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not signal_cache_file.exists():
            continue

        with open(signal_cache_file, 'rb') as f:
            signal_data = pickle.load(f)

        bars = signal_data["bars"]

        # Filter by direction if requested
        filtered_signals = signals
        if direction_filter:
            filtered_signals = [s for s in signals if s["direction"] == direction_filter]

        if not filtered_signals:
            continue

        # Simulate trades: one at a time (no overlapping positions)
        last_exit_time = None
        for signal in filtered_signals:
            entry_price = signal["entry_price"]
            direction = signal["direction"]
            atr = signal["atr"]
            confirm_bar_idx = signal["bar_index"] + 1

            if atr is None or atr <= 0:
                continue

            # No new entries after 4:50 PM ET
            confirm_time = signal["confirm_time"]
            if hasattr(confirm_time, 'tz_convert'):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz='UTC').tz_convert(ET)
            if "16:00" <= confirm_et.strftime("%H:%M") < "19:10":
                continue

            # Skip if previous trade hasn't exited yet
            if last_exit_time is not None and confirm_time <= last_exit_time:
                continue

            # Recompute SL/TP from ATR multiples (allows grid sweeps without cache rebuild)
            if direction == "long":
                sl_price = entry_price - atr * sl_mult
                tp_price = entry_price + atr * tp_mult
                if tp_price <= entry_price or sl_price >= entry_price:
                    continue
            else:  # short
                sl_price = entry_price + atr * sl_mult
                tp_price = entry_price - atr * tp_mult
                if tp_price >= entry_price or sl_price <= entry_price:
                    continue

            # Walk forward through bars after confirm bar
            exit_price = None
            exit_time = None
            exit_reason = None

            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue

                # Force close at 4:58 PM ET
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
                else:  # long
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

            # If no exit found, close at last bar (EOD)
            if exit_price is None:
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
                "stop_loss": sl_price,
                "target": tp_price,
                "exit_time": exit_et,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_points": pnl_points,
                "pnl_dollars": pnl_dollars,
                "vwap": signal["vwap"],
                "bias": signal["bias"],
                "atr": atr,
            }

            all_trades.append(trade)
            last_exit_time = exit_time

    # Results
    if not all_trades:
        print("No trades found.")
        return pd.DataFrame()

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

    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    gross_profit = winners["pnl_points"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_points"].sum()) if len(losers) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

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
    sess_exits = df[df["exit_reason"] == "session_close"]

    dir_label = direction_filter.upper() if direction_filter else "BOTH"
    print("=" * 80)
    print(f"VWAP REACTION STRATEGY - BACKTEST RESULTS (SL {sl_mult}x / TP {tp_mult}x ATR, {dir_label})")
    print("=" * 80)
    print(f"Period:          {start_date} to {end_date}")
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
    if len(shorts) > 0:
        s_wr = len(shorts[shorts['pnl_points'] > 0]) / len(shorts) * 100
        print(f"Shorts:          {len(shorts)} trades | {shorts['pnl_points'].sum():+.2f} pts | WR: {s_wr:.1f}%")
    else:
        print(f"Shorts:          0 trades")
    if len(longs) > 0:
        l_wr = len(longs[longs['pnl_points'] > 0]) / len(longs) * 100
        print(f"Longs:           {len(longs)} trades | {longs['pnl_points'].sum():+.2f} pts | WR: {l_wr:.1f}%")
    else:
        print(f"Longs:           0 trades")
    print()
    print(f"--- By Exit ---")
    print(f"Target:          {len(tp_exits)} ({len(tp_exits)/total*100:.1f}%)")
    print(f"Stop loss:       {len(sl_exits)} ({len(sl_exits)/total*100:.1f}%)")
    print(f"Session close:   {len(sess_exits)} ({len(sess_exits)/total*100:.1f}%)")
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

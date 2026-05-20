"""
Passive entry test — after confirmation bar closes, place limit/stop at
confirmation close price instead of market order.

For each signal:
  1. Entry price = confirmation bar close (same as current)
  2. Walk subsequent bars — does price come back to entry BEFORE SL/TP is hit?
  3. If yes: trade fills at exact entry price (zero slippage)
  4. If no: trade is missed (price ran away)

Also tracks HOW MANY bars later the fill happens (1 = next bar, which is
nearly instant since range bars close fast during active markets).
"""

import pickle
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
VWAP_REACTION_CACHE_DIR = DATA_DIR / "vwap_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0

SL_MULT = 1.0
TP_MULT = 2.0
ENTRY_CUTOFF = "16:00"
FORCE_CLOSE  = "16:58"

FULL_START = "2025-03-13"
FULL_END   = "2026-04-08"


def run_test():
    start = datetime.strptime(FULL_START, "%Y-%m-%d").date()
    end = datetime.strptime(FULL_END, "%Y-%m-%d").date()

    filled_trades = []
    missed_trades = []
    fill_bar_delays = []  # how many bars after confirm before fill

    for cache_file in sorted(VWAP_REACTION_CACHE_DIR.glob("*.pkl")):
        date_str = cache_file.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if file_date < start or file_date > end:
            continue
        with open(cache_file, 'rb') as f:
            vd = pickle.load(f)
        signals = vd["signals"]
        if not signals:
            continue
        scf = SIGNAL_CACHE_DIR / f"{date_str}.pkl"
        if not scf.exists():
            continue
        with open(scf, 'rb') as f:
            sd = pickle.load(f)
        bars = sd["bars"]

        last_exit_time = None

        for signal in signals:
            ep = signal["entry_price"]
            d = signal["direction"]
            atr = signal["atr"]
            cbi = signal["bar_index"] + 1

            if atr is None or atr <= 0:
                continue
            ct = signal["confirm_time"]
            if hasattr(ct, 'tz_convert'):
                cet = ct.tz_convert(ET)
            else:
                cet = pd.Timestamp(ct, tz='UTC').tz_convert(ET)
            if "16:00" <= cet.strftime("%H:%M") < "19:10":
                continue
            if last_exit_time is not None and ct <= last_exit_time:
                continue

            if d == "long":
                sl = ep - atr * SL_MULT
                tp = ep + atr * TP_MULT
                if tp <= ep or sl >= ep: continue
            else:
                sl = ep + atr * SL_MULT
                tp = ep - atr * TP_MULT
                if tp >= ep or sl <= ep: continue

            # Now simulate passive entry:
            # Walk bars after confirmation. Check if entry price is touched
            # BEFORE SL or TP would be hit (checking SL/TP from the entry price).
            #
            # Key distinction: the order hasn't filled yet, so SL/TP aren't active.
            # But if price reaches SL/TP level BEFORE coming back to entry,
            # the trade setup is invalidated (price already moved the full SL/TP
            # distance without filling us).

            order_filled = False
            fill_bar_idx = 0
            exit_price = None
            exit_time = None

            for j in range(cbi + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue

                bct = bar.close_time
                if hasattr(bct, 'tz_convert'):
                    bet = bct.tz_convert(ET)
                else:
                    bet = pd.Timestamp(bct, tz='UTC').tz_convert(ET)

                if not order_filled:
                    # Order not yet filled — check if this bar touches entry price
                    # AND check if SL/TP level is hit first within this bar

                    entry_touched = bar.low <= ep <= bar.high

                    if d == "long":
                        # SL is below entry, TP is above entry
                        # If bar.low hits SL before entry is touched, setup invalidated
                        sl_hit = bar.low <= sl
                        tp_hit = bar.high >= tp
                    else:
                        sl_hit = bar.high >= sl
                        tp_hit = bar.low <= tp

                    if entry_touched and not sl_hit and not tp_hit:
                        # Entry filled, SL/TP not yet hit in this bar
                        order_filled = True
                        fill_bar_idx = j - cbi
                        # Now continue to find exit
                    elif entry_touched and (sl_hit or tp_hit):
                        # Both entry and SL/TP touched in same bar
                        # Conservative: assume entry fills first (price came back
                        # to entry, then continued to SL/TP)
                        # This is the same as current backtest behavior
                        order_filled = True
                        fill_bar_idx = j - cbi
                        # Exit happens in this same bar
                        if sl_hit:
                            exit_price = sl
                            exit_time = bct
                        else:
                            exit_price = tp
                            exit_time = bct
                        break
                    elif sl_hit or tp_hit:
                        # SL or TP hit without entry being touched — MISSED
                        break

                    # Force close check while waiting for fill
                    if bet.strftime("%H:%M") >= FORCE_CLOSE:
                        # Session ending, order never filled
                        break

                else:
                    # Order filled — now run normal SL/TP exit logic
                    if bet.strftime("%H:%M") >= FORCE_CLOSE:
                        exit_price = bar.close
                        exit_time = bct
                        break
                    if d == "short":
                        if bar.high >= sl:
                            exit_price = sl; exit_time = bct; break
                        if bar.low <= tp:
                            exit_price = tp; exit_time = bct; break
                    else:
                        if bar.low <= sl:
                            exit_price = sl; exit_time = bct; break
                        if bar.high >= tp:
                            exit_price = tp; exit_time = bct; break

            if order_filled:
                if exit_price is None:
                    exit_price = bars[-1].close
                    exit_time = bars[-1].close_time

                pnl = (ep - exit_price) if d == "short" else (exit_price - ep)
                filled_trades.append({
                    "date": date_str,
                    "direction": d,
                    "entry": ep,
                    "exit": exit_price,
                    "pnl": pnl,
                    "fill_delay_bars": fill_bar_idx,
                })
                fill_bar_delays.append(fill_bar_idx)
                last_exit_time = exit_time
            else:
                # Trade missed — price ran away
                missed_trades.append({
                    "date": date_str,
                    "direction": d,
                    "entry": ep,
                })
                # Don't update last_exit_time — we never entered

    return filled_trades, missed_trades, fill_bar_delays


def main():
    print("Running passive entry test (4pm cutoff)...\n", flush=True)
    filled, missed, delays = run_test()

    total = len(filled) + len(missed)
    fill_rate = len(filled) / total * 100 if total > 0 else 0
    miss_rate = len(missed) / total * 100 if total > 0 else 0

    print("=" * 90)
    print("PASSIVE ENTRY TEST — Limit/Stop at confirmation close")
    print("=" * 90)
    print(f"Total signals:     {total}")
    print(f"Filled:            {len(filled)} ({fill_rate:.1f}%)")
    print(f"Missed:            {len(missed)} ({miss_rate:.1f}%)")
    print()

    # Stats on filled trades
    if filled:
        pnl_arr = np.array([t["pnl"] for t in filled])
        w = pnl_arr[pnl_arr > 0]
        l = pnl_arr[pnl_arr < 0]
        wr = len(w) / len(pnl_arr) * 100
        gp = w.sum() if len(w) else 0
        gl = abs(l.sum()) if len(l) else 1
        pf = gp / gl if gl > 0 else 999
        exp = pnl_arr.mean()
        total_pnl = pnl_arr.sum()

        print("FILLED TRADE STATS (zero slippage — exact entry at confirm close):")
        print(f"  Trades:      {len(filled)}")
        print(f"  Win rate:    {wr:.1f}%")
        print(f"  PF:          {pf:.2f}")
        print(f"  Expectancy:  {exp:+.2f} pts (${exp * POINT_VALUE:+.0f})")
        print(f"  Total P&L:   {total_pnl:+.1f} pts (${total_pnl * POINT_VALUE:+,.0f})")
        print()

    # Compare to market order (all 446 trades)
    print("COMPARISON TO MARKET ORDER (all signals, potential slippage):")
    print(f"  Market order:  {total} trades | PF 1.54 | Total $+72,123")
    if filled:
        print(f"  Passive order: {len(filled)} trades | PF {pf:.2f} | Total ${total_pnl * POINT_VALUE:+,.0f}")
        diff_trades = total - len(filled)
        print(f"  Trades lost:   {diff_trades} ({miss_rate:.1f}%)")
    print()

    # Fill delay distribution
    if delays:
        delay_counts = Counter(delays)
        print("FILL DELAY (bars after confirmation):")
        for bars_delay in sorted(delay_counts.keys()):
            count = delay_counts[bars_delay]
            pct = count / len(delays) * 100
            bar_str = "+" * int(pct)
            print(f"  Bar +{bars_delay:>2d}: {count:4d} ({pct:5.1f}%) {bar_str}")

        print(f"\n  Mean delay:   {np.mean(delays):.1f} bars")
        print(f"  Median delay: {np.median(delays):.0f} bars")
        pct_immediate = delay_counts.get(1, 0) / len(delays) * 100
        print(f"  Filled on next bar: {delay_counts.get(1, 0)} ({pct_immediate:.1f}%)")
    print()

    # What did the missed trades look like? Were they winners or just ran away?
    if missed:
        print("MISSED TRADES ANALYSIS:")
        long_missed = sum(1 for t in missed if t["direction"] == "long")
        short_missed = sum(1 for t in missed if t["direction"] == "short")
        print(f"  Long missed:  {long_missed}")
        print(f"  Short missed: {short_missed}")

        # Show a few examples
        print(f"\n  Examples (first 10):")
        for t in missed[:10]:
            print(f"    {t['date']} {t['direction']:>5s} entry={t['entry']:.2f} -- price ran away")
    print()

    # Net impact: if passive entry saves 1-2 ticks slippage on filled trades
    # but misses some trades, what's the net?
    if filled:
        print("NET IMPACT ANALYSIS:")
        for slip_ticks in [1, 2, 3]:
            slip_cost = slip_ticks * 0.25 * 2  # round-trip
            # Market order: all trades but with slippage
            market_pnl = 3606.16 - (total * slip_cost)  # from backtest total
            # Passive: filled trades with zero slippage
            passive_pnl = total_pnl
            advantage = passive_pnl - market_pnl
            print(f"  At {slip_ticks}-tick slippage: Market ${market_pnl * POINT_VALUE:+,.0f} vs "
                  f"Passive ${passive_pnl * POINT_VALUE:+,.0f} -> "
                  f"Passive {'saves' if advantage > 0 else 'loses'} ${abs(advantage) * POINT_VALUE:,.0f}")


if __name__ == "__main__":
    main()

"""Test CVD filter variants on POC reaction strategy."""

import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
POC_REACTION_CACHE_DIR = DATA_DIR / "poc_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0

ENTRY_START = "09:00"
ENTRY_END = "11:00"


def compute_recent_cvd(bars, up_to_index, lookback):
    start_idx = max(0, up_to_index - lookback)
    cvd = 0.0
    for i in range(start_idx, min(up_to_index, len(bars))):
        bar = bars[i]
        if not bar.closed:
            continue
        for price, lv in bar.levels.items():
            cvd += lv.delta
    return cvd


def run_with_cvd_filter(filter_mode, cvd_lookback=10):
    """
    filter_mode:
      "none"          - no filter (baseline)
      "longs_only"    - only take longs, require negative CVD
      "contra_both"   - both directions, require CVD against trade direction
      "contra_longs"  - longs require negative CVD, shorts unfiltered
      "contra_moderate_longs" - longs require CVD between -200 and -50
    """
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

            # Compute CVD
            cvd = compute_recent_cvd(bars, confirm_bar_idx + 1, cvd_lookback)

            # Apply filter
            if filter_mode == "longs_only":
                if direction == "short":
                    continue
                if cvd >= 0:
                    continue
            elif filter_mode == "contra_both":
                if direction == "short" and cvd <= 0:
                    continue
                if direction == "long" and cvd >= 0:
                    continue
            elif filter_mode == "contra_longs":
                if direction == "long" and cvd >= 0:
                    continue
            elif filter_mode == "contra_moderate_longs":
                if direction == "long" and not (-200 <= cvd < -50):
                    continue
            elif filter_mode == "contra_longs_no_shorts":
                if direction == "short":
                    continue
                if cvd >= 0:
                    continue

            in_trade = True
            exit_price = None
            exit_reason = None

            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                if direction == "short":
                    if bar.high >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                        break
                    if bar.low <= target:
                        exit_price = target
                        exit_reason = "target"
                        break
                else:
                    if bar.low <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                        break
                    if bar.high >= target:
                        exit_price = target
                        exit_reason = "target"
                        break

            if exit_price is None:
                last_bar = bars[-1]
                exit_price = last_bar.close
                exit_reason = "eod"

            if direction == "short":
                pnl_points = entry_price - exit_price
            else:
                pnl_points = exit_price - entry_price

            all_trades.append({
                "date": date_str,
                "direction": direction,
                "pnl_points": pnl_points,
                "pnl_dollars": pnl_points * POINT_VALUE,
                "exit_reason": exit_reason,
            })
            in_trade = False

    return pd.DataFrame(all_trades)


def print_results(df, label):
    if df.empty or len(df) == 0:
        print(f"  {label}: 0 trades")
        print()
        return

    total = len(df)
    winners = df[df["pnl_points"] > 0]
    losers = df[df["pnl_points"] < 0]
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
    eod = df[df["exit_reason"] == "eod"]

    shorts = df[df["direction"] == "short"]
    longs = df[df["direction"] == "long"]

    print(f"{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"Trades:      {total}")
    print(f"Win rate:    {win_rate:.1f}%")
    print(f"Avg win:     {avg_win:+.2f} pts (${avg_win * POINT_VALUE:+,.0f})")
    print(f"Avg loss:    {avg_loss:+.2f} pts (${avg_loss * POINT_VALUE:+,.0f})")
    print(f"Expectancy:  {expectancy:+.2f} pts/trade (${expectancy * POINT_VALUE:+,.0f})")
    print(f"PF:          {profit_factor:.2f}")
    print(f"Total P&L:   {df['pnl_points'].sum():+.2f} pts (${df['pnl_dollars'].sum():+,.0f})")
    print(f"Max DD:      {max_dd:.2f} pts (${max_dd * POINT_VALUE:+,.0f})")
    print()
    print(f"Target: {len(tp)} | SL: {len(sl)} | EOD: {len(eod)}")
    if len(shorts) > 0:
        s_wr = len(shorts[shorts['pnl_points'] > 0]) / len(shorts) * 100
        print(f"Shorts: {len(shorts)} | {shorts['pnl_points'].sum():+.1f} pts | WR: {s_wr:.1f}%")
    if len(longs) > 0:
        l_wr = len(longs[longs['pnl_points'] > 0]) / len(longs) * 100
        print(f"Longs:  {len(longs)} | {longs['pnl_points'].sum():+.1f} pts | WR: {l_wr:.1f}%")
    print()


if __name__ == "__main__":
    configs = [
        ("none", "BASELINE (no filter)"),
        ("contra_longs", "CONTRA CVD LONGS + all shorts"),
        ("contra_both", "CONTRA CVD both directions"),
        ("longs_only", "LONGS ONLY + contra CVD"),
        ("contra_longs_no_shorts", "LONGS ONLY + contra CVD (no shorts)"),
        ("contra_moderate_longs", "MODERATE CONTRA CVD longs (-200 to -50) + all shorts"),
    ]

    for mode, label in configs:
        df = run_with_cvd_filter(mode)
        print_results(df, label)

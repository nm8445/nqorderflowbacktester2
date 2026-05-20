"""Analyze SL/TP hit rates and speed."""

import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
POC_REACTION_CACHE_DIR = DATA_DIR / "poc_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0

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
        if "09:00" <= hour_min < "11:00":
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
        bars_to_exit = 0

        for j in range(confirm_bar_idx + 1, len(bars)):
            bar = bars[j]
            if not bar.closed:
                continue
            bars_to_exit += 1

            if direction == "short":
                if bar.high >= stop_loss:
                    exit_price = stop_loss
                    exit_time = bar.close_time
                    exit_reason = "stop_loss"
                    break
                if bar.low <= target:
                    exit_price = target
                    exit_time = bar.close_time
                    exit_reason = "target"
                    break
            else:
                if bar.low <= stop_loss:
                    exit_price = stop_loss
                    exit_time = bar.close_time
                    exit_reason = "stop_loss"
                    break
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

        confirm_time = signal["confirm_time"]
        if hasattr(confirm_time, "tz_convert"):
            confirm_et = confirm_time.tz_convert(ET)
        else:
            confirm_et = pd.Timestamp(confirm_time, tz="UTC").tz_convert(ET)

        if hasattr(exit_time, "tz_convert"):
            exit_et = exit_time.tz_convert(ET)
        else:
            exit_et = pd.Timestamp(exit_time, tz="UTC").tz_convert(ET)

        duration_mins = (exit_et - confirm_et).total_seconds() / 60
        target_dist = abs(target - entry_price)
        sl_dist = abs(stop_loss - entry_price)

        all_trades.append({
            "date": date_str,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "exit_reason": exit_reason,
            "pnl_points": pnl_points,
            "bars_to_exit": bars_to_exit,
            "duration_mins": duration_mins,
            "target_dist": target_dist,
            "sl_dist": sl_dist,
            "rr_ratio": target_dist / sl_dist if sl_dist > 0 else 0,
        })
        in_trade = False

df = pd.DataFrame(all_trades)

sl = df[df["exit_reason"] == "stop_loss"]
tp = df[df["exit_reason"] == "target"]
eod = df[df["exit_reason"] == "eod"]

print("=" * 80)
print("EXIT ANALYSIS")
print("=" * 80)
print(f"Target hits:   {len(tp)} ({len(tp)/len(df)*100:.1f}%)")
print(f"Stop losses:   {len(sl)} ({len(sl)/len(df)*100:.1f}%)")
print(f"EOD exits:     {len(eod)} ({len(eod)/len(df)*100:.1f}%)")
print()

print("--- STOP LOSS SPEED ---")
print(f"Avg bars to SL:     {sl['bars_to_exit'].mean():.1f}")
print(f"Median bars to SL:  {sl['bars_to_exit'].median():.0f}")
print(f"Avg mins to SL:     {sl['duration_mins'].mean():.1f}")
print(f"Median mins to SL:  {sl['duration_mins'].median():.1f}")
print()

print("SL hit within (bars):")
for n in [1, 2, 3, 5, 10, 20]:
    count = len(sl[sl["bars_to_exit"] <= n])
    print(f"  {n:2d} bars: {count:3d} ({count/len(sl)*100:.1f}%)")
print()

print("SL hit within (time):")
for m in [1, 5, 10, 30, 60, 120]:
    count = len(sl[sl["duration_mins"] <= m])
    print(f"  {m:3d} min: {count:3d} ({count/len(sl)*100:.1f}%)")
print()

print("--- TARGET SPEED ---")
print(f"Avg bars to TP:     {tp['bars_to_exit'].mean():.1f}")
print(f"Median bars to TP:  {tp['bars_to_exit'].median():.0f}")
print(f"Avg mins to TP:     {tp['duration_mins'].mean():.1f}")
print(f"Median mins to TP:  {tp['duration_mins'].median():.1f}")
print()

print("TP hit within (bars):")
for n in [1, 2, 3, 5, 10, 20, 50, 100]:
    count = len(tp[tp["bars_to_exit"] <= n])
    print(f"  {n:3d} bars: {count:3d} ({count/len(tp)*100:.1f}%)")
print()

print("--- R:R RATIO AT ENTRY ---")
print(f"Avg target dist:    {df['target_dist'].mean():.1f} pts")
print(f"Avg SL dist:        {df['sl_dist'].mean():.1f} pts")
print(f"Avg R:R ratio:      {df['rr_ratio'].mean():.2f}")
print(f"Median R:R ratio:   {df['rr_ratio'].median():.2f}")
print()

print("--- IMMEDIATE STOPS (<=3 bars) ---")
imm = sl[sl["bars_to_exit"] <= 3]
print(f"Count: {len(imm)} of {len(sl)} SLs ({len(imm)/len(sl)*100:.1f}%)")
print(f"  Shorts: {len(imm[imm['direction']=='short'])}")
print(f"  Longs:  {len(imm[imm['direction']=='long'])}")
print()

print("--- BY DIRECTION ---")
sl_short = sl[sl["direction"] == "short"]
sl_long = sl[sl["direction"] == "long"]
tp_short = tp[tp["direction"] == "short"]
tp_long = tp[tp["direction"] == "long"]
print(f"Short SLs: {len(sl_short)} | avg {sl_short['bars_to_exit'].mean():.1f} bars | med {sl_short['duration_mins'].median():.1f} min")
print(f"Short TPs: {len(tp_short)} | avg {tp_short['bars_to_exit'].mean():.1f} bars | med {tp_short['duration_mins'].median():.1f} min")
print(f"Long SLs:  {len(sl_long)} | avg {sl_long['bars_to_exit'].mean():.1f} bars | med {sl_long['duration_mins'].median():.1f} min")
print(f"Long TPs:  {len(tp_long)} | avg {tp_long['bars_to_exit'].mean():.1f} bars | med {tp_long['duration_mins'].median():.1f} min")

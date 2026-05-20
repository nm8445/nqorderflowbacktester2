"""Analyze whether cumulative volume delta at entry time predicts trade success."""

import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
POC_REACTION_CACHE_DIR = DATA_DIR / "poc_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0

ENTRY_START = "09:00"
ENTRY_END = "11:00"


def compute_cvd_from_bars(bars, up_to_index):
    """Compute cumulative volume delta from bar 0 up to (not including) up_to_index."""
    cvd = 0.0
    for i in range(min(up_to_index, len(bars))):
        bar = bars[i]
        if not bar.closed:
            continue
        for price, lv in bar.levels.items():
            cvd += lv.delta
    return cvd


def compute_recent_cvd(bars, up_to_index, lookback=20):
    """Compute CVD over the last N bars before entry."""
    start_idx = max(0, up_to_index - lookback)
    cvd = 0.0
    for i in range(start_idx, min(up_to_index, len(bars))):
        bar = bars[i]
        if not bar.closed:
            continue
        for price, lv in bar.levels.items():
            cvd += lv.delta
    return cvd


def run_analysis():
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

            # Compute CVD metrics at entry
            session_cvd = compute_cvd_from_bars(bars, confirm_bar_idx + 1)
            recent_cvd_20 = compute_recent_cvd(bars, confirm_bar_idx + 1, lookback=20)
            recent_cvd_10 = compute_recent_cvd(bars, confirm_bar_idx + 1, lookback=10)
            recent_cvd_5 = compute_recent_cvd(bars, confirm_bar_idx + 1, lookback=5)

            # Simulate trade
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

            win = pnl_points > 0

            all_trades.append({
                "date": date_str,
                "direction": direction,
                "pnl_points": pnl_points,
                "win": win,
                "exit_reason": exit_reason,
                "session_cvd": session_cvd,
                "cvd_20": recent_cvd_20,
                "cvd_10": recent_cvd_10,
                "cvd_5": recent_cvd_5,
            })
            in_trade = False

    df = pd.DataFrame(all_trades)

    print("=" * 80)
    print("CVD vs TRADE OUTCOME ANALYSIS")
    print("=" * 80)
    print(f"Total trades: {len(df)}")
    print()

    # --- CVD alignment analysis ---
    # For shorts: negative CVD = aligned (selling pressure), positive = against
    # For longs: positive CVD = aligned (buying pressure), negative = against
    for cvd_col, label in [
        ("session_cvd", "Session CVD (all bars)"),
        ("cvd_20", "Recent CVD (20 bars)"),
        ("cvd_10", "Recent CVD (10 bars)"),
        ("cvd_5", "Recent CVD (5 bars)"),
    ]:
        print(f"--- {label} ---")

        # Create alignment flag
        df["aligned"] = False
        df.loc[(df["direction"] == "short") & (df[cvd_col] < 0), "aligned"] = True
        df.loc[(df["direction"] == "long") & (df[cvd_col] > 0), "aligned"] = True

        aligned = df[df["aligned"]]
        against = df[~df["aligned"]]

        a_wr = aligned["win"].mean() * 100 if len(aligned) > 0 else 0
        g_wr = against["win"].mean() * 100 if len(against) > 0 else 0
        a_pnl = aligned["pnl_points"].sum()
        g_pnl = against["pnl_points"].sum()
        a_exp = aligned["pnl_points"].mean() if len(aligned) > 0 else 0
        g_exp = against["pnl_points"].mean() if len(against) > 0 else 0

        print(f"  Aligned:   {len(aligned):3d} trades | WR: {a_wr:5.1f}% | P&L: {a_pnl:+8.1f} pts | Exp: {a_exp:+.2f} pts/trade")
        print(f"  Against:   {len(against):3d} trades | WR: {g_wr:5.1f}% | P&L: {g_pnl:+8.1f} pts | Exp: {g_exp:+.2f} pts/trade")
        print()

    # --- CVD magnitude buckets ---
    print("=" * 80)
    print("CVD 10-BAR MAGNITUDE BUCKETS")
    print("=" * 80)

    # Use absolute aligned CVD (positive = aligned, negative = against)
    df["aligned_cvd"] = df.apply(
        lambda r: r["cvd_10"] if r["direction"] == "long" else -r["cvd_10"], axis=1
    )

    buckets = [
        ("Strong against (<-200)", df["aligned_cvd"] < -200),
        ("Against (-200 to -50)", (df["aligned_cvd"] >= -200) & (df["aligned_cvd"] < -50)),
        ("Neutral (-50 to +50)", (df["aligned_cvd"] >= -50) & (df["aligned_cvd"] <= 50)),
        ("With (+50 to +200)", (df["aligned_cvd"] > 50) & (df["aligned_cvd"] <= 200)),
        ("Strong with (>+200)", df["aligned_cvd"] > 200),
    ]

    for label, mask in buckets:
        subset = df[mask]
        if len(subset) == 0:
            print(f"  {label:30s}: 0 trades")
            continue
        wr = subset["win"].mean() * 100
        pnl = subset["pnl_points"].sum()
        exp = subset["pnl_points"].mean()
        print(f"  {label:30s}: {len(subset):3d} trades | WR: {wr:5.1f}% | P&L: {pnl:+8.1f} pts | Exp: {exp:+.2f} pts/trade")

    print()

    # --- By direction separately ---
    for dir_label, dir_val in [("SHORT", "short"), ("LONG", "long")]:
        print(f"--- {dir_label} trades: CVD 10-bar alignment ---")
        ddir = df[df["direction"] == dir_val]

        aligned = ddir[ddir["aligned"]]  # uses last computed 'aligned' col but let's recompute
        # Recompute aligned for cvd_10
        if dir_val == "short":
            al = ddir[ddir["cvd_10"] < 0]
            ag = ddir[ddir["cvd_10"] >= 0]
        else:
            al = ddir[ddir["cvd_10"] > 0]
            ag = ddir[ddir["cvd_10"] <= 0]

        al_wr = al["win"].mean() * 100 if len(al) > 0 else 0
        ag_wr = ag["win"].mean() * 100 if len(ag) > 0 else 0
        print(f"  Aligned:   {len(al):3d} trades | WR: {al_wr:5.1f}% | P&L: {al['pnl_points'].sum():+8.1f} pts")
        print(f"  Against:   {len(ag):3d} trades | WR: {ag_wr:5.1f}% | P&L: {ag['pnl_points'].sum():+8.1f} pts")
        print()


if __name__ == "__main__":
    run_analysis()

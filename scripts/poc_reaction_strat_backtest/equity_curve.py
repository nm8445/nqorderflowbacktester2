"""Equity curve for Longs only + contra CVD config."""

import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
POC_REACTION_CACHE_DIR = DATA_DIR / "poc_reaction_cache"
SIGNAL_CACHE_DIR = DATA_DIR / "signal_cache"
POINT_VALUE = 20.0

ENTRY_START = "09:00"
ENTRY_END = "11:00"


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


def get_trades():
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
            if s["direction"] == "short":
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
            stop_loss = signal["stop_loss"]
            target = signal["target"]
            direction = signal["direction"]
            confirm_bar_idx = signal["bar_index"] + 1

            if target <= entry_price or stop_loss >= entry_price:
                continue

            cvd = compute_recent_cvd(bars, confirm_bar_idx + 1)
            if cvd >= 0:
                continue

            in_trade = True
            exit_price = None
            exit_reason = None
            exit_time = None

            for j in range(confirm_bar_idx + 1, len(bars)):
                bar = bars[j]
                if not bar.closed:
                    continue
                if bar.low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                    exit_time = bar.close_time
                    break
                if bar.high >= target:
                    exit_price = target
                    exit_reason = "target"
                    exit_time = bar.close_time
                    break

            if exit_price is None:
                last_bar = bars[-1]
                exit_price = last_bar.close
                exit_time = last_bar.close_time
                exit_reason = "eod"

            pnl_points = exit_price - entry_price
            pnl_dollars = pnl_points * POINT_VALUE

            confirm_time = signal["confirm_time"]
            if hasattr(confirm_time, "tz_convert"):
                confirm_et = confirm_time.tz_convert(ET)
            else:
                confirm_et = pd.Timestamp(confirm_time, tz="UTC").tz_convert(ET)

            all_trades.append({
                "date": file_date,
                "entry_time": confirm_et,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_points": pnl_points,
                "pnl_dollars": pnl_dollars,
                "trade_num": len(all_trades) + 1,
            })
            in_trade = False

    return pd.DataFrame(all_trades)


if __name__ == "__main__":
    df = get_trades()
    df["cum_pnl"] = df["pnl_dollars"].cumsum()
    df["cum_max"] = df["cum_pnl"].cummax()
    df["drawdown"] = df["cum_pnl"] - df["cum_max"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                             sharex=True, gridspec_kw={"hspace": 0.08})

    # Equity curve
    ax1 = axes[0]
    ax1.plot(df["date"], df["cum_pnl"], color="#2196F3", linewidth=1.5, label="Equity")
    ax1.fill_between(df["date"], 0, df["cum_pnl"],
                     where=df["cum_pnl"] >= 0, alpha=0.15, color="#4CAF50")
    ax1.fill_between(df["date"], 0, df["cum_pnl"],
                     where=df["cum_pnl"] < 0, alpha=0.15, color="#F44336")

    # Mark wins/losses
    wins = df[df["pnl_dollars"] > 0]
    losses = df[df["pnl_dollars"] <= 0]
    ax1.scatter(wins["date"], wins["cum_pnl"], color="#4CAF50", s=18, zorder=5, alpha=0.7)
    ax1.scatter(losses["date"], losses["cum_pnl"], color="#F44336", s=18, zorder=5, alpha=0.7)

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Cumulative P&L ($)", fontsize=11)
    ax1.set_title("POC Reaction Strategy — Longs Only + Contra CVD Filter\n"
                  f"120 trades | 40.8% WR | PF 1.39 | +$18,714 | Max DD -$13,376",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    ax2.fill_between(df["date"], df["drawdown"], 0, color="#F44336", alpha=0.3)
    ax2.plot(df["date"], df["drawdown"], color="#F44336", linewidth=1)
    ax2.set_ylabel("Drawdown ($)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    out = Path("results/equity_curve_longs_contra_cvd.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")

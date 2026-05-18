"""Replay harness for the OD engine — validates parity vs backtest.

Loads:
  - 20-min bars from 1-min parquet (matches backtest's _resample_to_20min)
  - Backtest reference trade log at `live/overnight drift/trades.csv`

Pipes 20-min bars through ODEngine and diffs trades.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from live.combined.bar_builder import Bar
from live.combined.od_engine import ODEngine, ODSignal, Direction
from live.combined.config import ET_TZ

NQ_1MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
BACKTEST_TRADELOG = Path("C:/trading/nqorderflowbacktester/live/overnight drift/trades.csv")
OUT_CSV = Path(__file__).parent / "state" / "live_od_trades.csv"
NQ_POINT_VALUE = 20.0


def build_20min_bars(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Build 20-min bars anchored to midnight ET from 1-min parquet
    (matches backtest's _resample_to_20min: origin=start_day, label=left, closed=left)."""
    df = pd.read_parquet(NQ_1MIN)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET_TZ)
    df = df.sort_index()
    if start: df = df[df.index >= pd.Timestamp(start, tz=ET_TZ)]
    if end:   df = df[df.index <= pd.Timestamp(end,   tz=ET_TZ)]
    bars = (df["close"].resample("20min", origin="start_day",
                                  label="left", closed="left").last().rename("close").to_frame()
            .assign(open  = df["open"] .resample("20min", origin="start_day", label="left", closed="left").first(),
                    high  = df["high"] .resample("20min", origin="start_day", label="left", closed="left").max(),
                    low   = df["low"]  .resample("20min", origin="start_day", label="left", closed="left").min())
            .dropna(subset=["open","high","low","close"])
            [["open","high","low","close"]])
    return bars


def replay() -> pd.DataFrame:
    print("=" * 70)
    print("OD REPLAY")
    print("=" * 70)
    bars20 = build_20min_bars()
    print(f"[replay_od] built {len(bars20):,} 20-min bars  "
          f"({bars20.index[0]} -> {bars20.index[-1]})")

    engine = ODEngine()

    completed: list[dict] = []
    current_entry: dict = {}

    def on_signal(sig: ODSignal):
        nonlocal current_entry
        if sig.event == "ENTRY":
            current_entry = {"entry_ts": sig.timestamp, "entry": sig.price, "qty": sig.qty}
        elif sig.event == "EXIT" and current_entry:
            pnl_pts = sig.price - current_entry["entry"]
            qty = current_entry["qty"]
            completed.append({
                **current_entry, "exit_ts": sig.timestamp, "exit": sig.price,
                "reason": sig.reason, "qty_exit": qty,
                "pnl_points": pnl_pts,
                "pnl_dollars": pnl_pts * NQ_POINT_VALUE * qty,
            })
            current_entry = {}

    engine.subscribe(on_signal)

    for ts, row in bars20.iterrows():
        bar = Bar(
            open_time=ts, close_time=ts + pd.Timedelta(minutes=20),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            buy_vol=0, sell_vol=0, tick_count=0, timeframe_secs=1200,
        )
        engine.on_20min_bar(bar)

    print(f"\n[replay_od] done. bars={engine.n_bars_seen}  "
          f"entries={engine.n_entries}  exits={engine.n_exits}")

    df = pd.DataFrame(completed)
    if not df.empty:
        wins = df[df["pnl_dollars"] > 0]; losses = df[df["pnl_dollars"] < 0]
        wr = 100 * len(wins) / len(df)
        pf = wins["pnl_dollars"].sum() / abs(losses["pnl_dollars"].sum()) if len(losses) else float("inf")
        cum = df["pnl_dollars"].cumsum()
        mdd = (cum - cum.cummax()).min()
        print(f"\n=== OD LIVE-ENGINE REPLAY STATS ===")
        print(f"  Trades: {len(df)}   WR: {wr:.1f}%   PF: {pf:.2f}")
        print(f"  PnL: ${df['pnl_dollars'].sum():,.0f}   MaxDD: ${mdd:,.0f}")
        print(f"  Reason breakdown: {df['reason'].value_counts().to_dict()}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    return df


def diff_vs_backtest(live_df: pd.DataFrame) -> None:
    print(f"\n=== DIFF vs backtest ({BACKTEST_TRADELOG.name}) ===")
    if not BACKTEST_TRADELOG.exists():
        print(f"  Backtest file not found"); return
    bt = pd.read_csv(BACKTEST_TRADELOG)
    bt["entry_ts"] = pd.to_datetime(bt["entry_time"], utc=True).dt.tz_convert(ET_TZ)
    bt["exit_ts"]  = pd.to_datetime(bt["exit_time"],  utc=True).dt.tz_convert(ET_TZ)
    live_df["entry_ts"] = pd.to_datetime(live_df["entry_ts"])
    if live_df["entry_ts"].dt.tz is None:
        live_df["entry_ts"] = live_df["entry_ts"].dt.tz_localize(ET_TZ)
    bt["key"] = bt["entry_ts"].dt.strftime("%Y-%m-%d %H:%M")
    live_df["key"] = live_df["entry_ts"].dt.strftime("%Y-%m-%d %H:%M")
    bt_keys = set(bt["key"]); live_keys = set(live_df["key"])
    matched = bt_keys & live_keys
    only_bt = bt_keys - live_keys
    only_lv = live_keys - bt_keys
    print(f"  BT: {len(bt)}  Live: {len(live_df)}  Matched: {len(matched)}  "
          f"Only-BT: {len(only_bt)}  Only-Live: {len(only_lv)}")
    if matched:
        bt_m = bt[bt["key"].isin(matched)].set_index("key")
        lv_m = live_df[live_df["key"].isin(matched)].set_index("key")
        j = bt_m.join(lv_m, lsuffix="_bt", rsuffix="_lv")
        if "pnl_dollars_lv" in j.columns and "pnl_dollars" in j.columns:
            diff = j["pnl_dollars_lv"] - j["pnl_dollars"]
            print(f"  PnL diff on matched: mean ${diff.mean():+,.2f}  "
                  f"abs mean ${diff.abs().mean():,.2f}  max abs ${diff.abs().max():,.2f}")
            print(f"  Total live: ${j['pnl_dollars_lv'].sum():+,.0f}  "
                  f"Total bt: ${j['pnl_dollars'].sum():+,.0f}")


def main():
    df = replay()
    diff_vs_backtest(df)


if __name__ == "__main__":
    main()

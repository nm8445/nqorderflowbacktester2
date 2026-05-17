"""Replay harness for the Fabio ORB engine — validates parity vs backtest.

Loads:
  - 5-min volumetric bars from `D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet`
    (aggregated per-bar: open/high/low/close/buy_vol/sell_vol)
  - Backtest trade log at `D:/trading_pythonbacktest_data/fabio orb/trades_final_modeA.csv`

Pipes bars through FabioORBEngine and diffs the trade list.

Usage:
  python live/combined/replay_fabio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from live.combined.bar_builder import Bar
from live.combined.fabio_orb_engine import FabioORBEngine, FabioSignal, Direction
from live.combined.config import ET_TZ

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
BACKTEST_TRADELOG = Path("D:/trading_pythonbacktest_data/fabio orb/trades_final_modeA.csv")
OUT_CSV = Path(__file__).parent / "state" / "live_fabio_trades.csv"
NQ_POINT_VALUE = 20.0
TICK_SIZE = 0.25
TICK_VALUE = 5.0
SLIP_TICKS = 1
COMM_USD = 5.0


def build_5min_bars() -> list[Bar]:
    """Load volumetric parquet → per-bar aggregated 5-min bars with delta."""
    print(f"[replay_fabio] loading {VOL_PARQUET}...")
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
    ).dropna(subset=["open", "high", "low", "close"])
    agg = agg.sort_values("bar_open_time").reset_index(drop=True)

    # Convert to ET-aware timestamps
    ot = pd.to_datetime(agg["bar_open_time"])
    if ot.dt.tz is None:
        ot = ot.dt.tz_localize("UTC")
    ot = ot.dt.tz_convert(ET_TZ)
    agg["open_time"] = ot
    agg["close_time"] = ot + pd.Timedelta(minutes=5)

    bars: list[Bar] = []
    for _, r in agg.iterrows():
        bars.append(Bar(
            open_time=r["open_time"], close_time=r["close_time"],
            open=float(r["open"]), high=float(r["high"]),
            low=float(r["low"]), close=float(r["close"]),
            buy_vol=float(r["buy_vol"]), sell_vol=float(r["sell_vol"]),
            tick_count=0, timeframe_secs=300,
        ))
    print(f"[replay_fabio] built {len(bars):,} 5-min bars  "
          f"({bars[0].close_time} -> {bars[-1].close_time})")
    return bars


def replay() -> pd.DataFrame:
    print("=" * 70)
    print("FABIO ORB REPLAY")
    print("=" * 70)
    bars = build_5min_bars()
    engine = FabioORBEngine()

    completed: list[dict] = []
    current_entry: dict = {}

    def on_signal(sig: FabioSignal):
        nonlocal current_entry
        if sig.event == "ENTRY":
            current_entry = {
                "entry_ts": sig.timestamp,
                "entry": sig.price,
                "qty": sig.qty,
            }
        elif sig.event == "EXIT" and current_entry:
            pnl_pts = sig.price - current_entry["entry"]
            # Match backtest costs: 1 tick slippage each side + $5 commission r/t
            slip_pts = 2 * SLIP_TICKS * TICK_SIZE   # round-trip slippage
            gross = pnl_pts * NQ_POINT_VALUE * current_entry["qty"]
            net = (pnl_pts - slip_pts) * NQ_POINT_VALUE * current_entry["qty"] - COMM_USD * current_entry["qty"]
            completed.append({
                **current_entry, "exit_ts": sig.timestamp,
                "exit": sig.price, "reason": sig.reason,
                "pnl_points": pnl_pts,
                "gross_dollars": gross,
                "net_dollars": net,
            })
            current_entry = {}

    engine.subscribe(on_signal)

    for bar in bars:
        engine.on_5min_bar(bar)

    print(f"\n[replay_fabio] done. bars={engine.n_bars_seen:,}  "
          f"entries={engine.n_entries}  exits={engine.n_exits}  "
          f"blocked_skip930={engine.n_blocked_skip_bucket}  "
          f"blocked_no_confirm={engine.n_blocked_no_confirm}  "
          f"blocked_delta={engine.n_blocked_delta}")

    df = pd.DataFrame(completed)
    if not df.empty:
        wins = df[df["net_dollars"] > 0]
        losses = df[df["net_dollars"] < 0]
        wr = 100 * len(wins) / len(df)
        pf = wins["net_dollars"].sum() / abs(losses["net_dollars"].sum()) if len(losses) else float("inf")
        cum = df["net_dollars"].cumsum()
        mdd = (cum - cum.cummax()).min()
        print(f"\n=== FABIO LIVE-ENGINE REPLAY STATS ===")
        print(f"  Trades: {len(df)}")
        print(f"  Win rate: {wr:.1f}%")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Net PnL: ${df['net_dollars'].sum():,.0f}")
        print(f"  Max DD: ${mdd:,.0f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    return df


def diff_vs_backtest(live_df: pd.DataFrame) -> None:
    print(f"\n=== DIFF vs backtest ({BACKTEST_TRADELOG.name}) ===")
    if not BACKTEST_TRADELOG.exists():
        print(f"  Backtest file not found"); return
    bt = pd.read_csv(BACKTEST_TRADELOG)
    bt = bt[bt["mode"] == "A"].copy()
    bt["entry_ts"] = pd.to_datetime(bt["entry_time"], utc=True).dt.tz_convert(ET_TZ)
    bt["exit_ts"]  = pd.to_datetime(bt["exit_time"],  utc=True).dt.tz_convert(ET_TZ)
    live_df["entry_ts"] = pd.to_datetime(live_df["entry_ts"])
    if live_df["entry_ts"].dt.tz is None:
        live_df["entry_ts"] = live_df["entry_ts"].dt.tz_localize(ET_TZ)
    else:
        live_df["entry_ts"] = live_df["entry_ts"].dt.tz_convert(ET_TZ)
    live_df["exit_ts"] = pd.to_datetime(live_df["exit_ts"])
    if live_df["exit_ts"].dt.tz is None:
        live_df["exit_ts"] = live_df["exit_ts"].dt.tz_localize(ET_TZ)
    else:
        live_df["exit_ts"] = live_df["exit_ts"].dt.tz_convert(ET_TZ)

    bt["key"] = bt["entry_ts"].dt.strftime("%Y-%m-%d %H:%M")
    live_df["key"] = live_df["entry_ts"].dt.strftime("%Y-%m-%d %H:%M")
    bt_keys = set(bt["key"])
    live_keys = set(live_df["key"])
    matched = bt_keys & live_keys
    only_bt = bt_keys - live_keys
    only_live = live_keys - bt_keys

    print(f"  Backtest trades: {len(bt)}   Live: {len(live_df)}")
    print(f"  Matched: {len(matched)}   Only backtest: {len(only_bt)}   Only live: {len(only_live)}")
    if len(bt) + len(only_live):
        rate = 100 * len(matched) / (len(matched) + len(only_bt) + len(only_live))
        print(f"  Match rate: {rate:.1f}%")

    # PnL agreement on matched trades
    bt_m = bt[bt["key"].isin(matched)].set_index("key")
    lv_m = live_df[live_df["key"].isin(matched)].set_index("key")
    joined = bt_m.join(lv_m, lsuffix="_bt", rsuffix="_lv")
    if not joined.empty:
        diff = joined["net_dollars_lv"] - joined["net_dollars_bt"]
        print(f"\n  Net-PnL diff on matched trades:")
        print(f"    mean abs diff: ${diff.abs().mean():,.2f}")
        print(f"    max  abs diff: ${diff.abs().max():,.2f}")
        print(f"    total live: ${joined['net_dollars_lv'].sum():+,.0f}")
        print(f"    total bt:   ${joined['net_dollars_bt'].sum():+,.0f}")

    if only_bt:
        print(f"\n  First 5 'only backtest' trades:")
        for k in sorted(only_bt)[:5]:
            r = bt[bt["key"] == k].iloc[0]
            print(f"    {k}  entry=${r['entry']:.2f} exit=${r['exit']:.2f} reason={r['reason']} net=${r['net_dollars']:+,.0f}")
    if only_live:
        print(f"\n  First 5 'only live' trades:")
        for k in sorted(only_live)[:5]:
            r = live_df[live_df["key"] == k].iloc[0]
            print(f"    {k}  entry=${r['entry']:.2f} exit=${r['exit']:.2f} reason={r['reason']} net=${r['net_dollars']:+,.0f}")


def main():
    live_df = replay()
    diff_vs_backtest(live_df)


if __name__ == "__main__":
    main()

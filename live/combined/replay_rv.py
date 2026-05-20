"""Replay harness for the live RV engine — validates parity vs backtest.

Loads:
  - 20-min bars from D:/trading_pythonbacktest_data/timebars_5min_*/ pickles (aggregated)
  - Per-bar level volumes from D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet

Pipes each bar (with its level-volume dict) through the live RVEngine.

Output:
  - live_rv_trades.csv: trades emitted by the live engine
  - prints summary: total entries, exits, win rate, PF, gross PnL

Comparison against backtest trades:
  python live/combined/replay_rv.py --diff results/inspect_v3_N400_v3_trades.csv

Usage:
  python live/combined/replay_rv.py
  python live/combined/replay_rv.py --start 2025-01-01 --end 2025-06-30
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from live.combined.bar_builder import Bar
from live.combined.rv_engine import RVEngine, Signal, Direction
from live.combined.config import TIMEBARS_5MIN_DIRS, ET_TZ

VOLUMETRIC_PATH = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
NQ_POINT_VALUE = 20.0


def load_5min_bars_to_20min(start_date: str | None = None,
                             end_date: str | None = None) -> pd.DataFrame:
    """Load 5-min pickle archive and aggregate to 20-min bars (close-time indexed, ET tz)."""
    by_stem = {}
    for d in TIMEBARS_5MIN_DIRS:
        if not d.exists(): continue
        for f in sorted(d.glob("timebars_5min_*.pkl")):
            by_stem[f.stem] = f
    files = [by_stem[k] for k in sorted(by_stem.keys())]

    if start_date or end_date:
        sd = pd.Timestamp(start_date).date() if start_date else None
        ed = pd.Timestamp(end_date).date() if end_date else None
        def _fdate(p: Path):
            parts = p.stem.replace("timebars_5min_", "").split("_")
            return pd.Timestamp(int(parts[0]), int(parts[1]), int(parts[2])).date()
        files = [f for f in files if (sd is None or _fdate(f) >= sd) and (ed is None or _fdate(f) <= ed)]

    frames = []
    for f in files:
        with open(f, "rb") as fh:
            bars = pickle.load(fh)
        if not bars: continue
        rows = [{"ts": b["open_time"], "open": b["open"], "high": b["high"],
                 "low": b["low"], "close": b["close"],
                 "buy_vol": b.get("buy_vol", 0), "sell_vol": b.get("sell_vol", 0)} for b in bars]
        df5 = pd.DataFrame(rows).set_index("ts").sort_index()
        df5["group"] = df5.index.floor("20min")
        agg = df5.groupby("group").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
            buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
        )
        # Close time = group + 20 minutes (= bar close)
        agg.index = agg.index + pd.Timedelta(minutes=20)
        frames.append(agg)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # Coerce to DatetimeIndex with ET tz
    df.index = pd.DatetimeIndex([
        (pd.Timestamp(t).tz_convert(ET_TZ) if hasattr(pd.Timestamp(t), "tz_convert") and pd.Timestamp(t).tzinfo is not None
         else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET_TZ))
        for t in df.index
    ])
    return df


def build_level_volume_map(start_date: str | None = None,
                            end_date: str | None = None) -> dict[pd.Timestamp, dict[float, tuple[int, int]]]:
    """Load volumetric parquet and aggregate to per-20min-bar level dicts.
    Returns {bar_close_time_et: {level_price: (buy_vol, sell_vol)}}.
    """
    print(f"[replay] loading volumetric data from {VOLUMETRIC_PATH} ...")
    vol = pd.read_parquet(VOLUMETRIC_PATH)
    print(f"  loaded {len(vol):,} rows")
    vol["bar_open_time"] = pd.to_datetime(vol["bar_open_time"])
    if vol["bar_open_time"].dt.tz is None:
        vol["bar_open_time"] = vol["bar_open_time"].dt.tz_localize(ET_TZ)
    else:
        vol["bar_open_time"] = vol["bar_open_time"].dt.tz_convert(ET_TZ)
    if start_date:
        vol = vol[vol["bar_open_time"].dt.date >= pd.Timestamp(start_date).date()]
    if end_date:
        vol = vol[vol["bar_open_time"].dt.date <= pd.Timestamp(end_date).date()]
    # Aggregate to 20-min bar close time
    vol["group"] = vol["bar_open_time"].dt.floor("20min")
    vol["bar_close_time"] = vol["group"] + pd.Timedelta(minutes=20)
    grp = vol.groupby(["bar_close_time", "level_price"], as_index=False).agg(
        buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
    )
    print(f"  grouping into per-bar level dicts ({len(grp):,} (bar, level) rows)...")
    result: dict[pd.Timestamp, dict[float, tuple[int, int]]] = {}
    for bct, sub in grp.groupby("bar_close_time"):
        level_map = {float(row["level_price"]): (int(row["buy_vol"]), int(row["sell_vol"]))
                     for _, row in sub.iterrows()}
        result[pd.Timestamp(bct)] = level_map
    print(f"  built {len(result):,} per-bar level dicts")
    return result


def replay(start_date: str | None = None, end_date: str | None = None,
           out_csv: Path | None = None) -> pd.DataFrame:
    """Replay historical bars through live RVEngine. Returns DataFrame of trades."""
    print("=" * 70)
    print("RV REPLAY")
    print("=" * 70)
    bars_df = load_5min_bars_to_20min(start_date, end_date)
    print(f"[replay] loaded {len(bars_df):,} 20-min bars  "
          f"({bars_df.index[0]} -> {bars_df.index[-1]})")

    levels_map = build_level_volume_map(start_date, end_date)

    engine = RVEngine()

    # Collect signals as we go
    trades_in_flight: dict = {}  # entry_signal id -> partial trade
    completed_trades: list[dict] = []
    current_entry: dict | None = None

    def on_signal(sig: Signal):
        nonlocal current_entry
        if sig.event == "ENTRY":
            current_entry = {
                "entry_ts": sig.timestamp,
                "direction": "LONG" if sig.direction == Direction.LONG else "SHORT",
                "entry_price": sig.price,
            }
        elif sig.event == "EXIT" and current_entry is not None:
            t = current_entry
            sign = 1 if t["direction"] == "LONG" else -1
            pnl_pts = sign * (sig.price - t["entry_price"])
            completed_trades.append({
                **t,
                "exit_ts": sig.timestamp,
                "exit_price": sig.price,
                "exit_reason": sig.reason,
                "pnl_points": pnl_pts,
                "pnl_dollars": pnl_pts * NQ_POINT_VALUE,
            })
            current_entry = None

    engine.subscribe(on_signal)

    # Feed bars
    print(f"[replay] running engine through {len(bars_df):,} bars ...")
    n_bars_with_levels = 0
    for ts, row in bars_df.iterrows():
        bar = Bar(
            open_time=ts - pd.Timedelta(minutes=20),
            close_time=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            buy_vol=int(row["buy_vol"]),
            sell_vol=int(row["sell_vol"]),
            tick_count=0,
            timeframe_secs=1200,
        )
        lv = levels_map.get(ts)
        if lv is not None:
            engine.set_bar_levels(lv)
            n_bars_with_levels += 1
        engine.on_bar(bar)

    print(f"[replay] done. "
          f"bars_seen={engine.n_bars_seen}  with_levels={n_bars_with_levels}  "
          f"entries={engine.n_entries}  exits={engine.n_exits}  "
          f"trades_completed={len(completed_trades)}")

    df = pd.DataFrame(completed_trades)
    if not df.empty:
        wins = df[df["pnl_dollars"] > 0]
        losses = df[df["pnl_dollars"] < 0]
        wr = 100 * len(wins) / len(df) if len(df) else 0
        pf = wins["pnl_dollars"].sum() / abs(losses["pnl_dollars"].sum()) if len(losses) else float("inf")
        cum = df["pnl_dollars"].cumsum()
        mdd = (cum - cum.cummax()).min()
        print(f"\n=== LIVE-ENGINE REPLAY STATS ===")
        print(f"  Trades: {len(df)}")
        print(f"  Win rate: {wr:.1f}%")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Gross PnL: ${df['pnl_dollars'].sum():,.0f}")
        print(f"  Max DD: ${mdd:,.0f}")

    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")

    return df


def diff_vs_backtest(live_df: pd.DataFrame, backtest_csv: Path) -> None:
    """Compare live engine trades vs backtest reference, filtered to overlap window."""
    print(f"\n=== DIFF vs backtest ({backtest_csv}) ===")
    if not backtest_csv.exists():
        print(f"  Backtest file not found, skipping diff")
        return
    bt = pd.read_csv(backtest_csv)
    if "entry_ts" not in bt.columns and "entry_time" in bt.columns:
        bt = bt.rename(columns={"entry_time": "entry_ts"})
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True).dt.tz_convert(ET_TZ)
    live_df["entry_ts"] = pd.to_datetime(live_df["entry_ts"])
    if live_df["entry_ts"].dt.tz is None:
        live_df["entry_ts"] = live_df["entry_ts"].dt.tz_localize(ET_TZ)

    # Restrict backtest to the live engine's time window
    live_min = live_df["entry_ts"].min()
    live_max = live_df["entry_ts"].max()
    bt_in_window = bt[(bt["entry_ts"] >= live_min) & (bt["entry_ts"] <= live_max)]
    print(f"  Replay window: {live_min} -> {live_max}")
    print(f"  Backtest trades in window: {len(bt_in_window)}  (out of {len(bt)} total)")
    print(f"  Live engine trades: {len(live_df)}")

    bt_set = set(bt_in_window["entry_ts"].dt.tz_convert("UTC").astype("int64"))
    live_set = set(live_df["entry_ts"].dt.tz_convert("UTC").astype("int64"))

    both = bt_set & live_set
    only_bt = bt_set - live_set
    only_live = live_set - bt_set
    print(f"  Matched entry times: {len(both)}")
    print(f"  Only in backtest:    {len(only_bt)}")
    print(f"  Only in live engine: {len(only_live)}")
    base = max(len(bt_set), len(live_set))
    match_pct = 100 * len(both) / base if base else 0
    print(f"  Match rate (vs max of both): {match_pct:.1f}%  (target: 95%+)")

    if only_bt:
        print(f"\n  Sample of trades only in BACKTEST (first 10):")
        miss_ts = sorted([pd.Timestamp(t, unit="ns", tz="UTC").tz_convert(ET_TZ) for t in only_bt])
        for t in miss_ts[:10]:
            row = bt_in_window[bt_in_window["entry_ts"] == t].iloc[0]
            d = row.get("direction") or row.get("side") or "?"
            print(f"    {t} {d}")
    if only_live:
        print(f"\n  Sample of trades only in LIVE engine (first 10):")
        extra_ts = sorted([pd.Timestamp(t, unit="ns", tz="UTC").tz_convert(ET_TZ) for t in only_live])
        for t in extra_ts[:10]:
            row = live_df[live_df["entry_ts"] == t].iloc[0]
            print(f"    {t} {row['direction']} @ {row['entry_price']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    ap.add_argument("--out", type=str, default="live/combined/state/live_rv_trades.csv")
    ap.add_argument("--diff", type=str, default=None,
                    help="Path to backtest trades CSV for parity diff")
    args = ap.parse_args()

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    live_df = replay(args.start, args.end, out_csv)

    if args.diff:
        diff_vs_backtest(live_df, Path(args.diff))


if __name__ == "__main__":
    main()

"""Replay harness for the B2 engine — validates exit-logic parity vs backtest.

Loads:
  - Pre-computed entry signals from `scripts/overnight range strat/scripts/parquets/entry_signal_trades.parquet`
    (already filtered with the locked B2 variant/X/N/D/STRICT/BAND_K params per backtest)
  - 20-min bars (with atr_y) built from 5-min pickle archive
  - Gamma signs from `D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet`

Pipes each entry signal + 20-min bars through the live B2 engine.

Compares output trades to the locked backtest trade log:
  `scripts/overnight range strat/tradelogs/robust_configs/locked_v2_k08_lock045_mart_fc_filtered_trades.csv`

Usage:
  python live/combined/replay_b2.py
  python live/combined/replay_b2.py --start 2025-01-01 --end 2026-04-17
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
from live.combined.b2_engine import B2Engine, B2Signal, Direction
from live.combined.config import TIMEBARS_5MIN_DIRS, ET_TZ

SIGNAL_PARQUET = Path("C:/trading/nqorderflowbacktester/scripts/overnight range strat/scripts/parquets/entry_signal_trades.parquet")
SIGNAL_PARQUET_OOS = Path("C:/trading/nqorderflowbacktester/scripts/overnight range strat/scripts/parquets/entry_signal_trades_oos.parquet")
GAMMA_PARQUET = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
BACKTEST_TRADELOG = Path("C:/trading/nqorderflowbacktester/scripts/overnight range strat/tradelogs/robust_configs/locked_v2_k08_lock045_mart_fc_filtered_trades.csv")
NQ_POINT_VALUE = 20.0


NQ_1MIN_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")


def build_20min_bars_with_atr(start: str | None, end: str | None) -> pd.DataFrame:
    """Build 20-min bars from 1-min source — MATCHES backtest's build_20min_bars()
    in test_pure_ratchet_exits.py (1-min resample label="left", closed="left").

    Index = bar OPEN time (ET-aware). atr_y = Wilder ATR(14) on 20-min closes.
    """
    df = pd.read_parquet(NQ_1MIN_PARQUET)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET_TZ)
    df = df.sort_index()
    if start or end:
        sd = pd.Timestamp(start, tz=ET_TZ) if start else None
        ed = pd.Timestamp(end, tz=ET_TZ) if end else None
        if sd is not None: df = df[df.index >= sd]
        if ed is not None: df = df[df.index <= ed]

    # 1-min → 20-min, label="left" (index = bar open), closed="left"
    bars = (df["close"].resample("20min", label="left", closed="left").last().rename("close").to_frame()
            .assign(open  = df["open"] .resample("20min", label="left", closed="left").first(),
                    high  = df["high"] .resample("20min", label="left", closed="left").max(),
                    low   = df["low"]  .resample("20min", label="left", closed="left").min())
            .dropna(subset=["open","high","low","close"]))
    bars = bars[["open","high","low","close"]]

    # Wilder ATR(14)
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        (bars["high"] - bars["low"]).abs(),
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.copy()
    n = 14
    if len(tr) > n:
        atr.iloc[n - 1] = tr.iloc[:n].mean()
        for i in range(n, len(tr)):
            atr.iloc[i] = (atr.iloc[i-1] * (n-1) + tr.iloc[i]) / n
    bars["atr_y"] = atr
    return bars


def load_entry_signals(start: str | None, end: str | None) -> pd.DataFrame:
    """Load pre-computed entry signals and apply the locked B2 filter stack:
       variant=B2, pinbar_ratio>=0.75, |abs_delta_w15|>=70, strict_short=True (for SHORTs),
       band_K=0.25 proximity, |conf_delta_half_w5|>=75 in trade direction."""
    from live.combined.b2_engine import X, N_DELTA, D_DELTA, STRICT_SHORT, BAND_K, CONF_N, CONF_D, VARIANT

    if not SIGNAL_PARQUET.exists():
        print(f"  WARNING: signal parquet not found at {SIGNAL_PARQUET}")
        return pd.DataFrame()
    frames = [pd.read_parquet(SIGNAL_PARQUET)]
    if SIGNAL_PARQUET_OOS.exists():
        frames.append(pd.read_parquet(SIGNAL_PARQUET_OOS))
    df = pd.concat(frames, ignore_index=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    if df["entry_time"].dt.tz is None:
        df["entry_time"] = df["entry_time"].dt.tz_localize("UTC").dt.tz_convert(ET_TZ)
    else:
        df["entry_time"] = df["entry_time"].dt.tz_convert(ET_TZ)

    # Apply locked filters (match apply_filters in range_break_entry_summary.py)
    n_before = len(df)
    sub = df[df["variant"] == VARIANT].copy()
    sub = sub[sub["pinbar_ratio"] >= X]
    sub = sub[sub[f"abs_delta_w{N_DELTA}"].abs() >= D_DELTA]
    if STRICT_SHORT:
        long_mask = sub["direction"] == "LONG"
        strict_ok = sub["strict_short"].fillna(False).astype(bool)
        sub = sub[long_mask | strict_ok]
    # Proximity band — clip(BAND_K * atr_at_entry, 5, 20)
    band = np.clip(BAND_K * sub["atr_at_entry"].values, 5, 20)
    overlap = ((sub["signal_low"].values <= sub["near_level"].values + band) &
               (sub["signal_high"].values >= sub["near_level"].values - band))
    sub = sub[overlap]
    # Confirmation: |conf_delta_half_w5| >= CONF_D in trade direction
    conf_col = f"conf_delta_half_w{CONF_N}"
    long_ok = (sub["direction"] == "LONG") & (sub[conf_col].notna()) & (sub[conf_col] >= CONF_D)
    short_ok = (sub["direction"] == "SHORT") & (sub[conf_col].notna()) & (sub[conf_col] <= -CONF_D)
    sub = sub[long_ok | short_ok].copy()

    # NOTE: backtest applies mode1_chained_dedupe(TP_M=1.0, SL_M=1.0) HERE using
    # look-ahead simplified TP/SL hit indices. That rule is NOT live-feasible,
    # so we skip it. The live engine handles dedupe naturally via its own
    # "drop signal if position is open" rule in B2Engine._try_fire_pending().
    sub = sub.sort_values("entry_time").reset_index(drop=True)
    print(f"  filters: {n_before} candidates -> {len(sub)} after B2 stack")

    if start:
        sub = sub[sub["entry_time"] >= pd.Timestamp(start, tz=ET_TZ)]
    if end:
        sub = sub[sub["entry_time"] <= pd.Timestamp(end, tz=ET_TZ).replace(hour=23, minute=59)]
    return sub


def load_gamma_signs() -> dict:
    """Returns {date: prior-day gamma_sign}.
    For each date, prior-day gamma = the gamma row from the latest date < this date."""
    if not GAMMA_PARQUET.exists():
        return {}
    df = pd.read_parquet(GAMMA_PARQUET, columns=["date", "qqq_gamma_sign"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["qqq_gamma_sign"]).sort_values("date")
    sorted_dates = list(df["date"])
    sign_by_date = dict(zip(df["date"], df["qqq_gamma_sign"].astype(int)))
    # Build prior-day lookup
    prior = {}
    for i, d in enumerate(sorted_dates):
        if i == 0: continue
        prior[d] = sign_by_date[sorted_dates[i-1]]
    return prior


def replay(start: str | None = None, end: str | None = None,
           out_csv: Path | None = None) -> pd.DataFrame:
    print("=" * 70)
    print("B2 REPLAY")
    print("=" * 70)
    bars20 = build_20min_bars_with_atr(start, end)
    print(f"[replay] loaded {len(bars20):,} 20-min bars  "
          f"({bars20.index[0]} -> {bars20.index[-1]})")
    signals = load_entry_signals(start, end)
    print(f"[replay] loaded {len(signals):,} entry candidates")
    gamma = load_gamma_signs()
    print(f"[replay] loaded {len(gamma)} gamma sign records")

    engine = B2Engine()
    # Seed gamma signs
    for d, sign in gamma.items():
        engine.set_gamma_sign(d, sign)

    # Seed pending entries — for each signal, queue with entry_price=close at signal time
    # entry_price from signals parquet (should have a `entry_price` column)
    n_queued = 0
    for _, row in signals.iterrows():
        if "direction" not in row or "entry_price" not in row:
            continue
        # Find atr_y at the 20-min bar PRIOR to entry (per backtest: bars20.iloc[init_idx])
        ets = row["entry_time"]
        # init_idx = searchsorted(ets, side="right") - 1
        idx = bars20.index.searchsorted(ets, side="right") - 1
        if idx < 0 or np.isnan(bars20["atr_y"].iloc[idx]):
            continue
        atr_y = float(bars20["atr_y"].iloc[idx])
        engine.set_pending_entry(
            entry_time=ets, direction=row["direction"],
            entry_price=float(row["entry_price"]), atr_y=atr_y,
        )
        n_queued += 1
    print(f"[replay] queued {n_queued} entry candidates")

    # Collect trades
    completed: list[dict] = []
    current_entry = {}

    def on_signal(sig: B2Signal):
        nonlocal current_entry
        if sig.event == "ENTRY":
            current_entry = {
                "entry_ts": sig.timestamp,
                "direction": "LONG" if sig.direction == Direction.LONG else "SHORT",
                "entry_price": sig.price,
                "qty": sig.qty,
            }
        elif sig.event == "EXIT" and current_entry:
            sign = 1 if current_entry["direction"] == "LONG" else -1
            pnl_pts = sign * (sig.price - current_entry["entry_price"])
            qty = current_entry.get("qty", 1)
            completed.append({
                **current_entry, "exit_ts": sig.timestamp,
                "exit_price": sig.price, "exit_reason": sig.reason,
                "pnl_points": pnl_pts,
                "pnl_dollars": pnl_pts * NQ_POINT_VALUE,           # 1-contract baseline
                "scaled_pnl_dollars": pnl_pts * NQ_POINT_VALUE * qty,  # mart-scaled
            })
            current_entry = {}

    engine.subscribe(on_signal)

    # Feed bars — bars20.index is now bar OPEN time (label="left", matches backtest)
    for ts, row in bars20.iterrows():
        bar = Bar(
            open_time=ts, close_time=ts + pd.Timedelta(minutes=20),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            buy_vol=0, sell_vol=0, tick_count=0, timeframe_secs=1200,
        )
        atr_now = float(row["atr_y"]) if not np.isnan(row["atr_y"]) else None
        engine.on_20min_bar(bar, current_atr_y=atr_now)

    print(f"\n[replay] done. bars={engine.n_bars_seen}  entries={engine.n_entries}  "
          f"exits={engine.n_exits}  blocked_hour={engine.n_blocked_hour}  "
          f"blocked_pos_short={engine.n_blocked_pos_short}")

    df = pd.DataFrame(completed)
    if not df.empty:
        def _stats(col):
            wins = df[df[col] > 0][col]
            losses = df[df[col] < 0][col]
            wr = 100 * len(wins) / len(df) if len(df) else 0
            pf = wins.sum() / abs(losses.sum()) if len(losses) else float("inf")
            cum = df[col].cumsum()
            mdd = (cum - cum.cummax()).min()
            return wr, pf, df[col].sum(), mdd
        wr1, pf1, pnl1, mdd1 = _stats("pnl_dollars")
        wr2, pf2, pnl2, mdd2 = _stats("scaled_pnl_dollars")
        n_size2 = int((df["qty"] == 2).sum()) if "qty" in df.columns else 0
        print(f"\n=== B2 LIVE-ENGINE REPLAY STATS ===")
        print(f"  Trades: {len(df)}   (size-2 mart trades: {n_size2})")
        print(f"  --- 1-contract baseline (no mart) ---")
        print(f"  WR: {wr1:.1f}%   PF: {pf1:.2f}   PnL: ${pnl1:,.0f}   MaxDD: ${mdd1:,.0f}")
        print(f"  --- FC-only mart (size-2 on FC losses) ---")
        print(f"  WR: {wr2:.1f}%   PF: {pf2:.2f}   PnL: ${pnl2:,.0f}   MaxDD: ${mdd2:,.0f}")

    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")
    return df


def diff_vs_backtest(live_df: pd.DataFrame, backtest_csv: Path) -> None:
    print(f"\n=== DIFF vs backtest ({backtest_csv.name}) ===")
    if not backtest_csv.exists():
        print(f"  Backtest file not found, skipping")
        return
    bt = pd.read_csv(backtest_csv)
    if "entry_ts" not in bt.columns:
        for c in ("entry_time", "entry_time_et"):
            if c in bt.columns:
                bt = bt.rename(columns={c: "entry_ts"})
                break
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True).dt.tz_convert(ET_TZ)
    live_df["entry_ts"] = pd.to_datetime(live_df["entry_ts"])
    if live_df["entry_ts"].dt.tz is None:
        live_df["entry_ts"] = live_df["entry_ts"].dt.tz_localize(ET_TZ)

    live_min, live_max = live_df["entry_ts"].min(), live_df["entry_ts"].max()
    bt_in = bt[(bt["entry_ts"] >= live_min) & (bt["entry_ts"] <= live_max)]
    print(f"  Replay window: {live_min} -> {live_max}")
    print(f"  Backtest trades in window: {len(bt_in)}  (out of {len(bt)} total)")
    print(f"  Live engine trades: {len(live_df)}")

    bt_set = set(bt_in["entry_ts"].dt.tz_convert("UTC").astype("int64"))
    live_set = set(live_df["entry_ts"].dt.tz_convert("UTC").astype("int64"))
    both = bt_set & live_set
    only_bt = bt_set - live_set
    only_live = live_set - bt_set
    print(f"  Matched: {len(both)}  Only backtest: {len(only_bt)}  Only live: {len(only_live)}")
    base = max(len(bt_set), len(live_set))
    match_pct = 100 * len(both) / base if base else 0
    print(f"  Match rate: {match_pct:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--out", type=str, default="live/combined/state/live_b2_trades.csv")
    ap.add_argument("--diff", type=str, default=str(BACKTEST_TRADELOG))
    args = ap.parse_args()

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    live_df = replay(args.start, args.end, out_csv)
    if args.diff:
        diff_vs_backtest(live_df, Path(args.diff))


if __name__ == "__main__":
    main()

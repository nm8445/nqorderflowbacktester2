"""Augment trade parquets with confirmation-candle absorption metrics.

For each B2 trade:
  - confirmation candle = entry_bar_idx - 1
  - midpoint = (low + high) / 2

Two search areas computed per trade (both stored):

  conf_delta_wick_w{N}  - wick-only zone:
    LONG  -> price < body_low  (bottom wick)
    SHORT -> price > body_high (top wick)

  conf_delta_half_w{N}  - bottom/top HALF of candle (regardless of body):
    LONG  -> price < midpoint  (bottom half by price range)
    SHORT -> price > midpoint  (top    half by price range)

For each window size N in {5, 10, 15, 20}, find the BEST window where delta
is most extreme in the bias direction (most positive for LONG, most negative
for SHORT). Stored as the SIGNED scalar.

Skips B1 trades (no confirmation candle).

Usage: python augment_with_confirmation_absorption.py [is|oos|both]
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_signal_study import load_5min_features

PARQUET_DIR = Path(__file__).parent / "parquets"
TRADES_IS   = PARQUET_DIR / "entry_signal_trades.parquet"
TRADES_OOS  = PARQUET_DIR / "entry_signal_trades_oos.parquet"

WINDOW_NS = [5, 10, 15, 20]


def _best_directional_delta(zone_levels: pd.DataFrame, direction: str,
                             window_ns: list[int]) -> dict:
    """Helper: scan zone for best directional delta per window size."""
    if zone_levels.empty:
        return {N: 0.0 for N in window_ns}
    zone = zone_levels.sort_values("level_price")
    bv = zone["buy_vol"].values.astype(float)
    sv = zone["sell_vol"].values.astype(float)
    n_levels = len(zone)
    results = {}
    cb = np.cumsum(bv); cs = np.cumsum(sv)
    for N in window_ns:
        if n_levels < N:
            results[N] = np.nan
            continue
        win_buy  = np.concatenate([[cb[N-1]], cb[N:] - cb[:-N]])
        win_sell = np.concatenate([[cs[N-1]], cs[N:] - cs[:-N]])
        deltas = win_buy - win_sell
        if direction == "LONG":
            results[N] = float(deltas.max())   # most positive (buyers absorbing)
        else:
            results[N] = float(deltas.min())   # most negative (sellers absorbing)
    return results


def compute_conf_delta_zones(bar_levels: pd.DataFrame, low: float, high: float,
                              body_low: float, body_high: float, direction: str,
                              window_ns: list[int]) -> tuple[dict, dict]:
    """Compute directional delta for both WICK-only and HALF-of-candle zones.
    Returns (wick_results, half_results) — each is dict of N -> delta.
    """
    if bar_levels.empty:
        empty = {N: np.nan for N in window_ns}
        return empty, empty
    if direction == "LONG":
        wick_zone = bar_levels[bar_levels["level_price"] < body_low]
        half_zone = bar_levels[bar_levels["level_price"] < (low + high) / 2]
    else:
        wick_zone = bar_levels[bar_levels["level_price"] > body_high]
        half_zone = bar_levels[bar_levels["level_price"] > (low + high) / 2]
    return (_best_directional_delta(wick_zone, direction, window_ns),
            _best_directional_delta(half_zone, direction, window_ns))


def augment_trades(trades_path: Path, label: str):
    print(f"\n=== Augmenting {label} ({trades_path.name}) ===")
    trades = pd.read_parquet(trades_path)
    print(f"  loaded {len(trades):,} trades")

    # Filter to B2 only — only B2 has a confirmation candle
    b2_mask = trades["variant"].astype(str).str.startswith("B2") if "variant" in trades.columns else None
    if b2_mask is None:
        # fall back to subtype
        b2_mask = trades["subtype"].astype(str).str.startswith("B2") if "subtype" in trades.columns else pd.Series([True] * len(trades))
    n_b2 = int(b2_mask.sum())
    print(f"  B2 trades: {n_b2:,}  (B1 trades will get NaN conf_delta_w*)")

    trades["date"] = pd.to_datetime(trades["date"]).dt.date
    date_min = trades["date"].min()
    date_max = trades["date"].max()
    print(f"  date range: {date_min} -> {date_max}")

    print(f"  loading volumetric 5-min for full range...")
    bars_all, levels_all = load_5min_features((date_min, date_max + dt.timedelta(days=1)))
    print(f"  bars: {len(bars_all):,}  levels: {len(levels_all):,}")

    bars_by_day   = dict(list(bars_all.groupby("session_date", sort=True)))
    levels_by_day = dict(list(levels_all.groupby("session_date", sort=True)))

    # init conf_delta columns (both wick and half variants)
    for N in WINDOW_NS:
        trades[f"conf_delta_wick_w{N}"] = np.nan
        trades[f"conf_delta_half_w{N}"] = np.nan

    t0 = time.time()
    processed = 0
    skipped = 0
    for d, day_trades in trades.groupby("date", sort=False):
        if d not in bars_by_day:
            skipped += len(day_trades)
            continue
        day_bars = bars_by_day[d].sort_values("bar_open_time").reset_index(drop=True)
        day_levels = levels_by_day.get(d, pd.DataFrame())
        if not day_levels.empty:
            day_levels = day_levels.sort_values("bar_open_time")
            levels_by_bar = dict(list(day_levels.groupby("bar_open_time")))
        else:
            levels_by_bar = {}

        for trade_idx in day_trades.index:
            t = trades.loc[trade_idx]
            if not b2_mask.loc[trade_idx]:
                continue
            entry_idx = int(t["entry_bar_idx"])
            conf_idx  = entry_idx - 1
            if conf_idx < 0 or conf_idx >= len(day_bars):
                continue
            conf_bar = day_bars.iloc[conf_idx]
            body_low  = float(min(conf_bar["open"], conf_bar["close"]))
            body_high = float(max(conf_bar["open"], conf_bar["close"]))
            bar_low   = float(conf_bar["low"])
            bar_high  = float(conf_bar["high"])
            bar_lvls = levels_by_bar.get(conf_bar["bar_open_time"], pd.DataFrame())
            wick_res, half_res = compute_conf_delta_zones(
                bar_lvls, bar_low, bar_high, body_low, body_high,
                t["direction"], WINDOW_NS)
            for N, v in wick_res.items():
                trades.at[trade_idx, f"conf_delta_wick_w{N}"] = v
            for N, v in half_res.items():
                trades.at[trade_idx, f"conf_delta_half_w{N}"] = v
            processed += 1

    elapsed = time.time() - t0
    print(f"  processed {processed:,} B2 trades in {elapsed:.0f}s  (skipped {skipped})")

    # Coverage report
    for col_prefix in ["conf_delta_wick_w", "conf_delta_half_w"]:
        for N in WINDOW_NS:
            col = f"{col_prefix}{N}"
            n_pop = trades[col].notna().sum()
            n_nan = trades[col].isna().sum()
            if n_pop > 0:
                print(f"  {col:<25}  populated: {n_pop:>6,}  NaN: {n_nan:>6,}  "
                      f"range: {trades[col].min():>+8.1f} -> {trades[col].max():>+8.1f}  median: {trades[col].median():+.1f}")

    trades.to_parquet(trades_path, compression="zstd", index=False)
    print(f"  wrote back to {trades_path}")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "both"
    if target in ("is", "both"):
        augment_trades(TRADES_IS, "IN-SAMPLE")
    if target in ("oos", "both"):
        augment_trades(TRADES_OOS, "OUT-OF-SAMPLE")


if __name__ == "__main__":
    sys.exit(main())

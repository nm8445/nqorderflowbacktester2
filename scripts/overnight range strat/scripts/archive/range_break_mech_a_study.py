"""Mechanism A — consistent bid/ask zone (NT8 footprint "circles") study.

CURRENT DEFAULTS (synced with Mech B):
  - LEVEL_BAND captured at 30 NQ pts (wide); adaptive band applied at summary time
  - Lookahead-fixed MQ levels (trade day D uses prior-day's EOD MQ)
  - SYMMETRIC wick-anchored exits (TP_dist = base + tp_M*ATR;
    SL_dist = base + sl_P*ATR;  R:R = 1:1 when tp_M = sl_P)
  - Chained Mode 1 applied at summary time

For each bar, scan the prior K bars and find the strongest 2-tick price zone
(bid_level + bid_level+0.25) where total volume across the K-bar window
exceeds threshold M. The zone may sit anywhere in the bar's range (body or
wick). When detected, emit a signal in the prevailing bias direction.

Variants:
  A1 = enter on detection bar close (next-bar open)
  A2 = wait for next bar to close in bias direction (bullish for LONG,
       bearish for SHORT), then enter at bar after.

Sweeps (applied at summary time):
  K_BARS in {2, 3, 4}
  M (min contracts in 2-tick zone over K bars) in [30, 50, ..., 510]
  band_K, tp_M, sl_M (same as Mech B)

Reuses bias-state segmentation, level-proximity check, and forward-simulation
from range_break_entry_signal_study. Output: separate trades parquet with
mechanism='A'. Same column schema as B trades.
"""

from __future__ import annotations

import datetime as dt
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Reuse helpers from the Mech B script (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_signal_study import (  # noqa: E402
    SWEEP, IN_SAMPLE_END, ENTRY_N, RECLAIM_FLIP, LEVEL_BAND, TICK,
    load_range_per_day, load_mq_levels, levels_for_date,
    load_5min_features, compute_bias_path, near_level, simulate_trade,
)

OUT_DIR    = Path(__file__).parent
PARQUET_DIR = OUT_DIR / "parquets"
PARQUET_DIR.mkdir(exist_ok=True)
TRADES_OUT = PARQUET_DIR / "entry_signal_trades_mech_a.parquet"

K_BARS     = SWEEP["A_K_BARS"]
MIN_M_FLOOR = min(SWEEP["A_min_M"])  # 30 (record any zone >= floor; filter at summary)


def detect_mech_a_signals(levels_day: pd.DataFrame, bars_day: pd.DataFrame
                           ) -> list[dict]:
    """For each (bar_index, K) pair, find the strongest 2-tick zone over the
    last K bars. Record (bar_open_time, K, zone_bid, accumulated_vol).
    Returns one row per (bar, K) where vol >= MIN_M_FLOOR."""
    if levels_day.empty or bars_day.empty:
        return []
    # Build per-(bar, level) total vol matrix
    g = (levels_day.assign(vol=levels_day["buy_vol"] + levels_day["sell_vol"])
                   .groupby(["bar_open_time", "level_price"])["vol"].sum()
                   .unstack(fill_value=0.0)
                   .sort_index())
    if g.empty:
        return []
    # Align bar_open_times with bars_day; use only bars present in bars_day
    valid_times = pd.DatetimeIndex(bars_day["bar_open_time"].tolist())
    g = g.loc[g.index.isin(valid_times)]
    if len(g) < min(K_BARS):
        return []

    levels_arr = np.array(sorted(g.columns.tolist()))
    g = g[levels_arr.tolist()]
    vol_arr = g.values  # rows: bars, cols: levels
    times = g.index.tolist()

    # Map bar_open_time -> idx in bars_day
    bot_to_idx = {t: i for i, t in enumerate(bars_day["bar_open_time"].tolist())}

    # Pre-compute "is adjacent 2-tick pair?" for each level pair j -> j+1
    is_adjacent = np.abs(np.diff(levels_arr) - TICK) < 1e-9   # bool length L-1

    detections = []
    for K in K_BARS:
        if K > len(vol_arr):
            continue
        # rolling sum across K rows
        cum = np.cumsum(vol_arr, axis=0)
        # rolling[i] = sum of rows [i-K+1, i]; for i >= K-1
        rolling = np.empty_like(vol_arr)
        rolling[K-1:] = cum[K-1:] - np.r_[np.zeros((1, vol_arr.shape[1])),
                                           cum[:-K]]
        # For each bar i >= K-1, find best 2-tick zone
        for i in range(K - 1, len(vol_arr)):
            row = rolling[i]
            # 2-tick zone vol at column j (bid) + column j+1 (ask)
            zone_vol = row[:-1] + row[1:]
            zone_vol = np.where(is_adjacent, zone_vol, -1.0)
            j_best = int(np.argmax(zone_vol))
            best = float(zone_vol[j_best])
            if best < MIN_M_FLOOR:
                continue
            t = times[i]
            bidx = bot_to_idx.get(t)
            if bidx is None:
                continue
            detections.append({
                "bar_idx": bidx,
                "bar_open_time": t,
                "K": K,
                "zone_bid": float(levels_arr[j_best]),
                "zone_ask": float(levels_arr[j_best + 1]),
                "accum_vol": best,
            })
    return detections


def process_one_day(date: dt.date, bars_day: pd.DataFrame,
                    levels_day: pd.DataFrame, rng_row: pd.Series,
                    mq_levels: np.ndarray) -> list[dict]:
    if bars_day.empty or levels_day.empty:
        return []
    bars_day = bars_day.sort_values("bar_open_time").reset_index(drop=True)
    ohi = float(rng_row["overnight_high"])
    olo = float(rng_row["overnight_low"])

    # Bias state path (same as Mech B)
    bias, origin, inside_cnt, location = compute_bias_path(bars_day, ohi, olo)
    bars_day["bias"] = bias
    bars_day["origin"] = origin
    bars_day["inside_cnt"] = inside_cnt
    bars_day["location"] = location

    # Pre-entry-extreme reference (same as Mech B)
    high_above = []; cum_h = -np.inf
    for _, b in bars_day.iterrows():
        if b["high"] > ohi: cum_h = max(cum_h, b["high"])
        high_above.append(cum_h if np.isfinite(cum_h) else None)
    bars_day["high_above_hod_to_now"] = high_above
    low_below = []; cum_l = np.inf
    for _, b in bars_day.iterrows():
        if b["low"] < olo: cum_l = min(cum_l, b["low"])
        low_below.append(cum_l if np.isfinite(cum_l) else None)
    bars_day["low_below_lod_to_now"] = low_below

    # Detect Mech A signals
    detections = detect_mech_a_signals(levels_day, bars_day)
    if not detections:
        return []

    trades = []
    for det in detections:
        idx = det["bar_idx"]
        bar = bars_day.iloc[idx]
        if bar["bias"] == "NEUTRAL":
            continue
        direction = bar["bias"]

        # Short close-location filter: must close below LOD (strict) or wick below
        if direction == "SHORT":
            if bar["close"] >= olo and bar["low"] >= olo:
                continue
            strict_short = bar["close"] < olo
        else:
            strict_short = None

        # Level proximity (signal candle [low, high])
        near, lvl = near_level(bar["low"], bar["high"], mq_levels)
        if not near:
            continue

        next_bar = bars_day.iloc[idx + 1] if idx + 1 < len(bars_day) else None

        for variant in ["A1", "A2"]:
            if variant == "A1":
                entry_idx = idx + 1
            else:
                if next_bar is None:
                    continue
                if direction == "LONG" and not next_bar["bullish"]:
                    continue
                if direction == "SHORT" and next_bar["bullish"]:
                    continue
                entry_idx = idx + 2

            if entry_idx >= len(bars_day):
                continue

            atr = float(bars_day.iloc[entry_idx]["atr14"])
            if not np.isfinite(atr) or atr <= 0:
                continue

            sl_anchor = (float(bar["low"]) if direction == "LONG"
                          else float(bar["high"]))

            if direction == "LONG":
                if bar["origin"] == "LONG_BREAK":
                    te = bar["high_above_hod_to_now"]
                    target_extreme = te if te is not None else ohi
                else:
                    target_extreme = ohi
            else:
                te = bar["low_below_lod_to_now"]
                target_extreme = te if te is not None else olo

            sim = simulate_trade(bars_day, entry_idx, direction,
                                  sl_anchor, atr, ohi, olo, target_extreme)
            if sim is None:
                continue

            trade = {
                "date": date,
                "entry_time": bars_day.iloc[entry_idx]["bar_open_time"],
                "signal_time": bar["bar_open_time"],
                "mechanism": "A",
                "subtype": f"A_K{det['K']}",   # K_BARS as subtype identifier
                "variant": variant,
                "direction": direction,
                "bias_origin": bar["origin"],
                "location": bar["location"],
                "inside_cnt": int(bar["inside_cnt"]),
                "strict_short": strict_short,
                "near_level": lvl,
                "level_dist": abs(((bar["low"] + bar["high"]) / 2) - lvl),
                # Mech A specific
                "a_K_bars":      int(det["K"]),
                "a_zone_bid":    det["zone_bid"],
                "a_zone_ask":    det["zone_ask"],
                "a_accum_vol":   det["accum_vol"],
                # Mech B fields not applicable; placeholder NaN
                "lw_sell_ratio":    np.nan,
                "lw_total":         np.nan,
                "lw_pinbar_ratio":  np.nan,
                "uw_buy_ratio":     np.nan,
                "uw_total":         np.nan,
                "uw_pinbar_ratio":  np.nan,
                "ohi": ohi, "olo": olo,
                "signal_open": bar["open"], "signal_close": bar["close"],
                "signal_high": bar["high"], "signal_low": bar["low"],
                **sim,
            }
            trades.append(trade)
    return trades


def main(test_only_first_n: int | None = None):
    print("loading per-day range data...")
    rng = load_range_per_day()
    print(f"  range parquet: {len(rng)} days")

    print("loading MenthorQ levels...")
    mq = load_mq_levels()
    print(f"  MQ levels: {len(mq)} days")

    print("loading volumetric 5-min...")
    bars_all, levels_all = load_5min_features(
        (dt.date(2020, 12, 1), IN_SAMPLE_END))

    bars_by_day = dict(list(bars_all.groupby("session_date", sort=True)))
    levels_by_day = dict(list(levels_all.groupby("session_date", sort=True)))
    print(f"  {len(bars_by_day)} session days to process")

    # Lookahead-free MQ lookup: trade day D uses prior MQ-available day's EOD levels.
    mq_dates_sorted = sorted(mq.index.tolist())

    def prior_mq_day(session_date):
        prev = None
        for md in mq_dates_sorted:
            if md < session_date: prev = md
            else: break
        return prev

    if test_only_first_n is not None:
        keys = list(bars_by_day.keys())[:test_only_first_n]
        bars_by_day = {k: bars_by_day[k] for k in keys}

    all_trades = []
    t0 = time.time()
    for i, (d, bars_day) in enumerate(bars_by_day.items(), 1):
        if d not in rng.index:
            continue
        rng_row = rng.loc[d]
        levels_day = levels_by_day.get(d, pd.DataFrame())
        prior_d = prior_mq_day(d)
        if prior_d is None:
            continue
        mq_levels = levels_for_date(mq, prior_d)
        if mq_levels.size == 0:
            continue
        try:
            day_trades = process_one_day(d, bars_day, levels_day, rng_row, mq_levels)
            all_trades.extend(day_trades)
        except Exception as e:
            print(f"  ! {d}: {type(e).__name__}: {e}")
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(bars_by_day)}  elapsed={elapsed:.0f}s  "
                  f"trades={len(all_trades)}")

    if not all_trades:
        print("no Mech A trades produced")
        return

    df = pd.DataFrame(all_trades)
    df.to_parquet(TRADES_OUT, compression="zstd", index=False)
    print(f"\nwrote {TRADES_OUT}  ({len(df)} trades, {len(df.columns)} cols)")


if __name__ == "__main__":
    test_n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(main(test_n))

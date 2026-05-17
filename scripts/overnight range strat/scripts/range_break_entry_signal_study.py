"""Range-break entry-signal study (current defaults).

Pipeline per session day:
  1. Bias-state segmentation over RTH (9:30-17:00 ET):
       - First 3-consec close outside overnight range fires LONG_BREAK or SHORT_BREAK
       - SHORT_BREAK with 5+ inside closes flips to SHORT_FLIP_LONG bias
       - Fresh opposite break (3-consec) overrides bias once

  2. Per-bar signal detection (Mechanism B — single-candle wick absorption):
       - Pinbar shape: pinbar_ratio = relevant_wick / max(body, TICK) >= PINBAR_X_FLOOR
         (X=0.75 floor; finer X1/X2/X3 sweeps confirmed not to add edge — see
         shape-bucket study `b2_shape_bucket_breakdown.txt`)
       - Level proximity: signal candle [low, high] overlaps level zone
         [level - band, level + band]. Backtest CAPTURES wide (LEVEL_BAND=30 NQ pts)
         and the summary applies an adaptive band per trade:
           band = clip(BAND_K * ATR_5m, BAND_MIN=5, BAND_MAX=20)   # default BAND_K=0.25
       - Approach direction: prior bar close > level (LONG) / < level (SHORT)
       - Strict-shorts gate: close < OLO (loose variant kept as flag for sweeping)
       - Windowed orderflow scan: per 5-tick window in wick zone, sum buy/sell;
         pick max-ratio window. Filter: |delta| in best window >= delta_D.

     Variants (entry timing):
         B1 = signal candle alone -> enter at next bar open
         B2 = wait one bar; require it to close in bias direction; enter the bar after
         B3 = B2 + that confirmation bar must have positive (LONG) / negative (SHORT) total delta

  3. Forward simulation under SYMMETRIC WICK-ANCHORED exits:
        base_distance = |entry - wick_anchor|
        SL_distance   = base + sl_P * ATR    (SL_price = wick - sign*sl_P*ATR)
        TP_distance   = base + tp_M * ATR    (TP equidistant to SL when tp_M=sl_P)
     Walk forward bar-by-bar, record first-hit bar index for every (tp_M, sl_M) combo.

  4. Trade dedupe (CHAINED Mode 1, applied at summary time):
       Walk signals chronologically per day. Take a trade only when no position
       is currently open (previous trade exited via TP/SL or end-of-day).

  5. MenthorQ level lookup is LOOKAHEAD-FREE: trade day D uses prior-day's EOD MQ
     levels (computed from D-1 close, available before D's RTH open).

Sample: in-sample only (session_date < 2025-01-01). MenthorQ levels parquet
must be built first via build_menthorq_levels_bulk.py.

Top in-sample config:
  B2, X=1.25, N=5, D=70, strict=True, BAND_K=0.25, tp_M=sl_P=1.0
  -> n=657, total +3,471 pts, mean +5.28, WR 54.3%, PF 1.30, Sharpe 2.09
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

# ============================== Config ==============================

VOL5M       = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
MQ_LEVELS   = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
OUT_DIR     = Path(__file__).parent
PARQUET_DIR = OUT_DIR / "parquets"
PARQUET_DIR.mkdir(exist_ok=True)
RANGE_DAY   = PARQUET_DIR / "range_break_full_sequence_per_day.parquet"
TRADES_OUT  = PARQUET_DIR / "entry_signal_trades.parquet"
SUMMARY_OUT = PARQUET_DIR / "entry_signal_summary.parquet"

IN_SAMPLE_END = dt.date(2025, 1, 1)   # exclusive — 2020-12 to 2024-12

ENTRY_N      = 3        # consecutive 5-min closes outside range to fire trigger
RECLAIM_FLIP = 5        # SHORT_BREAK -> LONG flip threshold (consecutive inside)
LEVEL_BAND   = 30.0     # +/- NQ points around each MQ level (WIDE CAPTURE — adaptive
                         # filter applied post-hoc in summary; this is the upper
                         # bound for any adaptive sweep BAND_K * ATR config)
TICK         = 0.25
TICK_VAL     = 5.0      # USD per NQ point... actually $5 per tick; $20/pt
POINT_VAL    = 20.0     # $20 per NQ point per contract

# Sweeps -- recorded as attributes per trade; aggregated post-hoc
SWEEP = {
    "B_pinbar_X":      [0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50],
    "B_window_N":      [5, 10, 15, 20],          # ticks (1.25 / 2.5 / 3.75 / 5.0 NQ pts)
    "B_ratio_R":       [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "B_delta_D":       [30, 50, 100, 200, 500],
    "A_K_BARS":        [2, 3, 4],
    "A_min_M":         list(range(30, 511, 20)),
    # Symmetric wick-anchored exits (locked in — only framework used):
    #   base_dist = |entry - wick_anchor|
    #   SL_dist   = base_dist + sl_P * ATR    (SL_price = wick - sign*sl_P*ATR)
    #   TP_dist   = base_dist + tp_M * ATR    (TP equidistant from entry as SL when tp_M=sl_P)
    "TP_atr_mults":    [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
                         2.25, 2.50, 2.75, 3.00],
    "SL_atr_mults":    [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
                         2.25, 2.50, 2.75, 3.00],
}

PINBAR_X_FLOOR = 0.75   # min pinbar shape to even consider as signal candidate
WINDOW_NS = SWEEP["B_window_N"]
ATR_PERIOD = 14   # 5-min ATR


# ========================== Data loading ============================

def load_range_per_day() -> pd.DataFrame:
    df = pd.read_parquet(RANGE_DAY)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.set_index("date")


def load_mq_levels() -> pd.DataFrame:
    df = pd.read_parquet(MQ_LEVELS)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    level_cols = [c for c in df.columns
                  if c.endswith("_nq") and (c.startswith("qqq_") or c.startswith("ndx_"))]
    return df.set_index("date")[level_cols]


def levels_for_date(mq: pd.DataFrame, d: dt.date) -> np.ndarray:
    """Return non-NaN array of NQ-mapped level prices for the given date."""
    if d not in mq.index:
        return np.array([])
    row = mq.loc[d]
    return row.dropna().values.astype(float)


def load_5min_features(date_filter: tuple[dt.date, dt.date]) -> pd.DataFrame:
    """Load volumetric 5-min level data and aggregate to per-bar features.

    Per-bar features: open, high, low, close, total_vol, total_buy, total_sell,
    lower_wick_buy, lower_wick_sell, upper_wick_buy, upper_wick_sell,
    plus per-level dictionary for Mech-A scanning (kept as separate frame).

    Returns (bars_df, levels_df) where bars_df has one row per (date, bar_open_time)
    and levels_df has one row per (date, bar_open_time, level_price).
    """
    print(f"  loading volumetric 5-min ...")
    df = pd.read_parquet(VOL5M)
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    lo, hi = date_filter
    df = df[(df["session_date"] >= lo) & (df["session_date"] < hi)].copy()
    print(f"  filtered to {len(df):,} level rows")

    # Filter to RTH only (9:30 to 17:00 ET inclusive)
    bot = pd.to_datetime(df["bar_open_time"], utc=True).dt.tz_convert("America/New_York")
    df["bar_open_time"] = bot
    df["bar_close_time"] = pd.to_datetime(df["bar_close_time"],
                                          utc=True).dt.tz_convert("America/New_York")
    rth_mask = ((bot.dt.time >= dt.time(9, 30)) &
                (bot.dt.time <= dt.time(17, 0)))
    df = df[rth_mask & df["open"].notna()].copy()
    print(f"  RTH-only rows: {len(df):,}")

    # Body / wick zone vectorized
    df["body_low"]  = df[["open", "close"]].min(axis=1)
    df["body_high"] = df[["open", "close"]].max(axis=1)
    in_lower_wick = df["level_price"] < df["body_low"]
    in_upper_wick = df["level_price"] > df["body_high"]
    df["lw_buy"]  = np.where(in_lower_wick, df["buy_vol"],  0)
    df["lw_sell"] = np.where(in_lower_wick, df["sell_vol"], 0)
    df["uw_buy"]  = np.where(in_upper_wick, df["buy_vol"],  0)
    df["uw_sell"] = np.where(in_upper_wick, df["sell_vol"], 0)

    print("  aggregating per-bar features...")
    bars = (df.groupby(["session_date", "bar_open_time"], sort=True)
              .agg(open=("open", "first"),
                   high=("high", "first"),
                   low=("low", "first"),
                   close=("close", "first"),
                   total_vol=("bar_total_vol", "first"),
                   total_buy=("buy_vol", "sum"),
                   total_sell=("sell_vol", "sum"),
                   lw_buy=("lw_buy", "sum"),
                   lw_sell=("lw_sell", "sum"),
                   uw_buy=("uw_buy", "sum"),
                   uw_sell=("uw_sell", "sum"))
              .reset_index())
    bars["delta"] = bars["total_buy"] - bars["total_sell"]
    bars["body_size"] = (bars["close"] - bars["open"]).abs()
    bars["lower_wick_size"] = bars[["open", "close"]].min(axis=1) - bars["low"]
    bars["upper_wick_size"] = bars["high"] - bars[["open", "close"]].max(axis=1)
    bars["range_size"] = bars["high"] - bars["low"]
    bars["bullish"] = bars["close"] > bars["open"]

    # 14-period ATR (Wilder) per day
    print("  computing per-day ATR-14...")
    bars["prev_close"] = bars.groupby("session_date")["close"].shift(1)
    bars["tr"] = np.maximum.reduce([
        (bars["high"] - bars["low"]).values,
        (bars["high"] - bars["prev_close"]).abs().values,
        (bars["low"] - bars["prev_close"]).abs().values,
    ])
    bars["atr14"] = (bars.groupby("session_date")["tr"]
                        .transform(lambda s: s.ewm(alpha=1/ATR_PERIOD,
                                                   adjust=False, min_periods=1).mean()))

    return bars, df  # bars = aggregated; df = level-resolved (for Mech A)


# ====================== Bias-state segmentation =====================

def compute_bias_path(bars_day: pd.DataFrame, ohi: float, olo: float):
    """Walk a day's 5-min closes and compute, per bar:
       bias_state ∈ {NEUTRAL, LONG, SHORT}
       bias_origin ∈ {NEUTRAL, LONG_BREAK, SHORT_BREAK, SHORT_FLIP_LONG}
       inside_count (consecutive closes back inside after a break)
       location ∈ {ABOVE_HOD, INSIDE, BELOW_LOD}

    Rules:
      - First trigger (3-consec close outside range) sets bias.
      - SHORT_BREAK: 5+ consecutive closes inside range -> flip to LONG (origin=SHORT_FLIP_LONG)
      - Fresh opposite break (3-consec) flips bias to align (max 1 flip via this path).
      - LONG_BREAK does NOT flip on inside-count alone.
    """
    n = len(bars_day)
    state = np.full(n, "NEUTRAL", dtype=object)
    origin = np.full(n, "NEUTRAL", dtype=object)
    inside_cnt = np.zeros(n, dtype=int)
    location = np.full(n, "INSIDE", dtype=object)

    closes = bars_day["close"].values
    above = 0; below = 0; inside_run = 0
    bias = "NEUTRAL"; orig = "NEUTRAL"; flipped = False

    for i, c in enumerate(closes):
        # location for this bar
        if c > ohi:
            location[i] = "ABOVE_HOD"
        elif c < olo:
            location[i] = "BELOW_LOD"
        else:
            location[i] = "INSIDE"

        # update streak counters using THIS bar's close
        if c > ohi:
            above += 1; below = 0
        elif c < olo:
            below += 1; above = 0
        else:
            above = 0; below = 0

        # update inside_run only if we already have a SHORT bias and close is inside
        if orig.startswith("SHORT") and not flipped and location[i] == "INSIDE":
            inside_run += 1
        elif location[i] != "INSIDE":
            inside_run = 0

        # check for first trigger (sets bias)
        if bias == "NEUTRAL":
            if above >= ENTRY_N:
                bias = "LONG"; orig = "LONG_BREAK"
            elif below >= ENTRY_N:
                bias = "SHORT"; orig = "SHORT_BREAK"
        else:
            # already in a bias; check for flip events
            if not flipped:
                if bias == "SHORT" and inside_run >= RECLAIM_FLIP:
                    bias = "LONG"; orig = "SHORT_FLIP_LONG"; flipped = True
                # fresh opposite-side break aligns bias and counts as a flip
                if bias == "SHORT" and above >= ENTRY_N:
                    bias = "LONG"; orig = "LONG_BREAK"; flipped = True
                if bias == "LONG" and below >= ENTRY_N and orig != "LONG_BREAK":
                    # only flip from LONG via a fresh SHORT break
                    bias = "SHORT"; orig = "SHORT_BREAK"; flipped = True
                # special: if originally LONG_BREAK and a SHORT break fires,
                # data does NOT support flipping (per user). So we do NOT flip
                # in that case. The above branch is gated so the first
                # LONG_BREAK ignores any later SHORT_BREAK.
            # (we deliberately ignore further state changes after one flip)

        state[i] = bias
        origin[i] = orig
        inside_cnt[i] = inside_run

    return state, origin, inside_cnt, location


# ========================= Signal detection =========================

def evaluate_mech_b(bar) -> dict:
    """Compute the achieved Mech-B feature values for this bar.
    Returns dict with all values; downstream filters by sweep thresholds.

    Long-side features (use lower wick):
      L_kind  = 'bullish' or 'bearish_pinbar' or None
      L_lw_sell, L_lw_buy, L_lw_total
      L_lw_sell_ratio = lw_sell / max(lw_buy, 1)
      L_pinbar_ratio  = lower_wick_size / max(body_size, TICK)

    Short-side features (use upper wick):
      S_kind  = 'bearish' or 'bullish_pinbar' or None
      S_uw_buy, S_uw_sell, S_uw_total
      S_uw_buy_ratio  = uw_buy / max(uw_sell, 1)
      S_pinbar_ratio  = upper_wick_size / max(body_size, TICK)
    """
    out = {}
    body = max(bar["body_size"], TICK)

    # LONG side
    if bar["bullish"] and bar["lw_sell"] > 0:
        out["L_kind"] = "bullish"
        out["L_lw_sell"]   = bar["lw_sell"]
        out["L_lw_buy"]    = bar["lw_buy"]
        out["L_lw_total"]  = bar["lw_sell"] + bar["lw_buy"]
        out["L_lw_sell_ratio"] = bar["lw_sell"] / max(bar["lw_buy"], 1)
        out["L_pinbar_ratio"]  = bar["lower_wick_size"] / body
    elif (not bar["bullish"]) and bar["lw_sell"] > 0:
        out["L_kind"] = "bearish_pinbar"
        out["L_lw_sell"]   = bar["lw_sell"]
        out["L_lw_buy"]    = bar["lw_buy"]
        out["L_lw_total"]  = bar["lw_sell"] + bar["lw_buy"]
        out["L_lw_sell_ratio"] = bar["lw_sell"] / max(bar["lw_buy"], 1)
        out["L_pinbar_ratio"]  = bar["lower_wick_size"] / body
    else:
        out["L_kind"] = None
        for k in ["L_lw_sell", "L_lw_buy", "L_lw_total",
                  "L_lw_sell_ratio", "L_pinbar_ratio"]:
            out[k] = 0.0

    # SHORT side
    if (not bar["bullish"]) and bar["uw_buy"] > 0:
        out["S_kind"] = "bearish"
        out["S_uw_buy"]    = bar["uw_buy"]
        out["S_uw_sell"]   = bar["uw_sell"]
        out["S_uw_total"]  = bar["uw_buy"] + bar["uw_sell"]
        out["S_uw_buy_ratio"]  = bar["uw_buy"] / max(bar["uw_sell"], 1)
        out["S_pinbar_ratio"]  = bar["upper_wick_size"] / body
    elif bar["bullish"] and bar["uw_buy"] > 0:
        out["S_kind"] = "bullish_pinbar"
        out["S_uw_buy"]    = bar["uw_buy"]
        out["S_uw_sell"]   = bar["uw_sell"]
        out["S_uw_total"]  = bar["uw_buy"] + bar["uw_sell"]
        out["S_uw_buy_ratio"]  = bar["uw_buy"] / max(bar["uw_sell"], 1)
        out["S_pinbar_ratio"]  = bar["upper_wick_size"] / body
    else:
        out["S_kind"] = None
        for k in ["S_uw_buy", "S_uw_sell", "S_uw_total",
                  "S_uw_buy_ratio", "S_pinbar_ratio"]:
            out[k] = 0.0
    return out


def compute_windowed_absorption(bar_levels: pd.DataFrame, body_low: float,
                                 body_high: float, direction: str,
                                 window_ns: list[int]) -> dict:
    """Sliding-window scan across the relevant wick (lower for LONG, upper for SHORT).
    For each window size N, find the window with the HIGHEST absorbed-side ratio.

    Absorbed-side ratio:
      LONG:  sell_vol / buy_vol  in window  (sellers absorbed, want ratio HIGH)
      SHORT: buy_vol  / sell_vol in window  (buyers  absorbed, want ratio HIGH)

    Returns dict: N -> (lo, hi, ratio, delta_signed_in_bias, total_vol)
      delta is buy - sell (negative when sellers dominate, positive when buyers do).
      For LONG signals, |delta| is what matters (we want sellers > buyers).
      For SHORT signals, |delta| same — we want buyers > sellers.
    """
    if bar_levels.empty:
        return {N: (None, None, 0.0, 0.0, 0.0) for N in window_ns}
    if direction == "LONG":
        wick = bar_levels[bar_levels["level_price"] < body_low].sort_values("level_price")
    else:
        wick = bar_levels[bar_levels["level_price"] > body_high].sort_values("level_price")
    if wick.empty:
        return {N: (None, None, 0.0, 0.0, 0.0) for N in window_ns}

    bv = wick["buy_vol"].values.astype(float)
    sv = wick["sell_vol"].values.astype(float)
    lp = wick["level_price"].values
    n_levels = len(wick)
    results = {}
    cb = np.cumsum(bv); cs = np.cumsum(sv)
    for N in window_ns:
        if n_levels < N:
            results[N] = (None, None, 0.0, 0.0, 0.0)
            continue
        win_buy  = np.concatenate([[cb[N-1]], cb[N:] - cb[:-N]])
        win_sell = np.concatenate([[cs[N-1]], cs[N:] - cs[:-N]])
        if direction == "LONG":
            ratios = win_sell / np.clip(win_buy, 1, None)
        else:
            ratios = win_buy / np.clip(win_sell, 1, None)
        best_idx = int(np.argmax(ratios))
        lo = float(lp[best_idx])
        hi = float(lp[best_idx + N - 1])
        buy_sum = float(win_buy[best_idx])
        sell_sum = float(win_sell[best_idx])
        ratio = float(ratios[best_idx])
        delta = buy_sum - sell_sum
        total = buy_sum + sell_sum
        results[N] = (lo, hi, ratio, delta, total)
    return results


def near_level(low: float, high: float, levels: np.ndarray, band: float = LEVEL_BAND
               ) -> tuple[bool, float | None]:
    """Bar [low, high] overlaps any [level-band, level+band] zone."""
    if levels.size == 0:
        return False, None
    lo, hi = low - band, high + band   # expanded bar interval
    # overlap: bar.lo <= level+band AND bar.hi >= level-band
    overlap_mask = (lo <= levels) & (hi >= levels)
    # but actually bar's [low, high] vs [level-band, level+band]: bar overlaps
    # zone iff bar.low <= level+band AND bar.high >= level-band
    overlap_mask = (low <= levels + band) & (high >= levels - band)
    if overlap_mask.any():
        # nearest level to bar midpoint
        mid = (low + high) / 2
        idx = int(np.argmin(np.abs(levels - mid)))
        return True, float(levels[idx])
    return False, None


# ======================= Mechanism A — consistent bid/ask zones =======================

def detect_mech_a_zones(day_levels: pd.DataFrame, k_bars: int, min_m: int):
    """For each bar index i in the day, check if there exists a 2-tick zone
    (bid_level, bid_level + TICK) such that across the K bars ending at i,
    the combined volume at those two adjacent levels exceeds `min_m`.
    Returns list of (bar_open_time, zone_bid_price, accumulated_vol) tuples
    indexed by bar-time-of-detection.

    Coarse implementation: per bar, compute total volume per level; sliding
    window across K bars sums per-level volumes; check 2-tick adjacent pair.
    """
    # Aggregate per (bar_open_time, level_price)
    agg = (day_levels.groupby(["bar_open_time", "level_price"])
                     .agg(vol=("buy_vol", lambda s: s.sum() + day_levels.loc[s.index, "sell_vol"].sum()))
                     .reset_index())
    # That's awkward; do it cleanly:
    g = (day_levels.assign(vol=lambda d: d["buy_vol"] + d["sell_vol"])
                   .groupby(["bar_open_time", "level_price"])["vol"].sum()
                   .unstack(fill_value=0))   # index=bar_open_time, cols=level_price
    g = g.sort_index()
    times = g.index.tolist()
    levels = sorted(g.columns.tolist())
    arr = g[levels].values   # rows = bars, cols = levels

    detections = []
    for i in range(k_bars - 1, len(times)):
        win = arr[i - k_bars + 1: i + 1].sum(axis=0)   # per-level total over K bars
        # Check 2-tick adjacent pairs (level_j + level_j+1 if they're TICK apart)
        for j in range(len(levels) - 1):
            if abs(levels[j + 1] - levels[j] - TICK) < 1e-9:
                if win[j] + win[j + 1] >= min_m:
                    detections.append((times[i], levels[j], win[j] + win[j + 1]))
                    break  # only the strongest zone for this bar
    return detections


# ======================= Forward simulation =========================

def simulate_trade(bars_day: pd.DataFrame, entry_idx: int, direction: str,
                    sl_anchor: float, atr: float, ohi: float, olo: float,
                    pre_entry_extreme: float | None) -> dict:
    """Walk forward from entry_idx (entry occurs at this bar's open).
    Records the time-of-first-touch for each TP and SL threshold.
    Trade exits at 17:00 (last bar in day) if no threshold hit.

    Returns a dict keyed by ('tp', mult) and ('sl', pad) -> hit_idx_or_None,
    plus 'mfe', 'mae', 'close_pnl' (signed, in NQ pts).
    """
    n = len(bars_day)
    if entry_idx + 1 >= n:
        return None
    entry_price = bars_day.iloc[entry_idx]["open"]
    sign = 1 if direction == "LONG" else -1

    # base_distance = |entry - wick_anchor|, common to both SL and TP
    base_dist = sign * (entry_price - sl_anchor)

    # Symmetric wick-anchored exits:
    #   SL_dist = base + sl_P * ATR   (SL_price = wick - sign*sl_P*ATR)
    #   TP_dist = base + tp_M * ATR   (TP equidistant from entry as SL when tp_M=sl_P)
    sl_thresholds = {p: sl_anchor - sign * p * atr for p in SWEEP["SL_atr_mults"]}
    tp_thresholds = {m: entry_price + sign * (base_dist + m * atr)
                      for m in SWEEP["TP_atr_mults"]}

    target_extreme = pre_entry_extreme
    tp_hit = {m: None for m in tp_thresholds}
    sl_hit = {p: None for p in sl_thresholds}
    extreme_hit = None

    mfe = 0.0
    mae = 0.0
    bars_held = 0

    for k in range(entry_idx, n):
        bar = bars_day.iloc[k]
        if k == entry_idx:
            continue
        bars_held += 1
        excursion_up = bar["high"] - entry_price
        excursion_dn = bar["low"] - entry_price
        if direction == "LONG":
            mfe = max(mfe, excursion_up)
            mae = min(mae, excursion_dn)
            for m, t in tp_thresholds.items():
                if tp_hit[m] is None and bar["high"] >= t:
                    tp_hit[m] = k
            for p, s in sl_thresholds.items():
                if sl_hit[p] is None and bar["low"] <= s:
                    sl_hit[p] = k
            if target_extreme is not None and extreme_hit is None and bar["high"] >= target_extreme:
                extreme_hit = k
        else:
            mfe = max(mfe, -excursion_dn)
            mae = min(mae, -excursion_up)
            for m, t in tp_thresholds.items():
                if tp_hit[m] is None and bar["low"] <= t:
                    tp_hit[m] = k
            for p, s in sl_thresholds.items():
                if sl_hit[p] is None and bar["high"] >= s:
                    sl_hit[p] = k
            if target_extreme is not None and extreme_hit is None and bar["low"] <= target_extreme:
                extreme_hit = k

    close_pnl = sign * (bars_day.iloc[-1]["close"] - entry_price)

    out = {
        "entry_price": float(entry_price),
        "atr_at_entry": float(atr),
        "sl_anchor": float(sl_anchor),
        "entry_to_anchor": float(base_dist),
        "entry_bar_idx": int(entry_idx),       # NEW: for chained Mode 1 dedupe
        "extreme_hit_idx": extreme_hit if extreme_hit is not None else -1,
        "extreme_target": float(target_extreme) if target_extreme is not None else np.nan,
        "mfe_pts": float(mfe),
        "mae_pts": float(mae),
        "close_pnl_pts": float(close_pnl),
        "bars_held": bars_held,
    }
    for p, k in sl_hit.items():
        out[f"sl_{p:.2f}_idx"] = k if k is not None else -1
    for m, k in tp_hit.items():
        out[f"tp_{m:.2f}_idx"] = k if k is not None else -1
    return out


# ======================== Main pipeline ===========================

def process_one_day(date: dt.date, bars_day: pd.DataFrame, levels_day: pd.DataFrame,
                    rng_row: pd.Series, mq_levels: np.ndarray) -> list[dict]:
    """Detect signals and simulate trades for one day. Returns list of trade dicts."""
    if bars_day.empty:
        return []
    bars_day = bars_day.sort_values("bar_open_time").reset_index(drop=True)
    ohi = float(rng_row["overnight_high"])
    olo = float(rng_row["overnight_low"])

    # 1) Bias state path
    bias, origin, inside_cnt, location = compute_bias_path(bars_day, ohi, olo)
    bars_day["bias"] = bias
    bars_day["origin"] = origin
    bars_day["inside_cnt"] = inside_cnt
    bars_day["location"] = location

    # 2) For each bar, compute Mech-B feature attributes
    feat_cols = ["L_kind", "L_lw_sell", "L_lw_buy", "L_lw_total",
                 "L_lw_sell_ratio", "L_pinbar_ratio",
                 "S_kind", "S_uw_buy", "S_uw_sell", "S_uw_total",
                 "S_uw_buy_ratio", "S_pinbar_ratio"]
    feat = bars_day.apply(evaluate_mech_b, axis=1, result_type="expand")
    bars_day = pd.concat([bars_day, feat], axis=1)

    # 3) Pre-compute the highest above-HOD price reached up to each bar
    #    (for LONG continuation TP target)
    high_above = []
    cum_high_above = -np.inf
    for i, b in bars_day.iterrows():
        if b["high"] > ohi:
            cum_high_above = max(cum_high_above, b["high"])
        high_above.append(cum_high_above if np.isfinite(cum_high_above) else None)
    bars_day["high_above_hod_to_now"] = high_above
    low_below = []
    cum_low_below = np.inf
    for i, b in bars_day.iterrows():
        if b["low"] < olo:
            cum_low_below = min(cum_low_below, b["low"])
        low_below.append(cum_low_below if np.isfinite(cum_low_below) else None)
    bars_day["low_below_lod_to_now"] = low_below

    # body_low / body_high for wick zone scans
    bars_day["body_low_calc"]  = bars_day[["open", "close"]].min(axis=1)
    bars_day["body_high_calc"] = bars_day[["open", "close"]].max(axis=1)

    # Group level-resolved data by bar_open_time for fast per-bar lookup
    levels_by_bar = {t: g for t, g in levels_day.groupby("bar_open_time")}

    # 4) Walk bars and detect Mech-B signal candidates with the new rule set:
    #    - Universal pinbar shape: relevant_wick / body >= PINBAR_X_FLOOR (0.75)
    #    - Approach direction: prior bar close > level (long), < level (short)
    #    - Windowed absorption scan recorded per window size in {5,10,15,20} ticks
    trades = []
    bars_day["state_id"] = (bars_day["origin"] + ":" +
                             bars_day["bias"]).astype("category").cat.codes

    for idx, bar in bars_day.iterrows():
        if bar["bias"] == "NEUTRAL":
            continue
        if idx == 0:
            continue   # need prior bar for approach filter

        direction = bar["bias"]
        body = max(bar["body_size"], TICK)

        # Universal pinbar shape filter
        if direction == "LONG":
            pinbar_ratio = bar["lower_wick_size"] / body
        else:
            pinbar_ratio = bar["upper_wick_size"] / body
        if pinbar_ratio < PINBAR_X_FLOOR:
            continue

        # Level proximity
        near, lvl = near_level(bar["low"], bar["high"], mq_levels)
        if not near:
            continue

        # Approach direction filter: prior bar close > level (long) / < level (short)
        prior_close = float(bars_day.iloc[idx - 1]["close"])
        if direction == "LONG" and prior_close <= lvl:
            continue
        if direction == "SHORT" and prior_close >= lvl:
            continue

        # SHORT close-location filter (strict / loose recorded)
        if direction == "SHORT":
            close_below = bar["close"] < olo
            wick_below  = bar["low"]   < olo
            if not (close_below or wick_below):
                continue
            strict_short = bool(close_below)
        else:
            strict_short = None

        # Windowed absorption scan across full wick
        bar_lvls = levels_by_bar.get(bar["bar_open_time"], pd.DataFrame())
        win_results = compute_windowed_absorption(
            bar_lvls, bar["body_low_calc"], bar["body_high_calc"],
            direction, WINDOW_NS)

        # Subtype is now descriptive only (universal pinbar means same filter)
        if direction == "LONG":
            subtype = "LONG_BULL" if bar["bullish"] else "LONG_BEAR"
        else:
            subtype = "SHORT_BEAR" if not bar["bullish"] else "SHORT_BULL"

        next_bar = bars_day.iloc[idx + 1] if idx + 1 < len(bars_day) else None

        for variant in ["B1", "B2", "B3"]:
            if variant == "B1":
                entry_idx = idx + 1
            else:
                if next_bar is None:
                    continue
                if direction == "LONG" and not next_bar["bullish"]:
                    continue
                if direction == "SHORT" and next_bar["bullish"]:
                    continue
                if variant == "B3":
                    if direction == "LONG" and next_bar["delta"] <= 0:
                        continue
                    if direction == "SHORT" and next_bar["delta"] >= 0:
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
                "mechanism": "B",
                "subtype": subtype,
                "variant": variant,
                "direction": direction,
                "bias_origin": bar["origin"],
                "location": bar["location"],
                "inside_cnt": int(bar["inside_cnt"]),
                "strict_short": strict_short,
                "near_level": lvl,
                "level_dist": abs(((bar["low"] + bar["high"]) / 2) - lvl),
                "prior_close": prior_close,
                "pinbar_ratio": float(pinbar_ratio),
                # NEW: directional wick dominance + body-as-pct-of-range
                "pinbar_dominance_ratio": (
                    float(bar["lower_wick_size"] / max(bar["upper_wick_size"], TICK))
                    if direction == "LONG"
                    else float(bar["upper_wick_size"] / max(bar["lower_wick_size"], TICK))
                ),
                "body_pct_range": (
                    float(bar["body_size"] / max(bar["range_size"], TICK))
                ),
                # Whole-wick stats kept for diagnostics (NOT used in new filters)
                "lw_sell_ratio": (bar["lw_sell"] / max(bar["lw_buy"], 1))
                                  if direction == "LONG" else 0.0,
                "lw_total":     float(bar["lw_sell"] + bar["lw_buy"])
                                  if direction == "LONG" else 0.0,
                "uw_buy_ratio": (bar["uw_buy"] / max(bar["uw_sell"], 1))
                                  if direction == "SHORT" else 0.0,
                "uw_total":     float(bar["uw_sell"] + bar["uw_buy"])
                                  if direction == "SHORT" else 0.0,
                "ohi": ohi, "olo": olo,
                "signal_open": bar["open"], "signal_close": bar["close"],
                "signal_high": bar["high"], "signal_low": bar["low"],
                **sim,
            }
            # Windowed absorption per window size
            for N in WINDOW_NS:
                lo, hi, ratio, delta, total = win_results[N]
                trade[f"abs_lo_w{N}"]    = lo if lo is not None else np.nan
                trade[f"abs_hi_w{N}"]    = hi if hi is not None else np.nan
                trade[f"abs_ratio_w{N}"] = float(ratio)
                trade[f"abs_delta_w{N}"] = float(delta)
                trade[f"abs_total_w{N}"] = float(total)
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

    # group by session_date for daily processing
    bars_by_day = dict(list(bars_all.groupby("session_date", sort=True)))
    print(f"  {len(bars_by_day)} session days to process")

    levels_by_day = dict(list(levels_all.groupby("session_date", sort=True)))

    # Lookahead-free MQ lookup: trade day D uses prior MQ-available day's EOD levels.
    mq_dates_sorted = sorted(mq.index.tolist())
    prev_mq_for = {}
    for i, md in enumerate(mq_dates_sorted):
        if i == 0:
            continue
        # for any session date strictly AFTER mq_dates_sorted[i-1], that's the prior MQ day
        prev_mq_for[md] = mq_dates_sorted[i-1]

    def prior_mq_day(session_date: dt.date) -> dt.date | None:
        """Return the latest MQ-available date STRICTLY before session_date."""
        # binary search-style: walk sorted MQ dates
        prev = None
        for md in mq_dates_sorted:
            if md < session_date:
                prev = md
            else:
                break
        return prev

    if test_only_first_n is not None:
        keys = list(bars_by_day.keys())[:test_only_first_n]
        bars_by_day = {k: bars_by_day[k] for k in keys}

    all_trades = []
    t0 = time.time()
    skipped_no_mq = 0
    for i, (d, bars_day) in enumerate(bars_by_day.items(), 1):
        if d not in rng.index:
            continue
        rng_row = rng.loc[d]
        levels_day = levels_by_day.get(d, pd.DataFrame())
        # Use PRIOR trading day's EOD MQ levels (lookahead-free)
        prior_d = prior_mq_day(d)
        if prior_d is None:
            skipped_no_mq += 1
            continue
        mq_levels = levels_for_date(mq, prior_d)
        if mq_levels.size == 0:
            skipped_no_mq += 1
            continue
        try:
            day_trades = process_one_day(d, bars_day, levels_day, rng_row, mq_levels)
            all_trades.extend(day_trades)
        except Exception as e:
            print(f"  ! {d}: {type(e).__name__}: {e}")
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(bars_by_day)}  elapsed={elapsed:.0f}s  trades={len(all_trades)}")

    if not all_trades:
        print("no trades produced; check filters / data")
        return

    df = pd.DataFrame(all_trades)
    df.to_parquet(TRADES_OUT, compression="zstd", index=False)
    print(f"\nwrote {TRADES_OUT}  ({len(df)} trades, {len(df.columns)} cols)")


if __name__ == "__main__":
    test_n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(main(test_n))

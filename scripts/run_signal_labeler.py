"""
Label absorption signals across all RTH sessions in the training window.

For each trading date 2025-03-17 to 2025-11-30:
  1. Load enriched ticks for overnight + RTH window (from parquet cache if available)
  2. Build 40-range bars for the RTH session (09:30-11:00 ET)
  3. Compute VWAP over RTH ticks
  4. Compute session realized_vol from RTH range bar closes
  5. Load overnight regime label from output/directional_cache/
  6. Run label_absorption_signals()

Tick cache
----------
Normalized enriched ticks (18:00 prev day through 11:00 RTH end) are saved per
trading date as parquet files under output/tick_cache/.  On subsequent runs the
parquet is loaded directly, skipping the slow .dbn load + normalize_enriched()
step.  Delete a date's parquet file to force a refresh from the .dbn source.

After collecting all training signals:
  - Fit move_mean and move_std on the full training move_normalized distribution
  - Recompute move_z_score = (move_normalized - mean) / std
  - Recompute sustained = (move_z_score >= 1.5) and (mae_normalized <= 0.30)

Output: output/absorption_signals_labeled.csv
"""

import gc
import joblib
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize_enriched
from nqbt.analysis.range_bars import build_range_bars
from nqbt.analysis.vwap import compute_vwap
from nqbt.analysis.signal_labeler import label_absorption_signals
from nqbt.analysis.delta_nodes import load_nodes_cached, extract_delta_nodes
from nqbt.analysis.volume_profile import VolumeProfile

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START = "2025-03-17"
TRAIN_END   = "2025-11-30"

OUTPUT_CSV      = Path("output/absorption_signals_labeled.csv")
ON_MODEL_PATH   = Path("output/directional_overnight_hmm.pkl")
ON_CACHE_DIR    = Path("output/directional_cache")
TICK_CACHE_DIR  = Path("output/tick_cache")
DELTA_CACHE_DIR = Path("output/delta_node_cache")

ET       = "America/New_York"
MIN_BARS = 5
SUSTAINED_Z   = 1.5
SUSTAINED_MAE = 0.30
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_CSV.parent.mkdir(exist_ok=True)
TICK_CACHE_DIR.mkdir(exist_ok=True)


def trading_dates(start: str, end: str) -> list[date]:
    return [d.date() for d in pd.bdate_range(start=start, end=end)]


def load_overnight_label(trading_date: date, on_hmm) -> str:
    """Load overnight feature cache and return the HMM classification label."""
    p = ON_CACHE_DIR / f"{trading_date}_overnight.npy"
    if not p.exists() or on_hmm is None:
        return "unknown"
    try:
        feat = np.load(p)
        if feat.ndim != 2 or feat.shape[0] < 5:
            return "unknown"
        label, _ = on_hmm.classify(feat)
        return label
    except Exception:
        return "unknown"


def session_realized_vol(bars: list) -> float:
    """sqrt(sum(log_returns^2)) over range bar closes — consistent with features.py."""
    if len(bars) < 2:
        return 1e-6
    closes = np.array([b.close for b in bars], dtype=float)
    log_returns = np.log(closes[1:] / closes[:-1])
    rv = float(np.sqrt(np.sum(log_returns ** 2)))
    return rv if rv > 1e-9 else 1e-6


def _tick_cache_path(trading_date: date) -> Path:
    return TICK_CACHE_DIR / f"{trading_date}_ticks.parquet"


def _load_ticks(trading_date: date) -> pd.DataFrame | None:
    """
    Load enriched ticks for trading_date's window (18:00 prev day → 11:00 RTH).
    Returns from parquet cache if available, otherwise loads from .dbn and caches.
    Returns None on error.
    """
    cache = _tick_cache_path(trading_date)
    if cache.exists():
        return pd.read_parquet(cache)

    prev_day = trading_date - timedelta(days=1)
    try:
        raw      = fetch_and_load(str(prev_day), str(trading_date + timedelta(days=1)))
        enriched = normalize_enriched(raw)
        del raw
        gc.collect()
    except Exception as e:
        print(f"  ERROR loading {trading_date}: {e}")
        return None

    # Slice to the window we actually need: 18:00 prev day → 11:00 RTH end
    on_start = pd.Timestamp(f"{prev_day} 18:00:00",     tz=ET)
    rth_end  = pd.Timestamp(f"{trading_date} 11:00:00", tz=ET)
    window   = enriched[(enriched.index >= on_start) & (enriched.index <= rth_end)].copy()
    del enriched
    gc.collect()

    window.to_parquet(cache)
    return window


def process_date(trading_date: date, on_hmm) -> pd.DataFrame:
    ticks = _load_ticks(trading_date)
    if ticks is None:
        return pd.DataFrame()

    prev_day  = trading_date - timedelta(days=1)
    on_start  = pd.Timestamp(f"{prev_day} 18:00:00",     tz=ET)
    rth_start = pd.Timestamp(f"{trading_date} 09:30:00", tz=ET)
    rth_end   = pd.Timestamp(f"{trading_date} 11:00:00", tz=ET)

    rth_ticks     = ticks[(ticks.index >= rth_start) & (ticks.index <= rth_end)]
    profile_ticks = ticks[(ticks.index >= on_start)  & (ticks.index <= rth_end)]

    if len(rth_ticks) < 10:
        return pd.DataFrame()

    # 40-range bars for RTH
    rth_bars = build_range_bars(rth_ticks, range_ticks=40, ticks_per_level=5)
    if len(rth_bars) < MIN_BARS:
        return pd.DataFrame()

    # RTH VWAP at session end
    vwap_df  = compute_vwap(rth_ticks)
    vwap_val = float(vwap_df["vwap"].iloc[-1])  if not vwap_df.empty else 0.0
    vwap_std = float(vwap_df["std"].iloc[-1])   if not vwap_df.empty else 1.0
    if vwap_std < 1e-9:
        vwap_std = 1.0

    rv = session_realized_vol(rth_bars)

    # Build session value area for structural node tagging (RTH-based)
    try:
        node_profile = VolumeProfile.build(rth_ticks)
        price_levels = {"vah": node_profile.vah, "val": node_profile.val, "poc": node_profile.poc}
    except Exception:
        price_levels = {}

    from datetime import timedelta as _td

    # ── Three-source node lookup ──────────────────────────────────────────────
    # Source 1: prior 2 RTH sessions (by trading day, not calendar day)
    prior_nodes: list = []
    prior_bdates = pd.bdate_range(end=str(trading_date - _td(days=1)), periods=2)
    for p_bdate in prior_bdates:
        prior_nodes.extend(load_nodes_cached(p_bdate.date()))

    # Source 2: overnight nodes for today (built on-the-fly, NOT cached)
    on_ticks = ticks[
        (ticks.index >= pd.Timestamp(f"{trading_date - _td(days=1)} 18:00:00", tz=ET))
        & (ticks.index <  pd.Timestamp(f"{trading_date} 09:30:00",             tz=ET))
    ]
    overnight_nodes: list = []
    if len(on_ticks) >= 10:
        try:
            on_profile    = VolumeProfile.build(on_ticks)
            on_price_lvls = {"vah": on_profile.vah, "val": on_profile.val, "poc": on_profile.poc}
        except Exception:
            on_price_lvls = {}
        overnight_nodes = extract_delta_nodes(
            ticks            = on_ticks,
            price_levels     = on_price_lvls,
            node_delta_percentile = 50,
        )

    # Source 3: current RTH nodes (built from RTH ticks, cached with reaction scan)
    current_nodes = load_nodes_cached(
        session_date          = trading_date,
        ticks                 = rth_ticks,
        price_levels          = price_levels,
        node_delta_percentile = 50,
        rth_bars              = rth_bars,
        force_rebuild         = False,
    )

    # Combined node list (look-ahead bias prevention happens per-signal in labeler)
    delta_nodes = prior_nodes + overnight_nodes + current_nodes

    overnight_label = load_overnight_label(trading_date, on_hmm)
    rth_start_ts    = pd.Timestamp(f"{trading_date} 09:30:00", tz=ET)

    result = label_absorption_signals(
        bars                  = rth_bars,
        overnight_label       = overnight_label,
        profile_ticks         = profile_ticks,
        vwap                  = vwap_val,
        vwap_std              = vwap_std,
        session_realized_vol  = rv,
        delta_nodes           = delta_nodes,
        rth_start_ts          = rth_start_ts,
    )
    return result


def print_sustained_moves(df: pd.DataFrame) -> None:
    sustained = df[df["sustained"]].copy()
    if sustained.empty:
        print("No sustained moves found.")
        return

    # Convert signal_time to ET
    sustained["signal_time_et"] = pd.to_datetime(sustained["signal_time"]).dt.tz_convert(ET)

    print(f"\n{'Date':<12}  {'Time (ET)':<10}  {'Side':<14}  {'Entry':>8}  {'Exit':>8}  {'Points':>8}  {'Z-Score':>8}  {'Location'}")
    print("-" * 95)
    for _, row in sustained.sort_values("signal_time").iterrows():
        side      = row["absorption_side"]
        entry     = row["signal_price"]
        exit_     = row["end_price"]
        points    = exit_ - entry if side == "sell_absorbed" else entry - exit_
        z         = row["move_z_score"]
        loc       = row["price_location"]
        time_str  = row["signal_time_et"].strftime("%H:%M:%S")
        date_str  = str(row["date"])
        print(f"{date_str:<12}  {time_str:<10}  {side:<14}  {entry:>8.2f}  {exit_:>8.2f}  {points:>+8.2f}  {z:>8.2f}  {loc}")


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    if total == 0:
        print("No signals found.")
        return

    buy  = (df["absorption_side"] == "buy_absorbed").sum()
    sell = (df["absorption_side"] == "sell_absorbed").sum()
    sust = df["sustained"].sum()

    print(f"Total signals:          {total}")
    print(f"  buy_absorbed:         {buy}  ({buy/total:.0%})")
    print(f"  sell_absorbed:        {sell}  ({sell/total:.0%})")
    print(f"Sustained moves:        {sust}  ({sust/total:.0%} of total)")

    for loc in ["at_vah", "at_val", "outside_vah", "outside_val", "inside_va"]:
        sub  = df[df["price_location"] == loc]
        n    = len(sub)
        s    = sub["sustained"].sum()
        pct  = f"{s/n:.0%}" if n > 0 else "n/a"
        print(f"  {loc:<14}:  {n:>4}  ({pct} sustained)")

    print("By overnight regime:")
    for regime in df["overnight_regime"].unique():
        sub = df[df["overnight_regime"] == regime]
        n   = len(sub)
        s   = sub["sustained"].sum()
        pct = f"{s/n:.0%}" if n > 0 else "n/a"
        print(f"  {regime:<20}: {n:>4}  ({pct} sustained)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load overnight HMM for regime labels
    on_hmm = None
    if ON_MODEL_PATH.exists():
        print(f"Loading overnight model from {ON_MODEL_PATH}...")
        on_hmm = joblib.load(ON_MODEL_PATH)
    else:
        print(f"WARNING: {ON_MODEL_PATH} not found — overnight_regime will be 'unknown'")

    dates = trading_dates(TRAIN_START, TRAIN_END)
    print(f"\nProcessing {len(dates)} training dates ({TRAIN_START} to {TRAIN_END})...")

    all_frames = []
    for i, d in enumerate(dates):
        cached = _tick_cache_path(d).exists()
        print(f"  {d}  ({i+1}/{len(dates)}){'  [cached]' if cached else ''}")
        frame = process_date(d, on_hmm)
        if not frame.empty:
            all_frames.append(frame)

    if not all_frames:
        print("No signals found across any session.")
        raise SystemExit(0)

    df = pd.concat(all_frames, ignore_index=True)

    # ── Fit cross-session move normalization on training data ─────────────────
    print(f"\nFitting move normalization on {len(df)} training signals...")
    move_vals = df["move_normalized"].to_numpy()
    move_mean = float(np.mean(move_vals))
    move_std  = float(np.std(move_vals))
    if move_std < 1e-9:
        move_std = 1.0
    print(f"  move_normalized  mean={move_mean:+.6f}  std={move_std:.6f}")

    # Recompute z-scores and sustained with fitted normalization
    df["move_z_score"] = (df["move_normalized"] - move_mean) / move_std
    df["sustained"]    = (
        (df["move_z_score"]   >= SUSTAINED_Z)
        & (df["mae_normalized"] <= SUSTAINED_MAE)
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  SIGNAL LABELER SUMMARY")
    print("=" * 55)
    print_summary(df)
    print()
    print(f"  move normalization: mean={move_mean:+.6f}  std={move_std:.6f}")
    print(f"  sustained threshold: z >= {SUSTAINED_Z}  mae <= {SUSTAINED_MAE}")

    print()
    print("=" * 95)
    print("  SUSTAINED MOVES")
    print("=" * 95)
    print_sustained_moves(df)

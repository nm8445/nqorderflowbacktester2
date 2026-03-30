"""
Feature vector builder: computes ML-ready feature vectors for each reaction event.

Features are computed at multiple lookback windows (5, 10, 20, 30, 50 bars)
plus per-level delta features, overnight session features, and contextual features.

Input: reaction_events DataFrame (output of reaction_scanner.scan_date_range)
       bars_by_date: dict mapping date_str -> list[RangeBar] (precomputed)
       ticks_by_date: dict mapping date_str -> pd.DataFrame (RTH ticks, precomputed)
Output: feature_vectors DataFrame, one row per reaction event
"""

from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from nqbt.analysis.range_bars import RangeBar
from nqbt.analysis.per_level_delta import compute_per_level_delta
from nqbt.analysis.overnight_features import OVERNIGHT_FEATURE_NAMES

LOOKBACK_WINDOWS     = [5, 10, 20, 30, 50]
OVERNIGHT_CACHE_DIR  = Path("output/directional_cache")
HMM_PATH             = Path("output/directional_overnight_hmm.pkl")

ET = "America/New_York"
RTH_OPEN_TIME = "09:30"
RTH_TOTAL_MINUTES = 405  # 09:30 to 16:15

_PRICE_LOC_CODE = {
    "outside_vah": 0,
    "at_vah":      1,
    "inside_va":   2,
    "at_val":      3,
    "outside_val": 4,
}


def _load_hmm() -> object | None:
    """Load the DirectionalHMM joblib file, or return None if absent/unavailable."""
    if not HMM_PATH.exists():
        return None
    try:
        return joblib.load(HMM_PATH)
    except Exception:
        # hmmlearn or other dependency may be unavailable
        return None


def _load_overnight_matrix(session_date: str) -> np.ndarray | None:
    """
    Load overnight feature matrix (n_bars, 8) for a session.
    Returns None if file absent or shape invalid.
    """
    path = OVERNIGHT_CACHE_DIR / f"{session_date}_overnight.npy"
    if not path.exists():
        return None
    mat = np.load(str(path))
    if mat.ndim == 2 and mat.shape[1] == len(OVERNIGHT_FEATURE_NAMES):
        return mat
    return None


def _load_overnight_features(session_date: str) -> np.ndarray | None:
    """
    Return session-mean overnight feature vector (8,) by averaging the
    per-bar feature matrix across the overnight session bars.
    Returns None if file absent.
    """
    mat = _load_overnight_matrix(session_date)
    if mat is None:
        return None
    return np.nanmean(mat, axis=0)


def _classify_overnight_regime(hmm_model, session_date: str) -> str:
    """
    Use DirectionalHMM.classify() on the full overnight feature matrix.
    Returns 'directional', 'non_directional', or 'ambiguous'/'unknown'.
    """
    if hmm_model is None:
        return "unknown"
    mat = _load_overnight_matrix(session_date)
    if mat is None or mat.shape[0] == 0:
        return "unknown"
    try:
        label, _conf = hmm_model.classify(mat)
        return label
    except Exception:
        return "unknown"


def _bar_features_window(bars: list[RangeBar], up_to_idx: int, window: int) -> dict:
    """
    Compute summary statistics over a lookback window of bars ending at up_to_idx.

    Parameters
    ----------
    bars : list[RangeBar]
    up_to_idx : int
        Index of the event bar (inclusive).
    window : int
        Number of bars to look back.

    Returns
    -------
    dict
        Keys prefixed with w{window}_.
    """
    start = max(0, up_to_idx - window + 1)
    window_bars = bars[start : up_to_idx + 1]
    prefix = f"w{window}_"

    if not window_bars:
        return {
            f"{prefix}mean_delta":      0.0,
            f"{prefix}std_delta":       0.0,
            f"{prefix}mean_vol":        0.0,
            f"{prefix}std_vol":         0.0,
            f"{prefix}bull_frac":       0.0,
            f"{prefix}mean_delta_pct":  0.0,
            f"{prefix}cumulative_delta": 0,
        }

    deltas    = np.array([b.delta     for b in window_bars], dtype=float)
    vols      = np.array([b.total_vol for b in window_bars], dtype=float)
    closes    = np.array([b.close     for b in window_bars], dtype=float)
    opens     = np.array([b.open      for b in window_bars], dtype=float)

    mean_delta = float(np.mean(deltas))
    std_delta  = float(np.std(deltas, ddof=1)) if len(deltas) >= 2 else 0.0
    mean_vol   = float(np.mean(vols))
    std_vol    = float(np.std(vols, ddof=1))   if len(vols) >= 2 else 0.0
    bull_frac  = float(np.mean(closes > opens))

    # delta_pct per bar (skip bars with zero volume)
    valid_mask = vols > 0
    if np.any(valid_mask):
        delta_pcts = deltas[valid_mask] / vols[valid_mask]
        mean_delta_pct = float(np.mean(delta_pcts))
    else:
        mean_delta_pct = 0.0

    cumulative_delta = int(np.sum(deltas))

    return {
        f"{prefix}mean_delta":       mean_delta,
        f"{prefix}std_delta":        std_delta,
        f"{prefix}mean_vol":         mean_vol,
        f"{prefix}std_vol":          std_vol,
        f"{prefix}bull_frac":        bull_frac,
        f"{prefix}mean_delta_pct":   mean_delta_pct,
        f"{prefix}cumulative_delta": cumulative_delta,
    }


def _per_level_delta_features(event_bar: RangeBar, percentile: float = 70.0) -> dict:
    """
    Compute per-level delta features directly from bar.levels (BarLevel dict).

    Returns a dict with keys:
        zone_buy_vol, zone_sell_vol, zone_delta, zone_abs_delta,
        zone_pct_rank, zone_significant, n_significant_zones, bar_delta_pct
    """
    levels = event_bar.levels  # dict[float, BarLevel]

    default = {
        "zone_buy_vol":       0,
        "zone_sell_vol":      0,
        "zone_delta":         0,
        "zone_abs_delta":     0,
        "zone_pct_rank":      0.0,
        "zone_significant":   False,
        "n_significant_zones": 0,
        "bar_delta_pct":      0.0,
    }

    if not levels:
        return default

    zone_prices  = np.array(list(levels.keys()),  dtype=float)
    buy_vols     = np.array([lv.buy_vol  for lv in levels.values()], dtype=float)
    sell_vols    = np.array([lv.sell_vol for lv in levels.values()], dtype=float)
    deltas       = buy_vols - sell_vols
    abs_deltas   = np.abs(deltas)

    n = len(abs_deltas)
    if n > 1:
        pct_ranks = np.array([
            float(np.mean(abs_deltas < v)) * 100.0
            for v in abs_deltas
        ])
    else:
        pct_ranks = np.zeros(n)

    significant_mask = pct_ranks >= percentile
    n_significant    = int(np.sum(significant_mask))

    # Find nearest zone to bar.close
    event_zone = (event_bar.close // 1.25) * 1.25
    # Round to 4 decimal places to match how level keys are stored
    event_zone = round(event_zone, 4)

    # Find the matching zone key (exact float match with 4dp tolerance)
    match_idx = None
    for i, zp in enumerate(zone_prices):
        if abs(round(zp, 4) - event_zone) < 1e-6:
            match_idx = i
            break

    if match_idx is not None:
        zone_buy  = int(buy_vols[match_idx])
        zone_sell = int(sell_vols[match_idx])
        zone_d    = int(deltas[match_idx])
        zone_ad   = int(abs_deltas[match_idx])
        zone_pr   = float(pct_ranks[match_idx])
        zone_sig  = bool(significant_mask[match_idx])
    else:
        zone_buy  = 0
        zone_sell = 0
        zone_d    = 0
        zone_ad   = 0
        zone_pr   = 0.0
        zone_sig  = False

    total_vol = event_bar.total_vol
    bar_delta_pct = float(event_bar.delta / total_vol) if total_vol > 0 else 0.0

    return {
        "zone_buy_vol":        zone_buy,
        "zone_sell_vol":       zone_sell,
        "zone_delta":          zone_d,
        "zone_abs_delta":      zone_ad,
        "zone_pct_rank":       zone_pr,
        "zone_significant":    zone_sig,
        "n_significant_zones": n_significant,
        "bar_delta_pct":       bar_delta_pct,
    }


def build_feature_vectors(
    events: pd.DataFrame,
    bars_by_date: dict,
    ticks_by_date: dict,
) -> pd.DataFrame:
    """
    Build one feature vector row per reaction event.

    Parameters
    ----------
    events : pd.DataFrame
        Output of reaction_scanner.scan_date_range().
    bars_by_date : dict
        {date_str: list[RangeBar]} for all session dates present in events.
    ticks_by_date : dict
        {date_str: pd.DataFrame} RTH ticks for all session dates present.

    Returns
    -------
    pd.DataFrame
        One row per event, index reset.
    """
    hmm_model = _load_hmm()

    rows: list[dict] = []

    for _, event in events.iterrows():
        session_date = event["session_date"]
        bar_idx      = int(event["bar_idx"])

        bars = bars_by_date.get(session_date)
        if bars is None or bar_idx >= len(bars):
            continue

        event_bar = bars[bar_idx]

        feat: dict = {}

        # --- Lookback window features ---
        for w in LOOKBACK_WINDOWS:
            feat.update(_bar_features_window(bars, bar_idx, w))

        # --- Per-level delta features ---
        feat.update(_per_level_delta_features(event_bar))

        # --- Overnight features ---
        ov_vec = _load_overnight_features(session_date)
        if ov_vec is not None and len(ov_vec) == len(OVERNIGHT_FEATURE_NAMES):
            for name, val in zip(OVERNIGHT_FEATURE_NAMES, ov_vec):
                feat[f"ov_{name}"] = float(val)
        else:
            for name in OVERNIGHT_FEATURE_NAMES:
                feat[f"ov_{name}"] = float("nan")

        # HMM regime via DirectionalHMM.classify()
        feat["overnight_regime_str"] = _classify_overnight_regime(hmm_model, session_date)

        # --- Contextual features ---
        event_time_utc = event["event_time"]
        try:
            if hasattr(event_time_utc, "tz_convert"):
                event_time_et = event_time_utc.tz_convert(ET)
            else:
                event_time_et = pd.Timestamp(event_time_utc).tz_convert(ET)

            rth_open_et   = pd.Timestamp(f"{session_date} {RTH_OPEN_TIME}", tz=ET)
            minutes_since = (event_time_et - rth_open_et).total_seconds() / 60.0
            session_time_pct = float(minutes_since / RTH_TOTAL_MINUTES)
        except Exception:
            session_time_pct = float("nan")

        price_loc  = str(event["price_location"])
        feat["session_time_pct"]    = session_time_pct
        feat["bar_idx"]             = bar_idx
        feat["price_location_code"] = _PRICE_LOC_CODE.get(price_loc, 2)
        feat["vwap_band_proximity"] = int(bool(event["vwap_band_proximity"]))
        feat["vah_dist_pts"]        = float(event["bar_close"]) - float(event["vah"])
        feat["val_dist_pts"]        = float(event["bar_close"]) - float(event["val"])
        feat["vwap_dist_pts"]       = float(event["bar_close"]) - float(event["vwap"])
        feat["std2_dist_upper"]     = float(event["bar_close"]) - float(event["std2_upper"])
        feat["std2_dist_lower"]     = float(event["bar_close"]) - float(event["std2_lower"])

        # --- Pass-through columns ---
        feat["session_date"]    = session_date
        feat["event_time"]      = event["event_time"]
        feat["price_location"]  = price_loc
        feat["mfe_up"]          = float(event["mfe_up"])
        feat["mfe_down"]        = float(event["mfe_down"])
        feat["net_move"]        = float(event["net_move"])

        rows.append(feat)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)

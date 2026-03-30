"""
Inspect HMM regime predictions for specific trading dates.

Shows for each date:
  - Predicted overnight and RTH regime
  - Bar distribution: fraction of bars the Viterbi path assigned to each state
  - All three state emission means side-by-side for comparison
  - Bar-count and data quality info

Normalization:
  Overnight : per-session z-score (same as training)
  RTH       : cross-session z-score via output/rth_scaler.pkl (same scaler
              fitted on all training sessions during hmm_study.py)

Requires trained models in output/ — run hmm_study.py first.
Results written to output/regime_inspect.txt
"""

import gc
import json
import joblib
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize_enriched
from nqbt.analysis.minute_bars import build_minute_bars
from nqbt.analysis.vwap import compute_vwap
from nqbt.analysis.features import compute_session_features, FEATURE_NAMES
from nqbt.analysis.overnight_features import (
    compute_overnight_features,
    OVERNIGHT_FEATURE_NAMES,
)
from nqbt.analysis.hmm_regime import RegimeHMM

# ── Config ────────────────────────────────────────────────────────────────────
DATES = [
    d.strftime("%Y-%m-%d")
    for d in pd.bdate_range(start="2026-01-02", end="2026-03-09")
]

OUTPUT_FILE     = Path("output/regime_inspect.txt")
MINUTE_CACHE    = Path("output/minute_cache")
MODEL_OVERNIGHT = Path("output/overnight_hmm.pkl")
MODEL_RTH       = Path("output/rth_hmm.pkl")
SCALER_PATH     = Path("output/rth_scaler.pkl")
ET              = "America/New_York"
MIN_BARS        = 5

# Must stay in sync with hmm_study.py
CACHE_VERSION  = 1
_EXPECTED_NORM = {"overnight": "per_session", "rth": "raw"}
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FILE.parent.mkdir(exist_ok=True)


def _cache_valid(trading_date: date, session: str, expected_cols: int) -> bool:
    """Return True only if the cache file exists and its metadata is current."""
    npy  = MINUTE_CACHE / f"{trading_date}_{session}.npy"
    meta_file = MINUTE_CACHE / f"{trading_date}_{session}.meta.json"
    if not npy.exists() or not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text())
    except Exception:
        return False
    return (
        meta.get("version")       == CACHE_VERSION
        and meta.get("normalization") == _EXPECTED_NORM[session]
        and meta.get("n_features")    == expected_cols
    )


def load_session_features(
    trading_date: date,
    session: str,
    rth_scaler,
) -> tuple[np.ndarray, int]:
    """
    Load and compute features for one session on one date.
    session: 'overnight' or 'rth'
    Returns (feature_matrix, n_bars).
    Loads from feature cache if available (written by hmm_study.py).

    RTH cache stores RAW features; rth_scaler is applied here to match
    the cross-session normalization used during training.
    """
    is_overnight  = session == "overnight"
    expected_cols = len(OVERNIGHT_FEATURE_NAMES) if is_overnight else len(FEATURE_NAMES)

    if _cache_valid(trading_date, session, expected_cols):
        features = np.load(MINUTE_CACHE / f"{trading_date}_{session}.npy")
        if not is_overnight and rth_scaler is not None:
            features = rth_scaler.transform(
                np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features, features.shape[0]

    prev_day = trading_date - timedelta(days=1)
    raw      = fetch_and_load(str(prev_day), str(trading_date + timedelta(days=1)))
    enriched = normalize_enriched(raw)
    del raw
    gc.collect()

    if is_overnight:
        start = pd.Timestamp(f"{prev_day} 18:00:00", tz=ET)
        end   = pd.Timestamp(f"{trading_date} 09:30:00", tz=ET)
    else:
        start = pd.Timestamp(f"{trading_date} 09:30:00", tz=ET)
        end   = pd.Timestamp(f"{trading_date} 11:00:00", tz=ET)

    ticks = enriched[(enriched.index >= start) & (enriched.index <= end)]

    if len(ticks) < 10:
        return np.empty((0, expected_cols)), 0

    bars = build_minute_bars(ticks)
    if len(bars) < MIN_BARS:
        return np.empty((0, expected_cols)), 0

    if is_overnight:
        features = compute_overnight_features(
            ticks, bars, min_ticks=1, normalize=True
        )
    else:
        vwap_df  = compute_vwap(ticks)
        features = compute_session_features(
            ticks, bars, vwap_df,
            min_ticks=1, normalize=False,       # raw — apply cross-session scaler below
            include_session_features=True,
        )
        if rth_scaler is not None and features.shape[0] > 0:
            features = rth_scaler.transform(
                np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, len(bars)


def format_session_block(
    label:    str,
    features: np.ndarray,
    n_bars:   int,
    hmm,
    regime:   str,
) -> list[str]:
    lines = []
    lines.append(f"  {label} SESSION  ->  Predicted: {regime.upper()}  ({n_bars} bars)")
    lines.append(f"  {'-' * 72}")

    # State labels come from the HMM's state_map — works for both 2-state and 3-state models
    regimes     = [hmm.state_map[s] for s in sorted(hmm.state_map.keys())]
    reverse_map = {v: k for k, v in hmm.state_map.items()}
    n_feats    = hmm.model.means_.shape[1]
    feat_names = (
        OVERNIGHT_FEATURE_NAMES
        if n_feats == len(OVERNIGHT_FEATURE_NAMES)
        else FEATURE_NAMES[:n_feats]
    )

    proba = hmm.predict_proba_session(features)
    dist_parts = "  ".join(f"{name}: {proba.get(name, 0.0):>5.1%}" for name in regimes)
    lines.append(f"  Bar distribution:  {dist_parts}")
    lines.append(f"  {'-' * 72}")

    col_w = 12
    header = f"  {'Feature':<22}" + "".join(f"  {r:>{col_w}}" for r in regimes)
    lines.append(header)
    lines.append(f"  {'-' * 72}")

    for fi, fname in enumerate(feat_names):
        row = f"  {fname:<22}"
        for regime_name in regimes:
            state = reverse_map.get(regime_name)
            if state is not None:
                mean_val = hmm.model.means_[state, fi]
                marker = "*" if regime_name == regime else " "
                row += f"  {mean_val:>+9.4f}{marker}  "
        lines.append(row)

    lines.append(f"  (* = predicted/dominant state emission means)")
    return lines


def format_day(
    trading_date:   date,
    on_features:    np.ndarray,
    on_bars:        int,
    on_regime:      str,
    rth_features:   np.ndarray,
    rth_bars:       int,
    rth_regime:     str,
    overnight_hmm:  RegimeHMM,
    rth_hmm:        RegimeHMM,
) -> list[str]:
    lines = []
    lines.append("=" * 75)
    lines.append(f"  {trading_date.strftime('%A %B %d, %Y').upper()}")
    lines.append("=" * 75)
    lines.append("")
    lines += format_session_block("OVERNIGHT", on_features,  on_bars,  overnight_hmm, on_regime)
    lines.append("")
    lines += format_session_block("RTH",       rth_features, rth_bars, rth_hmm,       rth_regime)
    lines.append("")
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    missing = [p for p in (MODEL_OVERNIGHT, MODEL_RTH, SCALER_PATH) if not p.exists()]
    if missing:
        print(f"ERROR: Missing files: {missing}. Run scripts/hmm_study.py first.")
        exit(1)

    print("Loading models...")
    overnight_hmm: RegimeHMM = joblib.load(MODEL_OVERNIGHT)
    rth_hmm:       RegimeHMM = joblib.load(MODEL_RTH)
    rth_scaler                = joblib.load(SCALER_PATH)
    print(f"  RTH scaler loaded from {SCALER_PATH}")

    all_lines = []
    all_lines.append("HMM REGIME INSPECTION")
    all_lines.append(f"Dates: {DATES[0]} to {DATES[-1]}  ({len(DATES)} business days)")
    all_lines.append("Bar distribution = fraction of bars the Viterbi path assigned to each state")
    all_lines.append("Emission means shown for all three states (* = predicted state)")
    all_lines.append("")

    # Collect per-day summary rows for the compact table printed at the end
    summary_rows = []

    for date_str in DATES:
        trading_date = pd.Timestamp(date_str).date()
        print(f"Processing {trading_date}...")

        on_feat,  on_bars  = load_session_features(trading_date, "overnight", rth_scaler)
        rth_feat, rth_bars = load_session_features(trading_date, "rth",       rth_scaler)

        on_regime  = overnight_hmm.predict_session(on_feat)  if on_feat.shape[0]  >= MIN_BARS else "insufficient"
        rth_regime = rth_hmm.predict_session(rth_feat)       if rth_feat.shape[0] >= MIN_BARS else "insufficient"

        on_proba  = overnight_hmm.predict_proba_session(on_feat)  if on_feat.shape[0]  >= MIN_BARS else {}
        rth_proba = rth_hmm.predict_proba_session(rth_feat)       if rth_feat.shape[0] >= MIN_BARS else {}

        all_lines += format_day(
            trading_date,
            on_feat,  on_bars,  on_regime,
            rth_feat, rth_bars, rth_regime,
            overnight_hmm, rth_hmm,
        )

        summary_rows.append({
            "date":      trading_date,
            "on_bars":   on_bars,
            "on_regime": on_regime,
            "on_dir":    on_proba.get("directional",     0.0),
            "on_ndir":   on_proba.get("non_directional", 0.0),
            "rth_bars":  rth_bars,
            "rth_regime":rth_regime,
            "rth_tr":    rth_proba.get("trending", 0.0),
            "rth_ra":    rth_proba.get("ranging",  0.0),
            "rth_ch":    rth_proba.get("chop",     0.0),
        })

    # ── Compact summary table ──────────────────────────────────────────────────
    all_lines.append("=" * 75)
    all_lines.append("  SUMMARY TABLE")
    all_lines.append("=" * 75)
    hdr = (
        f"  {'Date':<12}  {'ON Bars':>7}  {'ON Regime':<15}  "
        f"{'dir%':>5} {'ndir%':>6}  "
        f"{'RTH Bars':>8}  {'RTH Regime':<10}  "
        f"{'tr%':>5} {'ra%':>5} {'ch%':>5}"
    )
    all_lines.append(hdr)
    all_lines.append("  " + "-" * 75)
    for r in summary_rows:
        all_lines.append(
            f"  {str(r['date']):<12}  {r['on_bars']:>7,}  {r['on_regime']:<15}  "
            f"{r['on_dir']:>4.0%} {r['on_ndir']:>5.0%}  "
            f"{r['rth_bars']:>8,}  {r['rth_regime']:<10}  "
            f"{r['rth_tr']:>4.0%} {r['rth_ra']:>4.0%} {r['rth_ch']:>4.0%}"
        )

    output = "\n".join(all_lines)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print(f"\nDone. Results written to {OUTPUT_FILE}")

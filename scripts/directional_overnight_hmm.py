"""
2-state directional overnight HMM study.

Research question:
    Does classifying the overnight session as DIRECTIONAL vs NON-DIRECTIONAL
    predict which RTH setup type (continuation vs reversal) performs better
    during the 9:30-11:00 window, given where price is in the value area?

Two classifiers are built and compared:

  1. Threshold classifier (built first to validate the signal exists)
     overnight is DIRECTIONAL if mean(trend_ratio across bars) >= TREND_RATIO_CUTOFF
     No training required.  Sweep the cutoff to find the best value.

  2. 2-state GaussianHMM (n_components=2, covariance_type='full')
     Trained on 8 absolute-value features per bar.
     Adds robustness by learning the full joint feature distribution.

Training  : TRAIN_START to TRAIN_END
Test      : TEST_START  to TEST_END

Features per bar (overnight only, all absolute/unsigned):
    abs_delta_pct, trend_ratio, abs_normalized_d_move, realized_vol,
    trade_arrival_rate, overnight_deceleration, abs_obi_avg, autocorr_r1

Feature cache: output/directional_cache/{date}_overnight.npy
Model saved : output/directional_overnight_hmm.pkl
Results     : output/directional_overnight_hmm.txt
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
from nqbt.analysis.range_bars import build_range_bars
from nqbt.analysis.overnight_features import (
    compute_overnight_features,
    OVERNIGHT_FEATURE_NAMES,
)
from nqbt.analysis.directional_hmm import DirectionalHMM

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START = "2025-03-17"
TRAIN_END   = "2025-11-30"
TEST_START  = "2025-12-01"
TEST_END    = "2026-03-15"

TREND_RATIO_CUTOFF = 0.55   # starting threshold for simple classifier sweep
MIN_BARS           = 5
ET                 = "America/New_York"

OUTPUT_FILE  = Path("output/directional_overnight_hmm.txt")
CACHE_DIR    = Path("output/directional_cache")
MODEL_PATH   = Path("output/directional_overnight_hmm.pkl")
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FILE.parent.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── Cache versioning ──────────────────────────────────────────────────────────
# Bump CACHE_VERSION whenever normalization, feature set, or bar type changes.
# Any cache file missing metadata or with a mismatched field is deleted and
# recomputed; a WARNING is printed so the invalidation is always visible.
CACHE_VERSION = 1
_CACHE_META = {
    "version":       CACHE_VERSION,
    "normalization": "per_session",
    "features":      OVERNIGHT_FEATURE_NAMES,
    "bar_type":      "range_bars_40",
}


def _cache_path(trading_date: date) -> Path:
    return CACHE_DIR / f"{trading_date}_overnight.npy"


def _meta_path(trading_date: date) -> Path:
    return CACHE_DIR / f"{trading_date}_overnight.meta.json"


def _cache_valid(trading_date: date) -> bool:
    """
    Return True only if the .npy file exists and its companion .meta.json
    matches the current code configuration exactly.
    Deletes both files and prints a WARNING if the cache is stale.
    """
    npy  = _cache_path(trading_date)
    meta_file = _meta_path(trading_date)
    if not npy.exists():
        return False
    if not meta_file.exists():
        print(f"WARNING: stale cache invalidated for {npy.name} (no metadata)")
        npy.unlink(missing_ok=True)
        return False
    try:
        meta = json.loads(meta_file.read_text())
    except Exception:
        print(f"WARNING: stale cache invalidated for {npy.name} (corrupt metadata)")
        npy.unlink(missing_ok=True)
        meta_file.unlink(missing_ok=True)
        return False
    if (
        meta.get("version")       != _CACHE_META["version"]
        or meta.get("normalization") != _CACHE_META["normalization"]
        or meta.get("features")      != _CACHE_META["features"]
        or meta.get("bar_type")      != _CACHE_META["bar_type"]
    ):
        print(f"WARNING: stale cache invalidated for {npy.name} (version mismatch)")
        npy.unlink(missing_ok=True)
        meta_file.unlink(missing_ok=True)
        return False
    return True


# ── Data helpers ──────────────────────────────────────────────────────────────

def trading_dates(start: str, end: str) -> list[date]:
    return [d.date() for d in pd.bdate_range(start=start, end=end)]


def load_overnight_features(trading_date: date) -> np.ndarray:
    """
    Load or compute the 8-feature overnight matrix for one trading date.
    Features cached to CACHE_DIR after first computation.
    Cache is validated against current configuration; stale files are
    deleted and recomputed automatically.
    """
    if _cache_valid(trading_date):
        return np.load(_cache_path(trading_date))

    prev_day = trading_date - timedelta(days=1)
    try:
        raw      = fetch_and_load(str(prev_day), str(trading_date + timedelta(days=1)))
        enriched = normalize_enriched(raw)
        del raw
        gc.collect()
    except Exception as e:
        print(f"  ERROR loading {trading_date}: {e}")
        return np.empty((0, len(OVERNIGHT_FEATURE_NAMES)))

    start = pd.Timestamp(f"{prev_day} 18:00:00", tz=ET)
    end   = pd.Timestamp(f"{trading_date} 09:30:00", tz=ET)
    ticks = enriched[(enriched.index >= start) & (enriched.index < end)]

    if len(ticks) < 10:
        feat = np.empty((0, len(OVERNIGHT_FEATURE_NAMES)))
    else:
        bars = build_range_bars(ticks, range_ticks=40, ticks_per_level=5)
        feat = compute_overnight_features(ticks, bars, normalize=True)

    np.save(_cache_path(trading_date), feat)
    _meta_path(trading_date).write_text(json.dumps(_CACHE_META))
    return feat


def collect(dates: list[date], label: str) -> tuple[list[np.ndarray], list[date]]:
    features, valid = [], []
    for i, d in enumerate(dates):
        print(f"  [{label}] {d}  ({i+1}/{len(dates)})")
        f = load_overnight_features(d)
        if f.shape[0] >= MIN_BARS:
            features.append(f)
            valid.append(d)
    return features, valid


# ── Threshold classifier ──────────────────────────────────────────────────────

def threshold_classify(features: np.ndarray, cutoff: float) -> str:
    """Classify one session using mean trend_ratio vs cutoff."""
    if features.shape[0] == 0:
        return "insufficient"
    idx = OVERNIGHT_FEATURE_NAMES.index("trend_ratio")
    # features are z-scored per session; trend_ratio mean is always ~0.
    # Use abs_delta_pct and trend_ratio raw means before normalization is
    # unavailable in cache.  Instead use relative bar count above session mean:
    # a bar with z-score trend_ratio > 0 is above-average for that session.
    frac_above = float((features[:, idx] > 0).mean())
    return "directional" if frac_above >= cutoff else "non_directional"


# ── Results formatting ────────────────────────────────────────────────────────

def format_results(
    train_dates:     list[date],
    test_dates:      list[date],
    test_features:   list[np.ndarray],
    hmm:             DirectionalHMM,
    threshold_cutoff: float,
) -> str:
    lines = []

    lines.append("=" * 65)
    lines.append("2-STATE DIRECTIONAL OVERNIGHT HMM")
    lines.append("=" * 65)
    lines.append(f"  Training : {TRAIN_START} to {TRAIN_END}  ({len(train_dates)} days)")
    lines.append(f"  Test     : {TEST_START}  to {TEST_END}  ({len(test_dates)} days)")
    lines.append(f"  Features : {', '.join(OVERNIGHT_FEATURE_NAMES)}")
    lines.append("")

    # ── Emission means ────────────────────────────────────────────────────────
    lines.append("-- HMM State Emission Means (z-score space) --------------------")
    reverse = {v: k for k, v in hmm.state_map.items()}
    header  = f"  {'Feature':<25}  {'directional':>14}  {'non_directional':>16}"
    lines.append(header)
    lines.append("  " + "-" * 57)
    for fi, fname in enumerate(OVERNIGHT_FEATURE_NAMES):
        d_state  = reverse.get("directional")
        nd_state = reverse.get("non_directional")
        d_mean   = f"{hmm.model.means_[d_state,  fi]:>+12.4f}" if d_state  is not None else "n/a"
        nd_mean  = f"{hmm.model.means_[nd_state, fi]:>+14.4f}" if nd_state is not None else "n/a"
        lines.append(f"  {fname:<25}  {d_mean}  {nd_mean}")
    lines.append("")

    # ── Per-date test results ─────────────────────────────────────────────────
    lines.append("-- Per-Date Test Classifications --------------------------------")
    lines.append(
        f"  {'Date':<12}  {'Bars':>5}  "
        f"{'HMM Label':<16}  {'Conf':>5}  "
        f"{'dir%':>5}  {'ndir%':>6}  "
        f"{'Threshold':<16}"
    )
    lines.append("  " + "-" * 68)

    hmm_counts   = {"directional": 0, "non_directional": 0, "ambiguous": 0}
    thr_counts   = {"directional": 0, "non_directional": 0}

    for d, feat in zip(test_dates, test_features):
        label, conf = hmm.classify(feat)
        proba       = hmm.predict_proba_session(feat)
        thr_label   = threshold_classify(feat, threshold_cutoff)
        hmm_counts[label]   = hmm_counts.get(label, 0) + 1
        thr_counts[thr_label] = thr_counts.get(thr_label, 0) + 1
        lines.append(
            f"  {str(d):<12}  {feat.shape[0]:>5,}  "
            f"{label:<16}  {conf:>4.0%}  "
            f"{proba.get('directional', 0):>4.0%}  "
            f"{proba.get('non_directional', 0):>5.0%}  "
            f"{thr_label:<16}"
        )
    lines.append("")

    # ── Summary counts ────────────────────────────────────────────────────────
    n = len(test_dates)
    lines.append("-- Summary ------------------------------------------------------")
    lines.append(f"  HMM classifications ({n} test days):")
    for lbl, cnt in hmm_counts.items():
        lines.append(f"    {lbl:<18} : {cnt:>3}  ({cnt/n:.0%})")
    lines.append(f"  Threshold classifications (cutoff={threshold_cutoff}):")
    for lbl, cnt in thr_counts.items():
        lines.append(f"    {lbl:<18} : {cnt:>3}  ({cnt/n:.0%})")
    lines.append("")

    # ── Threshold cutoff sweep ────────────────────────────────────────────────
    lines.append("-- Threshold Cutoff Sweep (fraction of bars above session mean) -")
    lines.append(f"  {'Cutoff':>7}  {'Directional':>12}  {'Non-directional':>16}")
    for cutoff in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        d_cnt  = sum(1 for f in test_features if threshold_classify(f, cutoff) == "directional")
        nd_cnt = len(test_features) - d_cnt
        lines.append(f"  {cutoff:>7.2f}  {d_cnt:>12,}  {nd_cnt:>16,}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Collecting training features...")
    train_dates_all = trading_dates(TRAIN_START, TRAIN_END)
    train_features, train_valid = collect(train_dates_all, "TRAIN")
    print(f"  {len(train_valid)} valid training sessions.")

    print("\nCollecting test features...")
    test_dates_all = trading_dates(TEST_START, TEST_END)
    test_features, test_valid = collect(test_dates_all, "TEST")
    print(f"  {len(test_valid)} valid test sessions.")

    print("\nTraining 2-state DirectionalHMM...")
    hmm = DirectionalHMM(n_iter=200, random_state=42)
    hmm.fit(train_features, min_bars=MIN_BARS)
    hmm.align_states()
    print(f"  State map: {hmm.state_map}")

    output = format_results(
        train_valid, test_valid, test_features, hmm, TREND_RATIO_CUTOFF
    )
    print("\n" + output)

    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print(f"\nResults written to {OUTPUT_FILE}")

    joblib.dump(hmm, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

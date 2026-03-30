"""
RTH session-level HMM study.

Research question:
    Does classifying the RTH session (09:30-11:00 ET) as DIRECTIONAL vs
    NON_DIRECTIONAL based on a single session-level feature vector produce
    a reliable signal, and does overnight regime predict it?

Architecture
------------
One shape-(8,) feature vector per RTH session.
  - Computed by compute_rth_session_features() from 1-minute bars.
  - Normalized cross-session: one StandardScaler fitted on all training
    session vectors stacked (n_train_sessions, 8), applied to train and test.
  - HMM input: (n_sessions, 8) with NO lengths array.

Overnight cross-reference
-------------------------
Loads output/directional_overnight_hmm.pkl and overnight feature caches from
output/directional_cache/ to produce a contingency table
P(RTH label | overnight label) and chi-squared test of independence.

Cache
-----
output/rth_session_cache/{date}_rth_session.npy  — raw (unscaled) shape-(8,)
Companion .meta.json records version, normalization, feature names, bar type.
Stale or missing metadata forces regeneration.

Results written to output/rth_session_hmm_study.txt
"""

import gc
import json
import joblib
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize_enriched
from nqbt.analysis.rth_session_features import (
    compute_rth_session_features,
    RTH_SESSION_FEATURE_NAMES,
)
from nqbt.analysis.rth_session_hmm import RTHSessionHMM, _DIRECTIONAL_EXAMPLES

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START = "2025-03-17"
TRAIN_END   = "2025-11-30"
TEST_START  = "2025-12-01"
TEST_END    = "2026-03-15"

OUTPUT_FILE       = Path("output/rth_session_hmm_study.txt")
CACHE_DIR         = Path("output/rth_session_cache")
MODEL_PATH        = Path("output/rth_session_hmm.pkl")
SCALER_PATH       = Path("output/rth_session_scaler.pkl")
ON_MODEL_PATH     = Path("output/directional_overnight_hmm.pkl")
ON_CACHE_DIR      = Path("output/directional_cache")

ET       = "America/New_York"
MIN_BARS = 5   # minimum 1-minute bars for a session to be included

# Diagnostic dates — checked before any training
DIAGNOSTIC_DATES = ["2025-12-08", "2026-02-03", "2026-02-18"]
THRESHOLD_DATE   = "2025-12-08"
THRESHOLD_FIELD  = "trend_consistency"
THRESHOLD_MIN    = 0.55   # stop if Dec 8 trend_consistency < this
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FILE.parent.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── Cache versioning ──────────────────────────────────────────────────────────
CACHE_VERSION = 1
_CACHE_META = {
    "version":       CACHE_VERSION,
    "normalization": "raw",
    "features":      RTH_SESSION_FEATURE_NAMES,
    "bar_type":      "minute_1",
}


def _npy_path(d: date) -> Path:
    return CACHE_DIR / f"{d}_rth_session.npy"


def _meta_path(d: date) -> Path:
    return CACHE_DIR / f"{d}_rth_session.meta.json"


def _cache_valid(d: date) -> bool:
    if not _npy_path(d).exists() or not _meta_path(d).exists():
        return False
    try:
        meta = json.loads(_meta_path(d).read_text())
    except Exception:
        return False
    return (
        meta.get("version")       == _CACHE_META["version"]
        and meta.get("normalization") == _CACHE_META["normalization"]
        and meta.get("features")      == _CACHE_META["features"]
        and meta.get("bar_type")      == _CACHE_META["bar_type"]
    )


def _save_cache(d: date, vec: np.ndarray) -> None:
    np.save(_npy_path(d), vec)
    _meta_path(d).write_text(json.dumps(_CACHE_META))


# ── Feature loading ───────────────────────────────────────────────────────────

def trading_dates(start: str, end: str) -> list[date]:
    return [d.date() for d in pd.bdate_range(start=start, end=end)]


def load_session_vector(trading_date: date) -> np.ndarray | None:
    """
    Return raw (unscaled) shape-(8,) session vector for one RTH session.
    Loads from cache if valid; otherwise fetches ticks, computes, and caches.
    Returns None if data is insufficient.
    """
    if _cache_valid(trading_date):
        return np.load(_npy_path(trading_date))

    prev_day = trading_date - timedelta(days=1)
    try:
        raw      = fetch_and_load(str(prev_day), str(trading_date + timedelta(days=1)))
        enriched = normalize_enriched(raw)
        del raw
        gc.collect()
    except Exception as e:
        print(f"  ERROR loading {trading_date}: {e}")
        return None

    start = pd.Timestamp(f"{trading_date} 09:30:00", tz=ET)
    end   = pd.Timestamp(f"{trading_date} 11:00:00", tz=ET)
    ticks = enriched[(enriched.index >= start) & (enriched.index <= end)]
    del enriched
    gc.collect()

    if len(ticks) < 10:
        return None

    # Quick bar count check before feature computation
    from nqbt.analysis.minute_bars import build_minute_bars
    bars = build_minute_bars(ticks)
    if len(bars) < MIN_BARS:
        return None

    vec = compute_rth_session_features(ticks)
    _save_cache(trading_date, vec)
    return vec


def collect(dates: list[date], label: str) -> tuple[list[np.ndarray], list[date]]:
    """Collect raw session vectors and valid dates."""
    vectors, valid = [], []
    for i, d in enumerate(dates):
        print(f"  [{label}] {d}  ({i+1}/{len(dates)})")
        vec = load_session_vector(d)
        if vec is not None:
            vectors.append(vec)
            valid.append(d)
    return vectors, valid


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_diagnostic(raw_vectors: dict[str, np.ndarray]) -> None:
    print()
    print("=" * 65)
    print("  PRE-TRAINING DIAGNOSTIC — raw unscaled session vectors")
    print("=" * 65)
    header = f"  {'Feature':<22}" + "".join(f"  {d:>12}" for d in DIAGNOSTIC_DATES)
    print(header)
    print(f"  {'-' * (22 + 14 * len(DIAGNOSTIC_DATES))}")
    for fi, fname in enumerate(RTH_SESSION_FEATURE_NAMES):
        row = f"  {fname:<22}"
        for d in DIAGNOSTIC_DATES:
            if d in raw_vectors:
                row += f"  {raw_vectors[d][fi]:>+12.4f}"
            else:
                row += f"  {'MISSING':>12}"
        print(row)
    print()
    print("  Expected on clean trend days:")
    print("    trend_consistency   > 0.62")
    print("    max_excursion_ratio > 0.50")
    print("    realized_vol        lower than chop days")
    print()


# ── Cross-session scaler ──────────────────────────────────────────────────────

def fit_scaler(raw_vectors: list[np.ndarray]) -> StandardScaler:
    X = np.vstack([v.reshape(1, -1) for v in raw_vectors])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return StandardScaler().fit(X)


def apply_scaler(scaler: StandardScaler, raw_vectors: list[np.ndarray]) -> list[np.ndarray]:
    return [
        np.nan_to_num(
            scaler.transform(v.reshape(1, -1))[0],
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        for v in raw_vectors
    ]


def scale_single(scaler: StandardScaler, vec: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        scaler.transform(vec.reshape(1, -1))[0],
        nan=0.0, posinf=0.0, neginf=0.0,
    )


# ── Overnight cross-reference ─────────────────────────────────────────────────

def load_overnight_label(trading_date: date, on_hmm) -> str | None:
    """Load overnight feature cache and return the HMM label, or None."""
    p = ON_CACHE_DIR / f"{trading_date}_overnight.npy"
    if not p.exists():
        return None
    feat = np.load(p)
    if feat.ndim != 2 or feat.shape[0] < 5:
        return None
    label, _ = on_hmm.classify(feat)
    return label


# ── Results formatting ────────────────────────────────────────────────────────

def format_results(
    test_dates:    list[date],
    test_scaled:   list[np.ndarray],
    hmm:           RTHSessionHMM,
    scaler:        StandardScaler,
    on_hmm,
) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append("RTH SESSION-LEVEL HMM STUDY")
    lines.append("=" * 68)
    lines.append(f"  Training : {TRAIN_START} to {TRAIN_END}")
    lines.append(f"  Test     : {TEST_START} to {TEST_END}")
    lines.append(f"  Features : {', '.join(RTH_SESSION_FEATURE_NAMES)}")
    lines.append(f"  Input    : one shape-(8,) vector per session (no bar sequence)")
    lines.append(f"  Norm     : cross-session StandardScaler fitted on training data")
    lines.append("")

    # ── Scaler stats ──────────────────────────────────────────────────────────
    lines.append("-- Cross-Session Scaler (training data) -------------------------")
    lines.append(f"  {'Feature':<22}  {'mean':>10}  {'std':>10}")
    lines.append(f"  {'-'*46}")
    for fi, fname in enumerate(RTH_SESSION_FEATURE_NAMES):
        lines.append(
            f"  {fname:<22}  {scaler.mean_[fi]:>+10.4f}  {scaler.scale_[fi]:>10.4f}"
        )
    lines.append("")

    # ── Emission means ────────────────────────────────────────────────────────
    lines.append("-- HMM State Emission Means (scaled space) ----------------------")
    reverse = {v: k for k, v in hmm.state_map.items()}
    header  = f"  {'Feature':<22}  {'directional':>14}  {'non_directional':>16}"
    lines.append(header)
    lines.append(f"  {'-'*56}")
    for fi, fname in enumerate(RTH_SESSION_FEATURE_NAMES):
        d_state  = reverse.get("directional")
        nd_state = reverse.get("non_directional")
        d_m  = f"{hmm.model.means_[d_state,  fi]:>+12.4f}" if d_state  is not None else "n/a"
        nd_m = f"{hmm.model.means_[nd_state, fi]:>+14.4f}" if nd_state is not None else "n/a"
        lines.append(f"  {fname:<22}  {d_m}  {nd_m}")
    lines.append("")

    # ── Per-date test classifications ─────────────────────────────────────────
    lines.append("-- Per-Date Test Classifications --------------------------------")
    lines.append(
        f"  {'Date':<12}  {'RTH Label':<16}  {'Conf':>5}  "
        f"{'dir%':>5}  {'ndir%':>6}  {'ON Label':<16}"
    )
    lines.append(f"  {'-'*70}")

    records = []
    rth_counts = {"directional": 0, "non_directional": 0, "ambiguous": 0}

    for d, vec in zip(test_dates, test_scaled):
        label, conf = hmm.classify(vec)
        _, d_conf   = hmm.predict(vec)
        proba_dict  = {}
        clean = np.nan_to_num(vec.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        raw_proba = hmm.model.predict_proba(clean)[0]
        for state, name in hmm.state_map.items():
            proba_dict[name] = float(raw_proba[state])

        on_label = load_overnight_label(d, on_hmm) if on_hmm is not None else None
        on_str   = on_label if on_label else "n/a"

        rth_counts[label] = rth_counts.get(label, 0) + 1
        records.append({"date": d, "rth": label, "overnight": on_label})

        lines.append(
            f"  {str(d):<12}  {label:<16}  {conf:>4.0%}  "
            f"{proba_dict.get('directional', 0):>4.0%}  "
            f"{proba_dict.get('non_directional', 0):>5.0%}  "
            f"{on_str:<16}"
        )
    lines.append("")

    # ── Summary counts ────────────────────────────────────────────────────────
    n = len(test_dates)
    lines.append("-- Summary ------------------------------------------------------")
    lines.append(f"  RTH classifications ({n} test days):")
    for lbl, cnt in rth_counts.items():
        lines.append(f"    {lbl:<18} : {cnt:>3}  ({cnt/n:.0%})")
    lines.append("")

    # ── Overnight cross-reference (only if overnight model loaded) ────────────
    pairs = pd.DataFrame([r for r in records if r["overnight"] is not None])

    if on_hmm is None:
        lines.append("-- Overnight Cross-Reference: SKIPPED (model not found) -----")
    elif pairs.empty or len(pairs) < 3:
        lines.append("-- Overnight Cross-Reference: insufficient paired data -------")
    else:
        raw_ct  = pd.crosstab(pairs["overnight"], pairs["rth"])
        norm_ct = pd.crosstab(pairs["overnight"], pairs["rth"], normalize="index")
        chi2, p_value, dof, _ = stats.chi2_contingency(raw_ct.values)

        lines.append("-- Contingency Table - P(RTH label | overnight label) --------")
        lines.append(norm_ct.to_string())
        lines.append("")
        lines.append("-- Raw Counts ------------------------------------------------")
        lines.append(raw_ct.to_string())
        lines.append("")
        lines.append("-- Chi-Squared Test of Independence -------------------------")
        lines.append(f"  chi2    : {round(chi2, 4)}")
        lines.append(f"  p-value : {round(p_value, 6)}")
        lines.append(f"  dof     : {dof}")
        sig = "SIGNIFICANT (p < 0.05)" if p_value < 0.05 else "NOT significant (p >= 0.05)"
        lines.append(f"  Result  : {sig}")
        lines.append("")

        lines.append("-- Per-State Match Rates ------------------------------------")
        for on_regime in pairs["overnight"].unique():
            subset = pairs[pairs["overnight"] == on_regime]
            lines.append(f"  Overnight = {on_regime}:")
            for rth_r, prob in subset["rth"].value_counts(normalize=True).items():
                lines.append(f"    RTH {rth_r:<16}: {prob:.1%}")
        lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Step 1: Collect diagnostic dates first (raw vectors) ──────────────────
    print("Loading diagnostic sessions...")
    diag_raw: dict[str, np.ndarray] = {}
    for ds in DIAGNOSTIC_DATES:
        d   = pd.Timestamp(ds).date()
        vec = load_session_vector(d)
        if vec is not None:
            diag_raw[ds] = vec
        else:
            print(f"  WARNING: could not compute features for {ds}")

    print_diagnostic(diag_raw)

    # ── Threshold check ────────────────────────────────────────────────────────
    if THRESHOLD_DATE in diag_raw:
        fi    = RTH_SESSION_FEATURE_NAMES.index(THRESHOLD_FIELD)
        value = float(diag_raw[THRESHOLD_DATE][fi])
        if value < THRESHOLD_MIN:
            print(f"STOP: {THRESHOLD_DATE} {THRESHOLD_FIELD} = {value:.4f} < {THRESHOLD_MIN}")
            print("Feature computation problem detected. Debug before retraining.")
            raise SystemExit(1)
        else:
            print(f"Threshold OK: {THRESHOLD_DATE} {THRESHOLD_FIELD} = {value:.4f} >= {THRESHOLD_MIN}")
    else:
        print(f"WARNING: threshold date {THRESHOLD_DATE} not available, skipping check.")
    print()

    # ── Step 2: Collect training and test features ────────────────────────────
    print("Collecting training features...")
    train_raw, train_valid = collect(trading_dates(TRAIN_START, TRAIN_END), "TRAIN")
    print(f"  {len(train_valid)} valid training sessions.\n")

    print("Collecting test features...")
    test_raw, test_valid = collect(trading_dates(TEST_START, TEST_END), "TEST")
    print(f"  {len(test_valid)} valid test sessions.\n")

    # ── Step 3: Fit cross-session scaler on training data only ────────────────
    print("Fitting cross-session scaler on training data...")
    scaler     = fit_scaler(train_raw)
    train_sc   = apply_scaler(scaler, train_raw)
    test_sc    = apply_scaler(scaler, test_raw)
    print(f"  Scaler fitted on {len(train_valid)} training sessions.\n")

    # ── Step 4: Build align_states example dict (scaled) ─────────────────────
    all_dates  = train_valid + test_valid
    all_raw    = train_raw  + test_raw
    raw_lookup = {str(d): v for d, v in zip(all_dates, all_raw)}

    example_scaled: dict[str, np.ndarray] = {}
    for ds in _DIRECTIONAL_EXAMPLES:
        if ds in raw_lookup:
            example_scaled[ds] = scale_single(scaler, raw_lookup[ds])

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    print("Training RTHSessionHMM...")
    hmm = RTHSessionHMM(n_iter=200, random_state=42)
    hmm.fit(train_sc)
    print()
    hmm.align_states(example_scaled)
    print()

    # ── Step 5b: Convergence check — stop before output if not converged ────────
    if not hmm.model.monitor_.converged:
        hist = hmm.model.monitor_.history
        delta = hist[-1] - hist[-2] if len(hist) >= 2 else float("nan")
        print("WARNING: model did not converge")
        print(f"  Final delta: {delta:.6f}")
        print("  Unreliable classifications — stopping. Try increasing n_iter.")
        raise SystemExit(1)
    print("Convergence: OK\n")

    # ── Step 6: Load overnight model for cross-reference ──────────────────────
    on_hmm = None
    if ON_MODEL_PATH.exists():
        print(f"Loading overnight model from {ON_MODEL_PATH}...")
        on_hmm = joblib.load(ON_MODEL_PATH)
    else:
        print(f"WARNING: {ON_MODEL_PATH} not found — overnight cross-reference skipped.")
    print()

    # ── Step 7: Format and write results ─────────────────────────────────────
    print("Formatting results...")
    output = format_results(test_valid, test_sc, hmm, scaler, on_hmm)
    print("\n" + output)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print(f"Results written to {OUTPUT_FILE}")

    joblib.dump(hmm,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved  to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

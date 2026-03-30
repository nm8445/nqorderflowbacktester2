"""
HMM Regime Study: Does overnight session predict RTH regime?

Dual bar type architecture:
  - 1-minute bars for all regime classification features (this file)
  - 40-range bars for execution-level orderflow features (unchanged elsewhere)

Training period : 2025-03-17 to 2025-11-30
Test period     : 2025-12-01 to 2026-03-15  (held out entirely during training)

Normalization:
  Overnight : per-session z-score via StandardScaler.fit_transform()
              (8 features, ~900 bars per session, variance is stable)
  RTH       : cross-session z-score — one StandardScaler fitted on ALL
              training sessions stacked, then applied to each session.
              Per-session z-scoring fails on 90-bar sessions because slow
              features (obi_avg, vwap_deviation, delta_acceleration) have
              near-zero within-session variance, collapsing to zero.

The fitted RTH scaler is saved to output/rth_scaler.pkl for inference.

Feature caches in output/minute_cache/:
  {date}_overnight.npy  — 8-col, per-session normalized
  {date}_rth.npy        — 13-col, RAW (un-normalized); scaler applied in main

Results written to output/hmm_study.txt
"""

import gc
import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize_enriched
from nqbt.analysis.minute_bars import build_minute_bars
from nqbt.analysis.vwap import compute_vwap
from nqbt.analysis.features import compute_session_features, FEATURE_NAMES
from nqbt.analysis.overnight_features import (
    compute_overnight_features,
    OVERNIGHT_FEATURE_NAMES,
)
from nqbt.analysis.hmm_regime import run_overnight_rth_study

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START = "2025-03-17"
TRAIN_END   = "2025-11-30"
TEST_START  = "2025-12-01"
TEST_END    = "2026-03-15"

OUTPUT_FILE  = Path("output/hmm_study.txt")
MINUTE_CACHE = Path("output/minute_cache")
SCALER_PATH  = Path("output/rth_scaler.pkl")
MIN_BARS     = 5

ET = "America/New_York"
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FILE.parent.mkdir(exist_ok=True)
MINUTE_CACHE.mkdir(exist_ok=True)

_N_ON  = len(OVERNIGHT_FEATURE_NAMES)   # 8  — per-session normalized
_N_RTH = len(FEATURE_NAMES)             # 13 — raw in cache, cross-session scaled in memory

# Cache versioning — bump when normalization method or feature schema changes.
# Any cache file missing metadata or with a mismatched field is silently
# regenerated rather than silently used with wrong normalization.
CACHE_VERSION = 1
_EXPECTED_NORM = {
    "overnight": "per_session",   # compute_overnight_features(normalize=True)
    "rth":       "raw",           # compute_session_features(normalize=False); cross-session scaler in memory
}


def _cache_path(trading_date: date, session: str) -> Path:
    return MINUTE_CACHE / f"{trading_date}_{session}.npy"


def _meta_path(trading_date: date, session: str) -> Path:
    return MINUTE_CACHE / f"{trading_date}_{session}.meta.json"


def _load_cache(trading_date: date, session: str) -> np.ndarray | None:
    p = _cache_path(trading_date, session)
    if not p.exists():
        return None
    mp = _meta_path(trading_date, session)
    if not mp.exists():
        return None   # pre-versioning cache — force regeneration
    try:
        meta = json.loads(mp.read_text())
    except Exception:
        return None
    expected_cols = _N_ON if session == "overnight" else _N_RTH
    if (
        meta.get("version")       != CACHE_VERSION
        or meta.get("normalization") != _EXPECTED_NORM[session]
        or meta.get("n_features")    != expected_cols
    ):
        return None   # stale — normalization or schema mismatch
    arr = np.load(p)
    if arr.ndim == 2 and arr.shape[1] != expected_cols:
        return None
    return arr


def _save_cache(trading_date: date, session: str, features: np.ndarray) -> None:
    np.save(_cache_path(trading_date, session), features)
    meta = {
        "version":       CACHE_VERSION,
        "normalization": _EXPECTED_NORM[session],
        "n_features":    features.shape[1] if features.ndim == 2 else 0,
    }
    _meta_path(trading_date, session).write_text(json.dumps(meta))


def trading_dates(start: str, end: str) -> list[date]:
    return [d.date() for d in pd.bdate_range(start=start, end=end)]


def extract_sessions(enriched: pd.DataFrame, trading_date: date):
    prev_day = trading_date - timedelta(days=1)
    on_start = pd.Timestamp(f"{prev_day} 18:00:00", tz=ET)
    on_end   = pd.Timestamp(f"{trading_date} 09:30:00", tz=ET)
    rth_end  = pd.Timestamp(f"{trading_date} 11:00:00", tz=ET)
    overnight = enriched[(enriched.index >= on_start) & (enriched.index < on_end)]
    rth       = enriched[(enriched.index >= on_end)   & (enriched.index <= rth_end)]
    return overnight, rth


def process_day(trading_date: date) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      on_feat  : (n, 8)  — overnight, per-session normalized
      rth_feat : (n, 13) — RTH, RAW (cross-session scaler applied in main)
    """
    cached_on  = _load_cache(trading_date, "overnight")
    cached_rth = _load_cache(trading_date, "rth")
    if cached_on is not None and cached_rth is not None:
        return cached_on, cached_rth

    prev_day = trading_date - timedelta(days=1)
    try:
        raw      = fetch_and_load(str(prev_day), str(trading_date + timedelta(days=1)))
        enriched = normalize_enriched(raw)
        del raw
        gc.collect()
    except Exception as e:
        print(f"  ERROR loading {trading_date}: {e}")
        return np.empty((0, _N_ON)), np.empty((0, _N_RTH))

    overnight_ticks, rth_ticks = extract_sessions(enriched, trading_date)
    del enriched
    gc.collect()

    # ── Overnight (8 features, per-session normalized) ────────────────────────
    if len(overnight_ticks) < 10:
        on_feat = np.empty((0, _N_ON))
    else:
        on_bars = build_minute_bars(overnight_ticks)
        if len(on_bars) < MIN_BARS:
            on_feat = np.empty((0, _N_ON))
        else:
            on_feat = compute_overnight_features(
                overnight_ticks, on_bars, min_ticks=1, normalize=True
            )

    # ── RTH (13 features, RAW — cross-session scaler applied later) ───────────
    if len(rth_ticks) < 10:
        rth_feat = np.empty((0, _N_RTH))
    else:
        rth_bars = build_minute_bars(rth_ticks)
        if len(rth_bars) < MIN_BARS:
            rth_feat = np.empty((0, _N_RTH))
        else:
            vwap_df  = compute_vwap(rth_ticks)
            rth_feat = compute_session_features(
                rth_ticks, rth_bars, vwap_df,
                min_ticks=1, normalize=False,        # raw — scaler applied in main
                include_session_features=True,
            )

    _save_cache(trading_date, "overnight", on_feat)
    _save_cache(trading_date, "rth",       rth_feat)
    return on_feat, rth_feat


def collect_features(dates: list[date], label: str) -> tuple[list, list, list]:
    """Returns (overnight_normalized, rth_raw, valid_dates)."""
    overnight_list, rth_list, valid_dates = [], [], []
    for i, d in enumerate(dates):
        print(f"  [{label}] {d}  ({i+1}/{len(dates)})")
        on_feat, rth_feat = process_day(d)
        if on_feat.shape[0] >= MIN_BARS and rth_feat.shape[0] >= MIN_BARS:
            overnight_list.append(on_feat)
            rth_list.append(rth_feat)
            valid_dates.append(d)
    return overnight_list, rth_list, valid_dates


def fit_rth_scaler(raw_sessions: list[np.ndarray]) -> StandardScaler:
    """Fit one StandardScaler on all training RTH sessions stacked."""
    valid = [X for X in raw_sessions if X.shape[0] >= MIN_BARS]
    X_all = np.vstack(valid)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    scaler.fit(X_all)
    return scaler


def apply_rth_scaler(
    scaler: StandardScaler, raw_sessions: list[np.ndarray]
) -> list[np.ndarray]:
    """Apply the fitted RTH scaler to each session."""
    result = []
    for X in raw_sessions:
        if X.shape[0] == 0:
            result.append(X)
            continue
        X_clean  = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X_clean)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        result.append(X_scaled)
    return result


def print_excursion_distributions(rth_sessions: list[np.ndarray]) -> None:
    """
    Print distribution statistics for the 3 session-level excursion features
    across all RTH training sessions (after cross-session scaling).
    Each session contributes one value per feature (constant across rows).
    """
    names = ["max_excursion_ratio", "trend_consistency", "drawdown_ratio"]
    values = {n: [] for n in names}
    for feat in rth_sessions:
        if feat.shape[0] > 0 and feat.shape[1] == _N_RTH:
            for i, name in enumerate(names):
                values[name].append(float(feat[0, 10 + i]))

    print("\n-- RTH Excursion Feature Distributions (cross-session scaled) ----")
    print(f"  Sessions: {len(next(iter(values.values())))}")
    print(f"  {'Feature':<22}  {'mean':>8}  {'std':>8}  {'p25':>8}  "
          f"{'p50':>8}  {'p75':>8}  {'p90':>8}")
    print(f"  {'-'*68}")
    for name in names:
        arr = np.array(values[name])
        if len(arr) == 0:
            continue
        print(
            f"  {name:<22}"
            f"  {np.mean(arr):>8.4f}"
            f"  {np.std(arr):>8.4f}"
            f"  {np.percentile(arr, 25):>8.4f}"
            f"  {np.percentile(arr, 50):>8.4f}"
            f"  {np.percentile(arr, 75):>8.4f}"
            f"  {np.percentile(arr, 90):>8.4f}"
        )
    print()


def print_normalization_diagnostic(
    raw_sessions:    list[np.ndarray],
    scaled_sessions: list[np.ndarray],
    session_dates:   list[date],
    n_sessions:      int = 5,
) -> None:
    """
    For the first n_sessions, print mean and std of all 13 RTH features
    after cross-session normalization. Confirms features are no longer
    collapsing to zero (as they did under per-session z-scoring).
    """
    print("-- RTH Cross-Session Normalization Diagnostic (first 5 sessions) -")
    print("   (per-session z-scoring produced mean=0 std=0 for slow features;")
    print("    cross-session scaling should show non-zero means and stds)")
    print()

    for idx in range(min(n_sessions, len(scaled_sessions))):
        d    = session_dates[idx]
        Xraw = raw_sessions[idx]
        X    = scaled_sessions[idx]
        print(f"  {d}  ({X.shape[0]} bars)")
        print(f"  {'Feature':<22}  {'raw_mean':>10}  {'scaled_mean':>12}  {'scaled_std':>10}")
        print(f"  {'-'*60}")
        for fi, fname in enumerate(FEATURE_NAMES):
            raw_mean    = float(np.mean(Xraw[:, fi]))
            scaled_mean = float(np.mean(X[:, fi]))
            scaled_std  = float(np.std(X[:, fi]))
            print(
                f"  {fname:<22}"
                f"  {raw_mean:>+10.4f}"
                f"  {scaled_mean:>+12.4f}"
                f"  {scaled_std:>10.4f}"
            )
        print()


def format_results(results: dict) -> str:
    lines = []

    lines.append("=" * 65)
    lines.append("HMM REGIME STUDY - Overnight -> RTH Predictability")
    lines.append("1-minute bars | overnight: per-session z-score | RTH: cross-session z-score")
    lines.append("=" * 65)
    lines.append(f"  Training : {TRAIN_START} to {TRAIN_END}")
    lines.append(f"  Test     : {TEST_START} to {TEST_END}")
    lines.append(f"  Test days used       : {results['n_test_days']}")
    lines.append(f"  Overnight features   : {', '.join(OVERNIGHT_FEATURE_NAMES)}")
    lines.append(f"  RTH features         : {', '.join(FEATURE_NAMES)}")
    lines.append("")

    lines.append("-- Chi-Squared Test of Independence ----------------------")
    lines.append(f"  chi2    : {results['chi2']}")
    lines.append(f"  p-value : {results['p_value']}")
    lines.append(f"  dof     : {results['dof']}")
    sig = "SIGNIFICANT (p < 0.05)" if results['p_value'] < 0.05 else "NOT significant (p >= 0.05)"
    lines.append(f"  Result  : {sig}")
    lines.append("")

    lines.append("-- Contingency Table - P(RTH regime | overnight regime) --")
    if results["contingency"] is not None:
        lines.append(results["contingency"].to_string())
    lines.append("")

    lines.append("-- Raw Counts --------------------------------------------")
    if results["raw_contingency"] is not None:
        lines.append(results["raw_contingency"].to_string())
    lines.append("")

    lines.append("-- Per-State Match Rates ---------------------------------")
    for on_regime, rth_dist in results["per_state_match"].items():
        lines.append(f"  Overnight = {on_regime}:")
        for rth_regime, prob in sorted(rth_dist.items(), key=lambda x: -x[1]):
            lines.append(f"    RTH {rth_regime:<10}: {prob:.1%}")
    lines.append("")

    lines.append("-- Overnight HMM Emission Means (2-state, 8 features) ----")
    hmm_on     = results["overnight_hmm"]
    on_regimes = list(hmm_on.state_map.values())
    on_reverse = {v: k for k, v in hmm_on.state_map.items()}
    on_header  = f"  {'Feature':<26}  " + "  ".join(f"{r:<16}" for r in on_regimes)
    lines.append(on_header)
    for fi, fname in enumerate(OVERNIGHT_FEATURE_NAMES):
        row = f"  {fname:<26}"
        for regime in on_regimes:
            state = on_reverse.get(regime)
            if state is not None:
                row += f"  {hmm_on.model.means_[state, fi]:>+14.4f}"
        lines.append(row)
    lines.append("")

    lines.append("-- RTH HMM Emission Means (3-state, 13 features) ---------")
    hmm_rth     = results["rth_hmm"]
    rth_regimes = ["trending", "ranging", "chop"]
    rth_reverse = {v: k for k, v in hmm_rth.state_map.items()}
    rth_header  = f"  {'Feature':<26}  " + "  ".join(f"{r:<10}" for r in rth_regimes)
    lines.append(rth_header)
    for fi, fname in enumerate(FEATURE_NAMES):
        row = f"  {fname:<26}"
        for regime in rth_regimes:
            state = rth_reverse.get(regime)
            if state is not None:
                row += f"  {hmm_rth.model.means_[state, fi]:>+10.4f}"
        lines.append(row)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import joblib

    print("Collecting training features...")
    train_dates = trading_dates(TRAIN_START, TRAIN_END)
    train_overnight, train_rth_raw, _ = collect_features(train_dates, "TRAIN")
    print(f"\nCollected {len(train_overnight)} valid training sessions.")

    print("\nFitting cross-session RTH scaler on training data...")
    rth_scaler = fit_rth_scaler(train_rth_raw)
    print(f"  Scaler fitted on {sum(X.shape[0] for X in train_rth_raw if X.shape[0] > 0):,} total bars")
    train_rth = apply_rth_scaler(rth_scaler, train_rth_raw)

    print_excursion_distributions(train_rth)

    print("Collecting test features...")
    test_dates_all = trading_dates(TEST_START, TEST_END)
    test_overnight, test_rth_raw, valid_test_dates = collect_features(test_dates_all, "TEST")
    print(f"Collected {len(test_overnight)} valid test sessions.")

    test_rth = apply_rth_scaler(rth_scaler, test_rth_raw)

    print_normalization_diagnostic(test_rth_raw, test_rth, valid_test_dates)

    print("Running study...")
    results = run_overnight_rth_study(
        train_overnight, train_rth,
        test_overnight,  test_rth,
        valid_test_dates,
        min_bars=MIN_BARS,
    )

    output = format_results(results)
    print("\n" + output)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print(f"\nResults written to {OUTPUT_FILE}")

    joblib.dump(results["overnight_hmm"], Path("output/overnight_hmm.pkl"))
    joblib.dump(results["rth_hmm"],       Path("output/rth_hmm.pkl"))
    joblib.dump(rth_scaler,               SCALER_PATH)
    print("Models saved to output/overnight_hmm.pkl, output/rth_hmm.pkl")
    print(f"RTH scaler saved to {SCALER_PATH}")

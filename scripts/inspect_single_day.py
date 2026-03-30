"""
Inspect HMM state emission means for a single date and session.

Edit TRADING_DATE and SESSION below, then run:
    python scripts/inspect_single_day.py
"""

import joblib
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize_enriched
from nqbt.analysis.range_bars import build_range_bars
from nqbt.analysis.vwap import compute_vwap
from nqbt.analysis.features import compute_session_features, FEATURE_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
TRADING_DATE = date(2025, 12, 8)
SESSION      = "rth"        # "overnight" or "rth"
ET           = "America/New_York"
FEATURE_CACHE = Path("output/feature_cache")
MODEL_OVERNIGHT = Path("output/overnight_hmm.pkl")
MODEL_RTH       = Path("output/rth_hmm.pkl")
# ─────────────────────────────────────────────────────────────────────────────

if SESSION == "rth":
    if not MODEL_RTH.exists():
        print("ERROR: output/rth_hmm.pkl not found. Run scripts/hmm_study.py first.")
        exit(1)
    hmm = joblib.load(MODEL_RTH)
    session_label = "RTH  09:30 to 11:00"
else:
    if not MODEL_OVERNIGHT.exists():
        print("ERROR: output/overnight_hmm.pkl not found. Run scripts/hmm_study.py first.")
        exit(1)
    hmm = joblib.load(MODEL_OVERNIGHT)
    session_label = "OVERNIGHT  18:00 prev day to 09:30"

reverse = {v: k for k, v in hmm.state_map.items()}

# ── Load features ─────────────────────────────────────────────────────────────
cache_path = FEATURE_CACHE / f"{TRADING_DATE}_{SESSION}.npy"
prev = TRADING_DATE - timedelta(days=1)

_cached = None
if cache_path.exists():
    _cached = np.load(cache_path)
    # Invalidate stale caches predating the 3 session-level excursion features
    if _cached.ndim == 2 and _cached.shape[1] < len(FEATURE_NAMES):
        _cached = None

if _cached is not None:
    feat = _cached
    print(f"Loaded from cache: {feat.shape[0]} bars")
else:
    print("Cache miss or stale (pre-13-feature) -- fetching raw data...")
    raw  = fetch_and_load(str(prev), str(TRADING_DATE + timedelta(days=1)))
    enr  = normalize_enriched(raw)

    if SESSION == "rth":
        start = pd.Timestamp(f"{TRADING_DATE} 09:30:00", tz=ET)
        end   = pd.Timestamp(f"{TRADING_DATE} 11:00:00", tz=ET)
    else:
        start = pd.Timestamp(f"{prev} 18:00:00", tz=ET)
        end   = pd.Timestamp(f"{TRADING_DATE} 09:30:00", tz=ET)

    ticks = enr[(enr.index >= start) & (enr.index <= end)]
    vwap  = compute_vwap(ticks)
    bars  = build_range_bars(ticks, range_ticks=40, ticks_per_level=5)
    feat  = compute_session_features(ticks, bars, vwap, normalize=True)
    print(f"Computed: {feat.shape[0]} bars")

# ── Predict ───────────────────────────────────────────────────────────────────
if feat.shape[0] < 5:
    print("Insufficient bars to classify.")
    exit(1)

regime = hmm.predict_session(feat)
proba  = hmm.predict_proba_session(feat)

# ── Output ────────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print(f"  {session_label}  {TRADING_DATE}".upper())
print("=" * 68)
print(f"  Predicted regime : {regime.upper()}")
print(
    f"  Bar distribution : "
    f"trending={proba['trending']:.1%}  "
    f"ranging={proba['ranging']:.1%}  "
    f"chop={proba['chop']:.1%}"
)
print(f"  Bars in session  : {feat.shape[0]}")
print()
print(f"  {'Feature':<22}  {'trending':>10}  {'ranging':>10}  {'chop':>10}")
print(f"  {'-' * 60}")
for fi, fname in enumerate(FEATURE_NAMES):
    row = f"  {fname:<22}"
    for rname in ["trending", "ranging", "chop"]:
        st     = reverse.get(rname)
        marker = " *" if rname == regime else "  "
        row   += f"  {hmm.model.means_[st, fi]:>+8.4f}{marker}"
    print(row)
print("  (* = predicted state)")

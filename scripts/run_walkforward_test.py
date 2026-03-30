"""
Walk-forward test of directional_absorption_setup_v1 on 2025-12-01 to 2026-03-15.

Uses training move_mean/move_std for sustained labeling (no refitting).
Rolling 20-session vol percentile seeds from the tail of training data.
Cluster_position and prior_absorption_count computed fresh on test signals.
"""

import gc
import sys
import json
import joblib
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize_enriched
from nqbt.analysis.range_bars import build_range_bars
from nqbt.analysis.vwap import compute_vwap
from nqbt.analysis.signal_labeler import label_absorption_signals

# ── Config ────────────────────────────────────────────────────────────────────
TEST_START  = "2025-12-01"
TEST_END    = "2026-03-15"

TRAIN_CSV       = PROJECT_ROOT / "output" / "absorption_signals_labeled.csv"
ON_MODEL_PATH   = PROJECT_ROOT / "output" / "directional_overnight_hmm.pkl"
ON_CACHE_DIR    = PROJECT_ROOT / "output" / "directional_cache"
TICK_CACHE_DIR  = PROJECT_ROOT / "output" / "tick_cache"
OUT_DIR         = PROJECT_ROOT / "output" / "walkforward"
OUT_TXT         = OUT_DIR / "directional_setup_walkforward.txt"

ET            = "America/New_York"
MIN_BARS      = 5
# Training-fitted normalization params (from absorption_signals_labeled.csv)
MOVE_MEAN     = -0.003491
MOVE_STD      =  0.194771
SUSTAINED_Z   = 1.5
SUSTAINED_MAE = 0.30

# Filter params from directional_setup_rules.json
ROLLING_VOL_WINDOW   = 20
ROLLING_VOL_PCT      = 60
TIME_START_MINS      = 30   # 09:30 ET = 30 mins after 09:00
TIME_END_MINS        = 60   # 10:00 ET
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
TICK_CACHE_DIR.mkdir(exist_ok=True)


def trading_dates(start: str, end: str) -> list[date]:
    return [d.date() for d in pd.bdate_range(start=start, end=end)]


def load_overnight_label(trading_date: date, on_hmm) -> str:
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
    if len(bars) < 2:
        return 1e-6
    closes = np.array([b.close for b in bars], dtype=float)
    log_returns = np.log(closes[1:] / closes[:-1])
    rv = float(np.sqrt(np.sum(log_returns ** 2)))
    return rv if rv > 1e-9 else 1e-6


def load_ticks(trading_date: date) -> pd.DataFrame | None:
    cache = TICK_CACHE_DIR / f"{trading_date}_ticks.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    prev_day = trading_date - timedelta(days=1)
    try:
        raw      = fetch_and_load(str(prev_day), str(trading_date + timedelta(days=1)))
        enriched = normalize_enriched(raw)
        del raw; gc.collect()
    except Exception as e:
        print(f"  ERROR loading {trading_date}: {e}")
        return None

    on_start = pd.Timestamp(f"{prev_day} 18:00:00",     tz=ET)
    rth_end  = pd.Timestamp(f"{trading_date} 11:00:00", tz=ET)
    window   = enriched[(enriched.index >= on_start) & (enriched.index <= rth_end)].copy()
    del enriched; gc.collect()
    window.to_parquet(cache)
    return window


def process_date(trading_date: date, on_hmm) -> pd.DataFrame:
    ticks = load_ticks(trading_date)
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

    rth_bars = build_range_bars(rth_ticks, range_ticks=40, ticks_per_level=5)
    if len(rth_bars) < MIN_BARS:
        return pd.DataFrame()

    vwap_df  = compute_vwap(rth_ticks)
    vwap_val = float(vwap_df["vwap"].iloc[-1]) if not vwap_df.empty else 0.0
    vwap_std = float(vwap_df["std"].iloc[-1])  if not vwap_df.empty else 1.0
    if vwap_std < 1e-9:
        vwap_std = 1.0

    rv = session_realized_vol(rth_bars)
    overnight_label = load_overnight_label(trading_date, on_hmm)

    result = label_absorption_signals(
        bars                 = rth_bars,
        overnight_label      = overnight_label,
        profile_ticks        = profile_ticks,
        vwap                 = vwap_val,
        vwap_std             = vwap_std,
        session_realized_vol = rv,
    )
    return result


def add_cluster_position(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate signal_diagnostics.signal_clustering logic."""
    df = df.copy()
    sig_et = pd.to_datetime(df["signal_time"], utc=True).dt.tz_convert(ET)
    df["_sig_et"] = sig_et
    df["_date_et"] = sig_et.dt.date
    positions = pd.Series("solo", index=df.index, dtype=object)

    for (d, side), grp in df.groupby(["_date_et", "absorption_side"]):
        grp_s = grp.sort_values("_sig_et")
        times = grp_s["_sig_et"].tolist()
        idxs  = grp_s.index.tolist()
        if len(times) < 2:
            continue
        clusters, cur = [], [0]
        for i in range(1, len(times)):
            if (times[i] - times[i-1]).total_seconds() <= 180:
                cur.append(i)
            else:
                clusters.append(cur); cur = [i]
        clusters.append(cur)
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            for pos, i in enumerate(cluster):
                positions[idxs[i]] = "first" if pos == 0 else ("second" if pos == 1 else "third_plus")

    df["cluster_position"] = positions
    return df.drop(columns=["_sig_et", "_date_et"])


def add_prior_absorption_count(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate signal_diagnostics.prior_absorption_count logic."""
    df = df.copy()
    sig_et = pd.to_datetime(df["signal_time"], utc=True).dt.tz_convert(ET)
    df["_sig_et"] = sig_et
    df["_date_et"] = sig_et.dt.date
    counts = pd.Series(0, index=df.index, dtype=int)

    for (d, side), grp in df.groupby(["_date_et", "absorption_side"]):
        grp_s = grp.sort_values("_sig_et")
        times = grp_s["_sig_et"].tolist()
        idxs  = grp_s.index.tolist()
        for i, (t, idx) in enumerate(zip(times, idxs)):
            window_start = t - pd.Timedelta(minutes=10)
            counts[idx] = sum(1 for prev_t in times[:i] if prev_t > window_start)

    df["prior_absorption_count"] = counts
    return df.drop(columns=["_sig_et", "_date_et"])


def add_rolling_vol_pct(df: pd.DataFrame, train_tail_vols: list[float],
                        train_tail_dates: list[date]) -> pd.DataFrame:
    """
    Compute rolling 20-session vol percentile for each signal.
    Seeds the window with the last N sessions from training data.
    """
    df = df.copy()
    all_dates = sorted(df["date"].unique())

    # Build combined session vol history: training tail + test sessions
    # For each test session, use median total_vol as the session representative
    session_vols: dict[date, list[float]] = {}
    for d, grp in df.groupby("date"):
        session_vols[d] = grp["total_vol"].tolist()

    # Ordered list of (date, [vols]) for rolling window
    history: list[tuple[date, list[float]]] = list(
        zip(train_tail_dates, [[v] * 1 for v in train_tail_vols])
    )
    # Replace with actual training vol lists (already aggregated as single representative)
    # Using per-signal vols: window = all signals from prior 20 sessions

    # Rebuild: training tail as flat vol pool per session
    train_pool: list[tuple[date, list[float]]] = list(zip(train_tail_dates,
                                                           [[v] for v in train_tail_vols]))

    result_pcts = pd.Series(np.nan, index=df.index)

    ordered_test_dates = sorted(all_dates)
    # Combined ordered history = training tail + test dates seen so far
    session_history: list[tuple[date, list[float]]] = train_pool.copy()

    for d in ordered_test_dates:
        # Window = prior 20 sessions (not including current day)
        window_sessions = session_history[-ROLLING_VOL_WINDOW:]
        window_vols = [v for _, vlist in window_sessions for v in vlist]

        if window_vols:
            mask = df["date"] == d
            for idx in df.index[mask]:
                sig_vol = df.at[idx, "total_vol"]
                result_pcts[idx] = float(np.mean([v <= sig_vol for v in window_vols])) * 100

        # Add today's signals to history for subsequent sessions
        if d in session_vols:
            session_history.append((d, session_vols[d]))

    df["rolling_vol_pct"] = result_pcts
    return df


def main():
    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    # ── Load overnight HMM ────────────────────────────────────────────────────
    on_hmm = None
    if ON_MODEL_PATH.exists():
        print(f"Loading overnight model from {ON_MODEL_PATH}...")
        on_hmm = joblib.load(ON_MODEL_PATH)
    else:
        print(f"WARNING: {ON_MODEL_PATH} not found — overnight_regime will be 'unknown'")

    # ── Seed rolling vol window from training tail ────────────────────────────
    print("Loading training tail for rolling vol seed...")
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["date"] = pd.to_datetime(train_df["date"]).dt.date
    train_dates_sorted = sorted(train_df["date"].unique())
    seed_dates = train_dates_sorted[-ROLLING_VOL_WINDOW:]
    seed_vols = [
        float(train_df[train_df["date"] == d]["total_vol"].median())
        for d in seed_dates
    ]
    print(f"  Seed window: {seed_dates[0]} to {seed_dates[-1]} ({len(seed_dates)} sessions)")

    # ── Process test dates ────────────────────────────────────────────────────
    test_dates = trading_dates(TEST_START, TEST_END)
    print(f"\nProcessing {len(test_dates)} test dates ({TEST_START} to {TEST_END})...")

    all_frames = []
    for i, d in enumerate(test_dates):
        cached = (TICK_CACHE_DIR / f"{d}_ticks.parquet").exists()
        print(f"  {d}  ({i+1}/{len(test_dates)}){'  [cached]' if cached else '  [building]'}")
        frame = process_date(d, on_hmm)
        if not frame.empty:
            frame["date"] = d
            all_frames.append(frame)

    if not all_frames:
        print("No signals found in test period.")
        return

    df = pd.concat(all_frames, ignore_index=True)
    print(f"\n{len(df)} raw signals across test period.")

    # ── Apply training normalization for sustained label ──────────────────────
    df["move_z_score"] = (df["move_normalized"] - MOVE_MEAN) / MOVE_STD
    df["sustained"] = (
        (df["move_z_score"] >= SUSTAINED_Z) &
        (df["mae_normalized"] <= SUSTAINED_MAE)
    )

    # ── Compute cluster_position and prior_absorption_count ───────────────────
    df = add_cluster_position(df)
    df = add_prior_absorption_count(df)

    # ── Compute rolling vol percentile (seeded from training tail) ────────────
    df = add_rolling_vol_pct(df, seed_vols, seed_dates)

    # ── Time filter ───────────────────────────────────────────────────────────
    sig_et = pd.to_datetime(df["signal_time"], utc=True).dt.tz_convert(ET)
    df["_mins"] = (sig_et.dt.hour - 9) * 60 + sig_et.dt.minute

    # ── Apply all 6 filters ───────────────────────────────────────────────────
    mask = (
        (df["overnight_regime"] == "directional") &
        (df["price_location"].isin(["outside_vah", "outside_val"])) &
        (df["cluster_position"].isin(["solo", "first"])) &
        (df["prior_absorption_count"] == 0) &
        (df["_mins"] >= TIME_START_MINS) & (df["_mins"] < TIME_END_MINS) &
        (df["rolling_vol_pct"] >= ROLLING_VOL_PCT)
    )
    filtered = df[mask].copy()

    sust    = filtered[filtered["sustained"]]
    non_sust = filtered[~filtered["sustained"]]

    # ── Directional days ──────────────────────────────────────────────────────
    dir_days = df[df["overnight_regime"] == "directional"]["date"].unique()

    # ── Build report ──────────────────────────────────────────────────────────
    n_total  = len(filtered)
    n_sust   = len(sust)
    sust_rate = n_sust / n_total * 100 if n_total > 0 else 0.0

    TRAIN_SUST_RATE   = 33.3
    TRAIN_MZ_SUST     = 2.422
    TRAIN_MAE_NONSUST = 0.2466
    MIN_SIGNALS_PASS  = 10
    MIN_RATE_PASS     = 25.0

    p("=" * 62)
    p("  WALK-FORWARD TEST: directional_absorption_setup_v1")
    p(f"  Test period : {TEST_START}  to  {TEST_END}")
    p(f"  Training    : 2025-03-17  to  2025-11-30  (baseline 33.3%)")
    p("=" * 62)
    p()
    p("  Applied filters:")
    p("    overnight_regime       = directional")
    p("    price_location         = outside_vah or outside_val")
    p("    cluster_position       = solo or first")
    p("    prior_absorption_count = 0")
    p("    signal_time (ET)       = 09:30 to 10:00")
    p(f"    total_vol              >= {ROLLING_VOL_PCT}th pct (rolling {ROLLING_VOL_WINDOW}-session)")
    p()
    p("-" * 62)
    p("  RAW PIPELINE COUNTS")
    p("-" * 62)
    p(f"  Test dates total           : {len(test_dates)}")
    p(f"  Dates with signals         : {df['date'].nunique()}")
    p(f"  Directional sessions       : {len(dir_days)}")
    p(f"  Total raw signals          : {len(df)}")
    p(f"  After all 6 filters        : {n_total}")
    p()
    p("-" * 62)
    p("  OVERALL RESULTS")
    p("-" * 62)
    p(f"  Signals (filtered)         : {n_total}")
    p(f"  Sustained                  : {n_sust}  ({sust_rate:.1f}%)")
    p(f"  Not sustained              : {len(non_sust)}  ({100-sust_rate:.1f}%)")
    p(f"  Training baseline          : 33.3%  (n=21)")
    delta = sust_rate - TRAIN_SUST_RATE
    p(f"  Delta vs training          : {delta:+.1f}pp")
    p()

    mz_s  = sust["move_z_score"].mean()     if n_sust > 0         else float("nan")
    mae_n = non_sust["mae_normalized"].mean() if len(non_sust) > 0 else float("nan")
    p(f"  Mean move_z_score (sust)   : {mz_s:.3f}  (train: {TRAIN_MZ_SUST:.3f})")
    p(f"  Mean mae_normalized (!sust): {mae_n:.4f}  (train: {TRAIN_MAE_NONSUST:.4f})")
    p()
    p("-" * 62)
    p("  BY PRICE LOCATION")
    p("-" * 62)
    p(f"  {'location':<14}  {'n':>5}  {'sust':>5}  {'sust%':>7}  {'train%':>8}")
    p(f"  {'-'*14}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*8}")
    train_by_loc = {"outside_vah": 40.0, "outside_val": 27.3}
    for loc in ["outside_vah", "outside_val"]:
        grp = filtered[filtered["price_location"] == loc]
        ns  = grp["sustained"].sum()
        n   = len(grp)
        r   = ns / n * 100 if n > 0 else 0
        p(f"  {loc:<14}  {n:>5}  {ns:>5}  {r:>6.1f}%  {train_by_loc[loc]:>7.1f}%")
    p()
    p("-" * 62)
    p("  DIRECTIONAL DAYS — SIGNAL DETAIL")
    p("-" * 62)
    p(f"  {'date':<12}  {'raw_sigs':>8}  {'filtered':>8}  {'sust':>6}  {'regime'}")
    p(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*12}")
    for d in sorted(dir_days):
        raw_n  = len(df[df["date"] == d])
        filt_n = len(filtered[filtered["date"] == d])
        sust_n = filtered[(filtered["date"] == d) & filtered["sustained"]]["sustained"].sum()
        regime = df[df["date"] == d]["overnight_regime"].iloc[0]
        p(f"  {str(d):<12}  {raw_n:>8}  {filt_n:>8}  {sust_n:>6}  {regime}")
    p()
    p("-" * 62)
    p("  PASS / FAIL ASSESSMENT")
    p("-" * 62)
    if n_total < MIN_SIGNALS_PASS:
        verdict = "INCONCLUSIVE"
        reason  = f"only {n_total} signals — need >= {MIN_SIGNALS_PASS} to assess"
    elif sust_rate >= MIN_RATE_PASS:
        verdict = "PASS"
        reason  = f"sustained rate {sust_rate:.1f}% >= {MIN_RATE_PASS}% threshold with n={n_total}"
    else:
        verdict = "FAIL"
        reason  = f"sustained rate {sust_rate:.1f}% < {MIN_RATE_PASS}% threshold (n={n_total})"
    p(f"  Verdict  : {verdict}")
    p(f"  Reason   : {reason}")
    p()
    p("=" * 62)

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to {OUT_TXT}")


if __name__ == "__main__":
    main()

"""
Run absorption-defended swing break detection over the full training range.

For each directional RTH session 2025-03-17 to 2025-11-30:
  1. Load ticks from output/tick_cache/{date}_ticks.parquet
  2. Build 40-range bars for RTH (09:30-11:00 ET)
  3. Compute session VAH/VAL from full RTH tick VolumeProfile (70% VA)
  4. Resolve overnight_regime from absorption_signals_labeled.csv
  5. Filter absorption signals for the date
  6. Run detect_swing_breaks()

Output: output/swing_break_signals.csv
"""

import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nqbt.analysis.range_bars import build_range_bars
from nqbt.analysis.volume_profile import VolumeProfile
from nqbt.analysis.swing_break_detector import detect_swing_breaks

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START = "2025-03-17"
TRAIN_END   = "2025-11-30"

TICK_CACHE_DIR = PROJECT_ROOT / "output" / "tick_cache"
SIGNALS_CSV    = PROJECT_ROOT / "output" / "absorption_signals_labeled.csv"
OUTPUT_CSV     = PROJECT_ROOT / "output" / "swing_break_signals.csv"
DEBUG_CSV      = PROJECT_ROOT / "output" / "swing_break_debug.csv"

ET              = "America/New_York"
RANGE_TICKS     = 40
TICKS_PER_LEVEL = 5

PARAMS = dict(
    swing_lookback_bars        = 8,
    min_pullback_bars          = 3,
    absorption_proximity_ticks = 8,
    vol_percentile_threshold   = 80,
    directional_only           = True,
)
# ─────────────────────────────────────────────────────────────────────────────


def trading_dates(start: str, end: str) -> list[date]:
    return [d.date() for d in pd.bdate_range(start=start, end=end)]


def load_signals() -> pd.DataFrame:
    if not SIGNALS_CSV.exists():
        print(f"ERROR: {SIGNALS_CSV} not found. Run run_signal_labeler.py first.")
        sys.exit(1)
    df = pd.read_csv(SIGNALS_CSV, parse_dates=["signal_time"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_session(d: date) -> tuple[list | None, pd.DataFrame | None]:
    """Return (bars, rth_ticks) or (None, None) if cache missing."""
    cache = TICK_CACHE_DIR / f"{d}_ticks.parquet"
    if not cache.exists():
        return None, None
    ticks = pd.read_parquet(cache)
    rth_start = pd.Timestamp(f"{d} 09:30:00", tz=ET)
    rth_end   = pd.Timestamp(f"{d} 11:00:00", tz=ET)
    rth_ticks = ticks[(ticks.index >= rth_start) & (ticks.index <= rth_end)]
    if len(rth_ticks) < 10:
        return None, None
    bars = build_range_bars(rth_ticks, range_ticks=RANGE_TICKS, ticks_per_level=TICKS_PER_LEVEL)
    if len(bars) < 5:
        return None, None
    return bars, rth_ticks


def session_value_area(rth_ticks: pd.DataFrame) -> tuple[float, float, float]:
    """Compute VAH, VAL, POC from full RTH ticks using 70% value area."""
    profile = VolumeProfile.build(rth_ticks, value_area_pct=0.70)
    return profile.vah, profile.val, profile.poc


def print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print("No setups found.")
        return

    n_bull   = (df["direction"] == "bull").sum()
    n_bear   = (df["direction"] == "bear").sum()
    mean_mae = df["mae"].mean()
    mean_mfe = df["mfe"].mean()
    pct_2x   = (df["mfe"] >= 2 * df["mae"]).mean() * 100
    pct_fwd  = df["single_bar_followthrough"].mean() * 100
    pct_sec  = df["secondary_absorption_confirmed"].mean() * 100

    print()
    print("=" * 60)
    print("  SWING BREAK DETECTOR -- SUMMARY")
    print("=" * 60)
    print(f"  Date range        : {TRAIN_START} to {TRAIN_END}")
    print(f"  directional_only  : {PARAMS['directional_only']}")
    print(f"  swing_lookback    : {PARAMS['swing_lookback_bars']}")
    print(f"  min_pullback_bars : {PARAMS['min_pullback_bars']}")
    print(f"  abs_prox_ticks    : {PARAMS['absorption_proximity_ticks']}")
    print(f"  vol_pct_thr       : {PARAMS['vol_percentile_threshold']}")
    print()
    print(f"  Total setups      : {n}")
    print(f"    Bull            : {n_bull}  ({n_bull/n*100:.1f}%)")
    print(f"    Bear            : {n_bear}  ({n_bear/n*100:.1f}%)")
    print()
    print(f"  Mean MAE (pts)    : {mean_mae:.2f}")
    print(f"  Mean MFE (pts)    : {mean_mfe:.2f}")
    print(f"  MFE >= 2x MAE     : {pct_2x:.1f}%")
    print(f"  1-bar followthru  : {pct_fwd:.1f}%")
    print(f"  Secondary abs     : {pct_sec:.1f}%")
    print()

    for sec_val, label in [(True, "secondary_abs=True"), (False, "secondary_abs=False")]:
        sub = df[df["secondary_absorption_confirmed"] == sec_val]
        if sub.empty:
            continue
        sm = sub["mae"].mean()
        sf = sub["mfe"].mean()
        sp = (sub["mfe"] >= 2 * sub["mae"]).mean() * 100
        print(f"  {label:<26} n={len(sub):>3}  mae={sm:.2f}  mfe={sf:.2f}  MFE>=2xMAE={sp:.1f}%")

    print()

    for d_label, sub in [("bull", df[df["direction"] == "bull"]),
                          ("bear", df[df["direction"] == "bear"])]:
        if sub.empty:
            continue
        sm = sub["mae"].mean()
        sf = sub["mfe"].mean()
        sp = (sub["mfe"] >= 2 * sub["mae"]).mean() * 100
        pf = sub["single_bar_followthrough"].mean() * 100
        print(f"  {d_label:<5}  n={len(sub):>3}  mae={sm:.2f}  mfe={sf:.2f}  MFE>=2x={sp:.1f}%  1-bar={pf:.1f}%")

    rc = df["bars_until_level_reclaimed"].dropna()
    if not rc.empty:
        print()
        print(f"  Level reclaimed (median bars) : {rc.median():.0f}")
        print(f"  Never reclaimed (in 20 bars)  : {df['bars_until_level_reclaimed'].isna().sum()}")

    print("=" * 60)


def main() -> None:
    print(f"Loading absorption signals from {SIGNALS_CSV} ...")
    signals_df = load_signals()
    n_dates = signals_df["date"].nunique() if "date" in signals_df.columns else "?"
    print(f"  {len(signals_df)} signals loaded across {n_dates} dates.")

    dates = trading_dates(TRAIN_START, TRAIN_END)
    print(f"\nScanning {len(dates)} trading dates ({TRAIN_START} to {TRAIN_END}) ...")
    print(f"Params: {PARAMS}\n")

    all_setups: list[dict] = []
    n_skipped_regime = 0
    n_skipped_cache  = 0
    n_processed      = 0

    for d in dates:
        sub = signals_df[signals_df["date"] == d] if "date" in signals_df.columns else pd.DataFrame()
        regime = str(sub["overnight_regime"].iloc[0]) if not sub.empty and "overnight_regime" in sub.columns else "unknown"

        if PARAMS["directional_only"] and regime != "directional":
            n_skipped_regime += 1
            continue

        bars, rth_ticks = load_session(d)
        if bars is None:
            n_skipped_cache += 1
            print(f"  {d}  [{regime}]  -- no tick cache / insufficient bars, skipping")
            continue

        try:
            vah, val, poc = session_value_area(rth_ticks)
        except Exception as e:
            print(f"  {d}  [{regime}]  -- VAH/VAL failed: {e}, skipping")
            continue

        setups = detect_swing_breaks(
            bars             = bars,
            absorption_df    = sub,
            overnight_regime = regime,
            trading_date     = d,
            session_vah      = vah,
            session_val      = val,
            debug            = True,
            **PARAMS,
        )

        n_processed += 1
        flag = f"  {len(setups)} setup(s)" if setups else "  --"
        print(f"  {d}  [{regime}]  bars={len(bars):>3}  vah={vah:.2f}  val={val:.2f}{flag}")
        all_setups.extend(setups)

    print(f"\nDates processed : {n_processed}")
    print(f"Skipped (regime): {n_skipped_regime}")
    print(f"Skipped (cache) : {n_skipped_cache}")

    if not all_setups:
        print("\nNo setups found. Consider relaxing swing_lookback_bars or absorption_proximity_ticks.")
        return

    df = pd.DataFrame(all_setups)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} setups to {OUTPUT_CSV}")

    # Debug CSV: per-setup detail for manual chart verification
    debug_cols = [
        "date", "direction", "entry_time", "entry_price",
        "swing_price", "price_location", "tradeable",
        "absorption_bar_time", "absorption_bar_price", "absorption_proximity_ticks_actual",
        "pullback_bars", "pullback_low",
        "break_bar_time", "break_bar_total_vol", "break_bar_vol_pct",
        "mae", "mfe", "overnight_regime",
    ]
    debug_df = df[[c for c in debug_cols if c in df.columns]]
    debug_df.to_csv(DEBUG_CSV, index=False)
    print(f"Saved debug detail to {DEBUG_CSV}")

    print_summary(df)


if __name__ == "__main__":
    main()

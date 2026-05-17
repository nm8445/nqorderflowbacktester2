"""Aggregate Mech A trades into per-variant/per-param summary metrics.

CURRENT DEFAULTS (synced with Mech B):
  - CHAINED Mode 1 (multiple non-overlapping trades per day)
  - Adaptive level band (BAND_K=0.25 default)
  - Symmetric wick-anchored exits (TP_dist, SL_dist = base + mult*ATR)
  - Lookahead-fixed (prior-day MQ)

Sweep dimensions:
  variant     : A1 / A2
  K_BARS      : {2, 3, 4}
  min_M       : {30, 50, ..., 510}
  strict      : {True, False}
  band_K      : {FIXED, 0.25, 0.5, 0.75, 1.0}   level-proximity band

NOTE: Mech A trades parquet must be re-run after framework changes (symmetric
exits + lookahead fix). Run range_break_mech_a_study.py first.
"""

from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import (  # noqa: E402
    trade_pnls_vectorized, mode1_chained_dedupe, stats,
    SL_MULTS, BAND_K_VALUES, BAND_MIN, BAND_MAX, FIXED_BAND_PT,
    SIGNAL_PASS_TP_M, SIGNAL_PASS_SL_M, N_WORKERS, CHUNKSIZE,
)

OUT_DIR  = Path(__file__).parent
PARQUET_DIR = OUT_DIR / "parquets"
PARQUET_DIR.mkdir(exist_ok=True)
TRADES   = PARQUET_DIR / "entry_signal_trades_mech_a.parquet"
SUMMARY  = PARQUET_DIR / "entry_signal_summary_mech_a.parquet"

K_BARS_VALUES = [2, 3, 4]
MIN_M_VALUES  = list(range(30, 511, 20))


def apply_filters(df: pd.DataFrame, variant: str, K_bars: int, min_M: float,
                  strict_shorts: bool, band_K=0.25) -> pd.DataFrame:
    """Filter Mech A trades:
       - variant + K_bars + a_accum_vol >= M
       - strict_shorts gate (SHORT direction only)
       - adaptive level proximity band (default BAND_K=0.25)
    """
    sub = df[(df["variant"] == variant) &
             (df["a_K_bars"] == K_bars) &
             (df["a_accum_vol"] >= min_M)].copy()
    if strict_shorts:
        short_mask = sub["direction"] == "SHORT"
        keep = ~short_mask | sub["strict_short"].fillna(False).astype(bool)
        sub = sub[keep].copy()
    # Adaptive band overlap check (requires signal_low / signal_high / near_level / atr_at_entry)
    if band_K == "FIXED":
        band = np.full(len(sub), FIXED_BAND_PT)
    else:
        band = np.clip(band_K * sub["atr_at_entry"].values, BAND_MIN, BAND_MAX)
    overlap = ((sub["signal_low"].values  <= sub["near_level"].values + band) &
               (sub["signal_high"].values >= sub["near_level"].values - band))
    return sub[overlap].copy()


# Module-level globals populated in each worker process at startup.
_W_DF = None
_W_TP = None
_W_SL = None


def _init_worker(df, tp_M, sl_M):
    warnings.filterwarnings("ignore", category=FutureWarning)
    global _W_DF, _W_TP, _W_SL
    _W_DF = df
    _W_TP = tp_M
    _W_SL = sl_M


def _evaluate_combo(args):
    variant, K, M_zone, strict, band_K = args
    filtered = apply_filters(_W_DF, variant, K, M_zone, strict, band_K)
    deduped  = mode1_chained_dedupe(filtered, _W_TP, _W_SL)
    if len(deduped) == 0:
        return None
    max_unrealized_dd = float(deduped["mae_pts"].min())
    pnl = trade_pnls_vectorized(deduped, _W_TP, _W_SL)
    s = stats(deduped, pnl)
    return {
        "variant":  variant,
        "K_bars":   K,
        "min_M":    M_zone,
        "strict":   strict,
        "band_K":   str(band_K),
        "tp_M":     _W_TP,
        "sl_M":     _W_SL,
        "max_unrealized_dd_pts": max_unrealized_dd,
        **s,
    }


def main():
    print(f"loading {TRADES}...")
    df = pd.read_parquet(TRADES)
    print(f"  {len(df):,} trades, {df['date'].min()} -> {df['date'].max()}")
    print(f"  variants: {df['variant'].value_counts().to_dict()}")
    print(f"  directions: {df['direction'].value_counts().to_dict()}")
    print(f"  K_bars: {df['a_K_bars'].value_counts().to_dict()}")
    print()

    combos = [(v, K, M_zone, strict, band_K)
              for v in ["A1", "A2"]
              for K in K_BARS_VALUES
              for M_zone in MIN_M_VALUES
              for strict in [True, False]
              for band_K in BAND_K_VALUES]
    print(f"sweeping {len(combos)} combos with {N_WORKERS} workers...")

    import time as _t
    t0 = _t.time()
    with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_init_worker,
            initargs=(df, SIGNAL_PASS_TP_M, SIGNAL_PASS_SL_M)) as ex:
        results = list(ex.map(_evaluate_combo, combos, chunksize=CHUNKSIZE))
    print(f"  swept in {_t.time()-t0:.0f}s")

    rows = [r for r in results if r is not None]
    if not rows:
        print("no rows produced")
        return

    summary = pd.DataFrame(rows)
    summary.to_parquet(SUMMARY, compression="zstd", index=False)
    print(f"\nwrote {SUMMARY}  ({len(summary)} combos)")

    big = summary[summary["n"] >= 30].copy()
    cols = ["variant","K_bars","min_M","strict","band_K","tp_M","sl_M",
            "n","n_long","n_short","total_pnl","mean_pnl","wr","wr_long","wr_short",
            "pf","sharpe","max_dd","max_unrealized_dd_pts"]

    print(f"\n{'='*100}\nTOP 20 BY SHARPE  (n>=30)\n{'='*100}")
    print(big.sort_values("sharpe", ascending=False).head(20)[cols]
             .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'='*100}\nTOP 20 BY TOTAL PROFIT  (n>=30)\n{'='*100}")
    print(big.sort_values("total_pnl", ascending=False).head(20)[cols]
             .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'='*100}\nTOP 20 BY PF  (n>=30, pf<999)\n{'='*100}")
    print(big[big["pf"] < 999.9].sort_values("pf", ascending=False).head(20)[cols]
             .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    sys.exit(main())

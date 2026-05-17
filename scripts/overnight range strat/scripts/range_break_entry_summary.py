"""Aggregate range_break_entry_signal_study trades into per-combo summary
metrics: Sharpe (daily-aggregated × √252), PF, MaxDD, MaxUnrealizedDD,
TotalProfit, TradeCount, WR, LongWR, ShortWR.

Sweep dimensions (current defaults):
  variant     : B1 / B2 / B3 (entry-timing variants)
  pinbar_X    : {0.75 .. 2.50}      pinbar_ratio >= X (loose floor — see study)
  window_N    : {5, 10, 15, 20}     ticks for windowed orderflow scan
  delta_D     : {30 .. 500 step 20} min |delta| in best window
  strict      : {True, False}       strict-shorts gate (close < OLO)
  band_K      : {FIXED, 0.25, 0.5, 0.75, 1.0}   level proximity band
                  band = clip(band_K * ATR, BAND_MIN=5, BAND_MAX=20)
                  "FIXED" = legacy 10-pt fixed band

Trade dedup: CHAINED Mode 1 — one trade in flight at a time globally per day.
  When a trade exits (TP/SL/end-of-day), the next valid signal can fire.
  Direction-agnostic; LONG exit allows next trade to be LONG or SHORT.

Exits: SYMMETRIC WICK-ANCHORED (signal-pass uses tp_M=sl_M=1.0, R:R=1:1).

Parallelism: 4 worker processes (configurable via N_WORKERS).
"""

from __future__ import annotations

import datetime as dt
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

N_WORKERS = 4    # parallel workers for the combo sweep
CHUNKSIZE = 50

OUT_DIR  = Path(__file__).parent
PARQUET_DIR = OUT_DIR / "parquets"
PARQUET_DIR.mkdir(exist_ok=True)
TRADES   = PARQUET_DIR / "entry_signal_trades.parquet"
SUMMARY  = PARQUET_DIR / "entry_signal_summary.parquet"

# Single-X pinbar classifier (X2/X3 confirmed not helpful via shape-bucket study)
PINBAR_X  = [0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50]    # lw/body (or uw/body for shorts)
WINDOW_NS = [5, 10, 15, 20]                      # ticks
DELTA_D   = list(range(30, 501, 20)) + [500]
DELTA_D   = sorted(set(DELTA_D))
# Adaptive level band: band = clip(BAND_K * ATR, BAND_MIN, BAND_MAX)
# Default BAND_K = 0.25 (best per in-sample sweep).
# Special value "FIXED" reproduces the legacy fixed-10pt band.
DEFAULT_BAND_K = 0.25
BAND_K_VALUES  = ["FIXED", 0.25, 0.50, 0.75, 1.00]
BAND_MIN       = 5.0
BAND_MAX       = 20.0
FIXED_BAND_PT  = 10.0
# PURE ATR exits: SL_dist = sl_M * ATR, TP_dist = tp_M * ATR (no wick anchor)
TP_MULTS  = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
             2.25, 2.50, 2.75, 3.00]
SL_MULTS  = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
             2.25, 2.50, 2.75, 3.00]
SL_PADS   = SL_MULTS    # backward-compat alias
# Signal-strength comparison: fixed TP=SL=1xATR (R:R = 1:1), no ratio_R filter
SIGNAL_PASS_TP_M = 1.00
SIGNAL_PASS_SL_M = 1.00


def trade_pnls_vectorized(df: pd.DataFrame, tp_M: float, sl_P: float) -> np.ndarray:
    """Vectorized realized PnL under symmetric wick-anchored exits.

    base_distance = |entry - wick_anchor|
    SL_distance   = base + sl_P * ATR
    TP_distance   = base + tp_M * ATR

    PnL when TP hit first  = +TP_distance
    PnL when SL hit first  = -SL_distance
    PnL when neither hit   = close_pnl_pts (held to 17:00)
    """
    sign = np.where(df["direction"].values == "LONG", 1, -1)
    atr   = df["atr_at_entry"].values
    entry = df["entry_price"].values
    anchor = df["sl_anchor"].values
    base = sign * (entry - anchor)
    sl_distance = base + sl_P * atr
    tp_distance = base + tp_M * atr
    sl_pnl = -sl_distance
    tp_pnl =  tp_distance

    sl_idx = df[f"sl_{sl_P:.2f}_idx"].values.astype(int)
    tp_idx = df[f"tp_{tp_M:.2f}_idx"].values.astype(int)

    pnl = np.where(
        (sl_idx >= 0) & ((tp_idx < 0) | (sl_idx < tp_idx)), sl_pnl,
        np.where(tp_idx >= 0, tp_pnl, df["close_pnl_pts"].values)
    )
    return pnl


def apply_filters(df: pd.DataFrame, variant: str, X: float, N: int,
                  D: float, strict_shorts: bool,
                  band_K=DEFAULT_BAND_K) -> pd.DataFrame:
    """Filter trades:
       - variant
       - pinbar_ratio   >= X
       - |abs_delta_w{N}| >= D
       - strict_shorts gate (only for SHORT direction trades)
       - level proximity zone:
           band_K == "FIXED" -> uses fixed FIXED_BAND_PT
           else              -> band = clip(band_K * ATR, BAND_MIN, BAND_MAX)
         keeps trade only if signal candle [low, high] overlaps [level±band].
    """
    sub = df[df["variant"] == variant].copy()
    keep = ((sub["pinbar_ratio"]          >= X) &
            (sub[f"abs_delta_w{N}"].abs() >= D))
    if strict_shorts:
        long_mask = sub["direction"] == "LONG"
        strict_ok = sub["strict_short"].fillna(False).astype(bool)
        keep &= (long_mask | strict_ok)
    # Adaptive band filter
    if band_K == "FIXED":
        band = np.full(len(sub), FIXED_BAND_PT)
    else:
        band = np.clip(band_K * sub["atr_at_entry"].values, BAND_MIN, BAND_MAX)
    overlap = ((sub["signal_low"].values  <= sub["near_level"].values + band) &
               (sub["signal_high"].values >= sub["near_level"].values - band))
    keep &= overlap
    return sub[keep].copy()


def mode1_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """LEGACY: per (date, bias_origin) keep earliest entry only.
    Replaced by mode1_chained_dedupe — kept for backward compat only."""
    if df.empty: return df
    return (df.sort_values("entry_time")
              .drop_duplicates(subset=["date", "bias_origin"], keep="first"))


def mode1_chained_dedupe(df: pd.DataFrame, tp_M: float, sl_M: float) -> pd.DataFrame:
    """Chained Mode 1: walk signals chronologically per day. Take a new trade
    only when no position is currently open (previous trade exited).

    Exit bar index per trade depends on (tp_M, sl_M):
      * earliest of tp_idx, sl_idx if either fires
      * end-of-day (large sentinel) if held to close (blocks all later same-day signals)

    Re-entry rule: signal trade i+1's entry_bar_idx must be STRICTLY GREATER than
    the previous trade's exit_bar_idx (the bar where it resolved). Same-bar
    re-entry not allowed (would overlap).

    Direction-agnostic globally per day (a LONG exit allows LONG or SHORT next).
    """
    if df.empty:
        return df

    tp_col = f"tp_{tp_M:.2f}_idx"
    sl_col = f"sl_{sl_M:.2f}_idx"

    keep_idx = []
    for _, group in df.groupby("date"):
        group = group.sort_values("entry_bar_idx")
        last_exit_bidx = -1   # no open position
        for idx, row in group.iterrows():
            if int(row["entry_bar_idx"]) > last_exit_bidx:
                keep_idx.append(idx)
                tp_idx = int(row[tp_col])
                sl_idx = int(row[sl_col])
                if tp_idx >= 0 and sl_idx >= 0:
                    exit_bidx = min(tp_idx, sl_idx)
                elif tp_idx >= 0:
                    exit_bidx = tp_idx
                elif sl_idx >= 0:
                    exit_bidx = sl_idx
                else:
                    exit_bidx = 10**9   # held to close — blocks everything else today
                last_exit_bidx = exit_bidx
    return df.loc[keep_idx]


def stats(deduped: pd.DataFrame, pnl: np.ndarray) -> dict:
    """Compute summary stats over the trade series.
    Sharpe uses DAILY-aggregated PnL × sqrt(252)."""
    n = len(pnl)
    if n == 0:
        return {"n": 0}
    wins = pnl > 0
    direction = deduped["direction"].values

    # Trade-level equity curve for MaxDD
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq - peak).min())

    pos = pnl[pnl > 0].sum()
    neg = -pnl[pnl < 0].sum()
    pf = (pos / neg) if neg > 0 else (np.inf if pos > 0 else 0.0)

    # Daily-aggregated PnL for Sharpe annualization
    daily = pd.Series(pnl, index=deduped["date"].values).groupby(level=0).sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    long_mask  = direction == "LONG"
    short_mask = direction == "SHORT"

    return {
        "n":           int(n),
        "n_long":      int(long_mask.sum()),
        "n_short":     int(short_mask.sum()),
        "total_pnl":   float(pnl.sum()),
        "mean_pnl":    float(pnl.mean()),
        "wr":          float(wins.mean()),
        "wr_long":     float(wins[long_mask].mean()) if long_mask.any() else np.nan,
        "wr_short":    float(wins[short_mask].mean()) if short_mask.any() else np.nan,
        "pf":          float(pf) if np.isfinite(pf) else 999.9,
        "sharpe":      float(sharpe),
        "max_dd":      float(max_dd),
        "n_days":      int(daily.size),
    }


# Module-level globals populated in each worker process at startup.
_W_DF   = None
_W_TP   = None
_W_SL   = None


def _init_worker(df, tp_M, sl_M):
    """ProcessPoolExecutor initializer — runs once per worker at startup."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    global _W_DF, _W_TP, _W_SL
    _W_DF = df
    _W_TP = tp_M
    _W_SL = sl_M


def _evaluate_combo(args):
    """Worker: process ONE (variant, X, N, D, strict, band_K) combo and
    return a row dict (or None if empty)."""
    variant, X, N, D, strict, band_K = args
    filtered = apply_filters(_W_DF, variant, X, N, D, strict, band_K)
    deduped  = mode1_chained_dedupe(filtered, _W_TP, _W_SL)
    if len(deduped) == 0:
        return None
    max_unrealized_dd = float(deduped["mae_pts"].min())
    pnl = trade_pnls_vectorized(deduped, _W_TP, _W_SL)
    s = stats(deduped, pnl)
    return {
        "variant":  variant,
        "pinbar_X": X,
        "window_N": N,
        "delta_D":  D,
        "strict":   strict,
        "band_K":   str(band_K),
        "tp_M":     _W_TP,
        "sl_M":     _W_SL,
        "max_unrealized_dd_pts": max_unrealized_dd,
        **s,
    }


def main():
    print(f"loading {TRADES} ...")
    df = pd.read_parquet(TRADES)
    print(f"  {len(df):,} trades, {df['date'].min()} -> {df['date'].max()}")
    print(f"  variants: {df['variant'].value_counts().to_dict()}")
    print(f"  directions: {df['direction'].value_counts().to_dict()}")
    print()

    combos = [(v, X, N, D, strict, band_K)
              for v in ["B1", "B2", "B3"]
              for X in PINBAR_X
              for N in WINDOW_NS
              for D in DELTA_D
              for strict in [True, False]
              for band_K in BAND_K_VALUES]
    total = len(combos)
    print(f"signal-strength pass: sweeping {total} combos "
          f"(variant x X x N x D x strict x BAND_K)")
    print(f"  fixed tp_M={SIGNAL_PASS_TP_M}  sl_M={SIGNAL_PASS_SL_M}  "
          f"(symmetric, R:R = 1:1, CHAINED Mode 1, {N_WORKERS} workers)")

    import time as _t
    t0 = _t.time()
    with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_init_worker,
            initargs=(df, SIGNAL_PASS_TP_M, SIGNAL_PASS_SL_M)) as ex:
        # ex.map preserves order; chunksize=50 batches tasks for IPC efficiency
        results = list(ex.map(_evaluate_combo, combos, chunksize=CHUNKSIZE))
    print(f"  swept in {_t.time()-t0:.0f}s")

    rows = [r for r in results if r is not None]
    if not rows:
        print("no rows produced")
        return

    summary = pd.DataFrame(rows)
    summary.to_parquet(SUMMARY, compression="zstd", index=False)
    print(f"\nwrote {SUMMARY}  ({len(summary)} combos)")

    # Filter to combos with reasonable sample size
    big = summary[summary["n"] >= 30].copy()
    print(f"\n{'='*100}")
    print(f"TOP 20 BY SHARPE  (n>=30 trades)")
    print(f"{'='*100}")
    cols = ["variant","pinbar_X","window_N","delta_D","strict","band_K",
            "tp_M","sl_M",
            "n","n_long","n_short","total_pnl","mean_pnl","wr","wr_long","wr_short",
            "pf","sharpe","max_dd","max_unrealized_dd_pts"]
    top_sharpe = big.sort_values("sharpe", ascending=False).head(20)
    print(top_sharpe[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'='*100}")
    print(f"TOP 20 BY TOTAL PROFIT  (n>=30)")
    print(f"{'='*100}")
    print(big.sort_values("total_pnl", ascending=False).head(20)[cols]
             .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'='*100}")
    print(f"TOP 20 BY PROFIT FACTOR  (n>=30, pf<999)")
    print(f"{'='*100}")
    print(big[big["pf"] < 999.9].sort_values("pf", ascending=False).head(20)[cols]
             .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    sys.exit(main())

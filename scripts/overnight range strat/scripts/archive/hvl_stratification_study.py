"""HVL / gamma-regime stratification study (QQQ-only).

Stratifications tested:
  1. By gamma_regime (qqq_gamma_sign) — always defined
       * POS gamma : sign = +1
       * NEG gamma : sign = -1
  2. By position vs QQQ HVL 0DTE (next-day-perspective filter)
       * ABOVE_HVL : entry > qqq_hvl_0dte_nq
       * BELOW_HVL : entry < qqq_hvl_0dte_nq
  3. By extended HVL (full coverage; no ±5% gate)

Direction split per stratum (LONG vs SHORT) for hypothesis testing.

Uses the user's selected B2 config:
  X=0.75, N=5, D=70, strict=True, BAND_K=0.25, tp_M=sl_M=1.0
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import (
    apply_filters, mode1_chained_dedupe, trade_pnls_vectorized, stats,
)

PARQUET_DIR  = Path(__file__).parent / "parquets"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
TRADELOG_DIR.mkdir(exist_ok=True)
TRADES = PARQUET_DIR / "entry_signal_trades.parquet"
MQ     = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")

VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 5, 70, True, 0.25
TP_M, SL_M = 1.0, 1.0


def main():
    print(f"loading trades and applying filters "
          f"(B2 X={X} N={N} D={D} strict={STRICT} BAND_K={BAND_K})...")
    df = pd.read_parquet(TRADES)
    filtered = apply_filters(df, VARIANT, X, N, D, STRICT, BAND_K)
    deduped  = mode1_chained_dedupe(filtered, TP_M, SL_M)
    deduped["pnl"] = trade_pnls_vectorized(deduped, TP_M, SL_M)
    deduped["date"] = pd.to_datetime(deduped["date"]).dt.date
    print(f"  {len(deduped)} trades after dedupe")
    print()

    # Load levels parquet
    print("loading MQ levels (QQQ HVL with NTD filter + gamma_sign + extended HVL)...")
    mq = pd.read_parquet(MQ)
    mq["date"] = pd.to_datetime(mq["date"]).dt.date
    mq = mq.set_index("date")
    needed = ["qqq_hvl_0dte_nq", "qqq_hvl_extended_nq", "qqq_gamma_sign"]
    missing_cols = [c for c in needed if c not in mq.columns]
    if missing_cols:
        print(f"  ERROR: missing columns {missing_cols} — re-run build_menthorq_levels_bulk.py")
        return

    mq_dates_sorted = sorted(mq.index.tolist())

    def prior_mq_day(d):
        prev = None
        for md in mq_dates_sorted:
            if md < d: prev = md
            else: break
        return prev

    # Map prior-day HVLs and gamma sign onto each trade
    cols_to_map = ["qqq_hvl_0dte_nq", "qqq_hvl_extended_nq", "qqq_hvl_dist_pct", "qqq_gamma_sign"]
    lookups = {c: {} for c in cols_to_map}
    for d in deduped["date"].unique():
        p = prior_mq_day(d)
        if p is None: continue
        for c in cols_to_map:
            if c in mq.columns:
                lookups[c][d] = mq.loc[p, c]

    for c in cols_to_map:
        deduped[c.replace("_nq","")] = deduped["date"].map(lookups[c])

    n_with_local = deduped["qqq_hvl_0dte"].notna().sum()
    n_with_ext   = deduped["qqq_hvl_extended"].notna().sum()
    n_with_gamma = deduped["qqq_gamma_sign"].notna().sum()
    print(f"  Trades with local QQQ HVL 0DTE  : {n_with_local}/{len(deduped)} "
          f"({n_with_local/len(deduped)*100:.1f}%)")
    print(f"  Trades with extended QQQ HVL    : {n_with_ext}/{len(deduped)} "
          f"({n_with_ext/len(deduped)*100:.1f}%)")
    print(f"  Trades with QQQ gamma_sign      : {n_with_gamma}/{len(deduped)} "
          f"({n_with_gamma/len(deduped)*100:.1f}%)")
    print()

    def stats_bucket(label, sub):
        if sub.empty:
            print(f"  {label:<22}  EMPTY")
            return
        pnl = sub["pnl"].values
        long_mask = (sub["direction"]=="LONG").values
        short_mask = (sub["direction"]=="SHORT").values
        wins = pnl > 0
        pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
        pf = pos/neg if neg > 0 else (np.inf if pos > 0 else 0.0)
        daily = pd.Series(pnl, index=sub["date"].values).groupby(level=0).sum()
        sharpe = daily.mean()/daily.std()*np.sqrt(252) if daily.std() > 0 else 0.0
        eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq-peak).min()
        wr_l = wins[long_mask].mean() if long_mask.any() else float("nan")
        wr_s = wins[short_mask].mean() if short_mask.any() else float("nan")
        print(f"  {label:<22}  n={len(sub):>4}"
              f"  L={int(long_mask.sum()):>4}/S={int(short_mask.sum()):>4}"
              f"  total={pnl.sum():>+8.1f}  mean={pnl.mean():>+5.2f}"
              f"  WR={wins.mean():>5.1%}  WR_L={wr_l:>5.1%}  WR_S={wr_s:>5.1%}"
              f"  PF={min(pf,999.0):>5.2f}  Sharpe={sharpe:>+5.2f}  MDD={max_dd:>+8.0f}")

    print("=" * 180)
    print("STRATIFICATION 1 — BY GAMMA REGIME (qqq_gamma_sign, near-100% coverage)")
    print("=" * 180)
    print(f"  {'bucket':<26}  n      L/S        total    mean    WR     WR_L    WR_S    PF    Sharpe   MDD")
    pos_g = deduped[deduped["qqq_gamma_sign"] == 1]
    neg_g = deduped[deduped["qqq_gamma_sign"] == -1]
    stats_bucket("POS gamma (sign=+1)", pos_g)
    stats_bucket("  -> LONG",          pos_g[pos_g["direction"]=="LONG"])
    stats_bucket("  -> SHORT",         pos_g[pos_g["direction"]=="SHORT"])
    stats_bucket("NEG gamma (sign=-1)", neg_g)
    stats_bucket("  -> LONG",          neg_g[neg_g["direction"]=="LONG"])
    stats_bucket("  -> SHORT",         neg_g[neg_g["direction"]=="SHORT"])

    print()
    print("=" * 180)
    print("STRATIFICATION 2 — BY POSITION vs LOCAL QQQ HVL 0DTE (next-day filter)")
    print("=" * 180)
    has_hvl = deduped["qqq_hvl_0dte"].notna()
    above = deduped[has_hvl & (deduped["entry_price"] > deduped["qqq_hvl_0dte"])]
    below = deduped[has_hvl & (deduped["entry_price"] < deduped["qqq_hvl_0dte"])]
    stats_bucket("ABOVE local HVL", above)
    stats_bucket("  -> LONG",       above[above["direction"]=="LONG"])
    stats_bucket("  -> SHORT",      above[above["direction"]=="SHORT"])
    stats_bucket("BELOW local HVL", below)
    stats_bucket("  -> LONG",       below[below["direction"]=="LONG"])
    stats_bucket("  -> SHORT",      below[below["direction"]=="SHORT"])

    print()
    print("=" * 180)
    print("STRATIFICATION 3 — BY POSITION vs EXTENDED QQQ HVL (full coverage)")
    print("=" * 180)
    has_ext = deduped["qqq_hvl_extended"].notna()
    above_x = deduped[has_ext & (deduped["entry_price"] > deduped["qqq_hvl_extended"])]
    below_x = deduped[has_ext & (deduped["entry_price"] < deduped["qqq_hvl_extended"])]
    stats_bucket("ABOVE extended HVL", above_x)
    stats_bucket("  -> LONG",          above_x[above_x["direction"]=="LONG"])
    stats_bucket("  -> SHORT",         above_x[above_x["direction"]=="SHORT"])
    stats_bucket("BELOW extended HVL", below_x)
    stats_bucket("  -> LONG",          below_x[below_x["direction"]=="LONG"])
    stats_bucket("  -> SHORT",         below_x[below_x["direction"]=="SHORT"])


if __name__ == "__main__":
    sys.exit(main())

"""Find configs in the X=0.75 area that work in BOTH in-sample and out-of-sample.

Holds locked: BAND_K=0.25, TP=SL=1.0, chained Mode 1.
Sweeps:
  X      in [0.50, 0.75, 1.00, 1.25, 1.50]
  N      in [5, 10, 15, 20]
  D      in [30, 50, 70, 100, 150, 200, 300, 500]
  strict in [True, False]

For each config: evaluate on both IS and OOS trades, then rank by:
  - Both must have n>=15 OOS trades
  - Both must have positive total PnL
  - Sort by minimum PF across IS/OOS  (most "honestly robust" criterion)
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import (
    apply_filters, mode1_chained_dedupe, trade_pnls_vectorized,
)

PARQUET_DIR = Path(__file__).parent / "parquets"
TRADES_IS   = PARQUET_DIR / "entry_signal_trades.parquet"
TRADES_OOS  = PARQUET_DIR / "entry_signal_trades_oos.parquet"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
TRADELOG_DIR.mkdir(exist_ok=True)
OUT_TXT     = TRADELOG_DIR / "robust_configs_is_oos.txt"

X_VALS      = [0.50, 0.75, 1.00, 1.25, 1.50]
N_VALS      = [5, 10, 15, 20]
D_VALS      = [30, 50, 70, 100, 150, 200, 300, 500]
STRICT_VALS = [True, False]
BAND_K      = 0.25
TP_M        = 1.0
SL_M        = 1.0


def evaluate(df: pd.DataFrame, X: float, N: int, D: int, strict: bool) -> dict | None:
    filtered = apply_filters(df, "B2", X, N, D, strict, BAND_K)
    deduped  = mode1_chained_dedupe(filtered, TP_M, SL_M)
    if len(deduped) == 0:
        return None
    deduped = deduped.copy()
    deduped["pnl"] = trade_pnls_vectorized(deduped, TP_M, SL_M)
    deduped["date"] = pd.to_datetime(deduped["date"]).dt.date
    pnl = deduped["pnl"].values
    long_mask = (deduped["direction"] == "LONG").values
    short_mask = (deduped["direction"] == "SHORT").values
    wins = pnl > 0
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0)
    daily = pd.Series(pnl, index=deduped["date"].values).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    return {
        "n": len(deduped), "n_long": int(long_mask.sum()), "n_short": int(short_mask.sum()),
        "total": pnl.sum(), "mean": pnl.mean(),
        "wr": wins.mean(),
        "wr_long": (pnl[long_mask] > 0).mean() if long_mask.any() else float("nan"),
        "wr_short": (pnl[short_mask] > 0).mean() if short_mask.any() else float("nan"),
        "pf": pf, "sharpe": sharpe, "max_dd": max_dd,
        "long_total": pnl[long_mask].sum(), "short_total": pnl[short_mask].sum(),
    }


def main():
    print("loading trade parquets...")
    is_df  = pd.read_parquet(TRADES_IS)
    oos_df = pd.read_parquet(TRADES_OOS)
    print(f"  IS:  {len(is_df):,} trades")
    print(f"  OOS: {len(oos_df):,} trades")

    combos = list(product(X_VALS, N_VALS, D_VALS, STRICT_VALS))
    print(f"sweeping {len(combos)} configs (X x N x D x strict, BAND_K=0.25, TP=SL=1.0)...")

    rows = []
    for i, (X, N, D, strict) in enumerate(combos, 1):
        is_s  = evaluate(is_df,  X, N, D, strict)
        oos_s = evaluate(oos_df, X, N, D, strict)
        if is_s is None or oos_s is None:
            continue
        rows.append({
            "X": X, "N": N, "D": D, "strict": strict,
            "is_n":     is_s["n"],     "is_total":   is_s["total"],
            "is_pf":    is_s["pf"],    "is_sharpe":  is_s["sharpe"],
            "is_wr":    is_s["wr"],    "is_mean":    is_s["mean"],
            "is_long_total":  is_s["long_total"],
            "is_short_total": is_s["short_total"],
            "oos_n":    oos_s["n"],    "oos_total":  oos_s["total"],
            "oos_pf":   oos_s["pf"],   "oos_sharpe": oos_s["sharpe"],
            "oos_wr":   oos_s["wr"],   "oos_mean":   oos_s["mean"],
            "oos_long_total":  oos_s["long_total"],
            "oos_short_total": oos_s["short_total"],
        })
        if i % 50 == 0:
            print(f"  {i}/{len(combos)}  rows kept={len(rows)}")

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} configs produced output on both IS and OOS")

    # Filter to robust candidates: positive on both, OOS n>=15
    robust = df[(df["is_total"] > 0) & (df["oos_total"] > 0) & (df["oos_n"] >= 15) & (df["is_n"] >= 100)].copy()
    robust["min_pf"]     = robust[["is_pf", "oos_pf"]].min(axis=1)
    robust["min_sharpe"] = robust[["is_sharpe", "oos_sharpe"]].min(axis=1)
    robust["min_wr"]     = robust[["is_wr", "oos_wr"]].min(axis=1)

    # Sort by minimum Sharpe (worst-case Sharpe — most pessimistic robustness metric)
    robust = robust.sort_values("min_sharpe", ascending=False)

    print(f"\n{len(robust)} ROBUST configs (positive on both, OOS n>=15, IS n>=100)")
    if robust.empty:
        print("\nNo robust configs found. Strategy may be regime-dependent or overfit.")
        # Still show what's least bad in OOS
        print("\nTop 20 by OOS PF (regardless of IS performance):")
        df_sorted = df[df["oos_n"] >= 15].sort_values("oos_pf", ascending=False).head(20)
        cols = ["X","N","D","strict","is_n","is_total","is_pf","is_sharpe",
                "oos_n","oos_total","oos_pf","oos_sharpe","oos_long_total","oos_short_total"]
        print(df_sorted[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        cols = ["X","N","D","strict","is_n","is_total","is_pf","is_sharpe",
                "oos_n","oos_total","oos_pf","oos_sharpe","oos_long_total","oos_short_total","min_pf","min_sharpe"]
        print(robust[cols].head(30).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Save full output
    lines = []
    lines.append("=" * 200)
    lines.append("ROBUST CONFIGS — IS vs OOS comparison")
    lines.append(f"Locked: BAND_K=0.25, TP=SL=1.0, chained Mode 1")
    lines.append(f"Sweep: X x N x D x strict  ({len(combos)} configs)")
    lines.append(f"IS:  2020-12 -> 2024-12  ({len(is_df):,} trades)")
    lines.append(f"OOS: 2025-01 -> 2025-11  ({len(oos_df):,} trades)")
    lines.append("=" * 200)

    if not robust.empty:
        lines.append("")
        lines.append(f"ROBUST CONFIGS — positive total on both IS and OOS, OOS n>=15, IS n>=100")
        lines.append(f"Found: {len(robust)} configs")
        lines.append("=" * 200)
        cols = ["X","N","D","strict","is_n","is_total","is_pf","is_sharpe","is_wr",
                "oos_n","oos_total","oos_pf","oos_sharpe","oos_wr",
                "is_long_total","is_short_total","oos_long_total","oos_short_total",
                "min_pf","min_sharpe"]
        lines.append(robust[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        lines.append("")
        lines.append("NO robust configs found — top 30 by OOS PF (regardless of IS):")
        df_sorted = df[df["oos_n"] >= 15].sort_values("oos_pf", ascending=False).head(30)
        cols = ["X","N","D","strict","is_n","is_total","is_pf","is_sharpe",
                "oos_n","oos_total","oos_pf","oos_sharpe","oos_long_total","oos_short_total"]
        lines.append(df_sorted[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Also show all X=0.75 configs specifically
    lines.append("")
    lines.append("=" * 200)
    lines.append("All X=0.75 configs (for direct comparison)")
    lines.append("=" * 200)
    x075 = df[df["X"] == 0.75].sort_values("oos_total", ascending=False)
    cols = ["X","N","D","strict","is_n","is_total","is_pf","is_sharpe",
            "oos_n","oos_total","oos_pf","oos_sharpe","oos_long_total","oos_short_total"]
    lines.append(x075[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}")


if __name__ == "__main__":
    sys.exit(main())

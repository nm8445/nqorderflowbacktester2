"""Dump the top 25 robust configs (positive on both IS and OOS, total >=500)
to a clean text file in tradelogs/robust_configs/.

Sweep params (must match sweep_confirmation_absorption.py):
  Locked B2 base: X=0.75 N=15 D=70 strict BAND_K=0.25 TP=SL=1.0 chained Mode 1
  Confirmation: directional delta in HALF-of-candle zone
  Sweep: conf_N x conf_D = 4 x 8 = 32 combos
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_break_entry_summary import (
    apply_filters, mode1_chained_dedupe, trade_pnls_vectorized,
)

PARQUET_DIR  = Path(__file__).parent / "parquets"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
ROBUST_DIR   = TRADELOG_DIR / "robust_configs"
ROBUST_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT      = ROBUST_DIR / "top_25_robust_configs.txt"

TRADES_IS  = PARQUET_DIR / "entry_signal_trades.parquet"
TRADES_OOS = PARQUET_DIR / "entry_signal_trades_oos.parquet"

VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
TP_M, SL_M = 1.0, 1.0
CONF_NS = [5, 10, 15, 20]
CONF_DS = [0, 25, 50, 75, 100, 150, 200, 300]
ZONE = "half"


def evaluate(df: pd.DataFrame, conf_N: int, conf_D: int) -> dict | None:
    filt = apply_filters(df, VARIANT, X, N, D, STRICT, BAND_K)
    col = f"conf_delta_{ZONE}_w{conf_N}"
    if col not in filt.columns:
        return None
    if conf_D == 0:
        cf = filt[((filt["direction"]=="LONG")  & (filt[col].notna()) & (filt[col] >=  0)) |
                  ((filt["direction"]=="SHORT") & (filt[col].notna()) & (filt[col] <=  0))]
    else:
        cf = filt[((filt["direction"]=="LONG")  & (filt[col].notna()) & (filt[col] >=  conf_D)) |
                  ((filt["direction"]=="SHORT") & (filt[col].notna()) & (filt[col] <= -conf_D))]
    ded = mode1_chained_dedupe(cf, TP_M, SL_M)
    if len(ded) == 0:
        return None
    ded = ded.copy()
    ded["pnl"]  = trade_pnls_vectorized(ded, TP_M, SL_M)
    ded["date"] = pd.to_datetime(ded["date"]).dt.date
    pnl = ded["pnl"].values
    long_mask = (ded["direction"] == "LONG").values
    short_mask = (ded["direction"] == "SHORT").values
    wins = pnl > 0
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0)
    daily = pd.Series(pnl, index=ded["date"].values).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    long_pnl = pnl[long_mask]; short_pnl = pnl[short_mask]
    return {
        "n": len(ded), "n_long": int(long_mask.sum()), "n_short": int(short_mask.sum()),
        "total": pnl.sum(), "mean": pnl.mean(),
        "wr": wins.mean(),
        "wr_long":  (long_pnl  > 0).mean() if len(long_pnl)  else float("nan"),
        "wr_short": (short_pnl > 0).mean() if len(short_pnl) else float("nan"),
        "pf": pf, "sharpe": sharpe, "max_dd": max_dd,
        "long_total": long_pnl.sum(), "short_total": short_pnl.sum(),
    }


def main():
    print("loading trades...")
    is_df  = pd.read_parquet(TRADES_IS)
    oos_df = pd.read_parquet(TRADES_OOS)
    print(f"  IS:  {len(is_df):,}    OOS: {len(oos_df):,}")

    print("evaluating all (conf_N, conf_D) combos on both IS and OOS...")
    rows = []
    for conf_N, conf_D in product(CONF_NS, CONF_DS):
        is_s  = evaluate(is_df,  conf_N, conf_D)
        oos_s = evaluate(oos_df, conf_N, conf_D)
        if is_s is None or oos_s is None:
            continue
        rows.append({
            "conf_N": conf_N, "conf_D": conf_D,
            "is_n": is_s["n"], "is_total": is_s["total"], "is_pf": is_s["pf"],
            "is_sharpe": is_s["sharpe"], "is_wr": is_s["wr"], "is_mean": is_s["mean"],
            "is_max_dd": is_s["max_dd"],
            "is_long_total": is_s["long_total"], "is_short_total": is_s["short_total"],
            "is_n_long": is_s["n_long"], "is_n_short": is_s["n_short"],
            "oos_n": oos_s["n"], "oos_total": oos_s["total"], "oos_pf": oos_s["pf"],
            "oos_sharpe": oos_s["sharpe"], "oos_wr": oos_s["wr"], "oos_mean": oos_s["mean"],
            "oos_max_dd": oos_s["max_dd"],
            "oos_long_total": oos_s["long_total"], "oos_short_total": oos_s["short_total"],
            "oos_n_long": oos_s["n_long"], "oos_n_short": oos_s["n_short"],
        })

    df = pd.DataFrame(rows)
    df["total_trades"] = df["is_n"] + df["oos_n"]
    df["min_pf"]       = df[["is_pf", "oos_pf"]].min(axis=1)
    df["min_sharpe"]   = df[["is_sharpe", "oos_sharpe"]].min(axis=1)
    df["combined_total_pnl"] = df["is_total"] + df["oos_total"]

    # Robust = positive on both, total >=500
    robust = df[(df["is_total"] > 0) & (df["oos_total"] > 0) & (df["total_trades"] >= 500)].copy()
    print(f"  robust configs: {len(robust)} (positive on both, total>=500)")

    # If fewer than 25 robust, drop the trade-count threshold to fill 25
    if len(robust) < 25:
        relaxed = df[(df["is_total"] > 0) & (df["oos_total"] > 0)].copy()
        # Backfill from positive-both configs sorted by min_sharpe
        relaxed = relaxed.sort_values("min_sharpe", ascending=False)
        already = set(zip(robust["conf_N"], robust["conf_D"]))
        backfill = relaxed[~relaxed.set_index(["conf_N","conf_D"]).index.isin(already)]
        n_needed = 25 - len(robust)
        robust = pd.concat([robust, backfill.head(n_needed)], ignore_index=True)
        print(f"  backfilled {n_needed} from relaxed criteria (positive both, any size)")

    # Sort by min_sharpe (most pessimistic robustness metric) for ranking
    robust = robust.sort_values("min_sharpe", ascending=False).head(25).reset_index(drop=True)
    robust.insert(0, "rank", range(1, len(robust) + 1))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 200)
    lines.append("TOP 25 ROBUST CONFIGS — IS vs OOS")
    lines.append("=" * 200)
    lines.append("")
    lines.append("LOCKED BASE CONFIG (held constant for all 25 below):")
    lines.append(f"  variant      : {VARIANT}")
    lines.append(f"  pinbar X     : {X}            (signal candle wick prominence floor)")
    lines.append(f"  window N     : {N} ticks      (signal candle absorption window)")
    lines.append(f"  delta D      : {D}            (signal candle absorption threshold)")
    lines.append(f"  strict       : {STRICT}        (require SHORT close < OLO)")
    lines.append(f"  BAND_K       : {BAND_K}        (level proximity = clip(0.25 x ATR, 5, 20))")
    lines.append(f"  TP_M / SL_M  : {TP_M} / {SL_M}    (symmetric wick-anchored, R:R = 1:1)")
    lines.append(f"  dedupe       : chained Mode 1 (multiple non-overlapping trades per day)")
    lines.append("")
    lines.append("CONFIRMATION CANDLE FILTER (varies per config):")
    lines.append(f"  zone         : HALF-of-candle (below midpoint for LONG, above for SHORT)")
    lines.append(f"  conf_N       : window size in ticks (best-window scan)")
    lines.append(f"  conf_D       : directional delta threshold (LONG: delta>=+conf_D ; SHORT: delta<=-conf_D)")
    lines.append("")
    lines.append("DATA WINDOWS:")
    lines.append(f"  IS  : 2020-12 -> 2024-12   ({len(is_df):,} trade candidates)")
    lines.append(f"  OOS : 2025-01 -> 2026-05   ({len(oos_df):,} trade candidates)")
    lines.append("")
    lines.append("RANKING METRIC:")
    lines.append("  Sorted by min(IS_Sharpe, OOS_Sharpe) — most pessimistic robustness signal.")
    lines.append("  All entries have positive total in both IS and OOS.")
    lines.append("  Threshold 'robust' = total trades (IS+OOS) >= 500. If fewer than 25 meet that,")
    lines.append("  list is backfilled from positive-both configs regardless of size.")
    lines.append("")
    lines.append("=" * 200)
    lines.append("TOP 25 (sorted by min Sharpe descending)")
    lines.append("=" * 200)
    lines.append("")

    cols_main = ["rank", "conf_N", "conf_D", "total_trades",
                 "is_n", "is_pf", "is_sharpe", "is_wr", "is_total",
                 "oos_n", "oos_pf", "oos_sharpe", "oos_wr", "oos_total",
                 "min_pf", "min_sharpe"]
    lines.append(robust[cols_main].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Long/short breakdown
    lines.append("")
    lines.append("=" * 200)
    lines.append("LONG / SHORT direction breakdown per config")
    lines.append("=" * 200)
    cols_ls = ["rank", "conf_N", "conf_D",
               "is_n_long", "is_long_total", "is_n_short", "is_short_total",
               "oos_n_long", "oos_long_total", "oos_n_short", "oos_short_total"]
    lines.append(robust[cols_ls].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # Drawdowns
    lines.append("")
    lines.append("=" * 200)
    lines.append("DRAWDOWN per config (realized P/L max drawdown, NQ pts)")
    lines.append("=" * 200)
    cols_dd = ["rank", "conf_N", "conf_D",
               "is_max_dd", "oos_max_dd",
               "is_total", "oos_total", "combined_total_pnl"]
    lines.append(robust[cols_dd].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # Top 5 detailed view
    lines.append("")
    lines.append("=" * 200)
    lines.append("TOP 5 — DETAILED VIEW")
    lines.append("=" * 200)
    for i in range(min(5, len(robust))):
        r = robust.iloc[i]
        lines.append("")
        lines.append(f"#{int(r['rank'])}  conf_N={int(r['conf_N'])}  conf_D={int(r['conf_D'])}")
        lines.append("-" * 80)
        lines.append(f"  IS  : n={int(r['is_n']):>4}  L={int(r['is_n_long']):>3}/S={int(r['is_n_short']):>3}  "
                     f"total={r['is_total']:>+8.1f}  PF={r['is_pf']:.2f}  Sharpe={r['is_sharpe']:+.2f}  "
                     f"WR={r['is_wr']:.1%}  MDD={r['is_max_dd']:+.0f}")
        lines.append(f"        L_total={r['is_long_total']:>+7.1f}     S_total={r['is_short_total']:>+7.1f}")
        lines.append(f"  OOS : n={int(r['oos_n']):>4}  L={int(r['oos_n_long']):>3}/S={int(r['oos_n_short']):>3}  "
                     f"total={r['oos_total']:>+8.1f}  PF={r['oos_pf']:.2f}  Sharpe={r['oos_sharpe']:+.2f}  "
                     f"WR={r['oos_wr']:.1%}  MDD={r['oos_max_dd']:+.0f}")
        lines.append(f"        L_total={r['oos_long_total']:>+7.1f}     S_total={r['oos_short_total']:>+7.1f}")
        lines.append(f"  Combined total: {r['combined_total_pnl']:+.1f} pts over {int(r['total_trades'])} trades")

    lines.append("")
    lines.append("=" * 200)
    lines.append("USAGE NOTES")
    lines.append("=" * 200)
    lines.append("- All configs use the same locked base; only conf_N and conf_D vary.")
    lines.append("- 'min Sharpe' is the worst of IS/OOS — higher means more consistent across periods.")
    lines.append("- Configs with very low total_trades (<400) are noisier despite high Sharpe.")
    lines.append("- IS LONG vs SHORT total can differ from OOS LONG vs SHORT total — direction balance shifts by regime.")
    lines.append(f"- Generated for OOS window through 2026-05-06 (extended Phase 2).")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}")
    print(f"  ({OUT_TXT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    sys.exit(main())

"""Sweep confirmation-candle absorption thresholds for the X=0.75 N=15 D=70
strict candidate config. Tests directional delta in confirmation wick:
  LONG  passes if conf_delta_w{conf_N} >= +conf_D
  SHORT passes if conf_delta_w{conf_N} <= -conf_D

Reports configs that hold up in BOTH IS and OOS (positive total, OOS n>=15).
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

PARQUET_DIR = Path(__file__).parent / "parquets"
TRADES_IS   = PARQUET_DIR / "entry_signal_trades.parquet"
TRADES_OOS  = PARQUET_DIR / "entry_signal_trades_oos.parquet"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
OUT_TXT     = TRADELOG_DIR / "confirmation_absorption_sweep.txt"

# Candidate config (locked)
VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
TP_M, SL_M = 1.0, 1.0

# Sweep dimensions for confirmation candle
CONF_NS = [5, 10, 15, 20]
CONF_DS = [0, 25, 50, 75, 100, 150, 200, 300]
ZONE    = "half"   # "wick" or "half" — search area for absorption
MIN_TOTAL_TRADES = 500   # IS + OOS combined


def apply_conf_filter(df: pd.DataFrame, conf_N: int, conf_D: int) -> pd.DataFrame:
    """Filter trades requiring directional absorption in confirmation candle.
    Uses conf_delta_{ZONE}_w{conf_N} column.
    LONG  passes if delta >= +conf_D
    SHORT passes if delta <= -conf_D
    NaN delta -> trade fails (insufficient levels in zone)
    """
    col = f"conf_delta_{ZONE}_w{conf_N}"
    if col not in df.columns:
        return df.iloc[0:0]
    long_mask  = (df["direction"] == "LONG")  & (df[col].notna()) & (df[col] >=  conf_D)
    short_mask = (df["direction"] == "SHORT") & (df[col].notna()) & (df[col] <= -conf_D)
    return df[long_mask | short_mask].copy()


def evaluate(df: pd.DataFrame, conf_N: int, conf_D: int) -> dict | None:
    # Step 1: apply existing locked filter (X, N, D, strict, BAND_K)
    filtered = apply_filters(df, VARIANT, X, N, D, STRICT, BAND_K)
    if len(filtered) == 0:
        return None
    # Step 2: NEW confirmation-absorption filter
    conf_filtered = apply_conf_filter(filtered, conf_N, conf_D)
    if len(conf_filtered) == 0:
        return None
    # Step 3: chained Mode 1 dedupe
    deduped = mode1_chained_dedupe(conf_filtered, TP_M, SL_M)
    if len(deduped) == 0:
        return None
    deduped = deduped.copy()
    deduped["pnl"]  = trade_pnls_vectorized(deduped, TP_M, SL_M)
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
        "n": len(deduped),
        "n_long": int(long_mask.sum()), "n_short": int(short_mask.sum()),
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
    print(f"  IS:  {len(is_df):,}    OOS: {len(oos_df):,}")

    # Baseline (no confirmation filter) for reference
    is_filt = apply_filters(is_df, VARIANT, X, N, D, STRICT, BAND_K)
    oos_filt = apply_filters(oos_df, VARIANT, X, N, D, STRICT, BAND_K)
    print(f"  after locked filters — IS: {len(is_filt):,}  OOS: {len(oos_filt):,}")

    is_dedup_base = mode1_chained_dedupe(is_filt, TP_M, SL_M)
    oos_dedup_base = mode1_chained_dedupe(oos_filt, TP_M, SL_M)

    def base_stats(df):
        df = df.copy()
        df["pnl"] = trade_pnls_vectorized(df, TP_M, SL_M)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        pnl = df["pnl"].values
        wins = pnl > 0
        pos = pnl[pnl>0].sum(); neg = -pnl[pnl<0].sum()
        pf = pos/neg if neg>0 else (np.inf if pos>0 else 0)
        daily = pd.Series(pnl, index=df["date"].values).groupby(level=0).sum()
        sharpe = daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
        return dict(n=len(df), total=pnl.sum(), mean=pnl.mean(), wr=wins.mean(),
                    pf=pf, sharpe=sharpe)
    bs_is = base_stats(is_dedup_base)
    bs_oos = base_stats(oos_dedup_base)

    # Sweep
    combos = list(product(CONF_NS, CONF_DS))
    print(f"\nsweeping {len(combos)} (conf_N, conf_D) combos...")
    rows = []
    for conf_N, conf_D in combos:
        is_s = evaluate(is_df, conf_N, conf_D)
        oos_s = evaluate(oos_df, conf_N, conf_D)
        if is_s is None or oos_s is None:
            continue
        rows.append({
            "conf_N": conf_N, "conf_D": conf_D,
            "is_n": is_s["n"], "is_total": is_s["total"], "is_pf": is_s["pf"],
            "is_sharpe": is_s["sharpe"], "is_wr": is_s["wr"], "is_mean": is_s["mean"],
            "is_long_total": is_s["long_total"], "is_short_total": is_s["short_total"],
            "oos_n": oos_s["n"], "oos_total": oos_s["total"], "oos_pf": oos_s["pf"],
            "oos_sharpe": oos_s["sharpe"], "oos_wr": oos_s["wr"], "oos_mean": oos_s["mean"],
            "oos_long_total": oos_s["long_total"], "oos_short_total": oos_s["short_total"],
        })

    if not rows:
        print("no combos produced output on both")
        return

    df = pd.DataFrame(rows)
    df["total_trades"] = df["is_n"] + df["oos_n"]
    robust = df[(df["is_total"] > 0) & (df["oos_total"] > 0) &
                (df["total_trades"] >= MIN_TOTAL_TRADES)].copy()
    robust["min_pf"] = robust[["is_pf","oos_pf"]].min(axis=1)
    robust["min_sharpe"] = robust[["is_sharpe","oos_sharpe"]].min(axis=1)
    robust = robust.sort_values("min_sharpe", ascending=False)

    # Output
    lines = []
    lines.append("=" * 200)
    lines.append("CONFIRMATION-CANDLE ABSORPTION SWEEP")
    lines.append(f"Locked: B2 X={X} N={N} D={D} strict={STRICT} BAND_K={BAND_K} TP=SL={TP_M} chained Mode 1")
    lines.append(f"Confirmation filter zone: {ZONE.upper()}  "
                 f"({'bottom/top wick only' if ZONE=='wick' else 'bottom/top HALF of candle (regardless of wick)'})")
    lines.append(f"  LONG  passes if conf_delta_{ZONE}_w{{conf_N}} >= +conf_D")
    lines.append(f"  SHORT passes if conf_delta_{ZONE}_w{{conf_N}} <= -conf_D")
    lines.append(f"Sweep: conf_N x conf_D = {len(CONF_NS)} x {len(CONF_DS)} = {len(combos)} combos")
    lines.append(f"Robustness threshold: total trades (IS+OOS) >= {MIN_TOTAL_TRADES}")
    lines.append(f"IS  trades (B2): 5,828   OOS trades (B2): 1,305")
    lines.append("=" * 200)
    lines.append("")
    lines.append("BASELINE (no confirmation filter — current locked candidate):")
    lines.append(f"  IS  : n={bs_is['n']:>5}  total={bs_is['total']:>+8.1f}  PF={bs_is['pf']:.2f}  Sharpe={bs_is['sharpe']:+.2f}  WR={bs_is['wr']:.1%}")
    lines.append(f"  OOS : n={bs_oos['n']:>5}  total={bs_oos['total']:>+8.1f}  PF={bs_oos['pf']:.2f}  Sharpe={bs_oos['sharpe']:+.2f}  WR={bs_oos['wr']:.1%}")
    lines.append("")
    lines.append(f"ROBUST CONFIGS (positive on both, total trades >= {MIN_TOTAL_TRADES}): {len(robust)}")
    lines.append("=" * 200)
    if not robust.empty:
        cols = ["conf_N","conf_D","is_n","is_total","is_pf","is_sharpe","is_wr",
                "oos_n","oos_total","oos_pf","oos_sharpe","oos_wr","total_trades",
                "is_long_total","is_short_total","oos_long_total","oos_short_total",
                "min_pf","min_sharpe"]
        lines.append(robust[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        lines.append(f"NONE — no setting both positive in IS+OOS with >= {MIN_TOTAL_TRADES} total trades.")

    # Also show conf_N effect — best conf_D per conf_N
    lines.append("")
    lines.append("=" * 200)
    lines.append("WHICH conf_N matters most? Best (positive both, max total trades) per conf_N value:")
    lines.append("=" * 200)
    df["min_pf"] = df[["is_pf","oos_pf"]].min(axis=1)
    df["min_sharpe"] = df[["is_sharpe","oos_sharpe"]].min(axis=1)
    pos_both = df[(df["is_total"]>0) & (df["oos_total"]>0)].copy()
    if not pos_both.empty:
        for n in CONF_NS:
            sub = pos_both[pos_both["conf_N"]==n]
            if sub.empty:
                lines.append(f"  conf_N={n}: no positive-both configs")
                continue
            lines.append(f"  conf_N={n}:")
            top = sub.sort_values("total_trades", ascending=False).head(3)
            for _, row in top.iterrows():
                lines.append(f"    conf_D={int(row['conf_D']):>3}  "
                             f"IS n={int(row['is_n']):>4} PF={row['is_pf']:.2f} Sharpe={row['is_sharpe']:+.2f}    "
                             f"OOS n={int(row['oos_n']):>3} PF={row['oos_pf']:.2f} Sharpe={row['oos_sharpe']:+.2f}    "
                             f"total={int(row['total_trades']):>4}")

    lines.append("")
    lines.append("=" * 200)
    lines.append("ALL CONFIGS (sorted by min Sharpe across IS/OOS):")
    lines.append("=" * 200)
    df["min_sharpe"] = df[["is_sharpe","oos_sharpe"]].min(axis=1)
    df["min_pf"] = df[["is_pf","oos_pf"]].min(axis=1)
    cols = ["conf_N","conf_D","is_n","is_total","is_pf","is_sharpe",
            "oos_n","oos_total","oos_pf","oos_sharpe",
            "is_long_total","is_short_total","oos_long_total","oos_short_total",
            "min_pf","min_sharpe"]
    lines.append(df.sort_values("min_sharpe", ascending=False)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(lines[:30]))
    print("..." if len(lines) > 30 else "")
    print("\n".join(lines[-30:]))


if __name__ == "__main__":
    sys.exit(main())

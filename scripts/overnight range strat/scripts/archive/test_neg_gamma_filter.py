"""Test EOD multi-DTE NEG-gamma filter on the X=0.75 N=15 D=70 strict candidate.

Compares baseline (no regime filter) against:
  Filter F1: NEG-gamma only       (drop all POS-gamma trades)
  Filter F2: NEG-gamma + LONG only (drop POS gamma + all shorts)

Reports IS and OOS side-by-side.
"""

from __future__ import annotations

import sys
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
EOD_MQ      = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
TRADELOG_DIR.mkdir(exist_ok=True)
OUT_TXT = TRADELOG_DIR / "neg_gamma_filter_test.txt"

# Candidate config
VARIANT, X, N, D, STRICT, BAND_K = "B2", 0.75, 15, 70, True, 0.25
TP_M, SL_M = 1.0, 1.0


def load_eod_gamma_lookup():
    eod = pd.read_parquet(EOD_MQ)
    eod["date"] = pd.to_datetime(eod["date"]).dt.date
    eod = eod.set_index("date")
    eod_dates = sorted(eod.index.tolist())
    def prior_mq(d):
        prev = None
        for md in eod_dates:
            if md < d: prev = md
            else: break
        return prev
    return eod, prior_mq


def stratify(df: pd.DataFrame, label: str) -> dict:
    pnl = df["pnl"].values if "pnl" in df.columns else trade_pnls_vectorized(df, TP_M, SL_M)
    if "pnl" not in df.columns:
        df = df.copy(); df["pnl"] = pnl
    long_mask = (df["direction"] == "LONG").values
    short_mask = (df["direction"] == "SHORT").values
    wins = pnl > 0
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0)
    daily = pd.Series(pnl, index=df["date"].values).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(eq); max_dd = (eq - peak).min()
    long_pnl = pnl[long_mask]; short_pnl = pnl[short_mask]
    return {
        "label": label, "n": len(df),
        "n_long": int(long_mask.sum()), "n_short": int(short_mask.sum()),
        "total": pnl.sum(), "mean": pnl.mean(),
        "wr": wins.mean(),
        "wr_long":  (long_pnl  > 0).mean() if len(long_pnl)  else float("nan"),
        "wr_short": (short_pnl > 0).mean() if len(short_pnl) else float("nan"),
        "pf": pf, "sharpe": sharpe, "max_dd": max_dd,
        "long_total": long_pnl.sum(), "short_total": short_pnl.sum(),
    }


def fmt(s):
    return (f"{s['label']:<32}  n={s['n']:>4}  L={s['n_long']:>4}/S={s['n_short']:>4}  "
            f"total={s['total']:>+8.1f}  L_total={s['long_total']:>+7.1f}  S_total={s['short_total']:>+7.1f}  "
            f"WR={s['wr']:>5.1%}  WR_L={s['wr_long']:>5.1%}  WR_S={s['wr_short']:>5.1%}  "
            f"PF={s['pf']:>5.2f}  Sharpe={s['sharpe']:>+5.2f}  MDD={s['max_dd']:>+8.0f}")


def process(label_period: str, df: pd.DataFrame, eod_lookup, prior_mq, lines: list):
    filtered = apply_filters(df, VARIANT, X, N, D, STRICT, BAND_K)
    deduped  = mode1_chained_dedupe(filtered, TP_M, SL_M)
    deduped  = deduped.copy()
    deduped["pnl"]  = trade_pnls_vectorized(deduped, TP_M, SL_M)
    deduped["date"] = pd.to_datetime(deduped["date"]).dt.date

    # Lookup EOD multi-DTE gamma_sign per trade (prior-day)
    g_lookup = {}
    for d in deduped["date"].unique():
        p = prior_mq(d)
        if p is None: continue
        g_lookup[d] = eod_lookup.loc[p, "qqq_gamma_sign"] if "qqq_gamma_sign" in eod_lookup.columns else np.nan
    deduped["eod_gamma"] = deduped["date"].map(g_lookup)

    n_with = deduped["eod_gamma"].notna().sum()
    lines.append(f"  trades with EOD gamma: {n_with}/{len(deduped)} ({n_with/len(deduped)*100:.1f}%)")
    lines.append("")
    lines.append(f"{label_period} BASELINE (no filter)")
    lines.append("  " + fmt(stratify(deduped, "BASELINE")))

    # Filter F1: NEG-gamma only (drop POS)
    f1 = deduped[deduped["eod_gamma"] == -1]
    lines.append("")
    lines.append(f"{label_period} F1: NEG-gamma only (drop POS-gamma trades)")
    lines.append("  " + fmt(stratify(f1, "F1 NEG-only")))
    lines.append("    (dropped: " + fmt(stratify(deduped[deduped['eod_gamma']==1], 'POS-gamma dropped')) + ")")

    # Filter F2: NEG-gamma + LONG only
    f2 = deduped[(deduped["eod_gamma"] == -1) & (deduped["direction"] == "LONG")]
    lines.append("")
    lines.append(f"{label_period} F2: NEG-gamma + LONG only")
    lines.append("  " + fmt(stratify(f2, "F2 NEG+LONG")))

    # Filter F3: drop only POS+SHORT (keep POS+LONG, NEG+anything)
    f3 = deduped[~((deduped["eod_gamma"] == 1) & (deduped["direction"] == "SHORT"))]
    lines.append("")
    lines.append(f"{label_period} F3: drop only POS-gamma SHORTs (most surgical filter)")
    lines.append("  " + fmt(stratify(f3, "F3 drop POS+SHORT")))

    return deduped


def main():
    print("loading trades...")
    is_df  = pd.read_parquet(TRADES_IS)
    oos_df = pd.read_parquet(TRADES_OOS)
    print(f"  IS:  {len(is_df):,}    OOS: {len(oos_df):,}")

    print("loading EOD multi-DTE gamma_sign lookup...")
    eod, prior_mq = load_eod_gamma_lookup()

    lines = []
    lines.append("=" * 200)
    lines.append("EOD NEG-GAMMA FILTER TEST  on candidate config:")
    lines.append(f"  B2 X={X} N={N} D={D} strict={STRICT} BAND_K={BAND_K} TP=SL={TP_M}  (chained Mode 1)")
    lines.append("=" * 200)
    lines.append("")
    lines.append("=" * 200)
    lines.append("IN-SAMPLE (2020-12 -> 2024-12)")
    lines.append("=" * 200)
    process("IS", is_df, eod, prior_mq, lines)

    lines.append("")
    lines.append("=" * 200)
    lines.append("OUT-OF-SAMPLE (2025-01 -> 2025-11)")
    lines.append("=" * 200)
    process("OOS", oos_df, eod, prior_mq, lines)

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())

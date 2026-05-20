"""POC scenario study extended with gamma regime at the open.

Stacks two filters:
  1. POC trend pattern (2-day rose/fell, 3-day rose/fell, inverted-V peak, V trough)
     + open relative to POC(D-1)
  2. Gamma regime at 9:30 ET — open in pos-gamma vs neg-gamma
     (using both QQQ-derived and NDX-derived classifications)

Output:
  - Cohort table per scenario × gamma regime: n, P(>0), mean ret (NQ pts),
    one-sample t-stat against 0, p-value
  - Highlights cells with p < 0.05 (significant) and p < 0.01 (strong)

Inputs:
  - POC per-day parquet:    .../poc_per_day.parquet
  - Intraday gamma parquet: D:/trading_pythonbacktest_data/QQQ_thetadata/
                              study_hvl0dte_intraday_regime.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

POC_PATH    = Path(__file__).parent / "poc_per_day.parquet"
GAMMA_PATH  = Path("D:/trading_pythonbacktest_data/QQQ_thetadata/study_hvl0dte_intraday_regime.parquet")


def load() -> pd.DataFrame:
    poc = pd.read_parquet(POC_PATH)
    gamma = pd.read_parquet(GAMMA_PATH)
    poc["date"] = pd.to_datetime(poc["date"])
    gamma["date"] = pd.to_datetime(gamma["date"])
    df = poc.merge(
        gamma[["date","qqq_regime_open","ndx_regime_open"]],
        on="date", how="inner",
    )
    return df


def cohort_stats(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n < 2:
        return {"n": n, "p_pos": np.nan, "mean": np.nan, "t": np.nan, "p": np.nan}
    rets = sub["day_ret_pts"].dropna().values
    n = len(rets)
    if n < 2:
        return {"n": n, "p_pos": np.nan, "mean": np.nan, "t": np.nan, "p": np.nan}
    p_pos = float((rets > 0).mean())
    mean = float(rets.mean())
    t, p = ss.ttest_1samp(rets, 0)
    return {"n": n, "p_pos": p_pos, "mean": mean, "t": float(t), "p": float(p)}


def fmt_row(label: str, st: dict) -> str:
    if not np.isfinite(st["mean"]):
        return f"  {label:<60}  n={st['n']:>4}  insufficient"
    sig = "  ***" if st["p"] < 0.001 else ("   **" if st["p"] < 0.01 else
          ("    *" if st["p"] < 0.05 else "     "))
    return (f"  {label:<60}  n={st['n']:>4}  P(>0)={st['p_pos']:.1%}  "
            f"mean={st['mean']:+7.2f} pts  t={st['t']:+5.2f}  p={st['p']:.4f}{sig}")


def trend_filters(df: pd.DataFrame):
    """Return list of (trend_label, base_filter_mask, [(open_cond_label, open_cond_mask)])"""
    poc1 = df["poc_d1"]; poc2 = df["poc_d2"]; poc3 = df["poc_d3"]
    op = df["open_930"]
    out = []

    # 2-day monotonic
    out.append(("2-day rose  (D-1 > D-2)",     poc1 > poc2,
                [("+ open >  POC(D-1)", op > poc1),
                 ("+ open <= POC(D-1)", op <= poc1)]))
    out.append(("2-day fell  (D-1 < D-2)",     poc1 < poc2,
                [("+ open <  POC(D-1)", op < poc1),
                 ("+ open >= POC(D-1)", op >= poc1)]))

    # 3-day monotonic
    out.append(("3-day rose  (D-1 > D-2 > D-3)", (poc1 > poc2) & (poc2 > poc3),
                [("+ open >  POC(D-1)", op > poc1),
                 ("+ open <= POC(D-1)", op <= poc1)]))
    out.append(("3-day fell  (D-1 < D-2 < D-3)", (poc1 < poc2) & (poc2 < poc3),
                [("+ open <  POC(D-1)", op < poc1),
                 ("+ open >= POC(D-1)", op >= poc1)]))

    # 3-day reversal
    out.append(("invV peak   (D-1 < D-2 > D-3)", (poc1 < poc2) & (poc2 > poc3),
                [("+ open >  POC(D-1)", op > poc1),
                 ("+ open <  POC(D-1)", op < poc1)]))
    out.append(("V  trough   (D-1 > D-2 < D-3)", (poc1 > poc2) & (poc2 < poc3),
                [("+ open <  POC(D-1)", op < poc1),
                 ("+ open >  POC(D-1)", op > poc1)]))

    return out


def main():
    df = load()
    valid = df.dropna(subset=["poc_d1","poc_d2","poc_d3"]).copy()
    print(f"loaded merged sample: {len(valid)} days "
          f"({valid['date'].min().date()} -> {valid['date'].max().date()})\n")

    # Baseline
    base = cohort_stats(valid)
    print(f"BASELINE  (all valid days):  {fmt_row('', base)}\n")

    print("Significance markers: * p<0.05, ** p<0.01, *** p<0.001\n")

    for source, regime_col in [("QQQ", "qqq_regime_open"),
                                ("NDX", "ndx_regime_open")]:
        print("=" * 100)
        print(f"GAMMA SOURCE: {source} (column: {regime_col})")
        print("=" * 100)

        for trend_label, trend_mask, open_conds in trend_filters(df):
            print(f"\n--- {trend_label} ---")
            base_trend = valid[trend_mask & valid[regime_col].notna()]
            print(fmt_row(f"  ALL  {trend_label}", cohort_stats(base_trend)))
            for open_label, open_mask in open_conds:
                cohort = valid[trend_mask & open_mask & valid[regime_col].notna()]
                print(fmt_row(f"  {open_label}  ALL gamma", cohort_stats(cohort)))
                for regime in ["pos", "neg"]:
                    sub = cohort[cohort[regime_col] == regime]
                    print(fmt_row(f"    {open_label} + {source}_{regime}-gamma",
                                  cohort_stats(sub)))


if __name__ == "__main__":
    sys.exit(main())

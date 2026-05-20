"""Analyze the saved HVL 0DTE mean reversion study parquet, with 4-cohort regime breakdown.

Regimes (each computed for QQQ and NDX 0-1 DTE chain):
  - above_flip   : flip exists, spot above HVL flip strike (positive-gamma regime)
  - below_flip   : flip exists, spot below HVL flip strike (negative-gamma regime)
  - deep_pos     : no flip, entire ±5%-of-spot band has positive cumulative GEX (deep positive gamma)
  - deep_neg     : no flip, entire ±5% band has negative cumulative GEX (deep negative gamma)
  - mixed_no_flip: edge case (rare)
  - no_data      : no chain available
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

PATH = Path("D:/trading_pythonbacktest_data/QQQ_thetadata/study_hvl0dte_meanrev.parquet")

df = pd.read_parquet(PATH)
df["abs_ret"] = df["ret"].abs()
print(f"loaded {len(df)} rows from {PATH}")
print(f"date range: {df['date'].min()} -> {df['date'].max()}\n")


def report(label, sub: pd.DataFrame):
    n = len(sub)
    if n == 0:
        print(f"  {label:<35} n=0"); return
    p_pos = (sub["ret"] > 0).mean()
    m = sub["ret"].mean(); md = sub["ret"].median(); std = sub["ret"].std()
    absm = sub["abs_ret"].mean(); absmd = sub["abs_ret"].median()
    t = m / (std / np.sqrt(n)) if std else float("nan")
    print(f"  {label:<35} n={n:>5}  P(>0)={p_pos:.2%}  "
          f"mean={m:+.3%}  med={md:+.3%}  |ret|_mean={absm:.3%}  |ret|_med={absmd:.3%}  t={t:+.2f}")


print("=== Headline ===")
report("BASE  (all valid days)", df)

REGIMES = ["above_flip", "below_flip", "deep_pos", "deep_neg", "mixed_no_flip", "no_data"]

for label_prefix, regime_col in [("QQQ", "qqq_regime"), ("NDX", "ndx_regime")]:
    print(f"\n=== {label_prefix}-derived regime breakdown ===")
    counts = df[regime_col].value_counts().reindex(REGIMES, fill_value=0)
    print(f"  Counts: {dict(counts)}")
    print()

    for r in REGIMES:
        sub = df[df[regime_col] == r]
        if len(sub) == 0: continue
        report(r, sub)

    # Mean-reversion-relevant comparisons
    print("\n  ----  Pin/mean-revert candidates (positive gamma)  ----")
    pos_pin = df[df[regime_col].isin(["above_flip", "deep_pos"])]
    print(f"  POS gamma combined  (above_flip + deep_pos)")
    report("    -> pos_gamma combined", pos_pin)

    print("  ----  Trend candidates (negative gamma)  ----")
    neg_trend = df[df[regime_col].isin(["below_flip", "deep_neg"])]
    print(f"  NEG gamma combined  (below_flip + deep_neg)")
    report("    -> neg_gamma combined", neg_trend)

    # |ret| t-test pos vs neg
    a = pos_pin["abs_ret"].dropna()
    b = neg_trend["abs_ret"].dropna()
    if len(a) > 1 and len(b) > 1:
        t, p = ss.ttest_ind(a, b, equal_var=False)
        print(f"\n  |ret| pos vs neg gamma:  pos={a.mean():.3%}  neg={b.mean():.3%}  "
              f"diff={a.mean()-b.mean():+.3%}  t={t:+.2f}  p={p:.4f}")

    # close>open hit-rate: pos vs neg
    if len(a) > 1 and len(b) > 1:
        from scipy.stats import chi2_contingency
        table = pd.crosstab(
            df[regime_col].isin(["above_flip","deep_pos"]).rename("is_pos_gamma"),
            (df["ret"] > 0).rename("close_above_open"),
        )
        chi2, pchi, _, _ = chi2_contingency(table.values)
        print(f"  P(close>open) chi2 (pos vs neg gamma): chi2={chi2:.2f}  p={pchi:.4f}")

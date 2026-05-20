"""Analyze the intraday-regime tracking parquet."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

PATH = Path("D:/trading_pythonbacktest_data/QQQ_thetadata/study_hvl0dte_intraday_regime.parquet")

df = pd.read_parquet(PATH)
print(f"loaded {len(df)} rows from {PATH}")
print(f"date range: {df['date'].min()} -> {df['date'].max()}\n")


def report(label, sub: pd.DataFrame):
    n = len(sub)
    if n == 0:
        print(f"  {label:<40} n=0"); return
    p_pos = (sub["ret"] > 0).mean()
    m = sub["ret"].mean(); std = sub["ret"].std()
    absm = sub["abs_ret"].mean(); absmd = sub["abs_ret"].median()
    t = m / (std/np.sqrt(n)) if std else float("nan")
    print(f"  {label:<40} n={n:>5}  P(>0)={p_pos:.2%}  "
          f"mean_ret={m:+.3%}  |ret|_mean={absm:.3%}  |ret|_med={absmd:.3%}  t={t:+.2f}")


print("=== Headline ===")
report("BASE  (all valid days)", df)

for src, prefix in [("QQQ", "qqq"), ("NDX", "ndx")]:
    pct_pos_col = f"{prefix}_pct_pos"
    flips_col   = f"{prefix}_n_flips"
    open_col    = f"{prefix}_regime_open"
    close_col   = f"{prefix}_regime_close"
    if pct_pos_col not in df.columns: continue
    print(f"\n=== {src}-derived intraday-regime cohorts ===")
    valid = df[pct_pos_col].notna()
    df_v = df[valid].copy()
    print(f"  valid sample: {len(df_v)}")

    # Cohorts based on time-in-pos-gamma
    pure_pin   = df_v[df_v[pct_pos_col] >= 0.95]
    mostly_pin = df_v[(df_v[pct_pos_col] >= 0.50) & (df_v[pct_pos_col] < 0.95)]
    mostly_vol = df_v[(df_v[pct_pos_col] >= 0.05) & (df_v[pct_pos_col] < 0.50)]
    pure_vol   = df_v[df_v[pct_pos_col] < 0.05]

    print("\n--- Time-in-pin cohorts ---")
    report("Pure pin    (>=95% pos-gamma)",  pure_pin)
    report("Mostly pin  (50-95%)",            mostly_pin)
    report("Mostly vol  (5-50%)",             mostly_vol)
    report("Pure vol    (<5% pos-gamma)",     pure_vol)

    # Transition vs no-transition
    no_flip = df_v[df_v[flips_col] == 0]
    one_flip = df_v[df_v[flips_col] == 1]
    multi_flip = df_v[df_v[flips_col] >= 2]
    print("\n--- By number of regime flips during day ---")
    report("0 flips (regime stayed same)",     no_flip)
    report("1 flip  (single transition)",       one_flip)
    report(">=2 flips (choppy)",                multi_flip)

    # Open vs close regime — does opening regime persist?
    same_open_close = df_v[df_v[open_col] == df_v[close_col]]
    diff_open_close = df_v[df_v[open_col] != df_v[close_col]]
    print("\n--- Open regime vs close regime ---")
    report("Open == Close (regime persisted)", same_open_close)
    report("Open != Close (regime flipped)",   diff_open_close)

    # Stats: how often does regime persist all day?
    persistence = (df_v[open_col] == df_v[close_col]).mean()
    print(f"\n  Open->Close persistence: {persistence:.2%} of days")
    print(f"  Mean # flips per day: {df_v[flips_col].mean():.2f}")
    print(f"  Median flips: {df_v[flips_col].median():.0f}")

    # Pure pin vs Pure vol |ret| t-test
    if len(pure_pin) > 1 and len(pure_vol) > 1:
        t, p = ss.ttest_ind(pure_pin["abs_ret"], pure_vol["abs_ret"], equal_var=False)
        print(f"\n  |ret| Pure pin vs Pure vol: pin={pure_pin['abs_ret'].mean():.3%}  "
              f"vol={pure_vol['abs_ret'].mean():.3%}  t={t:+.2f}  p={p:.4f}")

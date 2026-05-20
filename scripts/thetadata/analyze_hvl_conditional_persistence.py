"""Conditional regime-persistence analysis.

Question: if you open in regime X AND are still in regime X at time T,
what's the probability you stay in regime X through the close?

Answers the live-trading question: "the longer I see this regime persist,
the more confident I should be it'll last to the close."
"""

from pathlib import Path
import numpy as np
import pandas as pd

PATH = Path("D:/trading_pythonbacktest_data/QQQ_thetadata/study_hvl0dte_intraday_regime.parquet")

df = pd.read_parquet(PATH)
print(f"loaded {len(df)} rows from {PATH}\n")

CHECKPOINTS = ["regime_open", "regime_at_10", "regime_at_11",
               "regime_at_12", "regime_at_13", "regime_at_14",
               "regime_at_15", "regime_close"]
CP_LABELS  =  ["09:30",        "10:00",         "11:00",
               "12:00",        "13:00",         "14:00",
               "15:00",        "17:00 (close)"]


def conditional_persistence(df: pd.DataFrame, prefix: str, regime: str):
    """For each checkpoint T, compute:
       P(regime_close == regime | regime at all checkpoints <= T == regime)"""
    cols = [f"{prefix}_{c}" for c in CHECKPOINTS]
    valid = df.dropna(subset=cols).copy()
    print(f"{prefix.upper()} — Open in {regime}-gamma:  ", end="")
    print(f"valid sample after dropping missing checkpoints = {len(valid)}")

    # Filter to those that opened in target regime
    open_match = valid[valid[f"{prefix}_regime_open"] == regime]
    n_open = len(open_match)
    if n_open == 0:
        print(f"  no rows opened in {regime}"); return

    print(f"  total opened in {regime}: {n_open}")
    print(f"  {'Time':<14} {'Cum_in_regime':<15} {'P(close == open regime)':<25} {'n':<6}")
    print(f"  {'-'*14} {'-'*15} {'-'*25} {'-'*6}")

    # For each checkpoint, filter to days that have stayed in regime through that checkpoint
    cumulative_filter = pd.Series([True]*len(open_match), index=open_match.index)
    for col, label in zip(cols, CP_LABELS):
        # Update the cumulative filter to require this checkpoint also matches
        cumulative_filter &= (open_match[col] == regime)
        sub = open_match[cumulative_filter]
        n = len(sub)
        if n == 0:
            print(f"  {label:<14} {'(empty)':<15}")
            break
        # P(close == regime) given we've been in regime through this checkpoint
        p_close = (sub[f"{prefix}_regime_close"] == regime).mean()
        # And the % of days that GOT to this checkpoint without flipping
        cumulative_pct = n / n_open
        print(f"  {label:<14} {cumulative_pct:>7.1%} ({n:>4})  "
              f"{p_close:>19.1%}     n={n:>5}")
    print()


for prefix in ["qqq", "ndx"]:
    print("=" * 70)
    print(f"{prefix.upper()}-derived: Conditional persistence by checkpoint")
    print("=" * 70)
    for regime in ["pos", "neg"]:
        conditional_persistence(df, prefix, regime)

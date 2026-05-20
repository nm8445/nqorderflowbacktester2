"""Test: does negative total net GEX on day D predict positive NQ close-to-close
return from D to D+1?

Total net GEX = sum over all strikes & expirations (<=45 DTE) of:
    gamma * OI * 100 * spot^2,    with puts negated

Negative => dealers net short gamma => reflexive/volatile regime.
Positive => dealers net long gamma  => mean-reverting/pinned.

Output: full sample, hit rates, quintile buckets, t-stats, and writes a parquet
with date / gex / return columns for further work.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("D:/trading_pythonbacktest_data/QQQ_thetadata")
LEVELS = ROOT / "nq_levels.parquet"
OUT = ROOT / "gex_return_study.parquet"
MAX_DTE = 45


def total_net_gex(date: dt.date) -> float | None:
    g_path = ROOT / date.isoformat() / "greeks_eod.parquet"
    o_path = ROOT / date.isoformat() / "open_interest.parquet"
    if not g_path.exists() or not o_path.exists():
        return None
    g = pd.read_parquet(g_path)
    o = pd.read_parquet(o_path)
    g["expiration"] = pd.to_datetime(g["expiration"])
    o["expiration"] = pd.to_datetime(o["expiration"])
    g["dte"] = (g["expiration"] - pd.Timestamp(date)).dt.days
    g = g[(g["dte"] > 0) & (g["dte"] <= MAX_DTE)]
    if g.empty:
        return None
    chain = g.merge(o[["strike", "right", "expiration", "open_interest"]],
                    on=["strike", "right", "expiration"], how="left")
    chain["signed_gex"] = chain["gamma"] * chain["open_interest"].fillna(0) * 100
    chain.loc[chain["right"].str.upper() == "PUT", "signed_gex"] *= -1
    spot = chain["underlying_price"].iloc[0]
    chain["signed_gex"] *= spot ** 2
    return float(chain["signed_gex"].sum())


def main():
    levels = pd.read_parquet(LEVELS)[["date", "nq_spot"]].copy()
    levels["date"] = pd.to_datetime(levels["date"]).dt.date
    levels = levels.sort_values("date").reset_index(drop=True)

    print(f"computing total net GEX for {len(levels)} days...")
    gexes = []
    for i, d in enumerate(levels["date"]):
        gexes.append(total_net_gex(d))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(levels)}")
    levels["total_net_gex"] = gexes

    # Forward NQ close-to-close return: from D close to D+1 close
    levels["nq_close_next"] = levels["nq_spot"].shift(-1)
    levels["ret_next"] = (levels["nq_close_next"] - levels["nq_spot"]) / levels["nq_spot"]

    # Sample = days with both GEX and forward return
    df = levels.dropna(subset=["total_net_gex", "ret_next"]).copy()
    print(f"\nsample with valid GEX + next-day NQ return: {len(df)}")
    print(f"  GEX range: {df['total_net_gex'].min():.2e} to {df['total_net_gex'].max():.2e}")
    print(f"  return mean: {df['ret_next'].mean():.4%}  median: {df['ret_next'].median():.4%}")

    # ---- 1. Hit rate: P(ret_next > 0 | gex < 0)
    print("\n=== Hit rate analysis ===")
    base_pos = (df["ret_next"] > 0).mean()
    base_mean = df["ret_next"].mean()
    print(f"Base rate (all days):    P(ret>0)={base_pos:.2%}   "
          f"mean ret={base_mean:.4%}   n={len(df)}")

    neg_gex = df[df["total_net_gex"] < 0]
    pos_gex = df[df["total_net_gex"] > 0]
    if len(neg_gex):
        p_neg = (neg_gex["ret_next"] > 0).mean()
        m_neg = neg_gex["ret_next"].mean()
        print(f"GEX<0 days:              P(ret>0)={p_neg:.2%}   "
              f"mean ret={m_neg:.4%}   n={len(neg_gex)}")
    if len(pos_gex):
        p_pos = (pos_gex["ret_next"] > 0).mean()
        m_pos = pos_gex["ret_next"].mean()
        print(f"GEX>0 days:              P(ret>0)={p_pos:.2%}   "
              f"mean ret={m_pos:.4%}   n={len(pos_gex)}")

    # t-test: are next-day returns different between neg-GEX and pos-GEX days?
    if len(neg_gex) > 1 and len(pos_gex) > 1:
        t, p = stats.ttest_ind(neg_gex["ret_next"], pos_gex["ret_next"],
                               equal_var=False)
        print(f"\nWelch t-test (neg vs pos GEX next-day returns): t={t:.3f}  p={p:.4f}")

    # ---- 2. Quintile buckets
    print("\n=== Quintile buckets (Q1=most negative GEX, Q5=most positive) ===")
    df["q"] = pd.qcut(df["total_net_gex"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    by_q = df.groupby("q", observed=True)["ret_next"].agg(
        ["count", "mean", "median", "std",
         lambda s: (s > 0).mean()]).rename(columns={"<lambda_0>": "p_pos"})
    by_q["mean_ann"] = by_q["mean"] * 252
    by_q["t_vs_zero"] = by_q.apply(
        lambda r: r["mean"] / (r["std"] / np.sqrt(r["count"])) if r["std"] else np.nan,
        axis=1)
    print(by_q.to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- 3. Sign-only contingency table
    print("\n=== Contingency: GEX sign vs next-day return sign ===")
    df["gex_sign"] = np.where(df["total_net_gex"] < 0, "neg", "pos")
    df["ret_sign"] = np.where(df["ret_next"] > 0, "up", "down")
    table = pd.crosstab(df["gex_sign"], df["ret_sign"], margins=True)
    print(table)
    chi2, p, _, _ = stats.chi2_contingency(table.iloc[:2, :2])
    print(f"chi2={chi2:.3f}  p={p:.4f}")

    # ---- 4. Save
    out = df[["date", "nq_spot", "total_net_gex", "nq_close_next", "ret_next", "q"]]
    out.to_parquet(OUT, compression="zstd", index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

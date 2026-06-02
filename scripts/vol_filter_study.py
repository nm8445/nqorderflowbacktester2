"""Vol filter study: does QQQ IV regime predict per-strategy performance?

Computes from menthorq_levels_nq.parquet:
  - IV rank (percentile vs trailing 252 days)
  - IV change (day-over-day pct)
  - 20-day realized vol on NQ settle
  - VRP (IV - realized vol)

For each strategy (OD, B2, RV), buckets trades by each vol metric and reports
WR, net $, PF, MDD per bucket. Splits IS/OOS to validate robustness.

Goal: find a filter (skip trades in bucket X) that meaningfully improves
risk-adjusted return on IS AND OOS.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

TRADES_CSV = "live/combined deployment plan/combined_trades_with_mae.csv"
GAMMA_PATH = "D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet"

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)


def load_data():
    trades = pd.read_csv(TRADES_CSV)
    trades["date"] = pd.to_datetime(trades["date"])

    gamma = pd.read_parquet(GAMMA_PATH)
    gamma["date"] = pd.to_datetime(gamma["date"])
    gamma = gamma[["date", "qqq_iv", "ndx_iv", "nq_settle", "qqq_gamma_sign"]].sort_values("date").reset_index(drop=True)

    # Compute vol features (use prior-day values to avoid lookahead — trades reference yesterday's settle data)
    gamma["iv_rank_252"]  = gamma["qqq_iv"].rolling(252).rank(pct=True) * 100
    gamma["iv_chg"]       = gamma["qqq_iv"].pct_change() * 100
    gamma["nq_ret"]       = gamma["nq_settle"].pct_change()
    gamma["rv_20d"]       = gamma["nq_ret"].rolling(20).std() * np.sqrt(252) * 100   # in % annualized
    gamma["vrp"]          = (gamma["qqq_iv"] * 100) - gamma["rv_20d"]                # both in %

    # Shift to prior day to avoid lookahead bias for trade-day filtering
    for c in ["iv_rank_252", "iv_chg", "rv_20d", "vrp", "qqq_iv"]:
        gamma[f"prev_{c}"] = gamma[c].shift(1)

    return trades, gamma


def merge_and_split(trades, gamma):
    cols = ["date", "prev_iv_rank_252", "prev_iv_chg", "prev_rv_20d", "prev_vrp", "prev_qqq_iv"]
    df = trades.merge(gamma[cols], on="date", how="left")
    df = df.dropna(subset=["prev_iv_rank_252", "prev_iv_chg", "prev_vrp"])

    # IS/OOS split (60/40 chronological)
    df = df.sort_values("date").reset_index(drop=True)
    dates = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * 0.6)]
    df["phase"] = np.where(df["date"] < cutoff, "IS", "OOS")
    return df


def stats(g: pd.DataFrame) -> dict:
    p = g["pnl_$"].values
    n = len(p)
    if n == 0: return dict(n=0, wr=0, net=0, pf=0)
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum() / abs(l.sum()) if len(l) > 0 else 99.0
    return dict(n=n, wr=round((p > 0).mean()*100, 1), net=round(p.sum(), 0), pf=round(pf, 3))


def bucket_analyze(df, strat, feature, buckets):
    """For one strategy, bucket trades by feature and report IS/OOS stats."""
    sub = df[df["strat"] == strat].copy()
    if len(sub) == 0:
        return None
    sub["bucket"] = pd.cut(sub[feature], bins=buckets, include_lowest=True)
    rows = []
    for bk, g in sub.groupby("bucket", observed=True):
        all_s = stats(g)
        is_s  = stats(g[g["phase"] == "IS"])
        oos_s = stats(g[g["phase"] == "OOS"])
        rows.append({
            "bucket": str(bk),
            "n_all": all_s["n"], "wr_all": all_s["wr"], "net_all": all_s["net"], "pf_all": all_s["pf"],
            "n_is": is_s["n"], "net_is": is_s["net"], "pf_is": is_s["pf"],
            "n_oos": oos_s["n"], "net_oos": oos_s["net"], "pf_oos": oos_s["pf"],
        })
    return pd.DataFrame(rows)


def main():
    trades, gamma = load_data()
    print(f"Loaded {len(trades)} trades, {len(gamma)} gamma rows")

    df = merge_and_split(trades, gamma)
    print(f"After merge + lookahead-safe filter: {len(df)} trades ({df['phase'].value_counts().to_dict()})")
    print(f"Strategies: {df['strat'].value_counts().to_dict()}")

    # Show IV rank distribution
    print(f"\nIV rank summary: mean={df['prev_iv_rank_252'].mean():.1f}, "
          f"median={df['prev_iv_rank_252'].median():.1f}, "
          f"std={df['prev_iv_rank_252'].std():.1f}")

    iv_rank_bins = [0, 20, 40, 60, 80, 100]
    iv_chg_bins  = [-100, -10, -3, 3, 10, 100]    # daily pct change buckets
    vrp_bins     = [-100, -3, 0, 3, 6, 100]       # IV - RV
    iv_abs_bins  = [0, 0.10, 0.14, 0.18, 0.25, 1.0]  # raw IV levels

    feature_setups = [
        ("prev_iv_rank_252", iv_rank_bins, "IV RANK (252d percentile)"),
        ("prev_iv_chg",      iv_chg_bins,  "IV CHANGE day-over-day %"),
        ("prev_vrp",         vrp_bins,     "VRP (IV - RV20d in %)"),
        ("prev_qqq_iv",      iv_abs_bins,  "ABSOLUTE QQQ IV"),
    ]

    for strat in ["OD", "B2", "RV"]:
        print(f"\n{'='*100}")
        print(f"  STRATEGY: {strat}")
        print('='*100)
        for feature, bins, label in feature_setups:
            print(f"\n--- {label} ---")
            res = bucket_analyze(df, strat, feature, bins)
            if res is None or res.empty:
                print("  (no data)")
                continue
            print(res.to_string(index=False))

    # Look for STRONG SKIPS — buckets where BOTH IS and OOS are very negative or very weak
    print(f"\n{'='*100}")
    print("  CANDIDATE FILTERS (buckets to SKIP — net negative or PF <1 in BOTH IS+OOS)")
    print('='*100)
    candidates = []
    for strat in ["OD", "B2", "RV"]:
        for feature, bins, label in feature_setups:
            res = bucket_analyze(df, strat, feature, bins)
            if res is None: continue
            for _, r in res.iterrows():
                if r["n_is"] >= 30 and r["n_oos"] >= 20 and r["net_is"] < 0 and r["net_oos"] < 0:
                    candidates.append({
                        "strat": strat, "feature": label, "bucket": r["bucket"],
                        "n_is": r["n_is"], "net_is": r["net_is"], "pf_is": r["pf_is"],
                        "n_oos": r["n_oos"], "net_oos": r["net_oos"], "pf_oos": r["pf_oos"],
                    })
    if candidates:
        print(pd.DataFrame(candidates).to_string(index=False))
    else:
        print("  No clean skip buckets found (no bucket negative in BOTH IS+OOS with sufficient n).")

    # Look for STRONG KEEP — buckets where BOTH IS and OOS are clearly positive/high-PF
    print(f"\n{'='*100}")
    print("  STRONG-EDGE BUCKETS (PF > 1.4 in BOTH IS+OOS, n>=30 each phase)")
    print('='*100)
    edges = []
    for strat in ["OD", "B2", "RV"]:
        for feature, bins, label in feature_setups:
            res = bucket_analyze(df, strat, feature, bins)
            if res is None: continue
            for _, r in res.iterrows():
                if r["n_is"] >= 30 and r["n_oos"] >= 20 and r["pf_is"] > 1.4 and r["pf_oos"] > 1.4:
                    edges.append({
                        "strat": strat, "feature": label, "bucket": r["bucket"],
                        "n_is": r["n_is"], "pf_is": r["pf_is"], "net_is": r["net_is"],
                        "n_oos": r["n_oos"], "pf_oos": r["pf_oos"], "net_oos": r["net_oos"],
                    })
    if edges:
        print(pd.DataFrame(edges).to_string(index=False))
    else:
        print("  No strong-edge buckets (PF > 1.4 IS+OOS).")


if __name__ == "__main__":
    main()

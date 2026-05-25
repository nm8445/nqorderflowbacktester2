"""Gamma regime split — STRICTLY PRIOR DAY gamma (no lookahead).

Re-runs the analysis my earlier `gamma_split_analysis.py` did but with
the lookahead bug fixed. Tags each trade with the most recent gamma_sign
whose date is STRICTLY BEFORE the trade date.

Reports direction × gamma split for both top configs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from strategy_atr_sl import build_30min_bars, compute_indicators, run_strategy, stats_block

GAMMA_PATH = "D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet"


def tag_prior_gamma(trades: pd.DataFrame) -> pd.DataFrame:
    """Tag each trade with the gamma_sign from the most recent date STRICTLY BEFORE the trade date."""
    g = pd.read_parquet(GAMMA_PATH)
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g = g[["date", "qqq_gamma_sign"]].dropna(subset=["qqq_gamma_sign"]).sort_values("date")
    pairs = list(zip(g["date"].tolist(), g["qqq_gamma_sign"].tolist()))

    def lookup_prior(d):
        prev = None
        for gd, gs in pairs:
            if gd < d:
                prev = gs
            else:
                break
        return prev

    t = trades.copy()
    t["prior_gamma_sign"] = t["date"].apply(lookup_prior)
    return t


def report_split(trades: pd.DataFrame, label: str):
    print(f"\n{'='*100}")
    print(f"  {label}  ({len(trades)} trades)")
    print(f"{'='*100}")

    s = stats_block(trades["pnl_$"].values)
    print(f"  ALL:           n={s['n']:>5}  WR={s['wr']:>4.1f}%  net=${s['net']:>+10,.0f}  PF={s['pf']:.3f}  MDD=${s['mdd']:>+10,.0f}")

    # By gamma sign
    for sign, name in [(1, "POS gamma"), (-1, "NEG gamma")]:
        sub = trades[trades["prior_gamma_sign"] == sign]
        if len(sub) == 0: continue
        ss = stats_block(sub["pnl_$"].values)
        pct = len(sub) / len(trades) * 100
        print(f"  {name} ({pct:.0f}%):  n={ss['n']:>5}  WR={ss['wr']:>4.1f}%  net=${ss['net']:>+10,.0f}  PF={ss['pf']:.3f}  MDD=${ss['mdd']:>+10,.0f}")

    no_gamma = trades[trades["prior_gamma_sign"].isna()]
    if len(no_gamma) > 0:
        ng = stats_block(no_gamma["pnl_$"].values)
        print(f"  NO gamma:       n={ng['n']:>5}  WR={ng['wr']:>4.1f}%  net=${ng['net']:>+10,.0f}")

    # Direction × gamma sign — the key table
    print(f"\n  -- DIRECTION × PRIOR GAMMA --")
    print(f"  {'cell':<22} {'n':>5} {'WR':>5} {'net':>11} {'PF':>6} {'avg':>7} {'MDD':>10}")
    for sign, gname in [(1, "POS"), (-1, "NEG")]:
        for direction in ["LONG", "SHORT"]:
            sub = trades[(trades["prior_gamma_sign"] == sign) & (trades["direction"] == direction)]
            if len(sub) == 0: continue
            p = sub["pnl_$"].values
            ws = (p > 0).sum(); ls = (p < 0).sum()
            wr = ws/len(p)*100
            gw = p[p>0].sum(); gl = -p[p<0].sum()
            pf = gw/gl if gl > 0 else 99.0
            cum = p.cumsum(); mdd = float((cum - np.maximum.accumulate(cum)).min())
            avg = p.mean()
            print(f"  {gname} gamma + {direction:<5}{'':<6} {len(sub):>5} {wr:>4.1f}% ${p.sum():>+10,.0f} {pf:>6.3f} ${avg:>+6,.0f} ${mdd:>+9,.0f}")


def main():
    bars = build_30min_bars()

    print("Loading bars... done")

    # Config 1: risk-adjusted
    print("\nRunning unfiltered: EMA=60, ATR=28, SL=1.25")
    b1 = compute_indicators(bars, ema_n=60, atr_n=28)
    trades1 = run_strategy(b1, sl_mult=1.25, use_marti=True)
    trades1 = tag_prior_gamma(trades1)
    report_split(trades1, "EMA=60, ATR=28, SL=1.25 (risk-adj winner)")

    # Config 2: absolute $
    print("\nRunning unfiltered: EMA=60, ATR=10, SL=2.75")
    b2 = compute_indicators(bars, ema_n=60, atr_n=10)
    trades2 = run_strategy(b2, sl_mult=2.75, use_marti=True)
    trades2 = tag_prior_gamma(trades2)
    report_split(trades2, "EMA=60, ATR=10, SL=2.75 (abs $ winner)")

    # Save
    out_dir = Path(__file__).parent / "results"
    trades1.to_csv(out_dir / "trades_risk_adj_prior_gamma.csv", index=False)
    trades2.to_csv(out_dir / "trades_abs_dollar_prior_gamma.csv", index=False)
    print(f"\nSaved trades CSVs to {out_dir}")


if __name__ == "__main__":
    main()

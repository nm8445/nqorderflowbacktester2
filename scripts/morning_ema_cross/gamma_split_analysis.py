"""Gamma regime analysis for Morning EMA Cross strategy.

Tags each trade with the prior-day QQQ gamma sign (+1 / -1) and reports
stats split by regime. Tests whether filtering by regime improves results.

Intuition:
  POS gamma: dealer flows DAMPEN moves -> chop -> trend strategies suffer
  NEG gamma: dealer flows AMPLIFY moves -> trends -> trend strategies excel

Strategy is trend-following (EMA cross), so prediction:
  NEG gamma days should produce most of the edge.
  POS gamma days might be net flat or losing.

Best risk-adjusted config from sweep: EMA=60, ATR=28, SL=1.25.
Also tests the top absolute-$ config for comparison.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from strategy_atr_sl import build_30min_bars, compute_indicators, run_strategy, stats_block, NQ_PT

GAMMA_PATH = "D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet"


def tag_gamma(trades: pd.DataFrame) -> pd.DataFrame:
    gamma = pd.read_parquet(GAMMA_PATH)
    gamma["date"] = pd.to_datetime(gamma["date"])
    gamma = (gamma[["date", "qqq_gamma_sign"]]
             .dropna(subset=["qqq_gamma_sign"])
             .sort_values("date"))
    t = trades.copy()
    t["session_date"] = pd.to_datetime(t["date"])   # already a date col
    t = t.sort_values("session_date").reset_index(drop=True)
    t = pd.merge_asof(t, gamma, left_on="session_date", right_on="date",
                       direction="backward", tolerance=pd.Timedelta(days=5))
    return t


def analyze_config(bars, ema_n, atr_n, sl_mult, label: str):
    b = compute_indicators(bars, ema_n, atr_n)
    trades = run_strategy(b, sl_mult, use_marti=True)
    trades = tag_gamma(trades)

    print(f"\n{'='*100}")
    print(f"  CONFIG: EMA={ema_n}  ATR={atr_n}  SL_MULT={sl_mult}  ({label})")
    print(f"{'='*100}")

    # Overall
    s = stats_block(trades["pnl_$"].values)
    print(f"  ALL trades:        n={s['n']:>5}  WR={s['wr']:>4.1f}%  net=${s['net']:>+9,.0f}  PF={s['pf']:.3f}  MDD=${s['mdd']:>+9,.0f}")

    # By gamma sign
    for sign, name in [(1, "POS gamma"), (-1, "NEG gamma")]:
        sub = trades[trades["qqq_gamma_sign"] == sign]
        if len(sub) == 0: continue
        ss = stats_block(sub["pnl_$"].values)
        pct = len(sub) / len(trades) * 100
        print(f"  {name} ({pct:.0f}%):  n={ss['n']:>5}  WR={ss['wr']:>4.1f}%  net=${ss['net']:>+9,.0f}  PF={ss['pf']:.3f}  MDD=${ss['mdd']:>+9,.0f}")

    # Unmatched (no gamma sign — e.g., pre-2020-12 or post-latest-gamma-date)
    no_gamma = trades[trades["qqq_gamma_sign"].isna()]
    if len(no_gamma) > 0:
        ng = stats_block(no_gamma["pnl_$"].values)
        print(f"  NO gamma tag:      n={ng['n']:>5}  WR={ng['wr']:>4.1f}%  net=${ng['net']:>+9,.0f}")

    # By direction + gamma sign
    print(f"\n  -- Long vs Short × Gamma sign --")
    for sign, name in [(1, "POS"), (-1, "NEG")]:
        for direction in ["LONG", "SHORT"]:
            sub = trades[(trades["qqq_gamma_sign"] == sign) & (trades["direction"] == direction)]
            if len(sub) == 0: continue
            ss = stats_block(sub["pnl_$"].values)
            avg = sub["pnl_$"].mean()
            print(f"    {name} gamma + {direction:<5}: n={ss['n']:>4}  WR={ss['wr']:>4.1f}%  net=${ss['net']:>+8,.0f}  avg=${avg:>+6.0f}")

    # IS / OOS split on gamma-filtered subsets
    trades["session_date"] = pd.to_datetime(trades["session_date"])
    dates = sorted(trades["session_date"].dt.date.unique())
    cutoff_idx = int(len(dates) * 0.6)
    cutoff = dates[cutoff_idx]
    cutoff_ts = pd.Timestamp(cutoff)
    print(f"\n  -- Gamma-filtered IS/OOS (cutoff {cutoff}) --")
    for sign, name in [(1, "POS"), (-1, "NEG")]:
        sub = trades[trades["qqq_gamma_sign"] == sign]
        is_sub  = sub[sub["session_date"] <  cutoff_ts]
        oos_sub = sub[sub["session_date"] >= cutoff_ts]
        s_is = stats_block(is_sub["pnl_$"].values)
        s_oos = stats_block(oos_sub["pnl_$"].values)
        print(f"    {name} gamma only — IS: n={s_is['n']:>3} net=${s_is['net']:>+8,.0f} PF={s_is['pf']:.3f} | "
              f"OOS: n={s_oos['n']:>3} net=${s_oos['net']:>+8,.0f} PF={s_oos['pf']:.3f}")

    return trades


def main():
    print(f"Loading 30-min bars...")
    bars = build_30min_bars()
    print(f"  {len(bars):,} bars")

    # Best risk-adjusted
    t1 = analyze_config(bars, ema_n=60, atr_n=28, sl_mult=1.25, label="best risk-adjusted")
    # Best absolute $
    t2 = analyze_config(bars, ema_n=60, atr_n=10, sl_mult=2.75, label="best absolute $")

    # Save tagged trades for inspection
    out_dir = Path(__file__).parent / "results"
    t1.to_csv(out_dir / "best_risk_adj_gamma_tagged.csv", index=False)
    t2.to_csv(out_dir / "best_abs_gamma_tagged.csv", index=False)
    print(f"\nSaved gamma-tagged trades to {out_dir}")


if __name__ == "__main__":
    main()

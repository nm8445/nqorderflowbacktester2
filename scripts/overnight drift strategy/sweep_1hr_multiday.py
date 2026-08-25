"""OD 1hr — can we drop the force-close (hold for DAYS) and/or the green TP on a CFD account?

CFDs have no session force-out and (for this user's live $10k) no trailing DD, so two constraints
the futures config was built around can be relaxed:
  1. force_close 14:00 ET  -> remove it entirely; hold until an exit signal fires.
  2. green (TP)            -> remove it; let the trailing yellow be the only protection.

Baseline = the rank-1 1hr config from sweep_1hr_timeframe.py (project_od_1hr_config).
Entry is gated on `not in_pos`, so a multi-day hold simply skips subsequent 19:00 entries.

Implementation notes:
  - "no force close": forced_minute=30. Bars are anchored on :00, so the force-close time never
    matches a bar timestamp. Cleaner than special-casing the engine.
  - "no green": green_base = 1e7 -> green_val is unreachable (decay 0.17/bar can never erode it).

Run: python "scripts/overnight drift strategy/sweep_1hr_multiday.py"
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from overnight_drift_strategy import StrategyParams, run_backtest, trades_to_df  # noqa: E402
from sweep_1hr_timeframe import build_series  # noqa: E402

IS_END = pd.Timestamp("2023-12-31", tz="America/New_York")

# rank-1 1hr config (project_od_1hr_config.md)
BEST = dict(
    forced_hour=14, forced_minute=0,
    yellow_atr_len=7, yellow_atr_mult=0.99, yellow_mode="pure_ratchet",
    yellow_giveback=0.33, scale_by_body=False,
    max_giveback_atr=0.79, giveback_min_gap_atr=0.36,
    green_atr_len=7, green_atr_mult=3.40, green_base=102.4, green_decay=0.17,
    red_intercept=0.0, red_drift=0.45,
    use_martingale=False, use_be=False, base_qty=1,
)


def run(bars, **over):
    p = StrategyParams(**{**BEST, **over})
    tr = trades_to_df(run_backtest(bars, p))
    if tr.empty:
        return None
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    tr["exit_time"] = pd.to_datetime(tr["exit_time"])
    tr["hold_d"] = (tr["exit_time"] - tr["entry_time"]).dt.total_seconds() / 86400
    tr["period"] = np.where(tr["entry_time"] <= IS_END, "IS", "OOS")
    return tr


def stats(tr, label):
    def pf(s):
        g = s[s > 0].sum(); l = abs(s[s < 0].sum())
        return g / l if l > 0 else np.inf
    d = tr["pnl_dollars"]
    eq = tr.sort_values("entry_time")["pnl_dollars"].cumsum()
    mdd = (eq - eq.cummax()).min()
    isp = pf(tr.loc[tr.period == "IS", "pnl_dollars"])
    oos = pf(tr.loc[tr.period == "OOS", "pnl_dollars"])
    return dict(label=label, n=len(tr), wr=(d > 0).mean() * 100, IS=isp, OOS=oos,
                robust=min(isp, oos), FULL=pf(d), net=d.sum(), mdd=mdd,
                mar=d.sum() / abs(mdd) if mdd else np.nan,
                med_hold=tr.hold_d.median(), max_hold=tr.hold_d.max(),
                pct_gt1d=(tr.hold_d > 1).mean() * 100,
                reasons=tr.reason.value_counts().to_dict())


def main():
    print("Building 60-min bars...", flush=True)
    bars = build_series("60min")
    print(f"  {len(bars):,} bars\n", flush=True)

    variants = [
        ("A. baseline (force 14:00, green ON)", {}),
        ("B. NO force close (green ON)", dict(forced_minute=30)),
        ("C. force 14:00, NO green", dict(green_base=1e7)),
        ("D. NO force close, NO green", dict(forced_minute=30, green_base=1e7)),
    ]
    rows = []
    for lbl, over in variants:
        tr = run(bars, **over)
        if tr is None:
            print(f"{lbl}: no trades"); continue
        rows.append(stats(tr, lbl))

    hdr = (f"  {'variant':>36} {'n':>5} {'WR%':>6} {'IS':>6} {'OOS':>6} {'robust':>7} "
           f"{'net$':>10} {'maxDD$':>10} {'MAR':>6} {'medHold':>8} {'maxHold':>8} {'>1d%':>6}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['label']:>36} {r['n']:>5} {r['wr']:>5.1f}% {r['IS']:>6.3f} {r['OOS']:>6.3f} "
              f"{r['robust']:>7.3f} ${r['net']:>9,.0f} ${r['mdd']:>9,.0f} {r['mar']:>6.2f} "
              f"{r['med_hold']:>7.2f}d {r['max_hold']:>7.1f}d {r['pct_gt1d']:>5.1f}%")
    print("\nExit reason mix:")
    for r in rows:
        print(f"  {r['label']:>36}  {r['reasons']}")


if __name__ == "__main__":
    main()

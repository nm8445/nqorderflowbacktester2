"""Walk-forward (per calendar-year fold) of the FIXED FB giveback config k=1.5/gb0.3 vs static ORB_Low.

FB trades are independent (1/day, no martingale/cross-trade state), so we run each stop once over all
days and bucket by entry year. For a sweep-picked config the real overfit question is: does it beat the
static baseline in EVERY fold, or only in aggregate? Reports per-fold n / win% / PF / net / MaxDD for
both, the giveback-minus-static delta, and the overall MaxDD.

Run:  python scripts/fabio_orb/walk_forward_giveback.py
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_giveback_variant import load_days, run_static, run_giveback   # noqa: E402

ET = "America/New_York"
GB = dict(k=1.5, mode="drift_floor", drift=0.0, gb=0.3, scale_body=True, max_gb=0.5, min_gap=0.3)


def to_df(days, keys, run_fn, **kw):
    rows = [r for d in keys if (r := run_fn(days[d], **kw)) is not None]
    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["year"] = df["entry_time"].dt.tz_convert(ET).dt.year
    return df


def fold_stats(s):
    if len(s) == 0:
        return dict(n=0, win=0.0, pf=0.0, net=0.0, dd=0.0)
    wd = s.loc[s.net_dollars > 0, "net_dollars"].sum()
    ld = -s.loc[s.net_dollars < 0, "net_dollars"].sum()
    eq = s.sort_values("entry_time")["net_dollars"].cumsum()
    return dict(n=len(s), win=100 * (s.net_dollars > 0).mean(),
                pf=(wd / ld if ld > 0 else float("inf")), net=s.net_dollars.sum(),
                dd=(eq - eq.cummax()).min())


def main():
    print("Loading 5-min FB bars...", flush=True)
    days = load_days(); keys = sorted(days.keys())
    st = to_df(days, keys, run_static)
    gb = to_df(days, keys, run_giveback, **GB)
    years = sorted(set(st.year) | set(gb.year))

    print(f"\nWalk-forward per year — static ORB_Low  vs  giveback k=1.5/gb0.3\n")
    print(f"{'year':>6} | {'n':>4} {'st_PF':>6} {'gb_PF':>6} {'dPF':>6} | "
          f"{'st_net':>9} {'gb_net':>9} {'dnet':>9} | {'st_DD':>9} {'gb_DD':>9} | gb wins?")
    wins = 0
    for y in years:
        a = fold_stats(st[st.year == y]); b = fold_stats(gb[gb.year == y])
        won = b["pf"] >= a["pf"]
        wins += won
        print(f"{y:>6} | {a['n']:>4} {a['pf']:>6.3f} {b['pf']:>6.3f} {b['pf']-a['pf']:>+6.3f} | "
              f"{a['net']:>9,.0f} {b['net']:>9,.0f} {b['net']-a['net']:>+9,.0f} | "
              f"{a['dd']:>9,.0f} {b['dd']:>9,.0f} | {'YES' if won else 'no'}")

    A = fold_stats(st); B = fold_stats(gb)
    print(f"\n{'FULL':>6} | {A['n']:>4} {A['pf']:>6.3f} {B['pf']:>6.3f} {B['pf']-A['pf']:>+6.3f} | "
          f"{A['net']:>9,.0f} {B['net']:>9,.0f} {B['net']-A['net']:>+9,.0f} | "
          f"{A['dd']:>9,.0f} {B['dd']:>9,.0f} |")
    print(f"\nGiveback PF >= static in {wins}/{len(years)} folds.")
    print(f"Overall MaxDD: static ${A['dd']:,.0f}  ->  giveback ${B['dd']:,.0f}  "
          f"({100*(B['dd']-A['dd'])/A['dd']:+.0f}%)")


if __name__ == "__main__":
    main()

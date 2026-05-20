"""Daily-bucketed Sharpe + Sortino for rough vol, overnight drift, and combined."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
ROOT = HERE.parent.parent


def apply_mart_rv(df, streak=1, mult=1.5, max_doubles=2):
    qty = 1
    loss_streak = 0
    sized = np.zeros(len(df))
    for i, r in enumerate(df.itertuples()):
        sized[i] = r.pnl_dollars * qty
        if r.pnl_dollars < 0:
            loss_streak += 1
        else:
            loss_streak = 0
        if loss_streak >= streak:
            steps = min(loss_streak - streak + 1, max_doubles)
            qty = max(1, int(round(mult ** steps)))
        else:
            qty = 1
    return sized


# Rough vol with martingale
rv = pd.read_csv(RESULTS_DIR / "inspect_v3_FULL_log.csv")
rv["exit_ts"] = pd.to_datetime(rv["exit_ts"], utc=True).dt.tz_convert("America/New_York")
rv = rv.sort_values("exit_ts").reset_index(drop=True)
rv_sized = apply_mart_rv(rv, streak=1, mult=1.5, max_doubles=2)
rv["sized_pnl"] = rv_sized
# Baseline rough vol (no mart)
rv["base_pnl"] = rv["pnl_dollars"]

# Overnight drift
od = pd.read_csv(ROOT / "live" / "overnight drift" / "trades.csv")
od["exit_ts"] = pd.to_datetime(od["exit_time"], utc=True).dt.tz_convert("America/New_York")
od["sized_pnl"] = od["pnl_dollars"]

# Daily aggregation by exit date (ET)
rv["date"] = rv["exit_ts"].dt.date
od["date"] = od["exit_ts"].dt.date

rv_daily = rv.groupby("date")["sized_pnl"].sum().rename("rv_mart")
rv_base_daily = rv.groupby("date")["base_pnl"].sum().rename("rv_base")
od_daily = od.groupby("date")["sized_pnl"].sum().rename("od")

all_dates = sorted(set(rv_daily.index) | set(od_daily.index))
daily = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
daily = daily.join(rv_daily).join(rv_base_daily).join(od_daily).fillna(0)
daily["combined"] = daily["rv_mart"] + daily["od"]
daily["combined_base"] = daily["rv_base"] + daily["od"]


def sharpe(p, ann=252):
    p = np.asarray(p, dtype=float)
    if p.std(ddof=1) == 0: return 0.0
    return p.mean() / p.std(ddof=1) * np.sqrt(ann)


def sortino(p, ann=252):
    p = np.asarray(p, dtype=float)
    downside = p[p < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return 99.0
    return p.mean() / downside.std(ddof=1) * np.sqrt(ann)


def calmar(total_pnl, mdd, n_days):
    years = n_days / 252.0
    if mdd >= 0: return 99.0
    return (total_pnl / years) / abs(mdd)


def report(p, label):
    p = np.asarray(p)
    cum = p.cumsum()
    mdd = (cum - np.maximum.accumulate(cum)).min()
    sh = sharpe(p)
    so = sortino(p)
    cal = calmar(p.sum(), mdd, len(p))
    print(f"{label:>32}: trading_days={len(p):>4}  "
          f"avg/day=${p.mean():>+6,.0f}  std=${p.std(ddof=1):>5,.0f}  "
          f"Sharpe={sh:>5.2f}  Sortino={so:>5.2f}  Calmar={cal:>5.2f}  "
          f"TotPnL=${p.sum():>+8,.0f}  MDD=${mdd:>+8,.0f}")


print("=" * 130)
print("Daily-bucketed risk metrics (annualized × sqrt(252))")
print("=" * 130)
# Each strategy on its own (only its trading days, no zero-padding)
rv_only = daily.loc[rv_daily.index, "rv_mart"].to_numpy()
rv_base_only = daily.loc[rv_base_daily.index, "rv_base"].to_numpy()
od_only = daily.loc[od_daily.index, "od"].to_numpy()
print()
print("--- Per-strategy (active-day basis: only counting days each strategy traded) ---")
report(rv_base_only, "Rough vol v3 BASELINE")
report(rv_only, "Rough vol v3 + mart s1m1.5d2")
report(od_only, "Overnight drift (locked)")

print()
print("--- Combined (joint daily PnL, all trading dates) ---")
report(daily["combined_base"].to_numpy(), "Combined: RV baseline + OD")
report(daily["combined"].to_numpy(), "Combined: RV+mart + OD")

# Correlation of daily PnL
both_active = daily[(daily["rv_mart"] != 0) & (daily["od"] != 0)]
corr_mart = np.corrcoef(both_active["rv_mart"], both_active["od"])[0, 1]
corr_base = np.corrcoef(both_active["rv_base"], both_active["od"])[0, 1]
print()
print(f"Daily PnL correlation (both strats active, {len(both_active)} overlap days):")
print(f"  RV baseline x OD: {corr_base:+.3f}")
print(f"  RV + mart  x OD: {corr_mart:+.3f}")

# Year-by-year Sharpe for combined
print()
print("--- Combined Sharpe per year ---")
daily["year"] = daily.index.year
for y, g in daily.groupby("year"):
    p_base = g["combined_base"].to_numpy()
    p_mart = g["combined"].to_numpy()
    sh_b = sharpe(p_base); sh_m = sharpe(p_mart)
    cum_b = p_base.cumsum(); cum_m = p_mart.cumsum()
    mdd_b = (cum_b - np.maximum.accumulate(cum_b)).min()
    mdd_m = (cum_m - np.maximum.accumulate(cum_m)).min()
    print(f"  {y}: days={len(g):>3}  "
          f"BASE Sharpe={sh_b:>5.2f} PnL=${p_base.sum():>+9,.0f} MDD=${mdd_b:>+8,.0f}  | "
          f"MART Sharpe={sh_m:>5.2f} PnL=${p_mart.sum():>+9,.0f} MDD=${mdd_m:>+8,.0f}")

"""
Full trade distribution analysis for top rules in each window.

Rules analysed:
  1. profit_target(10)   in 10:00-10:45 + vwap_dist <= 50
  2. profit_target(30)   in 10:30-11:00 + vwap_dist <= 50
  3. fixed_stop(20)      in 10:15-10:45 + vwap_dist <= 50

For each rule:
  - Win / loss distribution (hit target vs session close)
  - Tail metrics (worst 10%, best 10%)
  - Commission-adjusted expectancy
  - Daily P&L series: Sharpe, max DD, consecutive losing days,
    prop-account payout metrics

Saves output/diagnostics/rule_full_distribution.txt
"""

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

FV_PATH  = PROJECT_ROOT / "output" / "diagnostics" / "feature_vectors.csv"
TR_PATH  = PROJECT_ROOT / "output" / "diagnostics" / "trailing_simulation_results.csv"
OUT_PATH = PROJECT_ROOT / "output" / "diagnostics" / "rule_full_distribution.txt"

ET               = "America/New_York"
DOLLARS_PER_PT   = 20.0
COMMISSION_PTS   = 0.25   # $5 round-trip = 0.25 NQ points
DAILY_STOP_LOSS  = -500.0 # dollars
PAYOUT_DAYS      = 21

# ---------------------------------------------------------------------------
# Load + derive
# ---------------------------------------------------------------------------

print("Loading feature vectors...")
fv = pd.read_csv(FV_PATH)

def _direction(loc):
    if loc in ("outside_vah", "at_vah"):  return "short"
    if loc in ("outside_val", "at_val"):  return "long"
    return "unknown"

fv["_dir"]    = fv["price_location"].apply(_direction)
fv["mfe_raw"] = np.where(fv["_dir"]=="short", fv["mfe_down"],
                np.where(fv["_dir"]=="long",  fv["mfe_up"],  float("nan")))
fv["mae_raw"] = np.where(fv["_dir"]=="short", fv["mfe_up"],
                np.where(fv["_dir"]=="long",  fv["mfe_down"], float("nan")))

def _bucket(mfe, mae):
    if pd.isna(mfe) or pd.isna(mae): return "unknown"
    if mfe < 10.0 or mae > mfe:      return "loss"
    if mfe < 25.0:                   return "scalp"
    if mfe < 50.0:                   return "medium"
    return "large"

fv["outcome_bucket"] = fv.apply(lambda r: _bucket(r["mfe_raw"], r["mae_raw"]), axis=1)
fv["_et_dt"]   = pd.to_datetime(fv["event_time"], utc=True).dt.tz_convert(ET)
fv["_et_hhmm"] = fv["_et_dt"].dt.hour * 60 + fv["_et_dt"].dt.minute

TRADING_DAYS  = fv["session_date"].nunique()
ALL_DATES     = sorted(fv["session_date"].unique())

fv_join = fv[["session_date", "event_time", "vwap_dist_pts", "_et_hhmm",
              "mae_raw", "mfe_raw", "outcome_bucket"]].copy()

print(f"  Trading days: {TRADING_DAYS}")

print("Loading trailing simulation results...")
tr = pd.read_csv(TR_PATH)
print(f"  Rows: {len(tr):,}")

print("Merging...")
tr = tr.merge(fv_join, on=["session_date", "event_time"], how="inner")
print(f"  Rows after merge: {len(tr):,}")

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

lines = []

def p(s=""):
    lines.append(s)


def _max_drawdown_pts(daily_pts: pd.Series) -> float:
    """Max peak-to-trough drawdown in points on cumulative P&L series."""
    cum = daily_pts.cumsum()
    peak = cum.cummax()
    dd   = (cum - peak)
    return float(dd.min())


def _max_consec_losing_days(daily_pts: pd.Series) -> int:
    """Max consecutive days with P&L < 0 (in points)."""
    max_consec = 0
    current    = 0
    for v in daily_pts:
        if v < 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    return max_consec


def _sharpe(daily_pts: pd.Series) -> float:
    """Annualised Sharpe using daily P&L (in dollars)."""
    daily_usd = daily_pts * DOLLARS_PER_PT
    if daily_usd.std() == 0:
        return float("nan")
    return float(daily_usd.mean() / daily_usd.std() * np.sqrt(252))


def _analyse_rule(df: pd.DataFrame, rule_label: str, target_pts: float | None,
                  window_label: str, trades_per_day: float):
    """
    df     : rows for this rule+window (already filtered)
    target_pts : profit target in pts if a PT rule, else None
    """
    pts   = df["captured_points"]
    n     = len(pts)

    # Win/loss split
    if target_pts is not None:
        wins  = df[df["exit_reason"] == "target"]
        loses = df[df["exit_reason"] != "target"]   # session close
    else:
        # fixed_stop: wins = session_close (not stopped), losses = stop
        wins  = df[df["exit_reason"] != "stop"]
        loses = df[df["exit_reason"] == "stop"]

    n_win  = len(wins)
    n_loss = len(loses)
    wr     = n_win / n if n > 0 else 0.0

    # Tail metrics
    p10_thresh = pts.quantile(0.10)
    p90_thresh = pts.quantile(0.90)
    worst10_mean = float(pts[pts <= p10_thresh].mean())
    best10_mean  = float(pts[pts >= p90_thresh].mean())

    # Expectancy and commission-adjusted
    raw_exp = float(pts.mean())
    adj_exp = raw_exp - COMMISSION_PTS   # subtract 0.25 pts per trade

    # Daily P&L (ALL trading days, including zero-trade days)
    daily = (df.groupby("session_date")["captured_points"]
               .sum()
               .reindex(ALL_DATES, fill_value=0.0))

    mean_daily_pts = float(daily.mean())
    std_daily_pts  = float(daily.std())
    worst_day_pts  = float(daily.min())
    best_day_pts   = float(daily.max())
    pct_pos_days   = float((daily > 0).mean()) * 100
    sharpe         = _sharpe(daily)
    max_dd_pts     = _max_drawdown_pts(daily)
    max_consec_l   = _max_consec_losing_days(daily)

    # Daily stop-loss breaches ($500 = 25 pts)
    daily_usd      = daily * DOLLARS_PER_PT
    n_stop_days    = int((daily_usd <= DAILY_STOP_LOSS).sum())

    # Payout cycle: simulate 21-day rolling windows
    cycle_results = []
    for i in range(0, len(ALL_DATES) - PAYOUT_DAYS + 1, PAYOUT_DAYS):
        chunk = daily.iloc[i:i + PAYOUT_DAYS]
        cycle_results.append({
            "start":       ALL_DATES[i],
            "total_pts":   chunk.sum(),
            "pos_days":    (chunk > 0).sum(),
            "neg_days":    (chunk < 0).sum(),
            "zero_days":   (chunk == 0).sum(),
            "profitable":  chunk.sum() > 0,
        })
    cycle_df = pd.DataFrame(cycle_results)
    pct_profitable_cycles = float((cycle_df["profitable"]).mean()) * 100
    mean_pos_days_per_cycle = float(cycle_df["pos_days"].mean())
    mean_neg_days_per_cycle = float(cycle_df["neg_days"].mean())

    # Distribution percentiles
    pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = {q: float(pts.quantile(q/100)) for q in pctiles}

    # Print
    p("=" * 72)
    p(f"RULE: {rule_label}")
    p(f"WINDOW: {window_label}   trades/day={trades_per_day:.2f}")
    p("=" * 72)
    p()
    p(f"  Total trades          : {n:,}")
    p(f"  Win rate              : {wr*100:.1f}%  ({n_win:,} wins / {n_loss:,} losses)")
    p()

    p(f"  POINTS DISTRIBUTION")
    p(f"  {'':30}  {'pts':>10}  {'dollars':>10}")
    p(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    if n_win > 0:
        p(f"  {'Mean on wins':<30}  {wins['captured_points'].mean():>10.2f}  "
          f"{wins['captured_points'].mean()*DOLLARS_PER_PT:>+10.0f}")
    else:
        p(f"  {'Mean on wins':<30}  {'n/a':>10}")
    if n_loss > 0:
        p(f"  {'Mean on losses':<30}  {loses['captured_points'].mean():>10.2f}  "
          f"{loses['captured_points'].mean()*DOLLARS_PER_PT:>+10.0f}")
    else:
        p(f"  {'Mean on losses':<30}  {'n/a':>10}")
    p(f"  {'Worst 10% mean':<30}  {worst10_mean:>10.2f}  "
      f"{worst10_mean*DOLLARS_PER_PT:>+10.0f}")
    p(f"  {'Best  10% mean':<30}  {best10_mean:>10.2f}  "
      f"{best10_mean*DOLLARS_PER_PT:>+10.0f}")
    p(f"  {'Raw expectancy':<30}  {raw_exp:>10.3f}  "
      f"{raw_exp*DOLLARS_PER_PT:>+10.2f}")
    p(f"  {'Adj expectancy (-0.25 comm)':<30}  {adj_exp:>10.3f}  "
      f"{adj_exp*DOLLARS_PER_PT:>+10.2f}")
    p()

    p(f"  FULL PERCENTILE TABLE (captured_points)")
    p(f"  {'Percentile':>12}  {'pts':>10}  {'dollars':>10}")
    p(f"  {'-'*12}  {'-'*10}  {'-'*10}")
    for q in pctiles:
        v = pct_vals[q]
        p(f"  {'p'+str(q):>12}  {v:>10.2f}  {v*DOLLARS_PER_PT:>+10.0f}")
    p()

    p(f"  DAILY P&L SERIES  (all {TRADING_DAYS} trading days, zero-trade days included)")
    p(f"  {'':30}  {'pts':>10}  {'dollars':>10}")
    p(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    p(f"  {'Mean daily P&L':<30}  {mean_daily_pts:>10.2f}  "
      f"{mean_daily_pts*DOLLARS_PER_PT:>+10.0f}")
    p(f"  {'Std daily P&L':<30}  {std_daily_pts:>10.2f}  "
      f"{std_daily_pts*DOLLARS_PER_PT:>+10.0f}")
    p(f"  {'Best day':<30}  {best_day_pts:>10.2f}  "
      f"{best_day_pts*DOLLARS_PER_PT:>+10.0f}")
    p(f"  {'Worst day':<30}  {worst_day_pts:>10.2f}  "
      f"{worst_day_pts*DOLLARS_PER_PT:>+10.0f}")
    p(f"  {'% profitable days':<30}  {pct_pos_days:>9.1f}%")
    p(f"  {'Annualised Sharpe':<30}  {sharpe:>10.2f}")
    p(f"  {'Max peak-to-trough DD (pts)':<30}  {max_dd_pts:>10.2f}  "
      f"{max_dd_pts*DOLLARS_PER_PT:>+10.0f}")
    p(f"  {'Max consec losing days':<30}  {max_consec_l:>10d}")
    p(f"  {'Days hitting -$500 stop':<30}  {n_stop_days:>10d}  "
      f"({n_stop_days/TRADING_DAYS*100:.1f}% of all days)")
    p()

    p(f"  PAYOUT CYCLE ANALYSIS ({PAYOUT_DAYS}-day cycles)")
    n_cycles = len(cycle_df)
    p(f"  Cycles analysed           : {n_cycles}")
    p(f"  Profitable cycles         : {cycle_df['profitable'].sum()} / {n_cycles}  "
      f"({pct_profitable_cycles:.1f}%)")
    p(f"  Mean profitable days/cycle: {mean_pos_days_per_cycle:.1f} / {PAYOUT_DAYS}")
    p(f"  Mean losing    days/cycle : {mean_neg_days_per_cycle:.1f} / {PAYOUT_DAYS}")
    p()

    # Per-cycle detail table
    p(f"  {'Cycle start':<14}  {'total_pts':>10}  {'total_$':>10}  "
      f"{'pos_days':>9}  {'neg_days':>9}  {'profitable':>11}")
    p(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*11}")
    for _, r in cycle_df.iterrows():
        flag = " ✓" if r["profitable"] else "  "
        p(f"  {r['start']:<14}  {r['total_pts']:>10.1f}  "
          f"{r['total_pts']*DOLLARS_PER_PT:>+10.0f}  "
          f"{int(r['pos_days']):>9}  {int(r['neg_days']):>9}  "
          f"{'YES'+flag:>11}")
    p()

    # Monthly P&L projection
    monthly_mean_usd = mean_daily_pts * DOLLARS_PER_PT * PAYOUT_DAYS
    adj_monthly      = (mean_daily_pts - COMMISSION_PTS * trades_per_day) * DOLLARS_PER_PT * PAYOUT_DAYS
    p(f"  PROJECTED MONTHLY P&L (unadjusted)     : {monthly_mean_usd:>+10,.0f}")
    p(f"  PROJECTED MONTHLY P&L (comm adjusted)  : {adj_monthly:>+10,.0f}")
    p(f"  × 20 prop accounts (adj)               : {adj_monthly*20:>+10,.0f}")
    p()


# ---------------------------------------------------------------------------
# Rule 1: profit_target(10) in 10:00-10:45 + vwap<=50
# ---------------------------------------------------------------------------

print("Analysing profit_target(10) in 10:00-10:45...")
_t0, _t1 = 10*60+0, 10*60+45
r1 = tr[
    (tr["rule_family"] == "profit_target") &
    (tr["params"] == "{'target_pts': 10.0}") &
    (tr["_et_hhmm"] >= _t0) & (tr["_et_hhmm"] < _t1) &
    (tr["vwap_dist_pts"].abs() <= 50.0)
].copy()
# trades/day = unique events in this window / trading days
ev_r1 = r1.drop_duplicates(["session_date", "event_time"])
tpd_r1 = len(ev_r1) / TRADING_DAYS

p("=" * 72)
p("FULL TRADE DISTRIBUTION ANALYSIS")
p(f"Training: {fv['session_date'].min()} to {fv['session_date'].max()}")
p(f"Trading days: {TRADING_DAYS}   Commission: {COMMISSION_PTS} pts ($5 RT)")
p(f"Daily stop-loss threshold: ${abs(DAILY_STOP_LOSS):.0f}")
p("=" * 72)
p()

_analyse_rule(r1, "profit_target(10)", target_pts=10.0,
              window_label="10:00-10:45 ET + |vwap_dist| <= 50",
              trades_per_day=tpd_r1)

# ---------------------------------------------------------------------------
# Rule 2: profit_target(30) in 10:30-11:00 + vwap<=50
# ---------------------------------------------------------------------------

print("Analysing profit_target(30) in 10:30-11:00...")
_t0, _t1 = 10*60+30, 11*60+0
r2 = tr[
    (tr["rule_family"] == "profit_target") &
    (tr["params"] == "{'target_pts': 30.0}") &
    (tr["_et_hhmm"] >= _t0) & (tr["_et_hhmm"] < _t1) &
    (tr["vwap_dist_pts"].abs() <= 50.0)
].copy()
ev_r2 = r2.drop_duplicates(["session_date", "event_time"])
tpd_r2 = len(ev_r2) / TRADING_DAYS

_analyse_rule(r2, "profit_target(30)", target_pts=30.0,
              window_label="10:30-11:00 ET + |vwap_dist| <= 50",
              trades_per_day=tpd_r2)

# ---------------------------------------------------------------------------
# Rule 3: fixed_stop(20) in 10:15-10:45 + vwap<=50
# ---------------------------------------------------------------------------

print("Analysing fixed_stop(20) in 10:15-10:45...")
_t0, _t1 = 10*60+15, 10*60+45
r3 = tr[
    (tr["rule_family"] == "fixed_stop") &
    (tr["params"] == "{'stop_pts': 20.0}") &
    (tr["_et_hhmm"] >= _t0) & (tr["_et_hhmm"] < _t1) &
    (tr["vwap_dist_pts"].abs() <= 50.0)
].copy()
ev_r3 = r3.drop_duplicates(["session_date", "event_time"])
tpd_r3 = len(ev_r3) / TRADING_DAYS

_analyse_rule(r3, "fixed_stop(20)", target_pts=None,
              window_label="10:15-10:45 ET + |vwap_dist| <= 50",
              trades_per_day=tpd_r3)

# ---------------------------------------------------------------------------
# Head-to-head daily P&L comparison
# ---------------------------------------------------------------------------

def _daily_series(df):
    return (df.groupby("session_date")["captured_points"]
              .sum()
              .reindex(ALL_DATES, fill_value=0.0))

daily_r1 = _daily_series(r1)
daily_r2 = _daily_series(r2)
daily_r3 = _daily_series(r3)

p("=" * 72)
p("HEAD-TO-HEAD DAILY P&L COMPARISON (pts per day, all dates)")
p("=" * 72)
p()
p(f"  {'Metric':<35}  {'PT(10) 10:00-45':>16}  {'PT(30) 10:30-00':>16}  {'FS(20) 10:15-45':>16}")
p(f"  {'-'*35}  {'-'*16}  {'-'*16}  {'-'*16}")

def _row(label, v1, v2, v3, fmt="{:+.2f}"):
    p(f"  {label:<35}  {fmt.format(v1):>16}  {fmt.format(v2):>16}  {fmt.format(v3):>16}")

_row("Mean daily pts",        daily_r1.mean(),      daily_r2.mean(),      daily_r3.mean())
_row("Std daily pts",         daily_r1.std(),       daily_r2.std(),       daily_r3.std(),       fmt="{:.2f}")
_row("Sharpe (annualised)",   _sharpe(daily_r1),    _sharpe(daily_r2),    _sharpe(daily_r3),    fmt="{:.2f}")
_row("Max drawdown pts",      _max_drawdown_pts(daily_r1), _max_drawdown_pts(daily_r2), _max_drawdown_pts(daily_r3))
_row("Max drawdown $",        _max_drawdown_pts(daily_r1)*DOLLARS_PER_PT,
                               _max_drawdown_pts(daily_r2)*DOLLARS_PER_PT,
                               _max_drawdown_pts(daily_r3)*DOLLARS_PER_PT, fmt="{:+,.0f}")
_row("Max consec losing days",_max_consec_losing_days(daily_r1),
                               _max_consec_losing_days(daily_r2),
                               _max_consec_losing_days(daily_r3), fmt="{:.0f}")
_row("% profitable days",     (daily_r1>0).mean()*100, (daily_r2>0).mean()*100, (daily_r3>0).mean()*100, fmt="{:.1f}%")
_row("Best day pts",          daily_r1.max(),       daily_r2.max(),       daily_r3.max())
_row("Worst day pts",         daily_r1.min(),       daily_r2.min(),       daily_r3.min())
_row("Best day $",            daily_r1.max()*DOLLARS_PER_PT, daily_r2.max()*DOLLARS_PER_PT, daily_r3.max()*DOLLARS_PER_PT, fmt="{:+,.0f}")
_row("Worst day $",           daily_r1.min()*DOLLARS_PER_PT, daily_r2.min()*DOLLARS_PER_PT, daily_r3.min()*DOLLARS_PER_PT, fmt="{:+,.0f}")
days_stop_r1 = ((daily_r1*DOLLARS_PER_PT) <= DAILY_STOP_LOSS).sum()
days_stop_r2 = ((daily_r2*DOLLARS_PER_PT) <= DAILY_STOP_LOSS).sum()
days_stop_r3 = ((daily_r3*DOLLARS_PER_PT) <= DAILY_STOP_LOSS).sum()
_row("Days hitting -$500 stop", days_stop_r1, days_stop_r2, days_stop_r3, fmt="{:.0f}")
_row("Monthly $ (unadj)",     daily_r1.mean()*DOLLARS_PER_PT*PAYOUT_DAYS,
                               daily_r2.mean()*DOLLARS_PER_PT*PAYOUT_DAYS,
                               daily_r3.mean()*DOLLARS_PER_PT*PAYOUT_DAYS, fmt="{:+,.0f}")
tpd_arr = [tpd_r1, tpd_r2, tpd_r3]
daily_list = [daily_r1, daily_r2, daily_r3]
adj_monthly = [(d.mean() - COMMISSION_PTS*tpd)*DOLLARS_PER_PT*PAYOUT_DAYS
               for d, tpd in zip(daily_list, tpd_arr)]
_row("Monthly $ (comm adj)",  adj_monthly[0], adj_monthly[1], adj_monthly[2], fmt="{:+,.0f}")
_row("× 20 accounts (adj)",   adj_monthly[0]*20, adj_monthly[1]*20, adj_monthly[2]*20, fmt="{:+,.0f}")
p()

# ---------------------------------------------------------------------------
# Viability summary
# ---------------------------------------------------------------------------

p("=" * 72)
p("PROP ACCOUNT VIABILITY SUMMARY")
p("=" * 72)
p()
for label, daily, tpd in [
    ("PT(10) 10:00-10:45", daily_r1, tpd_r1),
    ("PT(30) 10:30-11:00", daily_r2, tpd_r2),
    ("FS(20) 10:15-10:45", daily_r3, tpd_r3),
]:
    adj_exp_pts = daily.mean() - COMMISSION_PTS * tpd  # per-day adj
    monthly_usd = adj_exp_pts * DOLLARS_PER_PT * PAYOUT_DAYS
    sharpe      = _sharpe(daily)
    max_dd_usd  = _max_drawdown_pts(daily) * DOLLARS_PER_PT
    pct_pos     = (daily > 0).mean() * 100
    n_stop      = ((daily * DOLLARS_PER_PT) <= DAILY_STOP_LOSS).sum()

    p(f"  {label}")
    p(f"    Adj monthly P&L (1 account)  : {monthly_usd:>+9,.0f}")
    p(f"    Adj monthly P&L (20 accounts): {monthly_usd*20:>+9,.0f}")
    p(f"    Annualised Sharpe            : {sharpe:>9.2f}")
    p(f"    Max drawdown                 : {max_dd_usd:>+9,.0f}")
    p(f"    % profitable days            : {pct_pos:>8.1f}%")
    p(f"    Days breaching -$500 limit   : {n_stop:>9d} ({n_stop/TRADING_DAYS*100:.1f}% of days)")

    viable = (monthly_usd > 0 and sharpe > 0.5 and pct_pos >= 40
              and max_dd_usd > -5000)
    p(f"    Verdict: {'VIABLE for prop accounts' if viable else 'CAUTION — review metrics'}")
    p()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

report_text = "\n".join(lines)
OUT_PATH.write_text(report_text, encoding="utf-8")
print(f"\nReport saved to {OUT_PATH}")
try:
    print("\n" + report_text)
except UnicodeEncodeError:
    print("\n" + report_text.encode("ascii", errors="replace").decode("ascii"))

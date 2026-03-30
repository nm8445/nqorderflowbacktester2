"""
Stop-loss optimisation for profit-target rules.

Replays forward bars bar-by-bar for each filtered event, checking both
a take-profit target and a hard stop-loss on each bar.

Tie-break rule: if both target and stop are within a single bar's
high-low range, the stop is assumed to be hit first (conservative).

Combos tested
  Window 10:00-10:45 + |vwap_dist| <= 50
    PT(10) + stop(10 / 15 / 20 / 25)
    PT(20) + stop(10 / 15 / 20)
  Window 10:30-11:00 + |vwap_dist| <= 50
    PT(30) + stop(10 / 15 / 20 / 25 / 30)

Outputs:
  output/diagnostics/stop_loss_optimization.txt
  output/tearsheets/best_combination_tearsheet.html  (quantstats)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import quantstats as qs

from nqbt.analysis.reaction_scanner import _load_rth_ticks
from nqbt.analysis.range_bars import build_range_bars

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

FV_PATH      = PROJECT_ROOT / "output" / "diagnostics" / "feature_vectors.csv"
EV_PATH      = PROJECT_ROOT / "output" / "diagnostics" / "reaction_events.csv"
OUT_DIR      = PROJECT_ROOT / "output" / "diagnostics"
TEAR_DIR     = PROJECT_ROOT / "output" / "tearsheets"
TEAR_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH  = OUT_DIR / "stop_loss_optimization.txt"

ET               = "America/New_York"
DOLLARS_PER_PT   = 20.0
COMMISSION_PTS   = 0.25
DAILY_STOP_USD   = -500.0
PAYOUT_DAYS      = 21

# Account size for quantstats returns (use notional position value of 1 contract)
# NQ margin ~$16k; use $50k as reference account size for % returns
ACCOUNT_SIZE_USD = 50_000.0

COMBOS_W1 = [   # 10:00-10:45 window
    ("PT10+SL10",  10.0, 10.0),
    ("PT10+SL15",  10.0, 15.0),
    ("PT10+SL20",  10.0, 20.0),
    ("PT10+SL25",  10.0, 25.0),
    ("PT20+SL10",  20.0, 10.0),
    ("PT20+SL15",  20.0, 15.0),
    ("PT20+SL20",  20.0, 20.0),
]
COMBOS_W2 = [   # 10:30-11:00 window
    ("PT30+SL10",  30.0, 10.0),
    ("PT30+SL15",  30.0, 15.0),
    ("PT30+SL20",  30.0, 20.0),
    ("PT30+SL25",  30.0, 25.0),
    ("PT30+SL30",  30.0, 30.0),
]

# ---------------------------------------------------------------------------
# Load feature vectors, build filtered event lists
# ---------------------------------------------------------------------------

print("Loading feature vectors...")
fv = pd.read_csv(FV_PATH)

# Attach bar_close (entry price) from reaction_events.csv
ev_raw = pd.read_csv(EV_PATH, usecols=["session_date", "event_time", "bar_close"])
fv = fv.merge(ev_raw, on=["session_date", "event_time"], how="left")

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

TRADING_DAYS = fv["session_date"].nunique()
ALL_DATES    = sorted(fv["session_date"].unique())
print(f"  Trading days: {TRADING_DAYS}")

# Filter sets
ev_w1 = fv[
    (fv["_et_hhmm"] >= 10*60+0) & (fv["_et_hhmm"] < 10*60+45) &
    (fv["vwap_dist_pts"].abs() <= 50.0) &
    (fv["_dir"] != "unknown")
].copy()

ev_w2 = fv[
    (fv["_et_hhmm"] >= 10*60+30) & (fv["_et_hhmm"] < 11*60+0) &
    (fv["vwap_dist_pts"].abs() <= 50.0) &
    (fv["_dir"] != "unknown")
].copy()

print(f"  Window 1 (10:00-10:45): {len(ev_w1):,} events over {ev_w1['session_date'].nunique()} sessions")
print(f"  Window 2 (10:30-11:00): {len(ev_w2):,} events over {ev_w2['session_date'].nunique()} sessions")

# ---------------------------------------------------------------------------
# Bar-by-bar replay engine
# ---------------------------------------------------------------------------

def _simulate_pt_stop(forward_bars, entry: float, direction: str,
                      target_pts: float, stop_pts: float) -> tuple[float, str]:
    """
    Returns (captured_points, exit_reason).
    exit_reason: 'target' | 'stop' | 'session_close'
    Tie-break: if both target and stop within one bar, stop wins (conservative).
    """
    if direction == "short":
        target_price = entry - target_pts
        stop_price   = entry + stop_pts
    else:
        target_price = entry + target_pts
        stop_price   = entry - stop_pts

    for bar in forward_bars:
        if direction == "short":
            # Stop hit if bar.high reaches stop_price
            stop_hit   = bar.high >= stop_price
            target_hit = bar.low  <= target_price
        else:
            stop_hit   = bar.low  <= stop_price
            target_hit = bar.high >= target_price

        if stop_hit and target_hit:
            # Conservative: stop first
            return -stop_pts, "stop"
        if stop_hit:
            return -stop_pts, "stop"
        if target_hit:
            return target_pts, "target"

    # Session close
    if forward_bars:
        final = forward_bars[-1].close
    else:
        final = entry

    if direction == "short":
        captured = entry - final
    else:
        captured = final - entry
    return captured, "session_close"


def _replay_events(events_df: pd.DataFrame, combos: list) -> pd.DataFrame:
    """
    Replay all events for all combos.
    Returns a DataFrame with columns:
      session_date, event_time, direction, outcome_bucket,
      combo_label, target_pts, stop_pts, captured_points, exit_reason
    """
    results = []
    sessions = sorted(events_df["session_date"].unique())
    n_sessions = len(sessions)

    for si, session_date in enumerate(sessions):
        if si % 20 == 0:
            print(f"    Session {si+1}/{n_sessions}: {session_date}")

        ticks = _load_rth_ticks(session_date)
        if ticks.empty:
            continue
        bars = build_range_bars(ticks, range_ticks=40)
        if not bars:
            continue

        sess_events = events_df[events_df["session_date"] == session_date]

        for _, ev in sess_events.iterrows():
            bar_idx   = int(ev["bar_idx"])
            direction = ev["_dir"]
            entry     = float(ev["bar_close"])

            if bar_idx >= len(bars):
                continue

            forward_bars = bars[bar_idx + 1:]

            for (label, target_pts, stop_pts) in combos:
                cap, reason = _simulate_pt_stop(
                    forward_bars, entry, direction, target_pts, stop_pts
                )
                results.append({
                    "session_date":    session_date,
                    "event_time":      ev["event_time"],
                    "direction":       direction,
                    "outcome_bucket":  ev["outcome_bucket"],
                    "mfe_raw":         ev["mfe_raw"],
                    "mae_raw":         ev["mae_raw"],
                    "combo_label":     label,
                    "target_pts":      target_pts,
                    "stop_pts":        stop_pts,
                    "captured_points": cap,
                    "exit_reason":     reason,
                })

    return pd.DataFrame(results)


print("\nReplaying Window 1 (10:00-10:45)...")
res_w1 = _replay_events(ev_w1, COMBOS_W1)
print(f"  Rows: {len(res_w1):,}")

print("\nReplaying Window 2 (10:30-11:00)...")
res_w2 = _replay_events(ev_w2, COMBOS_W2)
print(f"  Rows: {len(res_w2):,}")

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _max_dd(daily_pts: pd.Series) -> float:
    cum  = daily_pts.cumsum()
    peak = cum.cummax()
    return float((cum - peak).min())

def _max_consec_losing(daily_pts: pd.Series) -> int:
    best = cur = 0
    for v in daily_pts:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best

def _sharpe(daily_pts: pd.Series) -> float:
    d = daily_pts * DOLLARS_PER_PT
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else float("nan")

def _payout_cycles(daily: pd.Series):
    cycles = []
    for i in range(0, len(ALL_DATES) - PAYOUT_DAYS + 1, PAYOUT_DAYS):
        chunk = daily.iloc[i:i + PAYOUT_DAYS]
        cycles.append(chunk.sum() > 0)
    return cycles

def _combo_metrics(df: pd.DataFrame, label: str, tpd: float) -> dict:
    pts    = df["captured_points"]
    n      = len(pts)
    wins   = df[df["exit_reason"] == "target"]
    stops  = df[df["exit_reason"] == "stop"]
    closes = df[df["exit_reason"] == "session_close"]

    wr      = len(wins) / n
    raw_exp = float(pts.mean())
    adj_exp = raw_exp - COMMISSION_PTS

    daily   = (df.groupby("session_date")["captured_points"]
                 .sum()
                 .reindex(ALL_DATES, fill_value=0.0))

    cycles  = _payout_cycles(daily)
    daily_usd = daily * DOLLARS_PER_PT

    return {
        "label":          label,
        "n":              n,
        "wr":             wr,
        "n_target":       len(wins),
        "n_stop":         len(stops),
        "n_close":        len(closes),
        "mean_win":       float(wins["captured_points"].mean()) if len(wins) > 0 else float("nan"),
        "mean_stop":      float(stops["captured_points"].mean()) if len(stops) > 0 else float("nan"),
        "mean_close":     float(closes["captured_points"].mean()) if len(closes) > 0 else float("nan"),
        "raw_exp":        raw_exp,
        "adj_exp":        adj_exp,
        "monthly":        adj_exp * tpd * PAYOUT_DAYS * DOLLARS_PER_PT,
        "pct_pos_days":   float((daily > 0).mean()) * 100,
        "days_stop_loss": int((daily_usd <= DAILY_STOP_USD).sum()),
        "max_dd_usd":     _max_dd(daily) * DOLLARS_PER_PT,
        "sharpe":         _sharpe(daily),
        "n_cycles":       len(cycles),
        "good_cycles":    sum(cycles),
        "tpd":            tpd,
        "daily_series":   daily,
        "p10":            float(pts.quantile(0.10)),
        "p50":            float(pts.quantile(0.50)),
        "p90":            float(pts.quantile(0.90)),
        "max_consec_lose": _max_consec_losing(daily),
    }


# Compute metrics for all combos
all_metrics = []

tpd_w1 = len(ev_w1) / TRADING_DAYS
tpd_w2 = len(ev_w2) / TRADING_DAYS

print("\nComputing metrics for Window 1...")
for label, target, stop in COMBOS_W1:
    sub = res_w1[res_w1["combo_label"] == label]
    m   = _combo_metrics(sub, f"{label} [10:00-10:45]", tpd_w1)
    m["window"] = "10:00-10:45"
    all_metrics.append(m)

print("Computing metrics for Window 2...")
for label, target, stop in COMBOS_W2:
    sub = res_w2[res_w2["combo_label"] == label]
    m   = _combo_metrics(sub, f"{label} [10:30-11:00]", tpd_w2)
    m["window"] = "10:30-11:00"
    all_metrics.append(m)

# Sort by adj_exp descending
all_metrics.sort(key=lambda x: x["adj_exp"], reverse=True)

# ---------------------------------------------------------------------------
# Identify best combo
# ---------------------------------------------------------------------------
best = all_metrics[0]
print(f"\nBest combo: {best['label']}  adj_exp={best['adj_exp']:.3f} pts")

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

lines = []
def p(s=""): lines.append(s)

p("=" * 80)
p("STOP-LOSS OPTIMISATION — PROFIT TARGET + HARD STOP")
p(f"Training: {fv['session_date'].min()} to {fv['session_date'].max()}")
p(f"Trading days: {TRADING_DAYS}   Commission: {COMMISSION_PTS} pts ($5 RT)")
p(f"Tie-break: if both target & stop in same bar → stop assumed hit first (conservative)")
p("=" * 80)
p()
p(f"  {'Combo [Window]':<35}  {'n':>5}  {'wr%':>6}  {'raw_exp':>8}  {'adj_exp':>8}  "
  f"{'mnth$/acc':>10}  {'pos%':>6}  {'sl_days':>8}  {'max_dd$':>9}  "
  f"{'sharpe':>7}  {'cyc':>4}")
p(f"  {'-'*35}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}  "
  f"{'-'*8}  {'-'*9}  {'-'*7}  {'-'*4}")

for m in all_metrics:
    p(f"  {m['label']:<35}  {m['n']:>5,}  {m['wr']*100:>5.1f}%  "
      f"{m['raw_exp']:>8.3f}  {m['adj_exp']:>8.3f}  "
      f"{m['monthly']:>+10,.0f}  {m['pct_pos_days']:>5.1f}%  "
      f"{m['days_stop_loss']:>8d}  {m['max_dd_usd']:>+9,.0f}  "
      f"{m['sharpe']:>7.2f}  {m['good_cycles']:>2}/{m['n_cycles']}")

p()
p()

# ---------------------------------------------------------------------------
# Detailed breakdown for each combo
# ---------------------------------------------------------------------------

p("=" * 80)
p("DETAILED BREAKDOWN PER COMBO")
p("=" * 80)
p()

for m in all_metrics:
    p(f"── {m['label']} ──")
    p(f"   Trades/day   : {m['tpd']:.2f}  Total trades: {m['n']:,}")
    p(f"   Win rate     : {m['wr']*100:.1f}%  "
      f"(target={m['n_target']:,}  stop={m['n_stop']:,}  close={m['n_close']:,})")
    p(f"   Mean on wins (target hit)  : {m['mean_win']:>7.2f} pts  "
      f"{m['mean_win']*DOLLARS_PER_PT if not pd.isna(m['mean_win']) else float('nan'):>+7.0f}")
    if not pd.isna(m.get('mean_stop', float('nan'))):
        p(f"   Mean on stops              : {m['mean_stop']:>7.2f} pts  "
          f"{m['mean_stop']*DOLLARS_PER_PT:>+7.0f}")
    if not pd.isna(m.get('mean_close', float('nan'))):
        p(f"   Mean on session close      : {m['mean_close']:>7.2f} pts  "
          f"{m['mean_close']*DOLLARS_PER_PT:>+7.0f}")
    p(f"   Raw expectancy             : {m['raw_exp']:>7.3f} pts  {m['raw_exp']*DOLLARS_PER_PT:>+7.2f}")
    p(f"   Adj expectancy             : {m['adj_exp']:>7.3f} pts  {m['adj_exp']*DOLLARS_PER_PT:>+7.2f}")
    p(f"   Monthly P&L (1 acct, adj)  : {m['monthly']:>+10,.0f}")
    p(f"   Monthly P&L (×20, adj)     : {m['monthly']*20:>+10,.0f}")
    p(f"   Percentiles: p10={m['p10']:>7.2f}  p50={m['p50']:>7.2f}  p90={m['p90']:>7.2f}")
    p(f"   % profitable days          : {m['pct_pos_days']:>6.1f}%")
    p(f"   Sharpe (annualised)        : {m['sharpe']:>6.2f}")
    p(f"   Max drawdown               : {m['max_dd_usd']:>+10,.0f}")
    p(f"   Max consec losing days     : {m['max_consec_lose']:>6d}")
    p(f"   Days breaching -$500 stop  : {m['days_stop_loss']:>6d}  ({m['days_stop_loss']/TRADING_DAYS*100:.1f}%)")
    p(f"   Profitable payout cycles   : {m['good_cycles']}/{m['n_cycles']}")
    p()

# ---------------------------------------------------------------------------
# Best combination full daily series
# ---------------------------------------------------------------------------

p("=" * 80)
p(f"BEST COMBINATION: {best['label']}")
p(f"Adj expectancy: {best['adj_exp']:.3f} pts ({best['adj_exp']*DOLLARS_PER_PT:+.2f}/trade)")
p("=" * 80)
p()
p("  DAILY P&L SERIES (pts → dollars at $20/pt)")
p()

daily_best = best["daily_series"]
daily_best_usd = daily_best * DOLLARS_PER_PT

p(f"  {'Date':<14}  {'daily_pts':>10}  {'daily_$':>10}  {'cum_pts':>10}  {'cum_$':>10}  {'flag'}")
p(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*15}")
cum = 0.0
for date in ALL_DATES:
    d_pts = float(daily_best.loc[date]) if date in daily_best.index else 0.0
    cum  += d_pts
    d_usd = d_pts * DOLLARS_PER_PT
    c_usd = cum   * DOLLARS_PER_PT
    flag  = ""
    if d_usd <= DAILY_STOP_USD:
        flag = "<-- DAILY STOP"
    elif d_pts > 0:
        flag = "+"
    p(f"  {date:<14}  {d_pts:>10.2f}  {d_usd:>+10.0f}  {cum:>10.2f}  {c_usd:>+10,}  {flag}")
p()

# Monthly returns
p("  MONTHLY RETURNS SUMMARY")
monthly_s = daily_best_usd.copy()
monthly_s.index = pd.to_datetime(monthly_s.index)
monthly_grouped = monthly_s.resample("ME").sum()
p()
p(f"  {'Month':<10}  {'P&L ($)':>10}  {'bar'}")
p(f"  {'-'*10}  {'-'*10}  {'-'*30}")
for ym, val in monthly_grouped.items():
    bar = "#" * int(abs(val) / 1000)
    sign = "+" if val >= 0 else "-"
    p(f"  {str(ym):<10}  {val:>+10,.0f}  {sign}{bar}")
p()

# Avg trades on profitable vs losing days
best_combo_label = best["label"].split(" [")[0]
best_window       = best["window"]
best_res          = res_w1 if best_window == "10:00-10:45" else res_w2
best_trades       = best_res[best_res["combo_label"] == best_combo_label]
trades_per_day_df = best_trades.groupby("session_date").size().reindex(ALL_DATES, fill_value=0)
pos_days_trades   = trades_per_day_df[daily_best > 0].mean()
neg_days_trades   = trades_per_day_df[daily_best < 0].mean()
p(f"  Avg trades on profitable days : {pos_days_trades:.1f}")
p(f"  Avg trades on losing days     : {neg_days_trades:.1f}")
p()

# ---------------------------------------------------------------------------
# Prop account viability matrix
# ---------------------------------------------------------------------------

p("=" * 80)
p("PROP ACCOUNT VIABILITY MATRIX")
p("Criteria: monthly$ > 0, sharpe > 0.5, pct_pos_days >= 40%, max_dd$ > -10000")
p("=" * 80)
p()
p(f"  {'Combo':<35}  {'monthly$':>9}  {'sharpe':>7}  {'pos%':>6}  "
  f"{'max_dd$':>9}  {'sl_days':>8}  {'cyc':>5}  {'PASS?':>6}")
p(f"  {'-'*35}  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*5}  {'-'*6}")
for m in all_metrics:
    passes = (m["monthly"] > 0 and m["sharpe"] > 0.5
              and m["pct_pos_days"] >= 40.0 and m["max_dd_usd"] > -10000)
    p(f"  {m['label']:<35}  {m['monthly']:>+9,.0f}  {m['sharpe']:>7.2f}  "
      f"{m['pct_pos_days']:>5.1f}%  {m['max_dd_usd']:>+9,.0f}  "
      f"{m['days_stop_loss']:>8d}  "
      f"{m['good_cycles']:>2}/{m['n_cycles']}  {'YES' if passes else 'no':>6}")
p()

# ---------------------------------------------------------------------------
# Save text report
# ---------------------------------------------------------------------------

report_text = "\n".join(lines)
REPORT_PATH.write_text(report_text, encoding="utf-8")
print(f"\nText report saved to {REPORT_PATH}")

# ---------------------------------------------------------------------------
# Quantstats tearsheet for best combo
# ---------------------------------------------------------------------------

print(f"Generating quantstats tearsheet for {best['label']}...")

daily_series_usd = best["daily_series"] * DOLLARS_PER_PT

# Build a proper DatetimeIndex from session_date strings
daily_returns = pd.Series(
    daily_series_usd.values / ACCOUNT_SIZE_USD,
    index=pd.to_datetime(best["daily_series"].index),
    name=best["label"],
)
daily_returns.index = pd.DatetimeIndex(daily_returns.index)

tear_path = TEAR_DIR / "best_combination_tearsheet.html"
try:
    qs.reports.html(
        daily_returns,
        output=str(tear_path),
        title=f"NQ Clean-Entry Filter — {best['label']}",
        download_filename=str(tear_path),
    )
    print(f"Tearsheet saved to {tear_path}")
except Exception as e:
    print(f"Tearsheet generation failed: {e}")
    # Fallback: save basic stats
    fallback = TEAR_DIR / "best_combination_stats.txt"
    qs.reports.basic(daily_returns)
    fallback.write_text(f"Tearsheet failed: {e}\n", encoding="utf-8")

# Print text report
try:
    print("\n" + report_text)
except UnicodeEncodeError:
    print("\n" + report_text.encode("ascii", errors="replace").decode("ascii"))

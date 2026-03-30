"""
Rule B Full Grid Optimization.

Applies Rule B (bar1 AND bar2 both close in reversal direction) and Rule A
(bar1 closes in reversal direction) as post-signal confirmation filters.

  Rule A entry: open of bar2 (signal_bar_idx + 2)
  Rule B entry: open of bar3 (signal_bar_idx + 3)

Filter: 10:00-10:45 ET + vwap_dist_pts <= 50 (same as stop_loss_optimization)

Grid tested (both rules):
  SL: 10, 15, 20, 25, 30
  PT: 10, 15, 20, 25, 30, 35, 40, 45, 50
  => 45 combos per rule

Also reports bar2_completion_time analysis for Rule B:
  mean time from signal bar to bar2 close
  entries that would fall after 11:00 ET / 11:30 ET

Outputs:
  output/diagnostics/rule_b_full_grid_optimization.txt
  output/tearsheets/rule_b_full_grid_best.html
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

# ──────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────────────────────────

FV_PATH     = PROJECT_ROOT / "output" / "diagnostics" / "feature_vectors.csv"
EV_PATH     = PROJECT_ROOT / "output" / "diagnostics" / "reaction_events.csv"
OUT_DIR     = PROJECT_ROOT / "output" / "diagnostics"
TEAR_DIR    = PROJECT_ROOT / "output" / "tearsheets"
TEAR_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUT_DIR / "rule_b_full_grid_optimization.txt"

ET               = "America/New_York"
DOLLARS_PER_PT   = 20.0
COMMISSION_PTS   = 0.25   # per side in points ($5 RT)
DAILY_STOP_USD   = -500.0
PAYOUT_DAYS      = 21
ACCOUNT_SIZE_USD = 50_000.0

SL_GRID = [10, 15, 20, 25, 30]
PT_GRID = [10, 15, 20, 25, 30, 35, 40, 45, 50]

COMBOS = [
    (f"PT{int(pt)}+SL{int(sl)}", float(pt), float(sl))
    for sl in SL_GRID
    for pt in PT_GRID
]

# ──────────────────────────────────────────────────────────────────────────────
# Load + filter events
# ──────────────────────────────────────────────────────────────────────────────

print("Loading data...")
fv = pd.read_csv(FV_PATH)
ev_raw = pd.read_csv(EV_PATH, usecols=["session_date", "event_time", "bar_close"])
fv = fv.merge(ev_raw, on=["session_date", "event_time"], how="left")

def _direction(loc):
    if loc in ("outside_vah", "at_vah"):  return "short"
    if loc in ("outside_val", "at_val"):  return "long"
    return "unknown"

fv["_dir"]    = fv["price_location"].apply(_direction)
fv["mfe_raw"] = np.where(fv["_dir"] == "short", fv["mfe_down"],
                np.where(fv["_dir"] == "long",  fv["mfe_up"],  float("nan")))
fv["mae_raw"] = np.where(fv["_dir"] == "short", fv["mfe_up"],
                np.where(fv["_dir"] == "long",  fv["mfe_down"], float("nan")))

fv["_et_dt"]   = pd.to_datetime(fv["event_time"], utc=True).dt.tz_convert(ET)
fv["_et_hhmm"] = fv["_et_dt"].dt.hour * 60 + fv["_et_dt"].dt.minute

ev = fv[
    (fv["_et_hhmm"] >= 10 * 60 + 0) & (fv["_et_hhmm"] < 10 * 60 + 45) &
    (fv["vwap_dist_pts"].abs() <= 50.0) &
    (fv["_dir"] != "unknown")
].copy()

TRADING_DAYS = fv["session_date"].nunique()
ALL_DATES    = sorted(fv["session_date"].unique())

print(f"  Base events (10:00-10:45 + vwap<=50): {len(ev):,} over "
      f"{ev['session_date'].nunique()} sessions")
print(f"  Trading days (full dataset): {TRADING_DAYS}")

# ──────────────────────────────────────────────────────────────────────────────
# Bar replay helpers
# ──────────────────────────────────────────────────────────────────────────────

def _close_direction(bar, direction):
    """True if bar closed in the reversal (trade) direction."""
    went_up = bar.close > bar.open
    return (not went_up) if direction == "short" else went_up


def _simulate_pt_stop(entry_bar, forward_bars, entry_price, direction,
                      target_pts, stop_pts):
    """
    Replay from entry_bar.open = entry_price.
    entry_bar is included first (may already hit target/stop within same bar).
    Tie-break: stop wins (conservative).
    Returns (captured_points, exit_reason).
    """
    if direction == "short":
        target_price = entry_price - target_pts
        stop_price   = entry_price + stop_pts
    else:
        target_price = entry_price + target_pts
        stop_price   = entry_price - stop_pts

    for bar in [entry_bar] + list(forward_bars):
        if direction == "short":
            stop_hit   = bar.high >= stop_price
            target_hit = bar.low  <= target_price
        else:
            stop_hit   = bar.low  <= stop_price
            target_hit = bar.high >= target_price

        if stop_hit and target_hit:
            return -stop_pts, "stop"
        if stop_hit:
            return -stop_pts, "stop"
        if target_hit:
            return target_pts, "target"

    # Session close
    final = forward_bars[-1].close if forward_bars else (
        entry_bar.close if entry_bar is not None else entry_price
    )
    captured = (entry_price - final) if direction == "short" else (final - entry_price)
    return captured, "session_close"


def _ts_to_et_minutes(ts):
    """Convert a pandas Timestamp to ET hour*60+minute."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_et = ts.tz_convert(ET)
    return ts_et.hour * 60 + ts_et.minute


def _replay_with_confirmation(events_df, rule_name):
    """
    For each qualifying event:
      Rule A: check bar1_close_direction; enter bar2.open
      Rule B: check bar1_close_direction AND bar2_close_direction; enter bar3.open

    Returns (results_df, timing_records_list, n_confirmed).
    """
    results  = []
    timing   = []
    sessions = sorted(events_df["session_date"].unique())
    n_sessions  = len(sessions)
    n_confirmed = 0
    n_no_bars   = 0

    for si, session_date in enumerate(sessions):
        if si % 20 == 0:
            print(f"    [{rule_name}] Session {si+1}/{n_sessions}: {session_date}")

        ticks = _load_rth_ticks(session_date)
        if ticks.empty:
            continue
        bars = build_range_bars(ticks, range_ticks=40)
        if not bars:
            continue

        sess_events = events_df[events_df["session_date"] == session_date]

        for _, row in sess_events.iterrows():
            bar_idx   = int(row["bar_idx"])
            direction = row["_dir"]

            # Need bar1 at minimum for Rule A, bar3 for Rule B
            min_bars_needed = bar_idx + (3 if rule_name == "Rule_B" else 2)
            if min_bars_needed >= len(bars):
                n_no_bars += 1
                continue

            signal_bar = bars[bar_idx]
            bar1       = bars[bar_idx + 1]
            bar2       = bars[bar_idx + 2]
            bar1_cd    = _close_direction(bar1, direction)

            if rule_name == "Rule_A":
                if not bar1_cd:
                    continue
                entry_bar    = bar2
                entry_price  = bar2.open
                confirm_time = bar1.close_time
                forward_bars = bars[bar_idx + 3:]

            else:  # Rule_B
                bar2_cd = _close_direction(bar2, direction)
                if not (bar1_cd and bar2_cd):
                    continue
                if bar_idx + 3 >= len(bars):
                    n_no_bars += 1
                    continue
                bar3         = bars[bar_idx + 3]
                entry_bar    = bar3
                entry_price  = bar3.open
                confirm_time = bar2.close_time
                forward_bars = bars[bar_idx + 4:]

            n_confirmed += 1

            # Timing record for confirmation-time analysis
            sig_close_min = _ts_to_et_minutes(signal_bar.close_time)
            conf_min      = _ts_to_et_minutes(confirm_time)
            if sig_close_min is not None and conf_min is not None:
                elapsed_bars = 2 if rule_name == "Rule_A" else 3  # approx
                timing.append({
                    "session_date":    session_date,
                    "sig_close_min":   sig_close_min,
                    "confirm_min":     conf_min,
                    "elapsed_min":     conf_min - sig_close_min,
                    "past_1100":       conf_min >= 11 * 60,
                    "past_1130":       conf_min >= 11 * 60 + 30,
                })

            # Replay all 45 combos
            for (label, target_pts, stop_pts) in COMBOS:
                cap, reason = _simulate_pt_stop(
                    entry_bar, forward_bars, entry_price, direction,
                    target_pts, stop_pts,
                )
                results.append({
                    "session_date":    session_date,
                    "event_time":      row["event_time"],
                    "direction":       direction,
                    "combo_label":     label,
                    "target_pts":      target_pts,
                    "stop_pts":        stop_pts,
                    "captured_points": cap,
                    "exit_reason":     reason,
                })

    print(f"    [{rule_name}] Confirmed: {n_confirmed:,}  "
          f"No-bars skipped: {n_no_bars:,}")
    return pd.DataFrame(results), timing, n_confirmed


print("\nReplaying Rule B...")
res_b, timing_b, n_b = _replay_with_confirmation(ev, "Rule_B")
print(f"  Rule B result rows: {len(res_b):,}")

print("\nReplaying Rule A...")
res_a, timing_a, n_a = _replay_with_confirmation(ev, "Rule_A")
print(f"  Rule A result rows: {len(res_a):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _max_dd(daily_pts):
    cum  = daily_pts.cumsum()
    peak = cum.cummax()
    return float((cum - peak).min())

def _max_consec_lose(daily_pts):
    best = cur = 0
    for v in daily_pts:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best

def _sharpe(daily_pts):
    d = daily_pts * DOLLARS_PER_PT
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else float("nan")

def _payout_cycles(daily):
    cycles = []
    for i in range(0, len(ALL_DATES) - PAYOUT_DAYS + 1, PAYOUT_DAYS):
        chunk = daily.iloc[i:i + PAYOUT_DAYS]
        cycles.append(float(chunk.sum()) > 0)
    return cycles

def _combo_metrics(df, label, tpd, target_pts, stop_pts):
    n = len(df)
    if n == 0:
        return None
    pts    = df["captured_points"]
    wins   = df[df["exit_reason"] == "target"]
    stops  = df[df["exit_reason"] == "stop"]
    closes = df[df["exit_reason"] == "session_close"]

    wr      = len(wins) / n
    raw_exp = float(pts.mean())
    adj_exp = raw_exp - COMMISSION_PTS

    daily     = (df.groupby("session_date")["captured_points"]
                   .sum()
                   .reindex(ALL_DATES, fill_value=0.0))
    daily_usd = daily * DOLLARS_PER_PT
    cycles    = _payout_cycles(daily)

    return {
        "label":          label,
        "target_pts":     target_pts,
        "stop_pts":       stop_pts,
        "n":              n,
        "tpd":            tpd,
        "wr":             wr,
        "n_target":       len(wins),
        "n_stop":         len(stops),
        "n_close":        len(closes),
        "mean_win":       float(wins["captured_points"].mean()) if len(wins) else float("nan"),
        "mean_loss":      float(stops["captured_points"].mean()) if len(stops) else float("nan"),
        "mean_close":     float(closes["captured_points"].mean()) if len(closes) else float("nan"),
        "raw_exp":        raw_exp,
        "adj_exp":        adj_exp,
        "monthly":        adj_exp * tpd * PAYOUT_DAYS * DOLLARS_PER_PT,
        "pct_pos_days":   float((daily > 0).mean()) * 100,
        "days_sl":        int((daily_usd <= DAILY_STOP_USD).sum()),
        "max_dd_usd":     _max_dd(daily) * DOLLARS_PER_PT,
        "sharpe":         _sharpe(daily),
        "max_consec":     _max_consec_lose(daily),
        "good_cycles":    sum(cycles),
        "n_cycles":       len(cycles),
        "daily_series":   daily,
    }


def _all_metrics(res_df, n_events, rule_name):
    tpd = n_events / TRADING_DAYS
    out = []
    for (label, target, stop) in COMBOS:
        sub = res_df[res_df["combo_label"] == label]
        m   = _combo_metrics(sub, label, tpd, target, stop)
        if m:
            m["rule"] = rule_name
            out.append(m)
    out.sort(key=lambda x: x["adj_exp"], reverse=True)
    return out


print("\nComputing Rule B metrics...")
metrics_b = _all_metrics(res_b, n_b, "Rule_B")
print("Computing Rule A metrics...")
metrics_a = _all_metrics(res_a, n_a, "Rule_A")

best_b = metrics_b[0] if metrics_b else None
best_a = metrics_a[0] if metrics_a else None

tpd_b = n_b / TRADING_DAYS
tpd_a = n_a / TRADING_DAYS

# ──────────────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────────────

lines = []
def p(s=""): lines.append(str(s))

p("=" * 84)
p("RULE B FULL GRID OPTIMIZATION — POST-SIGNAL CONFIRMATION ENTRIES")
p(f"Filter: 10:00-10:45 ET + vwap_dist_pts <= 50")
p(f"Training range: {fv['session_date'].min()} to {fv['session_date'].max()}")
p(f"Trading days: {TRADING_DAYS}   Commission: {COMMISSION_PTS} pts ($5 RT)")
p(f"Base events in window: {len(ev):,}   Rule B confirmed: {n_b:,} ({n_b/len(ev)*100:.1f}%)   "
  f"Rule A confirmed: {n_a:,} ({n_a/len(ev)*100:.1f}%)")
p(f"Events/day — Rule B: {tpd_b:.2f}   Rule A: {tpd_a:.2f}")
p(f"Tie-break: if both target & stop in same bar → stop hit first (conservative)")
p("=" * 84)
p()

# ── Section 1: Bar2 completion time (Rule B) ─────────────────────────────────
p("-" * 84)
p("SECTION 1 — BAR2 COMPLETION TIME ANALYSIS (Rule B)")
p("(confirmation completes when bar2 closes; entry = bar3 open)")
p("-" * 84)
p()

if timing_b:
    tb_df = pd.DataFrame(timing_b)
    mean_elapsed = tb_df["elapsed_min"].mean()
    median_elapsed = tb_df["elapsed_min"].median()
    n_past_1100  = int(tb_df["past_1100"].sum())
    n_past_1130  = int(tb_df["past_1130"].sum())
    n_timing     = len(tb_df)
    p(f"  Timing records: {n_timing:,}")
    p(f"  Mean elapsed (signal close → bar2 close): {mean_elapsed:.1f} min")
    p(f"  Median elapsed:                           {median_elapsed:.1f} min")
    p()
    p(f"  Entries that would fall AFTER 11:00 ET:  {n_past_1100:,} / {n_timing:,} "
      f"({n_past_1100/n_timing*100:.1f}%)")
    p(f"  Entries that would fall AFTER 11:30 ET:  {n_past_1130:,} / {n_timing:,} "
      f"({n_past_1130/n_timing*100:.1f}%)")
    p()
    # Distribution by confirm time bucket
    bins = [0, 10*60, 10*60+15, 10*60+30, 10*60+45, 11*60, 11*60+15, 11*60+30, 24*60]
    labels = ["<10:00","10:00-10:15","10:15-10:30","10:30-10:45","10:45-11:00",
              "11:00-11:15","11:15-11:30",">=11:30"]
    tb_df["time_bucket"] = pd.cut(tb_df["confirm_min"], bins=bins, labels=labels, right=False)
    tbl = tb_df.groupby("time_bucket", observed=False).size()
    p("  Confirmation-time distribution (bar2 close time ET):")
    for bucket, cnt in tbl.items():
        pct = cnt / n_timing * 100
        bar = "#" * int(pct / 2)
        p(f"    {str(bucket):<18}  {cnt:>5,}  ({pct:>5.1f}%)  {bar}")
    p()
else:
    p("  No timing data available.")
    p()

# ── Section 2: Rule A timing ─────────────────────────────────────────────────
p("-" * 84)
p("SECTION 2 — BAR1 COMPLETION TIME ANALYSIS (Rule A)")
p("(confirmation completes when bar1 closes; entry = bar2 open)")
p("-" * 84)
p()

if timing_a:
    ta_df = pd.DataFrame(timing_a)
    p(f"  Timing records: {len(ta_df):,}")
    p(f"  Mean elapsed (signal close → bar1 close): {ta_df['elapsed_min'].mean():.1f} min")
    p(f"  Entries after 11:00 ET: {ta_df['past_1100'].sum():,} / {len(ta_df):,} "
      f"({ta_df['past_1100'].mean()*100:.1f}%)")
    p(f"  Entries after 11:30 ET: {ta_df['past_1130'].sum():,} / {len(ta_df):,} "
      f"({ta_df['past_1130'].mean()*100:.1f}%)")
    p()
else:
    p("  No timing data available.")
    p()

# ── Section 3: Rule B — full grid ranked table ────────────────────────────────
p("-" * 84)
p("SECTION 3 — RULE B FULL GRID: ALL 45 COMBOS (ranked by adj_exp)")
p(f"  Events/day entering (Rule B): {tpd_b:.2f}")
p("-" * 84)
p()

hdr = (f"  {'Combo':<16}  {'n':>5}  {'wr%':>6}  {'raw_exp':>8}  {'adj_exp':>8}  "
       f"{'mnth$/acc':>10}  {'pos%':>6}  {'sl_days':>8}  "
       f"{'max_dd$':>10}  {'sharpe':>7}  {'cyc':>5}")
p(hdr)
p("  " + "-" * (len(hdr) - 2))

for m in metrics_b:
    p(f"  {m['label']:<16}  {m['n']:>5,}  {m['wr']*100:>5.1f}%  "
      f"{m['raw_exp']:>8.3f}  {m['adj_exp']:>8.3f}  "
      f"{m['monthly']:>+10,.0f}  {m['pct_pos_days']:>5.1f}%  "
      f"{m['days_sl']:>8d}  {m['max_dd_usd']:>+10,.0f}  "
      f"{m['sharpe']:>7.2f}  {m['good_cycles']:>2}/{m['n_cycles']}")
p()

# ── Section 4: Rule B heatmap — adj_exp by SL vs PT ──────────────────────────
p("-" * 84)
p("SECTION 4 — RULE B HEATMAP: ADJ_EXP (pts) — rows=SL, cols=PT")
p("-" * 84)
p()

# Build lookup dict
b_lookup = {m["label"]: m["adj_exp"] for m in metrics_b}

col_header = f"  {'SL\\PT':>6}" + "".join(f"  {'PT'+str(pt):>7}" for pt in PT_GRID)
p(col_header)
p("  " + "-" * (len(col_header) - 2))
for sl in SL_GRID:
    row = f"  {'SL'+str(sl):>6}"
    for pt in PT_GRID:
        label = f"PT{int(pt)}+SL{int(sl)}"
        val   = b_lookup.get(label, float("nan"))
        mark  = "*" if (not np.isnan(val) and val == max(b_lookup.values())) else " "
        row  += f"  {val:>6.3f}{mark}"
    p(row)
p()
p(f"  * = best cell.  Best overall: {max(b_lookup, key=b_lookup.get)} = {max(b_lookup.values()):.3f} pts")
p()

# ── Section 5: Rule B top-20 detailed breakdown ───────────────────────────────
p("-" * 84)
p("SECTION 5 — RULE B TOP 20 DETAILED BREAKDOWN")
p("-" * 84)
p()

for rank, m in enumerate(metrics_b[:20], 1):
    p(f"  #{rank:02d}  {m['label']}  [Rule B]")
    p(f"       n={m['n']:,}  ev/day={m['tpd']:.2f}  "
      f"win_rate={m['wr']*100:.1f}%  "
      f"(target={m['n_target']:,}  stop={m['n_stop']:,}  close={m['n_close']:,})")
    w_usd = m["mean_win"] * DOLLARS_PER_PT if not np.isnan(m["mean_win"]) else float("nan")
    l_usd = m["mean_loss"] * DOLLARS_PER_PT if not np.isnan(m["mean_loss"]) else float("nan")
    p(f"       mean_winner  : {m['mean_win']:>7.2f} pts  ({w_usd:>+7.0f})")
    p(f"       mean_loser   : {m['mean_loss']:>7.2f} pts  ({l_usd:>+7.0f})")
    p(f"       raw_exp      : {m['raw_exp']:>7.3f} pts  adj_exp: {m['adj_exp']:>7.3f} pts")
    p(f"       monthly/acc  : {m['monthly']:>+10,.0f}   monthly×20: {m['monthly']*20:>+11,.0f}")
    p(f"       pos_days%    : {m['pct_pos_days']:>5.1f}%   sl_days: {m['days_sl']:>4d}   "
      f"max_consec_lose: {m['max_consec']:>3d}")
    p(f"       max_dd       : {m['max_dd_usd']:>+10,.0f}   sharpe: {m['sharpe']:>6.2f}")
    p(f"       payout_cycles: {m['good_cycles']}/{m['n_cycles']}")
    p()

# ── Section 6: Rule A — full grid ranked table ────────────────────────────────
p("-" * 84)
p("SECTION 6 — RULE A FULL GRID: ALL 45 COMBOS (ranked by adj_exp)")
p(f"  Events/day entering (Rule A): {tpd_a:.2f}")
p("-" * 84)
p()

a_lookup = {m["label"]: m["adj_exp"] for m in metrics_a}

p(hdr)
p("  " + "-" * (len(hdr) - 2))
for m in metrics_a:
    p(f"  {m['label']:<16}  {m['n']:>5,}  {m['wr']*100:>5.1f}%  "
      f"{m['raw_exp']:>8.3f}  {m['adj_exp']:>8.3f}  "
      f"{m['monthly']:>+10,.0f}  {m['pct_pos_days']:>5.1f}%  "
      f"{m['days_sl']:>8d}  {m['max_dd_usd']:>+10,.0f}  "
      f"{m['sharpe']:>7.2f}  {m['good_cycles']:>2}/{m['n_cycles']}")
p()

# Rule A heatmap
p("-" * 84)
p("RULE A HEATMAP: ADJ_EXP (pts) — rows=SL, cols=PT")
p("-" * 84)
p()

p(col_header)
p("  " + "-" * (len(col_header) - 2))
for sl in SL_GRID:
    row = f"  {'SL'+str(sl):>6}"
    for pt in PT_GRID:
        label = f"PT{int(pt)}+SL{int(sl)}"
        val   = a_lookup.get(label, float("nan"))
        mark  = "*" if (not np.isnan(val) and val == max(a_lookup.values())) else " "
        row  += f"  {val:>6.3f}{mark}"
    p(row)
p()

# ── Section 7: Rule A top-10 detailed breakdown ───────────────────────────────
p("-" * 84)
p("SECTION 7 — RULE A TOP 10 DETAILED BREAKDOWN")
p("-" * 84)
p()

for rank, m in enumerate(metrics_a[:10], 1):
    p(f"  #{rank:02d}  {m['label']}  [Rule A]")
    p(f"       n={m['n']:,}  ev/day={m['tpd']:.2f}  "
      f"win_rate={m['wr']*100:.1f}%  "
      f"(target={m['n_target']:,}  stop={m['n_stop']:,}  close={m['n_close']:,})")
    w_usd = m["mean_win"] * DOLLARS_PER_PT if not np.isnan(m["mean_win"]) else float("nan")
    l_usd = m["mean_loss"] * DOLLARS_PER_PT if not np.isnan(m["mean_loss"]) else float("nan")
    p(f"       mean_winner  : {m['mean_win']:>7.2f} pts  ({w_usd:>+7.0f})")
    p(f"       mean_loser   : {m['mean_loss']:>7.2f} pts  ({l_usd:>+7.0f})")
    p(f"       raw_exp      : {m['raw_exp']:>7.3f} pts  adj_exp: {m['adj_exp']:>7.3f} pts")
    p(f"       monthly/acc  : {m['monthly']:>+10,.0f}   monthly×20: {m['monthly']*20:>+11,.0f}")
    p(f"       pos_days%    : {m['pct_pos_days']:>5.1f}%   sl_days: {m['days_sl']:>4d}   "
      f"max_consec_lose: {m['max_consec']:>3d}")
    p(f"       max_dd       : {m['max_dd_usd']:>+10,.0f}   sharpe: {m['sharpe']:>6.2f}")
    p(f"       payout_cycles: {m['good_cycles']}/{m['n_cycles']}")
    p()

# ── Section 8: Head-to-head comparison (best Rule B vs best Rule A vs unfiltered PT10+SL20) ──
p("-" * 84)
p("SECTION 8 — HEAD-TO-HEAD: BEST RULE B vs BEST RULE A")
p("-" * 84)
p()

if best_b and best_a:
    for label_str, m in [("Best Rule B", best_b), ("Best Rule A", best_a)]:
        p(f"  {label_str}: {m['label']}")
        p(f"    ev/day={m['tpd']:.2f}  win_rate={m['wr']*100:.1f}%  "
          f"adj_exp={m['adj_exp']:.3f} pts  monthly/acc={m['monthly']:>+,.0f}")
        p(f"    sharpe={m['sharpe']:.2f}  max_dd={m['max_dd_usd']:>+,.0f}  "
          f"pos_days={m['pct_pos_days']:.1f}%  cycles={m['good_cycles']}/{m['n_cycles']}")
        p()

# ── Section 9: Best Rule B daily series ──────────────────────────────────────
p("-" * 84)
p(f"SECTION 9 — BEST RULE B DAILY P&L SERIES: {best_b['label'] if best_b else 'N/A'}")
p("-" * 84)
p()

if best_b:
    daily_b = best_b["daily_series"]
    daily_b_usd = daily_b * DOLLARS_PER_PT
    p(f"  {'Date':<14}  {'daily_pts':>10}  {'daily_$':>10}  {'cum_pts':>10}  {'cum_$':>10}  flag")
    p(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")
    cum = 0.0
    for date in ALL_DATES:
        d_pts = float(daily_b.loc[date]) if date in daily_b.index else 0.0
        cum  += d_pts
        d_usd = d_pts * DOLLARS_PER_PT
        c_usd = cum   * DOLLARS_PER_PT
        flag  = "<-- DAILY STOP" if d_usd <= DAILY_STOP_USD else ("+" if d_pts > 0 else "")
        p(f"  {date:<14}  {d_pts:>10.2f}  {d_usd:>+10.0f}  {cum:>10.2f}  {c_usd:>+10,.0f}  {flag}")
    p()

    p("  MONTHLY RETURNS:")
    monthly_s = daily_b_usd.copy()
    monthly_s.index = pd.to_datetime(monthly_s.index)
    monthly_grouped = monthly_s.resample("ME").sum()
    p()
    p(f"  {'Month':<10}  {'P&L ($)':>10}  {'bar'}")
    p(f"  {'-'*10}  {'-'*10}  {'-'*30}")
    for ym, val in monthly_grouped.items():
        bar  = "#" * int(abs(val) / 500)
        sign = "+" if val >= 0 else "-"
        p(f"  {str(ym.date()):<10}  {val:>+10,.0f}  {sign}{bar}")
    p()

# ── Save report ───────────────────────────────────────────────────────────────
report_text = "\n".join(lines)
REPORT_PATH.write_text(report_text, encoding="utf-8")
print(f"\nReport saved to {REPORT_PATH}")

# ── Quantstats tearsheet for best Rule B combo ────────────────────────────────
if best_b:
    print(f"Generating quantstats tearsheet for {best_b['label']}...")
    daily_series_usd = best_b["daily_series"] * DOLLARS_PER_PT
    daily_returns = pd.Series(
        daily_series_usd.values / ACCOUNT_SIZE_USD,
        index=pd.DatetimeIndex(pd.to_datetime(best_b["daily_series"].index)),
        name=f"Rule B — {best_b['label']}",
    )
    tear_path = TEAR_DIR / "rule_b_full_grid_best.html"
    try:
        qs.reports.html(
            daily_returns,
            output=str(tear_path),
            title=f"NQ Rule B Confirmation — {best_b['label']}",
            download_filename=str(tear_path),
        )
        print(f"Tearsheet saved to {tear_path}")
    except Exception as e:
        print(f"Tearsheet generation failed: {e}")

# Print to console
try:
    print("\n" + report_text)
except UnicodeEncodeError:
    print("\n" + report_text.encode("ascii", errors="replace").decode("ascii"))

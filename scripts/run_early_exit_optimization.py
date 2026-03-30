"""
Early Exit Optimization.

Tests a post-entry early exit rule on top of fixed PT/SL:
  If bar+1 shows opposing_absorption=True AND close against trade
  → exit at bar+1 close (overrides PT/SL on that bar)

opposing_absorption for SHORT: net_delta < 0 (sellers dominated) AND bar went UP
opposing_absorption for LONG:  net_delta > 0 (buyers dominated) AND bar went DOWN
"close against trade" = same condition (bar close direction), so combined:
  SHORT early exit: net_delta < 0 AND bar.close > bar.open
  LONG  early exit: net_delta > 0 AND bar.close < bar.open

Position state management (applied to BOTH versions for fair comparison):
  - One trade at a time per session
  - 3-bar cooldown after any exit (next signal bar_idx >= exit_bar + 4)
  - No simultaneous long/short

Entry:  signal bar close (bar_close), same as original
Filter: 10:00-10:45 ET + vwap_dist_pts <= 50

Grid (45 combos per version):
  SL: 10, 15, 20, 25, 30
  PT: 10, 15, 20, 25, 30, 35, 40, 45, 50

Outputs:
  output/diagnostics/early_exit_optimization.txt
  output/tearsheets/early_exit_best.html
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
REPORT_PATH = OUT_DIR / "early_exit_optimization.txt"

ET               = "America/New_York"
DOLLARS_PER_PT   = 20.0
COMMISSION_PTS   = 0.25
DAILY_STOP_USD   = -500.0
PAYOUT_DAYS      = 21
ACCOUNT_SIZE_USD = 50_000.0
COOLDOWN_BARS    = 3     # bars to skip after exit (signal eligible at exit_bar + COOLDOWN_BARS + 1)

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
fv     = pd.read_csv(FV_PATH)
ev_raw = pd.read_csv(EV_PATH, usecols=["session_date", "event_time", "bar_close"])
fv     = fv.merge(ev_raw, on=["session_date", "event_time"], how="left")

def _direction(loc):
    if loc in ("outside_vah", "at_vah"):  return "short"
    if loc in ("outside_val", "at_val"):  return "long"
    return "unknown"

fv["_dir"]    = fv["price_location"].apply(_direction)
fv["_et_dt"]  = pd.to_datetime(fv["event_time"], utc=True).dt.tz_convert(ET)
fv["_hhmm"]   = fv["_et_dt"].dt.hour * 60 + fv["_et_dt"].dt.minute

ev = fv[
    (fv["_hhmm"] >= 10 * 60) & (fv["_hhmm"] < 10 * 60 + 45) &
    (fv["vwap_dist_pts"].abs() <= 50.0) &
    (fv["_dir"] != "unknown")
].copy()

TRADING_DAYS = fv["session_date"].nunique()
ALL_DATES    = sorted(fv["session_date"].unique())

print(f"  Base events (10:00-10:45 + vwap<=50): {len(ev):,} over "
      f"{ev['session_date'].nunique()} sessions")
print(f"  Trading days: {TRADING_DAYS}")

# ──────────────────────────────────────────────────────────────────────────────
# Single-trade simulation
# ──────────────────────────────────────────────────────────────────────────────

def _sim_trade(session_bars, signal_bar_idx, direction, entry,
               target_pts, stop_pts, use_early_exit):
    """
    Simulate one trade entered at `entry` (signal bar close).
    Forward bars: session_bars[signal_bar_idx + 1 : ]

    Early exit (use_early_exit=True): only checked on bar+1 (offset 0).
      SHORT: net_delta < 0 AND bar.close > bar.open  → exit at bar.close
      LONG:  net_delta > 0 AND bar.close < bar.open  → exit at bar.close

    Returns (captured_pts, exit_reason, abs_exit_bar_idx)
      exit_reason: 'target' | 'stop' | 'early_exit' | 'session_close'
      abs_exit_bar_idx: index in session_bars list
    """
    if direction == "short":
        target_price = entry - target_pts
        stop_price   = entry + stop_pts
    else:
        target_price = entry + target_pts
        stop_price   = entry - stop_pts

    n = len(session_bars)

    for abs_j in range(signal_bar_idx + 1, n):
        bar    = session_bars[abs_j]
        offset = abs_j - (signal_bar_idx + 1)   # 0 = bar+1, 1 = bar+2, …

        # ── Early exit: only on bar+1 ──────────────────────────────────────
        if use_early_exit and offset == 0:
            went_up = bar.close > bar.open
            nd      = bar.delta            # buy_vol − sell_vol
            if direction == "short":
                trigger = (nd < 0) and went_up      # sellers dominated, bar went up
            else:
                trigger = (nd > 0) and (not went_up) # buyers dominated, bar went down
            if trigger:
                captured = (entry - bar.close) if direction == "short" else (bar.close - entry)
                return captured, "early_exit", abs_j

        # ── PT / SL ────────────────────────────────────────────────────────
        if direction == "short":
            stop_hit   = bar.high >= stop_price
            target_hit = bar.low  <= target_price
        else:
            stop_hit   = bar.low  <= stop_price
            target_hit = bar.high >= target_price

        if stop_hit and target_hit:
            return -stop_pts, "stop", abs_j
        if stop_hit:
            return -stop_pts, "stop", abs_j
        if target_hit:
            return target_pts, "target", abs_j

    # Session close
    final = session_bars[-1].close if n > signal_bar_idx + 1 else entry
    captured = (entry - final) if direction == "short" else (final - entry)
    return captured, "session_close", n - 1


# ──────────────────────────────────────────────────────────────────────────────
# Session-level replay with position state management
# ──────────────────────────────────────────────────────────────────────────────

def _replay_sessions(ev_df, combos, use_early_exit, run_label):
    """
    For each session: load bars once, then for each combo run state-managed simulation.
    One trade at a time; COOLDOWN_BARS cooldown after exit.

    Returns dict[combo_label] → list of trade dicts.
    """
    results   = {lbl: [] for (lbl, _, _) in combos}
    sessions  = sorted(ev_df["session_date"].unique())
    n_sess    = len(sessions)

    for si, session_date in enumerate(sessions):
        if si % 20 == 0:
            print(f"    [{run_label}] Session {si+1}/{n_sess}: {session_date}")

        ticks = _load_rth_ticks(session_date)
        if ticks.empty:
            continue
        bars = build_range_bars(ticks, range_ticks=40)
        if not bars:
            continue

        sess_ev = (ev_df[ev_df["session_date"] == session_date]
                   .sort_values("bar_idx")
                   .reset_index(drop=True))

        for (combo_label, target_pts, stop_pts) in combos:
            # Session state: minimum bar_idx for next signal
            next_eligible = 0

            for _, row in sess_ev.iterrows():
                bar_idx = int(row["bar_idx"])
                if bar_idx < next_eligible:
                    continue

                direction = row["_dir"]
                entry     = float(row["bar_close"])

                captured, reason, abs_exit_bar = _sim_trade(
                    bars, bar_idx, direction, entry,
                    target_pts, stop_pts, use_early_exit
                )

                results[combo_label].append({
                    "session_date":    session_date,
                    "event_time":      row["event_time"],
                    "direction":       direction,
                    "captured_points": captured,
                    "exit_reason":     reason,
                })

                next_eligible = abs_exit_bar + COOLDOWN_BARS + 1

    return results


print("\nReplaying WITH early exit...")
res_ee    = _replay_sessions(ev, COMBOS, True,  "EarlyExit")
print("\nReplaying WITHOUT early exit...")
res_no_ee = _replay_sessions(ev, COMBOS, False, "NoEarlyExit")

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


def _compute_metrics(trade_list, label, target_pts, stop_pts):
    if not trade_list:
        return None
    df  = pd.DataFrame(trade_list)
    pts = df["captured_points"]
    n   = len(df)

    targets = df[df["exit_reason"] == "target"]
    stops   = df[df["exit_reason"] == "stop"]
    closes  = df[df["exit_reason"] == "session_close"]
    early   = df[df["exit_reason"] == "early_exit"]

    wr      = len(targets) / n
    raw_exp = float(pts.mean())
    adj_exp = raw_exp - COMMISSION_PTS
    tpd     = n / TRADING_DAYS

    daily     = (df.groupby("session_date")["captured_points"]
                   .sum()
                   .reindex(ALL_DATES, fill_value=0.0))
    daily_usd = daily * DOLLARS_PER_PT
    cycles    = _payout_cycles(daily)

    return {
        "label":         label,
        "target_pts":    target_pts,
        "stop_pts":      stop_pts,
        "n":             n,
        "tpd":           tpd,
        "wr":            wr,
        "n_target":      len(targets),
        "n_stop":        len(stops),
        "n_close":       len(closes),
        "n_early":       len(early),
        "pct_early":     len(early) / n * 100,
        "mean_win":      float(targets["captured_points"].mean()) if len(targets) else float("nan"),
        "mean_loss_stop":float(stops["captured_points"].mean())   if len(stops)   else float("nan"),
        "mean_close":    float(closes["captured_points"].mean())  if len(closes)  else float("nan"),
        "mean_early":    float(early["captured_points"].mean())   if len(early)   else float("nan"),
        "raw_exp":       raw_exp,
        "adj_exp":       adj_exp,
        "monthly":       adj_exp * tpd * PAYOUT_DAYS * DOLLARS_PER_PT,
        "pct_pos_days":  float((daily > 0).mean()) * 100,
        "days_sl":       int((daily_usd <= DAILY_STOP_USD).sum()),
        "max_dd_usd":    _max_dd(daily) * DOLLARS_PER_PT,
        "sharpe":        _sharpe(daily),
        "max_consec":    _max_consec_lose(daily),
        "good_cycles":   sum(cycles),
        "n_cycles":      len(cycles),
        "daily_series":  daily,
    }


def _build_metrics_list(results_dict):
    out = []
    for (label, target, stop) in COMBOS:
        m = _compute_metrics(results_dict.get(label, []), label, target, stop)
        if m:
            out.append(m)
    out.sort(key=lambda x: x["adj_exp"], reverse=True)
    return out


print("\nComputing metrics...")
metrics_ee    = _build_metrics_list(res_ee)
metrics_no_ee = _build_metrics_list(res_no_ee)

best_ee    = metrics_ee[0]    if metrics_ee    else None
best_no_ee = metrics_no_ee[0] if metrics_no_ee else None

ee_lookup    = {m["label"]: m["adj_exp"] for m in metrics_ee}
no_ee_lookup = {m["label"]: m["adj_exp"] for m in metrics_no_ee}

# ──────────────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────────────

lines = []
def p(s=""): lines.append(str(s))

p("=" * 88)
p("EARLY EXIT OPTIMIZATION — PT+SL WITH/WITHOUT BAR+1 OPPOSING ABSORPTION EXIT")
p(f"Filter: 10:00-10:45 ET + vwap_dist_pts <= 50")
p(f"Training: {fv['session_date'].min()} to {fv['session_date'].max()}")
p(f"Trading days: {TRADING_DAYS}   Commission: {COMMISSION_PTS} pts ($5 RT)")
p(f"Position mgmt: 1 trade/session, {COOLDOWN_BARS}-bar cooldown after exit")
p(f"Early exit: bar+1 opposing_absorption + close against trade → exit at bar+1.close")
p(f"Base qualifying events: {len(ev):,}   Sessions with events: {ev['session_date'].nunique()}")
p("=" * 88)
p()

if best_ee:
    total_ee    = sum(len(v) for v in res_ee.values()) // len(COMBOS)
    total_no_ee = sum(len(v) for v in res_no_ee.values()) // len(COMBOS)
    p(f"  Avg trades taken per session (representative combo PT10+SL10):")
    n_pt10_ee    = len(res_ee.get("PT10+SL10", []))
    n_pt10_no_ee = len(res_no_ee.get("PT10+SL10", []))
    p(f"    Early exit version:    {n_pt10_ee:,} trades  ({n_pt10_ee/TRADING_DAYS:.2f}/day)")
    p(f"    No early exit version: {n_pt10_no_ee:,} trades  ({n_pt10_no_ee/TRADING_DAYS:.2f}/day)")
    p()

# ── Section 1: Both grids ranked side-by-side summary ─────────────────────────
p("-" * 88)
p("SECTION 1 — EARLY EXIT VERSION: ALL 45 COMBOS (ranked by adj_exp)")
p("-" * 88)
p()

hdr = (f"  {'Combo':<16}  {'n':>5}  {'wr%':>6}  {'adj_exp':>8}  {'mnth$/acc':>10}  "
       f"{'pos%':>6}  {'sl_days':>8}  {'max_dd$':>10}  {'sharpe':>7}  "
       f"{'n_early':>8}  {'early%':>7}  {'cyc':>5}")
p(hdr)
p("  " + "-" * (len(hdr) - 2))

for m in metrics_ee:
    p(f"  {m['label']:<16}  {m['n']:>5,}  {m['wr']*100:>5.1f}%  "
      f"{m['adj_exp']:>8.3f}  {m['monthly']:>+10,.0f}  "
      f"{m['pct_pos_days']:>5.1f}%  {m['days_sl']:>8d}  "
      f"{m['max_dd_usd']:>+10,.0f}  {m['sharpe']:>7.2f}  "
      f"{m['n_early']:>8,}  {m['pct_early']:>6.1f}%  "
      f"{m['good_cycles']:>2}/{m['n_cycles']}")
p()

p("-" * 88)
p("SECTION 2 — NO EARLY EXIT VERSION: ALL 45 COMBOS (ranked by adj_exp)")
p("-" * 88)
p()

hdr2 = (f"  {'Combo':<16}  {'n':>5}  {'wr%':>6}  {'adj_exp':>8}  {'mnth$/acc':>10}  "
        f"{'pos%':>6}  {'sl_days':>8}  {'max_dd$':>10}  {'sharpe':>7}  {'cyc':>5}")
p(hdr2)
p("  " + "-" * (len(hdr2) - 2))

for m in metrics_no_ee:
    p(f"  {m['label']:<16}  {m['n']:>5,}  {m['wr']*100:>5.1f}%  "
      f"{m['adj_exp']:>8.3f}  {m['monthly']:>+10,.0f}  "
      f"{m['pct_pos_days']:>5.1f}%  {m['days_sl']:>8d}  "
      f"{m['max_dd_usd']:>+10,.0f}  {m['sharpe']:>7.2f}  "
      f"{m['good_cycles']:>2}/{m['n_cycles']}")
p()

# ── Section 3: Heatmaps side by side ──────────────────────────────────────────
p("-" * 88)
p("SECTION 3 — HEATMAP: ADJ_EXP (pts)  [EARLY EXIT vs NO EARLY EXIT]")
p("-" * 88)
p()

col_hdr = f"  {'SL\\PT':>6}" + "".join(f"  {'PT'+str(pt):>7}" for pt in PT_GRID)
sep_row = "  " + "-" * (len(col_hdr) - 2)

p("  EARLY EXIT version:")
p(col_hdr)
p(sep_row)
for sl in SL_GRID:
    row = f"  {'SL'+str(sl):>6}"
    for pt in PT_GRID:
        lbl = f"PT{int(pt)}+SL{int(sl)}"
        v   = ee_lookup.get(lbl, float("nan"))
        mk  = "*" if (not np.isnan(v) and v == max(ee_lookup.values())) else " "
        row += f"  {v:>6.3f}{mk}"
    p(row)
p()
best_ee_lbl = max(ee_lookup, key=ee_lookup.get)
p(f"  * best cell: {best_ee_lbl} = {ee_lookup[best_ee_lbl]:.3f} pts")
p()

p("  NO EARLY EXIT version:")
p(col_hdr)
p(sep_row)
for sl in SL_GRID:
    row = f"  {'SL'+str(sl):>6}"
    for pt in PT_GRID:
        lbl = f"PT{int(pt)}+SL{int(sl)}"
        v   = no_ee_lookup.get(lbl, float("nan"))
        mk  = "*" if (not np.isnan(v) and v == max(no_ee_lookup.values())) else " "
        row += f"  {v:>6.3f}{mk}"
    p(row)
p()
best_no_ee_lbl = max(no_ee_lookup, key=no_ee_lookup.get)
p(f"  * best cell: {best_no_ee_lbl} = {no_ee_lookup[best_no_ee_lbl]:.3f} pts")
p()

p("  DELTA heatmap (early_exit - no_early_exit, + = early exit better):")
p(col_hdr)
p(sep_row)
for sl in SL_GRID:
    row = f"  {'SL'+str(sl):>6}"
    for pt in PT_GRID:
        lbl  = f"PT{int(pt)}+SL{int(sl)}"
        ve   = ee_lookup.get(lbl, float("nan"))
        vn   = no_ee_lookup.get(lbl, float("nan"))
        diff = ve - vn if (not np.isnan(ve) and not np.isnan(vn)) else float("nan")
        mk   = "+" if diff > 0 else (" " if np.isnan(diff) else "-")
        row += f"  {diff:>+6.3f}{mk}"
    p(row)
p()

# ── Section 4: Early exit top 20 detailed ─────────────────────────────────────
p("-" * 88)
p("SECTION 4 — EARLY EXIT TOP 20 DETAILED BREAKDOWN")
p("-" * 88)
p()

for rank, m in enumerate(metrics_ee[:20], 1):
    w_usd  = m["mean_win"] * DOLLARS_PER_PT        if not np.isnan(m["mean_win"]) else float("nan")
    sl_usd = m["mean_loss_stop"] * DOLLARS_PER_PT  if not np.isnan(m["mean_loss_stop"]) else float("nan")
    ee_usd = m["mean_early"] * DOLLARS_PER_PT      if not np.isnan(m["mean_early"]) else float("nan")

    p(f"  #{rank:02d}  {m['label']}  [EarlyExit]")
    p(f"       n={m['n']:,}  ev/day={m['tpd']:.2f}  "
      f"win_rate={m['wr']*100:.1f}%  "
      f"(target={m['n_target']:,}  stop={m['n_stop']:,}  "
      f"close={m['n_close']:,}  early={m['n_early']:,} [{m['pct_early']:.1f}%])")
    p(f"       mean_winner      : {m['mean_win']:>7.2f} pts  ({w_usd:>+7.0f})")
    p(f"       mean_stop        : {m['mean_loss_stop']:>7.2f} pts  ({sl_usd:>+7.0f})")
    if not np.isnan(m["mean_early"]):
        p(f"       mean_early_exit  : {m['mean_early']:>7.2f} pts  ({ee_usd:>+7.0f})")
    p(f"       raw_exp : {m['raw_exp']:>7.3f} pts    adj_exp: {m['adj_exp']:>7.3f} pts")
    p(f"       monthly/acc : {m['monthly']:>+10,.0f}    monthly×20: {m['monthly']*20:>+11,.0f}")
    p(f"       pos_days%   : {m['pct_pos_days']:>5.1f}%   sl_days: {m['days_sl']:>4d}   "
      f"max_consec_lose: {m['max_consec']:>3d}")
    p(f"       max_dd      : {m['max_dd_usd']:>+10,.0f}    sharpe: {m['sharpe']:>6.2f}")
    p(f"       payout_cycles: {m['good_cycles']}/{m['n_cycles']}")
    p()

# ── Section 5: No early exit top 10 detailed ──────────────────────────────────
p("-" * 88)
p("SECTION 5 — NO EARLY EXIT TOP 10 DETAILED BREAKDOWN")
p("-" * 88)
p()

for rank, m in enumerate(metrics_no_ee[:10], 1):
    w_usd  = m["mean_win"] * DOLLARS_PER_PT       if not np.isnan(m["mean_win"]) else float("nan")
    sl_usd = m["mean_loss_stop"] * DOLLARS_PER_PT if not np.isnan(m["mean_loss_stop"]) else float("nan")

    p(f"  #{rank:02d}  {m['label']}  [NoEarlyExit]")
    p(f"       n={m['n']:,}  ev/day={m['tpd']:.2f}  "
      f"win_rate={m['wr']*100:.1f}%  "
      f"(target={m['n_target']:,}  stop={m['n_stop']:,}  close={m['n_close']:,})")
    p(f"       mean_winner  : {m['mean_win']:>7.2f} pts  ({w_usd:>+7.0f})")
    p(f"       mean_stop    : {m['mean_loss_stop']:>7.2f} pts  ({sl_usd:>+7.0f})")
    p(f"       raw_exp : {m['raw_exp']:>7.3f} pts    adj_exp: {m['adj_exp']:>7.3f} pts")
    p(f"       monthly/acc : {m['monthly']:>+10,.0f}    monthly×20: {m['monthly']*20:>+11,.0f}")
    p(f"       pos_days%   : {m['pct_pos_days']:>5.1f}%   sl_days: {m['days_sl']:>4d}   "
      f"max_consec_lose: {m['max_consec']:>3d}")
    p(f"       max_dd      : {m['max_dd_usd']:>+10,.0f}    sharpe: {m['sharpe']:>6.2f}")
    p(f"       payout_cycles: {m['good_cycles']}/{m['n_cycles']}")
    p()

# ── Section 6: Head-to-head ────────────────────────────────────────────────────
p("-" * 88)
p("SECTION 6 — HEAD-TO-HEAD: BEST EARLY EXIT vs BEST NO EARLY EXIT")
p("-" * 88)
p()

if best_ee and best_no_ee:
    for tag, m in [("EarlyExit", best_ee), ("NoEarlyExit", best_no_ee)]:
        p(f"  [{tag}] {m['label']}")
        p(f"    n={m['n']:,}  ev/day={m['tpd']:.2f}  wr={m['wr']*100:.1f}%  "
          f"adj_exp={m['adj_exp']:.3f} pts  monthly/acc={m['monthly']:>+,.0f}")
        p(f"    sharpe={m['sharpe']:.2f}  max_dd={m['max_dd_usd']:>+,.0f}  "
          f"pos_days={m['pct_pos_days']:.1f}%  cycles={m['good_cycles']}/{m['n_cycles']}")
        if tag == "EarlyExit":
            p(f"    early_exits: {m['n_early']:,} ({m['pct_early']:.1f}%)  "
              f"mean_early_pnl={m['mean_early']:.2f} pts  "
              f"({m['mean_early']*DOLLARS_PER_PT:+.0f})")
        p()

# ── Section 7: Best early exit daily series ────────────────────────────────────
p("-" * 88)
p(f"SECTION 7 — BEST EARLY EXIT DAILY SERIES: {best_ee['label'] if best_ee else 'N/A'}")
p("-" * 88)
p()

if best_ee:
    d    = best_ee["daily_series"]
    d_us = d * DOLLARS_PER_PT
    p(f"  {'Date':<14}  {'daily_pts':>10}  {'daily_$':>10}  {'cum_pts':>10}  {'cum_$':>10}  flag")
    p(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")
    cum = 0.0
    for date in ALL_DATES:
        dp  = float(d.loc[date]) if date in d.index else 0.0
        cum += dp
        p(f"  {date:<14}  {dp:>10.2f}  {dp*DOLLARS_PER_PT:>+10.0f}  "
          f"{cum:>10.2f}  {cum*DOLLARS_PER_PT:>+10,.0f}  "
          f"{'<-- DAILY STOP' if dp*DOLLARS_PER_PT <= DAILY_STOP_USD else ('+' if dp > 0 else '')}")
    p()

    p("  MONTHLY RETURNS:")
    ms = d_us.copy()
    ms.index = pd.to_datetime(ms.index)
    mg = ms.resample("ME").sum()
    p()
    p(f"  {'Month':<10}  {'P&L ($)':>10}  bar")
    p(f"  {'-'*10}  {'-'*10}  {'-'*30}")
    for ym, val in mg.items():
        bar  = "#" * int(abs(val) / 500)
        sign = "+" if val >= 0 else "-"
        p(f"  {str(ym.date()):<10}  {val:>+10,.0f}  {sign}{bar}")
    p()

# ── Save report ────────────────────────────────────────────────────────────────
report_text = "\n".join(lines)
REPORT_PATH.write_text(report_text, encoding="utf-8")
print(f"\nReport saved to {REPORT_PATH}")

# ── Quantstats tearsheet ───────────────────────────────────────────────────────
if best_ee:
    print(f"Generating quantstats tearsheet for {best_ee['label']}...")
    daily_usd = best_ee["daily_series"] * DOLLARS_PER_PT
    rets = pd.Series(
        daily_usd.values / ACCOUNT_SIZE_USD,
        index=pd.DatetimeIndex(pd.to_datetime(best_ee["daily_series"].index)),
        name=f"EarlyExit — {best_ee['label']}",
    )
    tear_path = TEAR_DIR / "early_exit_best.html"
    try:
        qs.reports.html(
            rets,
            output=str(tear_path),
            title=f"NQ Early Exit — {best_ee['label']}",
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

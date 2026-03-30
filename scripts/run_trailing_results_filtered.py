"""
Trailing simulation results filtered to clean-entry subsets.

Filters trailing_simulation_results.csv to:
  - Primary:    signal_time 10:30-11:00 ET  AND  |vwap_dist_pts| <= 50
  - Comparison: signal_time 10:15-10:45 ET  AND  |vwap_dist_pts| <= 50

For each filtered subset:
  1. All rule/param combos ranked by expectancy
  2. Top-10 detailed breakdown by outcome_bucket
  3. Fixed-stop comparison (10/15/20/25 pts)
  4. Practical P&L metrics at $20/pt (1 NQ contract)
  5. Additional time window analysis

Saves output/diagnostics/trailing_results_filtered.txt
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
OUT_PATH = PROJECT_ROOT / "output" / "diagnostics" / "trailing_results_filtered.txt"

ET              = "America/New_York"
DOLLARS_PER_PT  = 20.0          # $20/pt per NQ contract
TRADES_PER_DAY  = 6.79          # step-3 filter (10:30-11:00 + vwap<=50)
PAYOUT_DAYS     = 21
N_PROP_ACCOUNTS = 20

# ---------------------------------------------------------------------------
# Load feature vectors, derive columns
# ---------------------------------------------------------------------------

print("Loading feature vectors...")
fv = pd.read_csv(FV_PATH)

def _direction(loc):
    if loc in ("outside_vah", "at_vah"):  return "short"
    if loc in ("outside_val", "at_val"):  return "long"
    return "unknown"

fv["_dir"]     = fv["price_location"].apply(_direction)
fv["mfe_raw"]  = np.where(fv["_dir"] == "short", fv["mfe_down"],
                 np.where(fv["_dir"] == "long",  fv["mfe_up"],  float("nan")))
fv["mae_raw"]  = np.where(fv["_dir"] == "short", fv["mfe_up"],
                 np.where(fv["_dir"] == "long",  fv["mfe_down"], float("nan")))

def _bucket(mfe, mae):
    if pd.isna(mfe) or pd.isna(mae): return "unknown"
    if mfe < 10.0 or mae > mfe:      return "loss"
    if mfe < 25.0:                   return "scalp"
    if mfe < 50.0:                   return "medium"
    return "large"

fv["outcome_bucket"] = fv.apply(lambda r: _bucket(r["mfe_raw"], r["mae_raw"]), axis=1)
fv["is_clean_large"] = (fv["mfe_raw"] >= 50) & (fv["mae_raw"] < 10)
fv["is_stress_large"]= (fv["mfe_raw"] >= 50) & (fv["mae_raw"] >= 40)
fv["is_large"]       = (fv["mfe_raw"] >= 50) & (fv["mae_raw"] <= fv["mfe_raw"])

# ET time columns
fv["_et_dt"]    = pd.to_datetime(fv["event_time"], utc=True).dt.tz_convert(ET)
fv["_et_hhmm"]  = fv["_et_dt"].dt.hour * 60 + fv["_et_dt"].dt.minute

TRADING_DAYS = fv["session_date"].nunique()
print(f"  Trading days: {TRADING_DAYS}   events: {len(fv)}")

# Keep only columns needed for join
fv_join = fv[["session_date", "event_time", "vwap_dist_pts", "_et_hhmm",
              "mae_raw", "mfe_raw", "outcome_bucket",
              "is_clean_large", "is_stress_large", "is_large"]].copy()

# ---------------------------------------------------------------------------
# Load trailing results
# ---------------------------------------------------------------------------

print("Loading trailing simulation results...")
tr = pd.read_csv(TR_PATH)
print(f"  Rows: {len(tr):,}")

# Parse initial stop from params string
def _parse_stop(params_str: str, rule_family: str) -> float:
    try:
        d = ast.literal_eval(params_str)
        if rule_family == "fixed_stop":       return d.get("stop_pts", float("nan"))
        if rule_family == "bar_trail":        return d.get("trail_pts", float("nan"))
        if rule_family == "breakeven_trail":  return d.get("trigger_pts", float("nan"))
        if rule_family == "profit_target":    return d.get("target_pts", float("nan"))
        if rule_family == "combo":            return d.get("switch_after_pts", float("nan"))
    except Exception:
        pass
    return float("nan")

tr["initial_stop"] = [_parse_stop(p, f) for p, f in zip(tr["params"], tr["rule_family"])]

# Merge with fv metadata
print("Merging with feature vector metadata...")
tr = tr.merge(fv_join, on=["session_date", "event_time"], how="inner")
print(f"  Rows after merge: {len(tr):,}")

# ---------------------------------------------------------------------------
# Define time windows
# ---------------------------------------------------------------------------

WINDOWS = {
    "10:30-11:00": (10*60+30, 11*60+0),
    "10:15-10:45": (10*60+15, 10*60+45),
    "10:00-11:00": (10*60+0,  11*60+0),
    "10:00-10:30": (10*60+0,  10*60+30),
    "10:00-10:45": (10*60+0,  10*60+45),
}

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

lines = []

def p(s=""):
    lines.append(s)

def _events_from_subset(df):
    """Unique events (one row per event, not per rule) from a merged df."""
    return df.drop_duplicates(subset=["session_date", "event_time"])

def _rule_label(row):
    """Short label for rule+params."""
    try:
        d = ast.literal_eval(row["params"])
        vals = "/".join(str(v) for v in d.values())
    except Exception:
        vals = row["params"]
    return f"{row['rule_family']}({vals})"

def _analyse_window(df: pd.DataFrame, label: str, trades_per_day: float):
    """Full analysis for a filtered subset df (already filtered by time+vwap)."""

    events = _events_from_subset(df)
    n_ev   = len(events)
    n_cl   = int(events["is_clean_large"].sum())
    n_st   = int(events["is_stress_large"].sum())
    n_lg   = int(events["is_large"].sum())
    n_loss = int((events["outcome_bucket"] == "loss").sum())
    cf = n_cl / (n_cl + n_st) if (n_cl + n_st) > 0 else 0.0
    lf = n_lg / n_ev if n_ev > 0 else 0.0

    p("=" * 80)
    p(f"WINDOW: {label}")
    p("=" * 80)
    p()
    p(f"  Unique events        : {n_ev:,}   ({n_ev/TRADING_DAYS:.2f}/day)")
    p(f"  clean_large (mae<10) : {n_cl:,}   clean_frac={cf*100:.1f}%")
    p(f"  stress_large (mae≥40): {n_st:,}")
    p(f"  loss                 : {n_loss:,}")
    p(f"  large_fraction       : {lf*100:.1f}%")
    p()

    # ── ALL RULE/PARAM COMBOS ranked by expectancy ──────────────────────────
    p("-" * 80)
    p("ALL RULE/PARAM COMBOS — ranked by expectancy (mean captured_points)")
    p("-" * 80)
    p()

    combo_stats = []
    for (fam, params), grp in df.groupby(["rule_family", "params"]):
        n    = len(grp)
        mean = grp["captured_points"].mean()
        med  = grp["captured_points"].median()
        wr   = (grp["captured_points"] > 0).mean()
        cpct = grp["captured_pct_of_mfe"].mean()
        ist_mean = grp["initial_stop"].mean()
        ist_p90  = grp["initial_stop"].quantile(0.9)
        try:
            d = ast.literal_eval(params)
            vals = "/".join(str(v) for v in d.values())
        except Exception:
            vals = params
        lbl = f"{fam}({vals})"
        combo_stats.append({
            "label": lbl, "fam": fam, "params": params,
            "n": n, "expectancy": mean, "median": med,
            "win_rate": wr, "cap_pct_mfe": cpct,
            "stop_mean": ist_mean, "stop_p90": ist_p90,
        })

    combo_df = pd.DataFrame(combo_stats).sort_values("expectancy", ascending=False)

    hdr = f"  {'Rule/Params':<42}  {'n':>6}  {'expect':>8}  {'median':>8}  {'wr%':>6}  {'cap%mfe':>8}  {'stop_mn':>8}  {'stop_p90':>9}"
    p(hdr)
    p("  " + "-" * (len(hdr) - 2))
    for _, row in combo_df.iterrows():
        p(f"  {row['label']:<42}  {row['n']:>6,}  "
          f"{row['expectancy']:>8.2f}  {row['median']:>8.2f}  "
          f"{row['win_rate']*100:>5.1f}%  {row['cap_pct_mfe']:>7.3f}  "
          f"{row['stop_mean']:>8.2f}  {row['stop_p90']:>9.2f}")
    p()

    # ── TOP 10 DETAILED BY OUTCOME BUCKET ──────────────────────────────────
    p("-" * 80)
    p("TOP 10 RULES — detailed breakdown by outcome_bucket")
    p("-" * 80)
    p()

    top10 = combo_df.head(10)
    buckets = ["loss", "scalp", "medium", "large"]

    for _, row in top10.iterrows():
        grp = df[(df["rule_family"] == row["fam"]) & (df["params"] == row["params"])]
        p(f"  ── {row['label']} ──")
        p(f"     Overall: n={row['n']:,}  expect={row['expectancy']:.2f}  "
          f"median={row['median']:.2f}  wr={row['win_rate']*100:.1f}%  cap%mfe={row['cap_pct_mfe']:.3f}")
        for bkt in buckets:
            sub = grp[grp["outcome_bucket"] == bkt]
            if len(sub) == 0:
                continue
            bkt_mean = sub["captured_points"].mean()
            bkt_med  = sub["captured_points"].median()
            bkt_wr   = (sub["captured_points"] > 0).mean()
            p(f"     {bkt:<8}: n={len(sub):>5,}  expect={bkt_mean:>7.2f}  "
              f"median={bkt_med:>7.2f}  wr={bkt_wr*100:>5.1f}%")
        p()

    # ── FIXED STOP COMPARISON ────────────────────────────────────────────────
    p("-" * 80)
    p("FIXED STOP COMPARISON (from fixed_stop rule family)")
    p("-" * 80)
    p()

    fs_df = df[df["rule_family"] == "fixed_stop"].copy()
    p(f"  {'Stop':>6}  {'n_total':>8}  {'n_surv':>8}  {'surv%':>7}  "
      f"{'mean_all':>9}  {'mean_surv':>10}  {'med_surv':>9}  {'wr%':>7}")
    p(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*7}")
    for stop_val in [10.0, 15.0, 20.0, 25.0]:
        sub = fs_df[fs_df["initial_stop"] == stop_val]
        if len(sub) == 0:
            # 15/25 not in original sweep; compute from mfe/mae directly
            # Survival = mae_raw < stop_val (event not stopped)
            event_sub = events.copy()
            survivors = event_sub[event_sub["mae_raw"] < stop_val]
            n_surv = len(survivors)
            n_tot  = len(event_sub)
            # For survivors, captured = mfe_raw if no stop hit (session close exit)
            # Use 80% capture as proxy — but we can't know without replay.
            # Skip computed estimate for non-simulated stops.
            p(f"  {stop_val:>5.0f}   (not in simulation sweep — derive from mae_raw)")
            continue
        n_tot  = len(sub)
        # "survived" = not stopped (exit_reason != 'stop')
        surv   = sub[sub["exit_reason"] != "stop"]
        n_surv = len(surv)
        p(f"  {stop_val:>5.0f}  {n_tot:>8,}  {n_surv:>8,}  "
          f"{n_surv/n_tot*100:>6.1f}%  "
          f"{sub['captured_points'].mean():>9.2f}  "
          f"{surv['captured_points'].mean() if n_surv > 0 else float('nan'):>10.2f}  "
          f"{surv['captured_points'].median() if n_surv > 0 else float('nan'):>9.2f}  "
          f"{(sub['captured_points']>0).mean()*100:>6.1f}%")
    p()

    # Supplement fixed-stop comparison using mae_raw for stops not in sweep
    p("  Fixed-stop survival from mae_raw (all stop levels including 15/25):")
    p(f"  {'Stop':>6}  {'n_events':>9}  {'survived':>9}  {'surv%':>7}  {'mfe_surv_mean':>14}  {'mae_surv_mean':>14}")
    p(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*14}  {'-'*14}")
    for stop_val in [10.0, 15.0, 20.0, 25.0]:
        surv = events[events["mae_raw"] < stop_val]
        n_tot = len(events)
        n_surv = len(surv)
        mfe_surv = surv["mfe_raw"].mean() if n_surv > 0 else float("nan")
        mae_surv = surv["mae_raw"].mean() if n_surv > 0 else float("nan")
        p(f"  {stop_val:>5.0f}  {n_tot:>9,}  {n_surv:>9,}  "
          f"{n_surv/n_tot*100:>6.1f}%  {mfe_surv:>14.2f}  {mae_surv:>14.2f}")
    p()

    # ── PRACTICAL P&L METRICS ────────────────────────────────────────────────
    p("-" * 80)
    p(f"PRACTICAL P&L METRICS — 1 NQ contract (${DOLLARS_PER_PT:.0f}/pt)")
    p(f"Basis: {trades_per_day:.2f} trades/day × {PAYOUT_DAYS} days/month")
    p("-" * 80)
    p()
    p(f"  {'Rule/Params':<42}  {'mean$/tr':>9}  {'med$/tr':>9}  "
      f"{'month$':>10}  {'×20_accts':>10}")
    p(f"  {'-'*42}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*10}")
    for _, row in combo_df.head(15).iterrows():
        mean_pts = row["expectancy"]
        med_pts  = row["median"]
        mean_usd = mean_pts * DOLLARS_PER_PT
        med_usd  = med_pts  * DOLLARS_PER_PT
        monthly  = mean_usd * trades_per_day * PAYOUT_DAYS
        monthly20 = monthly * N_PROP_ACCOUNTS
        p(f"  {row['label']:<42}  {mean_usd:>+9.0f}  {med_usd:>+9.0f}  "
          f"{monthly:>+10,.0f}  {monthly20:>+10,.0f}")
    p()


# ---------------------------------------------------------------------------
# Apply primary filter: 10:30-11:00 + vwap_dist <= 50
# ---------------------------------------------------------------------------

print("Applying primary filter (10:30-11:00 + vwap_dist<=50)...")
_t0, _t1 = WINDOWS["10:30-11:00"]
tr_1030 = tr[(tr["_et_hhmm"] >= _t0) & (tr["_et_hhmm"] < _t1) &
             (tr["vwap_dist_pts"].abs() <= 50.0)]
print(f"  Rows: {len(tr_1030):,}   events: {tr_1030.drop_duplicates(['session_date','event_time']).shape[0]}")

# Apply comparison filter: 10:15-10:45 + vwap_dist <= 50
print("Applying comparison filter (10:15-10:45 + vwap_dist<=50)...")
_t0, _t1 = WINDOWS["10:15-10:45"]
tr_1015 = tr[(tr["_et_hhmm"] >= _t0) & (tr["_et_hhmm"] < _t1) &
             (tr["vwap_dist_pts"].abs() <= 50.0)]
print(f"  Rows: {len(tr_1015):,}   events: {tr_1015.drop_duplicates(['session_date','event_time']).shape[0]}")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

p("=" * 80)
p("TRAILING SIMULATION RESULTS — FILTERED TO CLEAN-ENTRY SUBSETS")
p(f"Training: {fv['session_date'].min()} to {fv['session_date'].max()}")
p(f"Trading days: {TRADING_DAYS}   NQ: ${DOLLARS_PER_PT:.0f}/pt per contract")
p("=" * 80)
p()

# ---------------------------------------------------------------------------
# Primary window analysis
# ---------------------------------------------------------------------------

trades_per_day_1030 = 1222 / TRADING_DAYS   # step-3 filter events count
_analyse_window(tr_1030, "10:30-11:00 ET + vwap_dist <= 50", trades_per_day_1030)

# ---------------------------------------------------------------------------
# Comparison window analysis
# ---------------------------------------------------------------------------

# 10:15-10:45 events/day from time window sensitivity: 3.01 (filtered)
_analyse_window(tr_1015, "10:15-10:45 ET + vwap_dist <= 50", trades_per_day=3.01)

# ---------------------------------------------------------------------------
# Additional time window comparison (vwap_dist<=50, raw clean fraction)
# ---------------------------------------------------------------------------

p("=" * 80)
p("ADDITIONAL TIME WINDOW COMPARISON (vwap_dist <= 50 applied to all)")
p("=" * 80)
p()
p(f"  {'Window':<22}  {'n_events':>9}  {'ev/day':>7}  "
  f"{'n_clean':>8}  {'n_stress':>9}  {'n_loss':>7}  {'clean%':>7}  {'large%':>7}")
p(f"  {'-'*22}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}")

extra_windows = [
    ("10:30-11:00",    WINDOWS["10:30-11:00"]),
    ("10:15-10:45",    WINDOWS["10:15-10:45"]),
    ("10:00-11:00",    WINDOWS["10:00-11:00"]),
    ("10:00-10:30",    WINDOWS["10:00-10:30"]),
    ("10:00-10:45",    WINDOWS["10:00-10:45"]),
]

for wlabel, (wlo, whi) in extra_windows:
    wfv = fv[(fv["_et_hhmm"] >= wlo) & (fv["_et_hhmm"] < whi) &
             (fv["vwap_dist_pts"].abs() <= 50.0)]
    n_ev  = len(wfv)
    n_cl  = int(wfv["is_clean_large"].sum())
    n_st  = int(wfv["is_stress_large"].sum())
    n_loss = int((wfv["outcome_bucket"] == "loss").sum())
    n_lg  = int(wfv["is_large"].sum())
    cf    = n_cl / (n_cl + n_st) * 100 if (n_cl + n_st) > 0 else 0.0
    lf    = n_lg / n_ev * 100 if n_ev > 0 else 0.0
    epd   = n_ev / TRADING_DAYS
    p(f"  {wlabel:<22}  {n_ev:>9,}  {epd:>7.2f}  "
      f"{n_cl:>8,}  {n_st:>9,}  {n_loss:>7,}  {cf:>6.1f}%  {lf:>6.1f}%")
p()

# Detailed P&L for 10:00-10:45 (highest frequency option)
p("-" * 80)
p("DETAILED: 10:00-10:45 + vwap_dist<=50")
p("(Widest window that still shows meaningful improvement over baseline)")
p("-" * 80)
p()
_t0, _t1 = WINDOWS["10:00-10:45"]
tr_1000_1045 = tr[(tr["_et_hhmm"] >= _t0) & (tr["_et_hhmm"] < _t1) &
                  (tr["vwap_dist_pts"].abs() <= 50.0)]
ev_1000_1045 = _events_from_subset(tr_1000_1045)
trades_1000_1045 = len(ev_1000_1045) / TRADING_DAYS

# Top 5 rules only for this window
combo_stats_w3 = []
for (fam, params), grp in tr_1000_1045.groupby(["rule_family", "params"]):
    try:
        d = ast.literal_eval(params)
        vals = "/".join(str(v) for v in d.values())
    except Exception:
        vals = params
    lbl = f"{fam}({vals})"
    combo_stats_w3.append({
        "label": lbl, "fam": fam, "params": params,
        "n": len(grp),
        "expectancy": grp["captured_points"].mean(),
        "median": grp["captured_points"].median(),
        "win_rate": (grp["captured_points"] > 0).mean(),
        "cap_pct_mfe": grp["captured_pct_of_mfe"].mean(),
    })
cdf3 = pd.DataFrame(combo_stats_w3).sort_values("expectancy", ascending=False)

p(f"  Events: {len(ev_1000_1045):,}  ({trades_1000_1045:.2f}/day)")
p()
p(f"  {'Rule/Params':<42}  {'expect':>8}  {'median':>8}  {'wr%':>6}  {'mean$/tr':>9}  {'month$':>10}  {'×20':>10}")
p(f"  {'-'*42}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*10}")
for _, row in cdf3.head(10).iterrows():
    mean_usd = row["expectancy"] * DOLLARS_PER_PT
    monthly  = mean_usd * trades_1000_1045 * PAYOUT_DAYS
    p(f"  {row['label']:<42}  {row['expectancy']:>8.2f}  {row['median']:>8.2f}  "
      f"{row['win_rate']*100:>5.1f}%  {mean_usd:>+9.0f}  {monthly:>+10,.0f}  {monthly*N_PROP_ACCOUNTS:>+10,.0f}")
p()

# ---------------------------------------------------------------------------
# Cross-window P&L comparison (best rule per window)
# ---------------------------------------------------------------------------

p("=" * 80)
p("CROSS-WINDOW BEST-RULE P&L COMPARISON")
p("(Top expectancy rule per window, vwap_dist<=50)")
p("=" * 80)
p()
p(f"  {'Window':<22}  {'ev/day':>7}  {'best_rule':<35}  "
  f"{'expect':>8}  {'mean$/tr':>9}  {'month$':>10}  {'×20':>10}")
p(f"  {'-'*22}  {'-'*7}  {'-'*35}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*10}")

for wlabel, (wlo, whi) in extra_windows:
    wtr = tr[(tr["_et_hhmm"] >= wlo) & (tr["_et_hhmm"] < whi) &
             (tr["vwap_dist_pts"].abs() <= 50.0)]
    wev_n = wtr.drop_duplicates(["session_date", "event_time"]).shape[0]
    wepd  = wev_n / TRADING_DAYS
    if len(wtr) == 0:
        p(f"  {wlabel:<22}  {wepd:>7.2f}  (no data)")
        continue
    best = max(
        [(fam, params, grp["captured_points"].mean())
         for (fam, params), grp in wtr.groupby(["rule_family", "params"])],
        key=lambda x: x[2]
    )
    fam, params, best_exp = best
    try:
        d = ast.literal_eval(params)
        vals = "/".join(str(v) for v in d.values())
    except Exception:
        vals = params
    lbl = f"{fam}({vals})"
    mean_usd = best_exp * DOLLARS_PER_PT
    monthly  = mean_usd * wepd * PAYOUT_DAYS
    p(f"  {wlabel:<22}  {wepd:>7.2f}  {lbl:<35}  "
      f"{best_exp:>8.2f}  {mean_usd:>+9.0f}  {monthly:>+10,.0f}  {monthly*N_PROP_ACCOUNTS:>+10,.0f}")
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

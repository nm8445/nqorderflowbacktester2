"""
Clean entry filter stack analysis.

Applies filters progressively on feature_vectors.csv and measures
how each filter changes: total events, clean_large count, stress_large count,
loss count, clean_fraction, large_fraction, and frequency metrics.

Also runs:
 - Time-window sensitivity analysis
 - Looser target (mae_raw < 20 instead of < 10)
 - Final subset outcome distributions

Saves output/diagnostics/clean_entry_filter_stack.txt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

FV_PATH    = PROJECT_ROOT / "output" / "diagnostics" / "feature_vectors.csv"
OUT_DIR    = PROJECT_ROOT / "output" / "diagnostics"
REPORT_PATH = OUT_DIR / "clean_entry_filter_stack.txt"

CLEAN_THRESHOLD    = 10.0   # mae_raw < 10 → clean
LOOSE_THRESHOLD    = 20.0   # mae_raw < 20 → loose-clean
LARGE_MFE_MIN      = 50.0
TRADING_DAYS       = None   # computed from data
PAYOUT_CYCLE_DAYS  = 21
ET                 = "America/New_York"

# ---------------------------------------------------------------------------
# Load and derive columns
# ---------------------------------------------------------------------------

print("Loading feature vectors...")
fv = pd.read_csv(FV_PATH)
print(f"  Rows: {len(fv)}")

# Direction
def _direction(loc: str) -> str:
    if loc in ("outside_vah", "at_vah"):
        return "short"
    if loc in ("outside_val", "at_val"):
        return "long"
    return "unknown"

fv["direction"] = fv["price_location"].apply(_direction)

# mfe_raw / mae_raw
fv["mfe_raw"] = np.where(
    fv["direction"] == "short", fv["mfe_down"],
    np.where(fv["direction"] == "long", fv["mfe_up"], float("nan"))
)
fv["mae_raw"] = np.where(
    fv["direction"] == "short", fv["mfe_up"],
    np.where(fv["direction"] == "long", fv["mfe_down"], float("nan"))
)

# Outcome bucket
def _bucket(mfe, mae):
    if pd.isna(mfe) or pd.isna(mae):
        return "unknown"
    if mfe < 10.0 or mae > mfe:
        return "loss"
    if mfe < 25.0:
        return "scalp"
    if mfe < 50.0:
        return "medium"
    return "large"

fv["outcome"] = fv.apply(lambda r: _bucket(r["mfe_raw"], r["mae_raw"]), axis=1)

# Clean / stress flags
fv["is_large"]        = (fv["mfe_raw"] >= LARGE_MFE_MIN) & (fv["mae_raw"] <= fv["mfe_raw"])
fv["is_clean_large"]  = fv["is_large"] & (fv["mae_raw"] < CLEAN_THRESHOLD)
fv["is_loose_large"]  = fv["is_large"] & (fv["mae_raw"] < LOOSE_THRESHOLD)
fv["is_stress_large"] = fv["is_large"] & (fv["mae_raw"] >= 40.0)
fv["is_loss"]         = fv["outcome"] == "loss"

# Direction-aligned w50_cumulative_delta
# long: want positive delta; short: want negative delta
fv["delta_aligned"] = np.where(
    fv["direction"] == "long",  fv["w50_cumulative_delta"],
    np.where(fv["direction"] == "short", -fv["w50_cumulative_delta"], float("nan"))
)

# event_time → ET hour/minute
fv["_et_dt"] = pd.to_datetime(fv["event_time"], utc=True).dt.tz_convert(ET)
fv["_et_hour"]   = fv["_et_dt"].dt.hour
fv["_et_minute"] = fv["_et_dt"].dt.minute
fv["_et_hhmm"]   = fv["_et_hour"] * 60 + fv["_et_minute"]  # minutes since midnight

# Session-level ov_trend_ratio median (for "above session median" filter)
# Since ov_* are z-scored per session, we compute the global median and filter on that.
ov_median = fv["ov_trend_ratio"].median()
fv["ov_trend_above_median"] = fv["ov_trend_ratio"] > ov_median

# Trading days from data
TRADING_DAYS = fv["session_date"].nunique()
print(f"  Trading days: {TRADING_DAYS}")
print(f"  ov_trend_ratio global median: {ov_median:.4f}")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

lines: list[str] = []


def p(s: str = "") -> None:
    lines.append(s)


def _freq_metrics(df: pd.DataFrame, label: str = "", loose: bool = False) -> dict:
    """Compute all frequency / quality metrics for a subset."""
    n_total   = len(df)
    n_cl      = int(df["is_clean_large"].sum())
    n_loose   = int(df["is_loose_large"].sum())
    n_st      = int(df["is_stress_large"].sum())
    n_loss    = int(df["is_loss"].sum())
    n_large   = int(df["is_large"].sum())

    clean_frac = n_cl / (n_cl + n_st) if (n_cl + n_st) > 0 else 0.0
    large_frac = n_large / n_total if n_total > 0 else 0.0
    loose_frac = n_loose / (n_loose + n_st) if (n_loose + n_st) > 0 else 0.0

    days_present = df["session_date"].nunique() if n_total > 0 else 0
    days_zero    = TRADING_DAYS - days_present
    per_day      = n_total / TRADING_DAYS
    per_week     = per_day * 5.0
    per_payout   = per_day * PAYOUT_CYCLE_DAYS

    return {
        "label":        label,
        "n_total":      n_total,
        "n_cl":         n_cl,
        "n_loose":      n_loose,
        "n_st":         n_st,
        "n_loss":       n_loss,
        "n_large":      n_large,
        "clean_frac":   clean_frac,
        "loose_frac":   loose_frac,
        "large_frac":   large_frac,
        "days_present": days_present,
        "days_zero":    days_zero,
        "per_day":      per_day,
        "per_week":     per_week,
        "per_payout":   per_payout,
    }


def _flag(per_day: float) -> str:
    if per_day < 0.5:
        return "  *** CRITICAL FREQUENCY — insufficient for prop payout cycles ***"
    if per_day < 1.0:
        return "  ** FREQUENCY WARNING — less than 1 event per day **"
    return ""


def _print_step(m: dict, step_label: str, include_loose: bool = False) -> None:
    p(f"  {step_label}")
    p(f"    total events       : {m['n_total']:>7,}")
    p(f"    per trading day    : {m['per_day']:>7.2f}")
    p(f"    per week (5d)      : {m['per_week']:>7.2f}")
    p(f"    per payout (21d)   : {m['per_payout']:>7.1f}")
    p(f"    days w/ ≥1 event   : {m['days_present']:>7d}  / {TRADING_DAYS}")
    p(f"    days w/ 0 events   : {m['days_zero']:>7d}")
    p(f"    clean_large        : {m['n_cl']:>7,}  (mae < 10)")
    p(f"    stress_large       : {m['n_st']:>7,}  (mae >= 40)")
    p(f"    loss               : {m['n_loss']:>7,}")
    p(f"    clean_fraction     : {m['clean_frac']*100:>6.1f}%  (clean / clean+stress)")
    p(f"    large_fraction     : {m['large_frac']*100:>6.1f}%  (large / total)")
    if include_loose:
        p(f"    loose_fraction     : {m['loose_frac']*100:>6.1f}%  (mae<20 / mae<20+stress)")
    flag = _flag(m['per_day'])
    if flag:
        p(flag)


# ---------------------------------------------------------------------------
# MAIN FILTER STACK (primary target: mae_raw < 10 = clean)
# ---------------------------------------------------------------------------

p("=" * 80)
p("CLEAN ENTRY FILTER STACK")
p(f"Training: {fv['session_date'].min()} to {fv['session_date'].max()}")
p(f"Trading days: {TRADING_DAYS}   Payout cycle: {PAYOUT_CYCLE_DAYS} days")
p(f"Clean definition: mae_raw < {CLEAN_THRESHOLD} pts (primary) / < {LOOSE_THRESHOLD} pts (loose)")
p("=" * 80)
p()
p("Targets: clean_fraction > 40% AND events/day > 1.0")
p()

# Reference counts from full dataset
_full = _freq_metrics(fv, "baseline")
n_large_total = _full["n_large"]

# Step 1: Baseline
p("-" * 60)
p("STEP 1 — BASELINE (all events, no filter)")
p("-" * 60)
m1 = _freq_metrics(fv, "step1")
_print_step(m1, "")
p()

# Step 2: Time 10:30-11:00
_10_30 = 10 * 60 + 30   # 630
_11_00 = 11 * 60 + 0    # 660
s2 = fv[(fv["_et_hhmm"] >= _10_30) & (fv["_et_hhmm"] < _11_00)]
p("-" * 60)
p("STEP 2 — add TIME FILTER: 10:30-11:00 ET")
p("-" * 60)
m2 = _freq_metrics(s2, "step2")
_print_step(m2, "")
p()

# Step 3: + vwap_dist_pts <= 50
s3 = s2[s2["vwap_dist_pts"].abs() <= 50.0]
p("-" * 60)
p("STEP 3 — add VWAP PROXIMITY: |vwap_dist_pts| <= 50")
p("-" * 60)
m3 = _freq_metrics(s3, "step3")
_print_step(m3, "")
p()

# Step 4: + val_dist_pts <= 100 OR vah_dist_pts <= 100
s4 = s3[(s3["val_dist_pts"].abs() <= 100.0) | (s3["vah_dist_pts"].abs() <= 100.0)]
p("-" * 60)
p("STEP 4 — add LEVEL PROXIMITY: |val_dist| <= 100 OR |vah_dist| <= 100")
p("-" * 60)
m4 = _freq_metrics(s4, "step4")
_print_step(m4, "")
p()

# Step 5: + delta aligned
s5 = s4[s4["delta_aligned"] > 0]
p("-" * 60)
p("STEP 5 — add DELTA ALIGNED: w50 cumulative delta in trade direction")
p("-" * 60)
m5 = _freq_metrics(s5, "step5")
_print_step(m5, "")
p()

# Step 6: + ov_trend_ratio above median
s6 = s5[s5["ov_trend_above_median"]]
p("-" * 60)
p("STEP 6 — add OVERNIGHT TREND: ov_trend_ratio above session median")
p("-" * 60)
m6 = _freq_metrics(s6, "step6")
_print_step(m6, "")
p()

# ---------------------------------------------------------------------------
# SAME STACK FOR 09:45-10:00 WINDOW (comparison)
# ---------------------------------------------------------------------------

p("=" * 80)
p("COMPARISON: FILTER STACK FOR 09:45-10:00 WINDOW")
p("=" * 80)
p()

_09_45 = 9 * 60 + 45    # 585
_10_00 = 10 * 60 + 0    # 600

cmp_base = fv[(fv["_et_hhmm"] >= _09_45) & (fv["_et_hhmm"] < _10_00)]
cmp_s3   = cmp_base[cmp_base["vwap_dist_pts"].abs() <= 50.0]
cmp_s4   = cmp_s3[(cmp_s3["val_dist_pts"].abs() <= 100.0) | (cmp_s3["vah_dist_pts"].abs() <= 100.0)]
cmp_s5   = cmp_s4[cmp_s4["delta_aligned"] > 0]
cmp_s6   = cmp_s5[cmp_s5["ov_trend_above_median"]]

for label, df in [
    ("Step 2 (09:45-10:00 only)", cmp_base),
    ("Step 3 (+ vwap_dist <= 50)", cmp_s3),
    ("Step 4 (+ level proximity)", cmp_s4),
    ("Step 5 (+ delta aligned)", cmp_s5),
    ("Step 6 (+ overnight trend)", cmp_s6),
]:
    cm = _freq_metrics(df, label)
    _print_step(cm, label)
    p()

# ---------------------------------------------------------------------------
# DIRECT COMPARISON TABLE: 09:45-10:00 vs 10:30-11:00 after all filters
# ---------------------------------------------------------------------------

p("=" * 80)
p("WINDOW COMPARISON — AFTER ALL FILTERS (steps 3-6 applied)")
p("=" * 80)
p()
p(f"  {'Metric':<30}  {'09:45-10:00':>14}  {'10:30-11:00':>14}")
p(f"  {'-'*30}  {'-'*14}  {'-'*14}")
for label, early, late in [
    ("total events",      cmp_s6,  s6),
    ("events / day",      None,    None),  # computed below
    ("clean_large (n)",   None,    None),
    ("stress_large (n)",  None,    None),
    ("clean_fraction",    None,    None),
    ("large_fraction",    None,    None),
]:
    pass  # handled below

def _cmp_row(label, v1, v2, fmt="{:.0f}"):
    p(f"  {label:<30}  {fmt.format(v1):>14}  {fmt.format(v2):>14}")

_cmp_row("total events",          len(cmp_s6),             len(s6),             "{:.0f}")
_cmp_row("events / day",          len(cmp_s6)/TRADING_DAYS, len(s6)/TRADING_DAYS, "{:.2f}")
_cmp_row("clean_large (n)",       cmp_s6["is_clean_large"].sum(), s6["is_clean_large"].sum(), "{:.0f}")
_cmp_row("stress_large (n)",      cmp_s6["is_stress_large"].sum(), s6["is_stress_large"].sum(), "{:.0f}")
cl_early = cmp_s6["is_clean_large"].sum(); st_early = cmp_s6["is_stress_large"].sum()
cl_late  = s6["is_clean_large"].sum();     st_late  = s6["is_stress_large"].sum()
_cmp_row("clean_fraction",        cl_early/(cl_early+st_early) if (cl_early+st_early)>0 else 0,
                                  cl_late/(cl_late+st_late)   if (cl_late+st_late)>0 else 0,
                                  "{:.1%}")
_cmp_row("large_fraction",        cmp_s6["is_large"].sum()/max(len(cmp_s6),1),
                                  s6["is_large"].sum()/max(len(s6),1),
                                  "{:.1%}")
p()

# ---------------------------------------------------------------------------
# FINAL SUBSET OUTCOME DISTRIBUTIONS
# ---------------------------------------------------------------------------

p("=" * 80)
p("FINAL FILTERED SUBSET OUTCOME DISTRIBUTIONS")
p("(10:30-11:00 + all 5 filters, step 6)")
p("=" * 80)
p()

final = s6
n_final = len(final)
large_final = final[final["is_large"]]
clean_final = final[final["is_clean_large"]]

p(f"  Total events in final subset : {n_final:,}")
p(f"  Events per trading day       : {n_final/TRADING_DAYS:.2f}")
p()

if len(large_final) > 0:
    mfe_vals = large_final["mfe_raw"].dropna()
    mae_vals = large_final["mae_raw"].dropna()
    p("  LARGE WINNERS (is_large = True) within final subset:")
    p(f"    count          : {len(large_final):,}")
    p(f"    mean mfe_raw   : {mfe_vals.mean():.1f} pts")
    p(f"    median mfe_raw : {mfe_vals.median():.1f} pts")
    p(f"    p90 mfe_raw    : {mfe_vals.quantile(0.9):.1f} pts")
    p(f"    mean mae_raw   : {mae_vals.mean():.1f} pts")
    p(f"    median mae_raw : {mae_vals.median():.1f} pts")
    p()

if len(clean_final) > 0:
    mfe_c = clean_final["mfe_raw"].dropna()
    mae_c = clean_final["mae_raw"].dropna()
    p("  CLEAN LARGE WINNERS (mae_raw < 10) within final subset:")
    p(f"    count          : {len(clean_final):,}")
    p(f"    mean mfe_raw   : {mfe_c.mean():.1f} pts")
    p(f"    median mfe_raw : {mfe_c.median():.1f} pts")
    p(f"    p90 mfe_raw    : {mfe_c.quantile(0.9):.1f} pts")
    p(f"    mean mae_raw   : {mae_c.mean():.1f} pts")
    p(f"    median mae_raw : {mae_c.median():.1f} pts")
    p()

p(f"  Coverage of all large events : {len(large_final)}/{n_large_total} = {len(large_final)/n_large_total*100:.1f}%")
p()

# ---------------------------------------------------------------------------
# LOOSE CLEAN DEFINITION (mae_raw < 20) — same filter stack
# ---------------------------------------------------------------------------

p("=" * 80)
p("LOOSE CLEAN DEFINITION: mae_raw < 20")
p("Same step 2-6 filter stack, target = large AND mae_raw < 20")
p("=" * 80)
p()

def _print_loose_step(label, df):
    n_total   = len(df)
    n_cl20    = int(df["is_loose_large"].sum())   # mae < 20
    n_cl10    = int(df["is_clean_large"].sum())   # mae < 10
    n_st      = int(df["is_stress_large"].sum())  # mae >= 40
    n_large   = int(df["is_large"].sum())
    per_day   = n_total / TRADING_DAYS
    frac_loose = n_cl20 / (n_cl20 + n_st) if (n_cl20 + n_st) > 0 else 0.0
    frac_clean = n_cl10 / (n_cl10 + n_st) if (n_cl10 + n_st) > 0 else 0.0
    flag = _flag(per_day)
    p(f"  {label}")
    p(f"    total           : {n_total:>6,}   per/day={per_day:.2f}   large={n_large:,}")
    p(f"    clean (mae<10)  : {n_cl10:>6,}   clean10_frac={frac_clean*100:.1f}%")
    p(f"    loose (mae<20)  : {n_cl20:>6,}   loose20_frac={frac_loose*100:.1f}%")
    p(f"    stress (mae>=40): {n_st:>6,}")
    if flag:
        p(flag)
    p()

_print_loose_step("Step 1 — baseline", fv)
_print_loose_step("Step 2 — 10:30-11:00", s2)
_print_loose_step("Step 3 — + vwap_dist<=50", s3)
_print_loose_step("Step 4 — + level proximity", s4)
_print_loose_step("Step 5 — + delta aligned", s5)
_print_loose_step("Step 6 — + overnight trend", s6)

# ---------------------------------------------------------------------------
# TIME WINDOW SENSITIVITY ANALYSIS
# ---------------------------------------------------------------------------

p("=" * 80)
p("TIME WINDOW SENSITIVITY ANALYSIS")
p("Each window shows events/day and clean_fraction (clean / clean+stress)")
p("Filters 3-6 (vwap_dist, level proximity, delta, overnight) applied within each window")
p("=" * 80)
p()

windows = [
    ("09:30-11:00 (full)",   9*60+30,  11*60+0),
    ("09:45-10:15",          9*60+45,  10*60+15),
    ("10:00-10:30",         10*60+0,  10*60+30),
    ("10:15-10:45",         10*60+15, 10*60+45),
    ("10:30-11:00",         10*60+30, 11*60+0),
]

p(f"  {'Window':<22}  {'n_raw':>7}  {'n_filt':>7}  {'raw/d':>7}  {'filt/d':>7}  {'clean%_raw':>10}  {'clean%_filt':>11}  {'large%_filt':>11}")
p(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*11}  {'-'*11}")

for wlabel, hhmm_lo, hhmm_hi in windows:
    wdf = fv[(fv["_et_hhmm"] >= hhmm_lo) & (fv["_et_hhmm"] < hhmm_hi)].copy()
    # Apply filters 3-6
    wf = wdf[wdf["vwap_dist_pts"].abs() <= 50.0]
    wf = wf[(wf["val_dist_pts"].abs() <= 100.0) | (wf["vah_dist_pts"].abs() <= 100.0)]
    wf = wf[wf["delta_aligned"] > 0]
    wf = wf[wf["ov_trend_above_median"]]

    def _cfrac(df):
        c = df["is_clean_large"].sum()
        s = df["is_stress_large"].sum()
        return c / (c + s) * 100 if (c + s) > 0 else 0.0

    n_raw  = len(wdf)
    n_filt = len(wf)
    raw_d  = n_raw  / TRADING_DAYS
    filt_d = n_filt / TRADING_DAYS
    cf_raw = _cfrac(wdf)
    cf_filt = _cfrac(wf)
    lf_filt = wf["is_large"].sum() / max(n_filt, 1) * 100

    flag = ""
    if filt_d < 0.5:
        flag = " *** CRITICAL ***"
    elif filt_d < 1.0:
        flag = " ** WARNING **"

    p(f"  {wlabel:<22}  {n_raw:>7,}  {n_filt:>7,}  {raw_d:>7.2f}  {filt_d:>7.2f}  {cf_raw:>9.1f}%  {cf_filt:>10.1f}%  {lf_filt:>10.1f}%{flag}")

p()

# ---------------------------------------------------------------------------
# TRADEOFF SUMMARY
# ---------------------------------------------------------------------------

p("=" * 80)
p("TRADEOFF SUMMARY")
p("=" * 80)
p()

steps_data = [
    ("Step 1 — baseline",             m1),
    ("Step 2 — 10:30-11:00",          m2),
    ("Step 3 — + vwap_dist<=50",      m3),
    ("Step 4 — + level proximity",    m4),
    ("Step 5 — + delta aligned",      m5),
    ("Step 6 — + overnight trend",    m6),
]

target_cf = 0.40
target_pd = 1.0
best_combined = None

p(f"  {'Step':<35}  {'per_day':>8}  {'clean%':>8}  {'large%':>8}  {'target?':>10}")
p(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
for slabel, sm in steps_data:
    cf = sm["clean_frac"] * 100
    pd_ = sm["per_day"]
    lf = sm["large_frac"] * 100
    meets = "YES ✓" if (cf >= target_cf*100 and pd_ >= target_pd) else "no"
    freq_flag = ""
    if pd_ < 0.5:
        freq_flag = " CRITICAL"
    elif pd_ < 1.0:
        freq_flag = " WARNING"
    p(f"  {slabel:<35}  {pd_:>8.2f}  {cf:>7.1f}%  {lf:>7.1f}%  {meets}{freq_flag}")
    if cf >= target_cf*100 and pd_ >= target_pd and best_combined is None:
        best_combined = (slabel, sm)

p()

if best_combined is not None:
    p(f"  ✓ Both targets met first at: {best_combined[0]}")
    p(f"    events/day = {best_combined[1]['per_day']:.2f}  clean_fraction = {best_combined[1]['clean_frac']*100:.1f}%")
else:
    p("  ✗ No single filter step achieves both targets simultaneously.")
    p()
    # Find best tradeoff: maximize clean_frac subject to per_day > 1.0
    freq_ok = [(sl, sm) for sl, sm in steps_data if sm["per_day"] >= 1.0]
    if freq_ok:
        best_freq_ok = max(freq_ok, key=lambda x: x[1]["clean_frac"])
        p(f"  Best clean_fraction with per_day >= 1.0: {best_freq_ok[0]}")
        p(f"    events/day = {best_freq_ok[1]['per_day']:.2f}  clean_fraction = {best_freq_ok[1]['clean_frac']*100:.1f}%")
        p()
        p("  RECOMMENDATION: To recover frequency, loosen filters in this order:")
        p("    1. Remove OVERNIGHT TREND filter (step 6) — overnight features noisy, small KS gain")
        p("    2. Widen LEVEL PROXIMITY to 200 pts (step 4) — val/vah distances have high variance")
        p("    3. Widen VWAP PROXIMITY to 100 pts (step 3) — some valid setups exist farther from VWAP")
        p("    4. Expand TIME WINDOW to 10:15-11:00 or 10:00-11:00 (step 2)")
    p()

p("=" * 80)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

report_text = "\n".join(lines)
REPORT_PATH.write_text(report_text, encoding="utf-8")
print(f"\nReport saved to {REPORT_PATH}")

try:
    print("\n" + report_text)
except UnicodeEncodeError:
    print("\n" + report_text.encode("ascii", errors="replace").decode("ascii"))

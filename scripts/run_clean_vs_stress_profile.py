"""
Clean vs stress large winner profile analysis.

Groups:
  clean_large  : mae_raw <  10 AND mfe_raw >= 50 AND mfe_raw > mae_raw  (n~2148)
  stress_large : mae_raw >= 40 AND mfe_raw >= 50 AND mfe_raw > mae_raw
  loss         : mfe_raw <  10 OR  mae_raw > mfe_raw

Computes pairwise separation + time-of-day, location, overnight, volume, delta profiles.
Saves output/diagnostics/clean_vs_stress_profile.txt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR     = PROJECT_ROOT / "output" / "diagnostics"
FV_PATH     = OUT_DIR / "feature_vectors.csv"
REPORT_PATH = OUT_DIR / "clean_vs_stress_profile.txt"

ET = "America/New_York"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading feature vectors...")
fv = pd.read_csv(FV_PATH)
print(f"  Rows: {len(fv)}")

# ---------------------------------------------------------------------------
# Direction, mfe_raw, mae_raw
# ---------------------------------------------------------------------------

def _direction(loc, std2_dist_upper, std2_dist_lower):
    if loc in ("outside_vah", "at_vah"):
        return "short"
    if loc in ("outside_val", "at_val"):
        return "long"
    if not (pd.isna(std2_dist_upper) or pd.isna(std2_dist_lower)):
        if std2_dist_upper >= 0:
            return "short"
        if std2_dist_lower <= 0:
            return "long"
    return "unknown"

fv["_dir"] = fv.apply(
    lambda r: _direction(
        str(r["price_location"]),
        r.get("std2_dist_upper", float("nan")),
        r.get("std2_dist_lower", float("nan")),
    ), axis=1
)

fv["_mfe_raw"] = np.where(fv["_dir"] == "short", fv["mfe_down"],
                  np.where(fv["_dir"] == "long",  fv["mfe_up"],  float("nan")))
fv["_mae_raw"] = np.where(fv["_dir"] == "short", fv["mfe_up"],
                  np.where(fv["_dir"] == "long",  fv["mfe_down"], float("nan")))

# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

def _bucket(mfe, mae):
    if pd.isna(mfe) or pd.isna(mae):
        return "unknown"
    if mfe < 10.0 or mae > mfe:
        return "loss"
    if mfe >= 50.0 and mae < 10.0:
        return "clean_large"
    if mfe >= 50.0 and mae >= 40.0:
        return "stress_large"
    if mfe >= 50.0:
        return "large_other"   # 10 <= mae < 40
    return "other"

fv["_group"] = fv.apply(lambda r: _bucket(r["_mfe_raw"], r["_mae_raw"]), axis=1)

grp = {g: fv[fv["_group"] == g] for g in ["clean_large", "stress_large", "loss"]}
counts = {g: len(grp[g]) for g in grp}

print(f"  clean_large  : {counts['clean_large']:,}")
print(f"  stress_large : {counts['stress_large']:,}")
print(f"  loss         : {counts['loss']:,}")

# ---------------------------------------------------------------------------
# Feature separation helpers
# ---------------------------------------------------------------------------

_EXCLUDE_COLS = {
    "session_date", "event_time", "price_location", "overnight_regime_str",
    "mfe_up", "mfe_down", "net_move",
    "_dir", "_mfe_raw", "_mae_raw", "_group",
}
all_num = [c for c in fv.columns
           if c not in _EXCLUDE_COLS and pd.api.types.is_numeric_dtype(fv[c])]
ov_cols  = [c for c in all_num if c.startswith("ov_")]
std_cols = [c for c in all_num if c not in ov_cols]


def sep_ratio(a_vals, b_vals):
    if len(a_vals) < 2 or len(b_vals) < 2:
        return float("nan")
    am, bm = np.mean(a_vals), np.mean(b_vals)
    av, bv = np.var(a_vals, ddof=1), np.var(b_vals, ddof=1)
    n_a, n_b = len(a_vals), len(b_vals)
    ps = np.sqrt((av * (n_a - 1) + bv * (n_b - 1)) / (n_a + n_b - 2))
    return abs(am - bm) / ps if ps > 1e-12 else 0.0


def ks(a_vals, b_vals):
    if len(a_vals) < 2 or len(b_vals) < 2:
        return float("nan"), float("nan")
    s, p = stats.ks_2samp(a_vals, b_vals)
    return float(s), float(p)


def col_means(col, groups):
    return {g: float(grp[g][col].dropna().mean()) for g in groups}


# Pairwise comparisons
PAIRS = [
    ("clean_large", "loss",         "cl_vs_loss"),
    ("clean_large", "stress_large", "cl_vs_stress"),
    ("stress_large","loss",         "stress_vs_loss"),
]


def feature_stats(col):
    row = {"feature": col}
    for g in ["clean_large", "stress_large", "loss"]:
        v = grp[g][col].dropna().to_numpy(dtype=float)
        row[f"{g}_mean"] = float(np.mean(v)) if len(v) else float("nan")
    for ga, gb, label in PAIRS:
        a = grp[ga][col].dropna().to_numpy(dtype=float)
        b = grp[gb][col].dropna().to_numpy(dtype=float)
        row[f"{label}_sep"] = sep_ratio(a, b)
        row[f"{label}_ks"],  row[f"{label}_ksp"] = ks(a, b)
    return row


print("Computing feature separation stats...")
std_stats = [feature_stats(c) for c in std_cols]
feat_df   = pd.DataFrame(std_stats).sort_values("cl_vs_loss_sep", ascending=False)

ov_stats = []
for c in ov_cols:
    row = {"feature": c}
    for g in ["clean_large", "stress_large", "loss"]:
        v = grp[g][c].dropna().to_numpy(dtype=float)
        row[f"{g}_med_abs"] = float(np.median(np.abs(v))) if len(v) else float("nan")
    for ga, gb, label in PAIRS:
        a = grp[ga][c].dropna().to_numpy(dtype=float)
        b = grp[gb][c].dropna().to_numpy(dtype=float)
        row[f"{label}_ks"], row[f"{label}_ksp"] = ks(a, b)
    ov_stats.append(row)
ov_df = pd.DataFrame(ov_stats).sort_values("cl_vs_stress_ks", ascending=False)

# ---------------------------------------------------------------------------
# Q1: Time of day (15-min buckets, 09:30-11:00)
# ---------------------------------------------------------------------------

print("Computing time-of-day analysis...")

fv["_event_et"] = pd.to_datetime(fv["event_time"], utc=True).dt.tz_convert(ET)
fv["_minute"]   = fv["_event_et"].dt.hour * 60 + fv["_event_et"].dt.minute

buckets_15m = [(9*60+30 + i*15, 9*60+30 + (i+1)*15) for i in range(6)]  # 09:30-11:00

tod_rows = []
for lo, hi in buckets_15m:
    h_lo, m_lo = divmod(lo, 60)
    h_hi, m_hi = divmod(hi, 60)
    label = f"{h_lo:02d}:{m_lo:02d}-{h_hi:02d}:{m_hi:02d}"
    in_win = (fv["_minute"] >= lo) & (fv["_minute"] < hi)
    n_cl   = int(((fv["_group"] == "clean_large")  & in_win).sum())
    n_st   = int(((fv["_group"] == "stress_large") & in_win).sum())
    total  = n_cl + n_st
    frac   = n_cl / total if total > 0 else float("nan")
    n_loss = int(((fv["_group"] == "loss") & in_win).sum())
    tod_rows.append({"window": label, "clean": n_cl, "stress": n_st,
                     "clean_frac": frac, "loss": n_loss})

tod_df = pd.DataFrame(tod_rows)

# ---------------------------------------------------------------------------
# Q2: Location breakdown
# ---------------------------------------------------------------------------

print("Computing location breakdown...")

loc_rows = []
for loc in sorted(fv["price_location"].dropna().unique()):
    sub = fv[fv["price_location"] == loc]
    n_tot  = len(sub)
    n_lrg  = int((sub["_group"].isin(["clean_large", "stress_large", "large_other"])).sum())
    n_cl   = int((sub["_group"] == "clean_large").sum())
    n_st   = int((sub["_group"] == "stress_large").sum())
    n_loss = int((sub["_group"] == "loss").sum())
    denom  = n_cl + n_st
    cl_frac = n_cl / denom if denom > 0 else float("nan")
    loc_rows.append({"location": loc, "total_events": n_tot,
                     "large_events": n_lrg, "clean": n_cl,
                     "stress": n_st, "loss": n_loss,
                     "clean_frac": cl_frac})

loc_df = pd.DataFrame(loc_rows).sort_values("clean_frac", ascending=False)

# ---------------------------------------------------------------------------
# Q3: Overnight feature KS (clean_large vs stress_large)
# Already in ov_df sorted by cl_vs_stress_ks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Q4: Volume profile
# ---------------------------------------------------------------------------

vol_cols = [c for c in std_cols if "mean_vol" in c]
vol_rows = []
for c in sorted(vol_cols):
    row = {"feature": c}
    for g in ["clean_large", "stress_large", "loss"]:
        row[g] = float(grp[g][c].dropna().mean())
    vol_rows.append(row)
vol_df = pd.DataFrame(vol_rows)

# ---------------------------------------------------------------------------
# Q5: Per-level delta alignment
# ---------------------------------------------------------------------------

# Delta alignment: does cumulative_delta in lookback window match expected direction?
# SHORT: aligned = cumulative_delta < 0 (more sell pressure)
# LONG:  aligned = cumulative_delta > 0 (more buy pressure)

delta_cols = [c for c in std_cols if "cumulative_delta" in c or "mean_delta_pct" in c]
delta_rows = []

for g in ["clean_large", "stress_large", "loss"]:
    sub = grp[g].copy()
    for col in sorted(delta_cols):
        vals = sub[col].dropna()
        # Signed: short expects negative delta, long expects positive
        # Create direction-aligned delta: multiply by +1 for long, -1 for short
        aligned = np.where(sub.loc[vals.index, "_dir"] == "long", vals, -vals)
        delta_rows.append({
            "feature": col,
            "group": g,
            "mean_raw": float(vals.mean()),
            "mean_aligned": float(np.mean(aligned)),
            "pct_aligned": float((aligned > 0).mean() * 100),
        })

delta_df = pd.DataFrame(delta_rows)
delta_pivot = delta_df.pivot_table(
    index="feature",
    columns="group",
    values=["mean_aligned", "pct_aligned"],
    aggfunc="first"
).round(3)

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

lines = []
def p(s=""): lines.append(s)

data_start = fv["session_date"].min() if "session_date" in fv.columns else "?"
data_end   = fv["session_date"].max() if "session_date" in fv.columns else "?"

p("=" * 80)
p("CLEAN vs STRESS LARGE WINNER PROFILE")
p(f"Training: {data_start} to {data_end}")
p("=" * 80)
p()
p("GROUP DEFINITIONS:")
p("  clean_large  : mfe_raw >= 50 AND mae_raw <  10 (minimal adverse, clean entry)")
p("  stress_large : mfe_raw >= 50 AND mae_raw >= 40 (large adverse survived before big move)")
p("  loss         : mfe_raw <  10 OR mae_raw > mfe_raw")
p()
p(f"  clean_large  : {counts['clean_large']:>6,} events")
p(f"  stress_large : {counts['stress_large']:>6,} events")
p(f"  loss         : {counts['loss']:>6,} events")
p(f"  large_other  : {int((fv['_group'] == 'large_other').sum()):>6,} events  (10 <= mae < 40, excluded from comparisons)")
p(f"  other/unknown: {int(fv['_group'].isin(['other','unknown']).sum()):>6,} events")
p()

# ── Feature separation ────────────────────────────────────────────────────────
p("=" * 80)
p("SECTION A  --  TOP 20 FEATURES by clean_large vs loss separation ratio")
p("sep_ratio = |mean_clean - mean_loss| / pooled_std")
p("Pairs: cl=clean_large, st=stress_large, ls=loss")
p("=" * 80)
p()

fw = 38
hdr = (f"{'feature':<{fw}}  {'cl_mean':>9}  {'st_mean':>9}  {'ls_mean':>9}  "
       f"{'cl_vs_ls':>9}  {'cl_vs_st':>9}  {'st_vs_ls':>9}")
sep_line = "-" * len(hdr)

p(hdr)
p(sep_line)
for _, r in feat_df.head(20).iterrows():
    p(
        f"{r['feature']:<{fw}}  "
        f"{r['clean_large_mean']:>9.3f}  {r['stress_large_mean']:>9.3f}  {r['loss_mean']:>9.3f}  "
        f"{r['cl_vs_loss_sep']:>9.4f}  {r['cl_vs_stress_sep']:>9.4f}  {r['stress_vs_loss_sep']:>9.4f}"
    )
p()

p("ALL STANDARD FEATURES (cl_vs_loss order)")
p(hdr)
p(sep_line)
for _, r in feat_df.iterrows():
    p(
        f"{r['feature']:<{fw}}  "
        f"{r['clean_large_mean']:>9.3f}  {r['stress_large_mean']:>9.3f}  {r['loss_mean']:>9.3f}  "
        f"{r['cl_vs_loss_sep']:>9.4f}  {r['cl_vs_stress_sep']:>9.4f}  {r['stress_vs_loss_sep']:>9.4f}"
    )
p()

# ── Q1: Time of day ───────────────────────────────────────────────────────────
p("=" * 80)
p("SECTION B  --  Q1: TIME OF DAY (15-min buckets, 09:30-11:00 ET)")
p("clean_frac = clean_large / (clean_large + stress_large)")
p("=" * 80)
p()
p(f"  {'window':<16}  {'clean':>7}  {'stress':>7}  {'loss':>7}  {'clean_frac':>12}")
p("  " + "-" * 55)
for _, r in tod_df.iterrows():
    cf = f"{r['clean_frac']*100:.1f}%" if not pd.isna(r["clean_frac"]) else "  n/a"
    p(f"  {r['window']:<16}  {r['clean']:>7}  {r['stress']:>7}  {r['loss']:>7}  {cf:>12}")
p()

# ── Q2: Location breakdown ────────────────────────────────────────────────────
p("=" * 80)
p("SECTION C  --  Q2: LOCATION BREAKDOWN")
p("clean_frac = clean_large / (clean_large + stress_large)")
p("Sorted by clean_frac descending")
p("=" * 80)
p()
p(f"  {'location':<16}  {'total':>7}  {'large':>7}  {'clean':>7}  {'stress':>7}  {'loss':>7}  {'clean_frac':>12}")
p("  " + "-" * 72)
for _, r in loc_df.iterrows():
    cf = f"{r['clean_frac']*100:.1f}%" if not pd.isna(r["clean_frac"]) else "  n/a"
    p(
        f"  {r['location']:<16}  {r['total_events']:>7}  {r['large_events']:>7}  "
        f"{r['clean']:>7}  {r['stress']:>7}  {r['loss']:>7}  {cf:>12}"
    )
p()

# ── Q3: Overnight features ────────────────────────────────────────────────────
p("=" * 80)
p("SECTION D  --  Q3: OVERNIGHT FEATURES (KS stat: clean_large vs stress_large)")
p("Note: ov_* features are z-scored per session; median_abs used for magnitude.")
p("Sorted by cl_vs_stress KS statistic.")
p("=" * 80)
p()
ov_fw = 32
ov_hdr = (
    f"  {'feature':<{ov_fw}}  {'cl_med_abs':>10}  {'st_med_abs':>10}  {'ls_med_abs':>10}  "
    f"{'cl_vs_st_ks':>12}  {'cl_vs_ls_ks':>12}  {'st_vs_ls_ks':>12}"
)
p(ov_hdr)
p("  " + "-" * (len(ov_hdr) - 2))
for _, r in ov_df.iterrows():
    p(
        f"  {r['feature']:<{ov_fw}}  "
        f"{r['clean_large_med_abs']:>10.4f}  {r['stress_large_med_abs']:>10.4f}  {r['loss_med_abs']:>10.4f}  "
        f"{r['cl_vs_stress_ks']:>12.4f}  {r['cl_vs_loss_ks']:>12.4f}  {r['stress_vs_loss_ks']:>12.4f}"
    )
p()

# ── Q4: Volume profile ────────────────────────────────────────────────────────
p("=" * 80)
p("SECTION E  --  Q4: VOLUME PROFILE (mean_vol at each lookback window)")
p("Hypothesis: low volume predicts clean entries")
p("=" * 80)
p()
p(f"  {'feature':<22}  {'clean_large':>12}  {'stress_large':>13}  {'loss':>12}  {'cl_lt_st?':>10}")
p("  " + "-" * 76)
for _, r in vol_df.iterrows():
    lt_flag = "YES" if r["clean_large"] < r["stress_large"] else "no"
    p(
        f"  {r['feature']:<22}  {r['clean_large']:>12.1f}  {r['stress_large']:>13.1f}  "
        f"{r['loss']:>12.1f}  {lt_flag:>10}"
    )
p()

# ── Q5: Delta alignment ───────────────────────────────────────────────────────
p("=" * 80)
p("SECTION F  --  Q5: PER-LEVEL DELTA ALIGNMENT")
p("mean_aligned: cumulative delta signed in trade direction (+ve = aligned with trade)")
p("  SHORT: aligned = negative delta (sell pressure)")
p("  LONG:  aligned = positive delta (buy pressure)")
p("pct_aligned: % of events where delta was in trade direction")
p("=" * 80)
p()

from nqbt.analysis.feature_vector_builder import LOOKBACK_WINDOWS
cum_delta_cols = sorted([c for c in delta_cols if "cumulative_delta" in c],
                        key=lambda x: int(x.split("w")[1].split("_")[0]))

p(f"  {'feature':<28}  {'cl_aligned':>12}  {'st_aligned':>12}  {'ls_aligned':>12}  "
  f"{'cl_pct%':>8}  {'st_pct%':>8}  {'ls_pct%':>8}")
p("  " + "-" * 94)
for col in cum_delta_cols:
    rows_col = delta_df[delta_df["feature"] == col]
    d = {r["group"]: r for _, r in rows_col.iterrows()}
    cl = d.get("clean_large",  {})
    st = d.get("stress_large", {})
    ls = d.get("loss",         {})
    p(
        f"  {col:<28}  "
        f"{cl.get('mean_aligned', float('nan')):>12.2f}  "
        f"{st.get('mean_aligned', float('nan')):>12.2f}  "
        f"{ls.get('mean_aligned', float('nan')):>12.2f}  "
        f"{cl.get('pct_aligned', float('nan')):>8.1f}  "
        f"{st.get('pct_aligned', float('nan')):>8.1f}  "
        f"{ls.get('pct_aligned', float('nan')):>8.1f}"
    )
p()

# mean_delta_pct
mean_delta_pct_cols = sorted([c for c in delta_cols if "mean_delta_pct" in c],
                              key=lambda x: int(x.split("w")[1].split("_")[0]))
p("  mean_delta_pct (direction-aligned, +ve = trade direction):")
p(f"  {'feature':<28}  {'clean_large':>12}  {'stress_large':>12}  {'loss':>12}")
p("  " + "-" * 68)
for col in mean_delta_pct_cols:
    rows_col = delta_df[delta_df["feature"] == col]
    d = {r["group"]: r for _, r in rows_col.iterrows()}
    cl = d.get("clean_large",  {})
    st = d.get("stress_large", {})
    ls = d.get("loss",         {})
    p(
        f"  {col:<28}  "
        f"{cl.get('mean_aligned', float('nan')):>12.4f}  "
        f"{st.get('mean_aligned', float('nan')):>12.4f}  "
        f"{ls.get('mean_aligned', float('nan')):>12.4f}"
    )
p()

# ── Summary ───────────────────────────────────────────────────────────────────
p("=" * 80)
p("KEY FINDINGS SUMMARY")
p("=" * 80)
p()

# Top feature cl_vs_loss
top_feat = feat_df.iloc[0]
p(f"  Top feature (cl vs loss): {top_feat['feature']}")
p(f"    clean_mean={top_feat['clean_large_mean']:.3f}  stress_mean={top_feat['stress_large_mean']:.3f}  loss_mean={top_feat['loss_mean']:.3f}")
p(f"    cl_vs_loss sep={top_feat['cl_vs_loss_sep']:.4f}  cl_vs_stress sep={top_feat['cl_vs_stress_sep']:.4f}")
p()

# Volume finding
if not vol_df.empty:
    w10_row = vol_df[vol_df["feature"] == "w10_mean_vol"]
    if not w10_row.empty:
        r = w10_row.iloc[0]
        p(f"  Volume (w10_mean_vol): clean={r['clean_large']:.0f}  stress={r['stress_large']:.0f}  loss={r['loss']:.0f}")
        direction = "LOWER" if r["clean_large"] < r["stress_large"] else "HIGHER"
        p(f"    Clean entries have {direction} pre-entry volume than stress entries.")
p()

# Best clean location
if not loc_df.empty:
    best_loc = loc_df[loc_df["clean"] >= 20].iloc[0] if len(loc_df[loc_df["clean"] >= 20]) > 0 else loc_df.iloc[0]
    p(f"  Cleanest location (min 20 clean events): {best_loc['location']}")
    p(f"    clean_frac={best_loc['clean_frac']*100:.1f}%  clean={best_loc['clean']:.0f}  stress={best_loc['stress']:.0f}")
p()

# Top overnight feature
if not ov_df.empty:
    top_ov = ov_df.iloc[0]
    p(f"  Top overnight feature (cl vs stress KS): {top_ov['feature']}")
    p(f"    cl_vs_stress KS={top_ov['cl_vs_stress_ks']:.4f}  p={top_ov['cl_vs_stress_ksp']:.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

report_text = "\n".join(lines)
REPORT_PATH.write_text(report_text, encoding="utf-8")
print(f"\nReport saved to {REPORT_PATH}")

try:
    print("\n" + "\n".join(lines[:50]))
except UnicodeEncodeError:
    safe = "\n".join(lines[:50]).encode("ascii", errors="replace").decode("ascii")
    print("\n" + safe)

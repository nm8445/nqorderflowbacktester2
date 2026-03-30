"""
Feature profiling with 4-tier outcome bucketing.

Outcome tiers (based on mfe_raw and mae_raw in NQ points):
  loss:   mfe_raw <  10 OR mae_raw > mfe_raw
  scalp:  mfe_raw >= 10 AND <  25 AND mfe_raw > mae_raw
  medium: mfe_raw >= 25 AND <  50 AND mfe_raw > mae_raw
  large:  mfe_raw >= 50          AND mfe_raw > mae_raw

mfe_raw / mae_raw are absolute NQ points:
  short setup (outside_vah, at_vah): mfe_raw = mfe_down, mae_raw = mfe_up
  long  setup (outside_val, at_val): mfe_raw = mfe_up,   mae_raw = mfe_down
  inside_va / VWAP-band only:        determined by band proximity side

Pairwise comparisons: loss_vs_scalp, loss_vs_medium, loss_vs_large, scalp_vs_large

Overnight features (ov_*) are z-scored per session so cross-event means ~= 0.
Ranked by KS statistic per pair instead of mean-based sep_ratio.

Saves output/diagnostics/feature_profiling_report.txt
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
REPORT_PATH = OUT_DIR / "feature_profiling_report.txt"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading feature vectors...")
fv = pd.read_csv(FV_PATH)

if fv.empty:
    print("No feature vectors found. Run run_new_study.py first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Assign direction and compute mfe_raw / mae_raw
# ---------------------------------------------------------------------------

def _assign_direction(row: pd.Series) -> str:
    loc = str(row.get("price_location", "inside_va"))
    if loc in ("outside_vah", "at_vah"):
        return "short"
    if loc in ("outside_val", "at_val"):
        return "long"
    std2_dist_upper = row.get("std2_dist_upper", float("nan"))
    std2_dist_lower = row.get("std2_dist_lower", float("nan"))
    if pd.isna(std2_dist_upper) or pd.isna(std2_dist_lower):
        return "unknown"
    if std2_dist_upper >= 0:
        return "short"
    if std2_dist_lower <= 0:
        return "long"
    return "unknown"


fv["_direction"] = fv.apply(_assign_direction, axis=1)

# mfe_raw / mae_raw per direction
def _mfe_raw(row: pd.Series) -> float:
    d = row["_direction"]
    if d == "short":
        return float(row.get("mfe_down", float("nan")))
    if d == "long":
        return float(row.get("mfe_up", float("nan")))
    return float("nan")


def _mae_raw(row: pd.Series) -> float:
    d = row["_direction"]
    if d == "short":
        return float(row.get("mfe_up", float("nan")))   # adverse for short = upward move
    if d == "long":
        return float(row.get("mfe_down", float("nan"))) # adverse for long = downward move
    return float("nan")


fv["_mfe_raw"] = fv.apply(_mfe_raw, axis=1)
fv["_mae_raw"] = fv.apply(_mae_raw, axis=1)

# ---------------------------------------------------------------------------
# Assign 4-tier outcome bucket
# ---------------------------------------------------------------------------

def _outcome_bucket(row: pd.Series) -> str:
    mfe = row["_mfe_raw"]
    mae = row["_mae_raw"]
    if pd.isna(mfe) or pd.isna(mae):
        return "unknown"
    if mfe < 10.0 or mae > mfe:
        return "loss"
    if mfe < 25.0:
        return "scalp"
    if mfe < 50.0:
        return "medium"
    return "large"


fv["_bucket"] = fv.apply(_outcome_bucket, axis=1)

fv_valid = fv[fv["_bucket"] != "unknown"].copy()

BUCKETS   = ["loss", "scalp", "medium", "large"]
PAIRS     = [
    ("loss",  "scalp",  "loss_vs_scalp"),
    ("loss",  "medium", "loss_vs_medium"),
    ("loss",  "large",  "loss_vs_large"),
    ("scalp", "large",  "scalp_vs_large"),
]

bucket_counts = {b: int((fv_valid["_bucket"] == b).sum()) for b in BUCKETS}
total_valid   = len(fv_valid)
bucket_groups = {b: fv_valid[fv_valid["_bucket"] == b] for b in BUCKETS}

print(f"  Total valid events: {total_valid}")
for b in BUCKETS:
    pct = 100.0 * bucket_counts[b] / total_valid if total_valid else 0.0
    print(f"    {b:<8}: {bucket_counts[b]:5d} ({pct:.1f}%)")

# ---------------------------------------------------------------------------
# Identify feature columns
# ---------------------------------------------------------------------------

_EXCLUDE_COLS = {
    "session_date", "event_time", "price_location",
    "overnight_regime_str",
    "mfe_up", "mfe_down", "net_move",
    "_direction", "_bucket", "_mfe_raw", "_mae_raw",
}

all_numeric = [
    c for c in fv_valid.columns
    if c not in _EXCLUDE_COLS
    and pd.api.types.is_numeric_dtype(fv_valid[c])
]

overnight_cols = [c for c in all_numeric if c.startswith("ov_")]
standard_cols  = [c for c in all_numeric if c not in overnight_cols]

# ---------------------------------------------------------------------------
# Helper: pairwise sep_ratio and KS for one feature
# ---------------------------------------------------------------------------

def _pair_stats(col: str, grp_a: pd.DataFrame, grp_b: pd.DataFrame) -> dict:
    a_vals = grp_a[col].dropna().to_numpy(dtype=float)
    b_vals = grp_b[col].dropna().to_numpy(dtype=float)

    if len(a_vals) < 2 or len(b_vals) < 2:
        return {"sep_ratio": float("nan"), "ks_stat": float("nan"), "ks_pvalue": float("nan")}

    a_mean = float(np.mean(a_vals))
    b_mean = float(np.mean(b_vals))
    a_var  = float(np.var(a_vals, ddof=1))
    b_var  = float(np.var(b_vals, ddof=1))
    n_a, n_b = len(a_vals), len(b_vals)
    pooled_std = float(np.sqrt(
        (a_var * (n_a - 1) + b_var * (n_b - 1)) / (n_a + n_b - 2)
    ))
    sep_ratio = abs(a_mean - b_mean) / pooled_std if pooled_std > 1e-12 else 0.0

    try:
        ks_stat, ks_pval = stats.ks_2samp(a_vals, b_vals)
    except Exception:
        ks_stat, ks_pval = float("nan"), float("nan")

    return {"sep_ratio": sep_ratio, "ks_stat": ks_stat, "ks_pvalue": ks_pval}

# ---------------------------------------------------------------------------
# Compute stats for all standard features
# ---------------------------------------------------------------------------

std_stats: list[dict] = []

for col in standard_cols:
    row: dict = {"feature": col}
    # Bucket means
    for b in BUCKETS:
        vals = bucket_groups[b][col].dropna().to_numpy(dtype=float)
        row[f"{b}_mean"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    # Pairwise sep + KS
    for grp_a_name, grp_b_name, pair_label in PAIRS:
        ps = _pair_stats(col, bucket_groups[grp_a_name], bucket_groups[grp_b_name])
        row[f"{pair_label}_sep"]  = ps["sep_ratio"]
        row[f"{pair_label}_ks"]   = ps["ks_stat"]
        row[f"{pair_label}_ksp"]  = ps["ks_pvalue"]
    std_stats.append(row)

feat_df = pd.DataFrame(std_stats)
feat_df_by_llg = feat_df.sort_values("loss_vs_large_sep", ascending=False)

# ---------------------------------------------------------------------------
# Overnight features — KS-based (per pair)
# ---------------------------------------------------------------------------

ov_stats: list[dict] = []

for col in overnight_cols:
    row_ov: dict = {"feature": col}
    for b in BUCKETS:
        vals = bucket_groups[b][col].dropna().to_numpy(dtype=float)
        row_ov[f"{b}_med_abs"] = float(np.median(np.abs(vals))) if len(vals) > 0 else float("nan")
    for grp_a_name, grp_b_name, pair_label in PAIRS:
        ps = _pair_stats(col, bucket_groups[grp_a_name], bucket_groups[grp_b_name])
        row_ov[f"{pair_label}_ks"]  = ps["ks_stat"]
        row_ov[f"{pair_label}_ksp"] = ps["ks_pvalue"]
    ov_stats.append(row_ov)

ov_df = pd.DataFrame(ov_stats).sort_values("loss_vs_large_ks", ascending=False)

# ---------------------------------------------------------------------------
# Lookback window ranking by average loss_vs_large sep_ratio
# ---------------------------------------------------------------------------

from nqbt.analysis.feature_vector_builder import LOOKBACK_WINDOWS

window_avgs = []
for w in LOOKBACK_WINDOWS:
    prefix = f"w{w}_"
    w_seps = feat_df[feat_df["feature"].str.startswith(prefix)]["loss_vs_large_sep"].dropna()
    window_avgs.append((w, float(w_seps.mean()) if len(w_seps) > 0 else 0.0))

window_avgs_sorted = sorted(window_avgs, key=lambda x: x[1], reverse=True)


def _best_window(feat: str) -> str:
    for w in LOOKBACK_WINDOWS:
        if feat.startswith(f"w{w}_"):
            return f"w{w}"
    return "n/a"

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

lines: list[str] = []


def p(s: str = "") -> None:
    lines.append(s)


data_start = fv_valid["session_date"].min() if "session_date" in fv_valid.columns else "?"
data_end   = fv_valid["session_date"].max() if "session_date" in fv_valid.columns else "?"

p("=" * 80)
p("FEATURE PROFILING REPORT")
p(f"Training: {data_start} to {data_end}")
p("=" * 80)
p()
p("OUTCOME BUCKET DEFINITIONS (NQ points from entry to session close):")
p("  loss:   mfe_raw <  10 pts  OR  mae_raw > mfe_raw")
p("  scalp:  mfe_raw >= 10 AND < 25 AND mfe_raw > mae_raw")
p("  medium: mfe_raw >= 25 AND < 50 AND mfe_raw > mae_raw")
p("  large:  mfe_raw >= 50          AND mfe_raw > mae_raw")
p()
p("OUTCOME BUCKET DISTRIBUTION:")
p(f"  {'bucket':<10} {'count':>7}  {'pct':>7}")
p("  " + "-" * 28)
for b in BUCKETS:
    pct = 100.0 * bucket_counts[b] / total_valid if total_valid else 0.0
    p(f"  {b:<10} {bucket_counts[b]:>7}  {pct:>6.1f}%")
p(f"  {'TOTAL':<10} {total_valid:>7}  {100.0:>6.1f}%")
p()

# ── Section A: standard features ─────────────────────────────────────────────
p("=" * 80)
p("SECTION A  --  STANDARD FEATURES (ranked by loss_vs_large separation ratio)")
p("sep_ratio = |mean_bucket_a - mean_bucket_b| / pooled_std")
p("=" * 80)
p()

col_w = 38
pair_labels_short = ["L_vs_Sc", "L_vs_Md", "L_vs_Lg", "Sc_vs_Lg"]

hdr = (
    f"{'feature':<{col_w}}  {'loss_mean':>10}  {'scalp_mean':>10}  "
    f"{'medium_mean':>11}  {'large_mean':>10}  "
    + "  ".join(f"{pl:>9}" for pl in pair_labels_short)
)
sep_line = "-" * len(hdr)

p("TOP 20 BY LOSS_VS_LARGE")
p(hdr)
p(sep_line)
for _, row in feat_df_by_llg.head(20).iterrows():
    p(
        f"{row['feature']:<{col_w}}  {row['loss_mean']:>10.3f}  {row['scalp_mean']:>10.3f}  "
        f"{row['medium_mean']:>11.3f}  {row['large_mean']:>10.3f}  "
        + "  ".join(
            f"{row.get(f'{pl}_sep', float('nan')):>9.4f}"
            for _, _, pl in PAIRS
        )
    )
p()

p("ALL STANDARD FEATURES (loss_vs_large order)")
p(hdr)
p(sep_line)
for _, row in feat_df_by_llg.iterrows():
    p(
        f"{row['feature']:<{col_w}}  {row['loss_mean']:>10.3f}  {row['scalp_mean']:>10.3f}  "
        f"{row['medium_mean']:>11.3f}  {row['large_mean']:>10.3f}  "
        + "  ".join(
            f"{row.get(f'{pl}_sep', float('nan')):>9.4f}"
            for _, _, pl in PAIRS
        )
    )
p()

# ── Top 10 detail table ───────────────────────────────────────────────────────
p("=" * 80)
p("SECTION B  --  TOP 10 FEATURES: ALL PAIRS DETAIL")
p("For each top feature, sep_ratio and KS stat for all 4 comparisons")
p("=" * 80)
p()

pair_col_w = 14
detail_hdr = (
    f"{'feature':<{col_w}}  {'loss_mean':>10}  {'scalp_mean':>10}  "
    f"{'medium_mean':>11}  {'large_mean':>10}"
)
p(detail_hdr)
p("-" * len(detail_hdr))
for _, row in feat_df_by_llg.head(10).iterrows():
    p(
        f"{row['feature']:<{col_w}}  {row['loss_mean']:>10.3f}  {row['scalp_mean']:>10.3f}  "
        f"{row['medium_mean']:>11.3f}  {row['large_mean']:>10.3f}"
    )
    # Pair detail lines
    for grp_a, grp_b, pl in PAIRS:
        sep = row.get(f"{pl}_sep",  float("nan"))
        ks  = row.get(f"{pl}_ks",   float("nan"))
        ksp = row.get(f"{pl}_ksp",  float("nan"))
        p(f"    {pl:<22}  sep={sep:.4f}  ks={ks:.4f}  p={ksp:.4f}")
    p()

# ── Section C: lookback window ranking ───────────────────────────────────────
p("=" * 80)
p("SECTION C  --  LOOKBACK WINDOW RANKING")
p("Average loss_vs_large sep_ratio across all features in each window")
p("=" * 80)
p()
for w, avg_sep in window_avgs_sorted:
    p(f"  w{w:<4}: avg_loss_vs_large_sep = {avg_sep:.4f}")
p()

# ── Section D: overnight features ────────────────────────────────────────────
p("=" * 80)
p("SECTION D  --  OVERNIGHT FEATURES (ranked by loss_vs_large KS statistic)")
p("Note: ov_* features are z-scored per session; cross-event means ~ 0.")
p("      med_abs = median of |feature value| per bucket (magnitude proxy).")
p("=" * 80)
p()

ov_col_w = 34
ov_hdr = (
    f"{'feature':<{ov_col_w}}  {'loss_mabs':>9}  {'scalp_mabs':>10}  "
    f"{'med_mabs':>8}  {'large_mabs':>10}  "
    + "  ".join(f"{pl:>9}_ks" for pl in pair_labels_short)
)
p(ov_hdr)
p("-" * len(ov_hdr))
for _, row in ov_df.iterrows():
    ks_vals = "  ".join(
        f"{row.get(f'{pl}_ks', float('nan')):>11.4f}"
        for _, _, pl in PAIRS
    )
    p(
        f"{row['feature']:<{ov_col_w}}  "
        f"{row['loss_med_abs']:>9.4f}  {row['scalp_med_abs']:>10.4f}  "
        f"{row['medium_med_abs']:>8.4f}  {row['large_med_abs']:>10.4f}  "
        + ks_vals
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

report_text = "\n".join(lines)
REPORT_PATH.write_text(report_text, encoding="utf-8")
print(f"\nReport saved to {REPORT_PATH}")

# Safe console preview
preview_lines = lines[:40]
try:
    print("\n" + "\n".join(preview_lines))
except UnicodeEncodeError:
    safe = "\n".join(preview_lines).encode("ascii", errors="replace").decode("ascii")
    print("\n" + safe)

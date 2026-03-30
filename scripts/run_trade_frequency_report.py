"""
Trade frequency report for all three setups.
- Total signals, days with signal, avg/day, 0/1/2/3+ distribution, monthly breakdown
- Combined (union of all three)
- Sensitivity: long reversal proven node filter
- Time window distribution by 30-min bucket (signals are 09:30-11:00)
Saves to output/diagnostics/trade_frequency_report.txt
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from nqbt.analysis.signal_diagnostics import signal_clustering, prior_absorption_count

CSV_PATH = PROJECT_ROOT / "output" / "absorption_signals_labeled.csv"
OUT_DIR  = PROJECT_ROOT / "output" / "diagnostics"
OUT_PATH = OUT_DIR / "trade_frequency_report.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ET = "America/New_York"
TRAINING_START = pd.Timestamp("2025-03-17")
TRAINING_END   = pd.Timestamp("2025-11-30")
TOTAL_DAYS     = 183   # trading days in training window

# ── Load & prep ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
for col in ["sustained", "sustained_loose", "sustained_strict",
            "reversal_rejection_confirmed", "post_absorption_aggression_3bar",
            "node_proven_20t", "at_vwap_band"]:
    df[col] = df[col].astype(bool)

# Signal time in ET
sig_time_et = pd.to_datetime(df["signal_time"], utc=True).dt.tz_convert(ET)
df["_time_et"]  = sig_time_et
df["_date_et"]  = sig_time_et.dt.date
df["_month"]    = sig_time_et.dt.to_period("M")
df["_hour_min"] = sig_time_et.dt.hour * 60 + sig_time_et.dt.minute

# ── Compute derived columns ───────────────────────────────────────────────────
print("Computing cluster_position...")
df_clustered, _ = signal_clustering(df)
df["cluster_position"] = df_clustered["cluster_position"]

print("Computing prior_absorption_count...")
df_pac, _ = prior_absorption_count(df)
df["prior_absorption_count"] = df_pac["prior_absorption_count"]

print("Computing total_vol_rolling_pct...")
all_dates   = sorted(df["_date_et"].unique())
date_to_idx = {d: i for i, d in enumerate(all_dates)}
WINDOW      = 20

rolling_pct = pd.Series(np.nan, index=df.index)
for date in all_dates:
    d_idx = date_to_idx[date]
    mask  = df["_date_et"] == date
    if d_idx < 1:
        rolling_pct[mask] = 50.0
        continue
    prior_dates  = all_dates[max(0, d_idx - WINDOW): d_idx]
    prior_vols   = df[df["_date_et"].isin(prior_dates)]["total_vol"].values
    if len(prior_vols) == 0:
        rolling_pct[mask] = 50.0
        continue
    threshold_60 = float(np.percentile(prior_vols, 60))
    rolling_pct[mask] = (df.loc[mask, "total_vol"] >= threshold_60).astype(float) * 100
df["total_vol_rolling_pct"] = rolling_pct

print("Done with derived columns.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def sust(grp):
    n = len(grp)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    loose  = ((grp["move_z_score"] >= 1.2) & (grp["mae_normalized"] <= 0.35)).mean() * 100
    base   = ((grp["move_z_score"] >= 1.5) & (grp["mae_normalized"] <= 0.30)).mean() * 100
    strict = ((grp["move_z_score"] >= 2.0) & (grp["mae_normalized"] <= 0.25)).mean() * 100
    return n, loose, base, strict


def fmt_row(label, n, loose, base, strict, width=40):
    def pct(v):
        return f"{v:.1f}%" if not np.isnan(v) else "  n/a"
    return (
        f"  {label:<{width}}  {n:>5}  "
        f"{pct(loose):>8}  {pct(base):>8}  {pct(strict):>8}"
    )


def HDR(w=40):
    return f"  {'group':<{w}}  {'n':>5}  {'loose%':>8}  {'base%':>8}  {'strict%':>8}"


def SEP(w=40):
    return "  " + "-" * (w + 34)


def freq_stats(sub, label, total_days=TOTAL_DAYS):
    """Print frequency stats for a filtered subset."""
    lines = []
    lines.append(f"  Total signals:        {len(sub)}")
    if len(sub) == 0:
        lines.append(f"  (no signals)")
        return lines

    by_date = sub.groupby("_date_et").size()
    days_with_signal = len(by_date)
    avg_per_day = len(sub) / total_days

    lines.append(f"  Days with >=1 signal: {days_with_signal}  ({days_with_signal/total_days*100:.1f}% of {total_days} trading days)")
    lines.append(f"  Avg signals/day:      {avg_per_day:.3f}  (~{1/avg_per_day:.0f} trading days between signals)" if avg_per_day > 0 else "")

    # Distribution
    lines.append(f"")
    lines.append(f"  Daily signal distribution:")
    days_0 = total_days - days_with_signal
    days_1 = (by_date == 1).sum()
    days_2 = (by_date == 2).sum()
    days_3p = (by_date >= 3).sum()
    lines.append(f"    0 signals: {days_0:>4} days  ({days_0/total_days*100:.1f}%)")
    lines.append(f"    1 signal:  {days_1:>4} days  ({days_1/total_days*100:.1f}%)")
    lines.append(f"    2 signals: {days_2:>4} days  ({days_2/total_days*100:.1f}%)")
    lines.append(f"    3+ signals:{days_3p:>4} days  ({days_3p/total_days*100:.1f}%)")

    return lines


def monthly_breakdown(sub, width=18):
    """Monthly breakdown: signal count, days with signal, base sustained rate."""
    lines = []
    header = f"  {'month':<{width}}  {'n':>5}  {'days_w_sig':>10}  {'base%':>8}"
    sep    = "  " + "-" * (width + 30)
    lines.append(header)
    lines.append(sep)
    months = sorted(sub["_month"].unique())
    for m in months:
        msub = sub[sub["_month"] == m]
        n = len(msub)
        days_w = msub.groupby("_date_et").ngroups
        _, _, base, _ = sust(msub)
        def pct(v): return f"{v:.1f}%" if not np.isnan(v) else "  n/a"
        lines.append(f"  {str(m):<{width}}  {n:>5}  {days_w:>10}  {pct(base):>8}")
    return lines


# ── Build masks for each setup ────────────────────────────────────────────────

# DIRECTIONAL SETUP
# Filters: directional, outside_vah/val, solo/first, prior_absorption=0,
#          time 09:30-10:00, vol>=60th
mask_dir = (
    (df["overnight_regime"] == "directional") &
    (df["price_location"].isin(["outside_vah", "outside_val"])) &
    (df["cluster_position"].isin(["solo", "first"])) &
    (df["prior_absorption_count"] == 0) &
    (df["_hour_min"] >= 9*60+30) & (df["_hour_min"] < 10*60) &
    (df["total_vol_rolling_pct"] >= 60)
)

# SHORT REVERSAL SETUP (all regimes — regime is an optional overlay)
# Filters: outside_vah, sell_absorbed, +2s OR at_vwap_band, rejection=True,
#          proven, solo/first, vol>=60th
mask_short = (
    (df["price_location"] == "outside_vah") &
    (df["absorption_side"] == "sell_absorbed") &
    ((df["vwap_band_tag"] == "+2s") | (df["at_vwap_band"])) &
    (df["reversal_rejection_confirmed"]) &
    (df["node_proven_20t"]) &
    (df["cluster_position"].isin(["solo", "first"])) &
    (df["total_vol_rolling_pct"] >= 60)
)

# LONG REVERSAL SETUP (all regimes)
# Filters: outside_val, buy_absorbed, rejection=False, proven, solo/first, vol>=60th
mask_long = (
    (df["price_location"] == "outside_val") &
    (df["absorption_side"] == "buy_absorbed") &
    (~df["reversal_rejection_confirmed"]) &
    (df["node_proven_20t"]) &
    (df["cluster_position"].isin(["solo", "first"])) &
    (df["total_vol_rolling_pct"] >= 60)
)

dir_df   = df[mask_dir]
short_df = df[mask_short]
long_df  = df[mask_long]

# Combined union (no double-counting)
combined_mask = mask_dir | mask_short | mask_long
combined_df   = df[combined_mask]

# ── Output ────────────────────────────────────────────────────────────────────
lines = []
def p(s=""): lines.append(s)

p("=" * 80)
p("TRADE FREQUENCY REPORT")
p("Training period: 2025-03-17 to 2025-11-30  |  183 trading days")
p("=" * 80)
p()
p("Three setups:")
p("  DIRECTIONAL: directional regime, outside vah/val, solo/first, prior_abs=0,")
p("               09:30-10:00 ET, vol>=60th pct")
p("  SHORT REV:   outside_vah, sell_absorbed, +2s/at_band, rejection=True,")
p("               proven, solo/first, vol>=60th pct")
p("  LONG REV:    outside_val, buy_absorbed, rejection=False,")
p("               proven, solo/first, vol>=60th pct")
p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("1. DIRECTIONAL SETUP")
p("=" * 80)
p()
for l in freq_stats(dir_df, "DIRECTIONAL"):
    p(l)
p()
_, _, base, _ = sust(dir_df)
p(f"  Quality: base sustained = {base:.1f}%")
p()
p("  Monthly breakdown:")
for l in monthly_breakdown(dir_df):
    p(l)
p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("2. SHORT REVERSAL SETUP (all regimes)")
p("=" * 80)
p()
for l in freq_stats(short_df, "SHORT"):
    p(l)
p()
_, _, base, _ = sust(short_df)
p(f"  Quality: base sustained = {base:.1f}%")
p()
p("  Monthly breakdown:")
for l in monthly_breakdown(short_df):
    p(l)
p()

# Regime overlay for short
p("  Regime overlay:")
p(HDR(32))
p(SEP(32))
for regime in ["directional", "non_directional", "ambiguous"]:
    g = short_df[short_df["overnight_regime"] == regime]
    p(fmt_row(f"short / {regime}", *sust(g), width=32))
p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("3. LONG REVERSAL SETUP (all regimes)")
p("=" * 80)
p()
for l in freq_stats(long_df, "LONG"):
    p(l)
p()
_, _, base, _ = sust(long_df)
p(f"  Quality: base sustained = {base:.1f}%")
p()
p("  Monthly breakdown:")
for l in monthly_breakdown(long_df):
    p(l)
p()

# Regime overlay for long
p("  Regime overlay:")
p(HDR(32))
p(SEP(32))
for regime in ["directional", "non_directional", "ambiguous"]:
    g = long_df[long_df["overnight_regime"] == regime]
    p(fmt_row(f"long / {regime}", *sust(g), width=32))
p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("4. COMBINED (UNION OF ALL THREE SETUPS — no double counting)")
p("=" * 80)
p()
for l in freq_stats(combined_df, "COMBINED"):
    p(l)
p()
_, _, base, _ = sust(combined_df)
p(f"  Quality: base sustained = {base:.1f}%")
p()
p("  Overlap check:")
p(f"    dir-only signals:         {(mask_dir & ~mask_short & ~mask_long).sum()}")
p(f"    short-only signals:       {(mask_short & ~mask_dir & ~mask_long).sum()}")
p(f"    long-only signals:        {(mask_long & ~mask_dir & ~mask_short).sum()}")
p(f"    dir+short overlap:        {(mask_dir & mask_short).sum()}")
p(f"    dir+long overlap:         {(mask_dir & mask_long).sum()}")
p(f"    short+long overlap:       {(mask_short & mask_long).sum()}")
p(f"    all three overlap:        {(mask_dir & mask_short & mask_long).sum()}")
p()
p("  Monthly breakdown:")
for l in monthly_breakdown(combined_df):
    p(l)
p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("5. SUMMARY TABLE")
p("=" * 80)
p()
p(f"  {'setup':<22}  {'total_n':>8}  {'days_w_sig':>10}  {'avg/day':>8}  {'base%':>8}")
p("  " + "-" * 66)

for label, sub in [
    ("directional",   dir_df),
    ("short_reversal", short_df),
    ("long_reversal",  long_df),
    ("combined",       combined_df),
]:
    n = len(sub)
    days_w = sub.groupby("_date_et").ngroups if n else 0
    avg    = n / TOTAL_DAYS
    _, _, base, _ = sust(sub)
    def pct(v): return f"{v:.1f}%" if not np.isnan(v) else "  n/a"
    p(f"  {label:<22}  {n:>8}  {days_w:>10}  {avg:>8.3f}  {pct(base):>8}")

p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("6. SENSITIVITY: LONG REVERSAL — PROVEN NODE FILTER (node_proven_20t)")
p("=" * 80)
p()
p("  Tradeoff: removing proven node requirement adds frequency, costs quality.")
p()

# With proven (current)
mask_long_no_proven = (
    (df["price_location"] == "outside_val") &
    (df["absorption_side"] == "buy_absorbed") &
    (~df["reversal_rejection_confirmed"]) &
    (df["cluster_position"].isin(["solo", "first"])) &
    (df["total_vol_rolling_pct"] >= 60)
)
# Without proven but with all other filters
long_noproven_df = df[mask_long_no_proven]

p(HDR(48))
p(SEP(48))
p(fmt_row("long / proven=True  (current)",      *sust(long_df),          width=48))
p(fmt_row("long / proven=False (no proven req)", *sust(long_noproven_df), width=48))
p()
p("  Frequency with no proven filter:")
for l in freq_stats(long_noproven_df, "LONG_NO_PROVEN"):
    p(l)
p()

# Regime breakdown both ways
p("  Regime breakdown with vs without proven node:")
p(HDR(52))
p(SEP(52))
for regime in ["directional", "non_directional", "ambiguous"]:
    gw  = long_df[long_df["overnight_regime"] == regime]
    gnp = long_noproven_df[long_noproven_df["overnight_regime"] == regime]
    p(fmt_row(f"long/proven=T / {regime}",  *sust(gw),  width=52))
    p(fmt_row(f"long/proven=F / {regime}",  *sust(gnp), width=52))
    p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("7. TIME WINDOW ANALYSIS  (current window: 09:30-11:00 ET)")
p("=" * 80)
p()
p("  Current signals are captured in the 09:30-11:00 RTH window.")
p("  NOTE: Extending to 11:30 or 12:00 requires re-running the signal labeler")
p("        with an extended RTH end time. The analysis below shows signal")
p("        distribution within the current window by 30-min bucket.")
p()

# 30-min bucket definitions (minutes since midnight)
buckets = [
    ("09:30-10:00", 9*60+30, 10*60),
    ("10:00-10:30", 10*60,   10*60+30),
    ("10:30-11:00", 10*60+30, 11*60+1),
]

def time_bucket_table(sub, setup_label):
    lines = []
    lines.append(f"  {setup_label}:")
    total_n = len(sub)
    lines.append(f"  {'bucket':<15}  {'n':>5}  {'pct_of_total':>13}  {'base%':>8}")
    lines.append("  " + "-" * 46)
    for bname, bstart, bend in buckets:
        g = sub[(sub["_hour_min"] >= bstart) & (sub["_hour_min"] < bend)]
        n = len(g)
        _, _, base, _ = sust(g)
        def pct(v): return f"{v:.1f}%" if not np.isnan(v) else "  n/a"
        lines.append(f"  {bname:<15}  {n:>5}  {n/total_n*100 if total_n else 0:>12.1f}%  {pct(base):>8}")
    return lines

for label, sub in [
    ("DIRECTIONAL",    dir_df),
    ("SHORT_REVERSAL", short_df),
    ("LONG_REVERSAL",  long_df),
    ("COMBINED",       combined_df),
]:
    if len(sub) == 0:
        p(f"  {label}: no signals")
        p()
        continue
    for l in time_bucket_table(sub, label):
        p(l)
    p()

p("  Extrapolation to extended windows:")
p("  If signal density is uniform within 09:30-11:00 (90 min),")
p("  extending to 11:30 (+30 min, +33%) would add ~1/3 more signals.")
p("  Extending to 12:00 (+60 min, +67%) would add ~2/3 more signals.")
p()
# Compute the actual 10:30-11:00 count and extrapolate
for label, sub in [
    ("directional",   dir_df),
    ("short_reversal", short_df),
    ("long_reversal",  long_df),
]:
    last_bucket = sub[(sub["_hour_min"] >= 10*60+30) & (sub["_hour_min"] < 11*60+1)]
    n_last  = len(last_bucket)
    n_total = len(sub)
    ext_30  = round(n_last)   # same density as 10:30-11:00
    ext_60  = round(n_last * 2)
    p(f"  {label:<22}  last_30min={n_last:>3}  "
      f"est +30min={ext_30:>3}  est +60min={ext_60:>3}  "
      f"(total_now={n_total})")
p()
p("  WARNING: Signal quality in 11:00-12:00 is likely lower — the edge is")
p("  concentrated in the opening auction reaction window (09:30-10:30).")
p("  Re-labeling is required to verify actual quality of extended signals.")

# ── Save ──────────────────────────────────────────────────────────────────────
full_text = "\n".join(lines)
print(full_text)
OUT_PATH.write_text(full_text, encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

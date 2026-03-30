"""
Frequency vs quality tradeoff analysis.
Progressive filter stack from loosest to strictest.
Goal: find tier where avg/day is 0.5-1.0 with base% meaningfully above baseline.
Saves to output/diagnostics/frequency_quality_tradeoff.txt
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from nqbt.analysis.signal_diagnostics import signal_clustering, prior_absorption_count

CSV_PATH = PROJECT_ROOT / "output" / "absorption_signals_labeled.csv"
OUT_DIR  = PROJECT_ROOT / "output" / "diagnostics"
OUT_PATH = OUT_DIR / "frequency_quality_tradeoff.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ET          = "America/New_York"
TOTAL_DAYS  = 183

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
for col in ["sustained", "sustained_loose", "sustained_strict",
            "reversal_rejection_confirmed", "node_proven_20t", "at_vwap_band"]:
    df[col] = df[col].astype(bool)

sig_et      = pd.to_datetime(df["signal_time"], utc=True).dt.tz_convert(ET)
df["_date"] = sig_et.dt.date
df["_month"] = sig_et.dt.to_period("M")

# ── Derived columns ───────────────────────────────────────────────────────────
print("Computing cluster_position...")
df_c, _ = signal_clustering(df)
df["cluster_position"] = df_c["cluster_position"]

print("Computing prior_absorption_count...")
df_p, _ = prior_absorption_count(df)
df["prior_absorption_count"] = df_p["prior_absorption_count"]

print("Done.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def metrics(sub):
    n       = len(sub)
    days_w  = sub.groupby("_date").ngroups if n else 0
    avg_day = n / TOTAL_DAYS
    loose   = ((sub["move_z_score"] >= 1.2) & (sub["mae_normalized"] <= 0.35)).mean() * 100 if n else float("nan")
    base    = ((sub["move_z_score"] >= 1.5) & (sub["mae_normalized"] <= 0.30)).mean() * 100 if n else float("nan")
    return n, days_w, avg_day, loose, base


def pct(v):
    return f"{v:.1f}%" if not np.isnan(v) else "  n/a"


def flag(avg_day):
    """Mark tiers in the target frequency range."""
    if 0.5 <= avg_day <= 1.0:
        return " <-- TARGET RANGE"
    elif avg_day > 1.0:
        return " (too frequent)"
    else:
        return ""


HEADER_W = 46
def hdr():
    return (f"  {'tier / filter':<{HEADER_W}}  {'n':>5}  "
            f"{'days_w':>6}  {'avg/day':>8}  {'loose%':>8}  {'base%':>8}")

def sep():
    return "  " + "-" * (HEADER_W + 46)

def row(label, sub, width=HEADER_W):
    n, days_w, avg_day, loose, base = metrics(sub)
    freq_flag = flag(avg_day)
    return (f"  {label:<{width}}  {n:>5}  "
            f"{days_w:>6}  {avg_day:>8.3f}  {pct(loose):>8}  {pct(base):>8}"
            f"{freq_flag}")


# ── Build tier masks ──────────────────────────────────────────────────────────
# Tier 1: non_directional + solo/first + prior_abs=0
t1 = (
    (df["overnight_regime"] == "non_directional") &
    (df["cluster_position"].isin(["solo", "first"])) &
    (df["prior_absorption_count"] == 0)
)

# Tier 2: + outside_vah or outside_val
t2 = t1 & df["price_location"].isin(["outside_vah", "outside_val"])

# Tier 3: + at_vwap_band = True
t3 = t2 & df["at_vwap_band"]

# Tier 4: + rejection logic (asymmetric)
#   outside_vah: rejection=True (seller rejection confirms short)
#   outside_val: rejection=False (no rejection = demand held, confirms long)
t4 = t3 & (
    ((df["price_location"] == "outside_vah") & df["reversal_rejection_confirmed"]) |
    ((df["price_location"] == "outside_val") & ~df["reversal_rejection_confirmed"])
)

# Tier 5: + node_proven_20t = True
t5 = t4 & df["node_proven_20t"]

# ── Also build directional equivalent for comparison ─────────────────────────
td1 = (
    (df["overnight_regime"] == "directional") &
    (df["cluster_position"].isin(["solo", "first"])) &
    (df["prior_absorption_count"] == 0)
)
td2 = td1 & df["price_location"].isin(["outside_vah", "outside_val"])
td3 = td2 & df["at_vwap_band"]
td4 = td3 & (
    ((df["price_location"] == "outside_vah") & df["reversal_rejection_confirmed"]) |
    ((df["price_location"] == "outside_val") & ~df["reversal_rejection_confirmed"])
)
td5 = td4 & df["node_proven_20t"]

# ── ALL regimes combined (non_dir + directional) ──────────────────────────────
ta1 = t1 | td1
ta2 = t2 | td2
ta3 = t3 | td3
ta4 = t4 | td4
ta5 = t5 | td5

# ── Output ────────────────────────────────────────────────────────────────────
lines = []
def p(s=""): lines.append(s)

p("=" * 80)
p("FREQUENCY vs QUALITY TRADEOFF — PROGRESSIVE FILTER STACK")
p("Training: 2025-03-17 to 2025-11-30  |  183 trading days")
p("=" * 80)
p()
p("Target: avg/day 0.5-1.0 with base% meaningfully above 6.0% baseline.")
p("Goal: enough signals for 5 profitable days per prop cycle, no overtrading.")
p()
p("Baseline: all 2711 signals, no filters")
n, days_w, avg_day, loose, base = metrics(df)
p(f"  n={n}  days_active={days_w}  avg/day={avg_day:.3f}  loose={pct(loose)}  base={pct(base)}")
p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("SECTION A — NON_DIRECTIONAL REGIME ONLY")
p("=" * 80)
p()
p(hdr())
p(sep())

tiers_nd = [
    ("Baseline (all signals)",                          df),
    ("T1: non_dir + solo/first + prior_abs=0",         df[t1]),
    ("T2: + outside_vah/val",                          df[t2]),
    ("T3: + at_vwap_band=True",                        df[t3]),
    ("T4: + asymmetric rejection filter",              df[t4]),
    ("T5: + node_proven_20t=True",                     df[t5]),
]
for label, sub in tiers_nd:
    p(row(label, sub))
p()

# T2 breakdown by location
p("  T2 breakdown by price_location:")
p(hdr())
p(sep())
for loc in ["outside_vah", "outside_val"]:
    g = df[t2 & (df["price_location"] == loc)]
    p(row(f"  T2 / {loc}", g))
p()

# T3 breakdown by location
p("  T3 breakdown by price_location:")
p(hdr())
p(sep())
for loc in ["outside_vah", "outside_val"]:
    g = df[t3 & (df["price_location"] == loc)]
    p(row(f"  T3 / {loc}", g))
p()

# T4 breakdown by location
p("  T4 breakdown by price_location:")
p(hdr())
p(sep())
for loc in ["outside_vah", "outside_val"]:
    g = df[t4 & (df["price_location"] == loc)]
    p(row(f"  T4 / {loc}", g))
p()

# T5 breakdown by location
p("  T5 breakdown by price_location:")
p(hdr())
p(sep())
for loc in ["outside_vah", "outside_val"]:
    g = df[t5 & (df["price_location"] == loc)]
    p(row(f"  T5 / {loc}", g))
p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("SECTION B — DIRECTIONAL REGIME ONLY (same stack for comparison)")
p("=" * 80)
p()
p(hdr())
p(sep())

tiers_dir = [
    ("T1: directional + solo/first + prior_abs=0",    df[td1]),
    ("T2: + outside_vah/val",                         df[td2]),
    ("T3: + at_vwap_band=True",                       df[td3]),
    ("T4: + asymmetric rejection filter",             df[td4]),
    ("T5: + node_proven_20t=True",                    df[td5]),
]
for label, sub in tiers_dir:
    p(row(label, sub))
p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("SECTION C — ALL REGIMES COMBINED (non_dir + directional, no ambiguous)")
p("=" * 80)
p()
p(hdr())
p(sep())

tiers_all = [
    ("Baseline (all signals)",                         df),
    ("T1: (nd+dir) + solo/first + prior_abs=0",       df[ta1]),
    ("T2: + outside_vah/val",                         df[ta2]),
    ("T3: + at_vwap_band=True",                       df[ta3]),
    ("T4: + asymmetric rejection filter",             df[ta4]),
    ("T5: + node_proven_20t=True",                    df[ta5]),
]
for label, sub in tiers_all:
    p(row(label, sub))
p()

# T3-T5 combined breakdowns
for tier_mask, tier_name in [(ta3, "T3"), (ta4, "T4"), (ta5, "T5")]:
    p(f"  {tier_name} breakdown by location x regime:")
    p(hdr())
    p(sep())
    for loc in ["outside_vah", "outside_val"]:
        for regime in ["non_directional", "directional"]:
            g = df[tier_mask & (df["price_location"] == loc) & (df["overnight_regime"] == regime)]
            p(row(f"  {tier_name} / {loc} / {regime}", g))
    p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("SECTION D — LOOSE OUTCOME (z>=1.2, mae<=0.35) vs BASE (z>=1.5, mae<=0.30)")
p("Same tiers, both metrics shown side by side")
p("=" * 80)
p()

w2 = 46
p(f"  {'tier':<{w2}}  {'n':>5}  {'avg/day':>8}  {'loose%':>8}  {'base%':>8}  {'lift_loose':>10}  {'lift_base':>10}")
p("  " + "-" * (w2 + 58))

baseline_loose = ((df["move_z_score"] >= 1.2) & (df["mae_normalized"] <= 0.35)).mean() * 100
baseline_base  = ((df["move_z_score"] >= 1.5) & (df["mae_normalized"] <= 0.30)).mean() * 100

for label, sub in [
    ("Baseline (all signals)",                         df),
    ("T1 combined (nd+dir)",                           df[ta1]),
    ("T2 combined",                                    df[ta2]),
    ("T3 combined",                                    df[ta3]),
    ("T4 combined",                                    df[ta4]),
    ("T5 combined",                                    df[ta5]),
]:
    n, _, avg_day, loose, base = metrics(sub)
    ll = loose - baseline_loose
    lb = base  - baseline_base
    def pp(v): return f"{v:+.1f}pp" if not np.isnan(v) else "   n/a"
    freq_flag = flag(avg_day)
    p(f"  {label:<{w2}}  {n:>5}  {avg_day:>8.3f}  {pct(loose):>8}  {pct(base):>8}  {pp(ll):>10}  {pp(lb):>10}{freq_flag}")
p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("SECTION E — MONTHLY SIGNAL COUNTS AT EACH TIER (combined, T1-T5)")
p("=" * 80)
p()

months = sorted(df["_month"].unique())
tier_masks_combined = [ta1, ta2, ta3, ta4, ta5]
tier_labels = ["T1", "T2", "T3", "T4", "T5"]

# Header
col_w = 8
p(f"  {'month':<10}" + "".join(f"  {t:>{col_w}}" for t in tier_labels))
p("  " + "-" * (10 + len(tier_labels) * (col_w + 2)))

for m in months:
    row_str = f"  {str(m):<10}"
    for mask in tier_masks_combined:
        g = df[mask & (df["_month"] == m)]
        row_str += f"  {len(g):>{col_w}}"
    p(row_str)

p()
p(f"  {'TOTAL':<10}" + "".join(f"  {df[mask].shape[0]:>{col_w}}" for mask in tier_masks_combined))
p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("SECTION F — SWEET SPOT ANALYSIS")
p("=" * 80)
p()
p("  Target range: avg signals/day between 0.5 and 1.0")
p("  Baseline base sustained: 6.0%")
p()
p(f"  {'tier':<38}  {'avg/day':>8}  {'base%':>8}  {'in_target':>10}  {'lift_vs_base':>12}")
p("  " + "-" * 82)

all_tiers = [
    ("Baseline",           df),
    ("T1 non_dir",         df[t1]),
    ("T1 directional",     df[td1]),
    ("T1 combined",        df[ta1]),
    ("T2 non_dir",         df[t2]),
    ("T2 directional",     df[td2]),
    ("T2 combined",        df[ta2]),
    ("T3 non_dir",         df[t3]),
    ("T3 directional",     df[td3]),
    ("T3 combined",        df[ta3]),
    ("T4 non_dir",         df[t4]),
    ("T4 directional",     df[td4]),
    ("T4 combined",        df[ta4]),
    ("T5 non_dir",         df[t5]),
    ("T5 directional",     df[td5]),
    ("T5 combined",        df[ta5]),
]

for label, sub in all_tiers:
    n, _, avg_day, loose, base = metrics(sub)
    in_target = "YES" if 0.5 <= avg_day <= 1.0 else ("too_frequent" if avg_day > 1.0 else "too_rare")
    lb = base - baseline_base
    def pp(v): return f"{v:+.1f}pp" if not np.isnan(v) else "  n/a"
    p(f"  {label:<38}  {avg_day:>8.3f}  {pct(base):>8}  {in_target:>10}  {pp(lb):>12}")
p()

# ── Save ──────────────────────────────────────────────────────────────────────
full_text = "\n".join(lines)
print(full_text)
OUT_PATH.write_text(full_text, encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

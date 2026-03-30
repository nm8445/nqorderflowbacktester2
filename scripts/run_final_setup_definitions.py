"""
Final setup suite definition and frequency report.
Implements Setup A, B, C — computes training metrics, saves JSON rules and report.
Saves:
  output/diagnostics/setup_A_rules.json
  output/diagnostics/setup_B_rules.json
  output/diagnostics/setup_C_rules.json
  output/diagnostics/final_setup_frequency.txt
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from nqbt.analysis.signal_diagnostics import signal_clustering, prior_absorption_count

CSV_PATH   = PROJECT_ROOT / "output" / "absorption_signals_labeled.csv"
OUT_DIR    = PROJECT_ROOT / "output" / "diagnostics"
OUT_TXT    = OUT_DIR / "final_setup_frequency.txt"
OUT_A_JSON = OUT_DIR / "setup_A_rules.json"
OUT_B_JSON = OUT_DIR / "setup_B_rules.json"
OUT_C_JSON = OUT_DIR / "setup_C_rules.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ET         = "America/New_York"
TOTAL_DAYS = 183
CYCLE_DAYS = 21   # trading days per payout cycle

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
for col in ["sustained", "sustained_loose", "sustained_strict",
            "reversal_rejection_confirmed", "node_proven_20t", "at_vwap_band"]:
    df[col] = df[col].astype(bool)

sig_et       = pd.to_datetime(df["signal_time"], utc=True).dt.tz_convert(ET)
df["_date"]  = sig_et.dt.date
df["_month"] = sig_et.dt.to_period("M")

# ── Derived columns ───────────────────────────────────────────────────────────
print("Computing cluster_position...")
df_c, _ = signal_clustering(df)
df["cluster_position"] = df_c["cluster_position"]

print("Computing prior_absorption_count...")
df_p, _ = prior_absorption_count(df)
df["prior_absorption_count"] = df_p["prior_absorption_count"]

print("Done.\n")

# ── Setup masks ───────────────────────────────────────────────────────────────

# SETUP A — High Conviction Short Reversal
mask_A = (
    (df["price_location"] == "outside_vah") &
    (df["absorption_side"] == "sell_absorbed") &
    ((df["vwap_band_tag"] == "+2s") | df["at_vwap_band"]) &
    (df["reversal_rejection_confirmed"]) &
    (df["node_proven_20t"]) &
    (df["cluster_position"].isin(["solo", "first"])) &
    (df["overnight_regime"].isin(["directional", "non_directional"]))
)

# SETUP C — High Conviction Long Reversal (directional only)
mask_C = (
    (df["price_location"] == "outside_val") &
    (df["absorption_side"] == "buy_absorbed") &
    (df["overnight_regime"] == "directional") &
    (df["node_proven_20t"]) &
    (~df["reversal_rejection_confirmed"]) &
    (df["cluster_position"].isin(["solo", "first"]))
)

# SETUP B — Standard Short Reversal (bread and butter)
# Excludes signals that qualify for Setup A
mask_B = (
    (df["price_location"] == "outside_vah") &
    (df["absorption_side"] == "sell_absorbed") &
    (df["cluster_position"].isin(["solo", "first"])) &
    (df["prior_absorption_count"] == 0) &
    (df["overnight_regime"] == "non_directional") &
    (~mask_A)   # exclude Setup A signals
)

A = df[mask_A]
B = df[mask_B]
C = df[mask_C]

# ── Helpers ───────────────────────────────────────────────────────────────────

def sust_metrics(sub):
    n = len(sub)
    if n == 0:
        return dict(n=0, loose=float("nan"), base=float("nan"), strict=float("nan"),
                    mean_z_sustained=float("nan"))
    loose  = ((sub["move_z_score"] >= 1.2) & (sub["mae_normalized"] <= 0.35)).mean() * 100
    base   = ((sub["move_z_score"] >= 1.5) & (sub["mae_normalized"] <= 0.30)).mean() * 100
    strict = ((sub["move_z_score"] >= 2.0) & (sub["mae_normalized"] <= 0.25)).mean() * 100
    sust_sigs = sub[(sub["move_z_score"] >= 1.5) & (sub["mae_normalized"] <= 0.30)]
    mean_z = sust_sigs["move_z_score"].mean() if len(sust_sigs) else float("nan")
    return dict(n=n, loose=loose, base=base, strict=strict, mean_z_sustained=mean_z)


def freq_stats(sub):
    n      = len(sub)
    days_w = sub.groupby("_date").ngroups if n else 0
    avg    = n / TOTAL_DAYS
    by_d   = sub.groupby("_date").size() if n else pd.Series(dtype=int)
    d0  = TOTAL_DAYS - days_w
    d1  = (by_d == 1).sum() if n else 0
    d2  = (by_d == 2).sum() if n else 0
    d3p = (by_d >= 3).sum() if n else 0
    return dict(total=n, days_active=days_w, avg_per_day=avg,
                days_0=int(d0), days_1=int(d1), days_2=int(d2), days_3p=int(d3p))


def pct(v):
    return f"{v:.1f}%" if not np.isnan(v) else "  n/a"


def fmt_z(v):
    return f"{v:.3f}" if not np.isnan(v) else "  n/a"


lines = []
def p(s=""): lines.append(s)

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("FINAL SETUP SUITE — TRAINING PERIOD FREQUENCY AND QUALITY REPORT")
p("Training: 2025-03-17 to 2025-11-30  |  183 trading days  |  ~8.7 payout cycles")
p("=" * 80)
p()
p("Three setups defined:")
p("  SETUP A  High Conviction Short Reversal (size up)")
p("           outside_vah + sell_absorbed + +2s/at_band + rejection=True")
p("           + proven node + solo/first | all regimes")
p()
p("  SETUP B  Standard Short Reversal (bread and butter)")
p("           outside_vah + sell_absorbed + solo/first + prior_abs=0")
p("           + non_directional | excludes Setup A signals")
p()
p("  SETUP C  High Conviction Long Reversal (directional days, size up)")
p("           outside_val + buy_absorbed + directional + proven")
p("           + rejection=False + solo/first")
p()

# ── Baseline ──────────────────────────────────────────────────────────────────
p(f"Baseline: 2711 total signals, 6.0% base sustained, 9.8% loose")
p()

for setup_label, sub, expected_note in [
    ("SETUP A — HIGH CONVICTION SHORT REVERSAL", A,
     "Expected: ~0.05-0.08/day, ~25-35% base sustained"),
    ("SETUP B — STANDARD SHORT REVERSAL",        B,
     "Expected: ~0.3-0.5/day, ~10-12% base sustained"),
    ("SETUP C — HIGH CONVICTION LONG REVERSAL",  C,
     "Expected: ~0.05/day, ~44% base sustained"),
]:
    sm = sust_metrics(sub)
    fs = freq_stats(sub)

    p("=" * 80)
    p(setup_label)
    p("=" * 80)
    p(f"  {expected_note}")
    p()

    # Frequency
    p(f"  FREQUENCY")
    p(f"  {'Total signals:':<30} {fs['total']}")
    p(f"  {'Days active:':<30} {fs['days_active']}  ({fs['days_active']/TOTAL_DAYS*100:.1f}% of {TOTAL_DAYS} trading days)")
    p(f"  {'Avg signals/day:':<30} {fs['avg_per_day']:.3f}  (1 signal every ~{1/fs['avg_per_day']:.0f} trading days)" if fs['avg_per_day'] > 0 else "  Avg signals/day:              0")
    est_per_cycle = fs['avg_per_day'] * CYCLE_DAYS
    p(f"  {'Est signals/payout cycle:':<30} {est_per_cycle:.1f}  (at {CYCLE_DAYS} trading days/cycle)")
    p()

    # Distribution
    p(f"  DAILY DISTRIBUTION")
    p(f"    0 signals: {fs['days_0']:>4} days  ({fs['days_0']/TOTAL_DAYS*100:.1f}%)")
    p(f"    1 signal:  {fs['days_1']:>4} days  ({fs['days_1']/TOTAL_DAYS*100:.1f}%)")
    p(f"    2 signals: {fs['days_2']:>4} days  ({fs['days_2']/TOTAL_DAYS*100:.1f}%)")
    p(f"    3+ signals:{fs['days_3p']:>4} days  ({fs['days_3p']/TOTAL_DAYS*100:.1f}%)")
    p()

    # Quality
    p(f"  QUALITY")
    p(f"  {'Loose  (z>=1.2, mae<=0.35):':<30} {pct(sm['loose'])}")
    p(f"  {'Base   (z>=1.5, mae<=0.30):':<30} {pct(sm['base'])}")
    p(f"  {'Strict (z>=2.0, mae<=0.25):':<30} {pct(sm['strict'])}")
    p(f"  {'Mean z-score on sustained:':<30} {fmt_z(sm['mean_z_sustained'])}")
    p()

    # Regime breakdown
    p(f"  REGIME BREAKDOWN")
    w = 32
    p(f"  {'regime':<{w}}  {'n':>5}  {'loose%':>8}  {'base%':>8}  {'strict%':>8}")
    p("  " + "-" * (w + 34))
    for regime in ["directional", "non_directional", "ambiguous"]:
        g = sub[sub["overnight_regime"] == regime]
        sm2 = sust_metrics(g)
        p(f"  {regime:<{w}}  {sm2['n']:>5}  {pct(sm2['loose']):>8}  {pct(sm2['base']):>8}  {pct(sm2['strict']):>8}")
    p()

    # Location breakdown (for setups with both locations)
    if setup_label.startswith("SETUP A") or setup_label.startswith("SETUP B"):
        p(f"  VWAP BAND BREAKDOWN (Setup A only — n/a for B)")
        if setup_label.startswith("SETUP A"):
            w = 20
            p(f"  {'vwap_band_tag':<{w}}  {'n':>5}  {'loose%':>8}  {'base%':>8}")
            p("  " + "-" * (w + 28))
            for band in ["+2s", "+1s", "vwap", "-1s", "-2s"]:
                g = sub[sub["vwap_band_tag"] == band]
                sm2 = sust_metrics(g)
                p(f"  {band:<{w}}  {sm2['n']:>5}  {pct(sm2['loose']):>8}  {pct(sm2['base']):>8}")
            g_atband = sub[sub["at_vwap_band"] & (sub["vwap_band_tag"] != "+2s")]
            sm2 = sust_metrics(g_atband)
            p(f"  {'at_vwap_band (not +2s)':<{w}}  {sm2['n']:>5}  {pct(sm2['loose']):>8}  {pct(sm2['base']):>8}")
        p()

    # Monthly breakdown
    p(f"  MONTHLY SIGNAL COUNT")
    months = sorted(sub["_month"].unique()) if len(sub) else []
    p(f"  {'month':<12}  {'n':>5}  {'days_active':>12}  {'base%':>8}")
    p("  " + "-" * 44)
    for m in months:
        ms = sub[sub["_month"] == m]
        sm2 = sust_metrics(ms)
        dw = ms.groupby("_date").ngroups
        p(f"  {str(m):<12}  {sm2['n']:>5}  {dw:>12}  {pct(sm2['base']):>8}")
    p()

# ════════════════════════════════════════════════════════════════════════════
p("=" * 80)
p("COMBINED SETUP SUITE — CROSS-SETUP FREQUENCY ANALYSIS")
p("=" * 80)
p()

all_mask    = mask_A | mask_B | mask_C
combined_df = df[all_mask]
sm_comb     = sust_metrics(combined_df)
fs_comb     = freq_stats(combined_df)

p(f"  Total unique signals (A+B+C, no double-count): {fs_comb['total']}")
p(f"  Days with any setup firing:                    {fs_comb['days_active']}  ({fs_comb['days_active']/TOTAL_DAYS*100:.1f}%)")
p(f"  Avg signals/day (all setups):                  {fs_comb['avg_per_day']:.3f}")
p(f"  Est signals/payout cycle:                      {fs_comb['avg_per_day']*CYCLE_DAYS:.1f}")
p(f"  Combined base sustained:                       {pct(sm_comb['base'])}")
p()

# Overlap analysis
dates_A = set(A["_date"].unique())
dates_B = set(B["_date"].unique())
dates_C = set(C["_date"].unique())
dates_AB = dates_A & dates_B
dates_AC = dates_A & dates_C
dates_BC = dates_B & dates_C
dates_any = dates_A | dates_B | dates_C

p(f"  Setup overlap (days where multiple setups fire):")
p(f"    A and B both fire (sizing conflict):  {len(dates_AB)} days")
p(f"    A and C both fire:                    {len(dates_AC)} days")
p(f"    B and C both fire:                    {len(dates_BC)} days")
p(f"    Any setup fires:                      {len(dates_any)} days  ({len(dates_any)/TOTAL_DAYS*100:.1f}% of trading days)")
p()

# Per-setup daily signal count when all fire same day
if len(dates_AB) > 0:
    p(f"  Days A+B both fire — quality comparison:")
    w = 18
    p(f"  {'setup':<{w}}  {'n':>5}  {'base%':>8}")
    p("  " + "-" * (w + 18))
    A_ab = A[A["_date"].isin(dates_AB)]
    B_ab = B[B["_date"].isin(dates_AB)]
    sm_Aab = sust_metrics(A_ab)
    sm_Bab = sust_metrics(B_ab)
    p(f"  {'A (on A+B days)':<{w}}  {sm_Aab['n']:>5}  {pct(sm_Aab['base']):>8}")
    p(f"  {'B (on A+B days)':<{w}}  {sm_Bab['n']:>5}  {pct(sm_Bab['base']):>8}")
    p()

# Summary table
p(f"  SUMMARY TABLE")
p(f"  {'setup':<32}  {'n':>5}  {'avg/day':>8}  {'per_cycle':>10}  {'loose%':>8}  {'base%':>8}  {'strict%':>8}")
p("  " + "-" * 88)
for label, sub in [("A  High Conv Short", A), ("B  Standard Short", B),
                    ("C  High Conv Long",  C), ("COMBINED",          combined_df)]:
    sm = sust_metrics(sub)
    fs = freq_stats(sub)
    cycle = fs['avg_per_day'] * CYCLE_DAYS
    p(f"  {label:<32}  {sm['n']:>5}  {fs['avg_per_day']:>8.3f}  {cycle:>10.1f}  "
      f"{pct(sm['loose']):>8}  {pct(sm['base']):>8}  {pct(sm['strict']):>8}")
p()

# Payout cycle projection
p(f"  PAYOUT CYCLE PROJECTION (21 trading days/cycle)")
p(f"  Assuming base sustained hit rate holds:")
for label, sub in [("Setup A", A), ("Setup B", B), ("Setup C", C), ("Combined", combined_df)]:
    sm = sust_metrics(sub)
    fs = freq_stats(sub)
    cycle_n = fs['avg_per_day'] * CYCLE_DAYS
    if not np.isnan(sm['base']) and sm['base'] > 0:
        winning = cycle_n * sm['base'] / 100
        p(f"    {label:<12}  {cycle_n:.1f} signals/cycle  x  {pct(sm['base'])} base  =  ~{winning:.1f} sustained moves/cycle")
    else:
        p(f"    {label:<12}  {cycle_n:.1f} signals/cycle  x  {pct(sm['base'])} base")
p()

# Monthly combined
p(f"  COMBINED MONTHLY BREAKDOWN")
months_all = sorted(combined_df["_month"].unique())
p(f"  {'month':<12}  {'A':>5}  {'B':>5}  {'C':>5}  {'total':>6}  {'base%':>8}")
p("  " + "-" * 46)
for m in months_all:
    ma = A[A["_month"] == m]
    mb = B[B["_month"] == m]
    mc = C[C["_month"] == m]
    mt = combined_df[combined_df["_month"] == m]
    sm2 = sust_metrics(mt)
    p(f"  {str(m):<12}  {len(ma):>5}  {len(mb):>5}  {len(mc):>5}  {len(mt):>6}  {pct(sm2['base']):>8}")
p()

# ── Save text ─────────────────────────────────────────────────────────────────
full_text = "\n".join(lines)
print(full_text)
OUT_TXT.write_text(full_text, encoding="utf-8")
print(f"\nSaved to {OUT_TXT}")

# ── Save JSON rule files ──────────────────────────────────────────────────────
sm_A = sust_metrics(A)
sm_B = sust_metrics(B)
sm_C = sust_metrics(C)
fs_A = freq_stats(A)
fs_B = freq_stats(B)
fs_C = freq_stats(C)

def regime_stats(sub):
    out = {}
    for regime in ["directional", "non_directional"]:
        g = sub[sub["overnight_regime"] == regime]
        sm = sust_metrics(g)
        out[regime] = {
            "n": sm["n"],
            "loose_pct": round(sm["loose"], 1) if not np.isnan(sm["loose"]) else None,
            "base_pct":  round(sm["base"],  1) if not np.isnan(sm["base"])  else None,
            "strict_pct":round(sm["strict"],1) if not np.isnan(sm["strict"]) else None,
        }
    return out

def training_block(sm, fs):
    return {
        "total_signals":      fs["total"],
        "days_active":        fs["days_active"],
        "avg_per_day":        round(fs["avg_per_day"], 3),
        "est_per_cycle_21d":  round(fs["avg_per_day"] * CYCLE_DAYS, 1),
        "loose_pct":          round(sm["loose"],  1) if not np.isnan(sm["loose"])  else None,
        "base_pct":           round(sm["base"],   1) if not np.isnan(sm["base"])   else None,
        "strict_pct":         round(sm["strict"], 1) if not np.isnan(sm["strict"]) else None,
        "mean_z_sustained":   round(sm["mean_z_sustained"], 3) if not np.isnan(sm["mean_z_sustained"]) else None,
    }

rules_A = {
    "name": "setup_A_high_conviction_short_v1",
    "description": (
        "High conviction short reversal. Upper extreme rejection confirmed by "
        "high-vol opposing sellers at +2s VWAP band or within VWAP band. "
        "Proven delta node within 20 ticks. First or solo cluster signal. "
        "All regimes. Size up."
    ),
    "version": "1.0",
    "created":  "2026-03-19",
    "sizing":   "full_size",
    "filters": {
        "price_location":              {"op": "eq",    "value": "outside_vah"},
        "absorption_side":             {"op": "eq",    "value": "sell_absorbed"},
        "vwap_band":                   {"op": "logic", "value": "vwap_band_tag=+2s OR at_vwap_band=True"},
        "reversal_rejection_confirmed":{"op": "eq",    "value": True},
        "node_proven_20t":             {"op": "eq",    "value": True},
        "cluster_position":            {"op": "isin",  "value": ["solo", "first"]},
        "overnight_regime":            {"op": "isin",  "value": ["directional", "non_directional"]},
    },
    "training_stats": training_block(sm_A, fs_A),
    "training_stats_by_regime": regime_stats(A),
}

rules_B = {
    "name": "setup_B_standard_short_v1",
    "description": (
        "Standard short reversal. Sell absorption at upper extreme on non-directional "
        "overnight sessions. First or solo cluster, no recent prior absorption. "
        "Excludes signals that qualify for Setup A. Baseline sizing."
    ),
    "version": "1.0",
    "created":  "2026-03-19",
    "sizing":   "half_size",
    "filters": {
        "price_location":        {"op": "eq",   "value": "outside_vah"},
        "absorption_side":       {"op": "eq",   "value": "sell_absorbed"},
        "cluster_position":      {"op": "isin", "value": ["solo", "first"]},
        "prior_absorption_count":{"op": "eq",   "value": 0},
        "overnight_regime":      {"op": "eq",   "value": "non_directional"},
        "exclude_setup_A":       {"op": "eq",   "value": True,
                                  "note": "signals meeting Setup A criteria are excluded"},
    },
    "training_stats": training_block(sm_B, fs_B),
    "training_stats_by_regime": {"non_directional": {
        "n": fs_B["total"],
        "loose_pct": round(sm_B["loose"], 1) if not np.isnan(sm_B["loose"]) else None,
        "base_pct":  round(sm_B["base"],  1) if not np.isnan(sm_B["base"])  else None,
        "strict_pct":round(sm_B["strict"],1) if not np.isnan(sm_B["strict"]) else None,
    }},
}

rules_C = {
    "name": "setup_C_high_conviction_long_v1",
    "description": (
        "High conviction long reversal. Buy absorption at lower extreme on directional "
        "overnight sessions. Demand held quietly — no high-vol seller rejection. "
        "Proven delta node within 20 ticks. First or solo cluster signal. Size up."
    ),
    "version": "1.0",
    "created":  "2026-03-19",
    "sizing":   "full_size",
    "filters": {
        "price_location":              {"op": "eq",   "value": "outside_val"},
        "absorption_side":             {"op": "eq",   "value": "buy_absorbed"},
        "overnight_regime":            {"op": "eq",   "value": "directional"},
        "node_proven_20t":             {"op": "eq",   "value": True},
        "reversal_rejection_confirmed":{"op": "eq",   "value": False,
                                        "note": "rejection=False means no high-vol opposing sellers — demand held"},
        "cluster_position":            {"op": "isin", "value": ["solo", "first"]},
    },
    "training_stats": training_block(sm_C, fs_C),
    "training_stats_by_regime": {"directional": {
        "n": fs_C["total"],
        "loose_pct": round(sm_C["loose"], 1) if not np.isnan(sm_C["loose"]) else None,
        "base_pct":  round(sm_C["base"],  1) if not np.isnan(sm_C["base"])  else None,
        "strict_pct":round(sm_C["strict"],1) if not np.isnan(sm_C["strict"]) else None,
    }},
}

for path, rules in [(OUT_A_JSON, rules_A), (OUT_B_JSON, rules_B), (OUT_C_JSON, rules_C)]:
    path.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    print(f"Saved {path}")

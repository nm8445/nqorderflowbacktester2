"""
Cross-tab analysis for reversal_rejection_confirmed vs sustained rates.
Requires output/absorption_signals_labeled.csv to include
reversal_rejection_confirmed and reversal_rejection_bar columns.

Saves to output/diagnostics/reversal_rejection_report.txt
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

CSV_PATH = PROJECT_ROOT / "output" / "absorption_signals_labeled.csv"
OUT_DIR  = PROJECT_ROOT / "output" / "diagnostics"
OUT_PATH = OUT_DIR / "reversal_rejection_report.txt"

df = pd.read_csv(CSV_PATH)

REQUIRED = ["reversal_rejection_confirmed", "reversal_rejection_bar"]
missing  = [c for c in REQUIRED if c not in df.columns]
if missing:
    print(f"ERROR: missing columns {missing}")
    print("Re-run the signal labeler first.")
    sys.exit(1)

for col in ["sustained", "sustained_loose", "sustained_strict",
            "reversal_rejection_confirmed", "post_absorption_aggression_3bar"]:
    df[col] = df[col].astype(bool)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sust(grp):
    n = len(grp)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    loose  = ((grp["move_z_score"] >= 1.2) & (grp["mae_normalized"] <= 0.35)).mean() * 100
    base   = ((grp["move_z_score"] >= 1.5) & (grp["mae_normalized"] <= 0.30)).mean() * 100
    strict = ((grp["move_z_score"] >= 2.0) & (grp["mae_normalized"] <= 0.25)).mean() * 100
    return n, loose, base, strict


def lift(base_val, comp_val):
    """Absolute pp lift of comp over base."""
    if np.isnan(base_val) or np.isnan(comp_val):
        return float("nan")
    return comp_val - base_val


def fmt_row(label, n, loose, base, strict, width=46):
    def pct(v):
        return f"{v:.1f}%" if not np.isnan(v) else "  n/a"
    return (
        f"  {label:<{width}}  {n:>5}  "
        f"{pct(loose):>8}  {pct(base):>8}  {pct(strict):>8}"
    )


def HDR(w=46):
    return f"  {'group':<{w}}  {'n':>5}  {'loose%':>8}  {'base%':>8}  {'strict%':>8}"


def SEP(w=46):
    return "  " + "-" * (w + 34)


def rej_split(sub, label_prefix, width=46):
    """Print rejection=True vs False rows for a subset."""
    lines = []
    r_true  = sub[sub["reversal_rejection_confirmed"] == True]
    r_false = sub[sub["reversal_rejection_confirmed"] == False]
    n_t, lo_t, ba_t, st_t = sust(r_true)
    n_f, lo_f, ba_f, st_f = sust(r_false)
    # Also show "all" as baseline
    n_a, lo_a, ba_a, st_a = sust(sub)
    lines.append(fmt_row(f"{label_prefix}  [all]",        n_a, lo_a, ba_a, st_a, width))
    lines.append(fmt_row(f"{label_prefix}  rejection=True",  n_t, lo_t, ba_t, st_t, width))
    lines.append(fmt_row(f"{label_prefix}  rejection=False", n_f, lo_f, ba_f, st_f, width))
    # Lift row
    lo_lift = lift(lo_f, lo_t)
    ba_lift = lift(ba_f, ba_t)
    st_lift = lift(st_f, st_t)
    def pp(v): return f"{v:+.1f}pp" if not np.isnan(v) else "    n/a"
    lines.append(
        f"  {'  => lift (True vs False)':<{width}}  {'':>5}  "
        f"{pp(lo_lift):>8}  {pp(ba_lift):>8}  {pp(st_lift):>8}"
    )
    return lines


lines = []
def p(s=""):
    lines.append(s)


# ── Header ────────────────────────────────────────────────────────────────────
p("=" * 80)
p("REVERSAL REJECTION CONFIRMED — CROSS-TAB ANALYSIS")
p("=" * 80)
p()
p("Definition:")
p("  reversal_rejection_confirmed = True when, within 3 bars after signal:")
p("    sell_absorbed (bullish): sell_vol > buy_vol AND total_vol >= session 70th pct")
p("    buy_absorbed  (bearish): buy_vol > sell_vol AND total_vol >= session 70th pct")
p("  i.e. the OPPOSING side steps in aggressively at an extreme, confirming rejection.")
p()
total = len(df)
rej_n = df["reversal_rejection_confirmed"].sum()
p(f"Total signals: {total}  |  rejection=True: {rej_n} ({rej_n/total*100:.1f}%)  "
  f"|  rejection=False: {total-rej_n} ({(total-rej_n)/total*100:.1f}%)")
p()

# Rejection bar distribution
p("reversal_rejection_bar distribution (which post-bar triggered):")
for bar_n in [0, 1, 2, 3]:
    cnt = (df["reversal_rejection_bar"] == bar_n).sum()
    label = "none" if bar_n == 0 else f"bar +{bar_n}"
    p(f"  {label:<10} {cnt:>5}  ({cnt/total*100:.1f}%)")
p()

# Per-side breakdown
p("By absorption_side:")
w0 = 50
p(HDR(w0))
p(SEP(w0))
for side in ["sell_absorbed", "buy_absorbed"]:
    sub = df[df["absorption_side"] == side]
    r   = sub["reversal_rejection_confirmed"].sum()
    p(fmt_row(f"{side}  rejection=True",  r, *sust(sub[sub["reversal_rejection_confirmed"]])[1:], width=w0))
    p(fmt_row(f"{side}  rejection=False", len(sub)-r, *sust(sub[~sub["reversal_rejection_confirmed"]])[1:], width=w0))
    p()

# ── 1. +2s band × reversal_rejection ─────────────────────────────────────────
p("=" * 80)
p("1. VWAP_BAND_TAG = '+2s'  x  REVERSAL_REJECTION_CONFIRMED")
p("=" * 80)
p()
sub2s = df[df["vwap_band_tag"] == "+2s"]
p(f"  Signals at +2s: {len(sub2s)}")
p()
p(HDR())
p(SEP())
for row in rej_split(sub2s, "+2s band"):
    p(row)
p()

# Compare with post_absorption_aggression_3bar at +2s
p("  --- Comparison: post_agg_3bar at +2s (for reference) ---")
p(HDR())
p(SEP())
for agg_val, label in [(True, "+2s / post_agg=True"), (False, "+2s / post_agg=False")]:
    g = sub2s[sub2s["post_absorption_aggression_3bar"] == agg_val]
    p(fmt_row(label, *sust(g)))
p()

# ── 2. -1s band × reversal_rejection ─────────────────────────────────────────
p("=" * 80)
p("2. VWAP_BAND_TAG = '-1s'  x  REVERSAL_REJECTION_CONFIRMED")
p("=" * 80)
p()
sub1s = df[df["vwap_band_tag"] == "-1s"]
p(f"  Signals at -1s: {len(sub1s)}")
p()
p(HDR())
p(SEP())
for row in rej_split(sub1s, "-1s band"):
    p(row)
p()

# ── 3. outside_vah × reversal_rejection × regime ─────────────────────────────
p("=" * 80)
p("3. PRICE_LOCATION = outside_vah  x  REVERSAL_REJECTION  x  OVERNIGHT_REGIME")
p("=" * 80)
p()
vah = df[df["price_location"] == "outside_vah"]
p(f"  outside_vah signals: {len(vah)}")
p()
p(HDR())
p(SEP())
for row in rej_split(vah, "outside_vah (all regimes)"):
    p(row)
p()
for regime in ["directional", "non_directional", "ambiguous"]:
    sub = vah[vah["overnight_regime"] == regime]
    if len(sub) == 0:
        continue
    p(f"  --- {regime} (n={len(sub)}) ---")
    p(HDR())
    p(SEP())
    for row in rej_split(sub, f"  {regime}"):
        p(row)
    p()

# ── 4. outside_val × reversal_rejection × regime ─────────────────────────────
p("=" * 80)
p("4. PRICE_LOCATION = outside_val  x  REVERSAL_REJECTION  x  OVERNIGHT_REGIME")
p("=" * 80)
p()
val_ = df[df["price_location"] == "outside_val"]
p(f"  outside_val signals: {len(val_)}")
p()
p(HDR())
p(SEP())
for row in rej_split(val_, "outside_val (all regimes)"):
    p(row)
p()
for regime in ["directional", "non_directional", "ambiguous"]:
    sub = val_[val_["overnight_regime"] == regime]
    if len(sub) == 0:
        continue
    p(f"  --- {regime} (n={len(sub)}) ---")
    p(HDR())
    p(SEP())
    for row in rej_split(sub, f"  {regime}"):
        p(row)
    p()

# ── 5. proven node × reversal_rejection ──────────────────────────────────────
p("=" * 80)
p("5. NODE_PROVEN_20T = True  x  REVERSAL_REJECTION_CONFIRMED")
p("=" * 80)
p()
proven = df[df["node_proven_20t"] == True]
p(f"  proven-node signals: {len(proven)}")
p()
p(HDR())
p(SEP())
for row in rej_split(proven, "proven node (all locs)"):
    p(row)
p()

# Proven node broken down by location
p("  --- By price_location (proven nodes only) ---")
p(HDR())
p(SEP())
for loc in ["outside_vah", "outside_val", "at_vah", "at_val", "inside_va"]:
    sub = proven[proven["price_location"] == loc]
    if len(sub) == 0:
        continue
    r_t = sub[sub["reversal_rejection_confirmed"]]
    r_f = sub[~sub["reversal_rejection_confirmed"]]
    n_t, lo_t, ba_t, st_t = sust(r_t)
    n_f, lo_f, ba_f, st_f = sust(r_f)
    n_a, lo_a, ba_a, st_a = sust(sub)
    p(fmt_row(f"proven/{loc}  all",           n_a, lo_a, ba_a, st_a))
    p(fmt_row(f"proven/{loc}  rejection=True",  n_t, lo_t, ba_t, st_t))
    p(fmt_row(f"proven/{loc}  rejection=False", n_f, lo_f, ba_f, st_f))
    p()

# ── 6. Best combinations (triple confluence with rejection) ───────────────────
p("=" * 80)
p("6. TRIPLE CONFLUENCE WITH REVERSAL_REJECTION")
p("=" * 80)
p()
p("  Columns: band + location + rejection, compared to same without rejection filter")
p()
w2 = 52

combos = [
    ("+2s / outside_vah / rej=True",
     (df["vwap_band_tag"] == "+2s") & (df["price_location"] == "outside_vah") & df["reversal_rejection_confirmed"]),
    ("+2s / outside_vah / rej=False",
     (df["vwap_band_tag"] == "+2s") & (df["price_location"] == "outside_vah") & ~df["reversal_rejection_confirmed"]),
    ("+2s / outside_vah (all)",
     (df["vwap_band_tag"] == "+2s") & (df["price_location"] == "outside_vah")),
    ("---", None),
    ("-1s / outside_val / rej=True",
     (df["vwap_band_tag"] == "-1s") & (df["price_location"] == "outside_val") & df["reversal_rejection_confirmed"]),
    ("-1s / outside_val / rej=False",
     (df["vwap_band_tag"] == "-1s") & (df["price_location"] == "outside_val") & ~df["reversal_rejection_confirmed"]),
    ("-1s / outside_val (all)",
     (df["vwap_band_tag"] == "-1s") & (df["price_location"] == "outside_val")),
    ("---", None),
    ("+2s / proven / outside_vah / rej=True",
     (df["vwap_band_tag"] == "+2s") & (df["node_proven_20t"]) & (df["price_location"] == "outside_vah") & df["reversal_rejection_confirmed"]),
    ("+2s / proven / outside_vah / rej=False",
     (df["vwap_band_tag"] == "+2s") & (df["node_proven_20t"]) & (df["price_location"] == "outside_vah") & ~df["reversal_rejection_confirmed"]),
    ("---", None),
    ("-1s / proven / outside_val / rej=True",
     (df["vwap_band_tag"] == "-1s") & (df["node_proven_20t"]) & (df["price_location"] == "outside_val") & df["reversal_rejection_confirmed"]),
    ("-1s / proven / outside_val / rej=False",
     (df["vwap_band_tag"] == "-1s") & (df["node_proven_20t"]) & (df["price_location"] == "outside_val") & ~df["reversal_rejection_confirmed"]),
    ("---", None),
    # Directional regime
    ("+2s / outside_vah / directional / rej=True",
     (df["vwap_band_tag"] == "+2s") & (df["price_location"] == "outside_vah") & (df["overnight_regime"] == "directional") & df["reversal_rejection_confirmed"]),
    ("+2s / outside_vah / directional / rej=False",
     (df["vwap_band_tag"] == "+2s") & (df["price_location"] == "outside_vah") & (df["overnight_regime"] == "directional") & ~df["reversal_rejection_confirmed"]),
    ("-1s / outside_val / directional / rej=True",
     (df["vwap_band_tag"] == "-1s") & (df["price_location"] == "outside_val") & (df["overnight_regime"] == "directional") & df["reversal_rejection_confirmed"]),
    ("-1s / outside_val / directional / rej=False",
     (df["vwap_band_tag"] == "-1s") & (df["price_location"] == "outside_val") & (df["overnight_regime"] == "directional") & ~df["reversal_rejection_confirmed"]),
]

p(HDR(w2))
p(SEP(w2))
for label, mask in combos:
    if mask is None:
        p()
        continue
    g = df[mask]
    p(fmt_row(label, *sust(g), width=w2))

p()

# ── Save ──────────────────────────────────────────────────────────────────────
full_text = "\n".join(lines)
print(full_text)
OUT_PATH.write_text(full_text, encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")

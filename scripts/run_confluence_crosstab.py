"""
Confluence cross-tab analysis on output/absorption_signals_labeled.csv.
Saves to output/diagnostics/confluence_crosstab.txt
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np

df = pd.read_csv(PROJECT_ROOT / "output" / "absorption_signals_labeled.csv")
df["sustained"]        = df["sustained"].astype(bool)
df["sustained_loose"]  = df["sustained_loose"].astype(bool)
df["sustained_strict"] = df["sustained_strict"].astype(bool)

OUT_DIR = PROJECT_ROOT / "output" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sust(grp):
    n = len(grp)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    loose  = ((grp["move_z_score"] >= 1.2) & (grp["mae_normalized"] <= 0.35)).mean() * 100
    base   = ((grp["move_z_score"] >= 1.5) & (grp["mae_normalized"] <= 0.30)).mean() * 100
    strict = ((grp["move_z_score"] >= 2.0) & (grp["mae_normalized"] <= 0.25)).mean() * 100
    return n, loose, base, strict


def fmt_row(label, n, loose, base, strict, width=38):
    def pct(v):
        return f"{v:.1f}%" if not np.isnan(v) else "  n/a"
    return (
        f"  {label:<{width}}  {n:>5}  "
        f"{pct(loose):>8}  {pct(base):>8}  {pct(strict):>8}"
    )


def HDR(w=38):
    return f"  {'group':<{w}}  {'n':>5}  {'loose%':>8}  {'base%':>8}  {'strict%':>8}"


def SEP(w=38):
    return "  " + "-" * (w + 34)


lines = []
def p(s=""):
    lines.append(s)


# ── Header ────────────────────────────────────────────────────────────────────
p("=" * 75)
p("CONFLUENCE CROSS-TAB ANALYSIS")
p("=" * 75)
p(f"Total signals: {len(df)}  |  Training: 2025-03-17 to 2025-11-30")
vc = df["vwap_band_tag"].value_counts().to_dict()
p(f"VWAP band tag counts: {vc}")
p()

# ── 1. +2s x regime x location ───────────────────────────────────────────────
p("=" * 75)
p("1. +2s VWAP BAND x OVERNIGHT_REGIME x PRICE_LOCATION  (n >= 5)")
p("=" * 75)
p("   Definition: vwap_band_tag = '+2s'")
p()
sub = df[df["vwap_band_tag"] == "+2s"]
p(f"  Signals at +2s band: {len(sub)}")
p()
p(HDR(44))
p(SEP(44))

for regime in ["directional", "non_directional", "ambiguous", "unknown"]:
    r = sub[sub["overnight_regime"] == regime]
    if len(r) == 0:
        continue
    n_r, lo, ba, st = sust(r)
    p(fmt_row(f"{regime}  (all locs)", n_r, lo, ba, st, 44))
    for loc in ["outside_vah", "at_vah", "inside_va", "at_val", "outside_val"]:
        g = r[r["price_location"] == loc]
        if len(g) < 5:
            continue
        n_g, lo2, ba2, st2 = sust(g)
        p(fmt_row(f"  {regime} / {loc}", n_g, lo2, ba2, st2, 44))
    p()

# ── 2. +2s x node_proven_20t ─────────────────────────────────────────────────
p("=" * 75)
p("2. +2s VWAP BAND x NODE_PROVEN_20T")
p("=" * 75)
p("   Compares +2s signals with vs without a proven node within 20 ticks")
p()
p(HDR(42))
p(SEP(42))

at2s = df[df["vwap_band_tag"] == "+2s"]
for proven_val, label in [
    (True,  "+2s AND node_proven_20t=True"),
    (False, "+2s AND node_proven_20t=False"),
]:
    g = at2s[at2s["node_proven_20t"] == proven_val]
    n, lo, ba, st = sust(g)
    p(fmt_row(label, n, lo, ba, st, 42))

p()
p("  --- Baseline comparisons ---")
n, lo, ba, st = sust(at2s)
p(fmt_row("+2s band (all)", n, lo, ba, st, 42))
n, lo, ba, st = sust(df)
p(fmt_row("all signals", n, lo, ba, st, 42))
p()

# ── 3. +2s x post_absorption_aggression_3bar ─────────────────────────────────
p("=" * 75)
p("3. +2s VWAP BAND x POST_ABSORPTION_AGGRESSION_3BAR")
p("=" * 75)
p("   post_agg_3bar: majority of next 3 bars confirm absorption direction")
p()
p(HDR(46))
p(SEP(46))

at2s = df[df["vwap_band_tag"] == "+2s"]
for agg_val, label in [
    (True,  "+2s AND post_agg_3bar=True  (confirmed)"),
    (False, "+2s AND post_agg_3bar=False (not confirmed)"),
]:
    g = at2s[at2s["post_absorption_aggression_3bar"] == agg_val]
    n, lo, ba, st = sust(g)
    p(fmt_row(label, n, lo, ba, st, 46))

p()
p("  --- post_agg split across ALL signals (no band filter) ---")
p(HDR(46))
p(SEP(46))
for agg_val, label in [
    (True,  "all / post_agg=True"),
    (False, "all / post_agg=False"),
]:
    g = df[df["post_absorption_aggression_3bar"] == agg_val]
    n, lo, ba, st = sust(g)
    p(fmt_row(label, n, lo, ba, st, 46))
p()

# ── 4. -1s x regime x location ───────────────────────────────────────────────
p("=" * 75)
p("4. -1s VWAP BAND x OVERNIGHT_REGIME x PRICE_LOCATION  (n >= 5)")
p("=" * 75)
p("   Definition: vwap_band_tag = '-1s'")
p()
sub = df[df["vwap_band_tag"] == "-1s"]
p(f"  Signals at -1s band: {len(sub)}")
p()
p(HDR(44))
p(SEP(44))

for regime in ["directional", "non_directional", "ambiguous", "unknown"]:
    r = sub[sub["overnight_regime"] == regime]
    if len(r) == 0:
        continue
    n_r, lo, ba, st = sust(r)
    p(fmt_row(f"{regime}  (all locs)", n_r, lo, ba, st, 44))
    for loc in ["outside_vah", "at_vah", "inside_va", "at_val", "outside_val"]:
        g = r[r["price_location"] == loc]
        if len(g) < 5:
            continue
        n_g, lo2, ba2, st2 = sust(g)
        p(fmt_row(f"  {regime} / {loc}", n_g, lo2, ba2, st2, 44))
    p()

# ── 5. proven node x location x regime (outside only) ────────────────────────
p("=" * 75)
p("5. PROVEN NODE (node_proven_20t=True) x PRICE_LOCATION x OVERNIGHT_REGIME")
p("   Focus: outside_vah and outside_val rows")
p("=" * 75)
p()
proven   = df[df["node_proven_20t"] == True]
unproven = df[df["node_proven_20t"] == False]
outside_mask = df["price_location"].isin(["outside_vah", "outside_val"])
p(f"  Proven-node signals total: {len(proven)}  |  outside_vah/val: {len(proven[outside_mask])}")
p()
p(HDR(46))
p(SEP(46))

for loc in ["outside_vah", "outside_val"]:
    loc_grp = proven[proven["price_location"] == loc]
    if len(loc_grp) == 0:
        continue
    n, lo, ba, st = sust(loc_grp)
    p(fmt_row(f"{loc}  (all regimes)", n, lo, ba, st, 46))
    for regime in ["directional", "non_directional", "ambiguous", "unknown"]:
        g = loc_grp[loc_grp["overnight_regime"] == regime]
        if len(g) == 0:
            continue
        n2, lo2, ba2, st2 = sust(g)
        p(fmt_row(f"  {loc} / {regime}", n2, lo2, ba2, st2, 46))
    p()

p("  --- Comparison: UNproven node at same locations ---")
p(HDR(46))
p(SEP(46))
for loc in ["outside_vah", "outside_val"]:
    g = unproven[unproven["price_location"] == loc]
    if len(g) == 0:
        continue
    n, lo, ba, st = sust(g)
    p(fmt_row(f"unproven / {loc}", n, lo, ba, st, 46))
p()

# ── 6. Triple (and quad) confluence ──────────────────────────────────────────
p("=" * 75)
p("6. TRIPLE / QUAD CONFLUENCE  (all sample sizes shown)")
p("=" * 75)
p()

combos = [
    # Triple
    ("+2s / proven / outside_vah",
     (df["vwap_band_tag"] == "+2s") & (df["node_proven_20t"] == True)  & (df["price_location"] == "outside_vah")),
    ("+2s / proven / outside_val",
     (df["vwap_band_tag"] == "+2s") & (df["node_proven_20t"] == True)  & (df["price_location"] == "outside_val")),
    ("+2s / unproven / outside_vah",
     (df["vwap_band_tag"] == "+2s") & (df["node_proven_20t"] == False) & (df["price_location"] == "outside_vah")),
    ("-1s / proven / outside_val",
     (df["vwap_band_tag"] == "-1s") & (df["node_proven_20t"] == True)  & (df["price_location"] == "outside_val")),
    ("-1s / proven / outside_vah",
     (df["vwap_band_tag"] == "-1s") & (df["node_proven_20t"] == True)  & (df["price_location"] == "outside_vah")),
    ("-1s / unproven / outside_val",
     (df["vwap_band_tag"] == "-1s") & (df["node_proven_20t"] == False) & (df["price_location"] == "outside_val")),
    # + regime
    ("+2s / proven / outside_vah / directional",
     (df["vwap_band_tag"] == "+2s") & (df["node_proven_20t"] == True) &
     (df["price_location"] == "outside_vah") & (df["overnight_regime"] == "directional")),
    ("-1s / proven / outside_val / directional",
     (df["vwap_band_tag"] == "-1s") & (df["node_proven_20t"] == True) &
     (df["price_location"] == "outside_val") & (df["overnight_regime"] == "directional")),
    ("+2s / proven / outside_val / directional",
     (df["vwap_band_tag"] == "+2s") & (df["node_proven_20t"] == True) &
     (df["price_location"] == "outside_val") & (df["overnight_regime"] == "directional")),
    ("-1s / proven / outside_vah / directional",
     (df["vwap_band_tag"] == "-1s") & (df["node_proven_20t"] == True) &
     (df["price_location"] == "outside_vah") & (df["overnight_regime"] == "directional")),
    # + post_agg (quad)
    ("+2s / proven / outside_vah / dir / post_agg",
     (df["vwap_band_tag"] == "+2s") & (df["node_proven_20t"] == True) &
     (df["price_location"] == "outside_vah") & (df["overnight_regime"] == "directional") &
     (df["post_absorption_aggression_3bar"] == True)),
    ("-1s / proven / outside_val / dir / post_agg",
     (df["vwap_band_tag"] == "-1s") & (df["node_proven_20t"] == True) &
     (df["price_location"] == "outside_val") & (df["overnight_regime"] == "directional") &
     (df["post_absorption_aggression_3bar"] == True)),
    # Mirror: +2s at val, -1s at vah
    ("+2s / proven / outside_val / dir / post_agg",
     (df["vwap_band_tag"] == "+2s") & (df["node_proven_20t"] == True) &
     (df["price_location"] == "outside_val") & (df["overnight_regime"] == "directional") &
     (df["post_absorption_aggression_3bar"] == True)),
    ("-1s / proven / outside_vah / dir / post_agg",
     (df["vwap_band_tag"] == "-1s") & (df["node_proven_20t"] == True) &
     (df["price_location"] == "outside_vah") & (df["overnight_regime"] == "directional") &
     (df["post_absorption_aggression_3bar"] == True)),
]

p(HDR(54))
p(SEP(54))
for label, mask in combos:
    g = df[mask]
    n, lo, ba, st = sust(g)
    p(fmt_row(label, n, lo, ba, st, 54))
p()

# ── Appendix: band coverage by regime ────────────────────────────────────────
p("=" * 75)
p("APPENDIX: VWAP BAND COVERAGE BY REGIME")
p("=" * 75)
p()
p("  Signals per band tag within each overnight regime")
p()
for regime in ["directional", "non_directional", "ambiguous", "unknown"]:
    r = df[df["overnight_regime"] == regime]
    p(f"  {regime} (n={len(r)}):")
    vc = r["vwap_band_tag"].value_counts()
    for tag in ["+2s", "+1s", "vwap", "-1s", "-2s", "none"]:
        cnt = vc.get(tag, 0)
        pct = cnt / len(r) * 100 if len(r) else 0
        p(f"    {tag:<8} {cnt:>5}  ({pct:.1f}%)")
    p()

# ── Save ──────────────────────────────────────────────────────────────────────
full_text = "\n".join(lines)
print(full_text)
out_path = OUT_DIR / "confluence_crosstab.txt"
out_path.write_text(full_text, encoding="utf-8")
print(f"\nSaved to {out_path}")

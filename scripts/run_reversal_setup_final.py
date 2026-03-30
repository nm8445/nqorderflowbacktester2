"""
Progressive filter stacks for short and long reversal setups.
Computes cluster_position and total_vol_rolling_pct on the fly.
Saves output/diagnostics/reversal_setup_final.txt
Saves JSON rule files for both setups.
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
OUT_TXT  = OUT_DIR / "reversal_setup_final.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ET = "America/New_York"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
for col in ["sustained", "sustained_loose", "sustained_strict",
            "reversal_rejection_confirmed", "post_absorption_aggression_3bar",
            "node_proven_20t"]:
    df[col] = df[col].astype(bool)

# ── Compute cluster_position ──────────────────────────────────────────────────
df_clustered, _ = signal_clustering(df)
df["cluster_position"] = df_clustered["cluster_position"]

# ── Compute total_vol_rolling_pct ─────────────────────────────────────────────
# For each signal: percentile rank of its total_vol within the distribution
# of all total_vol values across the prior 20 trading sessions.
signal_time_et = pd.to_datetime(df["signal_time"], utc=True).dt.tz_convert(ET)
df["_date_et"] = signal_time_et.dt.date

all_dates    = sorted(df["_date_et"].unique())
date_to_idx  = {d: i for i, d in enumerate(all_dates)}
WINDOW       = 20

rolling_pct = pd.Series(np.nan, index=df.index)

for date in all_dates:
    d_idx = date_to_idx[date]
    if d_idx < 1:
        # No prior sessions — assign 50th pct (neutral)
        mask = df["_date_et"] == date
        rolling_pct[mask] = 50.0
        continue
    # Collect total_vol from prior WINDOW sessions
    prior_dates = all_dates[max(0, d_idx - WINDOW) : d_idx]
    prior_vols  = df[df["_date_et"].isin(prior_dates)]["total_vol"].values
    if len(prior_vols) == 0:
        mask = df["_date_et"] == date
        rolling_pct[mask] = 50.0
        continue
    # For each signal on this date compute its rolling percentile rank
    mask = df["_date_et"] == date
    for idx in df[mask].index:
        v = df.at[idx, "total_vol"]
        pct = float(np.mean(prior_vols <= v) * 100)
        rolling_pct[idx] = pct

df["total_vol_rolling_pct"] = rolling_pct
df = df.drop(columns=["_date_et"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def sust(grp):
    n = len(grp)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    lo = ((grp["move_z_score"] >= 1.2) & (grp["mae_normalized"] <= 0.35)).mean() * 100
    ba = ((grp["move_z_score"] >= 1.5) & (grp["mae_normalized"] <= 0.30)).mean() * 100
    st = ((grp["move_z_score"] >= 2.0) & (grp["mae_normalized"] <= 0.25)).mean() * 100
    return n, lo, ba, st


def fmt_step(step_label, grp, cumulative_filter_label=""):
    n, lo, ba, st = sust(grp)
    def pct(v): return f"{v:.1f}%" if not np.isnan(v) else "  n/a"
    note = f"  [{cumulative_filter_label}]" if cumulative_filter_label else ""
    return (
        f"  {step_label:<50}  {n:>5}  "
        f"{pct(lo):>8}  {pct(ba):>8}  {pct(st):>8}{note}"
    )

HDR_W = 50
def HDR():
    return f"  {'step':<{HDR_W}}  {'n':>5}  {'loose%':>8}  {'base%':>8}  {'strict%':>8}"
def SEP():
    return "  " + "-" * (HDR_W + 36)


lines = []
def p(s=""): lines.append(s)


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESSIVE FILTER STACK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_stack(title, steps, base_df):
    """
    steps: list of (label, filter_fn_or_None)
    filter_fn is applied cumulatively: each step filters the output of the prior.
    """
    p("=" * 85)
    p(title)
    p("=" * 85)
    p()
    p(HDR())
    p(SEP())

    current = base_df.copy()
    for label, fn in steps:
        if fn is not None:
            current = current[fn(current)].copy()
        p(fmt_step(label, current))

    p()
    return current   # return final filtered set


# ══════════════════════════════════════════════════════════════════════════════
# SHORT SETUP — UPPER EXTREME REJECTION
# ══════════════════════════════════════════════════════════════════════════════

p("=" * 85)
p("SHORT REVERSAL SETUP — UPPER EXTREME REJECTION")
p("Logic: sell_absorbed at outside_vah near +2s VWAP band, with high-vol opposing")
p("       sellers stepping in to confirm rejection. Node proven nearby.")
p("=" * 85)
p()

short_steps_base = [
    ("Baseline: all signals",                    None),
    ("Step 1: price_location = outside_vah",     lambda d: d["price_location"] == "outside_vah"),
    ("Step 2: absorption_side = sell_absorbed",  lambda d: d["absorption_side"] == "sell_absorbed"),
    ("Step 3: vwap_band_tag=+2s OR at_vwap_band",lambda d: (d["vwap_band_tag"] == "+2s") | d["at_vwap_band"]),
    ("Step 4: reversal_rejection_confirmed=True",lambda d: d["reversal_rejection_confirmed"] == True),
    ("Step 5: node_proven_20t=True",             lambda d: d["node_proven_20t"] == True),
    ("Step 6: cluster_position=solo or first",   lambda d: d["cluster_position"].isin(["solo", "first"])),
    ("Step 7: total_vol >= 60th rolling pct",    lambda d: d["total_vol_rolling_pct"] >= 60),
]

p("--- NON_DIRECTIONAL REGIME ---")
p()

short_nd_steps = short_steps_base + [
    ("Step 8: overnight_regime=non_directional", lambda d: d["overnight_regime"] == "non_directional"),
]
run_stack("SHORT / NON_DIRECTIONAL", short_nd_steps, df)

# Show step 7 with/without for comparison
p("  [ Vol filter impact at Step 6 -> 7 (no regime filter yet) ]")
p(HDR())
p(SEP())
_s6 = df.copy()
for _, fn in short_steps_base[:7]:   # steps 0-6
    if fn: _s6 = _s6[fn(_s6)]
p(fmt_step("  After step 6 (no vol filter)", _s6))
_s7 = _s6[_s6["total_vol_rolling_pct"] >= 60]
p(fmt_step("  After step 7 (vol >= 60th)",   _s7))
p()

p("--- DIRECTIONAL REGIME ---")
p()

short_dir_steps = short_steps_base + [
    ("Step 8: overnight_regime=directional",     lambda d: d["overnight_regime"] == "directional"),
]
run_stack("SHORT / DIRECTIONAL", short_dir_steps, df)

p("  [ Vol filter impact (directional) ]")
p(HDR())
p(SEP())
_s6d = df.copy()
for _, fn in short_steps_base[:7]:
    if fn: _s6d = _s6d[fn(_s6d)]
_s6d_dir = _s6d[_s6d["overnight_regime"] == "directional"]
_s7d_dir = _s6d_dir[_s6d_dir["total_vol_rolling_pct"] >= 60]
p(fmt_step("  After step 6 + directional (no vol)", _s6d_dir))
p(fmt_step("  After step 7 + directional (vol>=60)", _s7d_dir))
p()

p("--- ALL REGIMES COMBINED (final filter set) ---")
p()
run_stack("SHORT / ALL REGIMES", short_steps_base, df)

# ══════════════════════════════════════════════════════════════════════════════
# LONG SETUP — LOWER EXTREME DEMAND HOLD
# ══════════════════════════════════════════════════════════════════════════════

p("=" * 85)
p("LONG REVERSAL SETUP — LOWER EXTREME DEMAND HOLD")
p("Logic: buy_absorbed at outside_val WITHOUT immediate rejection (demand held).")
p("       Proven node nearby, no high-vol sellers stepping in = demand absorbed.")
p("=" * 85)
p()

long_steps_base = [
    ("Baseline: all signals",                    None),
    ("Step 1: price_location = outside_val",     lambda d: d["price_location"] == "outside_val"),
    ("Step 2: absorption_side = buy_absorbed",   lambda d: d["absorption_side"] == "buy_absorbed"),
    ("Step 3: reversal_rejection_confirmed=False",lambda d: d["reversal_rejection_confirmed"] == False),
    ("Step 4: node_proven_20t=True",             lambda d: d["node_proven_20t"] == True),
    ("Step 5: cluster_position=solo or first",   lambda d: d["cluster_position"].isin(["solo", "first"])),
    ("Step 6: total_vol >= 60th rolling pct",    lambda d: d["total_vol_rolling_pct"] >= 60),
]

p("--- NON_DIRECTIONAL REGIME ---")
p()

long_nd_steps = long_steps_base + [
    ("Step 7: overnight_regime=non_directional", lambda d: d["overnight_regime"] == "non_directional"),
]
run_stack("LONG / NON_DIRECTIONAL", long_nd_steps, df)

p("  [ Vol filter impact at Step 5 -> 6 (no regime filter yet) ]")
p(HDR())
p(SEP())
_l5 = df.copy()
for _, fn in long_steps_base[:6]:
    if fn: _l5 = _l5[fn(_l5)]
p(fmt_step("  After step 5 (no vol filter)", _l5))
_l6 = _l5[_l5["total_vol_rolling_pct"] >= 60]
p(fmt_step("  After step 6 (vol >= 60th)",   _l6))
p()

p("--- DIRECTIONAL REGIME ---")
p()

long_dir_steps = long_steps_base + [
    ("Step 7: overnight_regime=directional",     lambda d: d["overnight_regime"] == "directional"),
]
run_stack("LONG / DIRECTIONAL", long_dir_steps, df)

p("  [ Vol filter impact (directional) ]")
p(HDR())
p(SEP())
_l5d = df.copy()
for _, fn in long_steps_base[:6]:
    if fn: _l5d = _l5d[fn(_l5d)]
_l5d_dir = _l5d[_l5d["overnight_regime"] == "directional"]
_l6d_dir = _l5d_dir[_l5d_dir["total_vol_rolling_pct"] >= 60]
p(fmt_step("  After step 5 + directional (no vol)", _l5d_dir))
p(fmt_step("  After step 6 + directional (vol>=60)", _l6d_dir))
p()

p("--- ALL REGIMES COMBINED (final filter set) ---")
p()
run_stack("LONG / ALL REGIMES", long_steps_base, df)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

p("=" * 85)
p("FINAL SETUP SUMMARY (full filter stack including vol, excluding regime filter)")
p("=" * 85)
p()

def apply_stack(steps, base):
    cur = base.copy()
    for _, fn in steps:
        if fn: cur = cur[fn(cur)]
    return cur

short_final = apply_stack(short_steps_base, df)
long_final  = apply_stack(long_steps_base,  df)

p(HDR())
p(SEP())
p(fmt_step("SHORT: all_regimes (steps 1-7)", short_final))
p(fmt_step("SHORT: non_directional",  short_final[short_final["overnight_regime"] == "non_directional"]))
p(fmt_step("SHORT: directional",      short_final[short_final["overnight_regime"] == "directional"]))
p()
p(fmt_step("LONG:  all_regimes (steps 1-6)", long_final))
p(fmt_step("LONG:  non_directional",  long_final[long_final["overnight_regime"] == "non_directional"]))
p(fmt_step("LONG:  directional",      long_final[long_final["overnight_regime"] == "directional"]))
p()
p(fmt_step("Baseline: all signals",   df))
p()


# ── Print + Save text ─────────────────────────────────────────────────────────
full_text = "\n".join(lines)
print(full_text)
OUT_TXT.write_text(full_text, encoding="utf-8")
print(f"\nSaved to {OUT_TXT}")


# ── JSON rule files ───────────────────────────────────────────────────────────

short_rules = {
    "name": "short_reversal_setup_v1",
    "description": (
        "Upper extreme rejection: sell_absorbed at outside_vah near +2s VWAP band. "
        "High-vol opposing sellers confirm rejection. Proven node within 20t."
    ),
    "filters": {
        "price_location":              {"op": "eq",   "value": "outside_vah"},
        "absorption_side":             {"op": "eq",   "value": "sell_absorbed"},
        "vwap_band":                   {"op": "eq",   "value": "+2s",
                                        "alt": "at_vwap_band=True",
                                        "logic": "vwap_band_tag=+2s OR at_vwap_band=True"},
        "reversal_rejection_confirmed":{"op": "eq",   "value": True},
        "node_proven_20t":             {"op": "eq",   "value": True},
        "cluster_position":            {"op": "isin", "value": ["solo", "first"]},
        "total_vol_rolling_pct":       {"op": "gte",  "value": 60, "window_sessions": 20},
        "overnight_regime":            {"op": "eq",   "value": "non_directional",
                                        "note": "primary filter; directional adds n but not rate"}
    },
    "training_stats": {
        "non_directional": {
            "n":          int(short_final[short_final["overnight_regime"] == "non_directional"].shape[0]),
            "loose_pct":  round(sust(short_final[short_final["overnight_regime"] == "non_directional"])[1], 1),
            "base_pct":   round(sust(short_final[short_final["overnight_regime"] == "non_directional"])[2], 1),
            "strict_pct": round(sust(short_final[short_final["overnight_regime"] == "non_directional"])[3], 1),
        },
        "directional": {
            "n":          int(short_final[short_final["overnight_regime"] == "directional"].shape[0]),
            "loose_pct":  round(sust(short_final[short_final["overnight_regime"] == "directional"])[1], 1),
            "base_pct":   round(sust(short_final[short_final["overnight_regime"] == "directional"])[2], 1),
            "strict_pct": round(sust(short_final[short_final["overnight_regime"] == "directional"])[3], 1),
        },
        "all_regimes": {
            "n":          int(short_final.shape[0]),
            "loose_pct":  round(sust(short_final)[1], 1),
            "base_pct":   round(sust(short_final)[2], 1),
            "strict_pct": round(sust(short_final)[3], 1),
        }
    }
}

long_rules = {
    "name": "long_reversal_setup_v1",
    "description": (
        "Lower extreme demand hold: buy_absorbed at outside_val WITHOUT immediate "
        "high-vol rejection. Proven node within 20t. Demand absorbed quietly."
    ),
    "filters": {
        "price_location":              {"op": "eq",   "value": "outside_val"},
        "absorption_side":             {"op": "eq",   "value": "buy_absorbed"},
        "reversal_rejection_confirmed":{"op": "eq",   "value": False,
                                        "note": "rejection=False means no high-vol opposing side — demand held"},
        "node_proven_20t":             {"op": "eq",   "value": True},
        "cluster_position":            {"op": "isin", "value": ["solo", "first"]},
        "total_vol_rolling_pct":       {"op": "gte",  "value": 60, "window_sessions": 20},
        "overnight_regime":            {"op": "eq",   "value": "non_directional",
                                        "note": "primary filter; directional adds n but not rate"}
    },
    "training_stats": {
        "non_directional": {
            "n":          int(long_final[long_final["overnight_regime"] == "non_directional"].shape[0]),
            "loose_pct":  round(sust(long_final[long_final["overnight_regime"] == "non_directional"])[1], 1),
            "base_pct":   round(sust(long_final[long_final["overnight_regime"] == "non_directional"])[2], 1),
            "strict_pct": round(sust(long_final[long_final["overnight_regime"] == "non_directional"])[3], 1),
        },
        "directional": {
            "n":          int(long_final[long_final["overnight_regime"] == "directional"].shape[0]),
            "loose_pct":  round(sust(long_final[long_final["overnight_regime"] == "directional"])[1], 1),
            "base_pct":   round(sust(long_final[long_final["overnight_regime"] == "directional"])[2], 1),
            "strict_pct": round(sust(long_final[long_final["overnight_regime"] == "directional"])[3], 1),
        },
        "all_regimes": {
            "n":          int(long_final.shape[0]),
            "loose_pct":  round(sust(long_final)[1], 1),
            "base_pct":   round(sust(long_final)[2], 1),
            "strict_pct": round(sust(long_final)[3], 1),
        }
    }
}

short_json = OUT_DIR / "short_reversal_setup_rules.json"
long_json  = OUT_DIR / "long_reversal_setup_rules.json"

short_json.write_text(json.dumps(short_rules, indent=2), encoding="utf-8")
long_json.write_text(json.dumps(long_rules,  indent=2), encoding="utf-8")
print(f"Saved {short_json}")
print(f"Saved {long_json}")

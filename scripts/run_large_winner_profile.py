"""
Large winner adverse excursion profile.

Analyzes the 'large' outcome bucket (mfe_raw >= 50 AND mfe_raw > mae_raw)
to determine:
  1. Distribution of mae_raw (max adverse over session) for large winners
  2-4. How many large winners had mae_raw > 10 / 20 / 40 pts
  5. Bar-by-bar: how many bars elapsed before running_mfe first exceeded
     running_mae (i.e. how long underwater before trade turned "net positive")
  6. Subset with mae_raw < 10 (clean entries): count and mean mfe_raw

Saves output/diagnostics/large_winner_adverse_profile.txt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from nqbt.analysis.reaction_scanner import _load_rth_ticks
from nqbt.analysis.range_bars import build_range_bars

OUT_DIR     = PROJECT_ROOT / "output" / "diagnostics"
EVENTS_PATH = OUT_DIR / "reaction_events.csv"
REPORT_PATH = OUT_DIR / "large_winner_adverse_profile.txt"

# ---------------------------------------------------------------------------
# Load events and filter to large bucket
# ---------------------------------------------------------------------------

print("Loading reaction events...")
events = pd.read_csv(EVENTS_PATH)
print(f"  Total events: {len(events)}")

# Assign direction
# reaction_events.csv has std2_upper / std2_lower as absolute prices
def _direction(loc: str, bar_close: float, std2_upper: float, std2_lower: float) -> str:
    if loc in ("outside_vah", "at_vah"):
        return "short"
    if loc in ("outside_val", "at_val"):
        return "long"
    # VWAP-band-only event: use absolute band prices
    if not (pd.isna(std2_upper) or pd.isna(std2_lower) or pd.isna(bar_close)):
        if bar_close >= std2_upper:
            return "short"
        if bar_close <= std2_lower:
            return "long"
    return "unknown"

events["_dir"] = events.apply(
    lambda r: _direction(
        str(r["price_location"]),
        float(r["bar_close"]),
        float(r["std2_upper"]) if pd.notna(r.get("std2_upper")) else float("nan"),
        float(r["std2_lower"]) if pd.notna(r.get("std2_lower")) else float("nan"),
    ), axis=1
)

# Compute mfe_raw and mae_raw
events["_mfe_raw"] = np.where(
    events["_dir"] == "short", events["mfe_down"],
    np.where(events["_dir"] == "long", events["mfe_up"], float("nan"))
)
events["_mae_raw"] = np.where(
    events["_dir"] == "short", events["mfe_up"],
    np.where(events["_dir"] == "long", events["mfe_down"], float("nan"))
)

# Large bucket: mfe_raw >= 50 AND mfe_raw > mae_raw AND direction known
large = events[
    (events["_mfe_raw"] >= 50.0) &
    (events["_mfe_raw"] > events["_mae_raw"]) &
    (events["_dir"] != "unknown")
].copy()

print(f"  Large events: {len(large)}")

mae  = large["_mae_raw"].dropna()
mfe  = large["_mfe_raw"].dropna()
n_lg = len(large)

# ---------------------------------------------------------------------------
# Q1: Mae_raw distribution
# ---------------------------------------------------------------------------

mae_mean   = float(mae.mean())
mae_median = float(mae.median())
mae_p25    = float(mae.quantile(0.25))
mae_p75    = float(mae.quantile(0.75))
mae_p90    = float(mae.quantile(0.90))

# Q2-4
n_mae_gt10 = int((mae > 10).sum())
n_mae_gt20 = int((mae > 20).sum())
n_mae_gt40 = int((mae > 40).sum())

# Q6: clean entries (mae_raw < 10)
clean = large[large["_mae_raw"] < 10.0]
n_clean       = len(clean)
clean_mfe_mean = float(clean["_mfe_raw"].mean()) if n_clean > 0 else float("nan")
clean_frac     = n_clean / n_lg if n_lg > 0 else 0.0

# ---------------------------------------------------------------------------
# Q5: Bar-by-bar replay — bars until running_mfe > running_mae
# Group large events by session date to avoid reloading ticks repeatedly
# ---------------------------------------------------------------------------

print("Running bar-by-bar replay for Q5 (bars until mfe > mae)...")
print(f"  Processing {n_lg} large events across {large['session_date'].nunique()} sessions...")

crossover_bars: list[int | None] = []   # None = never crossed (shouldn't happen for large)
first_mae_before_crossover: list[float] = []

n_sessions = large["session_date"].nunique()
session_num = 0

for session_date, grp in large.groupby("session_date"):
    session_num += 1
    if session_num % 20 == 0:
        print(f"    Session {session_num}/{n_sessions}: {session_date} ({len(grp)} events)")

    ticks = _load_rth_ticks(session_date)
    if ticks.empty:
        for _ in range(len(grp)):
            crossover_bars.append(None)
            first_mae_before_crossover.append(float("nan"))
        continue

    bars = build_range_bars(ticks, range_ticks=40)

    for _, ev in grp.iterrows():
        bar_idx   = int(ev["bar_idx"])
        direction = ev["_dir"]
        entry     = float(ev["bar_close"])

        forward_bars = bars[bar_idx + 1:]

        if not forward_bars:
            crossover_bars.append(0)
            first_mae_before_crossover.append(0.0)
            continue

        running_mfe = 0.0
        running_mae = 0.0
        crossed_at  = None
        mae_at_cross = float("nan")

        for i, bar in enumerate(forward_bars):
            if direction == "short":
                bar_mfe = entry - bar.low
                bar_mae = bar.high - entry
            else:
                bar_mfe = bar.high - entry
                bar_mae = entry - bar.low

            running_mfe = max(running_mfe, bar_mfe)
            running_mae = max(running_mae, bar_mae)

            if crossed_at is None and running_mfe > running_mae:
                crossed_at   = i + 1   # 1-based bar count
                mae_at_cross = running_mae
                break   # stop once crossed — we have what we need

        if crossed_at is None:
            # Running mfe never exceeded running mae in forward bars
            # (shouldn't occur for true large events but handle gracefully)
            crossed_at   = len(forward_bars)
            mae_at_cross = running_mae

        crossover_bars.append(crossed_at)
        first_mae_before_crossover.append(mae_at_cross)

# Attach results back to large DataFrame
large = large.copy()
large["_crossover_bars"]  = crossover_bars
large["_mae_at_crossover"] = first_mae_before_crossover

cb = large["_crossover_bars"].dropna()
mc = large["_mae_at_crossover"].dropna()

n_within_3  = int((cb <= 3).sum())
n_within_5  = int((cb <= 5).sum())
n_within_10 = int((cb <= 10).sum())
n_within_20 = int((cb <= 20).sum())
n_never     = int(cb.isna().sum())   # truly never (edge case)

print("Done. Building report...")

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

lines: list[str] = []


def p(s: str = "") -> None:
    lines.append(s)


p("=" * 70)
p("LARGE WINNER ADVERSE EXCURSION PROFILE")
p(f"Large bucket: mfe_raw >= 50 pts AND mfe_raw > mae_raw")
p(f"Total large events: {n_lg:,}  ({100*n_lg/len(events):.1f}% of all {len(events):,} events)")
p("=" * 70)
p()

# ── Q1 ───────────────────────────────────────────────────────────────────────
p("-" * 70)
p("Q1: DISTRIBUTION OF mae_raw (max adverse excursion over full session)")
p("    This is the maximum adverse move from entry at any point in the session.")
p("    If your stop <= mae_raw, it would have been hit at some point.")
p("-" * 70)
p()
p(f"  mean   : {mae_mean:>8.2f} pts")
p(f"  median : {mae_median:>8.2f} pts")
p(f"  p25    : {mae_p25:>8.2f} pts")
p(f"  p75    : {mae_p75:>8.2f} pts")
p(f"  p90    : {mae_p90:>8.2f} pts")
p()

# Histogram buckets
buckets = [(0,5), (5,10), (10,15), (15,20), (20,30), (30,40), (40,50), (50, 9999)]
p("  mae_raw distribution (all large events):")
for lo, hi in buckets:
    label = f"  {lo:>2}-{hi if hi < 9999 else '50+':>4} pts"
    n = int(((mae >= lo) & (mae < hi)).sum())
    bar = "#" * int(n / n_lg * 50)
    p(f"  {lo:>3}-{str(hi) if hi < 9999 else '50+':>4} pts : {n:>5}  ({100*n/n_lg:>5.1f}%)  {bar}")
p()

# ── Q2-4 ─────────────────────────────────────────────────────────────────────
p("-" * 70)
p("Q2-4: HOW MANY LARGE WINNERS WOULD BE STOPPED AT EACH LEVEL?")
p("      (mae_raw > threshold => stop at that level would be triggered)")
p("-" * 70)
p()
for thr, n_hit in [(10, n_mae_gt10), (20, n_mae_gt20), (40, n_mae_gt40)]:
    pct_hit     = 100.0 * n_hit / n_lg if n_lg else 0.0
    n_survived  = n_lg - n_hit
    pct_survived = 100.0 - pct_hit
    p(f"  Stop at {thr:>2} pts : {n_hit:>5} stopped ({pct_hit:>5.1f}%)  |  {n_survived:>5} survived ({pct_survived:>5.1f}%)")
p()

# ── Q5 ───────────────────────────────────────────────────────────────────────
p("-" * 70)
p("Q5: BARS UNTIL running_mfe FIRST EXCEEDED running_mae")
p("    i.e. how many forward bars elapsed before the trade was 'net positive'")
p("    (running favorable excursion > running adverse excursion)")
p("-" * 70)
p()
p(f"  Within  3 bars : {n_within_3:>5}  ({100*n_within_3/n_lg:>5.1f}%)")
p(f"  Within  5 bars : {n_within_5:>5}  ({100*n_within_5/n_lg:>5.1f}%)")
p(f"  Within 10 bars : {n_within_10:>5}  ({100*n_within_10/n_lg:>5.1f}%)")
p(f"  Within 20 bars : {n_within_20:>5}  ({100*n_within_20/n_lg:>5.1f}%)")
if n_never > 0:
    p(f"  Never          : {n_never:>5}  ({100*n_never/n_lg:>5.1f}%)  [edge case]")
p()

# Distribution of crossover bars
cb_vals = cb.dropna().to_numpy()
if len(cb_vals) > 0:
    p(f"  Crossover bar stats:")
    p(f"    mean   : {cb_vals.mean():>6.1f} bars")
    p(f"    median : {np.median(cb_vals):>6.1f} bars")
    p(f"    p75    : {np.percentile(cb_vals, 75):>6.1f} bars")
    p(f"    p90    : {np.percentile(cb_vals, 90):>6.1f} bars")
    p(f"    max    : {cb_vals.max():>6.0f} bars")
p()

# MAE at crossover point
if len(mc) > 0:
    p(f"  MAE at crossover point (adverse absorbed before trade turned positive):")
    p(f"    mean   : {mc.mean():>6.2f} pts")
    p(f"    median : {mc.median():>6.2f} pts")
    p(f"    p75    : {mc.quantile(0.75):>6.2f} pts")
    p(f"    p90    : {mc.quantile(0.90):>6.2f} pts")
p()

# ── Q6 ───────────────────────────────────────────────────────────────────────
p("-" * 70)
p("Q6: CLEAN ENTRIES (mae_raw < 10 pts)")
p("    Events where adverse excursion never exceeded 10 pts during session")
p("-" * 70)
p()
p(f"  Clean large events : {n_clean:>5}  ({100*clean_frac:>5.1f}% of all {n_lg:,} large events)")
p(f"  Mean mfe_raw       : {clean_mfe_mean:>7.2f} pts")
if n_clean > 0:
    p(f"  Median mfe_raw     : {float(clean['_mfe_raw'].median()):>7.2f} pts")
    p(f"  Mean mae_raw       : {float(clean['_mae_raw'].mean()):>7.2f} pts  (by definition < 10)")
p()

# ── Summary / implications ────────────────────────────────────────────────────
p("=" * 70)
p("IMPLICATIONS FOR STOP PLACEMENT")
p("=" * 70)
p()
p(f"  A 10-pt stop would be triggered on {100*n_mae_gt10/n_lg:.1f}% of large winners")
p(f"  A 20-pt stop would be triggered on {100*n_mae_gt20/n_lg:.1f}% of large winners")
p(f"  A 40-pt stop would be triggered on {100*n_mae_gt40/n_lg:.1f}% of large winners")
p()
p(f"  Median time to trade turning 'net positive': {np.median(cb_vals):.0f} bars")
p(f"  90th percentile: {np.percentile(cb_vals, 90):.0f} bars to crossover")
p()
p(f"  {100*clean_frac:.1f}% of large winners never went more than 10 pts adverse.")
p(f"  These clean entries had mean mfe_raw = {clean_mfe_mean:.1f} pts.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

report_text = "\n".join(lines)
REPORT_PATH.write_text(report_text, encoding="utf-8")
print(f"\nReport saved to {REPORT_PATH}")

# Print report to terminal
try:
    print("\n" + report_text)
except UnicodeEncodeError:
    print("\n" + report_text.encode("ascii", errors="replace").decode("ascii"))

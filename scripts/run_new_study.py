"""
New study pipeline runner.

Processes 2025-03-17 to 2025-11-30. For each session:
  1. Runs reaction_scanner to find qualifying location events
  2. Builds feature vectors for each event
  3. Runs trailing stop simulations

Saves:
  output/diagnostics/reaction_events.csv
  output/diagnostics/feature_vectors.csv
  output/diagnostics/trailing_simulation_results.csv
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

START   = "2025-03-17"
END     = "2025-11-30"
OUT_DIR = PROJECT_ROOT / "output" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd
from nqbt.analysis.reaction_scanner import scan_date_range, _load_rth_ticks
from nqbt.analysis.range_bars import build_range_bars

_events_path = OUT_DIR / "reaction_events.csv"
if _events_path.exists():
    print("Step 1: Loading existing reaction_events.csv (skip rescan)...")
    events = pd.read_csv(_events_path)
    print(f"  -> {len(events)} events loaded")
else:
    print("Step 1: Scanning sessions for reaction events...")
    events = scan_date_range(START, END)
    events.to_csv(_events_path, index=False)
    print(f"  -> {len(events)} events saved")

# Step 2: preload bars and ticks for feature building
print("Step 2: Building feature vectors...")
bars_by_date  = {}
ticks_by_date = {}
for date_str in sorted(events["session_date"].unique()):
    ticks = _load_rth_ticks(date_str)
    bars_by_date[date_str]  = build_range_bars(ticks, range_ticks=40)
    ticks_by_date[date_str] = ticks

from nqbt.analysis.feature_vector_builder import build_feature_vectors
fv = build_feature_vectors(events, bars_by_date, ticks_by_date)
fv.to_csv(OUT_DIR / "feature_vectors.csv", index=False)
print(f"  -> {len(fv)} feature vectors saved")

# Step 3: trailing simulation
print("Step 3: Running trailing simulations...")
from nqbt.analysis.trailing_simulator import simulate_all_events
trail_results = simulate_all_events(events, bars_by_date)
trail_results.to_csv(OUT_DIR / "trailing_simulation_results.csv", index=False)
print(f"  -> {len(trail_results)} simulation rows saved")

print(f"\nAll outputs saved to {OUT_DIR}")

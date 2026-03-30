"""
Reaction scanner debug output.

Loads reaction_events.csv, picks the first 5 trading dates, takes up to
10 events per date (50 total), rebuilds range bars for those sessions, and
produces two debug CSVs for manual chart verification in NinjaTrader:

  output/debug/reaction_debug_sample.csv       -- one row per event
  output/debug/reaction_debug_forward_bars.csv -- forward bar replay per event

Also prints a human-readable summary of the first 5 events to the terminal.

Uses a reference stop of 10 pts adverse to determine exit_reason and bars_held.
(Consistent with the loss bucket threshold: mfe_raw < 10 = loss.)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from nqbt.analysis.reaction_scanner import _load_rth_ticks
from nqbt.analysis.range_bars import build_range_bars

ET              = "America/New_York"
STOP_PTS        = 10.0   # reference stop for exit_reason computation
MAX_DATES       = 5
MAX_PER_DATE    = 10

OUT_DIR   = PROJECT_ROOT / "output" / "diagnostics"
DEBUG_DIR = PROJECT_ROOT / "output" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_PATH    = OUT_DIR / "reaction_events.csv"
SAMPLE_PATH    = DEBUG_DIR / "reaction_debug_sample.csv"
FWD_BARS_PATH  = DEBUG_DIR / "reaction_debug_forward_bars.csv"

# ---------------------------------------------------------------------------
# Load events
# ---------------------------------------------------------------------------

print("Loading reaction events...")
events = pd.read_csv(EVENTS_PATH)
print(f"  Total events: {len(events)}")

# Pick first 5 dates
all_dates = sorted(events["session_date"].unique())[:MAX_DATES]
print(f"  Using first {len(all_dates)} dates: {all_dates}")

subset = pd.concat([
    events[events["session_date"] == d].head(MAX_PER_DATE)
    for d in all_dates
], ignore_index=True)
print(f"  Debug subset: {len(subset)} events")

# ---------------------------------------------------------------------------
# Rebuild bars for each date
# ---------------------------------------------------------------------------

bars_by_date: dict = {}
for d in all_dates:
    ticks = _load_rth_ticks(d)
    if not ticks.empty:
        bars_by_date[d] = build_range_bars(ticks, range_ticks=40)
    else:
        bars_by_date[d] = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_direction(loc: str, std2_dist_upper: float, std2_dist_lower: float) -> str:
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


def _nearest_band(bar_close: float, vwap: float,
                  std1_upper: float, std1_lower: float,
                  std2_upper: float, std2_lower: float) -> str:
    bands = {
        "VWAP":       vwap,
        "std1_upper": std1_upper,
        "std1_lower": std1_lower,
        "std2_upper": std2_upper,
        "std2_lower": std2_lower,
    }
    nearest = min(bands, key=lambda k: abs(bar_close - bands[k]))
    return nearest


def _outcome_bucket(mfe_raw: float, mae_raw: float) -> str:
    if pd.isna(mfe_raw) or pd.isna(mae_raw):
        return "unknown"
    if mfe_raw < 10.0 or mae_raw > mfe_raw:
        return "loss"
    if mfe_raw < 25.0:
        return "scalp"
    if mfe_raw < 50.0:
        return "medium"
    return "large"


def _simulate_exit(forward_bars: list, entry: float, direction: str, stop_pts: float):
    """
    Walk forward bars and return (exit_price, exit_time_utc, bars_held, exit_reason).
    Uses a reference fixed stop of stop_pts; no take-profit target.
    """
    if not forward_bars:
        return entry, None, 0, "session_close"

    if direction == "short":
        stop = entry + stop_pts
        for i, bar in enumerate(forward_bars):
            if bar.high >= stop:
                return stop, bar.close_time, i + 1, "stop_hit"
        last = forward_bars[-1]
        return last.close, last.close_time, len(forward_bars), "session_close"
    else:  # long
        stop = entry - stop_pts
        for i, bar in enumerate(forward_bars):
            if bar.low <= stop:
                return stop, bar.close_time, i + 1, "stop_hit"
        last = forward_bars[-1]
        return last.close, last.close_time, len(forward_bars), "session_close"


def _ts_to_et(ts) -> str:
    if ts is None:
        return ""
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.tz_convert(ET).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)

# ---------------------------------------------------------------------------
# Build debug rows
# ---------------------------------------------------------------------------

sample_rows   = []
fwd_bars_rows = []

for _, ev in subset.iterrows():
    d        = ev["session_date"]
    bar_idx  = int(ev["bar_idx"])
    bars     = bars_by_date.get(d, [])

    if bar_idx >= len(bars):
        continue

    event_bar     = bars[bar_idx]
    forward_bars  = bars[bar_idx + 1:]

    entry_price   = float(ev["bar_close"])
    direction     = _assign_direction(
        str(ev["price_location"]),
        float(ev.get("std2_dist_upper", float("nan"))),
        float(ev.get("std2_dist_lower", float("nan"))),
    )

    # mfe_raw / mae_raw
    if direction == "short":
        mfe_raw = float(ev["mfe_down"])
        mae_raw = float(ev["mfe_up"])
    elif direction == "long":
        mfe_raw = float(ev["mfe_up"])
        mae_raw = float(ev["mfe_down"])
    else:
        mfe_raw = float("nan")
        mae_raw = float("nan")

    outcome = _outcome_bucket(mfe_raw, mae_raw)

    # Final points = net_move signed by direction
    net_move = float(ev["net_move"])
    final_pts = -net_move if direction == "short" else net_move

    # Exit simulation (10-pt reference stop)
    exit_price, exit_time_utc, bars_held, exit_reason = _simulate_exit(
        forward_bars, entry_price, direction, STOP_PTS
    )

    # Nearest VWAP band
    nearest_band = _nearest_band(
        entry_price,
        float(ev.get("vwap", float("nan"))),
        float(ev.get("std1_upper", float("nan"))),
        float(ev.get("std1_lower", float("nan"))),
        float(ev.get("std2_upper", float("nan"))),
        float(ev.get("std2_lower", float("nan"))),
    )

    bar_time_et  = _ts_to_et(ev["event_time"])
    exit_time_et = _ts_to_et(exit_time_utc)

    sample_rows.append({
        "date":              d,
        "bar_time_ET":       bar_time_et,
        "bar_index":         bar_idx,
        "location_type":     ev["price_location"],
        "nearest_vwap_band": nearest_band,
        "vwap_at_bar":       round(float(ev.get("vwap", float("nan"))), 2),
        "vah":               round(float(ev.get("vah", float("nan"))), 2),
        "val":               round(float(ev.get("val", float("nan"))), 2),
        "poc":               round(float(ev.get("poc", float("nan"))), 2),
        "entry_price":       round(entry_price, 2),
        "direction":         direction,
        "event_type":        ev["price_location"],
        "outcome_bucket":    outcome,
        "mfe_raw":           round(mfe_raw, 2) if not pd.isna(mfe_raw) else float("nan"),
        "mae_raw":           round(mae_raw, 2) if not pd.isna(mae_raw) else float("nan"),
        "final_points":      round(final_pts, 2),
        "bars_held":         bars_held,
        "exit_time_ET":      exit_time_et,
        "exit_reason":       exit_reason,
    })

    # Forward bar replay
    running_mfe = 0.0
    running_mae = 0.0
    for fwd_i, fbar in enumerate(forward_bars):
        if direction == "short":
            cum_pts    = entry_price - fbar.close
            mfe_so_far = entry_price - min(
                b.low for b in forward_bars[:fwd_i + 1]
            )
            mae_so_far = max(b.high for b in forward_bars[:fwd_i + 1]) - entry_price
        elif direction == "long":
            cum_pts    = fbar.close - entry_price
            mfe_so_far = max(b.high for b in forward_bars[:fwd_i + 1]) - entry_price
            mae_so_far = entry_price - min(b.low  for b in forward_bars[:fwd_i + 1])
        else:
            cum_pts = mfe_so_far = mae_so_far = float("nan")

        fwd_bars_rows.append({
            "date":                   d,
            "event_bar_time_ET":      bar_time_et,
            "forward_bar_number":     fwd_i + 1,
            "forward_bar_time_ET":    _ts_to_et(fbar.close_time),
            "bar_open":               round(fbar.open,  2),
            "bar_high":               round(fbar.high,  2),
            "bar_low":                round(fbar.low,   2),
            "bar_close":              round(fbar.close, 2),
            "cumulative_points_so_far": round(cum_pts,    2) if not pd.isna(cum_pts)    else float("nan"),
            "mfe_so_far":               round(mfe_so_far, 2) if not pd.isna(mfe_so_far) else float("nan"),
            "mae_so_far":               round(mae_so_far, 2) if not pd.isna(mae_so_far) else float("nan"),
        })

# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------

sample_df   = pd.DataFrame(sample_rows)
fwd_bars_df = pd.DataFrame(fwd_bars_rows)

sample_df.to_csv(SAMPLE_PATH, index=False)
fwd_bars_df.to_csv(FWD_BARS_PATH, index=False)

print(f"\nSaved {len(sample_df)} events  -> {SAMPLE_PATH}")
print(f"Saved {len(fwd_bars_df)} forward-bar rows -> {FWD_BARS_PATH}")

# ---------------------------------------------------------------------------
# Terminal summary: first 5 events
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 5 EVENT SUMMARY (verify against NinjaTrader 40-range chart)")
print("=" * 70)

for i, row in sample_df.head(5).iterrows():
    print(f"\nEvent {i+1}:")
    print(f"  Date / bar time : {row['date']}  {row['bar_time_ET']}")
    print(f"  Location        : {row['location_type']}  (nearest band: {row['nearest_vwap_band']})")
    print(f"  VAH={row['vah']}  VAL={row['val']}  VWAP={row['vwap_at_bar']}")
    print(f"  Direction       : {row['direction']}")
    print(f"  Entry price     : {row['entry_price']}")
    print(f"  Exit price      : {row['exit_time_ET']}  ({row['exit_reason']})")
    print(f"  Points captured : {row['final_points']:.2f} pts  (bars held: {row['bars_held']})")
    print(f"  MFE={row['mfe_raw']} pts  MAE={row['mae_raw']} pts  -> outcome: {row['outcome_bucket']}")

"""
Preview the developing volume profile and 40-range bars for a session window.
Adjust SESSION_DATE and the time window as needed.
Results are written to output/profile_and_bars.txt for easy reading.
"""

import sys
from pathlib import Path
import pandas as pd
from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize
from nqbt.analysis.volume_profile import VolumeProfile
from nqbt.analysis.range_bars import build_range_bars, bars_to_df
from nqbt.analysis.vwap import compute_vwap
from nqbt.analysis.absorption import score_bars

# --- Config ---
SESSION_DATE = "2026-03-11"
WINDOW_START = "2026-03-11 06:00:00"   # ET
WINDOW_END   = "2026-03-11 11:00:00"   # ET
OUTPUT_FILE  = Path("output/profile_and_bars.txt")
# --------------

OUTPUT_FILE.parent.mkdir(exist_ok=True)

fetch_start = str(pd.Timestamp(SESSION_DATE) - pd.Timedelta(days=1))[:10]
fetch_end   = str(pd.Timestamp(SESSION_DATE) + pd.Timedelta(days=1))[:10]

print("Loading data...")
df_raw = fetch_and_load(fetch_start, fetch_end)
ticks  = normalize(df_raw)

session_open = pd.Timestamp(f"{str(pd.Timestamp(SESSION_DATE) - pd.Timedelta(days=1))[:10]} 18:00:00", tz="America/New_York")
window_end   = pd.Timestamp(WINDOW_END, tz="America/New_York")
window_start = pd.Timestamp(WINDOW_START, tz="America/New_York")

profile_ticks = ticks[(ticks.index >= session_open) & (ticks.index <= window_end)]
bar_ticks     = ticks[(ticks.index >= window_start) & (ticks.index <= window_end)]

profile = VolumeProfile.build(profile_ticks, ticks_per_level=10, value_area_pct=0.70)
bars    = build_range_bars(bar_ticks, range_ticks=40, ticks_per_level=5)
vwap_df    = compute_vwap(profile_ticks)
absorption = score_bars(
    bars,
    use_percentile=True,  lookback=500,          percentile=80.0,
    use_ratio=True,       min_ratio=0.40,         min_volume_floor=50,
    use_activity=False,
    use_zscore=False,
    use_atr=False,
    use_ewma=False,
    combine_mode="all",
)

lines = []

# --- Volume Profile ---
# --- VWAP snapshot at window end ---
vwap_snap = vwap_df.iloc[-1]
lines.append("=" * 60)
lines.append(f"VWAP SNAPSHOT — as of {WINDOW_END} ET")
lines.append("=" * 60)
lines.append(f"  VWAP        : {vwap_snap['vwap']:.2f}")
lines.append(f"  Std Dev     : {vwap_snap['std']:.2f}")
lines.append(f"  +1 Std      : {vwap_snap['std1_upper']:.2f}")
lines.append(f"  -1 Std      : {vwap_snap['std1_lower']:.2f}")
lines.append(f"  +2 Std      : {vwap_snap['std2_upper']:.2f}")
lines.append(f"  -2 Std      : {vwap_snap['std2_lower']:.2f}")
lines.append(f"  +3 Std      : {vwap_snap['std3_upper']:.2f}")
lines.append(f"  -3 Std      : {vwap_snap['std3_lower']:.2f}")
lines.append("")

lines.append("=" * 60)
lines.append(f"DEVELOPING VOLUME PROFILE — 6pm ET prev day to {WINDOW_END} ET")
lines.append("=" * 60)
lines.append(profile.summary())
lines.append("")

df_profile = profile.to_df()
df_profile.insert(0, "label", "")
df_profile.loc[profile.poc, "label"] = "<-- POC"
level_size = profile.ticks_per_level * 0.25
vah_level = round(profile.vah - level_size, 4)
if vah_level in df_profile.index:
    df_profile.loc[vah_level, "label"] += " <-- VAH"
if profile.val in df_profile.index:
    df_profile.loc[profile.val, "label"] += " <-- VAL"
lines.append(df_profile.to_string())

# --- 40-Range Bars ---
lines.append("")
lines.append("=" * 60)
lines.append(f"40-RANGE VOLUMETRIC BARS — {WINDOW_START} to {WINDOW_END} ET")
lines.append(f"Total bars: {len(bars)}")
lines.append("=" * 60)

for i, bar in enumerate(bars, 1):
    direction = "BULL" if bar.close >= bar.open else "BEAR"
    open_time = bar.open_time.tz_convert("America/New_York").strftime("%I:%M:%S %p ET")
    lines.append("")
    lines.append(f"Bar {i:>2} [{direction}]  {open_time}")
    lines.append(f"         O:{bar.open:.2f}  H:{bar.high:.2f}  L:{bar.low:.2f}  C:{bar.close:.2f}")
    lines.append(f"         Buy:{bar.buy_vol}  Sell:{bar.sell_vol}  Delta:{bar.delta:+}  Total:{bar.total_vol}")
    lines.append(f"         {'Price':>10}  {'Ask(buy)':>9}  {'Bid(sell)':>9}  {'Delta':>7}")
    lines.append(f"         {'-' * 42}")
    for lv in sorted(bar.levels.values(), key=lambda l: l.price, reverse=True):
        lines.append(f"         {lv.price:>10.2f}  {lv.buy_vol:>9}  {lv.sell_vol:>9}  {lv.delta:>+7}")

# --- Absorption Signals ---
lines.append("")
lines.append("=" * 60)
lines.append(f"ABSORPTION SIGNALS — {WINDOW_START} to {WINDOW_END} ET")
lines.append(f"  lookback=500 bars  percentile=80  min_ratio=0.40  vol_floor=50  require_both=True")
lines.append("=" * 60)

combined = absorption[absorption["signal_combined"]]
if combined.empty:
    lines.append("  No combined signals with current parameters.")
else:
    lines.append(f"  Combined signals: {len(combined)}")
    lines.append("")
    lines.append(f"  {'Time (ET)':<18}  {'Open':>9}  {'Close':>9}  {'Dir':<5}  {'Imbalance':>10}  {'Ratio':>7}  {'TotalVol':>9}  {'Pct':>6}  {'Side'}")
    lines.append(f"  {'-' * 100}")
    for ts, row in combined.iterrows():
        time_str = ts.tz_convert("America/New_York").strftime("%I:%M:%S %p ET")
        pct_str  = f"{row['percentile_rank']:.1f}" if row['percentile_rank'] is not None else "  n/a"
        lines.append(
            f"  {time_str:<18}  {row['price_open']:>9.2f}  {row['price_close']:>9.2f}  "
            f"{row['bar_direction'].upper():<5}  {row['raw_imbalance']:>+10}  "
            f"{row['imbalance_ratio']:>+7.3f}  {row['total_vol']:>9}  "
            f"{pct_str:>6}  {row['absorption_side']}"
        )

lines.append("")
lines.append("ALL BARS — imbalance scores")
lines.append(f"  {'Time (ET)':<18}  {'Dir':<5}  {'Imbalance':>10}  {'Ratio':>7}  {'TotalVol':>9}  {'Pct':>6}  {'SigPct':<8}  {'SigRat':<8}  {'Combined'}")
lines.append(f"  {'-' * 105}")
for ts, row in absorption.iterrows():
    time_str = ts.tz_convert("America/New_York").strftime("%I:%M:%S %p ET")
    pct_str  = f"{row['percentile_rank']:.1f}" if row['percentile_rank'] is not None else "  n/a"
    lines.append(
        f"  {time_str:<18}  {row['bar_direction'].upper():<5}  {row['raw_imbalance']:>+10}  "
        f"{row['imbalance_ratio']:>+7.3f}  {row['total_vol']:>9}  "
        f"{pct_str:>6}  {'YES' if row['signal_percentile'] else 'no':<8}  "
        f"{'YES' if row['signal_ratio'] else 'no':<8}  "
        f"{'*** ' + str(row['absorption_side']) if row['signal_combined'] else ''}"
    )

output = "\n".join(lines)
OUTPUT_FILE.write_text(output, encoding="utf-8")
print(f"Done. Results written to {OUTPUT_FILE}")

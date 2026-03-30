"""
Audit post_absorption_aggression_3bar logic.
1. Print exact conditions checked for each signal side.
2. Verify delta direction semantics against absorption.py definitions.
3. Show 5 example signals at vwap_band_tag=+2s with post_agg=True, printing
   the signal row and the 3 bars after it from the raw tick cache.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gc
from datetime import timedelta

import numpy as np
import pandas as pd

from nqbt.analysis.range_bars import build_range_bars
from nqbt.analysis.absorption import score_bars

TICK_CACHE = PROJECT_ROOT / "output" / "tick_cache"
CSV_PATH   = PROJECT_ROOT / "output" / "absorption_signals_labeled.csv"
ET         = "America/New_York"

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: EXACT CONDITIONS IN signal_labeler.py")
print("=" * 70)
print()
print("From absorption.py CLASSIFICATION RULES:")
print()
print("  buy_absorbed  = buy-dominated imbalance (buy_vol > sell_vol)")
print("                  AND bar closed BEARISH (close < open)")
print("                  Buyers were aggressive; sellers held ground.")
print("                  => Bearish signal: expect price to FALL")
print()
print("  sell_absorbed = sell-dominated imbalance (sell_vol > buy_vol)")
print("                  AND bar closed BULLISH (close > open)")
print("                  Sellers were aggressive; buyers held ground.")
print("                  => Bullish signal: expect price to RISE")
print()
print("-" * 70)
print("CURRENT post_absorption_aggression_3bar CONDITIONS (signal_labeler.py):")
print()
print("  if side == 'sell_absorbed':   # bullish reversal expected UP")
print("      # confirmation: subsequent bars should show buyers winning")
print("      agg_bars_idx = [i+1 for i, b in enumerate(post_bars)")
print("                      if b.buy_vol > b.sell_vol]")
print()
print("  else (side == 'buy_absorbed'):  # bearish reversal expected DOWN")
print("      # confirmation: subsequent bars should show sellers winning")
print("      agg_bars_idx = [i+1 for i, b in enumerate(post_bars)")
print("                      if b.sell_vol > b.buy_vol]")
print()
print("-" * 70)
print("VERDICT:")
print()
print("  The labeler logic is CORRECT vs absorption.py definitions.")
print()
print("  User note: 'sell_absorbed (passive sellers absorbed buyers)' is")
print("  NOT how the code defines sell_absorbed. In absorption.py:")
print("    sell_absorbed = SELLERS were aggressive, BUYERS absorbed them")
print("    => bar closed UP, bullish, expect continuation UP")
print()
print("  What the user described ('passive sellers absorbed aggressive buyers")
print("  => price falls') maps to BUY_ABSORBED in the code, not sell_absorbed.")
print()

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 2: SIGNAL SIDE DISTRIBUTION AT +2s BAND")
print("=" * 70)
print()

df = pd.read_csv(CSV_PATH)
df["sustained"] = df["sustained"].astype(bool)
df["post_absorption_aggression_3bar"] = df["post_absorption_aggression_3bar"].astype(bool)

at2s = df[df["vwap_band_tag"] == "+2s"]

print(f"  Total +2s signals: {len(at2s)}")
print()
print("  absorption_side breakdown at +2s:")
for side, cnt in at2s["absorption_side"].value_counts().items():
    sub = at2s[at2s["absorption_side"] == side]
    agg_t = sub["post_absorption_aggression_3bar"].sum()
    base = ((sub["move_z_score"] >= 1.5) & (sub["mae_normalized"] <= 0.30)).mean() * 100
    print(f"    {side:<16}  n={cnt:>4}  post_agg=True: {agg_t:>3}  base_sustained: {base:.1f}%")
print()

print("  post_agg=True at +2s — further breakdown:")
at2s_agg = at2s[at2s["post_absorption_aggression_3bar"] == True]
print(f"    n={len(at2s_agg)}")
for side, cnt in at2s_agg["absorption_side"].value_counts().items():
    sub = at2s_agg[at2s_agg["absorption_side"] == side]
    base = ((sub["move_z_score"] >= 1.5) & (sub["mae_normalized"] <= 0.30)).mean() * 100
    print(f"    {side:<16}  n={cnt:>4}  base_sustained: {base:.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 3: 5 EXAMPLE SIGNALS  (vwap_band_tag=+2s, post_agg=True)")
print("=" * 70)
print()

examples = at2s_agg.head(5)

for rank, (_, row) in enumerate(examples.iterrows(), 1):
    trading_date_str = str(row["date"])
    trading_date     = pd.Timestamp(trading_date_str).date()
    signal_time      = pd.Timestamp(row["signal_time"], tz="UTC")
    signal_time_et   = signal_time.tz_convert(ET)
    side             = row["absorption_side"]

    print(f"  EXAMPLE {rank} --------------------------------------------------")
    print(f"    date:           {trading_date_str}")
    print(f"    signal_time ET: {signal_time_et.strftime('%H:%M:%S')}")
    print(f"    side:           {side}")
    print(f"    signal_price:   {row['signal_price']:.2f}")
    print(f"    price_location: {row['price_location']}")
    print(f"    vwap_at_signal: {row['vwap_at_signal']:.2f}  (dist_ticks={row['vwap_dist_ticks']:.1f})")
    print(f"    move_z_score:   {row['move_z_score']:.3f}  sustained={row['sustained']}")
    print(f"    post_agg_bar:   {row['post_aggression_bar']}")
    print()

    # Load tick cache for this date
    cache_path = TICK_CACHE / f"{trading_date}_ticks.parquet"
    if not cache_path.exists():
        print(f"    [tick cache not found: {cache_path}]")
        print()
        continue

    ticks = pd.read_parquet(cache_path)
    prev_day  = trading_date - timedelta(days=1)
    rth_start = pd.Timestamp(f"{trading_date} 09:30:00", tz=ET)
    rth_end   = pd.Timestamp(f"{trading_date} 11:00:00", tz=ET)
    rth_ticks = ticks[(ticks.index >= rth_start) & (ticks.index <= rth_end)]

    bars = build_range_bars(rth_ticks, range_ticks=40, ticks_per_level=5)
    if not bars:
        print(f"    [no bars built]")
        print()
        continue

    # Find signal bar by matching open_time
    sig_idx = None
    for i, b in enumerate(bars):
        b_ts = pd.Timestamp(b.open_time)
        if b_ts.tzinfo is None:
            b_ts = b_ts.tz_localize("UTC")
        b_ts_et = b_ts.tz_convert(ET)
        if abs((b_ts - signal_time).total_seconds()) < 60:
            sig_idx = i
            break

    if sig_idx is None:
        print(f"    [could not match signal bar to bar list]")
        print()
        continue

    # Print signal bar + next 3
    print(f"    {'bar':<6}  {'open_time_ET':<12}  {'open':>8}  {'close':>8}  {'buy_vol':>8}  {'sell_vol':>8}  {'delta':>8}  {'total_vol':>10}  note")
    print(f"    {'-'*100}")

    for offset in range(4):
        bi = sig_idx + offset
        if bi >= len(bars):
            break
        b = bars[bi]
        b_ts = pd.Timestamp(b.open_time)
        if b_ts.tzinfo is None:
            b_ts = b_ts.tz_localize("UTC")
        b_ts_et = b_ts.tz_convert(ET)
        delta = b.buy_vol - b.sell_vol
        note = ""
        if offset == 0:
            note = f"<-- SIGNAL BAR ({side})"
        else:
            # What does the condition check?
            if side == "sell_absorbed":
                meets = b.buy_vol > b.sell_vol
                note = f"buy_vol>sell_vol? {'YES' if meets else 'no'}"
            else:  # buy_absorbed
                meets = b.sell_vol > b.buy_vol
                note = f"sell_vol>buy_vol? {'YES' if meets else 'no'}"
        print(
            f"    {'sig' if offset==0 else f'+{offset}':<6}  "
            f"{b_ts_et.strftime('%H:%M:%S'):<12}  "
            f"{b.open:>8.2f}  {b.close:>8.2f}  "
            f"{b.buy_vol:>8}  {b.sell_vol:>8}  "
            f"{delta:>+8}  {b.total_vol:>10}  {note}"
        )

    # Confirm post_aggression_bar
    post_bars = bars[sig_idx + 1 : sig_idx + 4]
    if side == "sell_absorbed":
        agg_bars = [i+1 for i, b in enumerate(post_bars) if b.buy_vol > b.sell_vol]
    else:
        agg_bars = [i+1 for i, b in enumerate(post_bars) if b.sell_vol > b.buy_vol]
    majority = len(post_bars) > 0 and len(agg_bars) > len(post_bars) / 2
    print()
    print(f"    Confirmed bars (meeting delta condition): {agg_bars}")
    print(f"    Majority? {majority}  ({len(agg_bars)}/{len(post_bars)} bars)  post_aggression_bar={agg_bars[0] if agg_bars else 0}")
    print()
    del ticks, rth_ticks, bars
    gc.collect()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("  The post_absorption_aggression_3bar condition is checking DELTA DIRECTION")
print("  in the post-signal bars, not volume magnitude.")
print()
print("  Note from cross-tab: post_agg=False OUTPERFORMS post_agg=True at +2s band")
print("  (18.8% vs 9.6% base sustained). Likely because at an extended VWAP band,")
print("  immediate confirmation via delta means the move is a breakout continuation,")
print("  NOT a failed test that reverses. The best signals at +2s are ones where")
print("  the test FAILS immediately (sellers overwhelm quickly at +2s band = failed")
print("  breakout that snaps back).")
print()
print("  The 'high total_vol' condition the user mentioned is NOT currently included")
print("  in post_agg_3bar — only delta direction (buy_vol > sell_vol or vice versa)")
print("  is checked, not volume magnitude.")

"""
Post-target continuation analysis for PT10+SL25 winning trades.

For the 506 trades that hit the PT10 target, measures what happened
from the target hit point onward to session close.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from nqbt.analysis.reaction_scanner import _load_rth_ticks
from nqbt.analysis.range_bars import build_range_bars
from nqbt.analysis.vwap import compute_vwap

FV_PATH = PROJECT_ROOT / "output" / "diagnostics" / "feature_vectors.csv"
EV_PATH = PROJECT_ROOT / "output" / "diagnostics" / "reaction_events.csv"
ET           = "America/New_York"
TARGET_PTS   = 10.0
STOP_PTS     = 25.0
COOLDOWN     = 3
REACH_LEVELS = [20, 30, 40, 50]

fv     = pd.read_csv(FV_PATH)
ev_raw = pd.read_csv(EV_PATH, usecols=["session_date","event_time","bar_close"])
fv     = fv.merge(ev_raw, on=["session_date","event_time"], how="left")

def _dir(loc):
    if loc in ("outside_vah","at_vah"): return "short"
    if loc in ("outside_val","at_val"): return "long"
    return "unknown"

fv["_dir"]  = fv["price_location"].apply(_dir)
fv["_et"]   = pd.to_datetime(fv["event_time"], utc=True).dt.tz_convert(ET)
fv["_hhmm"] = fv["_et"].dt.hour*60 + fv["_et"].dt.minute

ev = fv[
    (fv["_hhmm"] >= 600) & (fv["_hhmm"] < 645) &
    (fv["vwap_dist_pts"].abs() <= 50) &
    (fv["_dir"] != "unknown")
].copy()

TRADING_DAYS = fv["session_date"].nunique()
ALL_DATES    = sorted(fv["session_date"].unique())

# ── Step 1: re-run simulation, collect target hits ─────────────────────────────

print("Running PT10+SL25 simulation, collecting target hits...")

wins = []    # dicts for each trade that hit target

sessions = sorted(ev["session_date"].unique())
for si, sd in enumerate(sessions):
    if si % 25 == 0: print(f"  Session {si+1}/{len(sessions)}: {sd}")

    ticks = _load_rth_ticks(sd)
    if ticks.empty: continue
    bars = build_range_bars(ticks, range_ticks=40)
    if not bars: continue

    # pre-compute VWAP series for the session (indexed by timestamp)
    vwap_df = compute_vwap(ticks)   # same index as ticks

    sess_ev      = ev[ev["session_date"]==sd].sort_values("bar_idx").reset_index(drop=True)
    next_eligible = 0

    for _, row in sess_ev.iterrows():
        bidx  = int(row["bar_idx"])
        if bidx < next_eligible: continue

        direction = row["_dir"]
        entry     = float(row["bar_close"])
        tp        = entry - TARGET_PTS if direction == "short" else entry + TARGET_PTS
        sl        = entry + STOP_PTS   if direction == "short" else entry - STOP_PTS
        n         = len(bars)

        hit_target = False
        abs_exit   = n - 1
        for j in range(bidx+1, n):
            b = bars[j]
            if direction == "short":
                sh = b.high >= sl;  th = b.low  <= tp
            else:
                sh = b.low  <= sl;  th = b.high >= tp
            if sh and th: abs_exit = j; break
            if sh:         abs_exit = j; break
            if th:
                abs_exit   = j
                hit_target = True
                break

        if hit_target:
            wins.append({
                "session_date":   sd,
                "bar_idx":        bidx,
                "direction":      direction,
                "entry":          entry,
                "target_price":   tp,
                "target_bar_abs": abs_exit,
                "target_bar_obj": bars[abs_exit],
                "bars_list":      bars,
                "vwap_df":        vwap_df,
            })

        next_eligible = abs_exit + COOLDOWN + 1

print(f"  Target hits collected: {len(wins):,}")

# ── Step 2: compute post-target metrics ────────────────────────────────────────

print("Computing post-target continuation metrics...")

records = []
for w in wins:
    direction      = w["direction"]
    entry          = w["entry"]
    tp             = w["target_price"]
    bars           = w["bars_list"]
    target_bar_abs = w["target_bar_abs"]
    target_bar     = w["target_bar_obj"]
    vwap_df        = w["vwap_df"]
    n              = len(bars)

    # VWAP at the time the target was hit
    vwap_val = float("nan")
    if target_bar.close_time is not None:
        ct = target_bar.close_time
        if ct.tzinfo is None:
            ct = ct.tz_localize("UTC").tz_convert(ET)
        else:
            ct = ct.tz_convert(ET)
        vwap_slice = vwap_df[vwap_df.index <= ct]
        if not vwap_slice.empty:
            vwap_val = float(vwap_slice.iloc[-1]["vwap"])

    # Remaining bars from target_bar onward (inclusive — to capture full bar range)
    rem_bars = bars[target_bar_abs:]

    if not rem_bars:
        continue

    if direction == "short":
        # Continuation = further down from tp
        max_cont   = max((tp - b.low)  for b in rem_bars)
        reversal   = max((b.high - tp) for b in rem_bars)
        reach_n    = {lvl: any(b.low <= entry - lvl for b in rem_bars) for lvl in REACH_LEVELS}
        dist_vwap  = tp - vwap_val if not np.isnan(vwap_val) else float("nan")
        # positive = tp is above VWAP, more room to fall
        reach_vwap = any(b.low <= vwap_val for b in rem_bars) if not np.isnan(vwap_val) else float("nan")
    else:
        max_cont   = max((b.high - tp) for b in rem_bars)
        reversal   = max((tp - b.low)  for b in rem_bars)
        reach_n    = {lvl: any(b.high >= entry + lvl for b in rem_bars) for lvl in REACH_LEVELS}
        dist_vwap  = vwap_val - tp if not np.isnan(vwap_val) else float("nan")
        # positive = tp is below VWAP, more room to rise
        reach_vwap = any(b.high >= vwap_val for b in rem_bars) if not np.isnan(vwap_val) else float("nan")

    rec = {
        "session_date":    w["session_date"],
        "direction":       direction,
        "entry":           entry,
        "target_price":    tp,
        "max_continuation_pts": max_cont,
        "reversal_after_target": reversal,
        "vwap_at_target":  vwap_val,
        "dist_to_vwap":    dist_vwap,
        "reach_vwap":      reach_vwap,
    }
    for lvl in REACH_LEVELS:
        rec[f"did_reach_{lvl}pts"] = reach_n[lvl]
    records.append(rec)

df = pd.DataFrame(records)
n  = len(df)

print(f"  Analysed: {n:,} winning trades")

# ── Step 3: Print results ──────────────────────────────────────────────────────

print(f"\n{'='*64}")
print(f"POST-TARGET CONTINUATION ANALYSIS")
print(f"PT10+SL25  No-Early-Exit  (10:00-10:45 + vwap<=50)")
print(f"N = {n} winning trades that hit the +10pt target")
print(f"{'='*64}")

print(f"\n  CONTINUATION BEYOND +10 PT TARGET:")
print(f"  {'Level':<16}  {'Count':>6}  {'%':>7}")
print(f"  {'-'*16}  {'-'*6}  {'-'*7}")
for lvl in REACH_LEVELS:
    col   = f"did_reach_{lvl}pts"
    cnt   = df[col].sum()
    pct   = cnt / n * 100
    bar_s = "#" * int(pct / 3)
    print(f"  reached +{lvl:<7}   {cnt:>6,}  {pct:>6.1f}%  {bar_s}")

print(f"\n  Conditional reach rates (given +10 already hit):")
print(f"  P(reach +20 | hit +10)  = {df['did_reach_20pts'].mean()*100:.1f}%")
prev = df["did_reach_20pts"]
for cur_lvl, prev_lvl in [(30,20),(40,30),(50,40)]:
    cur_col  = f"did_reach_{cur_lvl}pts"
    prev_col = f"did_reach_{prev_lvl}pts"
    cond     = df[prev_col]
    rate     = df.loc[cond, cur_col].mean() * 100 if cond.sum() > 0 else float("nan")
    print(f"  P(reach +{cur_lvl} | hit +{prev_lvl})  = {rate:.1f}%")

print(f"\n  MAX CONTINUATION (pts beyond +10, in trade direction):")
print(f"  Mean   : {df['max_continuation_pts'].mean():>6.2f} pts")
print(f"  Median : {df['max_continuation_pts'].median():>6.2f} pts")
print(f"  p75    : {df['max_continuation_pts'].quantile(.75):>6.2f} pts")
print(f"  p90    : {df['max_continuation_pts'].quantile(.90):>6.2f} pts")
print(f"  p95    : {df['max_continuation_pts'].quantile(.95):>6.2f} pts")
print(f"  Max    : {df['max_continuation_pts'].max():>6.2f} pts")

print(f"\n  Distribution of max_continuation_pts:")
edges = [0,5,10,15,20,30,40,50,999]
labels = ["0-5","5-10","10-15","15-20","20-30","30-40","40-50","50+"]
for i,(lo,hi) in enumerate(zip(edges,edges[1:])):
    mask = (df["max_continuation_pts"] >= lo) & (df["max_continuation_pts"] < hi)
    cnt  = mask.sum()
    pct  = cnt/n*100
    bar_s = "#" * int(pct/2)
    print(f"    {labels[i]:<8}: {cnt:>5,}  ({pct:>5.1f}%)  {bar_s}")

print(f"\n  REVERSAL AFTER TARGET HIT (pts price moved AGAINST trade direction):")
print(f"  Mean   : {df['reversal_after_target'].mean():>6.2f} pts")
print(f"  Median : {df['reversal_after_target'].median():>6.2f} pts")
print(f"  p75    : {df['reversal_after_target'].quantile(.75):>6.2f} pts")
print(f"  p90    : {df['reversal_after_target'].quantile(.90):>6.2f} pts")
print(f"  Fraction reversed > 5 pts  : {(df['reversal_after_target']>5).mean()*100:.1f}%")
print(f"  Fraction reversed > 10 pts : {(df['reversal_after_target']>10).mean()*100:.1f}%")
print(f"  Fraction reversed > 20 pts : {(df['reversal_after_target']>20).mean()*100:.1f}%")

print(f"\n  NET CONTINUATION (max_continuation - reversal, positive = net gain):")
df["net_cont"] = df["max_continuation_pts"] - df["reversal_after_target"]
print(f"  Mean net     : {df['net_cont'].mean():>+6.2f} pts")
print(f"  % net > 0    : {(df['net_cont']>0).mean()*100:.1f}%  (continuation > reversal)")
print(f"  % net > 5    : {(df['net_cont']>5).mean()*100:.1f}%")
print(f"  % net > 10   : {(df['net_cont']>10).mean()*100:.1f}%")

# VWAP analysis
df_v = df[df["dist_to_vwap"].notna() & df["dist_to_vwap"].between(-200,200)]
print(f"\n  VWAP ANALYSIS AT TARGET HIT (n={len(df_v)}):")
print(f"  Mean dist to VWAP (in trade direction) : {df_v['dist_to_vwap'].mean():.2f} pts")
print(f"    (positive = more room in trade direction; negative = VWAP already passed)")
print(f"  Median dist to VWAP                    : {df_v['dist_to_vwap'].median():.2f} pts")
print(f"  % where target went PAST VWAP (dist<0) : {(df_v['dist_to_vwap']<0).mean()*100:.1f}%")
print(f"  % where VWAP within 5 pts of target    : {(df_v['dist_to_vwap'].abs()<5).mean()*100:.1f}%")
print(f"\n  Did reach VWAP after PT10 hit:")
dv2 = df_v[df_v["reach_vwap"].notna()]
if len(dv2):
    pct_vwap = dv2["reach_vwap"].mean()*100
    print(f"    Yes: {dv2['reach_vwap'].sum():.0f} / {len(dv2):,}  ({pct_vwap:.1f}%)")

    print(f"\n  Conditional: reach_vwap rate by distance bucket:")
    print(f"  {'dist_to_vwap':>18}  {'n':>5}  {'reach_vwap%':>12}")
    for lo,hi in [(-999,-0.01),(0,10),(10,20),(20,30),(30,50),(50,999)]:
        label = f"<0 (past)" if hi==-0.01 else (f"{lo}-{hi}" if hi!=999 else f"{lo}+")
        sub   = dv2[(dv2["dist_to_vwap"]>=lo) & (dv2["dist_to_vwap"]<hi)]
        if len(sub) == 0: continue
        r     = sub["reach_vwap"].mean()*100
        print(f"  {label:>18}  {len(sub):>5,}  {r:>11.1f}%")

print(f"\n  DIRECTION BREAKDOWN:")
for d in ["short","long"]:
    sub = df[df["direction"]==d]
    if len(sub) == 0: continue
    print(f"  {d.upper()} ({len(sub)} trades):")
    print(f"    reach +20: {sub['did_reach_20pts'].mean()*100:.1f}%   "
          f"reach +30: {sub['did_reach_30pts'].mean()*100:.1f}%   "
          f"reach +50: {sub['did_reach_50pts'].mean()*100:.1f}%")
    print(f"    mean cont: {sub['max_continuation_pts'].mean():.1f} pts   "
          f"mean reversal: {sub['reversal_after_target'].mean():.1f} pts")

print()

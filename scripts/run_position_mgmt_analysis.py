"""
Detailed analysis of PT10+SL25 no-early-exit with position management.
Reports: trades/day, skip reasons, bars held, exit breakdown.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from nqbt.analysis.reaction_scanner import _load_rth_ticks
from nqbt.analysis.range_bars import build_range_bars

FV_PATH = PROJECT_ROOT / "output" / "diagnostics" / "feature_vectors.csv"
EV_PATH = PROJECT_ROOT / "output" / "diagnostics" / "reaction_events.csv"
ET = "America/New_York"
TARGET_PTS = 10.0
STOP_PTS   = 25.0
COOLDOWN   = 3

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

def _sim(bars, sig_idx, direction, entry):
    tp = entry - TARGET_PTS if direction=="short" else entry + TARGET_PTS
    sl = entry + STOP_PTS   if direction=="short" else entry - STOP_PTS
    n  = len(bars)
    for j in range(sig_idx+1, n):
        b = bars[j]
        if direction == "short":
            sh = b.high >= sl;  th = b.low  <= tp
        else:
            sh = b.low  <= sl;  th = b.high >= tp
        if sh and th: return -STOP_PTS,   "stop",          j, j-(sig_idx+1)+1
        if sh:        return -STOP_PTS,   "stop",          j, j-(sig_idx+1)+1
        if th:        return  TARGET_PTS, "target",        j, j-(sig_idx+1)+1
    last = n-1
    cap  = (entry - bars[last].close) if direction=="short" else (bars[last].close - entry)
    return cap, "session_close", last, last-(sig_idx+1)+1 if last > sig_idx else 0

trades  = []   # taken trades
skipped = []   # skipped signals

sessions = sorted(ev["session_date"].unique())
for si, sd in enumerate(sessions):
    if si % 25 == 0: print(f"  Session {si+1}/{len(sessions)}: {sd}")
    ticks = _load_rth_ticks(sd)
    if ticks.empty: continue
    bars = build_range_bars(ticks, range_ticks=40)
    if not bars: continue

    sess_ev = ev[ev["session_date"]==sd].sort_values("bar_idx").reset_index(drop=True)

    # We need to track the "last trade exit bar" to distinguish position-open vs cooldown
    last_exit_bar  = -9999
    next_eligible  = 0

    for _, row in sess_ev.iterrows():
        bidx = int(row["bar_idx"])
        if bidx < next_eligible:
            # Was the trade still open, or in cooldown?
            if bidx <= last_exit_bar:
                reason = "position_open"
            else:
                reason = "cooldown"
            skipped.append({"session_date": sd, "bar_idx": bidx, "skip_reason": reason})
            continue

        direction = row["_dir"]
        entry     = float(row["bar_close"])
        cap, exit_r, abs_exit, bars_held = _sim(bars, bidx, direction, entry)

        trades.append({
            "session_date": sd,
            "bar_idx":      bidx,
            "direction":    direction,
            "captured":     cap,
            "exit_reason":  exit_r,
            "bars_held":    bars_held,
        })
        last_exit_bar = abs_exit
        next_eligible = abs_exit + COOLDOWN + 1

tr = pd.DataFrame(trades)
sk = pd.DataFrame(skipped) if skipped else pd.DataFrame(columns=["session_date","bar_idx","skip_reason"])

# ── Print analysis ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PT10+SL25  No-Early-Exit  Position-Managed  10:00-10:45 + vwap<=50")
print(f"{'='*60}")

n_trades   = len(tr)
n_skipped  = len(sk)
n_signals  = n_trades + n_skipped
tpd        = n_trades / TRADING_DAYS

print(f"\n  Total qualifying signals : {n_signals:,}")
print(f"  Trades taken             : {n_trades:,}  ({n_trades/n_signals*100:.1f}% of signals)")
print(f"  Signals skipped          : {n_skipped:,}  ({n_skipped/n_signals*100:.1f}% of signals)")
print(f"  Trading days             : {TRADING_DAYS}")
print(f"  Trades per day (avg)     : {tpd:.2f}")

# Sessions with events
sess_trades = tr.groupby("session_date").size().reindex(
    [d for d in ALL_DATES if d in ev["session_date"].values], fill_value=0
)
print(f"  Trades per session (avg) : {sess_trades.mean():.2f}")
print(f"  Sessions with >=1 trade  : {(sess_trades>0).sum()} / {len(sess_trades)}")

# Distribution: days with 0,1,2,3+ trades
daily_counts = tr.groupby("session_date").size().reindex(ALL_DATES, fill_value=0)
print(f"\n  Days by trade count:")
for k in [0,1,2,3]:
    label = f"{k}+" if k==3 else str(k)
    cnt   = (daily_counts >= k).sum() if k==3 else (daily_counts == k).sum()
    pct   = cnt/TRADING_DAYS*100
    bar   = "#"*int(pct/2)
    print(f"    {label} trades : {cnt:>4d} days  ({pct:5.1f}%)  {bar}")

# Skip reasons
print(f"\n  Skip breakdown ({n_skipped:,} total):")
if len(sk):
    vc = sk["skip_reason"].value_counts()
    for reason, cnt in vc.items():
        print(f"    {reason:<20}: {cnt:,}  ({cnt/n_skipped*100:.1f}%  "
              f"= {cnt/n_signals*100:.1f}% of all signals)")

# Bars held
print(f"\n  Bars held (signal bar to exit bar):")
print(f"    Mean   : {tr['bars_held'].mean():.2f}")
print(f"    Median : {tr['bars_held'].median():.0f}")
print(f"    p25    : {tr['bars_held'].quantile(.25):.0f}")
print(f"    p75    : {tr['bars_held'].quantile(.75):.0f}")
print(f"    p90    : {tr['bars_held'].quantile(.90):.0f}")
print(f"    Max    : {tr['bars_held'].max():.0f}")
print(f"\n  Distribution:")
for b_range, label in [(range(1,2),"1 bar"), (range(2,4),"2-3 bars"),
                       (range(4,7),"4-6 bars"), (range(7,16),"7-15 bars"),
                       (range(16,1000),"16+ bars")]:
    cnt = tr["bars_held"].between(b_range.start, b_range.stop-1).sum()
    print(f"    {label:<12}: {cnt:>5,}  ({cnt/n_trades*100:.1f}%)")

# Exit breakdown
print(f"\n  Exit breakdown:")
vc2 = tr["exit_reason"].value_counts()
for ex, cnt in vc2.items():
    mean_pts = tr[tr["exit_reason"]==ex]["captured"].mean()
    print(f"    {ex:<16}: {cnt:>5,}  ({cnt/n_trades*100:.1f}%)  "
          f"mean={mean_pts:>+7.2f} pts  ({mean_pts*20:>+7.0f})")

print(f"\n  Overall expectancy:")
print(f"    Raw : {tr['captured'].mean():>+.3f} pts")
print(f"    Adj : {tr['captured'].mean()-0.25:>+.3f} pts  (after $5 commission)")

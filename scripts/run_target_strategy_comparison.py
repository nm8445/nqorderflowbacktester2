"""
Target strategy comparison for PT10+SL25 baseline.

Strategies tested (10:00-10:45 + vwap_dist<=50, position management):
  Baseline  — PT10+SL25 (fixed target)
  Strategy1 — Scale-out: exit 50% at +10, trail 50% from breakeven
  Strategy2 — VWAP target: exit at session-VWAP, skip if VWAP < 10 pts away
  Strategy3 — Tiered: SHORT=PT20+SL25, LONG=PT10+SL25

Outputs:
  output/diagnostics/target_strategy_comparison.txt
  output/tearsheets/target_strategy_comparison.html
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import quantstats as qs

from nqbt.analysis.reaction_scanner import _load_rth_ticks
from nqbt.analysis.range_bars import build_range_bars

FV_PATH     = PROJECT_ROOT / "output" / "diagnostics" / "feature_vectors.csv"
EV_PATH     = PROJECT_ROOT / "output" / "diagnostics" / "reaction_events.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "diagnostics" / "target_strategy_comparison.txt"
TEAR_DIR    = PROJECT_ROOT / "output" / "tearsheets"
TEAR_DIR.mkdir(parents=True, exist_ok=True)

ET               = "America/New_York"
DOLLARS_PER_PT   = 20.0
COMMISSION_PTS   = 0.25
DAILY_STOP_USD   = -500.0
PAYOUT_DAYS      = 21
ACCOUNT_SIZE_USD = 50_000.0
COOLDOWN         = 3
TICK_BUFFER      = 0.50    # 2 ticks at 0.25 pts/tick

fv     = pd.read_csv(FV_PATH)
ev_raw = pd.read_csv(EV_PATH, usecols=["session_date","event_time","bar_close","vwap"])
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

print(f"Base events: {len(ev):,}  Sessions: {ev['session_date'].nunique()}  Trading days: {TRADING_DAYS}")

# ──────────────────────────────────────────────────────────────────────────────
# Simulation functions
# ──────────────────────────────────────────────────────────────────────────────

def _prices(entry, direction, target_pts, stop_pts):
    if direction == "short":
        return entry - target_pts, entry + stop_pts
    return entry + target_pts, entry - stop_pts

def _stopped(bar, direction, stop_price):
    return bar.high >= stop_price if direction=="short" else bar.low <= stop_price

def _targeted(bar, direction, target_price):
    return bar.low <= target_price if direction=="short" else bar.high >= target_price


def _sim_baseline(bars, bidx, direction, entry, target_pts=10.0, stop_pts=25.0):
    tp, sl = _prices(entry, direction, target_pts, stop_pts)
    n = len(bars)
    for j in range(bidx+1, n):
        b = bars[j]
        sh, th = _stopped(b,direction,sl), _targeted(b,direction,tp)
        if sh and th: return -stop_pts, "stop",          j
        if sh:        return -stop_pts, "stop",          j
        if th:        return  target_pts,"target",       j
    last = n-1
    cap = (entry - bars[last].close) if direction=="short" else (bars[last].close - entry)
    return cap, "session_close", last


def _sim_scale_out(bars, bidx, direction, entry, target_pts=10.0, stop_pts=25.0):
    """
    50% exits at target_pts, remaining 50% trails from breakeven.
    Returns (total_pnl, first_exit, second_exit, second_pnl, abs_exit_bar).
    """
    tp, sl = _prices(entry, direction, target_pts, stop_pts)
    n = len(bars)

    # Phase 1: look for first half target or stop
    first_bar = None
    for j in range(bidx+1, n):
        b = bars[j]
        sh, th = _stopped(b,direction,sl), _targeted(b,direction,tp)
        if sh and th:
            return -stop_pts, "stop", "na",           -stop_pts, j
        if sh:
            return -stop_pts, "stop", "na",           -stop_pts, j
        if th:
            first_bar = j
            break

    if first_bar is None:
        # Session close — both halves
        last = n-1
        cap  = (entry - bars[last].close) if direction=="short" else (bars[last].close - entry)
        return cap, "session_close", "session_close", cap, last

    # Phase 2: trail remaining 50% from breakeven
    trail_stop = entry

    # Update trail on the target bar itself (no stop check on this bar)
    b0 = bars[first_bar]
    b0_up = b0.close > b0.open
    closes_dir = (not b0_up) if direction=="short" else b0_up
    if closes_dir:
        prop = (b0.high + TICK_BUFFER) if direction=="short" else (b0.low - TICK_BUFFER)
        if direction=="short" and prop < trail_stop: trail_stop = prop
        if direction=="long"  and prop > trail_stop: trail_stop = prop

    for j in range(first_bar+1, n):
        b = bars[j]
        # Check trail stop hit
        if direction=="short":
            if b.high >= trail_stop:
                s2 = entry - trail_stop
                return 0.5*target_pts + 0.5*s2, "target", "trail_stop", s2, j
        else:
            if b.low <= trail_stop:
                s2 = trail_stop - entry
                return 0.5*target_pts + 0.5*s2, "target", "trail_stop", s2, j

        # Update trail if bar closes in trade direction
        b_up = b.close > b.open
        closes = (not b_up) if direction=="short" else b_up
        if closes:
            prop = (b.high + TICK_BUFFER) if direction=="short" else (b.low - TICK_BUFFER)
            if direction=="short" and prop < trail_stop: trail_stop = prop
            if direction=="long"  and prop > trail_stop: trail_stop = prop

    # Session close
    last = n-1
    s2   = (entry - bars[last].close) if direction=="short" else (bars[last].close - entry)
    return 0.5*target_pts + 0.5*s2, "target", "session_close", s2, last


def _sim_vwap_target(bars, bidx, direction, entry, vwap_val, stop_pts=25.0):
    """Returns (pnl, exit_reason, abs_exit_bar, skipped)."""
    dist = entry - vwap_val if direction=="short" else vwap_val - entry
    if dist < 10.0:
        return 0.0, "skipped", bidx, True
    tp  = vwap_val
    sl  = entry + stop_pts if direction=="short" else entry - stop_pts
    n   = len(bars)
    for j in range(bidx+1, n):
        b = bars[j]
        sh, th = _stopped(b,direction,sl), _targeted(b,direction,tp)
        if sh and th: return -stop_pts, "stop",          j, False
        if sh:        return -stop_pts, "stop",          j, False
        if th:
            cap = dist   # captured = vwap_dist_pts at signal
            return cap, "target", j, False
    last = n-1
    cap  = (entry - bars[last].close) if direction=="short" else (bars[last].close - entry)
    return cap, "session_close", last, False


def _sim_tiered(bars, bidx, direction, entry, stop_pts=25.0):
    """SHORT: PT20+SL25  LONG: PT10+SL25"""
    target_pts = 20.0 if direction=="short" else 10.0
    return _sim_baseline(bars, bidx, direction, entry, target_pts, stop_pts)


# ──────────────────────────────────────────────────────────────────────────────
# Session-level replay with position state
# ──────────────────────────────────────────────────────────────────────────────

def _run_all_strategies(ev_df):
    """Single pass over sessions; runs all 4 strategies concurrently."""
    baseline_trades = []
    s1_trades       = []
    s2_trades       = []
    s3_trades       = []

    sessions = sorted(ev_df["session_date"].unique())
    for si, sd in enumerate(sessions):
        if si % 25 == 0: print(f"  Session {si+1}/{len(sessions)}: {sd}")

        ticks = _load_rth_ticks(sd)
        if ticks.empty: continue
        bars = build_range_bars(ticks, range_ticks=40)
        if not bars: continue

        sess_ev = ev_df[ev_df["session_date"]==sd].sort_values("bar_idx").reset_index(drop=True)

        # Independent state per strategy
        next_el = {"base":0, "s1":0, "s2":0, "s3":0}

        for _, row in sess_ev.iterrows():
            bidx  = int(row["bar_idx"])
            dirn  = row["_dir"]
            entry = float(row["bar_close"])
            vwap  = float(row["vwap"])
            common = {"session_date":sd, "event_time":row["event_time"], "direction":dirn}

            # ── Baseline ──────────────────────────────────────────────────
            if bidx >= next_el["base"]:
                pnl, reason, aex = _sim_baseline(bars, bidx, dirn, entry)
                baseline_trades.append({**common,"pnl":pnl,"exit_reason":reason})
                next_el["base"] = aex + COOLDOWN + 1

            # ── Strategy 1 (scale-out) ────────────────────────────────────
            if bidx >= next_el["s1"]:
                total, f_ex, s_ex, s2_pnl, aex = _sim_scale_out(bars, bidx, dirn, entry)
                s1_trades.append({
                    **common, "pnl":total,
                    "first_exit":f_ex, "second_exit":s_ex,
                    "second_pnl":s2_pnl,
                    "first_hit": f_ex in ("target",),
                    # what PT10-only would have captured on this same trade
                    "pt10_equiv": 10.0 if f_ex=="target" else (-25.0 if f_ex=="stop" else s2_pnl),
                })
                next_el["s1"] = aex + COOLDOWN + 1

            # ── Strategy 2 (VWAP target) ───────────────────────────────────
            if bidx >= next_el["s2"]:
                pnl, reason, aex, skipped = _sim_vwap_target(bars, bidx, dirn, entry, vwap)
                if not skipped:
                    s2_trades.append({**common,"pnl":pnl,"exit_reason":reason,
                                      "vwap_dist": abs(entry - vwap)})
                    next_el["s2"] = aex + COOLDOWN + 1
                # skipped trades: do NOT advance next_el (signal ignored, next signal eligible)

            # ── Strategy 3 (tiered) ────────────────────────────────────────
            if bidx >= next_el["s3"]:
                pnl, reason, aex = _sim_tiered(bars, bidx, dirn, entry)
                s3_trades.append({**common,"pnl":pnl,"exit_reason":reason})
                next_el["s3"] = aex + COOLDOWN + 1

    return (pd.DataFrame(baseline_trades), pd.DataFrame(s1_trades),
            pd.DataFrame(s2_trades),       pd.DataFrame(s3_trades))


print("\nRunning all strategies...")
df_base, df_s1, df_s2, df_s3 = _run_all_strategies(ev)
print(f"  Baseline: {len(df_base):,}  S1: {len(df_s1):,}  S2: {len(df_s2):,}  S3: {len(df_s3):,}")

# ──────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _max_dd(d):
    c = d.cumsum(); return float((c - c.cummax()).min())

def _sharpe(d):
    u = d * DOLLARS_PER_PT
    return float(u.mean() / u.std() * np.sqrt(252)) if u.std() > 0 else float("nan")

def _payout_cycles(d):
    cycles = []
    for i in range(0, len(ALL_DATES)-PAYOUT_DAYS+1, PAYOUT_DAYS):
        cycles.append(float(d.iloc[i:i+PAYOUT_DAYS].sum()) > 0)
    return cycles

def _metrics(df, pnl_col="pnl", label=""):
    n   = len(df)
    pts = df[pnl_col]
    tgt = df[df["exit_reason"]=="target"]  if "exit_reason" in df else pd.DataFrame()
    stp = df[df["exit_reason"]=="stop"]    if "exit_reason" in df else pd.DataFrame()

    wr      = len(tgt)/n if n>0 else 0
    raw_exp = float(pts.mean()) if n>0 else 0
    adj_exp = raw_exp - COMMISSION_PTS
    tpd     = n / TRADING_DAYS

    daily     = df.groupby("session_date")[pnl_col].sum().reindex(ALL_DATES, fill_value=0.0)
    daily_usd = daily * DOLLARS_PER_PT
    cycles    = _payout_cycles(daily)

    return {
        "label":       label,
        "n":           n,
        "tpd":         tpd,
        "wr":          wr,
        "n_target":    len(tgt),
        "n_stop":      len(stp),
        "n_close":     len(df[df["exit_reason"]=="session_close"]) if "exit_reason" in df else 0,
        "mean_win":    float(tgt[pnl_col].mean()) if len(tgt) else float("nan"),
        "mean_loss":   float(stp[pnl_col].mean()) if len(stp) else float("nan"),
        "raw_exp":     raw_exp,
        "adj_exp":     adj_exp,
        "monthly":     adj_exp * tpd * PAYOUT_DAYS * DOLLARS_PER_PT,
        "pct_pos":     float((daily>0).mean())*100,
        "days_sl":     int((daily_usd<=DAILY_STOP_USD).sum()),
        "max_dd":      _max_dd(daily) * DOLLARS_PER_PT,
        "sharpe":      _sharpe(daily),
        "good_cycles": sum(cycles),
        "n_cycles":    len(cycles),
        "daily":       daily,
    }

m_base = _metrics(df_base, label="Baseline  PT10+SL25")
m_s1   = _metrics(df_s1,   label="Strategy1 Scale-Out (50%@PT10+trail)")
m_s2   = _metrics(df_s2,   label="Strategy2 VWAP target")
m_s3   = _metrics(df_s3,   label="Strategy3 Tiered (SHORT PT20 / LONG PT10)")

all_m  = [m_base, m_s1, m_s2, m_s3]
best_m = max(all_m, key=lambda x: x["sharpe"])

# ──────────────────────────────────────────────────────────────────────────────
# Build report
# ──────────────────────────────────────────────────────────────────────────────

lines = []
def p(s=""): lines.append(str(s))

p("="*80)
p("TARGET STRATEGY COMPARISON")
p(f"Filter: 10:00-10:45 ET + vwap_dist_pts <= 50   Trading days: {TRADING_DAYS}")
p(f"Training: {fv['session_date'].min()} to {fv['session_date'].max()}")
p(f"Commission: {COMMISSION_PTS} pts ($5 RT)   Cooldown: {COOLDOWN} bars   $20/pt")
p("="*80)
p()

# ── Summary comparison table ───────────────────────────────────────────────────
p("-"*80)
p("STRATEGY COMPARISON SUMMARY")
p("-"*80)
p()
hdr = (f"  {'Strategy':<44}  {'n':>5}  {'wr%':>6}  {'adj_exp':>8}  "
       f"{'mnth$/acc':>10}  {'pos%':>6}  {'sl_days':>8}  {'max_dd$':>10}  {'sharpe':>7}  {'cyc':>5}")
p(hdr)
p("  "+"-"*(len(hdr)-2))
for m in all_m:
    star = " <-- BEST" if m["label"]==best_m["label"] else ""
    p(f"  {m['label']:<44}  {m['n']:>5,}  {m['wr']*100:>5.1f}%  "
      f"{m['adj_exp']:>8.3f}  {m['monthly']:>+10,.0f}  "
      f"{m['pct_pos']:>5.1f}%  {m['days_sl']:>8d}  "
      f"{m['max_dd']:>+10,.0f}  {m['sharpe']:>7.2f}  "
      f"{m['good_cycles']:>2}/{m['n_cycles']}{star}")
p()

# ── Baseline detail ────────────────────────────────────────────────────────────
p("-"*80)
p("BASELINE — PT10+SL25 (position managed)")
p("-"*80)
m = m_base
p(f"  n={m['n']:,}  ev/day={m['tpd']:.2f}  win_rate={m['wr']*100:.1f}%")
p(f"  exit: target={m['n_target']:,}  stop={m['n_stop']:,}  close={m['n_close']:,}")
p(f"  mean_winner=+{m['mean_win']:.2f} pts   mean_loser={m['mean_loss']:.2f} pts")
p(f"  raw_exp={m['raw_exp']:+.3f} pts   adj_exp={m['adj_exp']:+.3f} pts")
p(f"  monthly/acc={m['monthly']:>+,.0f}   monthly x20={m['monthly']*20:>+,.0f}")
p(f"  pos_days={m['pct_pos']:.1f}%   sl_days={m['days_sl']}   max_dd={m['max_dd']:>+,.0f}   sharpe={m['sharpe']:.2f}")
p(f"  payout_cycles={m['good_cycles']}/{m['n_cycles']}")
p()

# ── Strategy 1 detail ──────────────────────────────────────────────────────────
p("-"*80)
p("STRATEGY 1 — SCALE-OUT (50% at PT10, trail 50% from breakeven)")
p("-"*80)
m = m_s1
p(f"  n={m['n']:,}  ev/day={m['tpd']:.2f}  win_rate={m['wr']*100:.1f}%  (first half hit target)")
p(f"  exit first half:  target={m['n_target']:,}  stop={m['n_stop']:,}  close={m['n_close']:,}")
p(f"  raw_exp={m['raw_exp']:+.3f} pts   adj_exp={m['adj_exp']:+.3f} pts")
p(f"  monthly/acc={m['monthly']:>+,.0f}   monthly x20={m['monthly']*20:>+,.0f}")
p(f"  pos_days={m['pct_pos']:.1f}%   sl_days={m['days_sl']}   max_dd={m['max_dd']:>+,.0f}   sharpe={m['sharpe']:.2f}")
p(f"  payout_cycles={m['good_cycles']}/{m['n_cycles']}")
p()

n_s1_fh  = df_s1["first_hit"].sum()
n_s1_tot = len(df_s1)

# Second half outcomes (only for trades where first half hit)
s1_fh = df_s1[df_s1["first_hit"]]
if len(s1_fh):
    s1_trail = s1_fh[s1_fh["second_exit"]=="trail_stop"]
    s1_close = s1_fh[s1_fh["second_exit"]=="session_close"]
    p(f"  SECOND HALF OUTCOMES (n={len(s1_fh):,} first-half hits):")
    p(f"    Trail stop hit     : {len(s1_trail):,}  ({len(s1_trail)/len(s1_fh)*100:.1f}%)  "
      f"mean 2nd-half = {s1_trail['second_pnl'].mean():>+.2f} pts")
    p(f"    Session close      : {len(s1_close):,}  ({len(s1_close)/len(s1_fh)*100:.1f}%)  "
      f"mean 2nd-half = {s1_close['second_pnl'].mean():>+.2f} pts")
    p(f"    Mean pts on 2nd half (all)  : {s1_fh['second_pnl'].mean():>+.2f} pts")
    p(f"    2nd half > breakeven (>0)   : {(s1_fh['second_pnl']>0).mean()*100:.1f}%")
    p(f"    2nd half > +10 pts          : {(s1_fh['second_pnl']>10).mean()*100:.1f}%")
    p(f"    2nd half > +20 pts          : {(s1_fh['second_pnl']>20).mean()*100:.1f}%")
    p(f"    2nd half > +50 pts          : {(s1_fh['second_pnl']>50).mean()*100:.1f}%")
    p()
    p(f"  Second half pnl distribution:")
    edges = [(-999,-10),(-10,0),(0,5),(5,10),(10,20),(20,50),(50,999)]
    lbls  = ["<-10","-10 to 0","0 to 5","5 to 10","10 to 20","20 to 50","50+"]
    for (lo,hi),lbl in zip(edges,lbls):
        cnt = s1_fh["second_pnl"].between(lo,hi,inclusive="left").sum()
        p(f"    {lbl:<12}: {cnt:>4,}  ({cnt/len(s1_fh)*100:.1f}%)")
    p()

p(f"  COMPARISON: Scale-Out vs PT10-Only (same trades taken):")
p(f"    PT10 equivalent total P&L     : {df_s1['pt10_equiv'].sum():>+.1f} pts  "
  f"({df_s1['pt10_equiv'].mean():>+.3f} mean)")
p(f"    Scale-out actual total P&L    : {df_s1['pnl'].sum():>+.1f} pts  "
  f"({df_s1['pnl'].mean():>+.3f} mean)")
p(f"    Delta (scale-out vs PT10)     : {(df_s1['pnl']-df_s1['pt10_equiv']).mean():>+.3f} pts/trade")
p()

# ── Strategy 2 detail ──────────────────────────────────────────────────────────
p("-"*80)
p("STRATEGY 2 — VWAP TARGET (exit at session VWAP, skip if < 10 pts away)")
p("-"*80)
m = m_s2

# Count skipped signals: re-compute
n_skipped_vwap = ev.apply(
    lambda r: abs(r["bar_close"]-r["vwap"])<10.0, axis=1
).sum()
p(f"  Base signals: {len(ev):,}   Skipped (VWAP<10pts): {n_skipped_vwap:,} "
  f"({n_skipped_vwap/len(ev)*100:.1f}%)   Eligible: {len(ev)-n_skipped_vwap:,}")
p(f"  Trades taken (after position mgmt): {m['n']:,}  ev/day={m['tpd']:.2f}")
p(f"  win_rate={m['wr']*100:.1f}%  (target={m['n_target']:,}  stop={m['n_stop']:,}  close={m['n_close']:,})")
if len(df_s2):
    p(f"  Mean VWAP distance at entry: {df_s2['vwap_dist'].mean():.2f} pts  "
      f"(range: {df_s2['vwap_dist'].min():.1f} to {df_s2['vwap_dist'].max():.1f})")
p(f"  raw_exp={m['raw_exp']:+.3f} pts   adj_exp={m['adj_exp']:+.3f} pts")
p(f"  monthly/acc={m['monthly']:>+,.0f}   monthly x20={m['monthly']*20:>+,.0f}")
p(f"  pos_days={m['pct_pos']:.1f}%   sl_days={m['days_sl']}   max_dd={m['max_dd']:>+,.0f}   sharpe={m['sharpe']:.2f}")
p(f"  payout_cycles={m['good_cycles']}/{m['n_cycles']}")
p()
if len(df_s2):
    s2_t = df_s2[df_s2["exit_reason"]=="target"]
    p(f"  Mean pts captured on winners: {s2_t['pnl'].mean():>+.2f} pts  (= mean VWAP dist for winners)")
    p(f"  Mean pts on all trades (raw): {df_s2['pnl'].mean():>+.2f} pts")
    p()

# ── Strategy 3 detail ──────────────────────────────────────────────────────────
p("-"*80)
p("STRATEGY 3 — TIERED (SHORT=PT20+SL25, LONG=PT10+SL25)")
p("-"*80)
m = m_s3
p(f"  n={m['n']:,}  ev/day={m['tpd']:.2f}  overall win_rate={m['wr']*100:.1f}%")
p(f"  raw_exp={m['raw_exp']:+.3f} pts   adj_exp={m['adj_exp']:+.3f} pts")
p(f"  monthly/acc={m['monthly']:>+,.0f}   monthly x20={m['monthly']*20:>+,.0f}")
p(f"  pos_days={m['pct_pos']:.1f}%   sl_days={m['days_sl']}   max_dd={m['max_dd']:>+,.0f}   sharpe={m['sharpe']:.2f}")
p(f"  payout_cycles={m['good_cycles']}/{m['n_cycles']}")
p()

for d_lbl, d_val in [("SHORT","short"),("LONG","long")]:
    sub = df_s3[df_s3["direction"]==d_val]
    if len(sub)==0: continue
    t_sub   = sub[sub["exit_reason"]=="target"]
    s_sub   = sub[sub["exit_reason"]=="stop"]
    wr_sub  = len(t_sub)/len(sub)*100
    exp_sub = sub["pnl"].mean()
    p(f"  {d_lbl} trades: n={len(sub):,}  wr={wr_sub:.1f}%  "
      f"mean_pnl={exp_sub:>+.2f} pts  "
      f"(target={len(t_sub):,}  stop={len(s_sub):,}  "
      f"close={len(sub[sub['exit_reason']=='session_close']):,})")
p()

# ── Monthly returns table ──────────────────────────────────────────────────────
p("-"*80)
p("MONTHLY P&L COMPARISON ($, 1 account)")
p("-"*80)
p()

all_daily = {
    "Base": m_base["daily"],
    "S1":   m_s1["daily"],
    "S2":   m_s2["daily"],
    "S3":   m_s3["daily"],
}
monthly_all = {}
for tag, d in all_daily.items():
    s = (d * DOLLARS_PER_PT).copy()
    s.index = pd.to_datetime(s.index)
    monthly_all[tag] = s.resample("ME").sum()

months = sorted(set().union(*[list(v.index) for v in monthly_all.values()]))
p(f"  {'Month':<10}  {'Base':>10}  {'S1 Scale':>10}  {'S2 VWAP':>10}  {'S3 Tierd':>10}")
p(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for mo in months:
    vals = {tag: monthly_all[tag].get(mo, 0) for tag in ["Base","S1","S2","S3"]}
    p(f"  {str(mo.date()):<10}  "
      f"{vals['Base']:>+10,.0f}  {vals['S1']:>+10,.0f}  "
      f"{vals['S2']:>+10,.0f}  {vals['S3']:>+10,.0f}")
p()

# ── Best strategy daily series ─────────────────────────────────────────────────
tag_map = {"Baseline  PT10+SL25": "Base", "Strategy1 Scale-Out (50%@PT10+trail)": "S1",
           "Strategy2 VWAP target": "S2", "Strategy3 Tiered (SHORT PT20 / LONG PT10)": "S3"}
best_tag = tag_map.get(best_m["label"], "Base")
best_daily = all_daily[best_tag]

p("-"*80)
p(f"BEST STRATEGY DAILY SERIES: {best_m['label']}")
p(f"(ranked by Sharpe ratio: {best_m['sharpe']:.2f})")
p("-"*80)
p()
p(f"  {'Date':<14}  {'pts':>8}  {'$':>8}  {'cum_pts':>10}  {'cum_$':>10}  flag")
p(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}")
cum = 0.0
for date in ALL_DATES:
    dp  = float(best_daily.loc[date]) if date in best_daily.index else 0.0
    cum += dp
    flag = "<-- DAILY STOP" if dp*DOLLARS_PER_PT <= DAILY_STOP_USD else ("+" if dp>0 else "")
    p(f"  {date:<14}  {dp:>8.2f}  {dp*DOLLARS_PER_PT:>+8.0f}  "
      f"{cum:>10.2f}  {cum*DOLLARS_PER_PT:>+10,.0f}  {flag}")
p()

# ── Save ───────────────────────────────────────────────────────────────────────
report = "\n".join(lines)
REPORT_PATH.write_text(report, encoding="utf-8")
print(f"\nReport saved to {REPORT_PATH}")

# ── Quantstats tearsheet ───────────────────────────────────────────────────────
print(f"Generating tearsheet for best strategy: {best_m['label']}...")
rets = pd.Series(
    best_daily.values * DOLLARS_PER_PT / ACCOUNT_SIZE_USD,
    index=pd.DatetimeIndex(pd.to_datetime(best_daily.index)),
    name=best_m["label"],
)
tear_path = TEAR_DIR / "target_strategy_comparison.html"
try:
    qs.reports.html(rets, output=str(tear_path),
                    title=f"NQ — {best_m['label']}",
                    download_filename=str(tear_path))
    print(f"Tearsheet saved to {tear_path}")
except Exception as e:
    print(f"Tearsheet failed: {e}")

try:
    print("\n" + report)
except UnicodeEncodeError:
    print("\n" + report.encode("ascii", errors="replace").decode("ascii"))

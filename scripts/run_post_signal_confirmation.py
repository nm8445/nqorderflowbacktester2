"""
Post-signal confirmation analysis: forward bars 1-3 after signal bar.

For clean_large and loss events (10:00-10:45 ET), computes per-level
delta features on bars +1, +2, +3 after the signal bar.

Tests whether any forward-bar confirmation signal separates clean_large
from loss significantly, and evaluates three simple confirmation rules.

Saves output/diagnostics/post_signal_confirmation.txt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from nqbt.analysis.reaction_scanner import _load_rth_ticks
from nqbt.analysis.range_bars import build_range_bars

EV_PATH  = PROJECT_ROOT / "output" / "diagnostics" / "reaction_events.csv"
OUT_PATH = PROJECT_ROOT / "output" / "diagnostics" / "post_signal_confirmation.txt"
ET       = "America/New_York"
SESSION_PCT = 60.0

# ---------------------------------------------------------------------------
# Load and filter events
# ---------------------------------------------------------------------------

print("Loading reaction events...")
ev = pd.read_csv(EV_PATH)
ev["_et"]   = pd.to_datetime(ev["event_time"], utc=True).dt.tz_convert(ET)
ev["_hhmm"] = ev["_et"].dt.hour * 60 + ev["_et"].dt.minute
ev = ev[(ev["_hhmm"] >= 10*60) & (ev["_hhmm"] < 10*60+45)].copy()

def _dir(loc):
    if loc in ("outside_vah","at_vah"):  return "short"
    if loc in ("outside_val","at_val"):  return "long"
    return "unknown"

ev["_dir"]    = ev["price_location"].apply(_dir)
ev["mfe_raw"] = np.where(ev["_dir"]=="short", ev["mfe_down"],
                np.where(ev["_dir"]=="long",  ev["mfe_up"],  float("nan")))
ev["mae_raw"] = np.where(ev["_dir"]=="short", ev["mfe_up"],
                np.where(ev["_dir"]=="long",  ev["mfe_down"], float("nan")))

def _bucket(r):
    if pd.isna(r["mfe_raw"]) or pd.isna(r["mae_raw"]): return "unknown"
    if r["mfe_raw"] < 10.0 or r["mae_raw"] > r["mfe_raw"]: return "loss"
    if r["mfe_raw"] < 25.0: return "scalp"
    if r["mfe_raw"] < 50.0: return "medium"
    return "large"

ev["outcome"]     = ev.apply(_bucket, axis=1)
ev["clean_large"] = (ev["outcome"] == "large") & (ev["mae_raw"] < 10.0)
ev["is_loss"]     = ev["outcome"] == "loss"

ev_cl   = ev[ev["clean_large"]].copy()
ev_loss = ev[ev["is_loss"]].copy()

TRADING_DAYS = ev["session_date"].nunique()
print(f"  clean_large: {len(ev_cl):,}   loss: {len(ev_loss):,}   sessions: {TRADING_DAYS}")

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _session_60th(bars):
    vals = []
    for b in bars:
        try:
            vals.extend(b.levels_df()["delta"].abs().tolist())
        except Exception:
            pass
    return float(np.percentile(vals, SESSION_PCT)) if vals else 0.0


def _bar_features(bar, direction: str, sess_60th: float) -> dict:
    """
    Features for a single forward bar.

    direction_aligned   : net_delta in reversal direction
                          SHORT → delta < 0 (sell pressure)
                          LONG  → delta > 0 (buy pressure)
    close_direction     : bar closed in reversal direction
                          SHORT → close < open
                          LONG  → close > open
    absorption_present  : close AGAINST dominant delta (any direction)
    opposing_absorption : close against dominant delta AND that signals
                          failure of the trade
                          SHORT → sellers dominated but bar closed UP
                          LONG  → buyers dominated but bar closed DOWN
    """
    net_delta   = float(bar.delta)
    total_vol   = float(bar.total_vol)
    bar_went_up = bar.close > bar.open

    direction_aligned = (net_delta < 0) if direction == "short" else (net_delta > 0)
    close_direction   = (not bar_went_up) if direction == "short" else bar_went_up

    dominant_buy = net_delta >= 0
    absorption   = (dominant_buy and not bar_went_up) or (not dominant_buy and bar_went_up)

    # Opposing absorption = trade is running into trouble
    if direction == "short":
        opposing = (net_delta < 0) and bar_went_up   # sellers tried, buyers won
    else:
        opposing = (net_delta > 0) and (not bar_went_up)  # buyers tried, sellers won

    try:
        levels    = bar.levels_df()
        zone_count = int((levels["delta"].abs() >= sess_60th).sum())
    except Exception:
        zone_count = 0

    return {
        "net_delta":          net_delta,
        "direction_aligned":  direction_aligned,
        "close_direction":    close_direction,
        "zone_count":         float(zone_count),
        "absorption_present": absorption,
        "opposing_absorption": opposing,
        "total_vol":          total_vol,
    }


def _process_events(events_df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    sessions = sorted(events_df["session_date"].unique())
    rows     = []

    for si, session_date in enumerate(sessions):
        if si % 25 == 0:
            print(f"    [{group_label}] Session {si+1}/{len(sessions)}: {session_date}")

        ticks = _load_rth_ticks(session_date)
        if ticks.empty:
            continue
        bars = build_range_bars(ticks, range_ticks=40)
        if not bars:
            continue

        sess_60th = _session_60th(bars)
        sess_ev   = events_df[events_df["session_date"] == session_date]

        for _, row in sess_ev.iterrows():
            bar_idx   = int(row["bar_idx"])
            direction = row["_dir"]

            # Need at least 3 forward bars; skip if near session end
            if bar_idx + 3 >= len(bars):
                continue

            fwd = [bars[bar_idx + i] for i in range(1, 4)]  # bars +1, +2, +3
            f   = [_bar_features(b, direction, sess_60th) for b in fwd]

            # Cumulative features
            cum_delta   = sum(fi["net_delta"] for fi in f)
            aligned_cnt = sum(1 for fi in f if fi["direction_aligned"])
            close_cnt   = sum(1 for fi in f if fi["close_direction"])

            # First bar (1-indexed) with opposing absorption; 0 = none in first 3
            first_opp = 0
            for i, fi in enumerate(f):
                if fi["opposing_absorption"]:
                    first_opp = i + 1
                    break

            record = {
                "session_date":   session_date,
                "event_time":     row["event_time"],
                "direction":      direction,
                "price_location": row["price_location"],
                "mfe_raw":        row["mfe_raw"],
                "mae_raw":        row["mae_raw"],
                "outcome":        row["outcome"],
                "group":          group_label,
                # bar+1
                "bar1_net_delta":          f[0]["net_delta"],
                "bar1_direction_aligned":  f[0]["direction_aligned"],
                "bar1_close_direction":    f[0]["close_direction"],
                "bar1_zone_count":         f[0]["zone_count"],
                "bar1_absorption_present": f[0]["absorption_present"],
                "bar1_opposing_absorption": f[0]["opposing_absorption"],
                "bar1_total_vol":          f[0]["total_vol"],
                # bar+2
                "bar2_net_delta":          f[1]["net_delta"],
                "bar2_direction_aligned":  f[1]["direction_aligned"],
                "bar2_close_direction":    f[1]["close_direction"],
                "bar2_zone_count":         f[1]["zone_count"],
                "bar2_absorption_present": f[1]["absorption_present"],
                "bar2_opposing_absorption": f[1]["opposing_absorption"],
                "bar2_total_vol":          f[1]["total_vol"],
                # bar+3
                "bar3_net_delta":          f[2]["net_delta"],
                "bar3_direction_aligned":  f[2]["direction_aligned"],
                "bar3_close_direction":    f[2]["close_direction"],
                "bar3_zone_count":         f[2]["zone_count"],
                "bar3_absorption_present": f[2]["absorption_present"],
                "bar3_opposing_absorption": f[2]["opposing_absorption"],
                "bar3_total_vol":          f[2]["total_vol"],
                # cumulative
                "bars_1to3_net_delta":             cum_delta,
                "bars_1to3_direction_aligned_count": float(aligned_cnt),
                "bars_1to3_close_aligned_count":    float(close_cnt),
                "first_opposing_absorption_bar":    float(first_opp),
            }
            rows.append(record)

    return pd.DataFrame(rows)


print("\nProcessing clean_large events...")
df_cl   = _process_events(ev_cl,   "clean_large")
print(f"  Processed: {len(df_cl):,}")

print("\nProcessing loss events...")
df_loss = _process_events(ev_loss, "loss")
print(f"  Processed: {len(df_loss):,}")

# ---------------------------------------------------------------------------
# Separation analysis
# ---------------------------------------------------------------------------

NUM_FEATURES = [
    "bar1_net_delta", "bar1_zone_count", "bar1_total_vol",
    "bar2_net_delta", "bar2_zone_count", "bar2_total_vol",
    "bar3_net_delta", "bar3_zone_count", "bar3_total_vol",
    "bars_1to3_net_delta",
    "bars_1to3_direction_aligned_count",
    "bars_1to3_close_aligned_count",
    "first_opposing_absorption_bar",
]
BOOL_FEATURES = [
    "bar1_direction_aligned", "bar1_close_direction",
    "bar1_absorption_present", "bar1_opposing_absorption",
    "bar2_direction_aligned", "bar2_close_direction",
    "bar2_absorption_present", "bar2_opposing_absorption",
    "bar3_direction_aligned", "bar3_close_direction",
    "bar3_absorption_present", "bar3_opposing_absorption",
]

def _sep(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)];      b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float("nan")
    vp = ((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1)) / (len(a)+len(b)-2)
    return abs(np.mean(a)-np.mean(b)) / np.sqrt(vp) if vp > 0 else float("nan")

def _ks(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)];      b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5: return float("nan"), float("nan")
    ks, p = scipy_stats.ks_2samp(a, b)
    return float(ks), float(p)

rows = []
for feat in NUM_FEATURES + BOOL_FEATURES:
    a = df_cl[feat].astype(float).values
    b = df_loss[feat].astype(float).values
    sr = _sep(a, b)
    ks, p = _ks(a, b)
    rows.append({
        "feature": feat,
        "cl_mean": float(np.nanmean(a)),
        "ls_mean": float(np.nanmean(b)),
        "sep_ratio": sr, "ks_stat": ks, "ks_p": p,
    })

sep_df = pd.DataFrame(rows).sort_values("sep_ratio", ascending=False)

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

lines = []
def p(s=""): lines.append(s)

n_cl   = len(df_cl)
n_loss = len(df_loss)

p("=" * 76)
p("POST-SIGNAL CONFIRMATION ANALYSIS")
p(f"Window: 10:00-10:45 ET   Forward bars: +1, +2, +3 after signal bar")
p(f"Zone threshold: session {SESSION_PCT:.0f}th percentile of abs_delta")
p(f"clean_large: n={n_cl:,}   loss: n={n_loss:,}   trading days: {TRADING_DAYS}")
p("=" * 76)
p()

# ── Section 1: Separation table ───────────────────────────────────────────
p("-" * 76)
p("SECTION 1 — FEATURE SEPARATION RATIOS (ranked descending)")
p("-" * 76)
p()
p(f"  {'Feature':<42}  {'cl_mean':>8}  {'ls_mean':>8}  {'sep':>7}  {'KS':>7}  {'p':>7}")
p(f"  {'-'*42}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
for _, r in sep_df.iterrows():
    p(f"  {r['feature']:<42}  {r['cl_mean']:>8.4f}  {r['ls_mean']:>8.4f}  "
      f"{r['sep_ratio']:>7.4f}  {r['ks_stat']:>7.4f}  {r['ks_p']:>7.4f}")
p()

# ── Section 2: Boolean rates for bar+1 ───────────────────────────────────
p("-" * 76)
p("SECTION 2 — BAR+1 BOOLEAN RATES")
p("-" * 76)
p()
for feat in ["bar1_direction_aligned","bar1_close_direction",
             "bar1_absorption_present","bar1_opposing_absorption"]:
    cl_r = df_cl[feat].mean()  * 100
    ls_r = df_loss[feat].mean() * 100
    p(f"  {feat}")
    p(f"    clean_large: {cl_r:5.1f}%  ({df_cl[feat].sum():,} / {n_cl:,})")
    p(f"    loss       : {ls_r:5.1f}%  ({df_loss[feat].sum():,} / {n_loss:,})")
    p(f"    lift       : {cl_r-ls_r:+.1f}pp")
    p()

# ── Section 3: Cumulative 3-bar close alignment ───────────────────────────
p("-" * 76)
p("SECTION 3 — CUMULATIVE 3-BAR CLOSE ALIGNMENT (bars_1to3_close_aligned_count)")
p("-" * 76)
p()
p(f"  {'aligned_count':>14}  {'clean_large':>14}  {'loss':>14}  {'lift':>8}")
p(f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*8}")
for cnt in range(4):
    cl_f = (df_cl["bars_1to3_close_aligned_count"] == cnt).mean() * 100
    ls_f = (df_loss["bars_1to3_close_aligned_count"] == cnt).mean() * 100
    p(f"  {cnt:>14}  {cl_f:>13.1f}%  {ls_f:>13.1f}%  {cl_f-ls_f:>+7.1f}pp")
p()
cl_ge2 = (df_cl["bars_1to3_close_aligned_count"] >= 2).mean() * 100
ls_ge2 = (df_loss["bars_1to3_close_aligned_count"] >= 2).mean() * 100
p(f"  >= 2 aligned: clean_large={cl_ge2:.1f}%   loss={ls_ge2:.1f}%   lift={cl_ge2-ls_ge2:+.1f}pp")
p()

# ── Section 4: First opposing absorption bar ─────────────────────────────
p("-" * 76)
p("SECTION 4 — first_opposing_absorption_bar DISTRIBUTION")
p("(0 = no opposing absorption in bars 1-3; 1/2/3 = first bar it appeared)")
p("-" * 76)
p()
p(f"  {'bar':>6}  {'clean_large':>14}  {'loss':>14}  {'lift':>8}")
p(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}")
for v in [0, 1, 2, 3]:
    cl_f = (df_cl["first_opposing_absorption_bar"] == v).mean() * 100
    ls_f = (df_loss["first_opposing_absorption_bar"] == v).mean() * 100
    lbl  = "none" if v == 0 else f"bar+{v}"
    p(f"  {lbl:>6}  {cl_f:>13.1f}%  {ls_f:>13.1f}%  {cl_f-ls_f:>+7.1f}pp")
p()

# ── Section 5: Key question — bar+1 combined condition ───────────────────
p("-" * 76)
p("SECTION 5 — KEY QUESTION: bar1 aligned close AND aligned delta")
p("bar1_close_direction=True AND bar1_direction_aligned=True")
p("-" * 76)
p()
df_cl["bar1_both"]   = df_cl["bar1_close_direction"]  & df_cl["bar1_direction_aligned"]
df_loss["bar1_both"] = df_loss["bar1_close_direction"] & df_loss["bar1_direction_aligned"]
cl_both  = df_cl["bar1_both"].mean()  * 100
ls_both  = df_loss["bar1_both"].mean() * 100
p(f"  bar1_close_direction AND bar1_direction_aligned:")
p(f"    clean_large: {cl_both:.1f}%  ({df_cl['bar1_both'].sum():,} / {n_cl:,})")
p(f"    loss       : {ls_both:.1f}%  ({df_loss['bar1_both'].sum():,} / {n_loss:,})")
p(f"    lift       : {cl_both-ls_both:+.1f}pp")
p()

# Within events where bar1_both=True, what is the outcome distribution?
cl_given_both  = df_cl[df_cl["bar1_both"]]
ls_given_both  = df_loss[df_loss["bar1_both"]]
total_both     = len(cl_given_both) + len(ls_given_both)
# Also include scalp/medium from original events where bar1_both is available
ev_all = pd.concat([df_cl, df_loss], ignore_index=True)
ev_all["bar1_both"] = ev_all["bar1_close_direction"] & ev_all["bar1_direction_aligned"]
p(f"  Of events where bar1_both=True ({total_both:,} total clean_large+loss):")
p(f"    clean_large: {len(cl_given_both):,} ({len(cl_given_both)/total_both*100:.1f}%)")
p(f"    loss:        {len(ls_given_both):,} ({len(ls_given_both)/total_both*100:.1f}%)")
p()

# ── Section 6: Confirmation rules ────────────────────────────────────────
p("-" * 76)
p("SECTION 6 — CONFIRMATION RULE EVALUATION")
p("-" * 76)
p()

# Combine clean_large and loss into one pool for rule evaluation
pool = pd.concat([df_cl, df_loss], ignore_index=True)
pool["is_clean"] = pool["group"] == "clean_large"
total_pool       = len(pool)
cl_pool          = pool["is_clean"].sum()
ls_pool          = total_pool - cl_pool
baseline_cf      = cl_pool / total_pool * 100

p(f"  BASELINE (no confirmation, clean_large + loss pool):")
p(f"    Total events : {total_pool:,}   ({total_pool/TRADING_DAYS:.2f}/day)")
p(f"    clean_frac   : {baseline_cf:.1f}%")
p()

rules = [
    ("Rule A", "bar1 closes in reversal direction",
     pool["bar1_close_direction"]),
    ("Rule B", "bar1 AND bar2 both close in reversal direction",
     pool["bar1_close_direction"] & pool["bar2_close_direction"]),
    ("Rule C", "bars_1to3_close_aligned_count >= 2",
     pool["bars_1to3_close_aligned_count"] >= 2),
    ("Rule D", "bar1_close_direction AND bar1_direction_aligned (both)",
     pool["bar1_close_direction"] & pool["bar1_direction_aligned"]),
    ("Rule E", "bar1_both AND bar2_close_direction",
     pool["bar1_close_direction"] & pool["bar1_direction_aligned"] & pool["bar2_close_direction"]),
    ("Rule F", "no opposing absorption in bar+1",
     ~pool["bar1_opposing_absorption"]),
    ("Rule G", "bar1_close_direction AND no bar1_opposing_absorption",
     pool["bar1_close_direction"] & ~pool["bar1_opposing_absorption"]),
]

p(f"  {'Rule':<8}  {'Description':<50}  {'n':>6}  {'ev/day':>7}  {'clean%':>8}  {'lift':>7}")
p(f"  {'-'*8}  {'-'*50}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*7}")
for name, desc, mask in rules:
    sub     = pool[mask]
    n       = len(sub)
    cf      = sub["is_clean"].mean() * 100 if n > 0 else 0.0
    epd     = n / TRADING_DAYS
    lift    = cf - baseline_cf
    p(f"  {name:<8}  {desc:<50}  {n:>6,}  {epd:>7.2f}  {cf:>7.1f}%  {lift:>+6.1f}pp")
p()

# Detailed breakdown per rule: clean_large count, loss count, events/day
p("  Detailed per-rule breakdown:")
p()
for name, desc, mask in rules:
    sub     = pool[mask]
    n       = len(sub)
    n_cl_r  = sub["is_clean"].sum()
    n_ls_r  = n - n_cl_r
    cf      = sub["is_clean"].mean() * 100 if n > 0 else 0.0
    epd     = n / TRADING_DAYS
    pct_cl_retained = n_cl_r / cl_pool * 100  # % of all clean events retained
    pct_ls_retained = n_ls_r / ls_pool * 100  # % of all loss events retained
    p(f"  {name}: {desc}")
    p(f"    total={n:,} ({epd:.2f}/day)  clean={n_cl_r:,} ({cf:.1f}%)  loss={n_ls_r:,}")
    p(f"    retained: {pct_cl_retained:.1f}% of clean_large  |  {pct_ls_retained:.1f}% of loss")
    p()

# ── Section 7: Cross-bar feature breakdown ────────────────────────────────
p("-" * 76)
p("SECTION 7 — PER-BAR ALIGNED CLOSE RATES (bars +1, +2, +3)")
p("-" * 76)
p()
p(f"  {'Bar':<8}  {'Feature':<30}  {'clean_large':>12}  {'loss':>12}  {'lift':>8}")
p(f"  {'-'*8}  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*8}")
for bi in [1, 2, 3]:
    for feat_suffix in ["close_direction", "direction_aligned", "opposing_absorption"]:
        feat = f"bar{bi}_{feat_suffix}"
        cl_r = df_cl[feat].mean() * 100
        ls_r = df_loss[feat].mean() * 100
        p(f"  bar+{bi}     {feat_suffix:<30}  {cl_r:>11.1f}%  {ls_r:>11.1f}%  {cl_r-ls_r:>+7.1f}pp")
    p()

# ── Section 8: Conditional clean_large rate for bar+1 combined conditions ─
p("-" * 76)
p("SECTION 8 — CONDITIONAL ANALYSIS: clean_large rate vs bar+1 conditions")
p("-" * 76)
p()
conditions = [
    ("bar1_close_direction=T, bar1_dir_aligned=T",
     pool["bar1_close_direction"] & pool["bar1_direction_aligned"]),
    ("bar1_close_direction=T, bar1_dir_aligned=F",
     pool["bar1_close_direction"] & ~pool["bar1_direction_aligned"]),
    ("bar1_close_direction=F, bar1_dir_aligned=T",
     ~pool["bar1_close_direction"] & pool["bar1_direction_aligned"]),
    ("bar1_close_direction=F, bar1_dir_aligned=F",
     ~pool["bar1_close_direction"] & ~pool["bar1_direction_aligned"]),
]
p(f"  {'Condition':<48}  {'n':>6}  {'clean%':>8}  {'vs_base':>8}")
p(f"  {'-'*48}  {'-'*6}  {'-'*8}  {'-'*8}")
for desc, mask in conditions:
    sub = pool[mask]
    cf  = sub["is_clean"].mean() * 100 if len(sub) > 0 else 0.0
    p(f"  {desc:<48}  {len(sub):>6,}  {cf:>7.1f}%  {cf-baseline_cf:>+7.1f}pp")
p()

# ── Summary ───────────────────────────────────────────────────────────────
p("=" * 76)
p("KEY FINDINGS SUMMARY")
p("=" * 76)
p()
best_sep = sep_df.iloc[0]
p(f"  Top separator: {best_sep['feature']}")
p(f"    clean_large={best_sep['cl_mean']:.4f}  loss={best_sep['ls_mean']:.4f}")
p(f"    sep_ratio={best_sep['sep_ratio']:.4f}  KS={best_sep['ks_stat']:.4f}  p={best_sep['ks_p']:.4f}")
p()
p(f"  bar1_close_direction (reversal close on bar+1):")
p(f"    clean_large: {df_cl['bar1_close_direction'].mean()*100:.1f}%")
p(f"    loss:        {df_loss['bar1_close_direction'].mean()*100:.1f}%")
p(f"    lift:        {(df_cl['bar1_close_direction'].mean()-df_loss['bar1_close_direction'].mean())*100:+.1f}pp")
p()
p(f"  Best confirmation rule by clean_fraction:")
best_rule = max(rules, key=lambda x: pool[x[2]]["is_clean"].mean() if len(pool[x[2]]) > 0 else 0)
sub_best  = pool[best_rule[2]]
p(f"    {best_rule[0]}: {best_rule[1]}")
p(f"    events={len(sub_best):,}  clean%={sub_best['is_clean'].mean()*100:.1f}%  "
  f"events/day={len(sub_best)/TRADING_DAYS:.2f}")
p()
p(f"  Answer to KEY QUESTION: does bar1 aligned close+delta predict clean_large?")
p(f"    bar1 both conditions: clean={cl_both:.1f}%  loss={ls_both:.1f}%  lift={cl_both-ls_both:+.1f}pp")
p(f"    Separation ratio: {sep_df[sep_df['feature']=='bar1_close_direction']['sep_ratio'].values[0]:.4f}")
if abs(cl_both - ls_both) < 3.0:
    p(f"    CONCLUSION: No meaningful separation. Bar+1 close direction does NOT")
    p(f"    predict clean_large outcomes better than random.")
else:
    p(f"    CONCLUSION: Modest separation found. Bar+1 conditions may have edge.")
p()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

report_text = "\n".join(lines)
OUT_PATH.write_text(report_text, encoding="utf-8")
print(f"\nReport saved to {OUT_PATH}")
try:
    print("\n" + report_text)
except UnicodeEncodeError:
    print("\n" + report_text.encode("ascii", errors="replace").decode("ascii"))

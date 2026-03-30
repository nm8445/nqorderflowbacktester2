"""
Orderflow signal profile: clean_large vs loss events, 10:00-10:45 ET.

For each qualifying event loads the signal bar's per-level delta profile
using bar.levels_df() (1.25-pt zones already built by range_bar engine).
Computes 10 orderflow features per event, then ranks them by separation
ratio (clean_large vs loss).

Saves output/diagnostics/orderflow_signal_profile.txt
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EV_PATH   = PROJECT_ROOT / "output" / "diagnostics" / "reaction_events.csv"
OUT_PATH  = PROJECT_ROOT / "output" / "diagnostics" / "orderflow_signal_profile.txt"
ET        = "America/New_York"

SESSION_PCT_THRESHOLD = 60.0   # percentile for "significant zone" flag

# ---------------------------------------------------------------------------
# Load and filter events
# ---------------------------------------------------------------------------

print("Loading reaction events...")
ev = pd.read_csv(EV_PATH)
print(f"  Total events: {len(ev):,}")

ev["_et"]    = pd.to_datetime(ev["event_time"], utc=True).dt.tz_convert(ET)
ev["_hhmm"]  = ev["_et"].dt.hour * 60 + ev["_et"].dt.minute

# Time window: 10:00-10:45 ET
ev = ev[(ev["_hhmm"] >= 10*60) & (ev["_hhmm"] < 10*60+45)].copy()
print(f"  After 10:00-10:45 filter: {len(ev):,}")

# Direction and mfe/mae
def _dir(loc):
    if loc in ("outside_vah", "at_vah"):  return "short"
    if loc in ("outside_val", "at_val"):  return "long"
    return "unknown"

ev["_dir"]    = ev["price_location"].apply(_dir)
ev["mfe_raw"] = np.where(ev["_dir"]=="short", ev["mfe_down"],
                np.where(ev["_dir"]=="long",  ev["mfe_up"],  float("nan")))
ev["mae_raw"] = np.where(ev["_dir"]=="short", ev["mfe_up"],
                np.where(ev["_dir"]=="long",  ev["mfe_down"], float("nan")))

def _bucket(r):
    if pd.isna(r["mfe_raw"]) or pd.isna(r["mae_raw"]): return "unknown"
    if r["mfe_raw"] < 10.0 or r["mae_raw"] > r["mfe_raw"]: return "loss"
    if r["mfe_raw"] < 25.0:  return "scalp"
    if r["mfe_raw"] < 50.0:  return "medium"
    return "large"

ev["outcome"]     = ev.apply(_bucket, axis=1)
ev["clean_large"] = (ev["outcome"] == "large") & (ev["mae_raw"] < 10.0)
ev["is_loss"]     = ev["outcome"] == "loss"

# Keep only clean_large and loss for comparison
ev_cl   = ev[ev["clean_large"]].copy()
ev_loss = ev[ev["is_loss"]].copy()

print(f"  clean_large: {len(ev_cl):,}")
print(f"  loss       : {len(ev_loss):,}")
print(f"  Sessions   : {ev['session_date'].nunique()}")

# ---------------------------------------------------------------------------
# Per-event orderflow feature extraction
# ---------------------------------------------------------------------------

def _compute_session_pct(bars, pct=SESSION_PCT_THRESHOLD):
    """60th percentile of abs_delta across all zone-bar observations."""
    vals = []
    for b in bars:
        try:
            df = b.levels_df()
            vals.extend(df["delta"].abs().tolist())
        except Exception:
            pass
    return float(np.percentile(vals, pct)) if vals else 0.0


def _zone_features(bar, bar_idx: int, bars: list,
                   direction: str, session_60th: float) -> dict:
    """Extract 10 orderflow features for the signal bar."""
    try:
        levels = bar.levels_df()
    except Exception:
        levels = pd.DataFrame()

    total_vol = bar.total_vol if bar.total_vol > 0 else 1

    # net delta for signal bar
    net_delta = float(bar.delta)

    # dominant side
    dominant_buy = net_delta >= 0

    # max abs zone delta
    if not levels.empty:
        abs_deltas = levels["delta"].abs()
        max_zone_delta = float(abs_deltas.max())
        # zones above session 60th pct
        above_mask     = abs_deltas >= session_60th
        zone_count     = int(above_mask.sum())
        # concentration: fraction of total_vol in above-threshold zones
        zone_vol       = float(levels.loc[above_mask, "total_vol"].sum()) if "total_vol" in levels.columns else 0.0
        zone_conc      = zone_vol / total_vol
    else:
        max_zone_delta = float("nan")
        zone_count     = 0
        zone_conc      = float("nan")

    # bar close direction (True = closed up)
    bar_went_up = bar.close > bar.open
    bar_neutral = bar.close == bar.open

    # absorption: bar closed AGAINST dominant delta
    # short: dominant buy but bar closed down = absorption of buying by sellers
    # long:  dominant sell but bar closed up  = absorption of selling by buyers
    if direction == "short":
        absorption = dominant_buy and (not bar_went_up) and (not bar_neutral)
    else:
        absorption = (not dominant_buy) and bar_went_up and (not bar_neutral)

    # delta_vs_close: net_delta direction opposite to bar close direction
    if bar_neutral:
        delta_vs_close = False
    else:
        delta_dir_up   = net_delta >= 0
        delta_vs_close = (delta_dir_up != bar_went_up)

    # Prior bars delta trend (slope via linear regression)
    prior5  = bars[max(0, bar_idx - 5):bar_idx]
    prior10 = bars[max(0, bar_idx - 10):bar_idx]

    def _slope(bar_list):
        if len(bar_list) < 2:
            return float("nan")
        deltas = [float(b.delta) for b in bar_list]
        x = np.arange(len(deltas), dtype=float)
        return float(np.polyfit(x, deltas, 1)[0])

    slope5  = _slope(prior5)
    slope10 = _slope(prior10)

    # prior_5bar_dominant_consistency:
    # fraction of prior 5 bars where dominant_side matches expected reversal
    # short: expect prior bars to be buy-dominated (running into resistance)
    # long:  expect prior bars to be sell-dominated (running into support)
    if prior5:
        if direction == "short":
            consistency = sum(1 for b in prior5 if b.delta > 0) / len(prior5)
        else:
            consistency = sum(1 for b in prior5 if b.delta < 0) / len(prior5)
    else:
        consistency = float("nan")

    return {
        "signal_bar_net_delta":             net_delta,
        "signal_bar_dominant_side":         "buy" if dominant_buy else "sell",
        "signal_bar_max_zone_delta":        max_zone_delta,
        "signal_bar_zone_count":            float(zone_count),
        "signal_bar_zone_concentration":    zone_conc,
        "signal_bar_absorption_present":    absorption,
        "signal_bar_delta_vs_close":        delta_vs_close,
        "prior_5bar_delta_trend":           slope5,
        "prior_10bar_delta_trend":          slope10,
        "prior_5bar_dominant_consistency":  consistency,
    }


def _process_events(events_df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    sessions = sorted(events_df["session_date"].unique())
    n_sess   = len(sessions)
    rows     = []

    for si, session_date in enumerate(sessions):
        if si % 20 == 0:
            print(f"    [{group_label}] Session {si+1}/{n_sess}: {session_date}")

        ticks = _load_rth_ticks(session_date)
        if ticks.empty:
            continue
        bars = build_range_bars(ticks, range_ticks=40)
        if not bars:
            continue

        # Session-level 60th percentile of abs zone delta
        sess_60th = _compute_session_pct(bars)

        sess_ev = events_df[events_df["session_date"] == session_date]

        for _, row in sess_ev.iterrows():
            bar_idx   = int(row["bar_idx"])
            direction = row["_dir"]
            if bar_idx >= len(bars):
                continue
            bar = bars[bar_idx]
            feats = _zone_features(bar, bar_idx, bars, direction, sess_60th)
            feats["session_date"]    = session_date
            feats["event_time"]      = row["event_time"]
            feats["price_location"]  = row["price_location"]
            feats["direction"]       = direction
            feats["mfe_raw"]         = row["mfe_raw"]
            feats["mae_raw"]         = row["mae_raw"]
            feats["outcome"]         = row["outcome"]
            feats["group"]           = group_label
            rows.append(feats)

    return pd.DataFrame(rows)


print("\nProcessing clean_large events...")
df_cl = _process_events(ev_cl, "clean_large")
print(f"  Processed: {len(df_cl):,}")

print("\nProcessing loss events...")
df_loss = _process_events(ev_loss, "loss")
print(f"  Processed: {len(df_loss):,}")

# ---------------------------------------------------------------------------
# Compute separation ratios
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "signal_bar_net_delta",
    "signal_bar_max_zone_delta",
    "signal_bar_zone_count",
    "signal_bar_zone_concentration",
    "prior_5bar_delta_trend",
    "prior_10bar_delta_trend",
    "prior_5bar_dominant_consistency",
]

BOOL_FEATURES = [
    "signal_bar_absorption_present",
    "signal_bar_delta_vs_close",
]

def _sep_ratio(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n_a, n_b  = len(a), len(b)
    var_pooled = ((n_a - 1)*np.var(a, ddof=1) + (n_b - 1)*np.var(b, ddof=1)) / (n_a + n_b - 2)
    if var_pooled <= 0:
        return float("nan")
    return abs(np.mean(a) - np.mean(b)) / np.sqrt(var_pooled)

def _ks(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    ks, p = scipy_stats.ks_2samp(a, b)
    return float(ks), float(p)

print("\nComputing separation ratios...")
sep_rows = []
for feat in NUMERIC_FEATURES:
    a = df_cl[feat].values
    b = df_loss[feat].values
    sr = _sep_ratio(a, b)
    ks, p = _ks(a, b)
    cl_mean  = float(np.nanmean(a))
    ls_mean  = float(np.nanmean(b))
    cl_med   = float(np.nanmedian(a))
    ls_med   = float(np.nanmedian(b))
    sep_rows.append({
        "feature": feat, "cl_mean": cl_mean, "ls_mean": ls_mean,
        "cl_med": cl_med, "ls_med": ls_med,
        "sep_ratio": sr, "ks_stat": ks, "ks_p": p,
    })

for feat in BOOL_FEATURES:
    a = df_cl[feat].astype(float).values
    b = df_loss[feat].astype(float).values
    sr = _sep_ratio(a, b)
    ks, p = _ks(a, b)
    cl_mean = float(np.nanmean(a))
    ls_mean = float(np.nanmean(b))
    sep_rows.append({
        "feature": feat, "cl_mean": cl_mean, "ls_mean": ls_mean,
        "cl_med": cl_mean, "ls_med": ls_mean,
        "sep_ratio": sr, "ks_stat": ks, "ks_p": p,
    })

sep_df = pd.DataFrame(sep_rows).sort_values("sep_ratio", ascending=False)

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

lines = []
def p(s=""): lines.append(s)

n_cl   = len(df_cl)
n_loss = len(df_loss)

p("=" * 76)
p("ORDERFLOW SIGNAL PROFILE: clean_large vs loss")
p(f"Window: 10:00-10:45 ET   Locations: all (outside_vah/val, at_vah/val)")
p(f"Zone threshold: session {SESSION_PCT_THRESHOLD:.0f}th percentile of abs_delta")
p(f"clean_large: n={n_cl:,}   loss: n={n_loss:,}")
p("=" * 76)
p()

# ── Section 1: Separation ratios ──────────────────────────────────────────
p("-" * 76)
p("SECTION 1 — FEATURE SEPARATION RATIOS (ranked descending)")
p("sep_ratio = |mean_clean - mean_loss| / pooled_std")
p("-" * 76)
p()
p(f"  {'Feature':<38}  {'cl_mean':>9}  {'ls_mean':>9}  {'sep_ratio':>10}  {'ks_stat':>8}  {'ks_p':>8}")
p(f"  {'-'*38}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*8}")
for _, row in sep_df.iterrows():
    p(f"  {row['feature']:<38}  {row['cl_mean']:>9.4f}  {row['ls_mean']:>9.4f}  "
      f"{row['sep_ratio']:>10.4f}  {row['ks_stat']:>8.4f}  {row['ks_p']:>8.4f}")
p()

# ── Section 2: Boolean feature rates ─────────────────────────────────────
p("-" * 76)
p("SECTION 2 — BOOLEAN SIGNAL RATES (% of events where True)")
p("-" * 76)
p()
for feat in BOOL_FEATURES:
    cl_rate  = df_cl[feat].mean()  * 100
    ls_rate  = df_loss[feat].mean() * 100
    lift     = cl_rate - ls_rate
    p(f"  {feat}")
    p(f"    clean_large : {cl_rate:>6.1f}%  ({df_cl[feat].sum():,} / {n_cl:,})")
    p(f"    loss        : {ls_rate:>6.1f}%  ({df_loss[feat].sum():,} / {n_loss:,})")
    p(f"    lift (cl-ls): {lift:>+6.1f}pp")
    p()

# Dominant side breakdown
p("-" * 76)
p("SECTION 3 — DOMINANT SIDE BREAKDOWN")
p("-" * 76)
p()
for label, df in [("clean_large", df_cl), ("loss", df_loss)]:
    vc = df["signal_bar_dominant_side"].value_counts()
    buy_pct  = vc.get("buy",  0) / len(df) * 100
    sell_pct = vc.get("sell", 0) / len(df) * 100
    p(f"  {label}:")
    p(f"    buy-dominated bars  : {buy_pct:.1f}%  ({vc.get('buy',0):,})")
    p(f"    sell-dominated bars : {sell_pct:.1f}%  ({vc.get('sell',0):,})")
    p()

# ── Section 4: zone_count distribution ───────────────────────────────────
p("-" * 76)
p("SECTION 4 — signal_bar_zone_count DISTRIBUTION")
p("(zones with abs_delta >= session 60th percentile)")
p("-" * 76)
p()
max_zc = int(max(df_cl["signal_bar_zone_count"].max(),
                 df_loss["signal_bar_zone_count"].max()))
p(f"  {'zone_count':>12}  {'clean_large':>14}  {'loss':>14}  {'lift':>8}")
p(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*8}")
for zc in range(0, min(max_zc+1, 12)):
    cl_frac  = (df_cl["signal_bar_zone_count"]   == zc).mean() * 100
    ls_frac  = (df_loss["signal_bar_zone_count"] == zc).mean() * 100
    lift     = cl_frac - ls_frac
    p(f"  {zc:>12}  {cl_frac:>13.1f}%  {ls_frac:>13.1f}%  {lift:>+7.1f}pp")
p()
# Stats
p(f"  clean_large: mean={df_cl['signal_bar_zone_count'].mean():.2f}  "
  f"median={df_cl['signal_bar_zone_count'].median():.1f}  "
  f"p75={df_cl['signal_bar_zone_count'].quantile(0.75):.1f}")
p(f"  loss       : mean={df_loss['signal_bar_zone_count'].mean():.2f}  "
  f"median={df_loss['signal_bar_zone_count'].median():.1f}  "
  f"p75={df_loss['signal_bar_zone_count'].quantile(0.75):.1f}")
p()

# ── Section 5: zone_concentration distribution ────────────────────────────
p("-" * 76)
p("SECTION 5 — signal_bar_zone_concentration DISTRIBUTION")
p("(fraction of bar volume in above-threshold zones)")
p("-" * 76)
p()
buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
           (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
p(f"  {'conc range':>14}  {'clean_large':>14}  {'loss':>14}  {'lift':>8}")
p(f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*8}")
for lo, hi in buckets:
    cl_frac = ((df_cl["signal_bar_zone_concentration"] >= lo) &
               (df_cl["signal_bar_zone_concentration"] < hi)).mean() * 100
    ls_frac = ((df_loss["signal_bar_zone_concentration"] >= lo) &
               (df_loss["signal_bar_zone_concentration"] < hi)).mean() * 100
    lbl     = f"{lo:.1f}-{hi:.1f}" if hi < 1.01 else f"{lo:.1f}-1.0"
    p(f"  {lbl:>14}  {cl_frac:>13.1f}%  {ls_frac:>13.1f}%  {cl_frac-ls_frac:>+7.1f}pp")
p()
p(f"  clean_large: mean={df_cl['signal_bar_zone_concentration'].mean():.3f}  "
  f"median={df_cl['signal_bar_zone_concentration'].median():.3f}")
p(f"  loss       : mean={df_loss['signal_bar_zone_concentration'].mean():.3f}  "
  f"median={df_loss['signal_bar_zone_concentration'].median():.3f}")
p()

# ── Section 6: net delta distribution ─────────────────────────────────────
p("-" * 76)
p("SECTION 6 — signal_bar_net_delta DISTRIBUTION")
p("(positive = buy dominated, negative = sell dominated)")
p("-" * 76)
p()
for label, df in [("clean_large", df_cl), ("loss", df_loss)]:
    nd = df["signal_bar_net_delta"].dropna()
    p(f"  {label}: mean={nd.mean():.2f}  median={nd.median():.2f}  "
      f"std={nd.std():.2f}  p10={nd.quantile(0.1):.2f}  p90={nd.quantile(0.9):.2f}")
p()
# Direction-signed: for long setups a negative delta (sell pressure) is the setup condition
p("  Direction-signed net_delta (negative = against direction = setup condition):")
df_cl["_signed_delta"]   = np.where(df_cl["direction"]=="long",   -df_cl["signal_bar_net_delta"],
                                     df_cl["signal_bar_net_delta"])
df_loss["_signed_delta"] = np.where(df_loss["direction"]=="long", -df_loss["signal_bar_net_delta"],
                                     df_loss["signal_bar_net_delta"])
for label, df in [("clean_large", df_cl), ("loss", df_loss)]:
    sd = df["_signed_delta"].dropna()
    frac_pos = (sd > 0).mean() * 100
    p(f"  {label}: mean={sd.mean():.2f}  median={sd.median():.2f}  "
      f"% setup-aligned (>0): {frac_pos:.1f}%")
p()

# ── Section 7: prior bar trend analysis ──────────────────────────────────
p("-" * 76)
p("SECTION 7 — PRIOR BAR DELTA TREND")
p("-" * 76)
p()
for feat, label in [("prior_5bar_delta_trend","5-bar"), ("prior_10bar_delta_trend","10-bar")]:
    p(f"  {label} slope:")
    for grp_label, df in [("clean_large", df_cl), ("loss", df_loss)]:
        vals = df[feat].dropna()
        frac_neg = (vals < 0).mean() * 100  # negative slope = declining delta = selling pressure building
        p(f"    {grp_label}: mean={vals.mean():.3f}  median={vals.median():.3f}  "
          f"frac_negative={frac_neg:.1f}%")
    p()

# ── Section 8: prior_5bar_dominant_consistency ────────────────────────────
p("-" * 76)
p("SECTION 8 — prior_5bar_dominant_consistency")
p("(fraction of prior 5 bars where dominant_side matches reversal expectation)")
p("short: expect buy-dominated prior bars  |  long: expect sell-dominated prior bars")
p("-" * 76)
p()
for label, df in [("clean_large", df_cl), ("loss", df_loss)]:
    vals = df["prior_5bar_dominant_consistency"].dropna()
    high_consistency = (vals >= 0.6).mean() * 100
    p(f"  {label}: mean={vals.mean():.3f}  median={vals.median():.3f}  "
      f"% >= 60% consistent: {high_consistency:.1f}%")
p()

# ── Section 9: By location type ───────────────────────────────────────────
p("-" * 76)
p("SECTION 9 — ABSORPTION RATE BY LOCATION TYPE")
p("-" * 76)
p()
p(f"  {'location':<16}  {'cl_n':>6}  {'cl_abs%':>8}  {'cl_dvc%':>8}  "
  f"{'ls_n':>6}  {'ls_abs%':>8}  {'ls_dvc%':>8}")
p(f"  {'-'*16}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}")
for loc in ["outside_vah", "at_vah", "outside_val", "at_val"]:
    cl_loc  = df_cl[df_cl["price_location"] == loc]
    ls_loc  = df_loss[df_loss["price_location"] == loc]
    cl_abs  = cl_loc["signal_bar_absorption_present"].mean()*100 if len(cl_loc)>0 else float("nan")
    cl_dvc  = cl_loc["signal_bar_delta_vs_close"].mean()*100      if len(cl_loc)>0 else float("nan")
    ls_abs  = ls_loc["signal_bar_absorption_present"].mean()*100  if len(ls_loc)>0 else float("nan")
    ls_dvc  = ls_loc["signal_bar_delta_vs_close"].mean()*100      if len(ls_loc)>0 else float("nan")
    p(f"  {loc:<16}  {len(cl_loc):>6,}  {cl_abs:>7.1f}%  {cl_dvc:>7.1f}%  "
      f"{len(ls_loc):>6,}  {ls_abs:>7.1f}%  {ls_dvc:>7.1f}%")
p()

# ── Section 10: composite score ───────────────────────────────────────────
p("-" * 76)
p("SECTION 10 — COMPOSITE ORDERFLOW SCORE")
p("Score = absorption_present + delta_vs_close + (zone_count >= 2) + (consistency >= 0.6)")
p("Range 0-4; higher = stronger setup signal")
p("-" * 76)
p()
for label, df in [("clean_large", df_cl), ("loss", df_loss)]:
    score = (df["signal_bar_absorption_present"].astype(int) +
             df["signal_bar_delta_vs_close"].astype(int) +
             (df["signal_bar_zone_count"] >= 2).astype(int) +
             (df["prior_5bar_dominant_consistency"] >= 0.6).astype(int))
    p(f"  {label}: mean={score.mean():.2f}  median={score.median():.1f}")
    for s in range(5):
        frac = (score == s).mean() * 100
        bar_str = "#" * int(frac / 2)
        p(f"    score {s}: {frac:5.1f}%  {bar_str}")
    p()

# ── Summary ────────────────────────────────────────────────────────────────
p("=" * 76)
p("KEY FINDINGS SUMMARY")
p("=" * 76)
p()
top_feat = sep_df.iloc[0]
p(f"  Top separator: {top_feat['feature']}")
p(f"    clean_large mean={top_feat['cl_mean']:.4f}  loss mean={top_feat['ls_mean']:.4f}")
p(f"    sep_ratio={top_feat['sep_ratio']:.4f}  KS={top_feat['ks_stat']:.4f}")
p()
abs_cl   = df_cl["signal_bar_absorption_present"].mean()  * 100
abs_ls   = df_loss["signal_bar_absorption_present"].mean() * 100
dvc_cl   = df_cl["signal_bar_delta_vs_close"].mean()  * 100
dvc_ls   = df_loss["signal_bar_delta_vs_close"].mean() * 100
p(f"  Absorption signal present:")
p(f"    clean_large: {abs_cl:.1f}%   loss: {abs_ls:.1f}%   lift: {abs_cl-abs_ls:+.1f}pp")
p(f"  Delta vs close divergence:")
p(f"    clean_large: {dvc_cl:.1f}%   loss: {dvc_ls:.1f}%   lift: {dvc_cl-dvc_ls:+.1f}pp")
p()
p(f"  Zone concentration (mean):")
p(f"    clean_large: {df_cl['signal_bar_zone_concentration'].mean():.3f}   "
  f"loss: {df_loss['signal_bar_zone_concentration'].mean():.3f}")
p(f"  Prior 5-bar consistency (mean):")
p(f"    clean_large: {df_cl['prior_5bar_dominant_consistency'].mean():.3f}   "
  f"loss: {df_loss['prior_5bar_dominant_consistency'].mean():.3f}")
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

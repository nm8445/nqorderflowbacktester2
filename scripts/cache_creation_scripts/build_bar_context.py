"""
Build per-bar context table at 1-min granularity. Downstream scripts join this
to the wick-delta index (at whatever timeframe) to get location context.

Columns:
  session_date, bar_open_time (1-min, tz-aware ET), is_rth, is_eth
  dev_poc, dev_vah, dev_val        # developing profile snapshot at that minute
  noise_upper_14, noise_lower_14   # zarattini noise band with 14d lookback
  noise_upper_90, noise_lower_90   # with 90d lookback
  hod_eth_locked, lod_eth_locked   # per-day values (6pm prev day -> 9:29 today)
  prev_rth_poc_d1/d2/d3            # last 3 finalized daily RTH POCs
  prev_week_iso_poc/vah/val        # prior ISO week Mon-Fri RTH composite
  trailing_5d_poc/vah/val          # last 5 trading sessions (rolling)

Output: D:/trading_pythonbacktest_data/bar_context_1min.parquet
"""
from __future__ import annotations

import pickle
import time
import sys
from pathlib import Path
from datetime import date, timedelta, datetime
from collections import defaultdict

import numpy as np
import pandas as pd

ET = "America/New_York"
DATA_DIR = Path("D:/trading_pythonbacktest_data")
DEV_DIR = DATA_DIR / "cache/profiles"
SIGMA_14 = DATA_DIR / "noise_band_sigma_cache/sigma_lookback_14.pkl"
SIGMA_90 = DATA_DIR / "noise_band_sigma_cache/sigma_lookback_90.pkl"
VOL_1MIN = DATA_DIR / "volumetric_1min_1tpl.parquet"
OUTPUT = DATA_DIR / "bar_context_1min.parquet"

BUCKET_TPL = 10  # dev profile is already at 10-tick (2.5pt) buckets


# ── Helper: per-day RTH OHLC & HOD/LOD locked at 9:30 ET ────────────────────
def build_daily_anchors():
    """For each session in the volumetric 1-min cache, compute:
    - today_open  (first RTH bar = 9:30 ET open)
    - last_close  (last RTH bar = 16:59 ET close, equals daily RTH close)
    - hod_eth_locked / lod_eth_locked (high/low from 18:00 prev day -> 9:29 today)
    """
    print("  [anchors] Scanning 1-min cache for daily anchors...", flush=True)
    df = pd.read_parquet(VOL_1MIN, columns=[
        "session_date", "bar_open_time", "open", "high", "low", "close"])
    df = df[df["open"].notna()].copy()  # drop empty-bar placeholders
    df["bar_et"] = pd.to_datetime(df["bar_open_time"]).dt.tz_convert(ET)
    df["hm"] = df["bar_et"].dt.strftime("%H:%M")

    # RTH bars per session for open/close
    rth = df[(df["hm"] >= "09:30") & (df["hm"] <= "16:59")]
    t_open = rth.groupby("session_date")["open"].first().rename("today_open")
    t_close = rth.groupby("session_date")["close"].last().rename("today_close")

    # ETH bars = everything before 9:30 ET of that session. The session starts
    # at 18:00 prev day and goes through 9:29 today -> all bars with hm NOT in
    # [09:30, 16:59] covers ETH (and 17:00+ which is post-close, also ETH-adj)
    # For HOD/LOD at 9:30 lock, we want bars in [18:00 prev day, 09:29 today]
    # which is hm >= "18:00" OR hm < "09:30". Both belong to the session already
    # since session_date assignment did that mapping.
    eth = df[(df["hm"] < "09:30") | (df["hm"] >= "18:00")]
    eth_hod = eth.groupby("session_date")["high"].max().rename("hod_eth_locked")
    eth_lod = eth.groupby("session_date")["low"].min().rename("lod_eth_locked")

    anchors = pd.concat([t_open, t_close, eth_hod, eth_lod], axis=1)
    anchors.index.name = "session_date"
    print(f"  [anchors] {len(anchors)} sessions with anchor data", flush=True)
    return anchors


# ── Helper: per-day dev profile minute snapshots → scalars ───────────────────
def extract_dev_snapshots(date_iso):
    """Return (minute_ts_utc, poc, vah, val) arrays for this date from the
    developing profile cache."""
    p = DEV_DIR / f"{date_iso}_refresh_minutes=1.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        snaps = pickle.load(f)
    rts = []; pocs = []; vahs = []; vals = []
    for s in snaps:
        rt = s["refresh_time"]
        if rt.tz is None: rt = rt.tz_localize("UTC")
        prof = s["profile"]
        rts.append(rt)
        pocs.append(prof.poc)
        vahs.append(prof.vah)
        vals.append(prof.val)
    return pd.DataFrame({
        "bar_open_time_utc": rts,
        "dev_poc": pocs, "dev_vah": vahs, "dev_val": vals,
    })


# ── Helper: compute per-session noise bands for lookback 14 and 90 ──────────
def compute_noise_bands(anchors, sigma14, sigma90):
    """Return DataFrame indexed by (session_date, hm) with upper/lower bands
    for both lookbacks. Later we'll join to per-minute bars by hm."""
    rows = []
    for sess_date, r in anchors.iterrows():
        if pd.isna(r["today_open"]):
            continue
        # yesterday_close = the prior session's today_close
        pass  # computed below via shift
    # Easier: shift anchors to get yesterday_close
    a = anchors.sort_index().copy()
    a["yesterday_close"] = a["today_close"].shift(1)
    a["upper_anchor"] = a[["today_open", "yesterday_close"]].max(axis=1)
    a["lower_anchor"] = a[["today_open", "yesterday_close"]].min(axis=1)

    rows = []
    for sess_date, r in a.iterrows():
        if pd.isna(r["upper_anchor"]):
            continue
        sig14 = sigma14.get(sess_date) if isinstance(sigma14, dict) else None
        sig90 = sigma90.get(sess_date) if isinstance(sigma90, dict) else None
        sess_date_obj = date.fromisoformat(sess_date) if isinstance(sess_date, str) else sess_date
        # Emit a row per minute in 18:00 prev day -> 17:00 today
        # (1380 minutes, same as the volumetric grid)
        base_start = pd.Timestamp(sess_date_obj - timedelta(days=1), tz=ET).replace(hour=18, minute=0)
        minutes = pd.date_range(base_start, base_start + pd.Timedelta(hours=23), freq="1min", tz=ET)
        for m in minutes:
            hm = m.strftime("%H:%M")
            s14 = sig14.get(hm) if sig14 else None
            s90 = sig90.get(hm) if sig90 else None
            upper14 = r["upper_anchor"] * (1 + s14) if s14 else np.nan
            lower14 = r["lower_anchor"] * (1 - s14) if s14 else np.nan
            upper90 = r["upper_anchor"] * (1 + s90) if s90 else np.nan
            lower90 = r["lower_anchor"] * (1 - s90) if s90 else np.nan
            rows.append((sess_date, m, upper14, lower14, upper90, lower90))
    return pd.DataFrame(rows, columns=["session_date", "bar_open_time",
                                         "noise_upper_14", "noise_lower_14",
                                         "noise_upper_90", "noise_lower_90"])


# ── Helper: compute RTH POC per day via snapshot subtraction ────────────────
def rth_poc_for_date(date_iso):
    p = DEV_DIR / f"{date_iso}_refresh_minutes=1.pkl"
    if not p.exists(): return None
    with open(p, "rb") as f:
        snaps = pickle.load(f)
    t_start = pd.Timestamp(date_iso + " 09:30:00", tz=ET)
    t_end   = pd.Timestamp(date_iso + " 16:59:00", tz=ET)

    best_s = best_e = None
    ds_s = ds_e = pd.Timedelta.max
    for s in snaps:
        rt = s["refresh_time"]
        if rt.tz is None: rt = rt.tz_localize("UTC")
        rt_et = rt.tz_convert(ET)
        if abs(rt_et - t_start) < ds_s: ds_s = abs(rt_et - t_start); best_s = s
        if abs(rt_et - t_end)   < ds_e: ds_e = abs(rt_et - t_end);   best_e = s
    if not best_s or not best_e: return None
    if ds_s > pd.Timedelta("5min") or ds_e > pd.Timedelta("5min"): return None

    start_lv = {pr: (lv.buy_vol, lv.sell_vol) for pr, lv in best_s["profile"].levels.items()}
    diff = {}
    for pr, lv in best_e["profile"].levels.items():
        bs = start_lv.get(pr, (0, 0))
        tot = (lv.buy_vol - bs[0]) + (lv.sell_vol - bs[1])
        if tot > 0: diff[pr] = tot
    if not diff: return None
    return max(diff, key=lambda k: diff[k])


# ── Helper: aggregate sessions into a composite profile (POC/VAH/VAL) ───────
def composite_profile_from_sessions(session_dates, value_area_pct=0.68):
    """Sum level dicts from the END-OF-SESSION snapshot for each given date,
    return (poc, vah, val). Returns (nan, nan, nan) if no usable data."""
    combined = defaultdict(lambda: [0, 0])  # level_price -> [buy, sell]
    for d in session_dates:
        p = DEV_DIR / f"{d}_refresh_minutes=1.pkl"
        if not p.exists(): continue
        with open(p, "rb") as f:
            snaps = pickle.load(f)
        # Last snapshot = full session profile
        last = snaps[-1]["profile"]
        for pr, lv in last.levels.items():
            combined[pr][0] += lv.buy_vol
            combined[pr][1] += lv.sell_vol
    if not combined:
        return (np.nan, np.nan, np.nan)
    prices_sorted = sorted(combined.keys())
    total = sum(bs[0] + bs[1] for bs in combined.values())
    if total == 0:
        return (np.nan, np.nan, np.nan)
    poc = max(combined, key=lambda k: combined[k][0] + combined[k][1])
    poc_idx = prices_sorted.index(poc)
    va_target = total * value_area_pct
    va_vol = combined[poc][0] + combined[poc][1]
    lo, hi = poc_idx, poc_idx
    while va_vol < va_target:
        can_up = hi + 1 < len(prices_sorted)
        can_dn = lo - 1 >= 0
        if not can_up and not can_dn: break
        up_v = (combined[prices_sorted[hi+1]][0] + combined[prices_sorted[hi+1]][1]) if can_up else -1
        dn_v = (combined[prices_sorted[lo-1]][0] + combined[prices_sorted[lo-1]][1]) if can_dn else -1
        if up_v >= dn_v: hi += 1; va_vol += up_v
        else:            lo -= 1; va_vol += dn_v
    vah = prices_sorted[hi] + BUCKET_TPL * 0.25  # level_size = 2.5 pt
    val = prices_sorted[lo]
    return (poc, vah, val)


# ── Main orchestrator ───────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("Loading sigma caches...", flush=True)
    with open(SIGMA_14, "rb") as f: sigma14 = pickle.load(f)
    with open(SIGMA_90, "rb") as f: sigma90 = pickle.load(f)
    print(f"  sigma14: {len(sigma14)} days   sigma90: {len(sigma90)} days")

    print("Building daily anchors (today open/close + ETH HOD/LOD)...", flush=True)
    anchors = build_daily_anchors()

    print("Computing RTH POC per session (snapshot subtraction)...", flush=True)
    rth_poc = {}
    sessions = sorted(anchors.index.tolist())
    for i, d in enumerate(sessions, 1):
        p = rth_poc_for_date(d)
        if p is not None: rth_poc[d] = p
        if i % 200 == 0: print(f"  [{i}/{len(sessions)}] rth_poc", flush=True)
    print(f"  {len(rth_poc)} sessions with RTH POC", flush=True)

    # Per-day prev RTH POCs
    d_list = sorted(rth_poc.keys())
    poc_series = pd.Series(rth_poc)
    prev_poc_d1 = poc_series.shift(1)
    prev_poc_d2 = poc_series.shift(2)
    prev_poc_d3 = poc_series.shift(3)

    # Per-week ISO composite (Mon-Fri). For each session, find the prior complete ISO week's sessions.
    print("Computing prev ISO-week composite profile (Mon-Fri, value area 0.68)...", flush=True)
    session_to_iso_week = {}
    iso_week_sessions = defaultdict(list)
    for d in sessions:
        dt = date.fromisoformat(d)
        iso_year, iso_week, _ = dt.isocalendar()
        key = (iso_year, iso_week)
        session_to_iso_week[d] = key
        iso_week_sessions[key].append(d)

    week_keys_sorted = sorted(iso_week_sessions.keys())
    week_composite = {}
    for i, key in enumerate(week_keys_sorted):
        if i == 0: continue
        prev_key = week_keys_sorted[i-1]
        prev_sessions = iso_week_sessions[prev_key]
        poc, vah, val = composite_profile_from_sessions(prev_sessions)
        week_composite[key] = (poc, vah, val)
        if (i+1) % 40 == 0: print(f"  [{i+1}/{len(week_keys_sorted)}] iso weeks", flush=True)

    # Trailing-5-session rolling composite: for each session, composite of prior 5 sessions
    print("Computing trailing 5-day rolling composite...", flush=True)
    trail_composite = {}
    for i, d in enumerate(sessions):
        if i < 5: continue
        prior_5 = sessions[i-5:i]
        poc, vah, val = composite_profile_from_sessions(prior_5)
        trail_composite[d] = (poc, vah, val)
        if (i+1) % 200 == 0: print(f"  [{i+1}/{len(sessions)}] trailing", flush=True)

    # Noise bands (per-minute grid)
    print("Computing per-minute noise bands (lookback 14 + 90)...", flush=True)
    noise_df = compute_noise_bands(anchors, sigma14, sigma90)
    print(f"  noise grid: {len(noise_df):,} rows", flush=True)

    # Dev profile per-minute POC/VAH/VAL
    print("Extracting per-minute dev profile snapshots (POC/VAH/VAL)...", flush=True)
    dev_pieces = []
    for i, d in enumerate(sessions, 1):
        snap = extract_dev_snapshots(d)
        if snap is not None:
            snap["session_date"] = d
            dev_pieces.append(snap)
        if i % 200 == 0: print(f"  [{i}/{len(sessions)}] dev profile", flush=True)
    dev_df = pd.concat(dev_pieces, ignore_index=True)
    dev_df["bar_open_time"] = pd.to_datetime(dev_df["bar_open_time_utc"]).dt.tz_convert(ET)
    # Snap to 1-minute floor to match the volumetric grid
    dev_df["bar_open_time"] = dev_df["bar_open_time"].dt.floor("1min")
    dev_df = dev_df[["session_date", "bar_open_time", "dev_poc", "dev_vah", "dev_val"]]
    print(f"  dev snapshots: {len(dev_df):,} rows", flush=True)

    # Build the full 1-min grid from the volumetric 1-min cache
    print("Building output grid from volumetric 1-min cache...", flush=True)
    grid = pd.read_parquet(VOL_1MIN, columns=["session_date", "bar_open_time"])
    grid = grid.drop_duplicates(subset=["session_date", "bar_open_time"]).reset_index(drop=True)
    grid["bar_open_time"] = pd.to_datetime(grid["bar_open_time"])
    if grid["bar_open_time"].dt.tz is None:
        grid["bar_open_time"] = grid["bar_open_time"].dt.tz_localize("UTC").dt.tz_convert(ET)
    else:
        grid["bar_open_time"] = grid["bar_open_time"].dt.tz_convert(ET)

    # Join dev profile snapshots
    grid = grid.merge(dev_df, on=["session_date", "bar_open_time"], how="left")

    # Join noise bands
    noise_df["bar_open_time"] = pd.to_datetime(noise_df["bar_open_time"])
    if noise_df["bar_open_time"].dt.tz is None:
        noise_df["bar_open_time"] = noise_df["bar_open_time"].dt.tz_localize(ET)
    grid = grid.merge(noise_df, on=["session_date", "bar_open_time"], how="left")

    # Join per-day broadcast columns
    anchors_b = anchors.reset_index()[["session_date", "hod_eth_locked", "lod_eth_locked"]]
    grid = grid.merge(anchors_b, on="session_date", how="left")

    prev_df = pd.DataFrame({
        "session_date": poc_series.index,
        "prev_rth_poc_d1": prev_poc_d1.values,
        "prev_rth_poc_d2": prev_poc_d2.values,
        "prev_rth_poc_d3": prev_poc_d3.values,
    })
    grid = grid.merge(prev_df, on="session_date", how="left")

    # Weekly and trailing composites (broadcast per day)
    iso_rows = []
    for d in sessions:
        key = session_to_iso_week[d]
        iso_idx = week_keys_sorted.index(key)
        if iso_idx == 0:
            iso_rows.append((d, np.nan, np.nan, np.nan)); continue
        prev_wk = week_keys_sorted[iso_idx - 1]
        p, v, l = week_composite.get(prev_wk, (np.nan, np.nan, np.nan))
        iso_rows.append((d, p, v, l))
    iso_df = pd.DataFrame(iso_rows, columns=["session_date", "prev_week_iso_poc",
                                               "prev_week_iso_vah", "prev_week_iso_val"])
    grid = grid.merge(iso_df, on="session_date", how="left")

    trail_rows = []
    for d in sessions:
        p, v, l = trail_composite.get(d, (np.nan, np.nan, np.nan))
        trail_rows.append((d, p, v, l))
    trail_df = pd.DataFrame(trail_rows, columns=["session_date", "trailing_5d_poc",
                                                   "trailing_5d_vah", "trailing_5d_val"])
    grid = grid.merge(trail_df, on="session_date", how="left")

    # is_rth / is_eth
    hm = grid["bar_open_time"].dt.strftime("%H:%M")
    grid["is_rth"] = (hm >= "09:30") & (hm <= "16:59")
    grid["is_eth"] = ~grid["is_rth"]

    grid = grid.sort_values(["session_date", "bar_open_time"]).reset_index(drop=True)
    grid.to_parquet(OUTPUT, compression="snappy", index=False)

    print(f"\nWrote {OUTPUT.name}:  {OUTPUT.stat().st_size/1e6:.1f} MB   ({len(grid):,} rows)")
    print(f"Total elapsed: {time.time()-t0:.0f}s  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
